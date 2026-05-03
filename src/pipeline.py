"""End-to-end inference pipeline (CPU-safe), aligned to PROJECT_PLAN_v10.md §2 and §15.

Supports:
- **Pre-extracted crops**: run on a folder of `frame_*.png` face crops (from
  `src/preprocessing/extract_faces.py`).
- **Raw video (local / CPU)**: sample frames, MTCNN + IoU tracker + aligner, then the same
  spatial → temporal → fusion path as crops mode.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.fusion.fusion_layer import FusionLayer
from src.modules.temporal import TemporalAnalyzer
from src.utils import get_device, load_config


def _find_full_c23(models_dir: Path) -> Path | None:
    for p in models_dir.rglob("full_c23.p"):
        return p
    return None


def _load_bgr_frames(frames_dir: Path, max_frames: int | None) -> list[np.ndarray]:
    import cv2

    paths = sorted(frames_dir.glob("frame_*.png"))
    if max_frames is not None:
        paths = paths[:max_frames]
    out: list[np.ndarray] = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is not None:
            out.append(im)
    return out


def _resolve_fusion_model_path() -> Path:
    """Default ``models/fusion_lr_final.pkl``; override with ``DEEPFAKE_FUSION_MODEL``."""
    raw = os.environ.get("DEEPFAKE_FUSION_MODEL")
    if raw is not None and raw.strip():
        return Path(raw.strip())
    return Path("models/fusion_lr_final.pkl")


def _resolve_attribution_ckpt_path() -> Path | None:
    """Default ``models/dsan_v31_demo_200/best_final_90pct.pt``.

    Set ``DEEPFAKE_ATTRIBUTION_CKPT`` to a checkpoint path; set it to empty to skip
    attribution loading.
    """
    raw = os.environ.get("DEEPFAKE_ATTRIBUTION_CKPT")
    if raw is not None:
        raw = raw.strip()
        if not raw:
            return None
        return Path(raw)
    return Path("models/dsan_v31_demo_200/best_final_90pct.pt")


@dataclass(frozen=True)
class PipelineConfig:
    max_frames: int = 30
    face_detection_interval: int = 5
    inference_config_path: Path = Path("configs/inference_config.yaml")
    xception_weights: Path | None = None
    models_dir: Path = Path("models")
    fusion_model: Path = field(default_factory=_resolve_fusion_model_path)
    attribution_model: Path | None = field(default_factory=_resolve_attribution_ckpt_path)



class Pipeline:
    def __init__(self, device: str | None = None, cfg: PipelineConfig | None = None) -> None:
        self.device = device if device is not None else get_device()
        self.cfg = cfg or PipelineConfig()

        self._spatial: Any | None = None
        self._temporal: TemporalAnalyzer | None = None
        self._fusion: FusionLayer | None = None
        self._attribution: Any | None = None
        self._inf_cfg: dict[str, Any] | None = None

    def load_models(self) -> None:
        """Load inference-time artifacts (weights + config)."""
        self._inf_cfg = load_config(self.cfg.inference_config_path)
        max_frames = int(self._inf_cfg.get("max_frames", self.cfg.max_frames))
        face_detection_interval = int(
            self._inf_cfg.get("face_detection_interval", self.cfg.face_detection_interval)
        )
        self.cfg = PipelineConfig(
            max_frames=max_frames,
            face_detection_interval=face_detection_interval,
            inference_config_path=self.cfg.inference_config_path,
            xception_weights=self.cfg.xception_weights,
            models_dir=self.cfg.models_dir,
            fusion_model=self.cfg.fusion_model,
            attribution_model=self.cfg.attribution_model,
        )

        wpath = self.cfg.xception_weights
        if wpath is None:
            wpath = _find_full_c23(self.cfg.models_dir)
        if wpath is None or not Path(wpath).is_file():
            raise FileNotFoundError(
                "Missing Xception weights (full_c23.p). Provide PipelineConfig.xception_weights "
                "or unzip FaceForensics weights under models/."
            )
        # Local import: keep non-torch usage (e.g. docs/scripts) working without torch.
        from src.modules.spatial import SpatialDetector

        # Load spatial directly onto GPU — stays resident between requests
        self._spatial = SpatialDetector(wpath, device=self.device)
        self._temporal = TemporalAnalyzer(inference_config_path=self.cfg.inference_config_path)
        self._fusion = FusionLayer(model_path=self.cfg.fusion_model)

        if self.cfg.attribution_model and Path(self.cfg.attribution_model).is_file():
            import torch
            from src.attribution.attribution_model_v31 import DSANv31
            self._attribution = DSANv31(num_classes=4, pretrained=False)
            state = torch.load(self.cfg.attribution_model, map_location="cpu")
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            self._attribution.load_state_dict(state, strict=False)
            # Load attribution directly onto GPU — stays resident between requests
            self._attribution.to(self.device)
            self._attribution.eval()

    def _should_generate_gradcam(self, enable_gradcam: bool | None) -> bool:
        if enable_gradcam is not None:
            return bool(enable_gradcam)
        if self._inf_cfg is None:
            return False
        return bool(self._inf_cfg.get("enable_gradcam", False))

    def _run_attribution(
        self,
        crops_bgr: list[np.ndarray],
        enable_gradcam: bool = True,
        heatmap_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        import cv2
        import torch
        from torchvision import transforms

        from src.attribution.dataset_v31 import DSANv31Dataset

        if not self._attribution or not crops_bgr:
            return {"attribution_method": "Unknown", "attribution_scores": {}, "heatmap_paths": {}}

        rgb_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        _mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        _std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
        all_logits: list[torch.Tensor] = []
        rgb_tensors: list[torch.Tensor] = []
        srm_tensors: list[torch.Tensor] = []

        # Subsample: DSAN only needs ~15 frames to classify manipulation type.
        # Xception already processed all frames for spatial accuracy.
        _MAX_ATTR_FRAMES = 15
        if len(crops_bgr) > _MAX_ATTR_FRAMES:
            step = len(crops_bgr) // _MAX_ATTR_FRAMES
            attr_crops = crops_bgr[::step][:_MAX_ATTR_FRAMES]
        else:
            attr_crops = crops_bgr

        with torch.no_grad():
            for bgr in attr_crops:
                rgb_cv = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = rgb_transform(rgb_cv)
                srm = DSANv31Dataset._srm_from_rgb(rgb, _mean, _std)
                rgb_tensors.append(rgb)
                srm_tensors.append(srm)
                logits = self._attribution.predict(
                    rgb.unsqueeze(0).to(self.device),
                    srm.unsqueeze(0).to(self.device),
                )
                all_logits.append(logits.cpu())

        if not all_logits:
            return {"attribution_method": "Unknown", "attribution_scores": {}, "heatmap_paths": {}}

        stacked = torch.cat(all_logits, dim=0)          # (N, 4)
        avg_logits = stacked.mean(dim=0)
        probs = torch.softmax(avg_logits, dim=0).numpy()
        pred_idx = int(torch.argmax(avg_logits).item())
        scores = {methods[i]: float(probs[i]) for i in range(4)}

        heatmap_paths: dict[str, str] = {}

        if enable_gradcam and len(rgb_tensors) > 0:
            try:
                from src.modules.explainability import ExplainabilityModule

                # Pick the frame with highest confidence (max logit for predicted class)
                best_frame_idx = int(torch.argmax(stacked[:, pred_idx]).item())
                best_rgb_t = rgb_tensors[best_frame_idx].unsqueeze(0)  # (1,3,H,W)
                best_srm_t = srm_tensors[best_frame_idx].unsqueeze(0)
                best_bgr   = crops_bgr[best_frame_idx]

                explainer = ExplainabilityModule(self._attribution, device=self.device)
                rgb_hm, freq_hm = explainer.generate_heatmaps(best_rgb_t, best_srm_t, pred_idx)

                # Convert best crop to RGB for overlay
                best_rgb_np = cv2.cvtColor(
                    cv2.resize(best_bgr, (380, 380)), cv2.COLOR_BGR2RGB
                )
                rgb_overlay  = explainer.overlay_heatmap(best_rgb_np, rgb_hm)
                freq_overlay = explainer.overlay_heatmap(best_rgb_np, freq_hm)

                # Save to heatmap_dir
                import tempfile
                out_dir = Path(heatmap_dir) if heatmap_dir else Path(
                    tempfile.mkdtemp(prefix="df_heatmaps_")
                )
                out_dir.mkdir(parents=True, exist_ok=True)

                rgb_path  = out_dir / "rgb_heatmap.png"
                freq_path = out_dir / "freq_heatmap.png"
                cv2.imwrite(str(rgb_path),  cv2.cvtColor(rgb_overlay,  cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(freq_path), cv2.cvtColor(freq_overlay, cv2.COLOR_RGB2BGR))

                heatmap_paths = {
                    "rgb_heatmap":  str(rgb_path),
                    "freq_heatmap": str(freq_path),
                }
            except Exception as _e:
                # Grad-CAM is best-effort — never crash the pipeline
                import traceback
                traceback.print_exc()
                heatmap_paths = {}

        return {
            "attribution_method": methods[pred_idx],
            "predicted_method": methods[pred_idx],
            "class_probabilities": scores,
            "heatmap_paths": heatmap_paths,
        }


    def run_on_crops_dir(
        self,
        crops_dir: str | Path,
        *,
        enable_gradcam: bool | None = None,
    ) -> dict[str, Any]:
        """Analyze an already-extracted crops directory containing frame_*.png."""
        if (
            self._spatial is None
            or self._temporal is None
            or self._fusion is None
            or self._inf_cfg is None
        ):
            self.load_models()
        assert self._spatial is not None and self._temporal is not None and self._fusion is not None

        t0 = time.perf_counter()
        d = Path(crops_dir).expanduser().resolve()
        t_load = time.perf_counter()
        frames = _load_bgr_frames(d, max_frames=self.cfg.max_frames)
        load_s = time.perf_counter() - t_load

        t_spatial = time.perf_counter()
        spatial_out = self._spatial.predict_video(frames)
        spatial_s = time.perf_counter() - t_spatial

        ss = float(spatial_out["spatial_score"])
        per_frame = [float(x) for x in spatial_out["per_frame_predictions"]]
        n_frames = int(spatial_out["num_frames"])

        ts: float | None
        temporal_out: dict[str, Any] | None
        if n_frames >= 2:
            temporal_out = self._temporal.analyze(per_frame)
            ts = float(temporal_out["temporal_score"])
        else:
            temporal_out = None
            ts = None

        fusion_out = self._fusion.predict(ss=ss, ts=ts, n_frames=n_frames)
        
        attr_data = {}
        gradcam_enabled = self._should_generate_gradcam(enable_gradcam)
        if self._attribution is not None:
            # Only generate heatmaps for FAKE — skip for REAL to save VRAM
            t_attr = time.perf_counter()
            attr_data = self._run_attribution(
                frames,
                enable_gradcam=(gradcam_enabled and fusion_out.verdict == "FAKE"),
            )
            attr_s = time.perf_counter() - t_attr
        else:
            attr_s = 0.0
            
        elapsed = time.perf_counter() - t0

        out = {
            "verdict": fusion_out.verdict,
            "fusion_score": fusion_out.fusion_score,
            "spatial_score": ss,
            "temporal_score": ts if ts is not None else "N/A",
            "per_frame_predictions": per_frame,
            "metadata": {
                "frames_analysed": n_frames,
                "crops_dir": str(d),
            },
            "technical": {
                "device": self.device,
                "inference_time_s": float(elapsed),
                "used_fallback": fusion_out.used_fallback,
                "timings_s": {
                    "load_frames": float(load_s),
                    "spatial": float(spatial_s),
                    "attribution": float(attr_s),
                },
            },
        }
        if attr_data:
            out["attribution"] = attr_data
        return out

    def run_on_video(
        self,
        video_path: str | Path,
        *,
        fps_sampling: int | None = None,
        max_frames: int | None = None,
        detector_backend: str = "mtcnn",
        enable_gradcam: bool | None = None,
    ) -> dict[str, Any]:
        """Analyze a raw video locally (CPU path; slow).

        Uses MTCNN + IoU tracker + 1.3x margin crops, aligned to plan §5.7 and §13.
        Multi-face policy: always select the highest-confidence face on (re-)detection frames,
        then track that face.
        """
        if (
            self._spatial is None
            or self._temporal is None
            or self._fusion is None
            or self._inf_cfg is None
        ):
            self.load_models()
        assert self._spatial is not None and self._temporal is not None and self._fusion is not None

        import cv2

        from src.preprocessing.face_aligner import FaceAligner
        from src.preprocessing.face_detector import FaceDetector
        from src.preprocessing.face_tracker import FaceTracker
        from src.preprocessing.frame_sampler import FrameSampler

        vpath = Path(video_path).expanduser().resolve()
        t0 = time.perf_counter()

        cfg_fps = int(self._inf_cfg.get("fps_sampling", 1)) if self._inf_cfg else 1
        cfg_max = (
            int(self._inf_cfg.get("max_frames", self.cfg.max_frames))
            if self._inf_cfg
            else self.cfg.max_frames
        )
        fps = int(fps_sampling) if fps_sampling is not None else cfg_fps
        mf = int(max_frames) if max_frames is not None else cfg_max

        sampler = FrameSampler(fps=fps, max_frames=mf)
        t_sample = time.perf_counter()
        frames_bgr, meta = sampler.sample(vpath)
        sample_s = time.perf_counter() - t_sample

        t_preprocess = time.perf_counter()
        detector = FaceDetector(backend=detector_backend, device=self.device)
        tracker = FaceTracker(detector=detector)
        aligner = FaceAligner(output_size=299, margin_factor=1.3)

        def pick_best_box(dets: list[dict]) -> list[int] | None:
            if not dets:
                return None
            best = max(dets, key=lambda d: float(d.get("confidence", 0.0)))
            return list(map(int, best["box"]))

        crops: list[np.ndarray] = []
        prev_box: list[int] | None = None
        redetect_every = max(1, int(self.cfg.face_detection_interval))
        detections_run = 0
        for idx, fr in enumerate(frames_bgr):
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            if prev_box is None:
                dets = detector.detect(rgb)
                detections_run += 1
                box = pick_best_box(dets)
                if box is None:
                    continue
                prev_box = box
            elif idx % redetect_every != 0:
                box = prev_box
            else:
                tr = tracker.update(fr, prev_box)
                detections_run += 1
                if tr.get("tracked"):
                    prev_box = tr["box"]
                else:
                    dets = detector.detect(rgb)
                    detections_run += 1
                    box = pick_best_box(dets)
                    if box is None:
                        continue
                    prev_box = box
            crops.append(aligner.align(fr, prev_box))
        preprocess_s = time.perf_counter() - t_preprocess

        t_spatial = time.perf_counter()
        spatial_out = self._spatial.predict_video(crops)
        spatial_s = time.perf_counter() - t_spatial

        ss = float(spatial_out["spatial_score"])
        per_frame = [float(x) for x in spatial_out["per_frame_predictions"]]
        n_frames = int(spatial_out["num_frames"])

        if n_frames >= 2:
            temporal_out = self._temporal.analyze(per_frame)
            ts: float | None = float(temporal_out["temporal_score"])
        else:
            temporal_out = None
            ts = None

        fusion_out = self._fusion.predict(ss=ss, ts=ts, n_frames=n_frames)
        
        attr_data = {}
        gradcam_enabled = self._should_generate_gradcam(enable_gradcam)
        if self._attribution is not None:
            # Only generate heatmaps for FAKE — skip for REAL to save VRAM
            t_attr = time.perf_counter()
            attr_data = self._run_attribution(
                crops,
                enable_gradcam=(gradcam_enabled and fusion_out.verdict == "FAKE"),
            )
            attr_s = time.perf_counter() - t_attr
        else:
            attr_s = 0.0
            
        elapsed = time.perf_counter() - t0

        out = {
            "verdict": fusion_out.verdict,
            "fusion_score": fusion_out.fusion_score,
            "spatial_score": ss,
            "temporal_score": ts if ts is not None else "N/A",
            "per_frame_predictions": per_frame,
            "metadata": {
                "video_path": str(vpath),
                "duration_s": float(meta.get("duration", 0.0)),
                "fps": float(meta.get("original_fps", 0.0)),
                "frames_analysed": n_frames,
                "sampling_fps": fps,
                "face_detections_run": detections_run,
                "face_detection_interval": redetect_every,
            },
            "technical": {
                "device": self.device,
                "inference_time_s": float(elapsed),
                "used_fallback": fusion_out.used_fallback,
                "timings_s": {
                    "sample_video": float(sample_s),
                    "preprocess_faces": float(preprocess_s),
                    "spatial": float(spatial_s),
                    "attribution": float(attr_s),
                },
            },
        }
        if attr_data:
            out["attribution"] = attr_data
        return out
