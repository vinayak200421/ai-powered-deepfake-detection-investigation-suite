"""End-to-end inference pipeline (CPU-safe), aligned to PROJECT_PLAN_v10.md §2 and §15.

Supports:
- **Pre-extracted crops**: run on a folder of `frame_*.png` face crops (from
  `src/preprocessing/extract_faces.py`).
- **Raw video (local / CPU)**: sample frames, MTCNN + IoU tracker + aligner, then the same
  spatial → temporal → fusion path as crops mode.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class PipelineConfig:
    max_frames: int = 30
    inference_config_path: Path = Path("configs/inference_config.yaml")
    xception_weights: Path | None = None
    models_dir: Path = Path("models")
    fusion_model: Path = Path("models/fusion_lr.pkl")
    attribution_model: Path | None = Path("models/dsan_v31_demo_200/best_demo.pt")



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
        self.cfg = PipelineConfig(
            max_frames=max_frames,
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

        # Load models onto CPU initially to save VRAM
        self._spatial = SpatialDetector(wpath, device="cpu")
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
            self._attribution.to("cpu")
            self._attribution.eval()

    def _run_attribution(self, crops_bgr: list[np.ndarray]) -> dict[str, Any]:
        import torch
        import cv2
        from torchvision import transforms
        from src.attribution.dataset_v31 import DSANv31Dataset

        if not self._attribution or not crops_bgr:
            return {"attribution_method": "Unknown", "attribution_scores": {}}

        rgb_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        _mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        _std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
        all_logits = []

        with torch.no_grad():
            for bgr in crops_bgr:
                rgb_cv = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = rgb_transform(rgb_cv)
                srm = DSANv31Dataset._srm_from_rgb(rgb, _mean, _std)

                rgb = rgb.unsqueeze(0).to(self.device)
                srm = srm.unsqueeze(0).to(self.device)

                logits = self._attribution.predict(rgb, srm)
                all_logits.append(logits.cpu())

        if not all_logits:
            return {"attribution_method": "Unknown", "attribution_scores": {}}

        avg_logits = torch.cat(all_logits, dim=0).mean(dim=0)
        probs = torch.softmax(avg_logits, dim=0).numpy()
        pred_idx = int(torch.argmax(avg_logits).item())

        scores = {methods[i]: float(probs[i]) for i in range(4)}

        return {
            "attribution_method": methods[pred_idx],
            "attribution_scores": scores
        }

    def run_on_crops_dir(self, crops_dir: str | Path) -> dict[str, Any]:
        """Analyze an already-extracted crops directory containing frame_*.png."""
        if (
            self._spatial is None
            or self._temporal is None
            or self._fusion is None
            or self._inf_cfg is None
        ):
            self.load_models()
        assert self._spatial is not None and self._temporal is not None and self._fusion is not None

        import torch

        t0 = time.perf_counter()
        d = Path(crops_dir).expanduser().resolve()
        frames = _load_bgr_frames(d, max_frames=self.cfg.max_frames)

        # Move Xception to GPU
        self._spatial.to(self.device)
        spatial_out = self._spatial.predict_video(frames)
        # Move Xception back to CPU
        self._spatial.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        if fusion_out.verdict == "FAKE" and self._attribution is not None:
            # Move DSAN to GPU
            self._attribution.to(self.device)
            attr_data = self._run_attribution(frames)
            # Move DSAN back to CPU
            self._attribution.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
            },
        }
        if attr_data:
            out.update(attr_data)
        return out

    def run_on_video(
        self,
        video_path: str | Path,
        *,
        fps_sampling: int | None = None,
        max_frames: int | None = None,
        detector_backend: str = "mtcnn",
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
        frames_bgr, meta = sampler.sample(vpath)

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
        for fr in frames_bgr:
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            if prev_box is None:
                dets = detector.detect(rgb)
                box = pick_best_box(dets)
                if box is None:
                    continue
                prev_box = box
            else:
                tr = tracker.update(fr, prev_box)
                if tr.get("tracked"):
                    prev_box = tr["box"]
                else:
                    dets = detector.detect(rgb)
                    box = pick_best_box(dets)
                    if box is None:
                        continue
                    prev_box = box
            crops.append(aligner.align(fr, prev_box))

        import torch

        # Move Xception to GPU
        self._spatial.to(self.device)
        spatial_out = self._spatial.predict_video(crops)
        # Move Xception back to CPU
        self._spatial.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        if fusion_out.verdict == "FAKE" and self._attribution is not None:
            # Move DSAN to GPU
            self._attribution.to(self.device)
            attr_data = self._run_attribution(crops)
            # Move DSAN back to CPU
            self._attribution.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
            },
            "technical": {
                "device": self.device,
                "inference_time_s": float(elapsed),
                "used_fallback": fusion_out.used_fallback,
            },
        }
        if attr_data:
            out.update(attr_data)
        return out
