"""Module 1: spatial deepfake score (Ss) via pretrained FaceForensics++ Xception."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchvision import transforms

from src.modules.network.xception_loader import load_xception
from src.utils import get_device


class SpatialDetector:
    """Per-frame P(fake) and mean spatial score Ss (299×299, mean/std 0.5)."""

    def __init__(self, model_path: str | Path, device: str | None = None) -> None:
        path = str(Path(model_path).expanduser().resolve())
        dev = device if device is not None else get_device()
        self.device = torch.device(dev)
        self.model = load_xception(path, device=str(self.device))
        self.model.to(self.device)
        self.model.eval()

    def to(self, device: str | torch.device) -> None:
        """Move the underlying model to the target device."""
        self.device = torch.device(device)
        self.model.to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def predict_frame(self, face_crop_bgr: Any) -> float:
        """Return P(fake) for one face crop (H×W×3 uint8/float BGR numpy)."""
        import numpy as np

        arr = np.asarray(face_crop_bgr)
        face_rgb = arr[:, :, ::-1]
        tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            p_fake = probs[0, 1].item()
        return float(p_fake)

    def predict_video(self, face_crops: list) -> dict[str, Any]:
        """Mean Ss over frames; empty input -> neutral 0.5."""
        if not face_crops:
            return {
                "spatial_score": 0.5,
                "per_frame_predictions": [],
                "num_frames": 0,
            }
        predictions = [self.predict_frame(c) for c in face_crops]
        ss = sum(predictions) / len(predictions)
        return {
            "spatial_score": float(ss),
            "per_frame_predictions": predictions,
            "num_frames": len(predictions),
        }
