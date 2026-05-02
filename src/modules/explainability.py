"""Dual Grad-CAM++ for DSAN v3.1 RGB + frequency streams (plan §11).

V8-04: set_srm before each CAM call.
Updated to support DSANv31 (EfficientNetV2-M + ResNet-50).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.attribution.gradcam_wrapper import DSANGradCAMWrapper


class ExplainabilityModule:
    """Builds spatial + frequency Grad-CAM++ targets from a trained DSANv31."""

    def __init__(self, dsan_model: nn.Module, device: str = "cpu") -> None:
        self.device = device
        dsan_model.eval()
        self.wrapper = DSANGradCAMWrapper(dsan_model)
        self.wrapper.to(device)
        self.dsan = dsan_model

        from pytorch_grad_cam import GradCAMPlusPlus

        rgb_target = self._find_rgb_target_layer(dsan_model)
        self.rgb_cam = GradCAMPlusPlus(model=self.wrapper, target_layers=[rgb_target])

        freq_target = self._find_freq_target_layer(dsan_model)
        self.freq_cam = GradCAMPlusPlus(model=self.wrapper, target_layers=[freq_target])

    @staticmethod
    def _find_rgb_target_layer(dsan: nn.Module) -> nn.Module:
        """Find the last spatial (non-1x1) Conv2d in the RGB stream backbone."""
        target: nn.Module | None = None
        # Access rgb_stream.backbone for both DSANv3 and DSANv31
        backbone = dsan.rgb_stream.backbone
        for _name, module in backbone.named_modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size not in ((1, 1), (1,)):
                target = module
        if target is None:
            # Fallback: last Conv2d of any size
            for _name, module in backbone.named_modules():
                if isinstance(module, nn.Conv2d):
                    target = module
        if target is None:
            raise RuntimeError("No Conv2d found in RGB backbone for Grad-CAM target")
        return target

    @staticmethod
    def _find_freq_target_layer(dsan: nn.Module) -> nn.Module:
        """Find conv2 of the last block in layer4 of ResNet frequency stream."""
        bb = dsan.freq_stream.backbone
        # backbone is a Sequential; layer4 is the second-to-last child (before avgpool)
        children = list(bb.children())
        # Find layer4 — it is an nn.Sequential of Bottleneck/BasicBlock
        layer4: nn.Module | None = None
        for ch in reversed(children):
            if isinstance(ch, nn.Sequential):
                layer4 = ch
                break
        if layer4 is None:
            raise RuntimeError("Could not find layer4 in freq stream backbone")
        last_block = list(layer4.children())[-1]
        # Try conv2 attribute (ResNet Bottleneck / BasicBlock)
        if hasattr(last_block, "conv2"):
            return last_block.conv2
        # Fallback: last Conv2d in that block
        target: nn.Module | None = None
        for _n, m in last_block.named_modules():
            if isinstance(m, nn.Conv2d):
                target = m
        if target is None:
            raise RuntimeError("Could not find conv2 in last freq stream block")
        return target

    def generate_heatmaps(
        self,
        rgb_tensor: torch.Tensor,
        srm_tensor: torch.Tensor,
        target_class: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        if srm_tensor.dim() == 3:
            srm_tensor = srm_tensor.unsqueeze(0)
        rgb_tensor = rgb_tensor.to(self.device)
        srm_tensor = srm_tensor.to(self.device)
        targets = [ClassifierOutputTarget(target_class)]

        # V8-04: set_srm before each CAM call
        self.wrapper.set_srm(srm_tensor)
        rgb_cam_output = self.rgb_cam(input_tensor=rgb_tensor, targets=targets)
        rgb_heatmap = self._norm01(np.asarray(rgb_cam_output[0], dtype=np.float32))

        self.wrapper.set_srm(srm_tensor)
        freq_cam_output = self.freq_cam(input_tensor=rgb_tensor, targets=targets)
        freq_heatmap = self._norm01(np.asarray(freq_cam_output[0], dtype=np.float32))

        return rgb_heatmap, freq_heatmap

    @staticmethod
    def _norm01(x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        return (x - lo) / (hi - lo)

    def overlay_heatmap(self, original_frame_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        from pytorch_grad_cam.utils.image import show_cam_on_image

        frame_float = original_frame_rgb.astype(np.float64) / 255.0
        return show_cam_on_image(frame_float, heatmap, use_rgb=True)
