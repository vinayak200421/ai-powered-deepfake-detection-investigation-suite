"""Load FaceForensics++ pretrained Xception (V9-02 — no last_linear rename in state_dict)."""

from __future__ import annotations

import torch.nn as nn

from .xception import Xception


def patch_relu_inplace(module: nn.Module) -> None:
    """Recursively replace ReLU(inplace=True) with inplace=False (AMP / PyTorch 2.x)."""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU) and child.inplace:
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            patch_relu_inplace(child)


def load_xception(weights_path: str, device: str = "cpu") -> Xception:
    """Build Xception(num_classes=2), patch ReLU, load FF++ ``full_c23.p`` with strict=True."""
    import torch

    model = Xception(num_classes=2)
    # Vendor ``__init__`` registers the head as ``fc`` but ``logits()`` uses ``last_linear``.
    # The FaceForensics ``xception()`` helper aliases them; FF++ checkpoints key ``last_linear``.
    if hasattr(model, "fc") and not hasattr(model, "last_linear"):
        model.last_linear = model.fc
        del model.fc

    patch_relu_inplace(model)

    import sys
    import types
    from src.modules import network
    import src.modules.network.models as models
    from src.modules.network import xception as local_xception
    
    sys.modules["network"] = network
    sys.modules["network.models"] = models
    
    pm = types.ModuleType("pretrainedmodels")
    pmm = types.ModuleType("pretrainedmodels.models")
    
    sys.modules["pretrainedmodels"] = pm
    sys.modules["pretrainedmodels.models"] = pmm
    sys.modules["pretrainedmodels.models.xception"] = local_xception

    # Legacy .p pickle; weights_only=False required on PyTorch 2.6+ (safe pickle from official URL).
    state = torch.load(
        weights_path,
        map_location=device,
        weights_only=False,
    )

    if hasattr(state, "state_dict"):
        state = state.state_dict()
    
    # The original TransferModel puts Xception inside self.model.
    # We strip 'model.' prefix if present.
    new_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            new_state[k[6:]] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state, strict=True)
    model.eval()
    return model
