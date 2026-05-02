"""SupCon + DSAN composite losses (plan §10.10 for v3; §12.2 for v3.1 multi-task).

v3 classes (``SupConLoss``, ``DSANLoss``) are preserved unchanged for
backwards compatibility. v3.1 adds :class:`DSANv31Loss`, a multi-task wrapper
that adds a blending-mask BCE term and supports per-sample loss masking so
SBI samples contribute only to the mask term.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.15) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features.float(), dim=1)
        b = features.shape[0]
        if b == 0:
            return features.sum() * 0.0
        mask_self = torch.eye(b, device=features.device, dtype=torch.bool)
        similarity = torch.matmul(features, features.T)
        similarity.masked_fill_(mask_self, -1e4)
        similarity = similarity / self.temperature

        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_pos = labels_eq.float() * (~mask_self).float()

        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        num_positives = mask_pos.sum(dim=1)
        pos_log_prob = torch.where(mask_pos.bool(), log_prob, torch.zeros_like(log_prob))
        mean_log_prob = pos_log_prob.sum(dim=1) / (num_positives + 1e-8)

        valid_mask = num_positives > 0
        if valid_mask.sum() == 0:
            warnings.warn(
                "SupCon: no positive pairs in batch — check StratifiedBatchSampler", stacklevel=2
            )
            return features.sum() * 0.0
        return -mean_log_prob[valid_mask].mean()


class DSANLoss(nn.Module):
    """v3 composite loss: ``alpha * CE + beta * SupCon``."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.2, temperature: float = 0.15) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        self.con_loss = SupConLoss(temperature=temperature)

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l_ce = self.ce_loss(logits, labels)
        l_con = self.con_loss(embeddings, labels)
        return self.alpha * l_ce + self.beta * l_con, l_ce, l_con


class DSANv31Loss(nn.Module):
    """v3.1 multi-task loss: ``alpha * CE + beta * SupCon + lambda_mask * BCE``.

    Supports per-sample classification masking via ``cls_mask``. For SBI
    samples, ``cls_mask = 0`` excludes them from CE and SupCon — they contribute
    only to the mask-head BCE, where ``mask_mask = 1``.

    Returns a tuple ``(total, ce, supcon, mask_bce)`` with unreduced components
    for logging.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.2,
        lambda_mask: float = 0.3,
        temperature: float = 0.15,
        mask_pos_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_mask = lambda_mask
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.con_loss = SupConLoss(temperature=temperature)
        self.register_buffer(
            "mask_pos_weight",
            torch.tensor(float(mask_pos_weight), dtype=torch.float32),
        )

    @staticmethod
    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        denom = m.sum().clamp(min=1.0)
        return (x * m).sum() / denom

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask_logits: torch.Tensor | None,
        mask_gt: torch.Tensor | None,
        cls_mask: torch.Tensor | None = None,
        mask_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b = logits.shape[0]
        device = logits.device
        dtype = logits.dtype

        if cls_mask is None:
            cls_mask = torch.ones(b, device=device, dtype=dtype)
        else:
            cls_mask = cls_mask.to(device=device, dtype=dtype)
        if mask_mask is None and mask_gt is not None:
            mask_mask = torch.ones(b, device=device, dtype=dtype)
        elif mask_mask is not None:
            mask_mask = mask_mask.to(device=device, dtype=dtype)

        ce_per = self.ce_loss(logits, labels)
        l_ce = self._masked_mean(ce_per, cls_mask)

        valid_sc = cls_mask > 0
        if valid_sc.any():
            l_con = self.con_loss(embeddings[valid_sc], labels[valid_sc])
        else:
            l_con = logits.sum() * 0.0

        if (
            mask_logits is not None
            and mask_gt is not None
            and mask_mask is not None
            and self.lambda_mask > 0.0
        ):
            bce_per_px = F.binary_cross_entropy_with_logits(
                mask_logits,
                mask_gt,
                pos_weight=self.mask_pos_weight,
                reduction="none",
            )
            bce_per_sample = bce_per_px.mean(dim=(1, 2, 3))
            l_mask = self._masked_mean(bce_per_sample, mask_mask)
        else:
            l_mask = logits.sum() * 0.0

        total = self.alpha * l_ce + self.beta * l_con + self.lambda_mask * l_mask
        return total, l_ce, l_con, l_mask
