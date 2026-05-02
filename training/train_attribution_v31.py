#!/usr/bin/env python3
"""DSAN v3.1 training — Excellence pass.

Consumes ``configs/train_config_max.yaml``. Implements, beyond v3:

* EfficientNetV2-M RGB backbone + ResNet-50 frequency backbone.
* Auxiliary blending-mask head (Face-X-ray-style).
* Self-Blended Images augmentation (CVPR 2022) via :class:`DSANv31Dataset`.
* Mixup α=0.2 on classification head (SBI samples excluded).
* EMA shadow + SWA averaging of late-epoch weights.
* Test-Time Augmentation at eval (center-crop + hflip).
* Temperature calibration is done offline via ``scripts/fit_calibration.py``.

CLI modes:

* ``--dry-run``    : 1 forward+backward on random tensors; no dataset, no W&B.
* ``--smoke-train``: 1 epoch × 2 batches on synthetic crops; CPU-safe; <30 s.
* Default         : full multi-day training. Checkpoints resume-able.

Extended flags (see ``docs/GPU_EXECUTION_PLAN.md`` §S-9, §S-10):

* ``--eval-only`` + ``--eval-ckpts A.pt,B.pt`` + ``--eval-split {val,test}`` + ``--tta``
* ``--select-by val_macro_f1 --out-winner models/dsan_v31/winner.pt``
* ``--override key=value ...`` (ablations; dot-notation into the YAML config)
* ``--resume PATH`` (post-crash recovery)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader

from src.attribution.attribution_model_v31 import DSANv31
from src.attribution.dataset_v31 import DSANv31Dataset
from src.attribution.ema import ExponentialMovingAverage
from src.attribution.losses import DSANv31Loss
from src.attribution.mixup import mixup_batch, mixup_ce_loss
from src.attribution.samplers import StratifiedBatchSampler
from src.attribution.sbi import SBIConfig
from src.utils import load_config

SEED = 42
METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dot-notation overrides like ``attribution.model.mask_head.enabled=false``.

    Values are parsed as JSON when possible, otherwise treated as strings.
    """
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--override expects key=value; got {item!r}")
        key, raw = item.split("=", 1)
        try:
            value: Any = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node: Any = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = value
    return cfg


def _load_pairs(split_json: Path) -> list[tuple[str, str]]:
    with split_json.open(encoding="utf-8") as f:
        raw = json.load(f)
    out: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((str(item[0]).strip(), str(item[1]).strip()))
    if not out:
        raise ValueError(f"No [src, tgt] pairs in {split_json}")
    return out


def _stem(src: str, tgt: str) -> str:
    return f"{src.strip()}_{tgt.strip()}"


def load_labeled_videos(
    split_json: Path, crop_dir: Path, methods: list[str] | tuple[str, ...]
) -> tuple[list[str], list[int]]:
    pairs = _load_pairs(split_json)
    video_ids: list[str] = []
    labels: list[int] = []
    for mi, method in enumerate(methods):
        for src, tgt in pairs:
            stem = _stem(src, tgt)
            cands = [crop_dir / method / stem]
            for comp in ("c23", "c40"):
                cands.append(crop_dir / method / comp / stem)
            for p in cands:
                if p.is_dir() and list(p.glob("frame_*.png")):
                    video_ids.append(f"{method}/{stem}")
                    labels.append(mi)
                    break
    return video_ids, labels


def load_originals_pool(
    split_json: Path, crop_dir: Path, originals_dir_name: str = "original"
) -> list[str]:
    pairs = _load_pairs(split_json)
    pool: list[str] = []
    origs_root = crop_dir / originals_dir_name
    if not origs_root.is_dir():
        return pool
    seen = set()
    for src, tgt in pairs:
        for ident in (src, tgt):
            if ident in seen:
                continue
            cand = origs_root / ident
            if cand.is_dir() and list(cand.glob("frame_*.png")):
                pool.append(ident)
                seen.add(ident)
                continue
            for comp in ("c23", "c40"):
                cc = origs_root / comp / ident
                if cc.is_dir() and list(cc.glob("frame_*.png")):
                    pool.append(ident)
                    seen.add(ident)
                    break
    return pool


def build_dataloaders(
    cfg: dict[str, Any], project_root: Path, device: torch.device
) -> tuple[DataLoader, DataLoader]:
    a = cfg["attribution"]
    data = a["data"]
    tr = a["training"]
    crop_dir = (project_root / str(data.get("crop_dir", "data/processed/faces"))).resolve()
    train_split = project_root / str(data["train_split"])
    val_split = project_root / str(data["val_split"])
    methods = list(data.get("methods", list(METHODS)))
    fpv = int(data.get("frames_per_video", 30))
    image_size = int(data.get("image_size", 380))

    masks_crop_dir = data.get("masks_crop_dir")
    masks_crop_dir = str(project_root / masks_crop_dir) if masks_crop_dir else None
    originals_dir_name = str(data.get("originals_dir_name", "original"))
    originals_crop_dir = str(crop_dir / originals_dir_name)

    tr_ids, tr_y = load_labeled_videos(train_split, crop_dir, methods)
    va_ids, va_y = load_labeled_videos(val_split, crop_dir, methods)
    if not tr_ids or not va_ids:
        raise FileNotFoundError(
            f"No training or val samples under {crop_dir} for the given splits. "
            "Add crops or use --smoke-train (synthetic) / --dry-run."
        )

    sbi_cfg_block = a.get("sbi", {}) or {}
    sbi_enabled = bool(sbi_cfg_block.get("enabled", False))
    sbi_ratio = float(sbi_cfg_block.get("ratio", 0.0)) if sbi_enabled else 0.0
    originals_pool = load_originals_pool(
        train_split, crop_dir, originals_dir_name
    ) if sbi_ratio > 0 else []

    mask_cfg = a["model"].get("mask_head", {})
    mask_out_size = int(mask_cfg.get("out_size", 64))
    sbi_cfg = SBIConfig(
        brightness=float(sbi_cfg_block.get("color_jitter", {}).get("brightness", 0.15)),
        contrast=float(sbi_cfg_block.get("color_jitter", {}).get("contrast", 0.15)),
        saturation=float(sbi_cfg_block.get("color_jitter", {}).get("saturation", 0.10)),
        blur_sigma_min=float(sbi_cfg_block.get("blur_sigma", [0.5, 1.5])[0]),
        blur_sigma_max=float(sbi_cfg_block.get("blur_sigma", [0.5, 1.5])[1]),
        mask_blur_min=float(sbi_cfg_block.get("mask_blur_sigma", [3.0, 7.0])[0]),
        mask_blur_max=float(sbi_cfg_block.get("mask_blur_sigma", [3.0, 7.0])[1]),
        out_mask_size=mask_out_size,
    )

    tr_ds = DSANv31Dataset(
        tr_ids, tr_y, str(crop_dir),
        masks_crop_dir=masks_crop_dir,
        originals_pool=originals_pool,
        originals_crop_dir=originals_crop_dir,
        augment=True,
        frames_per_video=fpv,
        methods=methods,
        image_size=image_size,
        mask_out_size=mask_out_size,
        sbi_ratio=sbi_ratio,
        sbi_cfg=sbi_cfg,
    )
    va_ds = DSANv31Dataset(
        va_ids, va_y, str(crop_dir),
        masks_crop_dir=masks_crop_dir,
        originals_pool=[],
        originals_crop_dir=originals_crop_dir,
        augment=False,
        frames_per_video=fpv,
        methods=methods,
        image_size=image_size,
        mask_out_size=mask_out_size,
        sbi_ratio=0.0,
    )

    bsz = int(tr["batch_size"])
    nw = int(tr.get("num_workers", 0))
    pin = bool(tr.get("pin_memory", False)) and device.type == "cuda"
    ytrain = np.asarray(tr_ds.labels, dtype=np.int64)
    sampler = StratifiedBatchSampler(ytrain, batch_size=bsz, min_per_class=2)
    t_loader = DataLoader(
        tr_ds, num_workers=nw, batch_sampler=sampler, pin_memory=pin,
        **({"prefetch_factor": int(tr.get("prefetch_factor", 2)), "persistent_workers": nw > 0} if nw > 0 else {}),
    )
    v_loader = DataLoader(
        va_ds, batch_size=min(bsz, 32), shuffle=False, num_workers=nw, pin_memory=pin,
        **({"prefetch_factor": int(tr.get("prefetch_factor", 2)), "persistent_workers": nw > 0} if nw > 0 else {}),
    )
    return t_loader, v_loader


def build_model(
    cfg: dict[str, Any], device: torch.device, *, pretrained: bool | None = None
) -> DSANv31:
    m = cfg["attribution"]["model"]
    data = cfg["attribution"].get("data", {})
    p = bool(m.get("pretrained", True)) if pretrained is None else bool(pretrained)
    mh = m.get("mask_head", {}) or {}
    model = DSANv31(
        num_classes=int(m.get("num_classes", 4)),
        fused_dim=int(m.get("fused_dim", 512)),
        pretrained=p,
        rgb_backbone=str(m.get("rgb_backbone", "tf_efficientnetv2_m")),
        freq_backbone=str(m.get("freq_backbone", "resnet50")),
        image_size=int(data.get("image_size", 380)),
        mask_head=bool(mh.get("enabled", True)),
        mask_out_size=int(mh.get("out_size", 64)),
        mask_hidden_dim=int(mh.get("hidden_dim", 256)),
    )
    return model.to(device)


def build_optim(
    cfg: dict[str, Any], model: nn.Module
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, list[float]]:
    ocfg = cfg["attribution"]["optimizer"]
    scfg = cfg["attribution"]["scheduler"]
    tcfg = cfg["attribution"]["training"]
    epochs = int(tcfg["epochs"])
    wu = int(scfg.get("warmup_epochs", 0))
    min_lr = float(scfg.get("min_lr", 1e-7))
    wd = float(ocfg.get("weight_decay", 0.0))
    b_lr = float(ocfg.get("backbone_lr", 1e-5))
    h_lr = float(ocfg.get("head_lr", 3e-4))
    m_lr = float(ocfg.get("mask_head_lr", h_lr))

    backbone: list[nn.Parameter] = []
    head: list[nn.Parameter] = []
    mask_params: list[nn.Parameter] = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("mask_decoder."):
            mask_params.append(p)
        elif "classifier" in n:
            head.append(p)
        else:
            backbone.append(p)

    groups: list[dict[str, Any]] = [
        {"params": backbone, "lr": b_lr, "name": "backbone"},
        {"params": head, "lr": h_lr, "name": "head"},
    ]
    if mask_params:
        groups.append({"params": mask_params, "lr": m_lr, "name": "mask_head"})

    opt = torch.optim.AdamW(groups, weight_decay=wd)

    t0 = int(scfg.get("T_0", max(1, epochs - wu)))
    t_mult = int(scfg.get("T_mult", 1))
    if str(scfg.get("type", "cosine_annealing")) == "cosine_warm_restart":
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=t0, T_mult=t_mult, eta_min=min_lr
        )
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, epochs - wu), eta_min=min_lr
        )
    base_lrs = [g["lr"] for g in groups]
    return opt, sched, base_lrs


def _autocast_device(device: torch.device) -> str:
    return "cuda" if device.type == "cuda" else "cpu"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: DSANv31Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixed_precision: bool,
    grad_accum: int,
    epoch: int,
    *,
    mixup_alpha: float = 0.0,
    ema: ExponentialMovingAverage | None = None,
    max_batches: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    model.train()
    use_amp = mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    ac_dev = _autocast_device(device)

    run_loss = 0.0
    run_ce = 0.0
    run_sc = 0.0
    run_mbce = 0.0
    run_n = 0.0
    step = -1
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        rgb, srm, mask_gt, y, cls_mask, mask_mask = [t.to(device, non_blocking=True) for t in batch]

        do_mixup = mixup_alpha > 0.0 and torch.rand(1).item() < 0.5
        if do_mixup:
            rgb_mix, y_a, y_b, lam = mixup_batch(rgb, y, alpha=mixup_alpha, rng=rng)
            rgb = rgb_mix
        else:
            y_a, y_b, lam = y, y, 1.0

        with torch.amp.autocast(ac_dev, enabled=use_amp):
            logits, emb, mask_logits = model(rgb, srm)
            if do_mixup:
                ce_only = nn.CrossEntropyLoss()
                l_ce = lam * ce_only(logits, y_a) + (1.0 - lam) * ce_only(logits, y_b)
                l_con = logits.sum() * 0.0
                if mask_logits is not None:
                    mbce = torch.nn.functional.binary_cross_entropy_with_logits(
                        mask_logits, mask_gt,
                        pos_weight=loss_fn.mask_pos_weight.to(device),
                        reduction="none",
                    ).mean(dim=(1, 2, 3))
                    denom = mask_mask.sum().clamp(min=1.0)
                    l_mask = (mbce * mask_mask).sum() / denom
                else:
                    l_mask = logits.sum() * 0.0
                loss = loss_fn.alpha * l_ce + loss_fn.beta * l_con + loss_fn.lambda_mask * l_mask
            else:
                loss, l_ce, l_con, l_mask = loss_fn(
                    logits, emb, y, mask_logits, mask_gt, cls_mask, mask_mask
                )

        run_loss += float(loss.detach().cpu().item())
        run_ce += float(l_ce.detach().cpu().item())
        run_sc += float(l_con.detach().cpu().item())
        run_mbce += float(l_mask.detach().cpu().item())
        run_n += 1.0

        if use_amp and scaler is not None:
            scaler.scale(loss / float(grad_accum)).backward()
        else:
            (loss / float(grad_accum)).backward()

        if (step + 1) % int(grad_accum) == 0:
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

    if step >= 0 and (step + 1) % int(grad_accum) != 0:
        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)

    return {
        "loss": run_loss / max(run_n, 1.0),
        "ce": run_ce / max(run_n, 1.0),
        "supcon": run_sc / max(run_n, 1.0),
        "mask_bce": run_mbce / max(run_n, 1.0),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    tta: list[str] | None = None,
) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    all_probs: list[np.ndarray] = []
    iou_sum = 0.0
    iou_n = 0.0
    for batch in loader:
        rgb, srm, mask_gt, y, cls_mask, mask_mask = [t.to(device) for t in batch]
        logits, _, mask_logits = model(rgb, srm)
        if tta and "hflip" in tta:
            rgb_f = torch.flip(rgb, dims=(-1,))
            srm_f = torch.flip(srm, dims=(-1,))
            logits_f, _, _ = model(rgb_f, srm_f)
            logits = (logits + logits_f) / 2.0
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        pred = logits.argmax(dim=1).cpu().numpy()
        keep_idx = (cls_mask > 0).nonzero(as_tuple=True)[0].cpu().numpy()
        y_true.extend(int(y[i].item()) for i in keep_idx)
        y_pred.extend(int(pred[i]) for i in keep_idx)

        if mask_logits is not None and (mask_mask > 0).any():
            probs_m = torch.sigmoid(mask_logits)
            pred_m = (probs_m > 0.5).float()
            inter = (pred_m * mask_gt).sum(dim=(1, 2, 3))
            union = ((pred_m + mask_gt) > 0).float().sum(dim=(1, 2, 3)).clamp(min=1.0)
            iou = inter / union
            iou_sum += float((iou * mask_mask).sum().cpu().item())
            iou_n += float(mask_mask.sum().cpu().item())

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0
    oa = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
    rec = (
        recall_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2, 3])
        if y_true else [0.0, 0.0, 0.0, 0.0]
    )
    per_class = {int(i): float(rec[i]) for i in range(len(rec))}
    mask_iou = (iou_sum / iou_n) if iou_n > 0 else float("nan")
    return {
        "macro_f1": macro_f1,
        "accuracy": oa,
        "per_class_recall": per_class,
        "mask_iou": mask_iou,
        "probs": np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 4)),
    }


def _dry_run(cfg: dict[str, Any], device: torch.device, *, pretrained: bool, seed: int) -> None:
    set_seed(seed)
    cfg["attribution"]["model"]["rgb_backbone"] = "tf_efficientnetv2_b0"
    cfg["attribution"]["model"]["freq_backbone"] = "resnet18"
    cfg["attribution"]["data"]["image_size"] = 128
    cfg["attribution"]["model"]["mask_head"]["out_size"] = 32
    cfg["attribution"]["model"]["mask_head"]["hidden_dim"] = 64
    cfg["attribution"]["model"]["pretrained"] = False
    model = build_model(cfg, device, pretrained=False)
    loss_fn = DSANv31Loss(
        alpha=1.0, beta=0.2, lambda_mask=0.3, temperature=0.15, mask_pos_weight=2.0
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    b = 8
    rgb = torch.randn(b, 3, 128, 128, device=device)
    srm = torch.clamp(torch.randn(b, 3, 128, 128, device=device), -1.0, 1.0)
    mask_gt = (torch.rand(b, 1, 32, 32, device=device) > 0.5).float()
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long, device=device)
    cls_mask = torch.ones(b, device=device)
    mask_mask = torch.ones(b, device=device)
    opt.zero_grad(set_to_none=True)
    logits, emb, mask_logits = model(rgb, srm)
    loss, l_ce, l_con, l_mask = loss_fn(logits, emb, labels, mask_logits, mask_gt, cls_mask, mask_mask)
    loss.backward()
    opt.step()
    print("dry-run v3.1 ok:", {
        "loss": float(loss.item()),
        "ce": float(l_ce.item()),
        "supcon": float(l_con.item()),
        "mask_bce": float(l_mask.item()),
    })


def _write_smoke_crops(target: Path, n_per_class: int = 4) -> None:
    from PIL import Image
    target.mkdir(parents=True, exist_ok=True)
    methods = list(METHODS) + ["original"]
    for m in methods:
        mdir = target / m
        mdir.mkdir(exist_ok=True)
        for i in range(n_per_class):
            vdir = mdir / f"v{m}_{i:03d}"
            vdir.mkdir(exist_ok=True)
            for j in range(2):
                img = Image.new("RGB", (96, 96), (
                    (ord(m[0]) * 3 + i * 20) % 255,
                    (50 + i * 30) % 255,
                    (80 + j * 40) % 255,
                ))
                img.save(vdir / f"frame_{j:03d}.png")


def run_smoke_train(device: torch.device) -> None:
    """1 epoch, 2 train batches, 96×96 synthetic PNG crops, CPU, pretrained=False."""
    os.environ["WANDB_MODE"] = "disabled"
    set_seed(SEED)

    cfg: dict[str, Any] = {
        "attribution": {
            "version": "v3_1",
            "model": {
                "rgb_backbone": "tf_efficientnetv2_b0",
                "freq_backbone": "resnet18",
                "fused_dim": 128,
                "num_classes": 4,
                "pretrained": False,
                "mask_head": {
                    "enabled": True,
                    "out_size": 16,
                    "hidden_dim": 32,
                    "lambda_mask": 0.3,
                },
            },
            "sbi": {
                "enabled": True, "ratio": 0.25, "mask_type": "elliptical",
                "color_jitter": {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1},
                "blur_sigma": [0.5, 1.0], "mask_blur_sigma": [2.0, 4.0],
            },
            "training": {
                "epochs": 1,
                "batch_size": 8,
                "gradient_accumulation_steps": 1,
                "num_workers": 0,
                "pin_memory": False,
                "sampler": "stratified_batch",
                "mixed_precision": False,
                "mixup": {"enabled": True, "alpha": 0.2},
            },
            "optimizer": {
                "type": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3,
                "mask_head_lr": 1e-3, "weight_decay": 0.0,
            },
            "scheduler": {"type": "cosine_annealing", "warmup_epochs": 0, "min_lr": 1e-7},
            "loss": {"alpha": 1.0, "beta": 0.2, "temperature": 0.15, "mask_pos_weight": 2.0},
            "data": {
                "crop_dir": None,
                "masks_crop_dir": None,
                "train_split": None,
                "val_split": None,
                "methods": list(METHODS),
                "frames_per_video": 2,
                "image_size": 96,
                "originals_dir_name": "original",
            },
            "normalization": {
                "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            },
        }
    }

    with tempfile.TemporaryDirectory() as tmp:
        crop = Path(tmp) / "crops"
        _write_smoke_crops(crop, n_per_class=4)

        video_ids: list[str] = []
        labels: list[int] = []
        for mi, m in enumerate(METHODS):
            for i in range(4):
                video_ids.append(f"{m}/v{m}_{i:03d}")
                labels.append(mi)

        originals_pool = [f"voriginal_{i:03d}" for i in range(4)]
        originals_crop_dir = str(crop / "original")

        tr_ds = DSANv31Dataset(
            video_ids, labels, str(crop),
            masks_crop_dir=None,
            originals_pool=originals_pool,
            originals_crop_dir=originals_crop_dir,
            augment=False,
            frames_per_video=2,
            methods=list(METHODS),
            image_size=96,
            mask_out_size=16,
            sbi_ratio=0.25,
            sbi_cfg=SBIConfig(out_mask_size=16),
        )
        va_ds = DSANv31Dataset(
            video_ids[:4], labels[:4], str(crop),
            masks_crop_dir=None,
            originals_pool=[],
            originals_crop_dir=originals_crop_dir,
            augment=False,
            frames_per_video=2,
            methods=list(METHODS),
            image_size=96,
            mask_out_size=16,
            sbi_ratio=0.0,
        )
        bsz = 4
        ytrain = np.asarray(tr_ds.labels, dtype=np.int64)
        sampler = StratifiedBatchSampler(ytrain, batch_size=bsz, min_per_class=1)
        t_loader = DataLoader(tr_ds, num_workers=0, batch_sampler=sampler)
        v_loader = DataLoader(va_ds, batch_size=bsz, shuffle=False, num_workers=0)

        model = build_model(cfg, device, pretrained=False)
        lc = cfg["attribution"]["loss"]
        loss_fn = DSANv31Loss(
            alpha=float(lc["alpha"]), beta=float(lc["beta"]),
            lambda_mask=float(cfg["attribution"]["model"]["mask_head"]["lambda_mask"]),
            temperature=float(lc["temperature"]),
            mask_pos_weight=float(lc["mask_pos_weight"]),
        ).to(device)
        opt, sched, base_lrs = build_optim(cfg, model)
        ema = ExponentialMovingAverage(model, decay=0.99)

        stats = train_one_epoch(
            model, t_loader, loss_fn, opt, device,
            mixed_precision=False, grad_accum=1, epoch=0,
            mixup_alpha=0.2, ema=ema, max_batches=2,
            rng=np.random.default_rng(SEED),
        )
        if not np.isfinite(stats["loss"]):
            raise RuntimeError("non-finite loss in v3.1 smoke")
        ev = evaluate(model, v_loader, device, tta=["hflip"])
        if not np.isfinite(float(ev["macro_f1"])):
            raise RuntimeError("non-finite macro_f1 in v3.1 smoke")

        out_dir = Path(tmp) / "ckpt"
        out_dir.mkdir()
        torch.save({"model": model.state_dict(), "config": {"smoke": True}}, out_dir / "epoch1.pt")
        torch.save({"model": model.state_dict(), "config": {"smoke": True}}, out_dir / "best.pt")
        torch.save({"model": ema.state_dict()}, out_dir / "ema.pt")
    print("smoke-train v3.1 ok:", {"train_loss": stats["loss"], "val_macro_f1": ev["macro_f1"]})


def run_eval_only(
    cfg: dict[str, Any],
    project_root: Path,
    device: torch.device,
    ckpt_paths: list[Path],
    split: str,
    use_tta: bool,
    select_by: str | None,
    out_winner: Path | None,
) -> dict[str, Any]:
    set_seed(SEED)
    a = cfg["attribution"]
    data = a["data"]
    crop_dir = (project_root / str(data.get("crop_dir", "data/processed/faces"))).resolve()
    split_file = project_root / str(data[f"{split}_split"])
    methods = list(data.get("methods", list(METHODS)))
    fpv = int(data.get("frames_per_video", 30))
    image_size = int(data.get("image_size", 380))
    ids, ys = load_labeled_videos(split_file, crop_dir, methods)
    ds = DSANv31Dataset(
        ids, ys, str(crop_dir),
        augment=False, frames_per_video=fpv, methods=methods,
        image_size=image_size,
        mask_out_size=int(a["model"].get("mask_head", {}).get("out_size", 64)),
        sbi_ratio=0.0,
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    results: dict[str, dict[str, Any]] = {}
    best_name: str | None = None
    best_metric = float("-inf")
    for p in ckpt_paths:
        model = build_model(cfg, device, pretrained=False)
        sd = torch.load(p, map_location=device)
        model.load_state_dict(sd.get("model", sd), strict=False)
        ev = evaluate(model, loader, device, tta=["hflip"] if use_tta else None)
        results[str(p.name)] = {
            "macro_f1": ev["macro_f1"],
            "accuracy": ev["accuracy"],
            "mask_iou": ev["mask_iou"],
            "per_class_recall": ev["per_class_recall"],
        }
        metric = ev.get(select_by, ev["macro_f1"]) if select_by else ev["macro_f1"]
        if metric > best_metric:
            best_metric = float(metric)
            best_name = str(p)

    print(json.dumps(results, indent=2))
    if out_winner is not None and best_name is not None:
        out_winner.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copyfile(best_name, out_winner)
        print(f"wrote winner: {out_winner} <- {best_name} ({select_by}={best_metric:.4f})")
    return results


@torch.no_grad()
def update_bn_dsan(loader: DataLoader, model: nn.Module, device: torch.device) -> None:
    """Refresh BatchNorm statistics for DSAN models that take ``(rgb, srm)`` inputs."""
    momenta: dict[nn.Module, float | None] = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            momenta[module] = module.momentum
            module.reset_running_stats()
            module.momentum = None

    if not momenta:
        return

    was_training = model.training
    model.train()
    try:
        for batch in loader:
            rgb = batch[0].to(device, non_blocking=True)
            srm = batch[1].to(device, non_blocking=True)
            model(rgb, srm)
    finally:
        for module, momentum in momenta.items():
            module.momentum = momentum
        model.train(was_training)


def run_training(
    cfg: dict[str, Any],
    project_root: Path,
    device: torch.device,
    output_dir: Path,
    seed: int,
    resume: Path | None,
) -> None:
    set_seed(seed)
    tr_loader, val_loader = build_dataloaders(cfg, project_root, device)
    tcfg = cfg["attribution"]["training"]
    epochs = int(tcfg["epochs"])
    wu = int(cfg["attribution"]["scheduler"].get("warmup_epochs", 0))
    mixed = bool(tcfg.get("mixed_precision", False)) and device.type == "cuda"
    gaccum = int(tcfg.get("gradient_accumulation_steps", 1))
    mixup_alpha = float(tcfg.get("mixup", {}).get("alpha", 0.0)) if tcfg.get("mixup", {}).get("enabled", True) else 0.0
    swa_cfg = tcfg.get("swa", {}) or {}
    swa_on = bool(swa_cfg.get("enabled", False))
    swa_start = int(swa_cfg.get("start_epoch", max(1, epochs - 5)))
    ema_cfg = tcfg.get("ema", {}) or {}
    ema_on = bool(ema_cfg.get("enabled", False))

    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg, device)
    lc = cfg["attribution"]["loss"]
    loss_fn = DSANv31Loss(
        alpha=float(lc["alpha"]), beta=float(lc["beta"]),
        lambda_mask=float(cfg["attribution"]["model"]["mask_head"]["lambda_mask"]),
        temperature=float(lc["temperature"]),
        mask_pos_weight=float(lc["mask_pos_weight"]),
    ).to(device)
    opt, sched, base_lrs = build_optim(cfg, model)
    ema = ExponentialMovingAverage(model, decay=float(ema_cfg.get("decay", 0.999))) if ema_on else None

    swa_model = torch.optim.swa_utils.AveragedModel(model) if swa_on else None

    start_epoch = 0
    best_metric = float("-inf")
    wait = 0
    if resume is not None and resume.is_file():
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start_epoch = int(ck.get("epoch", -1)) + 1
        best_metric = float(ck.get("val_macro_f1", best_metric))
        if ema is not None and "ema" in ck:
            ema.load_state_dict(ck["ema"])
        print(f"resumed from {resume} at epoch {start_epoch} (best={best_metric:.4f})")

    es_cfg = tcfg.get("early_stopping", {}) or {}
    es_on = bool(es_cfg.get("enabled", False))
    patience = int(es_cfg.get("patience", 10))
    rng = np.random.default_rng(seed)

    for epoch in range(start_epoch, epochs):
        if wu > 0 and epoch < wu:
            wf = (epoch + 1) / wu
            for pg, b0 in zip(opt.param_groups, base_lrs):
                pg["lr"] = b0 * wf
        stats = train_one_epoch(
            model, tr_loader, loss_fn, opt, device,
            mixed, gaccum, epoch,
            mixup_alpha=mixup_alpha, ema=ema, rng=rng,
        )
        if epoch >= wu and sched is not None:
            sched.step()
        if swa_on and epoch >= swa_start and swa_model is not None:
            swa_model.update_parameters(model)

        ev = evaluate(model, val_loader, device, tta=None)
        m_f1 = float(ev["macro_f1"])

        ckpt_payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "val_macro_f1": m_f1,
        }
        if ema is not None:
            ckpt_payload["ema"] = ema.state_dict()
        torch.save(ckpt_payload, output_dir / f"epoch_{epoch + 1}.pt")

        if m_f1 > best_metric:
            best_metric = m_f1
            wait = 0
            torch.save(ckpt_payload, output_dir / "best.pt")
        else:
            wait += 1

        print(f"[epoch {epoch + 1}/{epochs}] " + json.dumps({
            **{k: round(v, 4) for k, v in stats.items()},
            "val_macro_f1": round(m_f1, 4),
            "val_mask_iou": None if not np.isfinite(ev["mask_iou"]) else round(float(ev["mask_iou"]), 4),
            "best": round(best_metric, 4),
        }))

        if es_on and wait >= patience:
            print(f"early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

    if swa_on and swa_model is not None:
        update_bn_dsan(tr_loader, swa_model, device)
        torch.save({"model": swa_model.module.state_dict()}, output_dir / "swa.pt")
    if ema is not None:
        ema.apply_shadow(model)
        torch.save({"model": model.state_dict(), "ema_state": ema.state_dict()}, output_dir / "ema.pt")
        ema.restore(model)

    print("training finished", {"best_val_macro_f1": best_metric})


def main() -> None:
    p = argparse.ArgumentParser(description="Train DSAN v3.1 (Excellence pass).")
    p.add_argument("--config", type=Path, default=Path("configs/train_config_max.yaml"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke-train", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("models/dsan_v31"))
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--eval-ckpts", type=str, default=None,
                   help="Comma-separated list of checkpoint paths to evaluate.")
    p.add_argument("--eval-split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--tta", action="store_true", help="Horizontal-flip TTA in eval-only mode.")
    p.add_argument("--select-by", type=str, default="macro_f1")
    p.add_argument("--out-winner", type=Path, default=None)
    p.add_argument("--override", action="append", default=[],
                   help="key.path=value overrides. Repeat to set multiple.")
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    if "attribution" not in cfg:
        print("Config must contain top-level 'attribution' key.", file=sys.stderr)
        sys.exit(1)
    cfg = _apply_overrides(cfg, list(args.override or []))

    if args.smoke_train:
        dev = torch.device("cpu")
    elif args.device:
        dev = torch.device(args.device)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dry_run:
        _dry_run(cfg, dev, pretrained=False, seed=int(args.seed))
        return
    if args.smoke_train:
        run_smoke_train(dev)
        return
    if args.eval_only:
        if not args.eval_ckpts:
            print("--eval-only requires --eval-ckpts", file=sys.stderr)
            sys.exit(2)
        ckpts = [Path(p.strip()) for p in args.eval_ckpts.split(",") if p.strip()]
        run_eval_only(
            cfg, project_root, dev, ckpts,
            split=args.eval_split, use_tta=bool(args.tta),
            select_by=args.select_by, out_winner=args.out_winner,
        )
        return

    try:
        run_training(cfg, project_root, dev, args.output_dir.resolve(), int(args.seed), args.resume)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        print(
            "Use --dry-run, --smoke-train, or place FF++ crops and splits per configs/train_config_max.yaml.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
