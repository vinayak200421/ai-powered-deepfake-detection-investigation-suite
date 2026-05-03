"""PipelineConfig defaults and DEEPFAKE_* env overrides (no weights required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline import PipelineConfig


def test_pipeline_config_default_model_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPFAKE_FUSION_MODEL", raising=False)
    monkeypatch.delenv("DEEPFAKE_ATTRIBUTION_CKPT", raising=False)
    cfg = PipelineConfig()
    assert cfg.fusion_model == Path("models/fusion_lr_final.pkl")
    assert cfg.attribution_model == Path("models/dsan_v31_demo_200/best_final_90pct.pt")


def test_pipeline_config_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fus = tmp_path / "custom_fusion.pkl"
    att = tmp_path / "custom_best.pt"
    monkeypatch.setenv("DEEPFAKE_FUSION_MODEL", str(fus))
    monkeypatch.setenv("DEEPFAKE_ATTRIBUTION_CKPT", str(att))
    cfg = PipelineConfig()
    assert cfg.fusion_model == fus
    assert cfg.attribution_model == att


def test_pipeline_config_empty_attribution_env_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPFAKE_FUSION_MODEL", raising=False)
    monkeypatch.setenv("DEEPFAKE_ATTRIBUTION_CKPT", "")
    cfg = PipelineConfig()
    assert cfg.attribution_model is None
