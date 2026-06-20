"""MAISI VAE 체크포인트가 로드되고 frozen 상태로 유지되는지 검증.

load_frozen 이후 모든 파라미터가 requires_grad=False인지(인코더·디코더 포함),
eval 모드로 로드되는지를 확인한다. Report2CT/ours 파이프라인의 frozen-VAE 가정 보호.
체크포인트가 없으면 skip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.baselines.maisi import (
    MAISI_CKPT_PATH,
    is_fully_frozen,
    load_frozen,
)


@pytest.mark.skipif(
    not MAISI_CKPT_PATH.exists(), reason=f"MAISI ckpt missing at {MAISI_CKPT_PATH}"
)
def test_maisi_checkpoint_exists() -> None:
    assert MAISI_CKPT_PATH.is_file()
    assert MAISI_CKPT_PATH.stat().st_size > 1024 * 1024, "MAISI ckpt suspiciously small"


@pytest.mark.skipif(
    not MAISI_CKPT_PATH.exists(), reason=f"MAISI ckpt missing at {MAISI_CKPT_PATH}"
)
def test_maisi_loadable_with_loader_util() -> None:
    model = load_frozen(device="cpu")
    assert model is not None
    # Sanity: encoder + decoder modules exist
    assert hasattr(model, "encoder")
    assert hasattr(model, "decoder")


@pytest.mark.skipif(
    not MAISI_CKPT_PATH.exists(), reason=f"MAISI ckpt missing at {MAISI_CKPT_PATH}"
)
def test_maisi_parameters_are_frozen_after_load() -> None:
    """R6 mitigation: every param must have requires_grad=False after load_frozen."""
    model = load_frozen(device="cpu")
    assert is_fully_frozen(model), (
        "Found at least one trainable parameter after load_frozen. "
        "Frozen-VAE assumption is required by the Report2CT/ours pipeline."
    )
    # Spot-check: encoder.parameters() and decoder.parameters() both fully frozen
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(not p.requires_grad for p in model.decoder.parameters())


@pytest.mark.skipif(
    not MAISI_CKPT_PATH.exists(), reason=f"MAISI ckpt missing at {MAISI_CKPT_PATH}"
)
def test_maisi_eval_mode_after_load() -> None:
    """load_frozen(eval_mode=True) should leave model.training == False."""
    model = load_frozen(device="cpu", eval_mode=True)
    assert model.training is False
