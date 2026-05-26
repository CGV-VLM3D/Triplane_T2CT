"""GenerateCT adapter smoke tests (Day 3).

We only verify:
  1. The submodule imports OK via sys.path (no setup.py exec).
  2. CTViT class constructs with paper kwargs (no weights load).
  3. With ckpt present, CTViT.load() runs and the model exposes encoder/decoder.

Full text->volume sampling is heavy (transformer + CTViT decode) and is exercised
on GPU in Day 5 + Phase B. Day 3 test stays CPU-friendly to keep CI under ~30s.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.baselines.generatect_adapter import (
    CTVIT_CKPT,
    SUPERRES_CKPT,
    TRANSFORMER_CKPT,
    build_ctvit,
)


def test_transformer_maskgit_importable_via_sys_path() -> None:
    """sys.path approach (no pip install -e) lets us import the third_party package."""
    # Trigger lazy import.
    from transformer_maskgit import CTViT, MaskGit, MaskGITTransformer  # noqa: F401, PLC0415

    assert CTViT.__module__.startswith("transformer_maskgit")


def test_ctvit_constructs_with_paper_kwargs_no_weights() -> None:
    """CTViT(...) succeeds with the GenerateCT paper kwargs even without ckpts."""
    model = build_ctvit(load_weights=False, device="cpu")
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 10_000_000, f"CTViT param count unexpectedly small: {n_params}"


@pytest.mark.skipif(
    not CTVIT_CKPT.is_file(), reason=f"CTViT ckpt not downloaded yet at {CTVIT_CKPT}"
)
def test_ctvit_loads_pretrained_ckpt() -> None:
    """CTViT loads the pretrained ckpt successfully."""
    model = build_ctvit(load_weights=True, device="cpu")
    # Architecture sanity: CTViT has spatial + temporal transformer paths.
    assert hasattr(model, "spatial_transformer") or hasattr(model, "encode")
    # No grad needed (inference-only).
    assert model.training is False


def test_checkpoint_paths_resolve() -> None:
    """All 3 ckpt paths under /workspace/data/checkpoints/generatect/ are accounted for."""
    for path in (CTVIT_CKPT, TRANSFORMER_CKPT, SUPERRES_CKPT):
        # Files may or may not exist depending on download progress, but the path
        # must point inside our checkpoint dir.
        assert path.parent == Path("/workspace/data/checkpoints/generatect")
