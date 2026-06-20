"""GenerateCT 어댑터 스모크 테스트.

sys.path 방식으로 submodule이 import되는지, CTViT가 논문 kwargs로 생성되는지,
출력 spacing이 super-res 여부에 따라 맞게 나오는지, 체크포인트 경로가 우리 디렉토리
아래인지 확인한다. 가중치가 있으면 CTViT.load()까지 검증(없으면 CPU-safe).
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
    """The adapter's lazy helper puts the third_party package on sys.path and imports it.

    A bare ``from transformer_maskgit import ...`` is order-dependent: nothing is on
    sys.path until a build runs, and ct_clip ships a colliding top-level
    ``transformer_maskgit`` with an incompatible CTViT. Go through
    ``_import_transformer_maskgit`` (the scrub + path-prioritize entry point) so the test
    is self-contained and actually exercises the collision-safe import.
    """
    from src.baselines.generatect_adapter import (  # noqa: PLC0415
        _import_transformer_maskgit,
    )

    CTViT, _MaskGit, _MaskGITTransformer = _import_transformer_maskgit()
    assert CTViT.__module__.startswith("transformer_maskgit")


def test_ctvit_constructs_with_paper_kwargs_no_weights() -> None:
    """CTViT(...) succeeds with the GenerateCT paper kwargs even without ckpts."""
    model = build_ctvit(load_weights=False, device="cpu")
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 10_000_000, f"CTViT param count unexpectedly small: {n_params}"


def test_output_spacing_tracks_resolution() -> None:
    """output_spacing is in squeezed-array order (D, H, W) and depends on super-res.

    Hires (512²) → in-plane 0.75 mm; low-res (128²) → 0.75 × 512/128 = 3.0 mm; slice = 1.5 mm.
    Read by src/inference.py to write a truthful NIfTI affine (visualization path).
    """
    from src.baselines.generatect_adapter import GenerateCTAdapter  # noqa: PLC0415

    assert GenerateCTAdapter(load_super_resolution=True).output_spacing == (
        1.5,
        0.75,
        0.75,
    )
    assert GenerateCTAdapter(load_super_resolution=False).output_spacing == (
        1.5,
        3.0,
        3.0,
    )


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
