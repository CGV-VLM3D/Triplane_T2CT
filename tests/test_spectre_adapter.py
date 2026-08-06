"""SPECTRE 백본 어댑터 테스트 (REPA teacher).

가중치 없이 도는 부분이 이 파일의 본체다 — 특히 **재조립 순서**와 **center-crop 방지선**은
모델 없이 결정적으로 검증할 수 있고, 틀리면 학습이 크래시 없이 조용히 무의미해지는 항목이다.
가중치가 있으면 (`-k requires_weights`) 실제 forward까지 확인한다.
"""

from __future__ import annotations

import pytest
import torch

from src.baselines.spectre_adapter import (
    CKPT_COMBINER,
    CKPT_VLA,
    DEFAULT_CKPT,
    SpectreBackbone,
    import_spectre,
)

WAN_VOLUME_SHAPE = (512, 512, 256)  # wan 그리드 (253 → 256 end-pad 후)


def test_adapter_constructs_without_weights() -> None:
    """가중치 없이도 import·생성돼야 한다 (lazy build)."""
    backbone = SpectreBackbone(load_weights=False)
    assert backbone.dense_dim == 1080
    assert backbone.global_dim == 1080
    assert backbone.crop_size == (128, 128, 64)
    assert backbone.patch_size == (16, 16, 8)
    assert backbone._model is None  # 아직 build 안 됨


def test_default_checkpoint_paths_under_workspace_data() -> None:
    """모든 ckpt는 /workspace/data/checkpoints/spectre/ 아래에 있어야 한다."""
    for path in (DEFAULT_CKPT, CKPT_VLA, CKPT_COMBINER):
        assert str(path).startswith("/workspace/data/checkpoints/spectre/")
    assert "no_vla" in DEFAULT_CKPT.name, "기본값은 SSL-only 백본이어야 한다"


def test_token_grid_matches_wan_latent_geometry() -> None:
    """teacher token grid는 wan latent(64³)보다 축당 정확히 2배 거칠어야 한다."""
    assert SpectreBackbone.token_grid(WAN_VOLUME_SHAPE) == (32, 32, 32)


def test_crops_to_grid_is_an_exact_inverse_of_grid_patch() -> None:
    """재조립 순서를 **모델 없이** 증명한다: 각 patch에 자기 인덱스를 심고 왕복시킨다.

    crop 순서(`n=(h·n_w+w)·n_d+d`)·crop 내부 patch 순서(H,W,D)·CLS 제거를 한 번에 고정하는
    결정적 테스트. permutation이 하나라도 틀리면 여기서 실패한다.
    """
    window_scan = import_spectre().window_scan

    grid_tok = SpectreBackbone.token_grid(WAN_VOLUME_SHAPE)  # (32, 32, 32)
    patch = SpectreBackbone.patch_size
    idx = torch.arange(int(torch.tensor(grid_tok).prod()), dtype=torch.float32).reshape(
        grid_tok
    )
    vol = idx
    for axis, rep in enumerate(patch):
        vol = vol.repeat_interleave(rep, dim=axis)
    vol = vol.unsqueeze(0)  # (1, 512, 512, 256)

    crops, crop_grid = window_scan(
        vol, SpectreBackbone.crop_size, scale_intensity=False
    )
    assert crop_grid == (4, 4, 4)

    n = crops.shape[0]
    p = tuple(c // q for c, q in zip(SpectreBackbone.crop_size, patch))  # (8, 8, 8)
    per_patch = crops[:, 0].reshape(n, p[0], patch[0], p[1], patch[1], p[2], patch[2])
    per_patch = per_patch.mean(dim=(2, 4, 6)).reshape(n, p[0] * p[1] * p[2], 1)

    got = SpectreBackbone._crops_to_grid(per_patch, crop_grid)[..., 0]
    assert torch.equal(got, idx), "token grid 재조립 순서가 틀렸다"


def test_crops_to_grid_rejects_tokens_that_still_carry_cls() -> None:
    """CLS를 안 뗀 (N, 513, F)는 조용히 통과하면 안 된다."""
    tokens = torch.zeros(64, 513, 8)
    with pytest.raises(ValueError, match="CLS"):
        SpectreBackbone._crops_to_grid(tokens, (4, 4, 4))


def test_window_refuses_a_depth_that_would_be_center_cropped() -> None:
    """z=253을 그대로 넣으면 upstream이 192로 center-crop한다 — 어댑터가 먼저 막아야 한다."""
    backbone = SpectreBackbone(load_weights=False)
    vol = torch.full((1, 512, 512, 253), -1000.0)
    with pytest.raises(ValueError, match="CENTER-CROP"):
        backbone.window(vol)


def test_pool_grid_averages_and_validates() -> None:
    """32³ → 16³ pooling이 2×2×2 평균이고, 나누어떨어지지 않는 목표는 거부한다."""
    dense = torch.randn(4, 4, 4, 3)
    pooled = SpectreBackbone.pool_grid(dense, (2, 2, 2))
    assert pooled.shape == (2, 2, 2, 3)
    torch.testing.assert_close(pooled[0, 0, 0], dense[:2, :2, :2].mean(dim=(0, 1, 2)))
    with pytest.raises(ValueError, match="multiple"):
        SpectreBackbone.pool_grid(dense, (3, 3, 3))


# --- requires_weights ---------------------------------------------------------


@pytest.mark.skipif(
    not DEFAULT_CKPT.is_file(), reason=f"SPECTRE 백본 미다운로드: {DEFAULT_CKPT}"
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 필요")
def test_requires_weights_dense_encode_and_layer_tap() -> None:
    """requires_weights — 실제 forward: dense shape, CLS 제거, 마지막 layer tap 동치성.

    `forward_intermediates(indices=[depth-1], norm=True)`가 `forward_features`에서 CLS를 뗀
    것과 같아야 한다. 다르면 `layer=` 경로가 다른 표현을 주고 있다는 뜻이다.
    """
    backbone = SpectreBackbone(device_str="cuda", with_combiner=False)
    # 작은 볼륨 한 덩어리(crop 1개)면 계약 검증에 충분하다 — 전체 512³는 U0 스모크가 담당.
    vol = torch.randn(1, 128, 128, 64) * 300.0 - 500.0  # HU-ish
    crops, grid = backbone.window(vol)
    assert crops.shape == (1, 1, 128, 128, 64) and grid == (1, 1, 1)
    assert float(crops.min()) >= 0.0 and float(crops.max()) <= 1.0  # HU→[0,1] 내부 처리

    dense, global_ = backbone.encode_crops(crops, grid)
    assert dense.shape == (8, 8, 8, backbone.dense_dim)
    assert global_ is None
    assert torch.isfinite(dense).all()

    depth = len(backbone.model.backbone.blocks)
    dense_last, _ = backbone.encode_crops(crops, grid, layer=depth - 1)
    torch.testing.assert_close(dense, dense_last, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not (DEFAULT_CKPT.is_file() and CKPT_COMBINER.is_file()),
    reason="SPECTRE 백본/combiner 미다운로드",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 필요")
def test_requires_weights_global_encode() -> None:
    """requires_weights — combiner 경유 scan-level 임베딩은 (1080,)이어야 한다."""
    backbone = SpectreBackbone(device_str="cuda", with_combiner=True)
    vol = torch.randn(1, 128, 128, 64) * 300.0 - 500.0
    crops, grid = backbone.window(vol)
    dense, global_ = backbone.encode_crops(crops, grid, want_global=True)
    assert dense.shape == (8, 8, 8, backbone.dense_dim)
    assert global_ is not None and global_.shape == (backbone.global_dim,)
    assert torch.isfinite(global_).all()


@pytest.mark.skipif(
    not DEFAULT_CKPT.is_file(), reason=f"SPECTRE 백본 미다운로드: {DEFAULT_CKPT}"
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 필요")
def test_requires_weights_global_requires_combiner() -> None:
    """combiner 없이 want_global=True면 조용히 None을 주지 말고 실패해야 한다."""
    backbone = SpectreBackbone(device_str="cuda", with_combiner=False)
    vol = torch.randn(1, 128, 128, 64) * 300.0 - 500.0
    crops, grid = backbone.window(vol)
    with pytest.raises(RuntimeError, match="combiner"):
        backbone.encode_crops(crops, grid, want_global=True)
