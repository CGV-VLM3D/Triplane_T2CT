"""Unit tests for src.models.components.mask_conditioner.MaskConditioner (top-K soft).

Covers the load-bearing design properties (plan A4 / Unit-0 / LDM cross-check):
  * latent-shape agnostic (text2ct (128,128,32) and MAISI (120,120,64));
  * zero-init gamma => e_mask == 0 at init (== mask-free baseline);
  * gamma receives grad at init while the embedding is blocked (one-step warmup);
  * single-class content is unit-scale (matches scale_factor-normalized latent, std ~1);
  * exact equivalence to LDM's channel_mapper on the SOFT one-hot (1x1 conv, bias=False);
  * K=1, frac=1 reduces to a plain per-voxel lookup;
  * per-class dropout => PARTIAL mask (still nonzero, never the null), training-only.
"""

from __future__ import annotations

import torch

from src.models.components.mask_conditioner import MaskConditioner

NUM_CLASSES = 118
EMBED_DIM = 8


def _topk(b: int, k: int, grid: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Random top-K soft label field: classes (B,K,*grid) long, fracs (B,K,*grid) summing to 1."""
    classes = torch.randint(0, NUM_CLASSES, (b, k, *grid))
    fracs = (
        torch.ones(b, 1, *grid)
        if k == 1
        else torch.softmax(torch.randn(b, k, *grid), dim=1)
    )
    return classes, fracs


def _no_dropout() -> MaskConditioner:
    return MaskConditioner(NUM_CLASSES, EMBED_DIM, class_dropout=0.0)


def test_shape_agnostic_both_grids() -> None:
    mc = MaskConditioner(NUM_CLASSES, EMBED_DIM)
    for grid in [(128, 128, 32), (120, 120, 64)]:
        out = mc(*_topk(2, 4, grid))  # (2,K,H,W,D) -> (2,embed,H,W,D)
        assert out.shape == (2, EMBED_DIM, *grid), out.shape


def test_zero_init_is_baseline() -> None:
    mc = MaskConditioner(NUM_CLASSES, EMBED_DIM)
    out = mc(*_topk(2, 4, (16, 16, 8)))
    assert torch.count_nonzero(out) == 0, (
        "gamma=0 must give an all-zero mask embedding at init"
    )


def test_gamma_gets_grad_embedding_blocked_at_init() -> None:
    mc = _no_dropout()
    out = mc(*_topk(2, 4, (16, 16, 8)))
    target = torch.randn_like(out)
    ((out - target) ** 2).mean().backward()
    assert mc.gamma.grad is not None and mc.gamma.grad.abs().sum() > 0, (
        "gamma must learn from step 0"
    )
    assert mc.embedding.weight.grad.abs().sum() == 0, (
        "embedding grad is 0 at gamma=0 (gated by gamma)"
    )


def test_single_class_content_unit_scale() -> None:
    """K=1, frac=1, gamma=1 => pure lookup => e_mask std ~1 (matches scale_factor-normed latent)."""
    mc = _no_dropout()
    with torch.no_grad():
        mc.gamma.fill_(1.0)
    out = mc(*_topk(4, 1, (32, 32, 16)))
    assert 0.8 < out.std().item() < 1.2, out.std().item()


def test_reduces_to_lookup_at_k1() -> None:
    mc = _no_dropout()
    with torch.no_grad():
        mc.gamma.uniform_(-1, 1)
    classes, fracs = _topk(2, 1, (8, 8, 4))  # K=1, frac=1
    out = mc(classes, fracs)  # (2,embed,H,W,D)
    ref = (mc.embedding(classes[:, 0]) * mc.gamma).permute(
        0, 4, 1, 2, 3
    )  # plain lookup ×gamma
    assert torch.allclose(out, ref, atol=1e-6), (out - ref).abs().max().item()


def test_equivalence_to_soft_onehot_conv() -> None:
    """e_mask == channel_mapper(soft_one_hot): a 1x1 conv (bias=False) with weight = (E*gamma)^T."""
    mc = _no_dropout()
    with torch.no_grad():
        mc.gamma.uniform_(-1, 1)
    classes, fracs = _topk(2, 4, (8, 8, 4))
    out = mc(classes, fracs)  # (B,embed,H,W,D)
    b, k, h, w, d = classes.shape
    p = torch.zeros(
        b, h, w, d, NUM_CLASSES
    )  # soft one-hot (accumulate fracs at class ids)
    p.scatter_add_(-1, classes.permute(0, 2, 3, 4, 1), fracs.permute(0, 2, 3, 4, 1))
    w_eff = mc.embedding.weight * mc.gamma  # (num_classes, embed)
    ref = torch.einsum("bhwdc,ce->behwd", p, w_eff)  # 1x1 conv on the soft one-hot
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max().item()


def test_class_dropout_makes_partial_mask_not_null() -> None:
    """Per-class dropout drops organs to BACKGROUND (partial mask), never to the null."""
    mc = MaskConditioner(NUM_CLASSES, EMBED_DIM, class_dropout=1.0)
    with torch.no_grad():
        mc.gamma.fill_(1.0)
    mc.train()
    classes, fracs = _topk(2, 4, (8, 8, 4))
    out = mc(
        classes, fracs
    )  # every organ dropped -> all class ids become background(0)
    assert torch.count_nonzero(out) > 0, (
        "per-class dropout -> background embedding is NONZERO (not null)"
    )
    spatial_std = out.view(2, EMBED_DIM, -1).std(dim=2).max().item()
    assert spatial_std < 1e-5, "all-background => spatially uniform embedding"
