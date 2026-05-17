"""Tier-0 sanity: each registered model must overfit a single random batch.

Karpathy's "overfit one batch" recipe — if a model can't drive the loss down
on a single tiny batch, the pipeline (gradient flow, loss, optimizer wiring)
is broken before any real training matters.

Assertion is a RELATIVE loss-reduction check (final < 0.3 * initial) rather
than an absolute threshold. The point is to verify autograd/optimizer/forward
plumb together end-to-end, not that a 100-step CPU run produces a publication-
quality recon. Models are intentionally tiny (latent 8x8x4) so both tests
finish in < 30 s on CPU even with another GPU job running on the same box.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# Keep CPU usage low so this can run while a GPU training/precompute job is on
# the same machine. Setting before importing torch's heavy submodules.
torch.set_num_threads(2)

# Resolve src.models and the deferred reference/models/trivae2 import path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reference"))

from src.models import MODEL_REGISTRY  # noqa: E402

LATENT_SHAPE = (8, 8, 4)
N_STEPS = 250
LR = 5e-3
DECREASE_RATIO = 0.3  # final must be < 30% of initial
SEED = 0


def _overfit_loop(model: torch.nn.Module, x: torch.Tensor) -> tuple[float, float]:
    """Run N_STEPS Adam updates on a single fixed batch.

    Returns (initial_loss, final_loss). Initial loss is the L1 BEFORE the
    first optimizer step (with the freshly-initialized model).
    """
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    initial = None
    final = float("inf")
    for step in range(N_STEPS):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        # Both TriplaneAE and TriVQAEConv return (recon, _aux).
        recon = out[0] if isinstance(out, tuple) else out
        loss = F.l1_loss(recon, x)
        if initial is None:
            initial = float(loss.detach())
        loss.backward()
        opt.step()
        final = float(loss.detach())
    return initial, final  # type: ignore[return-value]


@pytest.mark.tier0
def test_overfit_triplane_ae_single_batch():
    torch.manual_seed(SEED)
    model = MODEL_REGISTRY["TriplaneAE"](
        in_channels=4,
        emb_dim=48,
        n_layers=2,
        n_heads=4,
        out_channels=8,
        decoder_hidden=16,
        latent_shape=LATENT_SHAPE,
        patch_size=2,
    )
    x = torch.randn(1, 4, *LATENT_SHAPE)
    initial, final = _overfit_loop(model, x)
    print(f"\nTriplaneAE  initial L1={initial:.4f}  final L1={final:.4f}")
    assert final < initial * DECREASE_RATIO, (
        f"TriplaneAE did not overfit: final L1={final:.4f}, "
        f"initial={initial:.4f}, ratio={final / initial:.3f} >= {DECREASE_RATIO}"
    )


@pytest.mark.tier0
def test_overfit_trivqaeconv_single_batch():
    torch.manual_seed(SEED)
    # Construct the underlying impl directly so we don't need a Hydra cfg here.
    from models.trivae2 import TriVQAEConv as TriVQAEImpl

    model = TriVQAEImpl(
        in_channels=4,
        out_channels=4,
        latent_shape=LATENT_SHAPE,
        encoder_channels=(8, 8),
        blocks_per_stage=1,
        plane_channels=8,
        final_hidden_channels=8,
        final_depth=2,
    )
    x = torch.randn(1, 4, *LATENT_SHAPE)
    initial, final = _overfit_loop(model, x)
    print(f"\nTriVQAEConv initial L1={initial:.4f}  final L1={final:.4f}")
    assert final < initial * DECREASE_RATIO, (
        f"TriVQAEConv did not overfit: final L1={final:.4f}, "
        f"initial={initial:.4f}, ratio={final / initial:.3f} >= {DECREASE_RATIO}"
    )
