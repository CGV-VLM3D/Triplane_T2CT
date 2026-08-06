"""Report2CTModule CFG context dropout: per-sample (text2ct) vs per-batch (report2ct).

per-sample (``cfg_per_sample=True``) drops a random SUBSET of a batch's contexts (text2ct
diff_model_train.py:335); per-batch (False, report2ct upstream :296-297) drops the whole batch
or none. Verified by hooking the context the UNet actually receives (post-dropout).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)

from src.baselines.rflow import RFlowScheduler
from src.models.report2ct_module import Report2CTModule

B, GRID = 8, (16, 16, 8)


def _unet() -> DiffusionModelUNetMaisi:
    return DiffusionModelUNetMaisi(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        num_channels=[64, 128, 256, 512],
        num_res_blocks=2,
        attention_levels=[False, False, True, True],
        num_head_channels=[0, 0, 32, 32],
        cross_attention_dim=768,
        num_class_embeds=128,
        include_fc=True,
        include_spacing_input=True,
        include_top_region_index_input=False,
        include_bottom_region_index_input=False,
        resblock_updown=True,
        with_conditioning=True,
        use_flash_attention=False,
    )


def _capture_contexts(per_sample: bool, n_steps: int = 12) -> list[torch.Tensor]:
    """Return the (B,1,768) context tensor the UNet received at each of n_steps training steps."""
    unet = _unet()
    m = Report2CTModule(
        unet=unet,
        noise_scheduler=RFlowScheduler(
            num_train_timesteps=1000,
            scale=1.4,
            use_timestep_transform=True,
            use_discrete_timesteps=False,
            sample_method="uniform",
        ),
        cfg_drop_prob=0.5,
        cfg_per_sample=per_sample,
    )
    m._trainer = SimpleNamespace(world_size=1)
    m.log = lambda *a, **k: None
    m.train()
    caps: list[torch.Tensor] = []
    unet.register_forward_pre_hook(
        lambda mod, args, kwargs: caps.append(kwargs["context"].detach().clone()),
        with_kwargs=True,
    )
    torch.manual_seed(0)
    for _ in range(n_steps):
        batch = {
            "image": torch.randn(B, 4, *GRID),
            "context": torch.randn(B, 1, 768)
            + 5.0,  # strongly nonzero so a drop => exact zeros
            "spacing": torch.rand(B, 3) + 0.5,
        }
        m.training_step(batch, 0)
    return caps


def _dropped_rows(c: torch.Tensor) -> torch.Tensor:
    """Boolean (B,) — True where a sample's whole context was zeroed."""
    return c.abs().sum(dim=(1, 2)) == 0


def test_cfg_per_sample_produces_partial_batches() -> None:
    caps = _capture_contexts(per_sample=True)
    partial = any(0 < int(_dropped_rows(c).sum()) < B for c in caps)
    assert partial, (
        "per-sample CFG must drop only a SUBSET of a batch's contexts at least once"
    )


def test_cfg_per_batch_is_all_or_nothing() -> None:
    caps = _capture_contexts(per_sample=False)
    assert not any(0 < int(_dropped_rows(c).sum()) < B for c in caps), (
        "per-batch CFG must never drop a partial batch"
    )
    # dropout actually fires over the run: some batches fully dropped AND some fully kept
    assert any(_dropped_rows(c).all() for c in caps), "expect >=1 fully-dropped batch"
    assert any((~_dropped_rows(c)).all() for c in caps), "expect >=1 fully-kept batch"
