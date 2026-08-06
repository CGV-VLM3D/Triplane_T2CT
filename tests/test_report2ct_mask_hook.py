"""Integration tests for the Report2CTModule mask-conditioning hook (M2).

Verifies:
  * mask_conditioner=None => byte-compatible with the text-only baseline (UNet in=4, no "mask" key);
  * with a conditioner, the UNet INPUT is 4+embed_dim channels but the OUTPUT stays 4 channels —
    i.e. the mask is concatenated at the input only and never predicted/noised (structural proof
    of the noise/conditioning separation, via forward hooks);
  * configure_optimizers includes the conditioner's params (the "optimizer gap" fix);
  * gamma receives gradient so the mask path actually trains.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)

from src.baselines.rflow import RFlowScheduler
from src.models.components.mask_conditioner import MaskConditioner
from src.models.report2ct_module import Report2CTModule

EMBED = 8
GRID = (32, 32, 16)


def _build_unet(in_ch: int) -> DiffusionModelUNetMaisi:
    return DiffusionModelUNetMaisi(
        spatial_dims=3,
        in_channels=in_ch,
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


def _scheduler() -> RFlowScheduler:
    return RFlowScheduler(
        num_train_timesteps=1000,
        scale=1.4,
        use_timestep_transform=True,
        use_discrete_timesteps=False,
        sample_method="uniform",
    )


def _build_module(mask: bool):
    unet = _build_unet(4 + EMBED if mask else 4)
    mc = MaskConditioner(118, EMBED, class_dropout=0.0) if mask else None
    m = Report2CTModule(unet=unet, noise_scheduler=_scheduler(), mask_conditioner=mc)
    m._trainer = SimpleNamespace(world_size=1, estimated_stepping_batches=100)
    m.log = lambda *a, **k: None
    return m, unet, mc


def _batch(mask: bool) -> dict:
    b = {
        "image": torch.randn(1, 4, *GRID) * 0.98,
        "context": torch.randn(1, 1, 768),
        "spacing": torch.tensor([[0.75, 0.75, 3.0]]),
    }
    if mask:
        k = 4
        b["mask_classes"] = torch.randint(0, 118, (1, k, *GRID))
        b["mask_fracs"] = torch.softmax(torch.randn(1, k, *GRID), dim=1)
    return b


def test_mask_off_backward_compatible() -> None:
    m, _, _ = _build_module(mask=False)
    loss = m.training_step(_batch(mask=False), 0)
    loss.backward()
    assert torch.isfinite(loss)


def test_mask_concat_input12_output4() -> None:
    """UNet sees 4+embed channels in, predicts only the 4-channel latent out (noise separation)."""
    m, unet, _ = _build_module(mask=True)
    cap: dict = {}

    def pre_hook(_mod, _args, kwargs):
        cap["in"] = tuple(kwargs["x"].shape)

    def fwd_hook(_mod, _inp, out):
        cap["out"] = tuple(out.shape)

    unet.register_forward_pre_hook(pre_hook, with_kwargs=True)
    unet.register_forward_hook(fwd_hook)

    loss = m.training_step(_batch(mask=True), 0)
    loss.backward()
    assert cap["in"] == (1, 4 + EMBED, *GRID), cap["in"]  # mask concatenated onto input
    assert cap["out"] == (1, 4, *GRID), cap["out"]  # only the latent is predicted
    assert torch.isfinite(loss)


def test_optimizer_includes_conditioner() -> None:
    m_on, unet_on, mc = _build_module(mask=True)
    opt_on = m_on.configure_optimizers()["optimizer"]
    n_on = sum(p.numel() for g in opt_on.param_groups for p in g["params"])
    n_expect = sum(p.numel() for p in unet_on.parameters()) + sum(
        p.numel() for p in mc.parameters()
    )
    assert n_on == n_expect, (n_on, n_expect)

    m_off, unet_off, _ = _build_module(mask=False)
    opt_off = m_off.configure_optimizers()["optimizer"]
    n_off = sum(p.numel() for g in opt_off.param_groups for p in g["params"])
    assert n_off == sum(p.numel() for p in unet_off.parameters())


def test_conditioner_gamma_learns() -> None:
    """gamma moves off its zero init. NOTE: on step 1 gamma's grad is exactly 0 — the UNet's
    zero-init output conv blocks ALL upstream gradients (gamma included), identical to the
    Unit-0 finding. From step 2 (output conv != 0) the gradient reaches gamma. One-step warmup.
    """
    m, unet, mc = _build_module(mask=True)
    opt = torch.optim.Adam(list(unet.parameters()) + list(mc.parameters()), lr=1e-3)
    gamma0 = mc.gamma.detach().clone()
    grads: list[float] = []
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = m.training_step(_batch(mask=True), 0)
        loss.backward()
        grads.append(
            float(mc.gamma.grad.abs().sum()) if mc.gamma.grad is not None else 0.0
        )
        opt.step()
    assert grads[0] == 0.0, (
        f"step-1 gamma grad must be 0 (zero-init output conv): {grads}"
    )
    assert grads[1] > 0.0 or grads[2] > 0.0, (
        f"gamma must receive gradient by step 2: {grads}"
    )
    assert not torch.allclose(mc.gamma.detach(), gamma0), (
        "gamma must move off its zero init"
    )
