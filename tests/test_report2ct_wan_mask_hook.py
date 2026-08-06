"""Integration tests for the report2ct_wan_mask (Wan mask-latent concat) hook.

Verifies the ``batch["mask_latent"]`` branch of ``Report2CTModule._shared_forward``:
  * no ``mask_latent`` key ⇒ plain 16-ch image latent at the UNet input (baseline path);
  * with ``mask_latent`` ⇒ the UNet INPUT is 32 channels (16 image + 16 mask) but the OUTPUT
    stays 16 (the mask is concatenated at the input only, never noised/predicted) — structural
    proof via forward hooks;
  * the mask is concatenated RAW (no MaskConditioner, no extra params in the optimizer);
  * the eval sampler loads a HWDC mask latent to ``(1, 16, 64, 64, 64)`` = (C, H, W, D).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)

from src.baselines.rflow import RFlowScheduler
from src.models.report2ct_module import Report2CTModule

GRID = (16, 16, 16)  # small latent grid for a fast CPU forward
C = 16  # Wan latent channels


def _build_unet(in_ch: int) -> DiffusionModelUNetMaisi:
    return DiffusionModelUNetMaisi(
        spatial_dims=3,
        in_channels=in_ch,
        out_channels=C,
        num_channels=[64, 128, 256, 512],
        num_res_blocks=2,
        attention_levels=[False, False, True, True],
        num_head_channels=[0, 0, 32, 32],
        cross_attention_dim=2560,
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
    unet = _build_unet(2 * C if mask else C)
    m = Report2CTModule(unet=unet, noise_scheduler=_scheduler())
    m._trainer = SimpleNamespace(world_size=1, estimated_stepping_batches=100)
    m.log = lambda *a, **k: None
    return m, unet


def _batch(mask: bool) -> dict:
    b = {
        "image": torch.randn(1, C, *GRID) * 0.98,
        "context": torch.randn(1, 2, 2560),
        "spacing": torch.tensor([[75.0, 75.0, 150.0]]),
    }
    if mask:
        b["mask_latent"] = torch.randn(
            1, C, *GRID
        )  # (1, 16, H, W, D) raw Wan mask latent
    return b


def test_mask_latent_off_baseline() -> None:
    """No mask_latent key ⇒ plain 16-ch latent path, finite loss."""
    m, _ = _build_module(mask=False)
    loss = m.training_step(_batch(mask=False), 0)
    loss.backward()
    assert torch.isfinite(loss)


def test_mask_latent_concat_input32_output16() -> None:
    """UNet sees 32 channels in (16 image + 16 mask), predicts only the 16-channel latent out."""
    m, unet = _build_module(mask=True)
    cap: dict = {}

    def pre_hook(_mod, _args, kwargs):
        cap["in"] = tuple(kwargs["x"].shape)

    def fwd_hook(_mod, _inp, out):
        cap["out"] = tuple(out.shape)

    unet.register_forward_pre_hook(pre_hook, with_kwargs=True)
    unet.register_forward_hook(fwd_hook)

    loss = m.training_step(_batch(mask=True), 0)
    loss.backward()
    assert cap["in"] == (1, 2 * C, *GRID), cap["in"]  # mask concatenated onto input
    assert cap["out"] == (1, C, *GRID), cap["out"]  # only the image latent predicted
    assert torch.isfinite(loss)


def test_optimizer_has_no_mask_params() -> None:
    """No MaskConditioner ⇒ optimizer holds exactly the UNet params (mask is not trained)."""
    m, unet = _build_module(mask=True)
    opt = m.configure_optimizers()["optimizer"]
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_opt == sum(p.numel() for p in unet.parameters())


def test_sampler_loads_mask_latent_chwd(tmp_path: Path) -> None:
    """Report2CTWanMaskLatentSampler._case_mask_latent: HWDC nii → (1, 16, 64, 64, 64) (C,H,W,D)."""
    from src.eval.samplers.report2ct_wan import Report2CTWanMaskLatentSampler

    # A fake mask latent stored in the precompute layout: HWDC (H, W, D, C).
    hwdc = np.random.randn(8, 8, 8, C).astype(np.float32)
    scan_id = "valid_9999_a_1"
    nib.save(
        nib.Nifti1Image(hwdc, np.eye(4)), str(tmp_path / f"{scan_id}_mask_emb.nii.gz")
    )

    s = Report2CTWanMaskLatentSampler.__new__(Report2CTWanMaskLatentSampler)
    s.mask_dir = tmp_path
    s._device = torch.device("cpu")
    from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

    s._mask_loader = Compose(
        [LoadImaged(keys="mask_latent"), EnsureChannelFirstd(keys="mask_latent")]
    )
    ml = s._case_mask_latent(SimpleNamespace(scan_id=scan_id))
    assert tuple(ml.shape) == (1, C, 8, 8, 8), ml.shape  # (1, C, H, W, D)
    # channel-first == the loaded HWDC moved to CHWD (EnsureChannelFirstd), matching training.
    assert torch.allclose(ml[0], torch.from_numpy(hwdc).permute(3, 0, 1, 2))
