"""Report2CT-Wan + Wan-latent mask LightningDataModule.

Thin subclass of :class:`~src.data.report2ct_datamodule.Report2CTDataModule` for the
``report2ct_wan_mask`` variant. It emits everything the plain-wan module does — ``image``
(Wan image latent ``(16, 64, 64, 64)``), ``spacing``, ``context_f``/``context_i`` (2560-d
findings+impression) — PLUS a ``mask_latent`` ``(16, 64, 64, 64)`` loaded from the precomputed
``<id>_mask_emb.nii.gz`` (a frozen-Wan encode of the painted 4-organ GT mask; see
``scripts/precompute_wan_mask_latents.py``).

The mask latent lives in the SAME normalized Wan latent space as the image latent (same VAE,
same per-channel normalization), so ``Report2CTModule._shared_forward`` channel-concatenates it
onto the noisy latent at the UNet input (in_channels 16 → 32). The datalist is produced by
``scripts/build_report2ct_datalist.py --mask-dir ...`` (adds the ``"mask_latent"`` entry key).
"""

from __future__ import annotations

import monai
from monai.transforms import Compose

from src.data.report2ct_datamodule import Report2CTDataModule, _build_transforms


class Report2CTWanMaskDataModule(Report2CTDataModule):
    """Report2CT-Wan datamodule that additionally loads the precomputed Wan mask latent."""

    def _transforms(self) -> Compose:
        """Base report2ct transforms + load the ``mask_latent`` NIfTI (HWDC → ``(16, 64, 64, 64)``)."""
        base = _build_transforms(self.spacing_multiplier)
        return Compose(
            [
                *base.transforms,
                monai.transforms.LoadImaged(keys=["mask_latent"]),  # (64,64,64,16)
                monai.transforms.EnsureChannelFirstd(
                    keys=["mask_latent"]
                ),  # (16,64,64,64)
            ]
        )


__all__ = ["Report2CTWanMaskDataModule"]
