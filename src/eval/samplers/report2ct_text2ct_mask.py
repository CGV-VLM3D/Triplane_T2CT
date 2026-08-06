"""report2ct_text2ct_mask inference sampler for VLM3D evaluation.

Same as ``Report2CTText2CTSampler`` (Report2CT backbone on text2ct's ``(4,128,128,32)`` latent
+ FrozenCLIP3D condition) PLUS the LDM-concat organ-mask conditioning: a precomputed top-K
latent-grid mask is embedded by the trained ``MaskConditioner`` (loaded from the same ckpt) and
concatenated onto the noisy latent at the UNet input every denoising step (UNet ``in_channels=8``,
``out_channels=4``). The mask is fixed per case (never noised), so ``e_mask`` is computed once in
``_case_to_context`` and read by the inherited ``_denoise`` via ``self._mask_embed``.

Masks come from ``scripts/precompute_mask_latentgrid.py`` (``<mask_dir>/<scan_id>.pt`` =
``{classes:(K,128,128,32) uint8, fracs:(K,128,128,32) fp16}``), keyed by the FULL vol id
(``valid_1000_a_1``) — unlike the CLIP3D cond ``.pt`` which is keyed by the collapsed scan id.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.baselines.maisi import load_frozen
from src.eval.samplers.base import EvalCase
from src.eval.samplers.report2ct import _build_unet
from src.eval.samplers.report2ct_text2ct import Report2CTText2CTSampler
from src.models.components.mask_conditioner import MaskConditioner

log = logging.getLogger(__name__)


class Report2CTText2CTMaskSampler(Report2CTText2CTSampler):
    """text2ct-latent Report2CT + FrozenCLIP3D cond + LDM-concat organ mask.

    Args:
        ckpt_path: Lightning ``.ckpt`` trained with ``experiment=report2ct_text2ct_mask``
            (UNet ``in_channels=8``, plus a ``mask_conditioner.*`` submodule).
        cond_dir: precomputed FrozenCLIP3D cond embeddings (``<scan_id>.pt``, each ``(1,768)``).
        mask_dir: precomputed top-K latent-grid masks (``<vol>.pt`` with ``classes``/``fracs``),
            e.g. ``data/text2ct_mask/valid`` from ``scripts/precompute_mask_latentgrid.py``.
    """

    _mask_embed_dim: int = 4  # UNet in_channels = latent 4 + this = 8

    def __init__(self, ckpt_path: str, cond_dir: str, mask_dir: str, **kwargs) -> None:
        super().__init__(ckpt_path=ckpt_path, cond_dir=cond_dir, **kwargs)
        self.mask_dir = Path(mask_dir)
        self._mask_conditioner: MaskConditioner | None = None

    def _init(self, device: torch.device) -> None:
        """Load the ``in_channels=8`` UNet + ``MaskConditioner`` (both from the ckpt) + MAISI VAE."""
        if self._unet is not None:
            return
        self._device = device

        # Disable FA3 on Blackwell (same as Report2CTModule.setup())
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

        log.info("Loading mask-cond UNet + MaskConditioner from %s …", self.ckpt_path)
        ckpt = torch.load(str(self.ckpt_path), map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        self._scale_factor = sd["scale_factor"].item()

        in_channels = self._latent_shape[0] + self._mask_embed_dim  # 4 + 4 = 8
        unet = _build_unet(
            cross_attention_dim=self.cross_attention_dim, in_channels=in_channels
        )
        unet_sd = {k[len("unet.") :]: v for k, v in sd.items() if k.startswith("unet.")}
        unet.load_state_dict(unet_sd, strict=True)
        self._unet = unet.to(device).eval()
        for p in self._unet.parameters():
            p.requires_grad_(False)

        mc_sd = {
            k[len("mask_conditioner.") :]: v
            for k, v in sd.items()
            if k.startswith("mask_conditioner.")
        }
        # num_classes from the trained embedding table (NOT hardcoded) — the same sampler serves
        # both the full-118 and the 4-organ-group (num_classes=5) checkpoints (baseline-clone #3).
        num_classes = mc_sd["embedding.weight"].shape[0]
        mc = MaskConditioner(num_classes=num_classes, embed_dim=self._mask_embed_dim)
        mc.load_state_dict(mc_sd, strict=True)
        self._mask_conditioner = mc.to(device).eval()
        for p in self._mask_conditioner.parameters():
            p.requires_grad_(False)

        log.info("Loading frozen MAISI VAE …")
        self._autoencoder = load_frozen(device=device)

    @torch.no_grad()
    def _case_to_context(self, case: EvalCase) -> torch.Tensor:
        """CLIP3D context (inherited) + compute this case's fixed mask embedding.

        Sets ``self._mask_embed`` ``(1, embed_dim, 128, 128, 32)`` (read by ``_denoise``) and
        returns the CLIP3D cross-attention context ``(1, 1, 768)``.
        """
        context = super()._case_to_context(case)  # CLIP3D (1, 1, 768)

        # Mask .pt is keyed by the FULL vol id (valid_1000_a_1), not the collapsed scan id.
        mp = self.mask_dir / f"{case.scan_id}.pt"
        if not mp.is_file():
            raise FileNotFoundError(
                f"mask missing: {mp}. Run scripts/precompute_mask_latentgrid.py "
                f"(--split valid_fixed) for this split first."
            )
        md = torch.load(mp, weights_only=True)
        classes = md["classes"].long().unsqueeze(0).to(self._device)  # (1, K, H, W, D)
        fracs = md["fracs"].float().unsqueeze(0).to(self._device)  # (1, K, H, W, D)
        # e_mask fixed across denoising steps; float32 to match z in _denoise's cat.
        self._mask_embed = self._mask_conditioner(
            classes, fracs
        ).float()  # (1, 4, H, W, D)
        return context


__all__ = ["Report2CTText2CTMaskSampler"]
