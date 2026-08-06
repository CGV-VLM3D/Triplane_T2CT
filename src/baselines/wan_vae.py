"""Wan2.1 VAE adapter — CT volume ⇄ Wan latent, for the report2ct_wan latent substrate.

Wraps the Wan2.1 video VAE (`diffusers.AutoencoderKLWan`, 8×8×4 spatial/temporal compression,
latent channels = 16) as a frozen encode/decode utility. This is the substrate for
`report2ct_wan` — the exact twin of `report2ct_text2ct` with the VAE swapped from text2ct's
MAISI-family VAE to Wan2.1.

Why Wan2.1 (vs the foundation-vae reference):
  * foundation-vae `Reconstruction/Wan/recon_ct_window_wan2.1.py` loads the **native** Wan VAE
    (`wan.modules.vae2_1.Wan2_1_VAE(vae_pth="Wan2.1_VAE.pth")`). The diffusers
    `AutoencoderKLWan` from ``Wan-AI/Wan2.1-T2V-1.3B-Diffusers`` wraps the **same** Wan2.1 VAE
    weights (8×8×4, ch16) but avoids the native flash_attn install. All Wan2.1 models
    (T2V/I2V/S2V/VACE/Animate + Wan2.2 A14B) share this one VAE; only Wan2.2 TI2V-5B uses the
    higher-compression 16×16×4 VAE.
  * CT preprocessing is version-agnostic and copied from foundation-vae `recon_ct.py` /
    `recon_ct_window_wan2.1.py`: HU clip[-1000,1000] → /1000 → [-1,1], grayscale repeated to
    the VAE's 3 input channels; on decode, mean over the 3 output channels → *1000 → HU.

Latent normalization (baseline-clone #1 — activation-range handoff):
  diffusers `AutoencoderKLWan.encode()` returns the **raw** posterior — it does NOT apply the
  per-channel `latents_mean`/`latents_std`; the WanPipeline does. So we apply it here so the
  stored latent lives in Wan's native diffusion latent space (each channel ~zero-mean/unit-std).
  This mirrors the native `Wan2_1_VAE.encode`, which folds the same `(mu-mean)/std` into its
  forward (vae2_1.py:537). Report2CTModule's single scalar `scale_factor=1/std` then lands ~1.0
  on top (a near no-op), which is correct — the two normalizations are different layers (per-
  channel VAE vs global diffusion) and compose without conflict.

Geometry: the caller (precompute script) resamples each CT to a FIXED (T, H, W) so the latent
grid is fixed for diffusion training. Recommended 125×512×512 → latent (16, 32, 64, 64):
`T_lat = 1 + (T-1)//4`, and T=125 (≡1 mod 4) makes 125→32→125 lossless while keeping every
latent spatial dim divisible by 8 (MAISI-UNet needs it). See docs/wan_latent_runbook.md.

CUDA + fp32 only: bf16 breaks Wan reconstruction; a full-volume encode is memory-heavy, so all
calls run under `torch.inference_mode()` (memory `inference-mode-for-gpu-smoke`). This adapter
lives in the dedicated `wan` conda env (diffusers ≥ 0.34); it is NOT imported by the main
training env (diffusers 0.31), which trains on the precomputed latents.
"""

from __future__ import annotations

import logging
from typing import Final

import torch
from einops import rearrange

log = logging.getLogger(__name__)

# Default diffusers repo for the Wan2.1 VAE. Swap to a Wan2.2 repo (+ adjust the expected
# latent geometry) for the follow-up experiment — that is the only model-side change.
DEFAULT_WAN_HF_ID: Final[str] = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN_INPUT_CHANNELS: Final[int] = 3  # video VAE expects 3-channel (RGB) frames
HU_CLIP: Final[float] = 1000.0  # HU clipped to [-1000, 1000] then scaled to [-1, 1]


class WanVAE:
    """Frozen Wan2.1 VAE wrapper exposing CT-HU ⇄ latent encode/decode.

    Args:
        hf_id: HuggingFace repo holding the Wan VAE (loaded with ``subfolder="vae"``).
        device: CUDA device string (Wan VAE is CUDA + fp32 only).
        subfolder: HF subfolder for the VAE weights (``"vae"`` for the T2V repos).
    """

    def __init__(
        self,
        hf_id: str = DEFAULT_WAN_HF_ID,
        device: str = "cuda:0",
        subfolder: str = "vae",
    ) -> None:
        self.hf_id = hf_id
        self.device = device
        self.subfolder = subfolder
        self._vae: torch.nn.Module | None = None
        # Per-channel normalization constants, filled in _ensure_built. Shape (1, 16, 1, 1, 1).
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    # -- build ---------------------------------------------------------------

    def _ensure_built(self) -> None:
        """Lazily load the Wan VAE (frozen, eval, fp32) and cache its latents_mean/std."""
        if self._vae is not None:
            return
        from diffusers import AutoencoderKLWan  # noqa: PLC0415 — only in the `wan` env

        vae = AutoencoderKLWan.from_pretrained(
            self.hf_id, subfolder=self.subfolder, torch_dtype=torch.float32
        )
        vae.requires_grad_(False).eval().to(self.device)

        # Per-channel latents_mean/std (length z_dim=16) — used for the manual normalization
        # that diffusers leaves to the pipeline. View as (1, C, 1, 1, 1) for broadcast over
        # the 5D latent (B, C, T, H, W).
        z_dim = len(vae.config.latents_mean)  # 16
        self._mean = torch.tensor(
            vae.config.latents_mean, dtype=torch.float32, device=self.device
        ).view(1, z_dim, 1, 1, 1)
        self._std = torch.tensor(
            vae.config.latents_std, dtype=torch.float32, device=self.device
        ).view(1, z_dim, 1, 1, 1)
        self._vae = vae
        log.info("Loaded Wan VAE %s (z_dim=%d) on %s", self.hf_id, z_dim, self.device)

    @property
    def latent_channels(self) -> int:
        """Number of latent channels (16 for Wan2.1)."""
        self._ensure_built()
        return self._mean.shape[1]

    # -- encode / decode -----------------------------------------------------

    @torch.inference_mode()
    def encode(self, ct_hu: torch.Tensor) -> torch.Tensor:
        """Encode one CT volume (HU) to a normalized Wan latent.

        Mirrors foundation-vae recon_ct.py preprocessing, then applies the per-channel
        latents_mean/std that diffusers leaves to the pipeline.

        Args:
            ct_hu: single CT volume in Hounsfield units, shape ``(T, H, W)`` = (depth, height,
                width). The caller fixes the geometry (e.g. 125×512×512); H and W must be
                divisible by 8.

        Returns:
            Normalized latent ``(16, T_lat, H//8, W//8)`` on CPU float32, where
            ``T_lat = 1 + (T-1)//4`` (e.g. 125 → 32).
        """
        self._ensure_built()
        x = ct_hu.to(self.device, torch.float32)
        x = torch.clamp(x, -HU_CLIP, HU_CLIP) / HU_CLIP  # (T, H, W) in [-1, 1]
        # grayscale slice -> 3-channel frame, then to VAE layout (C, T, H, W)
        x = x[:, None].repeat(1, WAN_INPUT_CHANNELS, 1, 1)  # (T, 3, H, W)
        x = rearrange(x, "t c h w -> c t h w")  # (3, T, H, W)
        x = x[None]  # (1, 3, T, H, W)

        z = self._vae.encode(x).latent_dist.mode()  # (1, 16, T_lat, H//8, W//8)
        z = (z - self._mean) / self._std  # per-channel normalize -> Wan diffusion space
        return z[0].cpu()  # (16, T_lat, H//8, W//8)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a normalized Wan latent back to a CT volume (HU).

        Inverse of :meth:`encode`: de-normalize, decode, average the 3 output channels back to
        grayscale, and map [-1, 1] → HU (baseline-clone #6 — Wan uses *1000, not MAISI's
        *2000-1000).

        Args:
            z: normalized latent ``(16, T_lat, h, w)`` (as produced by :meth:`encode`).

        Returns:
            Reconstructed CT volume in HU, shape ``(T, H, W)`` = ``(1+(T_lat-1)*4, h*8, w*8)``,
            on CPU float32 (caller casts to int16 for saving).
        """
        self._ensure_built()
        z = z.to(self.device, torch.float32)[None]  # (1, 16, T_lat, h, w)
        z = z * self._std + self._mean  # de-normalize back to raw VAE latent
        x = self._vae.decode(z).sample  # (1, 3, T, h*8, w*8)
        x = x[0].mean(0)  # channel-mean -> (T, H, W)
        x = torch.clamp(x, -1.0, 1.0) * HU_CLIP  # [-1, 1] -> HU
        return x.cpu()  # (T, H, W)


__all__ = ["WanVAE", "DEFAULT_WAN_HF_ID"]
