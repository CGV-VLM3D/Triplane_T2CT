"""Report2CT inference sampler for VLM3D evaluation.

Loads a trained Report2CTModule Lightning checkpoint, runs the RFlow denoising loop,
and saves generated volumes as .mha files.

Architecture kwargs match configs/model/report2ct.yaml exactly (annotated).
Flash attention is always disabled: Blackwell sm_120 crashes FA3 backward.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)
from monai.inferers.inferer import SlidingWindowInferer
from tqdm import tqdm

from src.baselines.maisi import load_frozen
from src.baselines.report2ct_text_encoder import Report2CTTextEncoder
from src.baselines.rflow import RFlowScheduler
from src.eval.samplers._orient import ras_to_lps
from src.eval.samplers.base import AbstractSampler, EvalCase

log = logging.getLogger(__name__)

# Latent spatial dims: (C=4, H=120, W=120, D=64)
_LATENT_SHAPE = (4, 120, 120, 64)
_LATENT_VOL = 120 * 120 * 64  # for set_timesteps input_img_size
_SPACING_MULTIPLIER = 100.0  # matches Report2CTDataModule default

# Sliding-window decode params (upstream diff_model_infer_vlm3D.py:255-263)
_SW_ROI = (80, 80, 80)
_SW_OVERLAP = 0.4


class _ReconModel(torch.nn.Module):
    """Decode latent with scale_factor — mirrors upstream sample.py:ReconModel."""

    def __init__(self, autoencoder, scale_factor: float):
        """Wrap the frozen MAISI autoencoder with its learned latent scale factor.

        Args:
            autoencoder: frozen MAISI VAE (from ``load_frozen``).
            scale_factor: scalar by which latents were divided before encoding
                during training; divides ``z`` before decoding to undo that scaling.
        """
        super().__init__()
        self.autoencoder = autoencoder
        self.scale_factor = scale_factor

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a scaled latent to a reconstructed CT volume.

        Args:
            z: MAISI latent tensor ``(1, 4, 120, 120, 64)``.

        Returns:
            Reconstructed volume ``(1, 1, 480, 480, 256)`` in approximately ``[0, 1]``.
        """
        return self.autoencoder.decode_stage_2_outputs(z / self.scale_factor)


def _dynamic_infer(
    inferer: SlidingWindowInferer, model, images: torch.Tensor
) -> torch.Tensor:
    """Match upstream utils.py:dynamic_infer — clamp roi to spatial dims before inferring."""
    spatial_dims = images.shape[2:]
    orig_roi = inferer.roi_size
    adjusted_roi = [min(r, s) for r, s in zip(orig_roi, spatial_dims)]
    inferer.roi_size = adjusted_roi
    out = inferer(network=model, inputs=images)
    inferer.roi_size = orig_roi
    return out


def _build_unet(
    cross_attention_dim: int = 2560, in_channels: int = 4
) -> DiffusionModelUNetMaisi:
    """Instantiate UNet with kwargs from configs/model/report2ct.yaml.

    cross_attention_dim defaults to Report2CT's 2560; the fVLM-conditioned variant
    (src/eval/samplers/report2ct_fvlm.py) passes 256. in_channels defaults to 4 (plain
    latent); the mask-conditioned variant passes 4+embed_dim (LDM concat) — out_channels
    stays 4 (the UNet predicts only the latent).
    """
    return DiffusionModelUNetMaisi(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=4,
        num_channels=[64, 128, 256, 512],
        num_res_blocks=2,
        attention_levels=[False, False, True, True],
        num_head_channels=[0, 0, 32, 32],
        cross_attention_dim=cross_attention_dim,
        num_class_embeds=128,
        include_fc=True,
        include_spacing_input=True,
        include_top_region_index_input=False,
        include_bottom_region_index_input=False,
        resblock_updown=True,
        with_conditioning=True,
        use_flash_attention=False,  # disabled: Blackwell sm_120 FA3 crash
    )


def _load_checkpoint(
    ckpt_path: str | Path,
    device: torch.device,
    cross_attention_dim: int = 2560,
    in_channels: int = 4,
):
    """Load UNet weights and scale_factor from a Lightning .ckpt file.

    Returns (unet, scale_factor).
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    scale_factor: float = sd["scale_factor"].item()
    log.info("scale_factor from checkpoint: %.4f", scale_factor)

    unet = _build_unet(cross_attention_dim=cross_attention_dim, in_channels=in_channels)
    unet_sd = {k[len("unet.") :]: v for k, v in sd.items() if k.startswith("unet.")}
    missing, unexpected = unet.load_state_dict(unet_sd, strict=True)
    if missing or unexpected:
        log.warning(
            "UNet load_state_dict: missing=%s unexpected=%s", missing, unexpected
        )

    unet = unet.to(device).eval()
    for p in unet.parameters():
        p.requires_grad_(False)

    return unet, scale_factor


class Report2CTSampler(AbstractSampler):
    """Inference sampler using a trained Report2CT Lightning checkpoint.

    Args:
        ckpt_path: path to Lightning .ckpt (epoch_NNN_valloss_*.ckpt or last.ckpt).
        n_steps: number of RFlow denoising steps (default 100).
        modality_class_label: CT=1 (matches training config).
        spacing_mm: voxel spacing (mm) used BOTH as UNet conditioning and as the saved .mha
            affine. Defaults to upstream's (1.0, 1.0, 1.5) (diff_model_infer_vlm3D.py:95,320).
    """

    # Latent shape (C, H, W, D) the RFlow loop samples noise at. Report2CT's MAISI latents
    # are (4, 120, 120, 64); a subclass trained on a different VAE-latent geometry
    # (e.g. text2ct's (4, 128, 128, 32)) overrides this so generation matches its training.
    _latent_shape: tuple[int, int, int, int] = _LATENT_SHAPE

    def __init__(
        self,
        ckpt_path: str,
        n_steps: int = 100,
        modality_class_label: int = 1,
        spacing_mm: list[float] | None = None,
        cfg_scale: float = 1.0,
        name: str | None = None,  # absorbed from Hydra model config; not used
    ) -> None:
        """Store inference configuration; model weights are loaded lazily on first ``generate``.

        Args:
            ckpt_path: path to a Lightning ``.ckpt`` file containing ``unet.*`` weights
                and a ``scale_factor`` scalar in ``state_dict``.
            n_steps: number of RFlow denoising steps.
            modality_class_label: class-conditioning label fed to the UNet (CT=1,
                matching the training config).
            spacing_mm: voxel spacing (mm) used both as UNet spatial conditioning and
                as the affine stamped on the saved ``.mha``.  Defaults to
                ``[1.0, 1.0, 1.5]`` (upstream diff_model_infer_vlm3D.py:95, 320).
            cfg_scale: classifier-free guidance scale; ``1.0`` disables CFG.  Values
                above ``1.0`` improve CLIPScore but raise FID per the paper ablation.
        """
        self.ckpt_path = Path(ckpt_path)
        self.n_steps = n_steps
        self.modality_class_label = modality_class_label
        # Classifier-free guidance scale. 1.0 == no guidance == faithful reproduction
        # (upstream diff_model_infer_vlm3D.py has NO cfg). >1.0 enables CFG for the
        # paper's cfg variant; uncond branch uses zeros_like(context) to match training
        # cfg dropout (diff_model_train_vlm3D_2560_multi_text.py:296).
        # NOTE: per paper ablation, cfg RAISES FID (3.79→4.04→4.19) and only improves
        # CLIPScore — it is NOT a tool for closing the FID gap. Keep default 1.0.
        self.cfg_scale = float(cfg_scale)
        # Voxel spacing (mm) used BOTH as UNet conditioning and as the saved .mha affine —
        # they must match, since a spacing-conditional model generates at the spacing it is
        # told. Upstream diff_model_infer_vlm3D.py conditions (:95) and saves (:320) from one
        # config value (1.0, 1.0, 1.5); we mirror that. (OmegaConf ListConfig → plain list.)
        self.spacing_mm = (
            list(spacing_mm) if spacing_mm is not None else [1.0, 1.0, 1.5]
        )

        # Deferred: loaded once in generate()
        self._unet: DiffusionModelUNetMaisi | None = None
        self._scale_factor: float | None = None
        self._text_encoder: Report2CTTextEncoder | None = None
        self._autoencoder = None
        self._device: torch.device | None = None
        # Optional LDM-concat mask embedding (B, embed_dim, H, W, D), set per-case by a
        # mask-conditioned subclass; None ⇒ the UNet input is the plain 4-channel latent.
        self._mask_embed: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    #  Lazy initialisation                                                 #
    # ------------------------------------------------------------------ #

    def _init(self, device: torch.device) -> None:
        """Lazily load UNet, text encoders, and frozen MAISI VAE onto ``device``.

        No-ops if already initialised.  Disables Flash-Attention 3 on Blackwell
        (sm_120) to avoid the FA3 backward crash, matching ``Report2CTModule.setup()``.

        Args:
            device: target device for all model weights and inference tensors.
        """
        if self._unet is not None:
            return
        self._device = device

        # Disable FA3 on Blackwell (same as Report2CTModule.setup())
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

        log.info("Loading UNet from %s …", self.ckpt_path)
        self._unet, self._scale_factor = _load_checkpoint(self.ckpt_path, device)

        log.info("Loading text encoders …")
        self._text_encoder = Report2CTTextEncoder(device=device)

        log.info("Loading frozen MAISI VAE …")
        self._autoencoder = load_frozen(device=device)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _case_to_context(self, case: EvalCase) -> torch.Tensor:
        """Build the cross-attention context for one case.

        Overridable: the fVLM-conditioned subclass loads a precomputed per-organ
        embedding instead of encoding findings/impression text.
        """
        return self._encode_text(case.findings, case.impression)

    def _encode_text(self, findings: str, impression: str) -> torch.Tensor:
        """Encode findings + impression → context tensor (1, 2, 2560)."""
        ctx_f, ctx_i = self._text_encoder.encode_pair(findings, impression)
        # Both are (2560,) CPU tensors; promote to (1, 1, 2560) then cat
        ctx_f = ctx_f.unsqueeze(0).unsqueeze(0).to(self._device)  # (1, 1, 2560)
        ctx_i = ctx_i.unsqueeze(0).unsqueeze(0).to(self._device)  # (1, 1, 2560)
        return torch.cat([ctx_f, ctx_i], dim=1)  # (1, 2, 2560)

    def _make_spacing_tensor(self, spacing_mm: list[float]) -> torch.Tensor:
        """Build spacing tensor (1, 3) scaled by spacing_multiplier."""
        s = torch.tensor(spacing_mm, dtype=torch.float32) * _SPACING_MULTIPLIER
        return s.unsqueeze(0).to(self._device)  # (1, 3)

    @torch.no_grad()
    def _denoise(
        self, context: torch.Tensor, spacing_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Run RFlow denoising → clean latent (1, 4, 120, 120, 64).

        Wrapped in bf16 autocast — Report2CT was trained with `precision: bf16-mixed`,
        and MAISI VAE's MaisiGroupNorm3D internally casts to float16 (norm_float16: true
        in the bundle config), so autocast is required for end-to-end fp32 forward.
        """
        scheduler = RFlowScheduler(
            num_train_timesteps=1000,
            scale=1.4,
            use_timestep_transform=True,
            use_discrete_timesteps=False,
            sample_method="uniform",
        )
        # input_img_size = H*W*D of the latent space (matches training sample_timesteps);
        # timestep_transform requires a torch tensor for .pow(); kept on CPU because
        # set_timesteps then converts the timestep list via np.array.
        _, h, w, d = self._latent_shape  # (C, H, W, D)
        latent_numel = h * w * d
        scheduler.set_timesteps(
            self.n_steps,
            device=self._device,
            input_img_size=torch.tensor(float(latent_numel)),
        )

        z = torch.randn(
            1, *self._latent_shape, device=self._device, dtype=torch.float32
        )
        class_labels = torch.tensor([self.modality_class_label], device=self._device)

        # Mirror upstream diff_model_infer_vlm3D.py:215-216 — append 0 to next_timesteps
        # so the final step (t_99 → 0) is also taken.
        all_timesteps = scheduler.timesteps
        all_next_timesteps = torch.cat(
            (
                all_timesteps[1:],
                torch.tensor(
                    [0], dtype=all_timesteps.dtype, device=all_timesteps.device
                ),
            )
        )

        # Classifier-free guidance setup: batch the unconditional (zeros_like context)
        # and conditional passes together so wall-clock ≈ a single batch-1 forward.
        # noise_pred = uncond + cfg_scale * (cond - uncond).
        use_cfg = self.cfg_scale != 1.0
        if use_cfg:
            uncond_context = torch.zeros_like(context)
            context_b = torch.cat([uncond_context, context], dim=0)  # (2, 2, 2560)
            class_labels_b = class_labels.repeat(2)  # (2,)
            spacing_b = spacing_tensor.repeat(2, 1)  # (2, 3)

        for t, next_t in zip(all_timesteps, all_next_timesteps):
            # set_timesteps casts to float16; pass float() so scheduler.step's
            # dt arithmetic stays in float64 and z keeps its float32 dtype.
            t_scalar = float(t.item())
            next_t_scalar = float(next_t.item())
            t_tensor = torch.tensor(
                [t_scalar], device=self._device, dtype=torch.float32
            )
            # LDM concat: append the (fixed) mask embedding to the UNet input each step; the
            # mask is never noised (matches Report2CTModule training). None ⇒ plain latent.
            x_single = (
                z
                if self._mask_embed is None
                else torch.cat(
                    [z, self._mask_embed], dim=1
                )  # (1, 4+embed_dim, H, W, D)
            )
            if use_cfg:
                pred_b = self._unet(
                    x=torch.cat([x_single, x_single], dim=0),
                    timesteps=t_tensor.repeat(2),
                    context=context_b,
                    class_labels=class_labels_b,
                    spacing_tensor=spacing_b,
                )
                pred_uncond, pred_cond = pred_b.chunk(2, dim=0)
                pred = pred_uncond + self.cfg_scale * (pred_cond - pred_uncond)
            else:
                pred = self._unet(
                    x=x_single,
                    timesteps=t_tensor,
                    context=context,
                    class_labels=class_labels,
                    spacing_tensor=spacing_tensor,
                )
            z = scheduler.step(pred, t_scalar, z, next_t_scalar)[0]

        return z.to(torch.float32)

    @torch.no_grad()
    def _decode_to_hu(self, z: torch.Tensor) -> np.ndarray:
        """Decode latent → HU int16 numpy array using SlidingWindowInferer.

        Mirrors upstream diff_model_infer_vlm3D.py:255-269 — ReconModel(z) over a
        SlidingWindowInferer(roi=80³, overlap=0.4, mode="gaussian"), then map
        [0, 1] → HU [-1000, 1000] and clip.
        """
        z = z.to(torch.float32)  # MAISI VAE weights are float32

        recon_model = _ReconModel(self._autoencoder, self._scale_factor).to(
            self._device
        )
        inferer = SlidingWindowInferer(
            roi_size=list(_SW_ROI),
            sw_batch_size=1,
            progress=False,
            mode="gaussian",
            overlap=_SW_OVERLAP,
            sw_device=self._device,
            device=self._device,
        )
        decoded = _dynamic_infer(inferer, recon_model, z)
        # decoded: (1, 1, 480, 480, 256) in approximately [0, 1]

        # upstream line 266-269: (data - b_min) / (b_max - b_min) * (a_max - a_min) + a_min
        # with b_min=0, b_max=1, a_min=-1000, a_max=1000 → data * 2000 - 1000
        vol = decoded.squeeze().cpu().float().numpy()
        hu = vol * 2000.0 - 1000.0
        hu = np.clip(hu, -1000.0, 1000.0).astype(np.int16)
        return hu  # (H, W, D)

    @staticmethod
    def _save_mha(hu: np.ndarray, spacing_mm: list[float], out_path: Path) -> None:
        """Save HU array as .mha. hu shape: (H, W, D) → SimpleITK (X=H, Y=W, Z=D)."""
        # SimpleITK GetImageFromArray expects (Z, Y, X); our hu is (H, W, D)
        # (H, W, D) == (X, Y, Z) in CT convention → transpose to (Z, Y, X) = (D, W, H)
        arr_zyx = hu.transpose(2, 1, 0)  # (D, W, H)
        # Decoder emits RAS content (Orientationd(RAS) precompute); flip in-plane to LPS so it
        # matches the LPS GT / real CT-RATE for eval + submission (plan: eval-wise-catmull).
        arr_zyx = ras_to_lps(arr_zyx)  # (Z, Y, X), X/Y reversed
        img = sitk.GetImageFromArray(arr_zyx)
        img.SetSpacing([float(s) for s in spacing_mm])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(img, str(out_path))

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        cases: list[EvalCase],
        out_dir: Path,
        device: torch.device,
    ) -> list[Path]:
        """Generate one .mha prediction per case."""
        self._init(device)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        for case in tqdm(cases, desc="Report2CT inference"):
            out_path = out_dir / f"{case.scan_id}.mha"
            if out_path.exists():
                log.info("Skipping %s (already exists)", out_path.name)
                written.append(out_path)
                continue

            context = self._case_to_context(case)
            # Condition AND stamp the same configured spacing (upstream (1.0,1.0,1.5)); a
            # spacing-conditional model generates at the spacing it is told, so the saved
            # affine must match the conditioning. (Not case.spacing_mm — see __init__.)
            spacing_tensor = self._make_spacing_tensor(self.spacing_mm)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z = self._denoise(context, spacing_tensor)
                hu = self._decode_to_hu(z)
            self._save_mha(hu, self.spacing_mm, out_path)
            written.append(out_path)
            log.info(
                "Saved %s  HU range [%.0f, %.0f]", out_path.name, hu.min(), hu.max()
            )

        return written


__all__ = ["Report2CTSampler"]
