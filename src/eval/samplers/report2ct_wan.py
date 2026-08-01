"""Wan2.1-latent inference samplers for VLM3D evaluation (report2ct_wan).

Because the Wan VAE lives in a separate `wan` conda env (diffusers ≥ 0.34) while training +
RFlow sampling run in the main env (diffusers 0.31), the Wan eval path is split in two:

  1. `Report2CTWanLatentSampler` (MAIN env) — RFlow-denoise a clean Wan latent per case and save
     it as a `.npy` (NO decode — the main env has no Wan VAE). Same Report2CT backbone + 2560-d
     findings+impression conditioning as `Report2CTSampler`; only the latent geometry (16-channel
     Wan latent) and the "save latent instead of decode-to-HU" tail differ. Driven by
     `scripts/generate_wan_latents.py`.
  2. `scripts/decode_wan_latents.py` (WAN env) — `.npy` latent → Wan VAE decode → `.mha`.
  3. `Report2CTWanEvalSampler` (MAIN env) — the sampler `scripts/run_eval.py` instantiates; it is
     a pass-through that expects the `.mha` predictions to already be on disk (decoded in step 2)
     and returns them, so `CTGenEvaluator` can score. It never builds a UNet or VAE.

The UNet build differs from `report2ct.py._build_unet` only in in/out channels (16 vs 4), so we
carry a local 16-channel builder + checkpoint loader rather than editing the shared 4-channel
one (third_party/CLAUDE.md: surgical changes; mirrors the fvlm/clip3d subclass pattern).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import torch
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged
from tqdm import tqdm

from src.baselines.report2ct_text_encoder import Report2CTTextEncoder
from src.baselines.rflow import RFlowScheduler
from src.data.report2ct_wan_seq_datamodule import (
    ENCODER_HIDDEN_DIMS,
    SEQ_LEN_FINDINGS,
    SEQ_LEN_IMPRESSION,
    _pad_or_truncate,
)
from src.eval.samplers.base import AbstractSampler, EvalCase
from src.eval.samplers.report2ct import Report2CTSampler
from src.models.components.maisi_unet_text_pooled import (
    DiffusionModelUNetMaisiTextPooled,
)
from src.models.components.text_sequence_projector import TextSequenceProjector

log = logging.getLogger(__name__)

# Wan2.1 latent in the UNet's (C, H, W, D) convention: 16 channels, 64×64 in-plane, 64 depth
# (from a 512×512×253 CT → Wan 8×8×4 → (16, 64, 64, 64) in Wan (C,T_lat,H,W); the datamodule's
# EnsureChannelFirstd on the (H,W,D,C) latent yields (C,H,W,D)=(16,64,64,64)).
WAN_LATENT_SHAPE = (16, 64, 64, 64)
WAN_LATENT_VOL = (
    64 * 64 * 64
)  # set_timesteps input_img_size (matches training sample_timesteps)


def _noise_seed(case: EvalCase) -> int:
    """Deterministic RNG seed for one case's initial noise, from (target scan, manifest seed).

    Keyed on the TARGET, never on ``sample_id``: every condition of one target must start from
    the SAME latent noise, so that the difference between a ``gt`` volume and a swapped-mask
    volume is the intervention and not a different random draw. Plain runs have ``case.seed is
    None``, are never seeded here, and keep drawing from the global RNG exactly as before.
    """
    digest = hashlib.sha256(f"{case.scan_id}|{case.seed}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _build_wan_unet(
    cross_attention_dim: int = 2560,
    in_channels: int = 16,
    text_pooled: bool = False,
) -> DiffusionModelUNetMaisi:
    """UNet with configs/model/report2ct_wan.yaml kwargs (out=16, cross_attention_dim=2560).

    Mirrors report2ct._build_unet; report2ct_wan conditions on the report2ct 2560-d findings+
    impression context (NOT CLIP3D 768), so cross_attention_dim defaults to 2560. in_channels
    defaults to 16 (plain image latent); the mask-conditioned variant passes 32 (16 image + 16
    Wan mask latent, concat) — out_channels stays 16 (the UNet predicts only the image latent).

    Args:
        text_pooled: build ``DiffusionModelUNetMaisiTextPooled`` (adds the pooled-text ``emb``
            path + masked cross-attention, see
            ``src/models/components/maisi_unet_text_pooled.py``) instead of the stock class.
            ``report2ct_wan_seq`` (arm B/C) checkpoints need this; every other checkpoint in
            this file does not.
    """
    cls = DiffusionModelUNetMaisiTextPooled if text_pooled else DiffusionModelUNetMaisi
    return cls(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=16,
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


def _load_wan_checkpoint(
    ckpt_path: str | Path,
    device: torch.device,
    in_channels: int = 16,
    text_pooled: bool = False,
):
    """Load UNet weights + scale_factor (+ optional learned null mask / text projector) from a
    Lightning .ckpt.

    Returns ``(unet, scale_factor, no_mask_embed, text_projector)``. ``no_mask_embed`` is the
    learned null mask tensor ``(16,)`` for dual-CFG checkpoints (report2ct_wan_mask_v2, trained
    with mask_cfg=true) or ``None``. ``text_projector`` is a ``TextSequenceProjector`` loaded from
    the checkpoint's ``text_projector.*`` keys (report2ct_wan_seq, arm B/C) or ``None`` — both are
    ``None`` for every checkpoint that predates these features, so this signature change is safe
    for the mask / mask_v2 subclasses that inherit ``_init`` unchanged.

    Args:
        in_channels: 16 (plain report2ct_wan) or 32 (report2ct_wan_mask[_v2], image+mask concat).
        text_pooled: see ``_build_wan_unet``.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    scale_factor: float = sd["scale_factor"].item()
    log.info("scale_factor from checkpoint: %.4f", scale_factor)

    unet = _build_wan_unet(in_channels=in_channels, text_pooled=text_pooled)
    unet_sd = {k[len("unet.") :]: v for k, v in sd.items() if k.startswith("unet.")}
    missing, unexpected = unet.load_state_dict(unet_sd, strict=True)
    if missing or unexpected:
        log.warning(
            "UNet load_state_dict: missing=%s unexpected=%s", missing, unexpected
        )
    unet = unet.to(device).eval()
    for p in unet.parameters():
        p.requires_grad_(False)

    no_mask_embed = sd.get("no_mask_embed")  # (16,) for mask_cfg ckpts, else None
    if no_mask_embed is not None:
        no_mask_embed = no_mask_embed.to(device)
        log.info(
            "Loaded learned null mask embedding no_mask_embed %s",
            tuple(no_mask_embed.shape),
        )

    text_projector = None
    proj_sd = {
        k[len("text_projector.") :]: v
        for k, v in sd.items()
        if k.startswith("text_projector.")
    }
    if proj_sd:
        text_projector = TextSequenceProjector(
            encoder_dims=ENCODER_HIDDEN_DIMS, out_dim=2560
        )
        text_projector.load_state_dict(proj_sd, strict=True)
        text_projector = text_projector.to(device).eval()
        for p in text_projector.parameters():
            p.requires_grad_(False)
        log.info("Loaded text_projector (%d tensors)", len(proj_sd))

    return unet, scale_factor, no_mask_embed, text_projector


class Report2CTWanLatentSampler(Report2CTSampler):
    """RFlow-denoise a Wan2.1 latent per case and save it as ``.npy`` (no decode).

    Reuses ``Report2CTSampler._case_to_context`` (findings+impression → 2560-d context
    ``(1, 2, 2560)`` via ``Report2CTTextEncoder``) and ``_make_spacing_tensor``; overrides model
    load (16-ch UNet, no MAISI VAE), the denoise latent shape, and the output tail (save latent
    instead of decode-to-HU + .mha). Decode runs in the separate ``wan`` env.
    """

    cross_attention_dim = 2560
    # UNet input channels: 16 (plain image latent); the mask subclass sets 32 (image + mask concat).
    unet_in_channels: int = 16
    # Per-case Wan mask latent (1, 16, 64, 64, 64), set by generate() via _case_mask_latent; None ⇒
    # plain image latent at the UNet input (byte-identical to the non-mask path).
    _mask_latent: torch.Tensor | None = None
    # Learned null mask embedding (16,) loaded from the ckpt (report2ct_wan_mask_v2 dual CFG); None
    # for plain / non-dual-CFG checkpoints. Set by _init from _load_wan_checkpoint.
    _null_mask: torch.Tensor | None = None

    def _init(self, device: torch.device) -> None:
        """Load the UNet + report2ct text encoder. No MAISI VAE (decode is a separate env)."""
        if self._unet is not None:
            return
        self._device = device
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        log.info("Loading Wan-latent UNet from %s …", self.ckpt_path)
        self._unet, self._scale_factor, self._null_mask, _ = _load_wan_checkpoint(
            self.ckpt_path, device, in_channels=self.unet_in_channels
        )
        log.info("Loading Report2CT text encoders …")
        self._text_encoder = Report2CTTextEncoder(device=device)

    def _case_mask_latent(self, case: EvalCase) -> torch.Tensor | None:
        """Per-case Wan mask latent for the UNet concat; ``None`` for the plain (non-mask) sampler."""
        return None

    @torch.no_grad()
    def _denoise(
        self, context: torch.Tensor, spacing_tensor: torch.Tensor
    ) -> torch.Tensor:
        """RFlow denoise → clean Wan latent ``(1, 16, 64, 64, 64)``.

        Mirrors ``Report2CTSampler._denoise`` (report2ct.py) but on the Wan latent shape; keeps
        the final ``t→0`` step (append 0 to next_timesteps — baseline-clone #2). Divides the
        clean latent by ``scale_factor`` here so the saved ``.npy`` is back in the stored
        (Wan-normalized) latent space that ``decode_wan_latents.py`` de-normalizes.

        Args:
            context: report2ct findings+impression condition ``(1, 2, 2560)``.
            spacing_tensor: ``(1, 3)`` spacing × 100.

        Returns:
            Clean latent ``(1, 16, 64, 64, 64)`` (÷ scale_factor), float32.
        """
        scheduler = RFlowScheduler(
            num_train_timesteps=1000,
            scale=1.4,
            use_timestep_transform=True,
            use_discrete_timesteps=False,
            sample_method="uniform",
        )
        scheduler.set_timesteps(
            self.n_steps,
            device=self._device,
            input_img_size=torch.tensor(float(WAN_LATENT_VOL)),
        )

        z = torch.randn(1, *WAN_LATENT_SHAPE, device=self._device, dtype=torch.float32)
        class_labels = torch.tensor([self.modality_class_label], device=self._device)

        all_t = scheduler.timesteps
        all_next_t = torch.cat(
            (all_t[1:], torch.tensor([0], dtype=all_t.dtype, device=all_t.device))
        )

        for t, next_t in zip(all_t, all_next_t):
            t_scalar = float(t.item())
            next_t_scalar = float(next_t.item())
            t_tensor = torch.tensor(
                [t_scalar], device=self._device, dtype=torch.float32
            )
            pred = self._predict(z, t_tensor, context, class_labels, spacing_tensor)
            z = scheduler.step(pred, t_scalar, z, next_t_scalar)[0]

        return (z / self._scale_factor).to(torch.float32)

    def _predict(
        self,
        z: torch.Tensor,
        t_tensor: torch.Tensor,
        context: torch.Tensor,
        class_labels: torch.Tensor,
        spacing_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """One RFlow-step model prediction with single-condition (text) CFG.

        The (fixed) Wan mask latent — if any — is concatenated to the UNet input each step; it is
        never noised and is present on BOTH the cond and uncond branches (text-only CFG). The
        report2ct_wan_mask_v2 subclass overrides this with a 3-pass dual (text + mask) CFG.

        Args:
            z: current clean-latent estimate ``(1, 16, 64, 64, 64)``.
            t_tensor: timestep ``(1,)``.
            context: report2ct condition ``(1, 2, 2560)``.
            class_labels: modality class ``(1,)``.
            spacing_tensor: spacing ``(1, 3)`` × 100.

        Returns:
            Model prediction ``(1, 16, 64, 64, 64)`` for ``scheduler.step``.
        """
        # None ⇒ plain 16-ch image latent; else concat the mask → 32-ch UNet input (z stays 16-ch).
        x_single = (
            z
            if self._mask_latent is None
            else torch.cat([z, self._mask_latent], dim=1)  # (1, 32, 64, 64, 64)
        )
        if self.cfg_scale != 1.0:
            uncond = torch.zeros_like(context)
            pred_b = self._unet(
                x=torch.cat([x_single, x_single], dim=0),
                timesteps=t_tensor.repeat(2),
                context=torch.cat([uncond, context], dim=0),
                class_labels=class_labels.repeat(2),
                spacing_tensor=spacing_tensor.repeat(2, 1),
            )
            pred_uncond, pred_cond = pred_b.chunk(2, dim=0)
            return pred_uncond + self.cfg_scale * (pred_cond - pred_uncond)
        return self._unet(
            x=x_single,
            timesteps=t_tensor,
            context=context,
            class_labels=class_labels,
            spacing_tensor=spacing_tensor,
        )

    @torch.no_grad()
    def generate(
        self, cases: list[EvalCase], out_dir: Path, device: torch.device
    ) -> list[Path]:
        """Generate one ``.npy`` Wan latent per case (resumable; existing files skipped).

        One file per CASE, named ``case.out_stem`` — the scan_id in a plain run, the manifest
        ``sample_id`` in a mask-intervention run (where the same scan is generated several times
        under different conditioning masks).

        The saved latent is in Wan VAE's native ``(16, 64, 64, 64)`` = ``(C, T_lat, H, W)``
        layout (permuted from the UNet's ``(C, H, W, D)``) so ``decode_wan_latents.py`` can call
        ``WanVAE.decode`` directly.
        """
        self._init(device)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        for case in tqdm(cases, desc="Wan latent gen"):
            out_path = out_dir / f"{case.out_stem}.npy"
            if out_path.exists():
                written.append(out_path)
                continue

            context = self._case_to_context(case)  # (1, 2, 2560)
            spacing_tensor = self._make_spacing_tensor(self.spacing_mm)  # (1, 3)
            self._mask_latent = self._case_mask_latent(case)  # (1,16,64,64,64) or None
            if case.seed is not None:  # manifest run: pin the noise (see _noise_seed)
                torch.manual_seed(_noise_seed(case))
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z = self._denoise(context, spacing_tensor)  # (1, 16, 64, 64, 64)
            # UNet (C,H,W,D)=(16,64,64,64) -> Wan (C,T_lat,H,W)=(16,64,64,64) for decode
            z_wan = z.squeeze(0).permute(0, 3, 1, 2).contiguous()  # (16, 64, 64, 64)
            np.save(out_path, z_wan.cpu().float().numpy())
            written.append(out_path)
            log.info("Saved latent %s  shape=%s", out_path.name, tuple(z_wan.shape))

        return written


class Report2CTWanMaskLatentSampler(Report2CTWanLatentSampler):
    """report2ct_wan_mask: RFlow-denoise a Wan latent while concatenating a precomputed Wan mask
    latent at the UNet input each step (report2ct_wan + Wan-latent mask conditioning).

    Identical to :class:`Report2CTWanLatentSampler` except the UNet is 32-channel (16 image + 16
    mask) and each case loads its ``<scan_id>_mask_emb.nii.gz`` Wan mask latent (the frozen-Wan
    encode of the painted 4-organ GT mask, produced by ``scripts/precompute_wan_mask_latents.py``).
    The mask is loaded with the SAME MONAI transforms as training so the ``(16,64,64,64)=(C,H,W,D)``
    axis order matches the image latent exactly; it is concatenated RAW (no scale_factor), always
    present (never CFG-dropped — CFG is text-only), on both the cond and uncond branches.

    Args:
        ckpt_path: Lightning .ckpt from ``experiment=report2ct_wan_mask`` (32-ch UNet).
        mask_dir: dir of ``<vol_id>_mask_emb.nii.gz`` Wan mask latents (precompute --out-dir).
    """

    unet_in_channels = 32

    def __init__(self, ckpt_path: str, mask_dir: str, **kwargs) -> None:
        super().__init__(ckpt_path=ckpt_path, **kwargs)
        self.mask_dir = Path(mask_dir)
        # Same loader as training (report2ct_wan_mask_datamodule): HWDC (64,64,64,16) → (16,64,64,64).
        self._mask_loader = Compose(
            [LoadImaged(keys="mask_latent"), EnsureChannelFirstd(keys="mask_latent")]
        )

    def _mask_source_id(self, case: EvalCase) -> str | None:
        """Which scan's mask conditions this case: the target's own, or a manifest-designated one.

        A plain run (``sample_id is None``) always uses the case's own scan. A manifest run uses
        ``cond_mask_source_id`` verbatim — the target's own id for ``condition=gt``, a donor for
        the swaps, and ``None`` for ``condition=null``, which has no designated mask at all.
        """
        if case.sample_id is None:
            return case.scan_id
        return case.cond_mask_source_id

    def _case_mask_latent(self, case: EvalCase) -> torch.Tensor | None:
        """Load ``<mask_dir>/<source_id>_mask_emb.nii.gz`` → ``(1, 16, 64, 64, 64)`` (C,H,W,D) on device.

        ``None`` only for a manifest ``condition=null`` row, which the caller renders with the
        checkpoint's learned ``no_mask_embed``; a model without one cannot express "no mask".
        """
        source_id = self._mask_source_id(case)
        if source_id is None:
            if self._null_mask is None:
                raise RuntimeError(
                    f"{case.sample_id}: condition=null needs the learned no_mask_embed, which "
                    f"{self.ckpt_path} does not have. Only report2ct_wan_mask_v2 (mask_cfg=true) "
                    "supports it; an empty mask here would be an input the model never saw."
                )
            return None

        path = self.mask_dir / f"{source_id}_mask_emb.nii.gz"
        if not path.is_file():
            raise FileNotFoundError(
                f"Wan mask latent missing for {source_id}: {path}. Run "
                f"scripts/precompute_wan_mask_latents.py (wan env) first — see docs/wan_latent_runbook.md."
            )
        out = self._mask_loader({"mask_latent": str(path)})
        m = out["mask_latent"]  # (16, 64, 64, 64) = (C, H, W, D)
        return (
            m.as_tensor().unsqueeze(0).to(self._device, torch.float32)
        )  # (1,16,64,64,64)


class Report2CTWanMaskV2LatentSampler(Report2CTWanMaskLatentSampler):
    """report2ct_wan_mask_v2: text-inner / mask-outer dual-condition CFG (InstructPix2Pix Eq. 3).

    Same 32-ch UNet + per-case Wan mask latent as :class:`Report2CTWanMaskLatentSampler`, but the
    single-scale text CFG ``_predict`` is replaced by a 3-pass dual CFG:

        pred = e(∅m,∅t) + s_t·[e(∅m,t) − e(∅m,∅t)] + s_m·[e(m,t) − e(∅m,t)]

    where ∅m is the LEARNED null mask embedding (``no_mask_embed``, loaded from the ckpt by
    ``_load_wan_checkpoint``) and ∅t is the zero text context. text is the inner condition (measured
    with the mask absent); the mask is the outer condition (measured on top of text), so **s_m is a
    clean mask-effect dial**: s_m=0 ignores the mask (pure text→CT), s_m=1 uses it naturally, s_m>1
    amplifies organ geometry. Three forward passes per step: (∅m,∅t), (∅m,t), (m,t).

    Args:
        ckpt_path: Lightning .ckpt from ``experiment=report2ct_wan_mask_v2`` (32-ch UNet + no_mask_embed).
        mask_dir: dir of ``<vol_id>_mask_emb.nii.gz`` Wan mask latents (precompute --out-dir).
        cfg_scale_text: s_t, text guidance scale (report2ct eval uses ~5).
        cfg_scale_mask: s_m, mask guidance scale (0 → mask off, 1 → natural, >1 → amplified).
    """

    def __init__(
        self,
        ckpt_path: str,
        mask_dir: str,
        cfg_scale_text: float,
        cfg_scale_mask: float,
        **kwargs,
    ) -> None:
        # Pass cfg_scale=cfg_scale_text to satisfy the base (its 2-pass path is overridden here).
        super().__init__(
            ckpt_path=ckpt_path, mask_dir=mask_dir, cfg_scale=cfg_scale_text, **kwargs
        )
        self.cfg_scale_text = cfg_scale_text
        self.cfg_scale_mask = cfg_scale_mask

    def _predict(
        self,
        z: torch.Tensor,
        t_tensor: torch.Tensor,
        context: torch.Tensor,
        class_labels: torch.Tensor,
        spacing_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """3-pass text-inner / mask-outer dual CFG prediction (see class docstring)."""
        if self._null_mask is None:
            raise RuntimeError(
                "report2ct_wan_mask_v2 checkpoint has no `no_mask_embed` — was it trained with "
                "mask_cfg=true? Dual-condition CFG needs the learned null mask."
            )
        null_m = self._null_mask.to(z.dtype).view(1, -1, 1, 1, 1)  # (1, 16, 1, 1, 1)
        null_mask = null_m.expand(z.shape[0], -1, *z.shape[2:])  # (1, 16, 64, 64, 64)
        # A manifest `condition=null` row designates NO mask, so the "with mask" branch IS the
        # learned null: e(m,t) == e(∅m,t), the s_m term is exactly 0, and the row reduces to
        # text-only CFG — which is what "generate without a mask" means for this model.
        mask = null_mask if self._mask_latent is None else self._mask_latent
        null_x = torch.cat([z, null_mask], dim=1)  # (1, 32, 64, 64, 64) ∅m
        real_x = torch.cat([z, mask], dim=1)  # (1, 32, 64, 64, 64) m
        zeros_ctx = torch.zeros_like(context)  # ∅t
        x_b = torch.cat([null_x, null_x, real_x], dim=0)  # (3, 32, 64, 64, 64)
        ctx_b = torch.cat([zeros_ctx, context, context], dim=0)  # (∅t, t, t)
        pred_b = self._unet(
            x=x_b,
            timesteps=t_tensor.repeat(3),
            context=ctx_b,
            class_labels=class_labels.repeat(3),
            spacing_tensor=spacing_tensor.repeat(3, 1),
        )
        p_uu, p_ut, p_mt = pred_b.chunk(3, dim=0)  # e(∅m,∅t), e(∅m,t), e(m,t)
        return (
            p_uu
            + self.cfg_scale_text * (p_ut - p_uu)
            + self.cfg_scale_mask * (p_mt - p_ut)
        )


class Report2CTWanSeqLatentSampler(Report2CTWanLatentSampler):
    """report2ct_wan_seq (arms B/C, docs/vlm3d_research_roadmap.md Phase 1e): per-encoder token
    SEQUENCES instead of the pooled 2-token context, with the padding-aware UNet + optional
    pooled-text AdaLN path from Unit 3a/3b.

    Must reproduce ``Report2CTSeqDataModule``'s (training-side) padding *exactly* — same lengths,
    same encoder order, same mask convention — or the model sees a context distribution shifted
    from what it trained on. Both sides now share ``_pad_or_truncate`` /
    ``SEQ_LEN_FINDINGS`` / ``SEQ_LEN_IMPRESSION`` / ``ENCODER_HIDDEN_DIMS``
    (``src/data/report2ct_wan_seq_datamodule.py``) rather than each re-deriving them, so this
    cannot drift from training by editing only one side.

    Args:
        text_pooled_adaln: whether the checkpoint was trained with the pooled findings/impression
            -> ``emb`` path active (arm C). ``False`` (default, arm B) leaves that path idle —
            the UNet is always ``DiffusionModelUNetMaisiTextPooled`` either way (its pooled path
            is a zero-init no-op when unused, verified in Unit 3a), only whether this sampler
            *feeds* it differs.
    """

    def __init__(self, *args, text_pooled_adaln: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.text_pooled_adaln = text_pooled_adaln
        self._text_projector: TextSequenceProjector | None = None
        # Per-case state set by _case_to_context, read by _predict — same pattern as the mask
        # subclasses' _mask_latent (an instance-attribute side channel is required here because
        # the base generate()/generate_wan_latents.py call chain threads only `context` through
        # _denoise -> _predict, and overriding _case_to_context's return type would break that
        # contract for every other subclass that inherits it unchanged).
        self._context_mask: torch.Tensor | None = None
        self._text_pooled_f: torch.Tensor | None = None
        self._text_pooled_i: torch.Tensor | None = None

    def _init(self, device: torch.device) -> None:
        """Load the TextPooled UNet + text_projector. No MAISI VAE (decode is a separate env)."""
        if self._unet is not None:
            return
        self._device = device
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        log.info("Loading Wan-latent TextPooled UNet from %s …", self.ckpt_path)
        self._unet, self._scale_factor, self._null_mask, self._text_projector = (
            _load_wan_checkpoint(
                self.ckpt_path,
                device,
                in_channels=self.unet_in_channels,
                text_pooled=True,
            )
        )
        if self._text_projector is None:
            raise RuntimeError(
                f"{self.ckpt_path} has no text_projector.* weights — was it trained with "
                "experiment=report2ct_wan_seq (or report2ct_wan_seq_adaln)? "
                "Report2CTWanSeqLatentSampler needs a checkpoint trained with text_projector set."
            )
        log.info("Loading Report2CT text encoders …")
        self._text_encoder = Report2CTTextEncoder(device=device)

    def _case_to_context(self, case: EvalCase) -> torch.Tensor:
        """Encode findings+impression to token sequences, pad+mask+project, cache mask/pooled.

        Mirrors ``Report2CTModule._prepare_seq_context`` + ``_shared_forward``'s pooled-text
        read, but for one live-encoded case instead of a batch loaded from precomputed sidecars.

        Returns:
            Cross-attention context, ``(1, n_encoders*(SEQ_LEN_FINDINGS+SEQ_LEN_IMPRESSION),
            2560)`` — same return type as the base class, so ``generate()`` needs no override.
            The mask (and, if ``text_pooled_adaln``, the pooled vectors) are stashed on
            ``self`` for ``_predict`` to read.
        """
        f_seqs, i_seqs = self._text_encoder.encode_tokens_pair(
            case.findings, case.impression
        )  # each: list of (n_tokens_k, encoder_dims[k]) CPU float32
        f_pad, f_mask, i_pad, i_mask = [], [], [], []
        for f_seq, i_seq in zip(f_seqs, i_seqs):
            fs, fm = _pad_or_truncate(f_seq, SEQ_LEN_FINDINGS)
            is_, im = _pad_or_truncate(i_seq, SEQ_LEN_IMPRESSION)
            f_pad.append(
                fs.unsqueeze(0).to(self._device)
            )  # (1, SEQ_LEN_FINDINGS, dim_k)
            f_mask.append(fm.unsqueeze(0).to(self._device))
            i_pad.append(is_.unsqueeze(0).to(self._device))
            i_mask.append(im.unsqueeze(0).to(self._device))

        with torch.no_grad():
            context, context_mask = self._text_projector(f_pad, f_mask, i_pad, i_mask)
        self._context_mask = context_mask  # (1, T)

        if self.text_pooled_adaln:
            # Same encoder_pair + mean-pool identity Phase 1a verified exactly
            # (encode_tokens' mean == encode()'s pooled output): reuse encode_pair directly
            # rather than re-deriving the pooled vector from the just-computed sequences.
            ctx_f, ctx_i = self._text_encoder.encode_pair(
                case.findings, case.impression
            )
            self._text_pooled_f = ctx_f.unsqueeze(0).to(self._device)  # (1, 2560)
            self._text_pooled_i = ctx_i.unsqueeze(0).to(self._device)

        return context

    def _predict(
        self,
        z: torch.Tensor,
        t_tensor: torch.Tensor,
        context: torch.Tensor,
        class_labels: torch.Tensor,
        spacing_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Text-only CFG, extended with the context mask and (if enabled) pooled-text AdaLN.

        The CFG-dropped branch mirrors ``Report2CTModule``'s training-time joint drop: the
        cross-attention context AND the pooled vectors are zeroed together, never independently
        — see ``src/models/report2ct_module.py::_shared_forward``.
        """
        x_single = (
            z if self._mask_latent is None else torch.cat([z, self._mask_latent], dim=1)
        )
        extra: dict[str, torch.Tensor] = {}
        if self.text_pooled_adaln:
            extra["text_pooled_f"] = self._text_pooled_f
            extra["text_pooled_i"] = self._text_pooled_i

        if self.cfg_scale != 1.0:
            uncond = torch.zeros_like(context)
            mask_b = self._context_mask.repeat(2, 1)
            extra_b = (
                {
                    k: torch.cat([torch.zeros_like(v), v], dim=0)
                    for k, v in extra.items()
                }
                if extra
                else {}
            )
            pred_b = self._unet(
                x=torch.cat([x_single, x_single], dim=0),
                timesteps=t_tensor.repeat(2),
                context=torch.cat([uncond, context], dim=0),
                class_labels=class_labels.repeat(2),
                spacing_tensor=spacing_tensor.repeat(2, 1),
                context_mask=mask_b,
                **extra_b,
            )
            pred_uncond, pred_cond = pred_b.chunk(2, dim=0)
            return pred_uncond + self.cfg_scale * (pred_cond - pred_uncond)
        return self._unet(
            x=x_single,
            timesteps=t_tensor,
            context=context,
            class_labels=class_labels,
            spacing_tensor=spacing_tensor,
            context_mask=self._context_mask,
            **extra,
        )


class Report2CTWanEvalSampler(AbstractSampler):
    """Pass-through sampler for ``run_eval.py``: expects Wan ``.mha`` predictions already decoded.

    The Wan latent → HU decode happens in the ``wan`` conda env
    (``scripts/decode_wan_latents.py``), which writes ``<pred_dir>/<scan_id>.mha``. This sampler
    just verifies they exist and returns them so ``CTGenEvaluator`` can score — it never loads a
    UNet or VAE. Run the generate+decode steps before scoring (see docs/wan_latent_runbook.md).

    Args:
        name: absorbed from the Hydra eval/model config; unused.
    """

    def __init__(self, name: str | None = None) -> None:
        self.name = name

    def generate(
        self, cases: list[EvalCase], out_dir: Path, device: torch.device
    ) -> list[Path]:
        """Return the pre-decoded ``.mha`` per case; error if any is missing.

        Looks for ``case.out_stem`` — scan_id in a plain run, the manifest sample_id in a
        mask-intervention run (which is what the decode step wrote, since the latent it decoded
        was itself named after the sample).
        """
        out_dir = Path(out_dir)
        written: list[Path] = []
        missing: list[str] = []
        for case in cases:
            p = out_dir / f"{case.out_stem}.mha"
            (written if p.exists() else missing).append(
                p if p.exists() else case.out_stem
            )
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} Wan predictions missing in {out_dir} (e.g. {missing[:3]}). "
                f"Run scripts/generate_wan_latents.py then scripts/decode_wan_latents.py "
                f"(wan env) first — see docs/wan_latent_runbook.md."
            )
        return written


__all__ = [
    "Report2CTWanLatentSampler",
    "Report2CTWanMaskLatentSampler",
    "Report2CTWanMaskV2LatentSampler",
    "Report2CTWanSeqLatentSampler",
    "Report2CTWanEvalSampler",
]
