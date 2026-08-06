"""Report2CT training LightningModule.

Re-implements the training-loop logic of upstream
`third_party/report2ct/src/maisi/scripts/diff_model_train_vlm3D_2560_multi_text.py`
in Lightning-native form. Each line is annotated with the upstream `file:line` source
so drift can be audited.

Architecture (UNet, RFlowScheduler) is **not** rewritten — instantiated directly via
Hydra `_target_:` from the corresponding MONAI classes (`DiffusionModelUNetMaisi`,
`RFlowScheduler`). Parity test `tests/test_report2ct_parity.py` enforces kwarg equality
against the upstream JSON config.

Source references (all paths relative to `third_party/report2ct/src/maisi/scripts/`):
  - diff_model_train_vlm3D_2560_multi_text.py
      :151-172   calculate_scale_factor (first-batch stddev)
      :186       optimizer = Adam(lr)
      :189-200   lr_scheduler = PolynomialLR(power=2.0)
      :262-273   images = data['image'] * scale_factor; spacing_tensor
      :275-286   context_f/context_i load, unsqueeze if dim==2, cat dim=1
      :294-297   CFG drop_prob=0.15, context = zeros_like(context)
      :301-308   noise sampling + RFlowScheduler.sample_timesteps + add_noise
      :311-331   UNet input dict (x, timesteps, spacing_tensor, context, class_labels)
      :333-346   prediction_type branching (EPSILON/SAMPLE/V_PREDICTION)
      :348       loss = MSELoss(model_output.float(), model_gt.float())
      :512-515   total_steps = (n_epochs * dataset_size) / batch_size
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from lightning.pytorch import LightningModule
from monai.networks.schedulers.ddpm import DDPMPredictionType
from torch import nn

from src.baselines.rflow import RFlowScheduler
from torch.nn import functional as F

log = logging.getLogger(__name__)

CFG_DROP_PROB_DEFAULT: float = 0.15  # upstream :295


class Report2CTModule(LightningModule):
    """Lightning training module for Report2CT-style text→CT latent diffusion.

    The UNet and noise scheduler are constructed externally (e.g., by Hydra
    `instantiate(...)`) and passed in. This module only owns the training loop.
    """

    def __init__(
        self,
        unet: nn.Module,
        noise_scheduler: Any,
        mask_conditioner: nn.Module | None = None,
        lr: float = 1e-4,
        cfg_drop_prob: float = CFG_DROP_PROB_DEFAULT,
        cfg_per_sample: bool = False,
        spacing_scale: float = 1.0,
        modality_class_label: int = 1,
        lr_scheduler_power: float = 2.0,
        warmup_steps: int = 0,
        mask_cfg: bool = False,
        cfg_uncond_prob: float = 0.05,
        mask_channels: int = 16,
        no_mask_init_std: float = 0.02,
        repa: nn.Module | None = None,
        repa_weight: float = 0.5,
        repa_tap: str = "middle_block",
        repa_stop_step: int | None = None,
        init_from_ckpt: str | None = None,
        text_projector: nn.Module | None = None,
        text_pooled_cond: bool = False,
    ) -> None:
        """Wire up the training loop around an externally-built UNet + scheduler.

        Args:
            unet: denoising UNet, e.g. ``DiffusionModelUNetMaisi`` (Hydra-instantiated).
            noise_scheduler: ``RFlowScheduler`` (or any DDPM-style scheduler exposing
                ``add_noise`` / ``sample_timesteps``).
            mask_conditioner: optional ``MaskConditioner`` whose embedding of a latent-grid organ
                mask is concatenated onto the noisy latent at the UNet input (LDM concat
                conditioning). ``None`` ⇒ identical to the text-only baseline; when set, the UNet's
                ``in_channels`` must be ``latent_channels + mask_conditioner.embed_dim``.
            lr: Adam learning rate (upstream :186).
            cfg_drop_prob: probability of zeroing the context for classifier-free
                guidance during training (upstream :295).
            cfg_per_sample: if True, apply the CFG context dropout INDEPENDENTLY per sample
                (text2ct convention, diff_model_train.py:335); if False, drop the whole batch's
                context together (report2ct upstream :296-297; faithful default).
            spacing_scale: extra multiplier on the spacing tensor; ``1.0`` keeps the
                DataModule's already-scaled values untouched.
            modality_class_label: fixed modality class id; ``1`` = CT (upstream :270).
            lr_scheduler_power: exponent of the ``PolynomialLR`` decay (upstream :189-200).
            warmup_steps: linear-warmup steps prepended before the ``PolynomialLR`` decay; ``0``
                (default) reproduces report2ct's plain PolynomialLR byte-for-byte, ``>0``
                (report2ct_wan) ramps the LR in over this many optimizer steps first.
            mask_cfg: enable dual-condition CFG (report2ct_wan_mask_v2) — IP2P 5/5/5 dropout of the
                text + Wan mask conditions, with a LEARNED null mask embedding for dropped masks.
                ``False`` (default) ⇒ byte-identical to every other experiment.
            cfg_uncond_prob: per-bucket base prob ``p`` for the IP2P 5/5/5 dropout (each condition
                dropped with prob ``2p``); only used when ``mask_cfg`` is True.
            mask_channels: Wan mask-latent channel count for the learned null embedding (16).
            no_mask_init_std: stddev of the small-random init for the learned null embedding.
            repa: optional ``RepaAligner`` (``src/models/components/repa.py``) that aligns a
                middle-stage UNet feature with a frozen CT foundation model's token grid
                (U-REPA). ``None`` ⇒ the loss path is bit-identical to the baseline. Requires
                the batch to carry ``spectre`` ``(B, T, 1080)``.
            repa_weight: outer coefficient λ in ``L = L_velocity + λ·L_align`` (U-REPA Eq.5;
                their optimum is 0.5). The hard/relational split lives inside the aligner.
            repa_tap: attribute path of the UNet block to tap, e.g. ``"middle_block"`` (U-REPA's
                choice for U-Nets) or ``"up_blocks.0"``. Only reached through a forward hook, so
                the UNet's own forward is untouched.
            repa_stop_step: drop the alignment term entirely after this many global steps
                (HASTE's one-shot termination — alignment helps early and can hurt late).
                ``None`` keeps it on for the whole run.
            init_from_ckpt: **finetune** from a previous run's weights — loads only ``unet.*`` and
                ``scale_factor`` from a Lightning ``.ckpt``. This is deliberately not Lightning's
                ``ckpt_path=``, which also restores the optimizer and LR schedule: resuming a run
                whose ``PolynomialLR`` has already decayed to ~0 would train at essentially no
                learning rate. Use this with a small explicit ``lr`` instead.
            text_projector: optional ``TextSequenceProjector``
                (``src/models/components/text_sequence_projector.py``). When set, the
                cross-attention context becomes the per-encoder **token sequences** projected to
                ``cross_attention_dim`` and concatenated — replacing the pooled 2-token context —
                and a key-padding mask is passed alongside it. Requires the batch to carry the
                ``context_{f,i}_{seq,mask}_<k>`` keys emitted by ``Report2CTSeqDataModule``.
                ``None`` ⇒ the pooled path, byte-identical to the baseline.
            text_pooled_cond: route the pooled findings/impression vectors into the UNet's ``emb``
                (the vector every ResBlock reads) in addition to whatever the cross-attention
                context is. Requires ``unet`` to be a ``DiffusionModelUNetMaisiTextPooled``.
                Independent of ``text_projector``, so the two knobs give the A/B/C arms of
                docs/vlm3d_research_roadmap.md Phase 1e: both off ⇒ A, projector only ⇒ B,
                both on ⇒ C.
        """
        super().__init__()
        # Hydra-instantiated objects are not serializable in hparams; ignore them.
        self.save_hyperparameters(
            ignore=[
                "unet",
                "noise_scheduler",
                "mask_conditioner",
                "repa",
                "text_projector",
            ]
        )
        self.unet = unet
        # Token-sequence text conditioning (opt-in). Registered as a real submodule so it lands in
        # state_dict / .to(device); its params must ALSO be appended in configure_optimizers —
        # that method builds its list from self.unet, so a submodule alone would silently never
        # train (same trap as mask_conditioner / repa below).
        self.text_projector = text_projector
        self.text_pooled_cond = text_pooled_cond
        # Optional LDM concat mask conditioner (nn.Module, registered so it lands in state_dict /
        # .to(device) / .train()/.eval()). None ⇒ behaviour byte-identical to the text-only baseline.
        self.mask_conditioner = mask_conditioner
        # RFlowScheduler inherits nn.Module but doesn't call super().__init__(),
        # so _parameters is missing and model.parameters() crashes.
        # Bypass nn.Module.__setattr__ to avoid submodule registration.
        object.__setattr__(self, "noise_scheduler", noise_scheduler)
        self.lr = lr
        self.cfg_drop_prob = cfg_drop_prob
        self.cfg_per_sample = cfg_per_sample
        self.spacing_scale = spacing_scale
        self.modality_class_label = modality_class_label
        self.lr_scheduler_power = lr_scheduler_power
        self.warmup_steps = warmup_steps
        self.mask_cfg = mask_cfg
        self.cfg_uncond_prob = cfg_uncond_prob
        # Dual-condition CFG (report2ct_wan_mask_v2): a LEARNED null mask embedding replaces the Wan
        # mask latent for CFG-dropped samples (SAM `no_mask_embed` pattern — a per-channel vector
        # broadcast over space). Small-random init keeps the null ~neutral in the per-channel-
        # normalized Wan latent space. Created ONLY when mask_cfg=True so every other experiment's
        # state_dict stays byte-identical.
        if mask_cfg:
            self.no_mask_embed = nn.Parameter(
                torch.randn(mask_channels) * no_mask_init_std
            )  # (16,)

        # REPA (opt-in). The aligner is a real submodule so it lands in state_dict / .to(device),
        # but its params must ALSO be appended in configure_optimizers — that method builds the
        # param list from self.unet only, so a submodule alone would silently never train.
        self.repa = repa
        self.repa_weight = repa_weight
        self.repa_tap = repa_tap
        self.repa_stop_step = repa_stop_step
        self._repa_tap_feat: torch.Tensor | None = None
        self._repa_handle: Any = None
        # Dedicated RNG for the aligner's token sampling so switching REPA on does not shift the
        # global RNG stream (keeps a repa_weight=0 run bit-identical to the baseline).
        self._repa_generator: torch.Generator | None = None

        self.register_buffer("scale_factor", torch.tensor(1.0))
        self._scale_factor_initialized: bool = False
        if init_from_ckpt is not None:
            self._load_weights_only(init_from_ckpt)

    def _load_weights_only(self, ckpt_path: str) -> None:
        """Load ``unet.*`` weights + ``scale_factor`` from a Lightning checkpoint, nothing else.

        The scale factor comes along because it is a data statistic, not a learned parameter:
        recomputing it from the first finetune batch would put the resumed UNet in a slightly
        different latent scale than it was trained in.
        """
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        unet_sd = {
            k[len("unet.") :]: v for k, v in state.items() if k.startswith("unet.")
        }
        missing, unexpected = self.unet.load_state_dict(unet_sd, strict=True)
        if "scale_factor" in state:
            self.scale_factor = state["scale_factor"].to(self.scale_factor.device)
            self._scale_factor_initialized = True
        log.info(
            "init_from_ckpt %s: %d unet tensors, scale_factor=%.4f (missing=%s unexpected=%s)",
            ckpt_path,
            len(unet_sd),
            float(self.scale_factor),
            missing,
            unexpected,
        )

    def _maybe_init_scale_factor(self, images: torch.Tensor) -> None:
        """Set ``self.scale_factor`` once, from the first batch's latent stddev.

        Args:
            images: first batch of MAISI latents, ``(B, 4, 120, 120, 64)``.

        No-op after the first call (DDP-synced via all-gather mean).
        """
        # upstream calculate_scale_factor :163-172
        if self._scale_factor_initialized:
            return
        sf = 1.0 / images.detach().float().std().clamp(min=1e-8)
        # Sync across DDP ranks: upstream uses dist.all_reduce(AVG); Lightning
        # all_gather returns (world_size,) so .mean() is the equivalent.
        if self.trainer is not None and getattr(self.trainer, "world_size", 1) > 1:
            sf = self.all_gather(sf).float().mean()
        self.scale_factor = sf.to(self.scale_factor.dtype).to(self.scale_factor.device)
        self._scale_factor_initialized = True

    def _prepare_context(
        self, context_f: torch.Tensor, context_i: torch.Tensor
    ) -> torch.Tensor:
        """Stack the findings + impression embeddings into one context tensor.

        Args:
            context_f: findings embedding, ``(B, D)`` or ``(B, 1, D)``.
            context_i: impression embedding, ``(B, D)`` or ``(B, 1, D)``.

        Returns:
            Concatenated context, ``(B, 2, D)`` (D = 2560 for Report2CT).
        """
        # upstream :280-286 — promote (B, D) → (B, 1, D), then cat along dim=1
        if context_f.dim() == 2:
            context_f = context_f.unsqueeze(1)
        if context_i.dim() == 2:
            context_i = context_i.unsqueeze(1)
        return torch.cat((context_f, context_i), dim=1)

    def _prepare_seq_context(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the per-encoder token sequences into one cross-attention context + mask.

        Args:
            batch: must carry ``context_f_seq_<k>`` / ``context_f_mask_<k>`` /
                ``context_i_seq_<k>`` / ``context_i_mask_<k>`` for each encoder ``k``, as emitted
                by ``Report2CTSeqDataModule``. Kept as separate keys because the encoders' hidden
                widths differ (1024/768/768) and cannot share one stacked tensor.

        Returns:
            Tuple ``(context, context_mask)``: ``(B, sum_of_lengths, cross_attention_dim)`` and
            ``(B, sum_of_lengths)`` bool (``True`` = real token).
        """
        n = len(self.text_projector.projections)
        return self.text_projector(
            [batch[f"context_f_seq_{k}"] for k in range(n)],
            [batch[f"context_f_mask_{k}"] for k in range(n)],
            [batch[f"context_i_seq_{k}"] for k in range(n)],
            [batch[f"context_i_mask_{k}"] for k in range(n)],
        )

    def _shared_forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run one denoising step and return the diffusion loss.

        Shared by ``training_step`` and ``validation_step``: scales the latents,
        assembles the conditioning context (with CFG dropout in training), samples
        noise + timesteps, runs the UNet, and compares against the prediction-type
        target (EPSILON / SAMPLE / V_PREDICTION).

        Args:
            batch: dict with ``image`` ``(B, 4, 120, 120, 64)``, ``spacing`` ``(B, 3)``,
                and either a pre-assembled ``context`` ``(B, num_organs, D)`` or the
                ``context_f`` / ``context_i`` pair ``(B, D)`` each.

        Returns:
            Scalar MSE loss between the UNet output and the target, both ``(B, 4, 120, 120, 64)``.
        """
        images = batch["image"]  # (B, C, H, W, D)
        self._maybe_init_scale_factor(images)
        images = images * self.scale_factor  # upstream :263

        # Conditioner-agnostic: a datamodule may emit a pre-assembled `context`
        # (e.g. fVLM (B, 4, 256)); otherwise fall back to Report2CT's findings +
        # impression pair → (B, 2, 2560). Everything downstream (CFG, UNet) is identical.
        context_mask: torch.Tensor | None = None
        if self.text_projector is not None:
            # Token sequences replace the pooled 2-token context; the mask travels with them.
            context, context_mask = self._prepare_seq_context(batch)
        elif "context" in batch:
            context = batch["context"]  # (B, num_organs, D) — e.g. fVLM (B, 4, 256)
        else:
            context = self._prepare_context(
                batch["context_f"], batch["context_i"]
            )  # upstream :286
        # Pooled findings/impression for the UNet's `emb` path (arm C). Read BEFORE the CFG block
        # so the same drop can be applied to them; `None` leaves the UNet's text_pooled path idle.
        text_pooled_f: torch.Tensor | None = None
        text_pooled_i: torch.Tensor | None = None
        if self.text_pooled_cond:
            text_pooled_f = batch["context_f"]
            text_pooled_i = batch["context_i"]
        # Classifier-free-guidance dropout (training only). Three mutually-exclusive regimes:
        #   (a) mask_cfg — dual-condition IP2P 5/5/5: ONE per-sample uniform draw drops TEXT and MASK
        #       independently (each dropped 2p=10% = single 5% + both 5%). The text drop zeros the
        #       context here; the mask drop is recorded in `mask_drop` and applied at the concat
        #       below by swapping in the learned null embedding. (report2ct_wan_mask_v2)
        #   (b) cfg_per_sample — text-only, independent per-sample drop (text2ct).
        #   (c) whole-batch text drop (report2ct upstream :296-297).
        mask_drop: torch.Tensor | None = None
        # Which samples had their TEXT condition dropped, recorded so the pooled `emb` path can be
        # dropped identically — a sample whose cross-attention context is zeroed but whose pooled
        # vector still carries the report would teach the model an inconsistent "unconditional".
        text_drop: torch.Tensor | None = None
        if self.training and self.mask_cfg:
            p = self.cfg_uncond_prob  # 0.05
            r = torch.rand(context.shape[0], device=context.device)
            drop_text = r < 2 * p  # [0, 2p): text-only-drop + both
            mask_drop = (r >= p) & (r < 3 * p)  # [p, 3p): both + mask-only-drop
            context = context.masked_fill(
                drop_text.view(-1, *([1] * (context.dim() - 1))), 0.0
            )
            text_drop = drop_text
        elif self.training and self.cfg_drop_prob > 0.0:  # upstream :296-297
            if (
                self.cfg_per_sample
            ):  # text2ct: independent per-sample drop (diff_model_train.py:335)
                drop = (
                    torch.rand(context.shape[0], device=context.device)
                    < self.cfg_drop_prob
                )
                context = context.masked_fill(
                    drop.view(-1, *([1] * (context.dim() - 1))), 0.0
                )  # zero the (T, D) of dropped samples only
                text_drop = drop
            elif torch.rand(()) < self.cfg_drop_prob:  # report2ct: whole-batch drop
                context = torch.zeros_like(context)
                text_drop = torch.ones(
                    context.shape[0], dtype=torch.bool, device=context.device
                )

        # Mirror the text drop onto the pooled `emb` inputs. The UNet's `text_pooled_proj` is a
        # Linear, so a zeroed input maps to its (learned) bias — a consistent null-text embedding,
        # the same pattern as the learned null mask used by mask_cfg.
        if text_drop is not None and text_pooled_f is not None:
            zero_at = text_drop.view(-1, *([1] * (text_pooled_f.dim() - 1)))
            text_pooled_f = text_pooled_f.masked_fill(zero_at, 0.0)
            text_pooled_i = text_pooled_i.masked_fill(zero_at, 0.0)

        spacing_tensor = batch["spacing"].to(
            images.device
        )  # already × 1e2 in DataModule (upstream :88)
        if self.spacing_scale != 1.0:
            spacing_tensor = spacing_tensor * self.spacing_scale

        # upstream :270 — modality fixed to CT (class 1)
        modality_tensor = torch.full(
            (images.shape[0],),
            self.modality_class_label,
            dtype=torch.long,
            device=images.device,
        )

        noise = torch.randn_like(images)  # upstream :301
        if isinstance(self.noise_scheduler, RFlowScheduler):  # upstream :303-306
            timesteps = self.noise_scheduler.sample_timesteps(images)
        else:
            num_train_timesteps = int(
                getattr(self.noise_scheduler, "num_train_timesteps", 1000)
            )
            timesteps = torch.randint(
                0, num_train_timesteps, (images.shape[0],), device=images.device
            ).long()

        noisy_latent = self.noise_scheduler.add_noise(  # upstream :308
            original_samples=images, noise=noise, timesteps=timesteps
        )

        # Concat mask conditioning at the UNet INPUT only. `noise`, `add_noise`, and `model_gt`
        # above all stay on the image latent, so the mask is never noised nor predicted — the UNet
        # grows its in_channels but still outputs the image-latent channels. Two mutually-exclusive
        # sources; None of either ⇒ x is the plain noisy latent (text-only baseline):
        #   (a) mask_conditioner  — TRAINED top-K soft organ embedding (report2ct_text2ct_mask).
        #   (b) batch["mask_latent"] — PRECOMPUTED frozen-Wan encode of the painted 4-organ mask,
        #       already in the same normalized latent space as the image latent, concatenated RAW
        #       (no scale_factor — it is conditioning, not the diffused variable; report2ct_wan_mask).
        if self.mask_conditioner is not None:
            # Top-K soft organ mask -> embedding (LDM concat). `.to(dtype)` keeps the cat AMP-safe:
            # the embedding output is fp32 while noisy_latent may be bf16 under autocast.
            e_mask = self.mask_conditioner(
                batch["mask_classes"], batch["mask_fracs"]
            ).to(noisy_latent.dtype)  # (B, embed_dim, H, W, D)
            if self.training:  # monitor whether the mask is actually engaging
                self._log_mask_stats(e_mask, noisy_latent)
            unet_x = torch.cat(
                [noisy_latent, e_mask], dim=1
            )  # (B, C+embed_dim, H, W, D)
        elif "mask_latent" in batch:
            mask_latent = batch["mask_latent"].to(
                noisy_latent.dtype
            )  # (B, 16, H, W, D) — raw, NOT × scale_factor
            if self.mask_cfg and self.training and mask_drop is not None:
                # Swap the LEARNED null mask (SAM no_mask_embed) into CFG-dropped samples. torch.where
                # + .view keep the graph intact, so no_mask_embed receives gradient from every dropped
                # position (no .detach / .data / in-place — those would sever the null's learning).
                null = self.no_mask_embed.to(noisy_latent.dtype).view(
                    1, -1, 1, 1, 1
                )  # (1, 16, 1, 1, 1) broadcast over batch + space
                mask_latent = torch.where(
                    mask_drop.view(-1, 1, 1, 1, 1), null, mask_latent
                )  # (B, 16, H, W, D)
            unet_x = torch.cat([noisy_latent, mask_latent], dim=1)  # (B, 32, H, W, D)
        else:
            unet_x = noisy_latent  # (B, C, H, W, D)

        unet_inputs = {  # upstream :311-316 (+ class_labels at :325-330)
            "x": unet_x,
            "timesteps": timesteps,
            "spacing_tensor": spacing_tensor,
            "context": context,
            "class_labels": modality_tensor,
        }
        # Added only when in use: the stock DiffusionModelUNetMaisi accepts neither kwarg, so a
        # baseline experiment must keep calling it with exactly the upstream signature.
        if context_mask is not None:
            unet_inputs["context_mask"] = context_mask
        if text_pooled_f is not None:
            unet_inputs["text_pooled_f"] = text_pooled_f
            unet_inputs["text_pooled_i"] = text_pooled_i
        model_output = self.unet(**unet_inputs)  # upstream :331

        # REPA aux term. Stashed rather than returned so `_shared_forward` keeps returning a bare
        # scalar — tests/test_report2ct_parity.py and the two wan_mask hook tests assert
        # `loss.ndim == 0` on it, and `training_step` is where the two terms are combined.
        self._last_repa = self._repa_loss(batch)

        # upstream :333-346 — prediction-type branching.
        # Bundled RFlowScheduler has no prediction_type attr; its step() at
        # rectified_flow.py:155 uses velocity (v_pred = model_output), so target
        # = images − noise == V_PREDICTION.
        prediction_type = getattr(self.noise_scheduler, "prediction_type", None)
        if prediction_type is None and isinstance(self.noise_scheduler, RFlowScheduler):
            prediction_type = DDPMPredictionType.V_PREDICTION
        elif prediction_type is None:
            prediction_type = DDPMPredictionType.EPSILON
        if prediction_type == DDPMPredictionType.EPSILON:
            model_gt = noise
        elif prediction_type == DDPMPredictionType.SAMPLE:
            model_gt = images
        elif prediction_type == DDPMPredictionType.V_PREDICTION:
            model_gt = images - noise
        else:
            raise ValueError(f"Unknown prediction_type {prediction_type}")

        return F.mse_loss(
            model_output.float(), model_gt.float()
        )  # upstream :348 (MSELoss, switched from L1 at :517)

    def _log_mask_stats(self, e_mask: torch.Tensor, noisy_latent: torch.Tensor) -> None:
        """Log mask-conditioning diagnostics so we can see the mask actually engaging.

        Args:
            e_mask: mask embedding fed to the concat, ``(B, embed_dim, H, W, D)``.
            noisy_latent: the 4-channel noisy latent it is concatenated with, same spatial dims.

        Logs per training step: ``mask/gamma_absmean`` + per-channel ``mask/gamma_c{i}`` (learned
        LayerScale — grows off 0 as the mask is used), ``mask/e_std`` (effective mask signal
        magnitude), and ``mask/e_over_z`` (mask std relative to the latent it augments; ~O(1) means
        comparable, ~0 means the mask is being ignored).
        """
        g = self.mask_conditioner.gamma.detach()
        e_std = e_mask.detach().float().std()
        z_std = noisy_latent.detach().float().std().clamp(min=1e-6)
        common = {"on_step": True, "on_epoch": False, "sync_dist": True}
        self.log("mask/gamma_absmean", g.abs().mean(), **common)
        self.log("mask/e_std", e_std, **common)
        self.log("mask/e_over_z", e_std / z_std, **common)
        for i in range(g.numel()):
            self.log(f"mask/gamma_c{i}", g[i], **common)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Compute the diffusion loss on a training batch and log ``train/loss``.

        Args:
            batch: see :meth:`_shared_forward`.
            batch_idx: index of the batch within the epoch (unused; Lightning API).

        Returns:
            Scalar training loss.
        """
        loss_diff = self._shared_forward(batch)
        repa_loss, repa_logs = getattr(self, "_last_repa", (None, {}))
        loss = loss_diff
        if repa_loss is not None:
            # U-REPA Eq.5: L = L_velocity + λ·(hard·L_cos + w·L_rel); the bracket is the aligner's
            # output, λ is repa_weight. `train/loss` stays the TOTAL so src/callbacks/loss_curve.py
            # keeps plotting the same curve it always has.
            loss = loss_diff + self.repa_weight * repa_loss
            self.log(
                "train/loss_diff",
                loss_diff,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/loss_repa",
                repa_loss,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            for name, value in repa_logs.items():
                self.log(
                    f"train/{name}", value, on_step=True, on_epoch=True, sync_dist=True
                )
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        """Log the global L2 grad-norm (pre-clip) so clipping can be judged empirically.

        Pure diagnostics — no gradient modification. Runs after backward, before the optimizer
        step; grads are the real values under bf16 (no unscaling needed). Lets us see whether the
        UNet's grad-norm ever spikes (→ set ``gradient_clip_val``) or stays bounded (→ none needed).

        Args:
            optimizer: the optimizer about to step (unused; Lightning API).
        """
        grads = [p.grad.detach() for p in self.parameters() if p.grad is not None]
        if grads:
            total = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
            self.log(
                "train/grad_norm",
                total,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True,
            )

    def setup(self, stage: str) -> None:
        """Select a Blackwell-safe SDPA backend before training starts.

        Args:
            stage: Lightning stage string (``"fit"`` / ``"validate"`` / ...); unused.
        """
        # PyTorch 2.7 FA3 backend crashes on Blackwell (sm_120) during DDP backward.
        # Force memory-efficient SDPA instead; forward/backward both stable on sm_120.
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        self._register_repa_hook()

    # --- REPA plumbing --------------------------------------------------------
    def _register_repa_hook(self) -> None:
        """Attach a forward hook on the tapped UNet block (idempotent; no-op without REPA).

        A hook is used rather than a forward-returning-features variant so the UNet stays exactly
        the module the checkpoint expects. ⚠ The captured tensor is deliberately **not** detached:
        detaching it would leave the alignment loss training only the projector while the UNet
        learns nothing — a failure with no symptom other than a flat FID.
        """
        if self.repa is None or self._repa_handle is not None:
            return
        module: nn.Module = self.unet
        for part in self.repa_tap.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)

        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            # Down/up blocks may return (hidden, residuals); the hidden state is first.
            self._repa_tap_feat = output[0] if isinstance(output, tuple) else output

        self._repa_handle = module.register_forward_hook(hook)

    def _repa_active(self) -> bool:
        """Whether the alignment term contributes this step (HASTE one-shot termination)."""
        if self.repa is None or self.repa_weight == 0.0:
            return False
        if self.repa_stop_step is None:
            return True
        return int(self.global_step) < self.repa_stop_step

    def _repa_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor | None, dict]:
        """Consume the hooked feature and score it against the batch's teacher tokens.

        Must be called right after the UNet forward. Always clears `_repa_tap_feat` so the
        autograd graph is not held across steps (a leak that shows up as steadily growing memory).
        """
        feat, self._repa_tap_feat = self._repa_tap_feat, None
        # Validation shares this forward path; scoring there would mix REPA into val/loss and
        # change what the checkpoint callback sees. The clear above still runs unconditionally.
        if not self.training or not self._repa_active() or feat is None:
            return None, {}
        if "spectre" not in batch:
            raise KeyError(
                "REPA is enabled but the batch has no 'spectre' key — use "
                "configs/data/report2ct_wan_repa.yaml (Report2CTWanRepaDataModule)."
            )
        if self._repa_generator is None:
            # Fixed seed, deliberately independent of rank/step: the point is only that this
            # stream is SEPARATE from the global one. Sharing the token subset across DDP ranks
            # is harmless (each rank sees different volumes anyway), and a rank-dependent seed
            # would need a trainer, which unit tests do not have.
            self._repa_generator = torch.Generator(device=feat.device).manual_seed(1234)
        return self.repa(
            feat.float(),
            batch["spectre"].float(),
            teacher_global=batch.get("spectre_global"),
            generator=self._repa_generator,
        )

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Compute the diffusion loss on a validation batch and log ``val/loss``.

        Args:
            batch: see :meth:`_shared_forward`.
            batch_idx: index of the batch within the epoch (unused; Lightning API).

        Returns:
            Scalar validation loss.
        """
        loss = self._shared_forward(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Build the Adam optimizer + per-step ``PolynomialLR`` schedule.

        Returns:
            Lightning optimizer-config dict with ``optimizer`` and a step-interval
            ``lr_scheduler`` entry.
        """
        # Include the mask conditioner's params (Embedding + gamma) when present — otherwise Adam
        # never updates them and the mask silently stays off (gamma pinned at its 0 init).
        params = list(self.unet.parameters())

        # mask condition parameter를 optim에 추가함
        if self.mask_conditioner is not None:
            params += list(self.mask_conditioner.parameters())
        # Learned null mask embedding (dual-condition CFG, report2ct_wan_mask_v2) — without this Adam
        # never updates it and the null stays frozen at its random init.
        if getattr(self, "mask_cfg", False):
            params.append(self.no_mask_embed)
        # REPA projector — a submodule alone is not enough: this param list is built from
        # self.unet, so without this the projector never receives an update and the alignment
        # loss plateaus at its random-init value with no other symptom.
        if self.repa is not None:
            params += list(self.repa.parameters())
        # Text sequence projector — same trap: without this its per-encoder Linears never update,
        # so the cross-attention context stays a random projection of the token sequences.
        if self.text_projector is not None:
            params += list(self.text_projector.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)  # upstream :186
        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        # PolynomialLR (power 2) decays LR→0 over training (upstream :189-200). When warmup_steps>0
        # (report2ct_wan), prepend a linear warmup so the raised LR ramps in rather than hitting
        # step 0 cold; the decay then runs over the remaining steps. warmup_steps=0 (report2ct
        # baseline) keeps the plain PolynomialLR byte-for-byte.
        if self.warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-2, total_iters=self.warmup_steps
            )
            decay = torch.optim.lr_scheduler.PolynomialLR(
                optimizer,
                total_iters=max(total_steps - self.warmup_steps, 1),
                power=self.lr_scheduler_power,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup, decay], milestones=[self.warmup_steps]
            )
        else:
            scheduler = torch.optim.lr_scheduler.PolynomialLR(  # upstream :189-200
                optimizer, total_iters=total_steps, power=self.lr_scheduler_power
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


__all__ = ["Report2CTModule"]
