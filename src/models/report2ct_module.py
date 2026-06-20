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

from typing import Any

import torch
from lightning.pytorch import LightningModule
from monai.networks.schedulers.ddpm import DDPMPredictionType
from torch import nn

from src.baselines.rflow import RFlowScheduler
from torch.nn import functional as F

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
        lr: float = 1e-4,
        cfg_drop_prob: float = CFG_DROP_PROB_DEFAULT,
        spacing_scale: float = 1.0,
        modality_class_label: int = 1,
        lr_scheduler_power: float = 2.0,
    ) -> None:
        """Wire up the training loop around an externally-built UNet + scheduler.

        Args:
            unet: denoising UNet, e.g. ``DiffusionModelUNetMaisi`` (Hydra-instantiated).
            noise_scheduler: ``RFlowScheduler`` (or any DDPM-style scheduler exposing
                ``add_noise`` / ``sample_timesteps``).
            lr: Adam learning rate (upstream :186).
            cfg_drop_prob: probability of zeroing the context for classifier-free
                guidance during training (upstream :295).
            spacing_scale: extra multiplier on the spacing tensor; ``1.0`` keeps the
                DataModule's already-scaled values untouched.
            modality_class_label: fixed modality class id; ``1`` = CT (upstream :270).
            lr_scheduler_power: exponent of the ``PolynomialLR`` decay (upstream :189-200).
        """
        super().__init__()
        # Hydra-instantiated objects are not serializable in hparams; ignore them.
        self.save_hyperparameters(ignore=["unet", "noise_scheduler"])
        self.unet = unet
        # RFlowScheduler inherits nn.Module but doesn't call super().__init__(),
        # so _parameters is missing and model.parameters() crashes.
        # Bypass nn.Module.__setattr__ to avoid submodule registration.
        object.__setattr__(self, "noise_scheduler", noise_scheduler)
        self.lr = lr
        self.cfg_drop_prob = cfg_drop_prob
        self.spacing_scale = spacing_scale
        self.modality_class_label = modality_class_label
        self.lr_scheduler_power = lr_scheduler_power

        self.register_buffer("scale_factor", torch.tensor(1.0))
        self._scale_factor_initialized: bool = False

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
        if "context" in batch:
            context = batch["context"]  # (B, num_organs, D) — e.g. fVLM (B, 4, 256)
        else:
            context = self._prepare_context(
                batch["context_f"], batch["context_i"]
            )  # upstream :286
        if self.training and torch.rand(()) < self.cfg_drop_prob:  # upstream :296-297
            context = torch.zeros_like(context)

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

        unet_inputs = {  # upstream :311-316 (+ class_labels at :325-330)
            "x": noisy_latent,
            "timesteps": timesteps,
            "spacing_tensor": spacing_tensor,
            "context": context,
            "class_labels": modality_tensor,
        }
        model_output = self.unet(**unet_inputs)  # upstream :331

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
        loss = self._shared_forward(batch)
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

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
        optimizer = torch.optim.Adam(
            self.unet.parameters(), lr=self.lr
        )  # upstream :186
        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(  # upstream :189-200
            optimizer, total_iters=total_steps, power=self.lr_scheduler_power
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


__all__ = ["Report2CTModule"]
