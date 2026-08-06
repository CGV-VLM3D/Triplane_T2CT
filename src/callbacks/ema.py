"""Exponential moving average of the model weights (online EMA).

Standard practice for diffusion models (LDM / DiT / ADM / EDM2): the EMA of the weights
usually scores better than the raw weights. This repo previously approximated it offline
with ``scripts/average_wan_checkpoints.py`` ("a cheap standalone proxy for the online EMA
this run never tracked"); this callback tracks it online instead.

Purely additive — it never touches the loss, the optimizer or the parameters themselves, so
the raw ``epoch_NNN.ckpt`` a run writes stays bit-identical to a run without this callback.

This callback only *tracks* the average; it writes no files of its own. The shadow rides
along in Lightning's callback-state slot of every checkpoint the run saves
(``ckpt["callbacks"]["EMACallback"]``), which both resumes the average and makes each
epoch's EMA recoverable after the fact — ``scripts/extract_ema.py`` turns any snapshot into
a standalone ``{"state_dict": ...}`` checkpoint the eval samplers load unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class EMACallback(Callback):
    """Track an EMA of the model's float parameters into every saved checkpoint.

    Buffers are deliberately NOT averaged, which is why the shadow covers parameters only.
    ``Report2CTModule.scale_factor`` is a lazily initialised data statistic
    (``report2ct_module.py:168``) that jumps once on the first batch and then stays put; the
    eval samplers read it straight out of the saved ``state_dict``. Averaging it would leave
    it crawling up from its ``1.0`` init forever and silently generate at the wrong latent
    scale. ``scripts/extract_ema.py`` therefore takes buffers from the raw ``state_dict``.
    """

    def __init__(self, decay: float = 0.9999) -> None:
        """
        Args:
            decay: target EMA decay. The effective decay is ramped up from 0.1 as
                ``min(decay, (1 + n) / (10 + n))`` so the average is not dragged by the
                random init during the first steps.
        """
        super().__init__()
        self.decay = decay
        # param name -> fp32 shadow tensor; None until on_train_start (or a resume restore)
        self._shadow: dict[str, torch.Tensor] | None = None
        self._num_updates = 0

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Seed the shadow from the current weights, or move a resumed shadow to the device."""
        # load_state_dict() runs before this hook on resume — don't clobber a restored shadow.
        if self._shadow is None:
            self._shadow = {
                name: p.detach().to(torch.float32).clone()
                for name, p in pl_module.named_parameters()
                if p.is_floating_point()
            }
            log.info(
                "EMA: tracking %d tensors (%.2f M params), decay=%g",
                len(self._shadow),
                sum(t.numel() for t in self._shadow.values()) / 1e6,
                self.decay,
            )
        else:
            self._shadow = {
                name: t.to(device=pl_module.device, dtype=torch.float32)
                for name, t in self._shadow.items()
            }
            log.info("EMA: resumed at %d updates", self._num_updates)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Blend the current weights into the shadow (in-place, no grad)."""
        assert self._shadow is not None
        # Warmup ramp: 0.1 at the first update, approaching self.decay from below.
        d = min(self.decay, (1 + self._num_updates) / (10 + self._num_updates))
        for name, p in pl_module.named_parameters():
            shadow = self._shadow.get(name)
            if shadow is not None:
                # shadow <- shadow * d + p * (1 - d); params are fp32 masters under bf16-mixed
                shadow.lerp_(p.detach().to(torch.float32), 1.0 - d)
        self._num_updates += 1

    def state_dict(self) -> dict[str, Any]:
        """Persist the shadow into every saved checkpoint (resume + post-hoc extraction)."""
        assert self._shadow is not None
        return {
            "shadow": {k: v.detach().cpu() for k, v in self._shadow.items()},
            "num_updates": self._num_updates,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the shadow on resume; ``on_train_start`` moves it back to the device."""
        self._shadow = dict(state_dict["shadow"])
        self._num_updates = state_dict["num_updates"]
