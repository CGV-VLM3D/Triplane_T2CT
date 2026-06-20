"""Discord notification Lightning callback.

Posts rich embeds to a Discord webhook at key training events:
start, best-checkpoint saved, NaN detected, training end (success or crash).

Reads DISCORD_WEBHOOK_URL from the environment (set in .env.local).
Silently skips all posts when the URL is not set — never blocks training.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import dotenv
import requests
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

dotenv.load_dotenv("/workspace/.env.local", override=False)
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint


def _post(webhook_url: str, embed: dict) -> None:
    """Best-effort POST — logs on failure, never raises."""
    try:
        resp = requests.post(
            webhook_url,
            json={"embeds": [embed]},
            timeout=5,
        )
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"[DiscordNotifier] webhook post failed: {exc}")


class DiscordNotifier(Callback):
    """Lightning callback that sends Discord notifications during training."""

    # Discord embed colour codes
    _COLOR_INFO = 3447003  # blue
    _COLOR_GOOD = 3066993  # green
    _COLOR_WARN = 15105570  # orange
    _COLOR_BAD = 15158332  # red

    def __init__(
        self,
        experiment_name: str = "unnamed",
        webhook_url: str | None = None,
    ) -> None:
        super().__init__()
        self.experiment_name = experiment_name
        # Prefer explicit kwarg; fall back to env var loaded from .env.local
        self._url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "")
        self._start_time: float = 0.0
        self._best_metric: float | None = None
        self._nan_reported = False

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _send(self, title: str, description: str, color: int) -> None:
        if not self._url:
            return
        _post(self._url, {"title": title, "description": description, "color": color})

    def _elapsed(self) -> str:
        secs = int(time.time() - self._start_time)
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._start_time = time.time()
        n_gpu = trainer.num_devices
        n_params = sum(p.numel() for p in pl_module.parameters()) / 1e6
        self._send(
            title=f"Training started — {self.experiment_name}",
            description=(
                f"**GPUs:** {n_gpu}\n"
                f"**Params:** {n_params:.1f}M\n"
                f"**Max epochs:** {trainer.max_epochs}"
            ),
            color=self._COLOR_INFO,
        )

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Detect best-checkpoint updates by watching ModelCheckpoint callbacks."""
        for cb in trainer.callbacks:
            if not isinstance(cb, ModelCheckpoint):
                continue
            if cb.best_model_score is None:
                continue
            current = cb.best_model_score.item()
            if self._best_metric is None or current != self._best_metric:
                self._best_metric = current
                self._send(
                    title=f"New best checkpoint — {self.experiment_name}",
                    description=(
                        f"**{cb.monitor}:** {current:.4f}\n"
                        f"**Epoch:** {trainer.current_epoch}\n"
                        f"**Elapsed:** {self._elapsed()}"
                    ),
                    color=self._COLOR_GOOD,
                )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._nan_reported:
            return
        # Check logged loss for NaN/Inf
        loss = trainer.callback_metrics.get("train/loss")
        if loss is not None and (torch.isnan(loss) or torch.isinf(loss)):
            self._nan_reported = True
            self._send(
                title=f"NaN/Inf loss detected — {self.experiment_name}",
                description=(
                    f"**Step:** {trainer.global_step}\n"
                    f"**Epoch:** {trainer.current_epoch}\n"
                    f"**Loss:** {loss.item()}"
                ),
                color=self._COLOR_BAD,
            )

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        best = f"{self._best_metric:.4f}" if self._best_metric is not None else "n/a"
        self._send(
            title=f"Training finished — {self.experiment_name}",
            description=(
                f"**Best metric:** {best}\n**Total elapsed:** {self._elapsed()}"
            ),
            color=self._COLOR_GOOD,
        )

    def on_exception(
        self, trainer: Trainer, pl_module: LightningModule, exception: BaseException
    ) -> None:
        self._send(
            title=f"Training CRASHED — {self.experiment_name}",
            description=(
                f"**Error:** `{type(exception).__name__}`\n"
                f"**Message:** {str(exception)[:300]}\n"
                f"**Elapsed:** {self._elapsed()}"
            ),
            color=self._COLOR_BAD,
        )
