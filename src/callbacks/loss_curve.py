"""Loss-curve plotting Lightning callback.

Collects epoch-level ``train/loss`` and ``val/loss`` during training and, at
fit end, writes a PNG plot (``loss_curve.png``) plus the raw numbers
(``loss_history.csv``) into the Hydra output directory — so the loss history is
available locally in ``outputs/<task_name>/<KST-date>[_N]/`` independently of wandb.
"""

from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")  # headless: no display in the dev container
import matplotlib.pyplot as plt
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class LossCurvePlotter(Callback):
    """Save train/val loss curves as a PNG (+ CSV) into the output dir at fit end."""

    def __init__(
        self,
        output_dir: str,
        train_key: str = "train/loss_epoch",
        val_key: str = "val/loss",
        filename: str = "loss_curve.png",
    ) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.train_key = train_key
        self.val_key = val_key
        self.filename = filename
        # epoch -> {"train": float, "val": float}; either key may be absent
        self._history: dict[int, dict[str, float]] = {}

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Record this epoch's train/val loss from the logged metrics."""
        if trainer.sanity_checking:  # the pre-fit sanity val has no train loss yet
            return
        metrics = trainer.callback_metrics
        epoch = self._history.setdefault(trainer.current_epoch, {})
        train_loss = metrics.get(self.train_key)
        val_loss = metrics.get(self.val_key)
        if train_loss is not None:
            epoch["train"] = float(train_loss)
        if val_loss is not None:
            epoch["val"] = float(val_loss)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Render the collected history to PNG + CSV (rank-zero only, no-op if empty)."""
        if trainer.global_rank != 0 or not self._history:
            return
        epochs = sorted(self._history)
        train = [self._history[e].get("train") for e in epochs]
        val = [self._history[e].get("val") for e in epochs]

        os.makedirs(self.output_dir, exist_ok=True)
        self._write_csv(epochs, train, val)
        self._plot(epochs, train, val)

    def _write_csv(
        self, epochs: list[int], train: list[float | None], val: list[float | None]
    ) -> None:
        path = os.path.join(self.output_dir, "loss_history.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])
            for e, t, v in zip(epochs, train, val):
                writer.writerow([e, "" if t is None else t, "" if v is None else v])

    def _plot(
        self, epochs: list[int], train: list[float | None], val: list[float | None]
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        # drop None gaps so a partially-logged curve still plots cleanly
        t_pts = [(e, y) for e, y in zip(epochs, train) if y is not None]
        v_pts = [(e, y) for e, y in zip(epochs, val) if y is not None]
        if t_pts:
            ax.plot(*zip(*t_pts), marker="o", ms=3, label=self.train_key)
        if v_pts:
            ax.plot(*zip(*v_pts), marker="o", ms=3, label=self.val_key)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Training / validation loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, self.filename), dpi=120)
        plt.close(fig)
