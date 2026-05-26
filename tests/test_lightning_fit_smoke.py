"""End-of-Phase-A Lightning wiring proof.

Runs `lightning.Trainer.fit` for 1 step on a dummy LightningModule + LightningDataModule
to confirm the full pl.Trainer.fit happy-path works before Phase B kickoff. Catches
import-time issues, version mismatches, and basic config sanity in one go.

This is the Phase A acceptance criterion `pl.Trainer.fit runs >=1 step` (plan section
'Phase A — 5/26 -> 5/31' Day 5).
"""

from __future__ import annotations

import torch
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset


class _DummyModule(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x).squeeze(-1), y)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)


class _DummyDM(LightningDataModule):
    def setup(self, stage: str | None = None) -> None:
        self.ds = TensorDataset(torch.randn(8, 4), torch.randn(8))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.ds, batch_size=4)


def test_trainer_fit_one_step_on_dummy_module(tmp_path) -> None:
    trainer = Trainer(
        max_steps=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(_DummyModule(), datamodule=_DummyDM())
    # If fit returned without raising, Lightning end-to-end works.
    assert trainer.global_step >= 1
