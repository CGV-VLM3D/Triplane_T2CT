"""fVLM-conditioned ctgen LightningDataModule.

Sibling of `src/data/report2ct_datamodule.py`. Same `CacheDataset`/`DataLoader`
skeleton and the same image/spacing transforms, but the per-sample condition is the
frozen fVLM per-organ embedding `(4, 256)` instead of the 2560-d findings/impression
vectors. The datalist is produced by `scripts/build_fvlm_cond_datalist.py`.

Datalist JSON contract:
    {
      "training":   [{"image": "<abs _emb.nii.gz>", "spacing": [sx, sy, sz],
                       "context": "<abs .pt with a (4, 256) tensor>"}, ...],
      "validation": [...]
    }

The image path is the Report2CT `_emb.nii.gz` latent (identical to Report2CT
training); `spacing` is the per-scan voxel spacing already baked into the datalist
as a literal list; `context` points at one fVLM embedding `.pt`. The module emits
`batch["context"]` directly — consumed as-is by `Report2CTModule._shared_forward`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import monai
import torch
from lightning.pytorch import LightningDataModule
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose


def _load_context_pt(path: str) -> torch.Tensor:
    """Load a precomputed fVLM condition tensor `(num_organs, 256)` from a `.pt`."""
    return torch.load(path, weights_only=True).float()  # (4, 256)


def _build_transforms(spacing_multiplier: float = 1e2) -> Compose:
    """Builds the MONAI transform pipeline for fVLM-conditioned samples.

    Transforms applied in order:

    1. ``LoadImaged`` + ``EnsureChannelFirstd`` on ``"image"``:
       ``_emb.nii.gz`` (H, W, D, C=4) → ``(4, 120, 120, 64)``
    2. ``Lambdad`` on ``"spacing"``: literal ``[sx, sy, sz]`` list →
       ``float32`` tensor ``(3,)`` scaled by ``spacing_multiplier`` (default ×100).
    3. ``Lambdad`` on ``"context"``: ``.pt`` path →
       fVLM per-organ embedding ``(num_organs, 256)`` (typically ``(4, 256)``).

    Args:
        spacing_multiplier: Scalar applied to the raw voxel spacing values,
            matching the upstream Report2CT training script convention (×1e2).

    Returns:
        A ``Compose`` transform ready to pass to ``CacheDataset``.
    """
    return Compose(
        [
            # _emb.nii.gz is (H, W, D, C=4) → channel-first (C, H, W, D) = (4, 120, 120, 64)
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image"]),
            # spacing is a literal [sx, sy, sz] in the datalist → tensor × 1e2 (upstream :88)
            monai.transforms.Lambdad(
                keys="spacing",
                func=lambda x: (
                    torch.as_tensor(x, dtype=torch.float32) * spacing_multiplier
                ),
            ),
            # context path → (4, 256) fVLM per-organ embedding
            monai.transforms.Lambdad(keys="context", func=_load_context_pt),
        ]
    )


class FVLMCondDataModule(LightningDataModule):
    """LightningDataModule reading `_emb.nii.gz` image latents + fVLM condition `.pt`."""

    def __init__(
        self,
        datalist_path: str | Path,
        batch_size: int = 2,
        num_workers: int = 6,
        cache_rate: float = 0.0,
        spacing_multiplier: float = 1e2,
        pin_memory: bool = True,
    ) -> None:
        """Configures the fVLM-conditioned data module.

        Args:
            datalist_path: Path to the JSON produced by
                ``scripts/build_fvlm_cond_datalist.py`` containing
                ``"training"`` and ``"validation"`` entry lists.
            batch_size: Samples per batch passed to ``DataLoader``.
            num_workers: Parallel workers for ``DataLoader`` and ``CacheDataset``.
            cache_rate: Fraction of the dataset to cache in RAM (0.0 = no cache).
            spacing_multiplier: Scalar applied to voxel spacing values (default ×100,
                matching the upstream Report2CT training script).
            pin_memory: Whether to pin host memory for faster GPU transfer.
        """
        super().__init__()
        self.save_hyperparameters()
        self.datalist_path = Path(datalist_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.spacing_multiplier = spacing_multiplier
        self.pin_memory = pin_memory
        self.train_files: list[dict[str, Any]] = []
        self.val_files: list[dict[str, Any]] = []
        self.train_ds: CacheDataset | None = None
        self.val_ds: CacheDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Parses the datalist JSON and builds ``CacheDataset`` instances.

        Reads ``"training"`` and ``"validation"`` entry lists from the datalist,
        then constructs ``CacheDataset`` objects gated by ``stage``:
        ``"fit"`` builds train (and optionally val); ``"validate"`` builds val only;
        ``None`` builds both.
        """
        if not self.datalist_path.is_file():
            raise FileNotFoundError(
                f"fVLM-cond datalist not found at {self.datalist_path}"
            )
        with open(self.datalist_path) as f:
            data = json.load(f)
        self.train_files = data.get("training", [])
        self.val_files = data.get("validation", [])
        transforms = _build_transforms(self.spacing_multiplier)
        if stage in (None, "fit"):
            self.train_ds = CacheDataset(
                data=self.train_files,
                transform=transforms,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
            )
        if stage in (None, "fit", "validate") and self.val_files:
            self.val_ds = CacheDataset(
                data=self.val_files,
                transform=transforms,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
            )

    def train_dataloader(self) -> DataLoader:
        """Returns a shuffled ``DataLoader`` over the training split."""
        if self.train_ds is None:
            self.setup("fit")
        return DataLoader(
            self.train_ds,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns an unshuffled ``DataLoader`` over the validation split."""
        if self.val_ds is None:
            self.setup("validate")
        return DataLoader(
            self.val_ds,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
        )


__all__ = ["FVLMCondDataModule"]
