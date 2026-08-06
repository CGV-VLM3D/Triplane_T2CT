"""Report2CT LightningDataModule.

Transforms mirror upstream `diff_model_train_vlm3D_2560_multi_text.py:84-111`:
- LoadImaged + EnsureChannelFirstd on the embedding NIfTI (image)
- Lambdad to load `spacing` from JSON and multiply by 1e2 (upstream :87-88)
- Lambdad to load `findings_embeddings` / `impression_embeddings` from JSON (upstream :89-90)

Datalist JSON contract (produced by `scripts/build_report2ct_datalist.py`):
    {
      "training":   [{"image": "<abs.nii.gz>", "spacing": "<abs.json>",
                       "context_f": "<abs.json>", "context_i": "<abs.json>"} , ...],
      "validation": [...]
    }

All four entry values are absolute paths; the same JSON file is referenced 3 times
(spacing/context_f/context_i) so that all three keys can be extracted by separate
Lambdad transforms (matches upstream behavior).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import monai
import torch
from lightning.pytorch import LightningDataModule
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose
from torch.utils.data import Dataset

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=False)


class _SkipCorruptDataset(Dataset):
    """Wraps a dataset so a bad sample logs + retries instead of killing the whole run.

    A single truncated/corrupt file (e.g. disk-full-interrupted write) otherwise crashes
    every DDP rank minutes into a multi-day training job — one bad file out of tens of
    thousands shouldn't cost the whole run. Retries are bounded so a systemic problem
    (e.g. a stale datalist) still surfaces as an error instead of looping forever.
    """

    def __init__(self, dataset: Dataset, max_retries: int = 20) -> None:
        self.dataset = dataset
        self.max_retries = max_retries

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        for attempt in range(self.max_retries):
            try:
                return self.dataset[index]
            except Exception as exc:  # noqa: BLE001 — one bad sample must not kill training
                path = self.dataset.data[index].get("image", "<unknown>")
                log.error(
                    "Corrupt sample skipped (attempt %d/%d): %s — %s: %s",
                    attempt + 1,
                    self.max_retries,
                    path,
                    type(exc).__name__,
                    exc,
                )
                index = random.randrange(len(self.dataset))
        raise RuntimeError(
            f"{self.max_retries} consecutive corrupt samples starting near index "
            f"{index} — likely a systemic data problem, not a one-off bad file."
        )


def _load_json_key(file_path: str, key: str) -> torch.Tensor:
    """Load a numeric array under `key` from a JSON file → float32 tensor.

    Matches `_load_data_from_file` in upstream training script :80-82.
    """
    with open(file_path) as f:
        data = json.load(f)
    return torch.FloatTensor(data[key])


def _build_transforms(spacing_multiplier: float = 1e2) -> Compose:
    """Builds the MONAI transform pipeline mirroring the upstream Report2CT training script.

    Transforms applied in order (matching upstream ``diff_model_train_vlm3D_2560_multi_text.py:84-111``):

    1. ``LoadImaged`` + ``EnsureChannelFirstd`` on ``"image"``:
       ``_emb.nii.gz`` (H, W, D, C=4) → ``(4, 120, 120, 64)``
    2. ``Lambdad`` on ``"spacing"``: JSON path → ``"spacing"`` array → ``float32`` tensor
       ``(3,)`` scaled by ``spacing_multiplier`` (default ×100).
    3. ``Lambdad`` on ``"context_f"``: JSON path → ``findings_embeddings`` tensor
       (shape determined by the CT-CLIP text encoder; not fixed here).
    4. ``Lambdad`` on ``"context_i"``: JSON path → ``impression_embeddings`` tensor
       (same encoder, same variable shape).

    Args:
        spacing_multiplier: Scalar applied to voxel spacing values (default ×100,
            matching upstream script line :88).

    Returns:
        A ``Compose`` transform ready to pass to ``CacheDataset``.
    """
    return Compose(
        [
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image"]),
            monai.transforms.Lambdad(
                keys="spacing", func=lambda x: _load_json_key(x, "spacing")
            ),
            monai.transforms.Lambdad(
                keys="spacing", func=lambda x: x * spacing_multiplier
            ),
            monai.transforms.Lambdad(
                keys="context_f",
                func=lambda x: _load_json_key(x, "findings_embeddings"),
            ),
            monai.transforms.Lambdad(
                keys="context_i",
                func=lambda x: _load_json_key(x, "impression_embeddings"),
            ),
        ]
    )


class Report2CTDataModule(LightningDataModule):
    """LightningDataModule reading precomputed image + text embeddings from a datalist JSON."""

    def __init__(
        self,
        datalist_path: str | Path,
        batch_size: int = 2,
        num_workers: int = 6,
        cache_rate: float = 0.0,
        spacing_multiplier: float = 1e2,
        pin_memory: bool = True,
    ) -> None:
        """Configures the Report2CT data module.

        Args:
            datalist_path: Path to the JSON produced by
                ``scripts/build_report2ct_datalist.py`` containing
                ``"training"`` and ``"validation"`` entry lists, each with
                ``"image"``, ``"spacing"``, ``"context_f"``, and ``"context_i"`` keys.
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
        self.train_ds: _SkipCorruptDataset | None = None
        self.val_ds: _SkipCorruptDataset | None = None

    def _transforms(self) -> Compose:
        """Build the per-sample transform pipeline (overridable hook for subclass variants)."""
        return _build_transforms(self.spacing_multiplier)

    def setup(self, stage: str | None = None) -> None:
        """Parses the datalist JSON and builds ``CacheDataset`` instances.

        Reads ``"training"`` and ``"validation"`` entry lists from the datalist,
        then constructs ``CacheDataset`` objects gated by ``stage``:
        ``"fit"`` builds train (and optionally val); ``"validate"`` builds val only;
        ``None`` builds both.
        """
        if not self.datalist_path.is_file():
            raise FileNotFoundError(
                f"Report2CT datalist not found at {self.datalist_path}"
            )
        with open(self.datalist_path) as f:
            data = json.load(f)
        self.train_files = data.get("training", [])
        self.val_files = data.get("validation", [])
        transforms = self._transforms()
        if stage in (None, "fit"):
            self.train_ds = _SkipCorruptDataset(
                CacheDataset(
                    data=self.train_files,
                    transform=transforms,
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers,
                )
            )
        if stage in (None, "fit", "validate"):
            self.val_ds = _SkipCorruptDataset(
                CacheDataset(
                    data=self.val_files,
                    transform=transforms,
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers,
                )
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


__all__ = ["Report2CTDataModule"]
