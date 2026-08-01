"""Text2CT-latent + CLIP3D-cond + top-K organ-mask LightningDataModule.

Sibling of `src/data/fvlm_cond_datamodule.py`, for the mask-conditioning experiment on the
**text2ct VAE latent** substrate. Per sample it emits:
  - ``image``         : text2ct latent ``(4, 128, 128, 32)`` (from ``<vol>.nii.gz``, HWDC→CHWD)
  - ``context``       : FrozenCLIP3D findings+impression embedding ``(1, 768)`` (from ``.npy``)
  - ``mask_classes``  : top-K organ ids ``(K, 128, 128, 32)`` long   (from an M4 ``.pt``)
  - ``mask_fracs``    : matching area fractions ``(K, 128, 128, 32)`` float
  - ``spacing``       : ``(3,)`` voxel spacing × ``spacing_multiplier`` (text2ct is a fixed
                        ``[0.75, 0.75, 3.0]``)

``image``/``context``/``spacing`` are consumed by ``Report2CTModule`` exactly as the CLIP3D
variant; ``mask_classes``/``mask_fracs`` feed the injected ``MaskConditioner``. The datalist is
produced by ``scripts/build_text2ct_mask_datalist.py``.

Datalist JSON contract:
    {"training": [{"image": "<abs .nii.gz>", "context": "<abs .npy>",
                    "mask": "<abs M4 .pt>", "spacing": [0.75, 0.75, 3.0]}, ...],
     "validation": [...]}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import monai
import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose, MapTransform


def _load_context_npy(path: str) -> torch.Tensor:
    """Load a FrozenCLIP3D condition ``(1, 768)`` from a text2ct ``*_impression_xgem_3D.npy``."""
    return torch.from_numpy(np.load(path)).float()  # (1, 768)


class LoadMaskTopKd(MapTransform):
    """Load an top-K mask ``.pt`` (path at ``mask``) into ``mask_classes`` + ``mask_fracs``.

    Module-level class (picklable) so ``CacheDataset`` multiprocessing caching works.
    """

    def __init__(self, key: str = "mask") -> None:
        super().__init__(keys=[key])
        self.key = key

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        md = torch.load(d[self.key], weights_only=True)
        d["mask_classes"] = md["classes"].long()  # (K, H, W, D)
        d["mask_fracs"] = md["fracs"].float()  # (K, H, W, D)
        del d[self.key]
        return d


def _build_transforms(spacing_multiplier: float = 1e2) -> Compose:
    """MONAI transforms: text2ct latent (HWDC→CHWD), spacing×mult, CLIP3D npy, top-K mask ``.pt``."""
    return Compose(
        [
            monai.transforms.LoadImaged(keys=["image"]),  # (128,128,32,4)
            monai.transforms.EnsureChannelFirstd(keys=["image"]),  # (4,128,128,32)
            monai.transforms.Lambdad(
                keys="spacing",
                func=lambda x: (
                    torch.as_tensor(x, dtype=torch.float32) * spacing_multiplier
                ),
            ),
            monai.transforms.Lambdad(keys="context", func=_load_context_npy),  # (1,768)
            LoadMaskTopKd(
                key="mask"
            ),  # -> mask_classes (K,·) long, mask_fracs (K,·) float
        ]
    )


class Text2CTMaskDataModule(LightningDataModule):
    """LightningDataModule: text2ct latent + CLIP3D cond + top-K organ mask."""

    def __init__(
        self,
        datalist_path: str | Path,
        batch_size: int = 2,
        num_workers: int = 6,
        cache_rate: float = 0.0,
        spacing_multiplier: float = 1e2,
        pin_memory: bool = True,
    ) -> None:
        """Configure the module.

        Args:
            datalist_path: JSON from ``scripts/build_text2ct_mask_datalist.py`` with
                ``training``/``validation`` entry lists (image/context/mask/spacing).
            batch_size: samples per batch.
            num_workers: DataLoader / CacheDataset workers.
            cache_rate: fraction cached in RAM (0.0 = none). Note the top-K mask ``.pt`` is
                ~6 MB/scan, so caching the full 5k adds ~30 GB.
            spacing_multiplier: scalar on voxel spacing (×100, MAISI/Report2CT convention).
            pin_memory: pin host memory for faster GPU transfer.
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
        """Parse the datalist and build ``CacheDataset`` instances gated by ``stage``."""
        if not self.datalist_path.is_file():
            raise FileNotFoundError(
                f"text2ct-mask datalist not found at {self.datalist_path}"
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
        """Shuffled ``DataLoader`` over the training split."""
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
        """Unshuffled ``DataLoader`` over the validation split."""
        if self.val_ds is None:
            self.setup("validate")
        return DataLoader(
            self.val_ds,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
        )


__all__ = ["Text2CTMaskDataModule", "LoadMaskTopKd"]
