from __future__ import annotations

import glob
import os
from collections.abc import Sequence

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRanged,
)


def _scan_volumes(root: str) -> list[str]:
    """Find every .nii.gz file under <root>/train_*/<inner>/*.nii.gz."""
    pattern = os.path.join(root, "train_*", "*", "*.nii.gz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .nii.gz files found under {root} with pattern {pattern}")
    return files


class SaveSpacingd:
    """
    Save original voxel spacing from MONAI metadata.

    This is useful if later you want to condition a diffusion model on spacing,
    as in Report2CT/MAISI-style pipelines.
    """

    def __init__(self, key: str = "image"):
        self.key = key

    def __call__(self, d):
        meta = d.get(f"{self.key}_meta_dict", {})

        spacing = None

        if "pixdim" in meta:
            pixdim = meta["pixdim"]
            spacing = [float(pixdim[1]), float(pixdim[2]), float(pixdim[3])]

        # fallback from affine
        elif "affine" in meta:
            affine = np.asarray(meta["affine"])
            spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).tolist()

        if spacing is None:
            spacing = [1.0, 1.0, 1.0]

        d["spacing"] = torch.tensor(spacing, dtype=torch.float32)
        return d


def build_transforms(
    spatial_size: Sequence[int] = (480, 480, 256),
    hu_min: float = -1000.0,
    hu_max: float = 1000.0,
    out_min: float = 0.0,
    out_max: float = 1.0,
    train: bool = True,
) -> Compose:
    """
    Preprocessing for CT-RATE volumes (already stored in HU).

    Pipeline:
      load -> channel first -> save ORIGINAL spacing
      -> RAS orientation
      -> float32 -> clip HU [-1000, 1000] -> [0, 1]
      -> trilinear Resize to spatial_size (anisotropic stretch).

    No Spacingd: resampling to 1mm produces heavy air padding (e.g., a
    1024^2 0.34mm CT shrinks to 350^2, requiring ~36% air padding around it),
    which the AE wasn't trained on. Direct Resize keeps the body filling the
    full volume; effective spacing ends up ~0.7-1.3mm, closer to training.
    """

    keys = ["image"]

    transforms = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        SaveSpacingd(key="image"),
        Orientationd(keys=keys, axcodes="RAS"),
        EnsureTyped(keys=keys, dtype=torch.float32),
        ScaleIntensityRanged(
            keys=keys,
            a_min=hu_min,
            a_max=hu_max,
            b_min=out_min,
            b_max=out_max,
            clip=True,
        ),
        Resized(
            keys=keys,
            spatial_size=spatial_size,
            mode="trilinear",
            align_corners=False,
        ),
        EnsureTyped(keys=keys, dtype=torch.float32),
    ]

    return Compose(transforms)


class CTVolumeDataset(Dataset):
    """
    CT-RATE-style chest CT dataset for VQ-VAE / VQGAN training.

    Each item:
        batch["image"]:   [1, D, H, W] after transform
        batch["spacing"]: [3] original voxel spacing
    """

    def __init__(
        self,
        data_root: str = "/workspace/dataset/train",
        spatial_size: Sequence[int] = (480, 480, 256),
        hu_min: float = -1000.0,
        hu_max: float = 1000.0,
        out_min: float = 0.0,
        out_max: float = 1.0,
        train: bool = True,
        files: Sequence[str] | None = None,
    ):
        files = list(files) if files is not None else _scan_volumes(data_root)

        data = [{"image": f} for f in files]

        transform = build_transforms(
            spatial_size=spatial_size,
            hu_min=hu_min,
            hu_max=hu_max,
            out_min=out_min,
            out_max=out_max,
            train=train,
        )

        super().__init__(data=data, transform=transform)
        self.files = files


def build_dataloader(
    data_root: str = "/workspace/dataset/train",
    batch_size: int = 1,
    num_workers: int = 4,
    spatial_size: Sequence[int] = (480, 480, 256),
    hu_min: float = -1000.0,
    hu_max: float = 1000.0,
    out_min: float = 0.0,
    out_max: float = 1.0,
    train: bool = True,
    cache_rate: float = 0.0,
    val_split: float = 0.0,
    seed: int = 0,
) -> DataLoader | tuple[DataLoader, DataLoader]:
    """
    Build CT-RATE dataloader.

    If val_split > 0:
        returns train_loader, val_loader
    else:
        returns train_loader
    """

    files = _scan_volumes(data_root)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    files = [files[i] for i in idx]

    if val_split > 0.0:
        n_val = max(1, int(round(len(files) * val_split)))
        val_files = files[:n_val]
        train_files = files[n_val:]
    else:
        train_files, val_files = files, []

    def _make_dataset(file_list: Sequence[str], is_train: bool):
        data = [{"image": f} for f in file_list]

        transform = build_transforms(
            spatial_size=spatial_size,
            hu_min=hu_min,
            hu_max=hu_max,
            out_min=out_min,
            out_max=out_max,
            train=is_train,
        )

        if cache_rate > 0:
            return CacheDataset(
                data=data,
                transform=transform,
                cache_rate=cache_rate,
                num_workers=num_workers,
            )

        return Dataset(data=data, transform=transform)

    train_ds = _make_dataset(train_files, is_train=train)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )

    if val_split > 0.0:
        val_ds = _make_dataset(val_files, is_train=False)

        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=max(1, num_workers // 2),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        return train_loader, val_loader

    return train_loader


if __name__ == "__main__":
    loader = build_dataloader(
        data_root="/workspace/dataset/train",
        batch_size=1,
        num_workers=0,
        spatial_size=(480, 480, 256),
        cache_rate=0.0,
        val_split=0.0,
    )

    print("dataset size:", len(loader.dataset))
    print("num batches:", len(loader))

    batch = next(iter(loader))
    img = batch["image"]

    print("batch keys:", batch.keys())
    print("image shape:", tuple(img.shape))
    print("image dtype:", img.dtype)
    print("image min/max:", float(img.min()), float(img.max()))
    print("image mean/std:", float(img.mean()), float(img.std()))

    if "spacing" in batch:
        print("spacing:", batch["spacing"])