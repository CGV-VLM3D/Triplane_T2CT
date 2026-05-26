"""
Dataset for pre-extracted MAISI latents stored as per-patient folders:

    <latent_dir>/
    ├── train/
    │   ├── <patient_id>/
    │   │   ├── mu.pt        (Tensor [C, D, H, W])
    │   │   ├── sigma.pt     (Tensor [C, D, H, W])  -- optional
    │   │   └── src.txt
    │   └── ...
    └── valid/
        └── ...

Each __getitem__ returns:
    {"image": z, "patient_id": str}
where z = mu + sigma * eps (freshly sampled each call) when sigma exists and
sample_z is True. Otherwise z = mu.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class LatentDataset(Dataset):
    def __init__(
        self,
        latent_dir: str,
        split: str = "train",
        sample_z: bool = True,
        normalize: bool = True,
    ):
        root = Path(latent_dir) / split
        if not root.is_dir():
            raise FileNotFoundError(f"{root} does not exist")

        self.patient_dirs: list[Path] = sorted(
            p for p in root.iterdir() if p.is_dir() and (p / "mu.pt").is_file()
        )
        if not self.patient_dirs:
            raise FileNotFoundError(f"No patient folders with mu.pt under {root}")

        self.sample_z = sample_z

        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        if normalize:
            stats_path = Path(latent_dir) / "stats.json"
            if not stats_path.is_file():
                raise FileNotFoundError(
                    f"normalize=True but {stats_path} not found. "
                    f"Run scripts/compute_latent_stats.py first."
                )
            with open(stats_path) as f:
                stats = json.load(f)
            self.mean = torch.tensor(stats["channel_mean"], dtype=torch.float32).view(-1, 1, 1, 1)
            self.std = torch.tensor(stats["channel_std"], dtype=torch.float32).clamp(min=1e-6).view(-1, 1, 1, 1)

    def __len__(self) -> int:
        return len(self.patient_dirs)

    def __getitem__(self, idx: int) -> dict:
        pdir = self.patient_dirs[idx]
        mu = torch.load(pdir / "mu.pt", map_location="cpu").float()
        if self.sample_z and (pdir / "sigma.pt").is_file():
            sigma = torch.load(pdir / "sigma.pt", map_location="cpu").float()
            eps = torch.randn_like(mu)
            z = mu + sigma * eps
        else:
            z = mu
        if self.mean is not None:
            z = (z - self.mean) / self.std
        return {"image": z, "patient_id": pdir.name}


def build_latent_dataloader(
    latent_dir: str,
    split: str = "train",
    batch_size: int = 1,
    num_workers: int = 2,
    shuffle: bool = True,
    drop_last: bool = True,
    sample_z: bool = True,
    normalize: bool = True,
) -> DataLoader:
    ds = LatentDataset(
        latent_dir=latent_dir, split=split,
        sample_z=sample_z, normalize=normalize,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )
