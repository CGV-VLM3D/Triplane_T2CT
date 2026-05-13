from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

# stats.json lives at <root>/stats.json (sibling of the split subdirectories)


class MAISILatentDataset(Dataset):
    """MAISI latent dataset; always returns the deterministic posterior mean (mu)."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        normalize: bool = True,
    ) -> None:
        split_dir = Path(root) / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"{split_dir} does not exist")

        self.patient_dirs: list[Path] = sorted(
            p for p in split_dir.iterdir() if p.is_dir() and (p / "mu.pt").is_file()
        )
        if not self.patient_dirs:
            raise FileNotFoundError(f"No patient folders with mu.pt under {split_dir}")

        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        if normalize:
            stats_path = Path(root) / "stats.json"
            if not stats_path.is_file():
                raise FileNotFoundError(f"normalize=True but {stats_path} not found")
            with open(stats_path) as f:
                stats = json.load(f)
            self.mean = torch.tensor(stats["channel_mean"], dtype=torch.float32).view(
                -1, 1, 1, 1
            )
            self.std = (
                torch.tensor(stats["channel_std"], dtype=torch.float32)
                .clamp(min=1e-6)
                .view(-1, 1, 1, 1)
            )

    def __len__(self) -> int:
        return len(self.patient_dirs)

    def __getitem__(self, idx: int) -> dict:
        pdir = self.patient_dirs[idx]
        mu = torch.load(pdir / "mu.pt", weights_only=True).float()
        if self.mean is not None:
            mu = (mu - self.mean) / self.std
        return {"mu": mu, "patient_id": pdir.name}
