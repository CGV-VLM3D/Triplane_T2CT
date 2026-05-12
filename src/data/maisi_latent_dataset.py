from __future__ import annotations

import torch
from torch.utils.data import Dataset


class MAISILatentDataset(Dataset):
    """Stub — full implementation in the data pipeline step."""

    def __init__(self, latent_dir: str | None = None, normalize: bool = False) -> None:
        self.latent_dir = latent_dir
        self.normalize = normalize

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Returns a dummy tensor so import and basic instantiation don't break.
        return torch.zeros(4, 120, 120, 64)
