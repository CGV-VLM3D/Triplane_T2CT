from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MAISILatentDataset(Dataset):
    """MAISI latent dataset; returns the deterministic posterior mean (mu).

    Two storage layouts are supported, auto-detected from ``root``:

    1. **Shard layout** (fast). Built by ``scripts/build_latent_shard.py``:
           <root>/<split>.npy            fp16 memmap, shape [N, 4, H, W, D]
           <root>/<split>.index.json     patient_ids in row order
           <root>/stats.json             optional, only needed for normalize=True

    2. **Patient-dir layout** (legacy, per-file):
           <root>/<split>/<patient_id>/mu.pt   fp16, [4, H, W, D]
           <root>/stats.json

    The memmap path opens the .npy once in ``__init__`` with ``mmap_mode="r"``;
    DataLoader workers fork and share the kernel page cache, eliminating the
    per-sample open/close overhead that dominates patient-dir runs.
    """

    def __init__(
        self,
        root: str = "/workspace/datasets/datasets/latents",
        split: str = "train",
        normalize: bool = True,
        load_ct_recon: bool = False,
        ct_recon_root: str = "/workspace/data/maisi_latent_with_recon",
    ) -> None:
        self.split = split
        root_path = Path(root)

        shard_npy = root_path / f"{split}.npy"
        shard_idx = root_path / f"{split}.index.json"
        self._shard_mode = shard_npy.is_file() and shard_idx.is_file()

        if self._shard_mode:
            with open(shard_idx) as f:
                idx = json.load(f)
            self._patient_ids: list[str] = list(idx["patient_ids"])
            self._mmap = np.load(shard_npy, mmap_mode="r")
            if len(self._patient_ids) != self._mmap.shape[0]:
                raise RuntimeError(
                    f"shard length mismatch: index has {len(self._patient_ids)} "
                    f"entries but {shard_npy.name} has {self._mmap.shape[0]} rows"
                )
        else:
            split_dir = root_path / split
            if not split_dir.is_dir():
                raise FileNotFoundError(f"{split_dir} does not exist")
            patient_dirs = sorted(
                p for p in split_dir.iterdir() if p.is_dir() and (p / "mu.pt").is_file()
            )
            if not patient_dirs:
                raise FileNotFoundError(
                    f"No patient folders with mu.pt under {split_dir}"
                )
            self._patient_dirs: list[Path] = patient_dirs
            self._patient_ids = [p.name for p in patient_dirs]

        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        if normalize:
            stats_path = root_path / "stats.json"
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

        self.load_ct_recon = load_ct_recon
        if load_ct_recon:
            self.ct_recon_dir = Path(ct_recon_root) / split

    def __len__(self) -> int:
        return len(self._patient_ids)

    def _read_mu(self, idx: int) -> torch.Tensor:
        if self._shard_mode:
            # .copy() forces a writable fp16 buffer out of the read-only mmap
            # (silences torch.from_numpy's "not writable" UserWarning); .float()
            # then upcasts to a fresh fp32 tensor.
            return torch.from_numpy(self._mmap[idx].copy()).float()
        return torch.load(self._patient_dirs[idx] / "mu.pt", weights_only=True).float()

    def __getitem__(self, idx: int) -> dict:
        mu = self._read_mu(idx)
        if self.mean is not None:
            mu = (mu - self.mean) / self.std

        patient_id = self._patient_ids[idx]
        item = {"mu": mu, "patient_id": patient_id, "sample_id": patient_id}

        if self.load_ct_recon:
            ct_recon_path = self.ct_recon_dir / patient_id / "ct_recon.pt"
            if not ct_recon_path.is_file():
                raise FileNotFoundError(
                    f"ct_recon not found: {ct_recon_path}. "
                    f"Run scripts/measure_upper_bound.py --cache-recon "
                    f"--split={self.split} first."
                )
            ct_recon = torch.load(ct_recon_path, weights_only=True).float()
            if ct_recon.dim() == 5:
                ct_recon = ct_recon.squeeze(0)
            item["ct_recon"] = ct_recon

        return item
