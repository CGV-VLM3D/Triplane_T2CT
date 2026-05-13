from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = ROOT / "reference"
sys.path.insert(0, str(REFERENCE_DIR))
sys.path.insert(0, str(REFERENCE_DIR / "scripts"))

LATENT_ROOT_DEFAULT = "/workspace/datasets/datasets/latents"
RECON_ROOT_DEFAULT = "/workspace/data/maisi_latent_with_recon"

SPATIAL_SIZE = (480, 480, 256)
HU_MIN, HU_MAX = -1000.0, 1000.0
OUT_MIN, OUT_MAX = 0.0, 1.0


class MAISILatentDataset(Dataset):
    """Dataset over pre-extracted MAISI VAE latents.

    Directory layout expected:
      <root>/<split>/<sample_id>/mu.pt          float16, [4, 120, 120, 64]
      <root>/<split>/<sample_id>/src.txt        path to source NIfTI (for gt loading)
      <recon_root>/<split>/<sample_id>/ct_recon.pt  float16, [1, 480, 480, 256]

    Items returned:
      {
        "mu":       [4, 120, 120, 64] float32,
        "ct_recon": [1, 480, 480, 256] float32  (or None tensor if load_ct_recon=False),
        "gt":       [1, 480, 480, 256] float32  (or None tensor if load_gt=False),
        "sample_id": str,
      }
    """

    def __init__(
        self,
        split: str,
        root: str = LATENT_ROOT_DEFAULT,
        recon_root: str | None = RECON_ROOT_DEFAULT,
        load_ct_recon: bool = False,
        load_gt: bool = False,
        normalize: bool = False,
    ) -> None:
        self.split = split
        self.latent_dir = Path(root) / split
        self.recon_root = Path(recon_root) / split if recon_root else None
        self.load_ct_recon = load_ct_recon
        self.load_gt = load_gt
        self.normalize = normalize

        self.sample_dirs = sorted(self.latent_dir.iterdir())
        if not self.sample_dirs:
            raise FileNotFoundError(f"No samples found under {self.latent_dir}")

        if load_ct_recon and self.recon_root is None:
            raise ValueError("load_ct_recon=True requires recon_root to be set.")

        self._gt_transform = None
        if load_gt:
            from models.dataloader import build_transforms

            self._gt_transform = build_transforms(
                spatial_size=SPATIAL_SIZE,
                hu_min=HU_MIN,
                hu_max=HU_MAX,
                out_min=OUT_MIN,
                out_max=OUT_MAX,
                train=False,
            )

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> dict:
        sample_dir = self.sample_dirs[idx]
        sample_id = sample_dir.name

        mu = torch.load(sample_dir / "mu.pt", weights_only=True).float()

        if self.normalize:
            mu_min = mu.min()
            mu_max = mu.max()
            drange = (mu_max - mu_min).clamp(min=1e-6)
            mu = (mu - mu_min) / drange

        ct_recon = None
        if self.load_ct_recon:
            recon_path = self.recon_root / sample_id / "ct_recon.pt"
            if not recon_path.exists():
                raise FileNotFoundError(
                    f"Cached ct_recon not found at {recon_path}. "
                    "Run `python scripts/measure_upper_bound.py --cache-recon` first "
                    "to generate the MAISI reconstruction cache."
                )
            ct_recon = torch.load(recon_path, weights_only=True).float()

        gt = None
        if self.load_gt:
            src_path = sample_dir / "src.txt"
            nifti_path = src_path.read_text().strip()
            item = self._gt_transform({"image": nifti_path})
            gt_raw = item["image"]
            if hasattr(gt_raw, "as_tensor"):
                gt_raw = gt_raw.as_tensor()
            gt = gt_raw.float()

        return {
            "mu": mu,
            "ct_recon": ct_recon if ct_recon is not None else torch.tensor([]),
            "gt": gt if gt is not None else torch.tensor([]),
            "sample_id": sample_id,
        }
