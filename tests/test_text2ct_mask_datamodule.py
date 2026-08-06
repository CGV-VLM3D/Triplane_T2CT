"""Unit test for src.data.text2ct_mask_datamodule.Text2CTMaskDataModule.

Self-contained: synthesizes a tiny latent NIfTI (HWDC), a CLIP3D `.npy`, and an M4-style mask
`.pt`, writes a datalist, and checks one batch has the shapes/dtypes Report2CTModule + the
MaskConditioner expect.
"""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import torch

from src.data.text2ct_mask_datamodule import Text2CTMaskDataModule

H, W, D, C, K = 16, 12, 8, 4, 4


def _make_sample(root, vol: str) -> dict:
    # latent NIfTI stored (H, W, D, C) like report2ct/text2ct _emb latents
    lat = np.random.randn(H, W, D, C).astype(np.float32)
    lat_path = root / f"{vol}.nii.gz"
    nib.save(
        nib.Nifti1Image(lat, affine=np.diag([0.75, 0.75, 3.0, 1.0])), str(lat_path)
    )
    # CLIP3D cond (1, 768)
    ctx_path = root / f"{vol}_ctx.npy"
    np.save(ctx_path, np.random.randn(1, 768).astype(np.float32))
    # M4 top-K mask
    msk_path = root / f"{vol}_mask.pt"
    torch.save(
        {
            "classes": torch.randint(0, 118, (K, H, W, D), dtype=torch.uint8),
            "fracs": torch.softmax(torch.randn(K, H, W, D), dim=0).to(torch.float16),
        },
        msk_path,
    )
    return {
        "image": str(lat_path),
        "context": str(ctx_path),
        "mask": str(msk_path),
        "spacing": [0.75, 0.75, 3.0],
    }


def test_one_batch_shapes(tmp_path) -> None:
    entries = [_make_sample(tmp_path, f"train_{i}_a_1") for i in range(4)]
    dl_path = tmp_path / "datalist.json"
    dl_path.write_text(json.dumps({"training": entries[:3], "validation": entries[3:]}))

    dm = Text2CTMaskDataModule(
        datalist_path=dl_path, batch_size=2, num_workers=0, cache_rate=0.0
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))

    assert batch["image"].shape == (2, C, H, W, D), batch["image"].shape
    assert batch["context"].shape == (2, 1, 768), batch["context"].shape
    assert batch["mask_classes"].shape == (2, K, H, W, D)
    assert batch["mask_classes"].dtype == torch.long
    assert batch["mask_fracs"].shape == (2, K, H, W, D)
    assert batch["mask_fracs"].dtype == torch.float32
    assert batch["spacing"].shape == (2, 3)
    # spacing × 100 (MAISI convention): [0.75,0.75,3.0] -> [75,75,300]
    assert torch.allclose(
        batch["spacing"][0], torch.tensor([75.0, 75.0, 300.0]), atol=1e-3
    )
    # fracs sum to 1 over K (per voxel)
    assert torch.allclose(batch["mask_fracs"].sum(1), torch.ones(2, H, W, D), atol=1e-2)
    # "mask" path key consumed (replaced by the two tensors)
    assert "mask" not in batch
