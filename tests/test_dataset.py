from __future__ import annotations

from pathlib import Path

import pytest
import torch

DATASET_ROOT = "/workspace/datasets/datasets/latents"


def _skip_if_missing():
    if not Path(DATASET_ROOT).is_dir():
        pytest.skip(f"Dataset root {DATASET_ROOT} not found")


def _make_dataset(normalize: bool):
    from src.data.maisi_latent_dataset import MAISILatentDataset

    return MAISILatentDataset(root=DATASET_ROOT, split="train", normalize=normalize)


def test_shape_and_types():
    _skip_if_missing()
    ds = _make_dataset(normalize=False)
    item = ds[0]
    assert isinstance(item["mu"], torch.Tensor)
    assert item["mu"].shape == (4, 120, 120, 64)
    assert item["mu"].dtype == torch.float32
    assert isinstance(item["patient_id"], str) and len(item["patient_id"]) > 0


def test_normalization_changes_values():
    _skip_if_missing()
    ds_norm = _make_dataset(normalize=True)
    ds_raw = _make_dataset(normalize=False)
    assert not torch.equal(ds_norm[0]["mu"], ds_raw[0]["mu"])


def test_deterministic():
    _skip_if_missing()
    ds = _make_dataset(normalize=False)
    a = ds[0]["mu"]
    b = ds[0]["mu"]
    assert torch.equal(a, b)
