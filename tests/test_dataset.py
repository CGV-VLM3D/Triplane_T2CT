from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

DATASET_ROOT = "/workspace/datasets/datasets/latents"
CT_RECON_ROOT = "/workspace/data/maisi_latent_with_recon"


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
    assert "sample_id" in item


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


def test_sample_id_alias():
    _skip_if_missing()
    ds = _make_dataset(normalize=False)
    item = ds[0]
    assert item["sample_id"] == item["patient_id"]


def test_no_ct_recon_default():
    _skip_if_missing()
    ds = _make_dataset(normalize=False)
    item = ds[0]
    assert "ct_recon" not in item


def test_load_ct_recon():
    _skip_if_missing()
    from src.data.maisi_latent_dataset import MAISILatentDataset

    split = "valid"
    ct_recon_split_dir = Path(CT_RECON_ROOT) / split
    if not ct_recon_split_dir.is_dir() or not any(ct_recon_split_dir.iterdir()):
        pytest.skip(f"ct_recon cache not found at {ct_recon_split_dir}")

    valid_root = Path(DATASET_ROOT)
    if not (valid_root / split).is_dir():
        pytest.skip(f"Valid split not found at {valid_root / split}")

    ds = MAISILatentDataset(
        root=DATASET_ROOT,
        split=split,
        normalize=False,
        load_ct_recon=True,
        ct_recon_root=CT_RECON_ROOT,
    )
    item = ds[0]
    assert "ct_recon" in item
    assert item["ct_recon"].shape == (1, 480, 480, 256)
    assert item["ct_recon"].dtype == torch.float32
    assert float(item["ct_recon"].min()) >= 0.0
    assert float(item["ct_recon"].max()) <= 1.0


def test_missing_ct_recon_raises():
    _skip_if_missing()
    from src.data.maisi_latent_dataset import MAISILatentDataset

    # Use a real latent split so the dataset can initialize, but point ct_recon_root
    # at a tmp dir that contains no patient subdirs — triggering FileNotFoundError on
    # the first __getitem__ call.
    with tempfile.TemporaryDirectory() as tmp:
        ds = MAISILatentDataset(
            root=DATASET_ROOT,
            split="train",
            normalize=False,
            load_ct_recon=True,
            ct_recon_root=tmp,
        )
        with pytest.raises(FileNotFoundError) as exc_info:
            _ = ds[0]
        msg = str(exc_info.value)
        assert "ct_recon not found" in msg
        assert "--cache-recon" in msg
