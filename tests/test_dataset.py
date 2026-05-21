from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
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


def _build_synthetic_shard(
    root: Path, split: str, n: int, shape=(4, 6, 6, 4)
) -> list[str]:
    """Create a tiny <root>/<split>.npy + <split>.index.json for tests."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((n,) + shape).astype(np.float16)
    npy_path = root / f"{split}.npy"
    np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.float16, shape=arr.shape)[
        :
    ] = arr
    patient_ids = [f"{split}_p{i:04d}" for i in range(n)]
    with open(root / f"{split}.index.json", "w") as f:
        json.dump(
            {"patient_ids": patient_ids, "shape": list(arr.shape), "dtype": "float16"},
            f,
        )
    return patient_ids


def test_shard_mode_basic(tmp_path):
    from src.data.maisi_latent_dataset import MAISILatentDataset

    ids = _build_synthetic_shard(tmp_path, "train", n=3)
    ds = MAISILatentDataset(root=str(tmp_path), split="train", normalize=False)
    assert len(ds) == 3
    item = ds[1]
    assert isinstance(item["mu"], torch.Tensor)
    assert item["mu"].shape == (4, 6, 6, 4)
    assert item["mu"].dtype == torch.float32
    assert item["patient_id"] == ids[1]
    assert item["sample_id"] == ids[1]


def test_shard_mode_deterministic(tmp_path):
    from src.data.maisi_latent_dataset import MAISILatentDataset

    _build_synthetic_shard(tmp_path, "train", n=2)
    ds = MAISILatentDataset(root=str(tmp_path), split="train", normalize=False)
    a = ds[0]["mu"]
    b = ds[0]["mu"]
    assert torch.equal(a, b)


def test_shard_mode_normalize(tmp_path):
    from src.data.maisi_latent_dataset import MAISILatentDataset

    _build_synthetic_shard(tmp_path, "train", n=2)
    with open(tmp_path / "stats.json", "w") as f:
        json.dump(
            {"channel_mean": [0.0, 0.0, 0.0, 0.0], "channel_std": [2.0, 2.0, 2.0, 2.0]},
            f,
        )
    ds_raw = MAISILatentDataset(root=str(tmp_path), split="train", normalize=False)
    ds_norm = MAISILatentDataset(root=str(tmp_path), split="train", normalize=True)
    # std=2 => normalized values are half of raw values.
    assert torch.allclose(ds_norm[0]["mu"], ds_raw[0]["mu"] / 2.0, atol=1e-4)


def test_shard_takes_priority_over_patient_dirs(tmp_path):
    """When both layouts exist at the same root, the shard wins (it's the fast path)."""
    from src.data.maisi_latent_dataset import MAISILatentDataset

    # Build a 1-sample patient-dir layout AND a 2-sample shard at the same root.
    (tmp_path / "train" / "patient_X").mkdir(parents=True)
    torch.save(
        torch.zeros(4, 6, 6, 4, dtype=torch.float16),
        tmp_path / "train" / "patient_X" / "mu.pt",
    )
    _build_synthetic_shard(tmp_path, "train", n=2)

    ds = MAISILatentDataset(root=str(tmp_path), split="train", normalize=False)
    assert len(ds) == 2  # shard, not the 1-patient legacy dir


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
