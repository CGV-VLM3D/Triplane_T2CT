"""CT-RATE LightningDataModule smoke (metadata mode).

Phase A Day 2 acceptance: this file passes EOD.
Heavy volume-loading transforms are deferred to Day 4 EDA / Phase C.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.ct_rate_datamodule import (
    CT_RATE_ROOT,
    CTRateDataModule,
    CTRateRecord,
    _parse_spacing,
    _resolve_nifti_path,
    load_records,
)


def test_parse_spacing_handles_float_and_stringified_list() -> None:
    assert _parse_spacing(0.34) == pytest.approx(0.34)
    assert _parse_spacing("[0.341796875, 0.341796875]") == pytest.approx(0.341796875)
    assert _parse_spacing("1.5") == pytest.approx(1.5)
    assert _parse_spacing(None) is None
    assert _parse_spacing("") is None
    assert _parse_spacing("not-a-number") is None


def test_resolve_nifti_path_layout() -> None:
    path = _resolve_nifti_path("valid_1_a_1.nii.gz", Path("/tmp/v"))
    assert path == Path("/tmp/v/valid_1/valid_1_a/valid_1_a_1.nii.gz")


@pytest.mark.skipif(
    not (CT_RATE_ROOT / "valid_fixed").exists(), reason="CT-RATE dataset not mounted"
)
def test_load_records_valid_split() -> None:
    records = load_records("valid")
    assert len(records) > 1000, f"Expected ~3038 valid records, got {len(records)}"
    r = records[0]
    assert isinstance(r, CTRateRecord)
    assert r.volume_name.endswith(".nii.gz")
    assert r.nifti_path.is_file(), f"First valid sample NIfTI missing: {r.nifti_path}"
    assert r.findings  # non-empty
    assert r.impression
    assert r.spacing_xy is not None
    assert r.spacing_z is not None
    assert r.labels is not None and len(r.labels) == 18


@pytest.mark.skipif(
    not (CT_RATE_ROOT / "valid_fixed").exists(), reason="CT-RATE dataset not mounted"
)
def test_datamodule_metadata_mode_yields_batches() -> None:
    dm = CTRateDataModule(mode="metadata", num_workers=0, batch_size=4)
    dm.setup()
    batch = next(iter(dm.val_dataloader()))
    assert len(batch) == 4
    assert all(isinstance(b, CTRateRecord) for b in batch)
