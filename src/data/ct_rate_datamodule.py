"""CT-RATE LightningDataModule (scaffold).

Day 2 (5/27) status: metadata + reports + labels join is wired and unit-tested.
NIfTI volume loading is gated behind `mode='full'` and exercised by EDA on Day 4
and by Phase C training. Heavy transforms (HU clip, resize) are configurable so
the same DataModule serves EDA (light: metadata only) and training (full: volumes).

CT-RATE layout (collaborator read-only):
    /workspace/datasets/datasets/CT-RATE/dataset/
      ├── train_fixed/<patient>/<study>/<volume>.nii.gz   (20,000 patients, 47,148 scans)
      ├── valid_fixed/<patient>/<study>/<volume>.nii.gz   (1,304 patients, 3,038 scans)
      ├── metadata/{train,validation}_metadata.csv        (spacing, manufacturer, kernel, ...)
      ├── multi_abnormality_labels/{train,valid}_predicted_labels.csv  (18 binary labels)
      └── radiology_text_reports/{train,validation}_reports.csv  (Findings_EN, Impressions_EN)

The VolumeName column is the canonical join key everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset


CT_RATE_ROOT = Path("/workspace/datasets/datasets/CT-RATE/dataset")


@dataclass(frozen=True)
class CTRateRecord:
    """Per-sample metadata joined from reports + spacing + labels CSVs."""

    volume_name: str  # e.g. "valid_1_a_1.nii.gz"
    nifti_path: Path
    findings: str
    impression: str
    # Spacing (mm): may be None if metadata missing for this sample.
    spacing_xy: float | None
    spacing_z: float | None
    # 18-dim binary label vector (Cardiomegaly, Pleural effusion, ...). dict-of-int or None.
    labels: dict[str, int] | None


def _parse_spacing(raw) -> float | None:
    """CT-RATE metadata stores spacings as either floats or stringified lists like '[0.34, 0.34]'.
    Return the first scalar component as a float, or None if the cell is missing.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().strip("[]")
    if not s:
        return None
    head = s.split(",")[0].strip()
    try:
        return float(head)
    except ValueError:
        return None


def _resolve_nifti_path(volume_name: str, split_dir: Path) -> Path:
    """volume_name = '<split>_<patient>_<study>_<idx>.nii.gz' → split_dir/<patient_id>/<study_id>/volume."""
    # 'valid_1_a_1.nii.gz' -> patient='valid_1', study='valid_1_a'
    parts = volume_name.replace(".nii.gz", "").split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected VolumeName format: {volume_name}")
    patient = "_".join(parts[:2])  # "valid_1"
    study = "_".join(parts[:3])  # "valid_1_a"
    return split_dir / patient / study / volume_name


def load_records(
    split: Literal["train", "valid"], root: Path = CT_RATE_ROOT
) -> list[CTRateRecord]:
    """Join reports + metadata + labels for a split into a flat record list (metadata only, no NIfTI load)."""
    if split == "train":
        reports_csv = root / "radiology_text_reports" / "train_reports.csv"
        metadata_csv = root / "metadata" / "train_metadata.csv"
        labels_csv = root / "multi_abnormality_labels" / "train_predicted_labels.csv"
        split_dir = root / "train_fixed"
    else:
        reports_csv = root / "radiology_text_reports" / "validation_reports.csv"
        metadata_csv = root / "metadata" / "validation_metadata.csv"
        labels_csv = root / "multi_abnormality_labels" / "valid_predicted_labels.csv"
        split_dir = root / "valid_fixed"

    reports = pd.read_csv(reports_csv)[["VolumeName", "Findings_EN", "Impressions_EN"]]
    metadata = pd.read_csv(metadata_csv)
    labels = pd.read_csv(labels_csv)

    # Spacing column names vary across CT-RATE metadata releases — try the most common ones.
    sx_col = next(
        (
            c
            for c in metadata.columns
            if c in ("XYSpacing", "PixelSpacingX", "XSpacing", "PixelSpacing_X")
        ),
        None,
    )
    sz_col = next(
        (
            c
            for c in metadata.columns
            if c in ("ZSpacing", "SliceThickness", "PixelSpacingZ", "PixelSpacing_Z")
        ),
        None,
    )

    label_cols = [c for c in labels.columns if c != "VolumeName"]

    df = reports.merge(metadata, on="VolumeName", how="left").merge(
        labels, on="VolumeName", how="left"
    )

    records: list[CTRateRecord] = []
    for _, row in df.iterrows():
        d = row.to_dict()
        records.append(
            CTRateRecord(
                volume_name=d["VolumeName"],
                nifti_path=_resolve_nifti_path(d["VolumeName"], split_dir),
                findings=d.get("Findings_EN") or "",
                impression=d.get("Impressions_EN") or "",
                spacing_xy=_parse_spacing(d.get(sx_col)) if sx_col else None,
                spacing_z=_parse_spacing(d.get(sz_col)) if sz_col else None,
                labels={k: int(d[k]) for k in label_cols if pd.notna(d.get(k))}
                if label_cols
                else None,
            )
        )
    return records


class CTRateMetadataDataset(Dataset):
    """Lightweight dataset returning only per-sample metadata (no NIfTI load)."""

    def __init__(self, records: list[CTRateRecord]) -> None:
        """Wraps a pre-joined list of CTRateRecord dataclasses.

        Args:
            records: Flat list of records from `load_records()`.
        """
        self.records = records

    def __len__(self) -> int:
        """Returns the number of records in the dataset."""
        return len(self.records)

    def __getitem__(self, idx: int) -> CTRateRecord:
        """Returns the CTRateRecord at position `idx` (no NIfTI load; metadata only).

        Args:
            idx: Integer index into `self.records`.

        Returns:
            A `CTRateRecord` dataclass with volume_name, nifti_path, findings,
            impression, spacing_xy/z, and a dict of 18 binary abnormality labels
            (or None when labels are unavailable). No tensor — metadata mode only.
        """
        return self.records[idx]


class CTRateDataModule(LightningDataModule):
    """LightningDataModule for CT-RATE.

    `mode='metadata'` (default for EDA / Phase B B.3 retrieval diagnostic): returns CTRateRecord per sample.
    `mode='volume'`: also loads NIfTI volume with HU clip / scale (Phase C training; TODO).
    """

    def __init__(
        self,
        root: Path | str = CT_RATE_ROOT,
        mode: Literal["metadata", "volume"] = "metadata",
        batch_size: int = 1,
        num_workers: int = 8,
        pin_memory: bool = True,
    ) -> None:
        """Configures the CT-RATE data module.

        Args:
            root: Dataset root; defaults to the collaborator read-only mount at
                ``/workspace/datasets/datasets/CT-RATE/dataset``.
            mode: ``'metadata'`` returns ``CTRateRecord`` dataclasses (EDA / retrieval);
                ``'volume'`` (not yet implemented) will also load NIfTI volumes.
            batch_size: Samples per batch passed to ``DataLoader``.
            num_workers: Parallel workers for ``DataLoader``.
            pin_memory: Whether to pin host memory for faster GPU transfer.
        """
        super().__init__()
        self.save_hyperparameters()
        self.root = Path(root)
        self.mode = mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self._train_records: list[CTRateRecord] | None = None
        self._valid_records: list[CTRateRecord] | None = None

    def setup(self, stage: str | None = None) -> None:
        """Loads and joins train/valid CT-RATE records (idempotent; CSV-only, no NIfTI I/O)."""
        if self._train_records is None:
            self._train_records = load_records("train", self.root)
        if self._valid_records is None:
            self._valid_records = load_records("valid", self.root)

    def _make_loader(self, records: list[CTRateRecord], shuffle: bool) -> DataLoader:
        """Builds a ``DataLoader`` over ``CTRateMetadataDataset`` for the given records.

        Uses a pass-through ``collate_fn`` so each batch is a plain Python list of
        ``CTRateRecord`` dataclasses (no tensor collation in metadata mode).

        Args:
            records: Pre-joined record list for one split.
            shuffle: Whether to shuffle the dataset order each epoch.

        Returns:
            A ``DataLoader`` yielding lists of ``CTRateRecord`` objects.
            ``mode='volume'`` raises ``NotImplementedError``.
        """
        if self.mode != "metadata":
            raise NotImplementedError(
                "mode='volume' loader lands in Day 4 EDA or Phase C; use mode='metadata' for now."
            )
        return DataLoader(
            CTRateMetadataDataset(records),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=lambda batch: (
                batch
            ),  # keep dataclass list as-is for metadata mode
        )

    def train_dataloader(self) -> DataLoader:
        """Returns a shuffled ``DataLoader`` over the training split."""
        assert self._train_records is not None, "Call setup() first"
        return self._make_loader(self._train_records, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Returns an unshuffled ``DataLoader`` over the validation split."""
        assert self._valid_records is not None, "Call setup() first"
        return self._make_loader(self._valid_records, shuffle=False)
