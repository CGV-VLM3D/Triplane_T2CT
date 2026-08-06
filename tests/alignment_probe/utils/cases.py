"""Alignment-probing 실험(Experiment 1) 공유 케이스 로더 — 모델 비종속.

train(toy_v2 5000) + valid_v2(valid_fixed 1304) 두 split을 동일한 ``AlignmentCase``로 묶는다.
한 케이스 = scan_id + findings + impression + age/sex(T5 템플릿용) + 18-label.
텍스트 probe는 report만 필요하므로 볼륨/마스크는 여기서 안 만진다(image feature z는
train 단계에서 MAISI latent로 별도 처리).

재사용: ``ABNORMALITY_LABELS``(tests/saliency_map/data.py, 18종 단일 소스).
조인 규약: toy_v2 ``ids.json``의 VolumeName으로 제한 후 reports+metadata+labels를
VolumeName(재구성 인덱스 포함)에 1:1 조인.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from tests.saliency_map.data import ABNORMALITY_LABELS

_ROOT: Final[Path] = Path("/workspace/datasets/datasets/CT-RATE/dataset")
_TOY_V2: Final[Path] = Path("/workspace/data/ctrate_toy_v2")
_SUFFIX: Final[str] = ".nii.gz"

# split → (reports_csv, metadata_csv, labels_csv, toy_v2 ids.json)
_SPLIT_FILES: Final[dict[str, tuple[Path, Path, Path, Path]]] = {
    "train": (
        _ROOT / "radiology_text_reports" / "train_reports.csv",
        _ROOT / "metadata" / "train_metadata.csv",
        _ROOT / "multi_abnormality_labels" / "train_predicted_labels.csv",
        _TOY_V2 / "train" / "ids.json",
    ),
    "valid_v2": (
        _ROOT / "radiology_text_reports" / "validation_reports.csv",
        _ROOT / "metadata" / "validation_metadata.csv",
        _ROOT / "multi_abnormality_labels" / "valid_predicted_labels.csv",
        _TOY_V2 / "valid_v2" / "ids.json",
    ),
}


@dataclass(frozen=True)
class AlignmentCase:
    """텍스트 probe 1건(텐서 아님). VolumeName 단위(재구성 인덱스 포함)."""

    scan_id: str  # "train_10000_a_1" / "valid_1000_a_1"
    findings: str
    impression: str
    age: str  # CT-RATE raw "036Y" (T5 템플릿이 [:-1].zfill(3)로 파싱)
    sex: str  # "M" / "F"
    labels: dict[str, int]  # {abnormality: 0/1}, 18개


def _clean(value) -> str:
    """metadata 셀 → str, NaN/None → ''."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def load_cases(split: str) -> list[AlignmentCase]:
    """split('train'|'valid_v2')의 ``AlignmentCase`` 목록.

    toy_v2 ``ids.json``의 VolumeName 순서를 보존하여 reports+metadata+labels를 조인한다.
    조인 키가 어느 CSV에든 없으면 KeyError로 즉시 실패(experiment는 fail-fast).
    """
    if split not in _SPLIT_FILES:
        raise ValueError(f"split must be 'train'|'valid_v2', got {split!r}")
    reports_csv, meta_csv, labels_csv, ids_json = _SPLIT_FILES[split]

    frozen_ids = json.loads(ids_json.read_text())["ids"]  # ['train_10000_a_1', ...]
    want = [v if v.endswith(_SUFFIX) else v + _SUFFIX for v in frozen_ids]

    reports = pd.read_csv(reports_csv).set_index("VolumeName")
    meta = pd.read_csv(meta_csv).set_index("VolumeName")
    labels = pd.read_csv(labels_csv).set_index("VolumeName")

    cases: list[AlignmentCase] = []
    for vn in want:
        rep, m, lab = reports.loc[vn], meta.loc[vn], labels.loc[vn]
        cases.append(
            AlignmentCase(
                scan_id=vn[: -len(_SUFFIX)],
                findings=str(rep.get("Findings_EN", "") or ""),
                impression=str(rep.get("Impressions_EN", "") or ""),
                age=_clean(m.get("PatientAge")),
                sex=_clean(m.get("PatientSex")),
                labels={a: int(lab[a]) for a in ABNORMALITY_LABELS},
            )
        )
    return cases
