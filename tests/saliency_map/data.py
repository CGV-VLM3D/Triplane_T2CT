"""Saliency 실험 공유 케이스 로딩 front (모델 비종속).

scan_id 하나를 saliency runner가 필요로 하는 입력으로 묶는다 — 경로 / 메타 / 라벨만.
실제 볼륨·마스크 로딩과 전처리는 모델별 runner가 담당한다 (plan: 전처리는 runner에 번들).

재사용: load_eval_cases / _volume_name_to_nifti (ct_rate_cases), ts_mask_path (fvlm_preprocess).
"""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from src.baselines.fvlm_preprocess import ts_mask_path
from src.eval.ct_rate_cases import _volume_name_to_nifti, load_eval_cases

_ROOT: Final[Path] = Path("/workspace/datasets/datasets/CT-RATE/dataset")
_LABELS_CSV: Final[Path] = (
    _ROOT / "multi_abnormality_labels" / "valid_predicted_labels.csv" # valid 예측 라
)
_METADATA_CSV: Final[Path] = _ROOT / "metadata" / "validation_metadata.csv"

# 18종 abnormality — third_party/ct_clip/scripts/zero_shot.py 의 `pathologies` 와 동일 순서
# (labels CSV의 VolumeName 제외 컬럼과도 일치). 명시해서 upstream 과 1:1 대조 가능하게.
ABNORMALITY_LABELS: Final[list[str]] = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]


@dataclass
class SaliencyCase:
    """saliency 실행 1건의 입력 (텐서 아님)."""

    scan_id: str  # "valid_1_a_1"
    ct_path: Path  # 원시 CT NIfTI (HU, (X,Y,Z))
    ts_mask_path: Path  # TS 멀티라벨 마스크 (int, (X,Y,Z))
    labels: dict[str, int]  # {abnormality: 0/1} (18개)
    findings: str
    impression: str
    meta: dict  # CT-CLIP 전처리용: slope/intercept/xy_spacing/z_spacing


def select_cases(
    n: int | None = None,
    *,
    seed: int = 42,
    positive_for: list[str] | None = None,
    require_mask: bool = True,
) -> list[SaliencyCase]:
    """proxy_test(1304, valid_fixed 1-per-patient)에서 SaliencyCase 목록 구성.

    positive_for 지정 시 해당 abnormality 라벨이 모두 1인 케이스만(AND) 유지.
    n 캡은 필터 *후* seeded-shuffle+head (모델 간 동일 부분집합 → 공정 비교).
    """
    labels = pd.read_csv(_LABELS_CSV).set_index(
        "VolumeName"
    )  # 인덱스=VolumeName, 18 컬럼
    meta = pd.read_csv(_METADATA_CSV).set_index("VolumeName")
    if positive_for and (bad := set(positive_for) - set(ABNORMALITY_LABELS)):
        raise ValueError(f"unknown abnormality(s): {sorted(bad)}")

    cases: list[SaliencyCase] = []
    for ec in load_eval_cases(n_samples=None):
        vn = f"{ec.scan_id}.nii.gz"
        row = labels.loc[vn]
        lab = {a: int(row[a]) for a in ABNORMALITY_LABELS}  # {18 abnormality: 0/1}
        if positive_for and not all(lab[a] == 1 for a in positive_for):
            continue
        ct = _volume_name_to_nifti(vn)
        try:
            mask = ts_mask_path(ct)  # 파일 없으면 FileNotFoundError
        except FileNotFoundError:
            if require_mask:
                continue
            mask = ct
        m = meta.loc[vn]
        xy = ast.literal_eval(str(m["XYSpacing"]))  # '[x_mm, y_mm]' 문자열 → [x, y]
        cases.append(
            SaliencyCase(
                scan_id=ec.scan_id,
                ct_path=ct,
                ts_mask_path=mask,
                labels=lab,
                findings=ec.findings,
                impression=ec.impression,
                meta={
                    "slope": float(m["RescaleSlope"]),
                    "intercept": float(m["RescaleIntercept"]),
                    "xy_spacing": float(
                        xy[0]
                    ),  # x/y 동일 가정 (data_inference_nii.py 와 일치)
                    "z_spacing": float(m["ZSpacing"]),
                },
            )
        )

    if n is not None and n < len(cases): # n개만 평가에 사용하고 싶을 때
        rng = random.Random(seed)
        rng.shuffle(cases)
        cases = cases[:n]
    return cases
