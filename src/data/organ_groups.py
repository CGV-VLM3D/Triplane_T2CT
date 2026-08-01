"""CT-RATE ts_seg 117-label → 4 흉부 organ-group 병합 (ctgen mask conditioning).

TotalSegmentator v2 117 라벨(정의: `third_party/fvlm/data/resize.py:21-139`, 우리 on-disk ts_seg와
동일 스킴)을 **4개 흉부 organ**으로 축소한다. KEEP 대상만 group_id 1-4로 매핑하고 나머지 113개는 전부
배경(0). 근거는 7축 교차검증(플랜 `radiant-popping-fox.md`): 리포트 언급 빈도·질병 분포·FOV·mask↔질병 AUC·
fVLM released 선례 모두 lung/heart/aorta/esophagus로 수렴 (trachea·복부 드롭 확정).

병합 이유: (1) 117 fine 라벨은 이미지 VAE roundtrip에서 organ 복원이 붕괴(organDice≈0.36)하는데 4로 묶으면
ordinal 간격이 1/117→1/4로 넓어져 생존↑; (2) 리포트에 안 나오는 세부(개별 갈비뼈·척추·골반·사지)는 조건
가치가 없음. 마스크는 **geometry(위치/크기/모양)만** 담고 texture는 text가 담당 — 4 organ 전부 최소 1개
size-소견을 실측으로 보유(heart→cardiomegaly AUC 0.92, aorta→calcif 0.79, esophagus→hiatal 0.67,
lung→pleural effusion 0.23 역상관).
"""

from __future__ import annotations

import numpy as np

# ts_seg source label id → 병합 group id. 여기 없는 id는 전부 배경(0)으로 떨어진다.
# (fVLM released CT-RATE 4-organ과 동일: resize.py:141-151.)
TS_TO_GROUP: dict[int, int] = {
    10: 1,
    11: 1,
    12: 1,
    13: 1,
    14: 1,  # lung: 5 lobes(상/하좌, 상/중/하우)
    51: 2,
    61: 2,  # heart: heart + atrial_appendage_left
    52: 3,  # aorta
    15: 4,  # esophagus
}

# group_id → 사람이 읽는 이름 (0 = 배경).
GROUP_NAMES: dict[int, str] = {
    0: "background",
    1: "lung",
    2: "heart",
    3: "aorta",
    4: "esophagus",
}

# 배경 제외 organ 그룹 수. MaskConditioner의 num_classes = NUM_GROUPS + 1 = 5 (배경 포함 카운트).
NUM_GROUPS: int = 4


def apply_grouping(
    mask: np.ndarray, mapping: dict[int, int] = TS_TO_GROUP
) -> np.ndarray:
    """117-label ts_seg 마스크를 4-organ group 마스크로 병합.

    fVLM `resize.py:156-161`의 fused-mask 패턴과 동일. `mapping`에 없는 라벨은 배경(0).

    Args:
        mask: ts_seg 라벨 볼륨, 임의 shape, 값 ``{0..117}`` (정수 또는 정수값 float).
        mapping: source-label → group-id dict (기본 :data:`TS_TO_GROUP`; ablation용 교체 가능).

    Returns:
        `mask`와 동일 shape·dtype, 값 ``{0..4}`` (배경 0 + organ 1-4).
    """
    fused = np.zeros_like(mask)
    for src_id, group_id in mapping.items():
        fused[mask == src_id] = group_id
    return fused


__all__ = ["TS_TO_GROUP", "GROUP_NAMES", "NUM_GROUPS", "apply_grouping"]
