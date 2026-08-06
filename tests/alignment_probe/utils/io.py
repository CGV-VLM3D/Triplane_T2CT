"""임베딩/라벨 IO + scan_id 정렬 — probe(A/B)·train 공용.

embeddings/<split>/<encoder>.npz 로딩, 18-label dict, img_feat(img_feat.npz) 로딩,
공통 scan_id 정렬, npz 저장 헬퍼.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.alignment_probe.utils.cases import ABNORMALITY_LABELS, load_cases

EMB_ROOT = Path("tests/alignment_probe/embeddings")


def save_npz(path: str | Path, **arrays) -> None:
    """arrays를 npz로 저장(부모 디렉토리 자동 생성).

    Args:
        path: 저장 경로(.npz).
        **arrays: 이름→배열. ``np.savez``에 그대로 전달.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def encoders() -> list[str]:
    """train과 valid_v2 양쪽에 임베딩이 있는 인코더 이름(img_feat 제외).

    Returns:
        두 split 교집합 인코더 이름 list(알파벳순 정렬).
    """
    here = lambda split: {  # noqa: E731
        p.stem for p in (EMB_ROOT / split).glob("*.npz") if p.stem != "img_feat"
    }
    return sorted(here("train") & here("valid_v2"))


def emb_dict(encoder: str, split: str) -> dict[str, np.ndarray]:
    """한 인코더·split의 임베딩을 scan_id→벡터 dict로 로딩.

    Args:
        encoder: 인코더 이름(``embeddings/<split>/<encoder>.npz``).
        split: "train" 또는 "valid_v2".

    Returns:
        {scan_id: 임베딩 벡터 ``(D,)``} dict (D는 인코더별 차원).
    """
    z = np.load(EMB_ROOT / split / f"{encoder}.npz", allow_pickle=True)
    return dict(zip(z["scan_ids"], z["emb"]))


def label_dict(split: str) -> dict[str, np.ndarray]:
    """split의 18종 이상소견 라벨을 scan_id→멀티핫 벡터 dict로 구성.

    Args:
        split: "train" 또는 "valid_v2".

    Returns:
        {scan_id: 라벨 벡터 ``(18,)`` float32} dict (ABNORMALITY_LABELS 순서).
    """
    return {
        c.scan_id: np.array([c.labels[a] for a in ABNORMALITY_LABELS], dtype=np.float32)
        for c in load_cases(split)
    }


def img_feat_dict(split: str) -> dict[str, np.ndarray] | None:
    """img_feat.npz(있으면)를 scan_id→벡터 dict로 로딩.

    Args:
        split: "train" 또는 "valid_v2".

    Returns:
        scan_id → img_feat 임베딩 ``(D_z,)`` dict; 파일 없으면 None.
    """
    p = EMB_ROOT / split / "img_feat.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=True)
    return dict(zip(z["scan_ids"], z["emb"]))


def aligned(x_dict: dict, y_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """공통 scan_id(정렬)에 대해 (X, Y) 행렬 구성.

    Args:
        x_dict: scan_id → feature vector ``(Dx,)``.
        y_dict: scan_id → target vector ``(Dy,)``.

    Returns:
        (X, Y): ``(N, Dx)``와 ``(N, Dy)``; N = 공통 scan_id 수(정렬됨).
    """
    ids = sorted(set(x_dict) & set(y_dict))
    return np.stack([x_dict[i] for i in ids]), np.stack([y_dict[i] for i in ids])
