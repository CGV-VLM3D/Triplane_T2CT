"""마스크 VAE 데이터셋 — ts_seg 마스크 경로 리스트 → DataLoader.

비지도(text·이미지 불필요): 마스크 경로만 모아 [preprocess.build_mask_transform](preprocess.py)을 적용한다.
train = `train_fixed`, val = `valid_fixed` (계획 확정).
"""

from __future__ import annotations

import glob
from pathlib import Path

from monai.data import CacheDataset, DataLoader, Dataset

from tests.mask_vae.preprocess import PATCH_SIZE, build_mask_transform

_CTRATE_ROOT: Path = Path("/workspace/datasets/datasets/CT-RATE/dataset")
TS_TOTAL_DIR: Path = _CTRATE_ROOT / "ts_seg" / "ts_total"

# 사용자가 본 데이터셋에서 격리한 스캔들 — VAE 학습에서 제외.
# (ts_seg 트리는 정리 안 돼 있어 이 스캔들의 마스크가 아직 남아있음.)
EXCLUDE_DIRS: tuple[Path, ...] = (
    _CTRATE_ROOT / "error_ctrate_data",
    _CTRATE_ROOT / "no_chest_data",
)

_excluded_ids_cache: set[str] | None = None


def excluded_scan_ids() -> set[str]:
    """no_chest·error 디렉토리에서 제외 대상 스캔 id(파일 basename, `.nii.gz` 제거) 집합을 만든다."""
    global _excluded_ids_cache
    if _excluded_ids_cache is None:
        ids: set[str] = set()
        for d in EXCLUDE_DIRS:
            for p in glob.glob(str(d / "*" / "*" / "*" / "*.nii.gz")):
                ids.add(Path(p).name[: -len(".nii.gz")])
        _excluded_ids_cache = ids
    return _excluded_ids_cache


def list_mask_paths(split: str) -> list[str]:
    """split의 ts_seg 마스크 `.nii.gz` 경로를 정렬해 반환.

    Args:
        split: ``"train_fixed"`` 또는 ``"valid_fixed"``.

    Returns:
        정렬된 절대경로 리스트 (스캔당 1개; `<split>/<patient>/<scan>/<file>.nii.gz`).
    """
    if split not in ("train_fixed", "valid_fixed"):
        raise ValueError(f"split must be train_fixed|valid_fixed, got {split}")
    paths = sorted(glob.glob(str(TS_TOTAL_DIR / split / "*" / "*" / "*.nii.gz")))
    if not paths:
        raise FileNotFoundError(f"no ts_seg masks under {TS_TOTAL_DIR / split}")
    excl = excluded_scan_ids()  # no_chest + error 스캔 제외
    paths = [p for p in paths if Path(p).name[: -len(".nii.gz")] not in excl]
    return paths


def build_dataloader(
    split: str,
    is_train: bool,
    batch_size: int = 1,
    num_workers: int = 4,
    limit: int | None = None,
    patch_size: tuple[int, int, int] = PATCH_SIZE,
    pin_memory: bool = True,
    cache_rate: float = 0.0,
) -> DataLoader:
    """마스크 DataLoader 생성.

    Args:
        split: ``"train_fixed"`` | ``"valid_fixed"``.
        is_train: True면 64³ 랜덤 패치+flip+shuffle; False면 전체 `(480,480,256)` 볼륨.
        batch_size: 배치 크기 (전체 볼륨 val은 1 권장).
        num_workers: DataLoader 워커 수.
        limit: 앞에서 N개만 사용 (overfit 스모크/빠른 테스트용).
        patch_size: 학습 패치 크기.
        pin_memory: 호스트 메모리 pin.

    Returns:
        ``DataLoader`` — 배치 ``batch["mask"]`` 는
        학습 ``(B,1,64,64,64)`` / precompute ``(B,1,480,480,256)``, dtype long, 값 ``{0..117}``.
    """
    paths = list_mask_paths(split)
    if limit is not None:
        paths = paths[:limit]
    data = [{"mask": p} for p in paths]
    transform = build_mask_transform(is_train=is_train, patch_size=patch_size)
    # cache_rate>0면 CacheDataset이 랜덤 크롭 이전(Resize까지)을 캐싱 → 스텝당 크롭만 재실행.
    if cache_rate > 0:
        ds = CacheDataset(
            data=data,
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
    else:
        ds = Dataset(data=data, transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=is_train,
        pin_memory=pin_memory,
    )


__all__ = ["list_mask_paths", "build_dataloader", "TS_TOTAL_DIR"]
