"""ts_seg 마스크 전처리 — 마스크 VAE(raw 1채널 입력)용.

[src/baselines/report2ct_image_encoder.py](../../src/baselines/report2ct_image_encoder.py)의 공간 파이프라인을
**라벨용으로** 변형한다. seg latent이 이미지 latent와 voxel 단위로 정렬되도록 **동일한 고정 격자
`(480,480,256)`**로 Resize하되, 라벨이므로:
  - ScaleIntensityRange 제거 (라벨은 intensity가 아님)
  - Resize/crop 보간은 **nearest** (라벨은 절대 보간 금지)
학습은 MAISI VAE 레시피대로 64³ 랜덤 패치, precompute는 전체 `(480,480,256)` 격자를 그대로 인코딩한다.

정렬 근거: 이미지 latent는 CT → `Resized((480,480,256), trilinear)` → MAISI encode로 만들어졌다
([report2ct_image_encoder.py:62-64](../../src/baselines/report2ct_image_encoder.py#L62)). 같은 스캔의 마스크를
동일 격자로 Resize(nearest)하면 voxel이 1:1 대응한다.
"""

from __future__ import annotations

import monai
import torch
from monai.transforms import Compose

# 이미지 인코더와 반드시 동일한 격자 — 재정의하지 않고 재사용해 정렬을 강제한다.
from src.baselines.report2ct_image_encoder import OUTPUT_SPATIAL_SIZE

NUM_CLASSES: int = 118  # TS v2 라벨 0..117 (배경 0 포함)
PATCH_SIZE: tuple[int, int, int] = (
    64,
    64,
    64,
)  # config_maisi_vae_train.json:autoencoder_train.patch_size


def build_mask_transform(
    is_train: bool, patch_size: tuple[int, int, int] = PATCH_SIZE
) -> Compose:
    """ts_seg `.nii.gz` 경로 → canonical 라벨 볼륨/패치 (정수, channel-first) 변환기.

    딕셔너리 transform이며 키는 ``"mask"``.

    Args:
        is_train: True면 64³ 랜덤 패치 + flip 증강; False면 전체 격자(precompute용).
        patch_size: 학습 시 크롭할 패치 크기 ``(64, 64, 64)``.

    Returns:
        MONAI ``Compose`` — 출력 ``batch["mask"]`` 는
        학습 ``(1, 64, 64, 64)`` / precompute ``(1, 480, 480, 256)``, dtype long, 값 ``{0..117}``.
    """
    tfms: list = [
        monai.transforms.LoadImaged(keys="mask"),
        monai.transforms.EnsureChannelFirstd(keys="mask"),  # (1, X, Y, Z)
        monai.transforms.Orientationd(keys="mask", axcodes="RAS"),
        monai.transforms.Resized(
            keys="mask", spatial_size=OUTPUT_SPATIAL_SIZE, mode="nearest-exact"
        ),  # (1, 480, 480, 256) — 이미지 격자와 정렬, 라벨 보존
        # 캐시를 uint8로 저장(라벨 0~117, 무손실) → float32 대비 RAM/disk 4× 절감.
        # 랜덤 크롭 이전이라 CacheDataset이 이 uint8 볼륨을 캐싱.
        monai.transforms.EnsureTyped(keys="mask", dtype=torch.uint8),
    ]
    if is_train:
        # 공간 증강(라벨 안전): 좌우/전후/상하 flip. NV random_aug의 경량 부분만 차용.
        tfms += [
            monai.transforms.RandFlipd(keys="mask", prob=0.5, spatial_axis=axis)
            for axis in range(3)
        ]
        tfms += [
            monai.transforms.SpatialPadd(keys="mask", spatial_size=patch_size),
            monai.transforms.RandSpatialCropd(
                keys="mask",
                roi_size=patch_size,
                random_size=False,
                random_center=True,
            ),  # (1, 64, 64, 64)
        ]
    tfms.append(monai.transforms.EnsureTyped(keys="mask", dtype=torch.long))
    return Compose(tfms)


def to_vae_input(label: torch.Tensor) -> torch.Tensor:
    """정수 라벨 볼륨/패치 → raw pseudo-intensity VAE 입력 ``[0,1]``.

    라벨 id를 pseudo-intensity로 넣는 raw 방식(계획 확정). MAISI VAE의 [0,1] 규약에 맞춤.

    Args:
        label: ``(B, 1, ...)`` 또는 ``(1, ...)`` 정수 라벨 ``{0..117}``.

    Returns:
        같은 shape의 float 텐서, ``label / 117``.
    """
    return label.float() / float(NUM_CLASSES - 1)  # /117 → [0,1]


__all__ = ["build_mask_transform", "to_vae_input", "NUM_CLASSES", "PATCH_SIZE"]
