"""MAISI VAE latent 로더 — LatentClassifier 학습 / Probe B target 공용.

train·valid_v2 모두 Report2CT가 frozen MAISI로 인코딩한 `_emb.nii.gz` ((H,W,D,4), std≈0.95)를 쓴다.
두 경로 모두 toy_v2 아래 동일 포맷 심링크(train은 → report2ct_work_dir/image_embeddings):
  train    : data/ctrate_toy_v2/train/latents/<scan>_emb.nii.gz
  valid_v2 : data/ctrate_toy_v2/valid_v2/latents/<scan>_emb.nii.gz
주의: raw VAE mu `mu.pt`(std≈0.67)는 _emb과 스케일이 달라 latent으로 쓰지 말 것 —
현재 `datasets/datasets/latents/train/<scan>/mu.pt`에만 존재(toy_v2 심링크는 _emb.nii.gz로 교체됨).
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

_TRAIN_EMB = Path("/workspace/data/ctrate_toy_v2/train/latents")
_VALID_V2_EMB = Path("/workspace/data/ctrate_toy_v2/valid_v2/latents")


def latent_path(scan_id: str, split: str) -> Path:
    """scan의 ``_emb.nii.gz`` 절대 경로를 반환한다.

    Args:
        scan_id: 재구성 인덱스 포함 scan ID (예: ``'train_10000_a_1'``).
        split: ``'train'`` 또는 ``'valid_v2'``.

    Returns:
        ``data/ctrate_toy_v2/<split>/latents/<scan_id>_emb.nii.gz`` 경로.
    """
    root = _TRAIN_EMB if split == "train" else _VALID_V2_EMB
    return root / f"{scan_id}_emb.nii.gz"


def has_latent(scan_id: str, split: str) -> bool:
    """해당 scan의 latent 파일이 존재하면 True를 반환한다."""
    return latent_path(scan_id, split).is_file()


def load_latent(scan_id: str, split: str) -> torch.Tensor:
    """scan의 MAISI VAE latent → ``(4, 120, 120, 64)`` float32."""
    arr = nib.load(str(latent_path(scan_id, split))).get_fdata()  # (H, W, D, C=4)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(3, 0, 1, 2))).float()
