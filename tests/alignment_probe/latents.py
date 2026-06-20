"""MAISI VAE latent 로더 — Unit 2 z② 분류기 / Probe B target 공용.

train·valid_v2 모두 Report2CT가 frozen MAISI로 인코딩한 `_emb.nii.gz` ((H,W,D,4))를 쓴다
→ 동일 파이프라인·동일 스케일(std≈0.95). (`ctrate_toy_v2/train/.../mu.pt`는 raw VAE mu라
스케일이 달라(std 0.67) 쓰지 않는다.)
  train      : data/report2ct_work_dir/image_embeddings/<scan>_emb.nii.gz
  valid_v2 : data/ctrate_toy_v2/valid_v2/latents/<scan>_emb.nii.gz
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

_TRAIN_EMB = Path("/workspace/data/report2ct_work_dir/image_embeddings")
_VALID_V2_EMB = Path("/workspace/data/ctrate_toy_v2/valid_v2/latents")


def latent_path(scan_id: str, split: str) -> Path:
    root = _TRAIN_EMB if split == "train" else _VALID_V2_EMB
    return root / f"{scan_id}_emb.nii.gz"


def has_latent(scan_id: str, split: str) -> bool:
    return latent_path(scan_id, split).is_file()


def load_latent(scan_id: str, split: str) -> torch.Tensor:
    """scan의 MAISI VAE latent → ``(4, 120, 120, 64)`` float32."""
    arr = nib.load(str(latent_path(scan_id, split))).get_fdata()  # (H, W, D, C=4)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(3, 0, 1, 2))).float()
