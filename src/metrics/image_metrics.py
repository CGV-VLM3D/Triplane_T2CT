from __future__ import annotations

import torch
from monai.metrics import PSNRMetric, SSIMMetric


def image_psnr_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    """PSNR per sample for 3-D volumes [B,1,H,W,D]. Returns [B].

    Inputs expected in [0, 1] (MAISI decoder output convention).
    For HU-space inputs, pass data_range=2000.
    """
    metric = PSNRMetric(max_val=data_range)
    out = metric(pred, target)  # [B, 1]
    return out.squeeze(1)


def image_ssim_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    win_size: int = 11,
) -> torch.Tensor:
    """SSIM per sample for 3-D volumes [B,1,H,W,D]. Returns [B].

    Inputs expected in [0, 1] (MAISI decoder output convention).
    For HU-space inputs, pass data_range=2000.
    """
    metric = SSIMMetric(spatial_dims=3, data_range=data_range, win_size=win_size)
    out = metric(pred, target)  # [B, 1]
    return out.squeeze(1)
