from __future__ import annotations

import torch
from monai.metrics import PSNRMetric, SSIMMetric

# Module-level lazy cache: avoids re-loading VGG weights on every call.
_LPIPS: object | None = None


def _get_lpips(device: torch.device):
    global _LPIPS
    if _LPIPS is None:
        import lpips

        _LPIPS = lpips.LPIPS(net="vgg").eval()
    return _LPIPS.to(device)


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


def _random_slice_indices(size: int, n: int, generator: torch.Generator) -> list[int]:
    n = min(n, size)
    return torch.randperm(size, generator=generator)[:n].tolist()


def image_ssim_2d_avg(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    n_slices: int = 10,
    seed: int = 0,
    win_size: int = 11,
) -> torch.Tensor:
    """Average 2-D SSIM over random slices along all 3 axes. Returns [B].

    Inputs expected in [0, 1] (MAISI decoder output convention).
    For HU-space inputs, pass data_range=2000.

    Inputs: [B, 1, H, W, D]. Slices along axial (D), coronal (H), sagittal (W).
    """
    B, _, H, W, D = pred.shape
    metric = SSIMMetric(spatial_dims=2, data_range=data_range, win_size=win_size)
    gen = torch.Generator()
    gen.manual_seed(seed)
    scores: list[torch.Tensor] = []

    for idx in _random_slice_indices(D, n_slices, gen):  # axial [B,1,H,W]
        scores.append(metric(pred[:, :, :, :, idx], target[:, :, :, :, idx]).squeeze(1))
    for idx in _random_slice_indices(H, n_slices, gen):  # coronal [B,1,W,D]
        scores.append(metric(pred[:, :, idx, :, :], target[:, :, idx, :, :]).squeeze(1))
    for idx in _random_slice_indices(W, n_slices, gen):  # sagittal [B,1,H,D]
        scores.append(metric(pred[:, :, :, idx, :], target[:, :, :, idx, :]).squeeze(1))

    return torch.stack(scores, dim=0).mean(dim=0)  # [B]


def image_lpips_2d_avg(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_slices: int = 10,
    seed: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Average LPIPS over random 2-D slices along all 3 axes. Returns [B].

    Inputs: [B, 1, H, W, D] in HU (clamped to [-1000, 1000] internally).
    """
    if device is None:
        device = pred.device
    net = _get_lpips(device)
    B, _, H, W, D = pred.shape

    def _to_lpips(s: torch.Tensor) -> torch.Tensor:
        """[B, 1, h, w] HU → [B, 3, h, w] in [-1, 1]."""
        x = s.to(device=device, dtype=torch.float32).clamp(-1000.0, 1000.0) / 1000.0
        return x.expand(-1, 3, -1, -1)

    gen = torch.Generator()
    gen.manual_seed(seed)
    scores: list[torch.Tensor] = []

    for idx in _random_slice_indices(D, n_slices, gen):  # axial
        with torch.no_grad():
            v = net(
                _to_lpips(pred[:, :, :, :, idx]), _to_lpips(target[:, :, :, :, idx])
            )
        scores.append(v.reshape(B))
    for idx in _random_slice_indices(H, n_slices, gen):  # coronal
        with torch.no_grad():
            v = net(
                _to_lpips(pred[:, :, idx, :, :]), _to_lpips(target[:, :, idx, :, :])
            )
        scores.append(v.reshape(B))
    for idx in _random_slice_indices(W, n_slices, gen):  # sagittal
        with torch.no_grad():
            v = net(
                _to_lpips(pred[:, :, :, idx, :]), _to_lpips(target[:, :, :, idx, :])
            )
        scores.append(v.reshape(B))

    return torch.stack(scores, dim=0).mean(dim=0)  # [B]
