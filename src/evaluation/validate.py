"""Validation loop for the triplane autoencoder.

Primary metric: round-trip latent fidelity.
Optional image-domain metric: MAISI_decode(mu_hat) vs cached ct_recon,
  which isolates the triplane-AE error from the MAISI VAE's own error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from monai.inferers import SlidingWindowInferer
from monai.metrics import PSNRMetric, SSIMMetric

from src.metrics import latent_cosine_similarity, latent_l1, latent_mse, latent_psnr

UPPER_BOUND_JSON = Path("/workspace/results/upper_bound.json")

LATENT_ROI = (20, 20, 20)
SW_OVERLAP = 0.4
SW_BATCH = 16


def _latent_metrics(mu_hat: torch.Tensor, mu: torch.Tensor) -> dict[str, float]:
    """Per-batch latent-space metrics; returns scalars."""
    l1 = float(latent_l1(mu_hat, mu).mean())
    l2 = float(latent_mse(mu_hat, mu).mean().sqrt())
    data_range = float(mu.max() - mu.min()) or 1.0
    psnr = float(latent_psnr(mu_hat, mu, data_range=data_range).mean())
    return {
        "latent_l1": l1,
        "latent_l2": l2,
        "latent_psnr": psnr,
    }


def _build_inferer(device: torch.device) -> SlidingWindowInferer:
    return SlidingWindowInferer(
        roi_size=LATENT_ROI,
        sw_batch_size=SW_BATCH,
        mode="gaussian",
        overlap=SW_OVERLAP,
        device=device,
        sw_device=device,
        progress=False,
    )


def run_validation(
    model,
    maisi_decoder,
    val_loader,
    n_samples: int | None = 200,
    device: str = "cuda",
    compute_image_metrics: bool = True,
) -> dict[str, Any]:
    """Run validation and return aggregate metric dict.

    Parameters
    ----------
    model:
        TriplaneAE or IdentityAE.  Must implement forward(mu) -> (mu_hat, ...).
    maisi_decoder:
        Frozen callable net.decode_stage_2_outputs, or None.
        Required when compute_image_metrics=True.
    val_loader:
        Yields dict with key "mu" ([B,4,120,120,64]).
        When compute_image_metrics=True, must also yield "ct_recon" ([B,1,480,480,256]).
    n_samples:
        Stop after this many individual volumes.  None means all.
    device:
        Torch device string.
    compute_image_metrics:
        Whether to decode mu_hat through MAISI and compare against cached ct_recon.
    """
    if compute_image_metrics and maisi_decoder is None:
        raise ValueError(
            "compute_image_metrics=True requires maisi_decoder to be provided. "
            "Pass net.decode_stage_2_outputs from a loaded MAISI autoencoder, or "
            "set compute_image_metrics=False."
        )

    dev = torch.device(device)
    model.eval()

    psnr_metric_img = PSNRMetric(max_val=1.0, reduction="none")
    ssim_metric_img = SSIMMetric(
        spatial_dims=3, data_range=1.0, win_size=11, reduction="none"
    )
    inferer = _build_inferer(dev) if compute_image_metrics else None

    accum: dict[str, list[float]] = {
        "latent_l1": [],
        "latent_l2": [],
        "latent_psnr": [],
    }
    if compute_image_metrics:
        accum["image_psnr_3d"] = []
        accum["image_ssim_3d"] = []
        accum["image_psnr_3d_vs_gt"] = []
        accum["image_ssim_3d_vs_gt"] = []

    n_seen = 0

    with torch.no_grad():
        for batch in val_loader:
            if n_samples is not None and n_seen >= n_samples:
                break

            mu = batch["mu"].to(dev)  # [B, 4, 120, 120, 64]
            B = mu.shape[0]

            mu_hat, _ = model(mu)

            lm = _latent_metrics(mu_hat.detach(), mu.detach())
            for k, v in lm.items():
                accum[k].append(v)

            if compute_image_metrics:
                ct_recon = batch["ct_recon"].to(dev)  # [B, 1, 480, 480, 256]

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    our_recon = inferer(mu_hat.float(), maisi_decoder).clamp(0.0, 1.0)

                our_recon_f32 = our_recon.float()

                psnr_vals = psnr_metric_img(our_recon_f32, ct_recon)
                ssim_vals = ssim_metric_img(our_recon_f32, ct_recon)
                accum["image_psnr_3d"].append(float(psnr_vals.mean().item()))
                accum["image_ssim_3d"].append(float(ssim_vals.mean().item()))

                gt = batch.get("gt")
                if gt is not None and not (
                    isinstance(gt, torch.Tensor) and gt.numel() == 0
                ):
                    gt = gt.to(dev)
                    psnr_gt = psnr_metric_img(our_recon_f32, gt.float())
                    ssim_gt = ssim_metric_img(our_recon_f32, gt.float())
                    accum["image_psnr_3d_vs_gt"].append(float(psnr_gt.mean().item()))
                    accum["image_ssim_3d_vs_gt"].append(float(ssim_gt.mean().item()))

                del ct_recon, our_recon, our_recon_f32

            del mu, mu_hat
            torch.cuda.empty_cache()
            n_seen += B

    import numpy as np

    result: dict[str, Any] = {}
    for k, vals in accum.items():
        if vals:
            arr = np.array(vals, dtype=np.float64)
            result[k] = float(arr.mean())

    # Gap to upper bound
    gap_psnr = None
    gap_ssim = None
    if compute_image_metrics and UPPER_BOUND_JSON.exists():
        with open(UPPER_BOUND_JSON) as f:
            ub = json.load(f)
        ub_psnr = ub["psnr"]["mean"]
        ub_ssim = ub["ssim"]["mean"]
        if "image_psnr_3d" in result:
            gap_psnr = ub_psnr - result["image_psnr_3d"]
        if "image_ssim_3d" in result:
            gap_ssim = ub_ssim - result["image_ssim_3d"]

    result["gap_to_upper_psnr"] = gap_psnr
    result["gap_to_upper_ssim_3d"] = gap_ssim
    result["n_samples_evaluated"] = n_seen

    return result
