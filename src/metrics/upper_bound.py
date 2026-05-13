from __future__ import annotations

import json
from pathlib import Path

import torch

from .image_metrics import (
    image_lpips_2d_avg,
    image_psnr_3d,
    image_ssim_2d_avg,
    image_ssim_3d,
)


@torch.no_grad()
def compute_upper_bound(maisi_vae, dataloader, output_path: str | Path) -> dict:
    """MAISI VAE encode->decode round-trip metrics, saved as JSON.

    Args:
        maisi_vae: MAISI autoencoder with .encode() / .decode() or
                   .decode_stage_2_outputs() API. Caller loads weights and
                   moves to the desired device before passing in.
        dataloader: yields [B,1,H,W,D] float tensors or dicts with key "image".
        output_path: where the result JSON will be written.

    Returns:
        The summary dict written to JSON.
    """
    device = next(maisi_vae.parameters()).device

    all_psnr: list[float] = []
    all_ssim3: list[float] = []
    all_ssim2: list[float] = []
    all_lpips: list[float] = []
    per_sample: list[dict] = []
    sample_id = 0

    for batch in dataloader:
        x = batch["image"] if isinstance(batch, dict) else batch
        x = x.to(device=device, dtype=torch.float32)

        # Encode: handle (mu, sigma) tuple or distribution-like object.
        enc_out = maisi_vae.encode(x)
        if isinstance(enc_out, (tuple, list)):
            mu = enc_out[0]
        elif hasattr(enc_out, "mean"):
            mu = enc_out.mean
        else:
            mu = enc_out

        # Decode: prefer decode_stage_2_outputs when available (MAISI convention).
        if hasattr(maisi_vae, "decode_stage_2_outputs"):
            x_hat = maisi_vae.decode_stage_2_outputs(mu)
        else:
            x_hat = maisi_vae.decode(mu)

        psnr_b = image_psnr_3d(x_hat, x).cpu()
        ssim3_b = image_ssim_3d(x_hat, x).cpu()
        ssim2_b = image_ssim_2d_avg(x_hat, x).cpu()
        lpips_b = image_lpips_2d_avg(x_hat, x, device=device).cpu()

        for i in range(x.shape[0]):
            per_sample.append(
                {
                    "sample_id": sample_id,
                    "image_psnr_3d": psnr_b[i].item(),
                    "image_ssim_3d": ssim3_b[i].item(),
                    "image_ssim_2d_avg": ssim2_b[i].item(),
                    "image_lpips_2d_avg": lpips_b[i].item(),
                }
            )
            sample_id += 1

        all_psnr.extend(psnr_b.tolist())
        all_ssim3.extend(ssim3_b.tolist())
        all_ssim2.extend(ssim2_b.tolist())
        all_lpips.extend(lpips_b.tolist())

    def _stats(vals: list[float]) -> dict:
        t = torch.tensor(vals)
        return {
            "mean": t.mean().item(),
            "std": t.std().item(),
            "median": t.median().item(),
        }

    result = {
        "image_psnr_3d": _stats(all_psnr),
        "image_ssim_3d": _stats(all_ssim3),
        "image_ssim_2d_avg": _stats(all_ssim2),
        "image_lpips_2d_avg": _stats(all_lpips),
        "n_samples": sample_id,
        "data": per_sample,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result
