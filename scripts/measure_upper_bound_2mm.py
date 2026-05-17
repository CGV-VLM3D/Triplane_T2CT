"""
Measure MAISI VAE round-trip reconstruction upper bound on the 2mm toy latents.

This is the Tier-1 ceiling — the best PSNR/SSIM any triplane AE trained on
`/workspace/data/latents_2mm/` could possibly reach, since the triplane AE
sits between MAISI's encode and MAISI's decode and can never improve the
round-trip itself. Compare candidate Tier-1 trial_toy* runs against this
number to know how far below ceiling each model is.

Mirrors scripts/measure_upper_bound.py but with:
  - latent shape  : [4, 60, 60, 32]
  - GT spatial    : (240, 240, 128)
  - decode roi    : (10, 10, 10)  (= 40^3 image roi, scaled from 80^3 at 1mm)
  - Results saved : /workspace/results/upper_bound_2mm.json

The 2mm decode is ~10x cheaper than 1mm decode, so torch.compile warmup is
NOT worth it for a 1000-volume run; we skip it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import Dataset
from monai.inferers import SlidingWindowInferer
from monai.metrics import PSNRMetric, SSIMMetric
from torch.utils.data import DataLoader
from tqdm import tqdm

WORKSPACE = Path("/workspace")
BUNDLE_DIR = WORKSPACE / "maisi_bundle"
sys.path.insert(0, str(BUNDLE_DIR))
sys.path.insert(0, str(WORKSPACE / "reference"))
sys.path.insert(0, str(WORKSPACE / "reference" / "scripts"))

from models.dataloader import build_transforms  # noqa: E402
from extract_maisi_latent import (  # noqa: E402
    add_bundle_to_syspath,
    build_autoencoder,
    find_ae_ckpt,
    load_state,
)

LATENTS_2MM_ROOT = WORKSPACE / "data" / "latents_2mm"
RESULTS_DIR = WORKSPACE / "results"

# Decode settings (latent_roi=10 ≈ image_roi=40, scaled 1/2 from 1mm setup)
LATENT_ROI = (10, 10, 10)
SW_OVERLAP = 0.4
SW_BATCH = 32

# 2mm preprocessing matches the latent extraction pipeline
SPATIAL_SIZE = (240, 240, 128)
HU_MIN, HU_MAX = -1000.0, 1000.0
OUT_MIN, OUT_MAX = 0.0, 1.0

NUM_WORKERS = 8


class LatentGTDataset(Dataset):
    """Yields (mu [4,60,60,32] fp16, gt [1,240,240,128] fp32) pairs."""

    def __init__(self, latent_dirs: list[Path]):
        gt_transform = build_transforms(
            spatial_size=SPATIAL_SIZE,
            hu_min=HU_MIN,
            hu_max=HU_MAX,
            out_min=OUT_MIN,
            out_max=OUT_MAX,
            train=False,
        )
        data = []
        for ld in latent_dirs:
            src = (ld / "src.txt").read_text().strip()
            data.append({"image": src, "mu_path": str(ld / "mu.pt")})
        super().__init__(data=data, transform=gt_transform)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        gt = item["image"]
        if hasattr(gt, "as_tensor"):
            gt = gt.as_tensor()
        mu = torch.load(item["mu_path"], weights_only=True)
        return {"mu": mu, "gt": gt.float()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["valid", "train"], default="valid")
    parser.add_argument("--n", type=int, default=-1, help="cap n volumes (-1 = all)")
    parser.add_argument(
        "--out", type=str, default=str(RESULTS_DIR / "upper_bound_2mm.json")
    )
    args = parser.parse_args()

    device = torch.device("cuda:0")
    split_dir = LATENTS_2MM_ROOT / args.split

    # ---- Build and load frozen autoencoder ----------------------------------
    add_bundle_to_syspath(BUNDLE_DIR)
    net = build_autoencoder(BUNDLE_DIR, "autoencoder_def")
    ckpt_path = find_ae_ckpt(BUNDLE_DIR, None)
    print(f"Loading weights: {ckpt_path}")
    load_state(net, ckpt_path)
    net = net.to(device).eval()

    inferer = SlidingWindowInferer(
        roi_size=LATENT_ROI,
        sw_batch_size=SW_BATCH,
        mode="gaussian",
        overlap=SW_OVERLAP,
        device=device,
        sw_device=device,
        progress=False,
    )

    latent_dirs = sorted(split_dir.iterdir())
    if args.n > 0:
        latent_dirs = latent_dirs[: args.n]
    print(f"Found {len(latent_dirs)} {args.split} 2mm latents")

    ds = LatentGTDataset(latent_dirs)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    psnr_metric = PSNRMetric(max_val=1.0, reduction="none")
    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0, reduction="none")

    per_psnr: list[float] = []
    per_ssim: list[float] = []
    per_ids: list[str] = []

    for i, batch in enumerate(tqdm(loader, desc="UB 2mm", unit="vol")):
        sample_id = latent_dirs[i].name
        mu = batch["mu"].to(device)  # [1, 4, 60, 60, 32]
        gt = batch["gt"].to(device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            recon = inferer(mu, net.decode_stage_2_outputs).clamp(OUT_MIN, OUT_MAX)
        recon_f32 = recon.float()

        per_psnr.append(float(psnr_metric(recon_f32, gt).mean().item()))
        per_ssim.append(float(ssim_metric(recon_f32, gt).mean().item()))
        per_ids.append(sample_id)

        del mu, gt, recon, recon_f32
        torch.cuda.empty_cache()

    psnr_arr = np.array(per_psnr)
    ssim_arr = np.array(per_ssim)
    result = {
        "n_samples": len(per_psnr),
        "psnr": {
            "mean": float(psnr_arr.mean()),
            "std": float(psnr_arr.std()),
            "median": float(np.median(psnr_arr)),
        },
        "ssim": {
            "mean": float(ssim_arr.mean()),
            "std": float(ssim_arr.std()),
            "median": float(np.median(ssim_arr)),
        },
        "per_sample": {"psnr": per_psnr, "ssim": per_ssim},
        "sample_ids": per_ids,
        "notes": (
            "MAISI VAE encode->decode round-trip upper bound at 2mm spacing "
            "(spatial 240x240x128, latent [4,60,60,32]). Same intensity convention "
            "as 1mm upper bound: HU clip [-1000,1000] -> [0,1]. "
            f"Decode: SlidingWindowInferer(roi={LATENT_ROI}, overlap={SW_OVERLAP}, "
            f"sw_batch={SW_BATCH}). No torch.compile."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== Upper bound 2mm ({args.split}, n={result['n_samples']}) ===")
    print(
        f"PSNR  mean {result['psnr']['mean']:.2f} ± {result['psnr']['std']:.2f}  "
        f"median {result['psnr']['median']:.2f} dB"
    )
    print(
        f"SSIM  mean {result['ssim']['mean']:.4f} ± {result['ssim']['std']:.4f}  "
        f"median {result['ssim']['median']:.4f}"
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
