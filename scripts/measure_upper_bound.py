"""
Measure MAISI VAE round-trip reconstruction upper bound on the CT-RATE validation set.

Pipeline per sample:
  1. Load pre-extracted latent mu (shape [4, 120, 120, 64], float16).
  2. Sliding-window decode with the frozen MAISI autoencoder, using the same
     roi/overlap as the reference recon_test.py:
       latent_roi=(20,20,20), overlap=0.4, sw_batch_size=16, GPU stitching.
     The decoder is compiled once with torch.compile for speed (~17s/vol).
  3. Clip decoder output to [0, 1].
  4. Load ground-truth CT in background DataLoader workers, applying the same
     preprocessing as the latent extraction pipeline:
       load NIfTI -> RAS orient -> clip HU [-1000,1000] -> scale [0,1]
       -> trilinear resize to (480,480,256).
  5. Compute 3D PSNR and SSIM (data_range=1.0) on the [0,1] domain.

Results saved to /workspace/results/upper_bound.json.

Optional flags:
  --cache-recon   Save each volume's ct_recon to
                  /workspace/data/maisi_latent_with_recon/<split>/<sample_id>/ct_recon.pt
                  as float16 (~58 MB per volume; 1000-sample valid split = ~58 GB total).
  --split         Which split to process (default: valid).
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

LATENTS_ROOT = WORKSPACE / "datasets" / "datasets" / "latents"
RESULTS_DIR = WORKSPACE / "results"
RECON_CACHE_ROOT = WORKSPACE / "data" / "maisi_latent_with_recon"

# Decode settings matching reference recon_test.py (image roi 80^3 -> latent roi 20^3)
LATENT_ROI = (20, 20, 20)
SW_OVERLAP = 0.4
SW_BATCH = 16  # amortise per-window overhead; fits comfortably on A6000

# Preprocessing matching the extraction pipeline
SPATIAL_SIZE = (480, 480, 256)
HU_MIN, HU_MAX = -1000.0, 1000.0
OUT_MIN, OUT_MAX = 0.0, 1.0

NUM_WORKERS = 8  # parallel GT loading to overlap with GPU decode


class LatentGTDataset(Dataset):
    """Yields (mu [4,120,120,64] float16, gt [1,D,H,W] float32) pairs."""

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
        mu = torch.load(item["mu_path"], weights_only=True)  # [4, 120, 120, 64]
        return {"mu": mu, "gt": gt.float()}


def main():
    parser = argparse.ArgumentParser(description="Measure MAISI VAE upper bound.")
    parser.add_argument(
        "--cache-recon",
        action="store_true",
        default=False,
        help=(
            "Save each decoded ct_recon to "
            "RECON_CACHE_ROOT/<split>/<sample_id>/ct_recon.pt (float16). "
            "WARNING: ~58 MB per volume; full valid split ~58 GB."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["valid", "train"],
        default="valid",
        help="Which latent split to process (default: valid).",
    )
    args = parser.parse_args()

    split = args.split
    cache_recon = args.cache_recon

    latent_split_dir = LATENTS_ROOT / split
    device = torch.device("cuda:0")

    # ---- Build and load frozen autoencoder ----------------------------------
    add_bundle_to_syspath(BUNDLE_DIR)
    net = build_autoencoder(BUNDLE_DIR, "autoencoder_def")
    ckpt_path = find_ae_ckpt(BUNDLE_DIR, None)
    print(f"Loading weights: {ckpt_path}")
    load_state(net, ckpt_path)
    net = net.to(device).eval()

    # Compile the decoder for speed (warmup cost ~6 min, then ~17s/vol vs 25s)
    print("Compiling decoder with torch.compile (mode='reduce-overhead')...")
    compiled_decode = torch.compile(net.decode_stage_2_outputs, mode="reduce-overhead")

    inferer = SlidingWindowInferer(
        roi_size=LATENT_ROI,
        sw_batch_size=SW_BATCH,
        mode="gaussian",
        overlap=SW_OVERLAP,
        device=device,  # stitching accumulator on GPU
        sw_device=device,
        progress=False,
    )

    # ---- Dataset / DataLoader for parallel GT loading -----------------------
    latent_dirs = sorted(latent_split_dir.iterdir())
    print(f"Found {len(latent_dirs)} {split} latents")

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
    per_sample_ids: list[str] = []

    # Warm up torch.compile with 3 passes so the kernel cache is hot
    # before the timed loop. Uses the first latent (already on CPU cache).
    print("Warming up torch.compile (3 passes)...")
    _warm_mu = (
        torch.load(str(latent_dirs[0] / "mu.pt"), weights_only=True)
        .unsqueeze(0)
        .to(device)
    )
    for _ in range(3):
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = inferer(_warm_mu, compiled_decode)
        torch.cuda.synchronize()
    del _warm_mu
    torch.cuda.empty_cache()
    print("Warmup done. Starting measurement loop.")

    for i, batch in enumerate(tqdm(loader, desc="Measuring", unit="vol")):
        sample_id = latent_dirs[i].name
        mu = batch["mu"].to(device)  # [1, 4, 120, 120, 64]
        gt = batch["gt"].to(device)  # [1, 1, 480, 480, 256]

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                recon = inferer(mu, compiled_decode).clamp(OUT_MIN, OUT_MAX)

        recon_f32 = recon.float()
        psnr_val = psnr_metric(recon_f32, gt)
        ssim_val = ssim_metric(recon_f32, gt)

        per_psnr.append(float(psnr_val.mean().item()))
        per_ssim.append(float(ssim_val.mean().item()))
        per_sample_ids.append(sample_id)

        if cache_recon:
            cache_dir = RECON_CACHE_ROOT / split / sample_id
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Store as float16 to halve disk usage (~58 MB/vol at float16 vs ~116 MB float32).
            torch.save(recon_f32.half().cpu(), cache_dir / "ct_recon.pt")

        del mu, gt, recon, recon_f32
        torch.cuda.empty_cache()

    # ---- Aggregate and save -------------------------------------------------
    psnr_arr = np.array(per_psnr)
    ssim_arr = np.array(per_ssim)

    result = {
        "n_samples": len(per_psnr),
        "psnr": {"mean": float(psnr_arr.mean()), "std": float(psnr_arr.std())},
        "ssim": {"mean": float(ssim_arr.mean()), "std": float(ssim_arr.std())},
        "per_sample": {"psnr": per_psnr, "ssim": per_ssim},
        "sample_ids": per_sample_ids,
        "notes": (
            "MAISI VAE encode->decode round-trip upper bound on CT-RATE validation set "
            "(1000 volumes). Intensity convention: HU clipped to [-1000, 1000] then "
            "scaled to [0, 1]. Spatial size: 480x480x256 (trilinear resize, no spacing "
            "resampling). Decoder: AutoencoderKlMaisi.decode_stage_2_outputs via "
            f"SlidingWindowInferer (latent_roi={LATENT_ROI}, overlap={SW_OVERLAP}, "
            f"sw_batch_size={SW_BATCH}, GPU stitching, torch.compile reduce-overhead). "
            "Latent mu decoded directly — no scale_factor division (scale_factor is "
            "diffusion-pipeline-only and not applied during encode/decode). "
            "Metrics: MONAI PSNRMetric and SSIMMetric (spatial_dims=3), data_range=1.0."
        ),
    }

    # Only overwrite the canonical upper_bound.json when processing the valid split,
    # so train-split caching runs don't clobber the benchmark file.
    if split == "valid":
        out_path = RESULTS_DIR / "upper_bound.json"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n=== Upper bound (n={result['n_samples']}) ===")
        print(f"PSNR: {result['psnr']['mean']:.3f} +/- {result['psnr']['std']:.3f} dB")
        print(f"SSIM: {result['ssim']['mean']:.4f} +/- {result['ssim']['std']:.4f}")
        print(f"Results saved to {out_path}")
    else:
        out_path = RESULTS_DIR / f"upper_bound_{split}.json"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n=== {split} split (n={result['n_samples']}) ===")
        print(f"PSNR: {result['psnr']['mean']:.3f} +/- {result['psnr']['std']:.3f} dB")
        print(f"SSIM: {result['ssim']['mean']:.4f} +/- {result['ssim']['std']:.4f}")
        print(f"Results saved to {out_path}")

    if cache_recon:
        print(f"Recon cache written to {RECON_CACHE_ROOT / split}/")


if __name__ == "__main__":
    main()
