"""
Sanity check for Path-B toy-latent strategy: does MAISI VAE round-trip survive
when we encode at coarser effective spacing?

For each of N CT-RATE valid volumes, we run TWO independent round-trips:

  1mm baseline : load -> Resized(480,480,256) -> encode -> decode -> PSNR/SSIM vs
                 the (480,480,256) ground truth.
  2mm toy      : load -> Resized(240,240,128) -> encode -> decode -> PSNR/SSIM vs
                 the (240,240,128) ground truth.

We compare each pipeline against ITS OWN resized GT (apples-to-apples), so the
delta isolates "does MAISI VAE handle coarser effective spacing well?" rather
than "does coarser spacing lose detail vs original CT?" (the latter is trivially
worse and not the question we're asking).

Gate decision: if median 2mm PSNR drop vs 1mm baseline is <= 1.5 dB, Path B is
viable for Tier-1 architecture sweeps.

Notes:
- Re-uses the same encode/decode sliding-window settings as the existing
  pipeline (encode roi=80, decode latent_roi=20, overlap=0.4).
- Skips torch.compile: warmup cost (~6 min x 2 stages) is not amortised across
  only 50 samples.
- Output: results/validate_2mm_spacing.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import Dataset
from monai.inferers import sliding_window_inference
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

LATENTS_VALID_DIR = WORKSPACE / "datasets" / "datasets" / "latents" / "valid"
RESULTS_DIR = WORKSPACE / "results"

SW_OVERLAP = 0.4
ENCODE_ROI = (80, 80, 80)  # MAISI bundle default
DECODE_LATENT_ROI = (20, 20, 20)  # encode_roi / 4
SW_BATCH = 16

HU_MIN, HU_MAX = -1000.0, 1000.0
OUT_MIN, OUT_MAX = 0.0, 1.0

BASELINE_SIZE = (480, 480, 256)  # ~1mm effective spacing
TOY_SIZE = (240, 240, 128)  # ~2mm effective spacing


class SrcCTDataset(Dataset):
    """Loads only the GT CT volume at the requested spatial_size."""

    def __init__(self, src_files: list[str], spatial_size: tuple[int, int, int]):
        gt_transform = build_transforms(
            spatial_size=spatial_size,
            hu_min=HU_MIN,
            hu_max=HU_MAX,
            out_min=OUT_MIN,
            out_max=OUT_MAX,
            train=False,
        )
        data = [{"image": f} for f in src_files]
        super().__init__(data=data, transform=gt_transform)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        gt = item["image"]
        if hasattr(gt, "as_tensor"):
            gt = gt.as_tensor()
        return {"gt": gt.float()}


@torch.no_grad()
def encode_decode(
    net: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Returns (reconstruction in [0,1], latent shape) for a single volume."""

    def _encode_pred(inp: torch.Tensor) -> torch.Tensor:
        mu, _ = net.encode(inp)
        return mu

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        mu = sliding_window_inference(
            inputs=x,
            roi_size=ENCODE_ROI,
            sw_batch_size=SW_BATCH,
            predictor=_encode_pred,
            overlap=SW_OVERLAP,
            mode="gaussian",
            progress=False,
        )

    latent_shape = tuple(mu.shape)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        recon = sliding_window_inference(
            inputs=mu,
            roi_size=DECODE_LATENT_ROI,
            sw_batch_size=SW_BATCH,
            predictor=net.decode_stage_2_outputs,
            overlap=SW_OVERLAP,
            mode="gaussian",
            progress=False,
        )

    recon = recon.float().clamp(OUT_MIN, OUT_MAX)
    return recon, latent_shape


def run_pipeline(
    net: torch.nn.Module,
    src_files: list[str],
    spatial_size: tuple[int, int, int],
    device: torch.device,
    label: str,
) -> dict:
    ds = SrcCTDataset(src_files, spatial_size)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    psnr_metric = PSNRMetric(max_val=1.0, reduction="none")
    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0, reduction="none")

    psnrs: list[float] = []
    ssims: list[float] = []
    latent_shape_seen: tuple[int, ...] | None = None

    for batch in tqdm(loader, desc=f"{label} ({spatial_size})", unit="vol"):
        gt = batch["gt"].to(device)  # [1, 1, D, H, W]
        recon, latent_shape = encode_decode(net, gt, device)
        if latent_shape_seen is None:
            latent_shape_seen = latent_shape

        psnrs.append(float(psnr_metric(recon, gt).mean().item()))
        ssims.append(float(ssim_metric(recon, gt).mean().item()))

        del gt, recon
        torch.cuda.empty_cache()

    arr_p = np.array(psnrs)
    arr_s = np.array(ssims)
    return {
        "spatial_size": list(spatial_size),
        "latent_shape": list(latent_shape_seen) if latent_shape_seen else None,
        "n": len(psnrs),
        "psnr_mean": float(arr_p.mean()),
        "psnr_std": float(arr_p.std()),
        "psnr_median": float(np.median(arr_p)),
        "ssim_mean": float(arr_s.mean()),
        "ssim_std": float(arr_s.std()),
        "ssim_median": float(np.median(arr_s)),
        "per_sample_psnr": psnrs,
        "per_sample_ssim": ssims,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of valid volumes to test (default: 50).",
    )
    parser.add_argument(
        "--out", type=str, default=str(RESULTS_DIR / "validate_2mm_spacing.json")
    )
    args = parser.parse_args()

    device = torch.device("cuda:0")

    # Pull first N valid volumes via their precomputed latents' src.txt pointers.
    latent_dirs = sorted(LATENTS_VALID_DIR.iterdir())[: args.n]
    src_files = [(ld / "src.txt").read_text().strip() for ld in latent_dirs]
    print(f"Selected {len(src_files)} valid volumes (first src: {src_files[0]})")

    # ---- Build and load frozen autoencoder ----------------------------------
    add_bundle_to_syspath(BUNDLE_DIR)
    net = build_autoencoder(BUNDLE_DIR, "autoencoder_def")
    ckpt_path = find_ae_ckpt(BUNDLE_DIR, None)
    print(f"Loading weights: {ckpt_path}")
    load_state(net, ckpt_path)
    net = net.to(device).eval()

    # ---- Run both pipelines -------------------------------------------------
    baseline = run_pipeline(net, src_files, BASELINE_SIZE, device, label="1mm baseline")
    toy = run_pipeline(net, src_files, TOY_SIZE, device, label="2mm toy")

    delta_psnr = toy["psnr_median"] - baseline["psnr_median"]
    delta_ssim = toy["ssim_median"] - baseline["ssim_median"]

    result = {
        "n_samples": args.n,
        "baseline_1mm": baseline,
        "toy_2mm": toy,
        "delta": {
            "psnr_median_2mm_minus_1mm": delta_psnr,
            "ssim_median_2mm_minus_1mm": delta_ssim,
        },
        "gate_decision": {
            "threshold_psnr_drop_db": 1.5,
            "viable": delta_psnr >= -1.5,
        },
        "notes": (
            "Per-pipeline PSNR/SSIM are against each pipeline's OWN resized GT, "
            "isolating MAISI VAE behavior at different effective spacings. "
            f"encode roi={ENCODE_ROI}, decode latent_roi={DECODE_LATENT_ROI}, "
            f"overlap={SW_OVERLAP}, sw_batch={SW_BATCH}. No torch.compile."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print(f"=== 1mm baseline (n={baseline['n']}) ===")
    print(f"  latent shape: {baseline['latent_shape']}")
    print(
        f"  PSNR  mean {baseline['psnr_mean']:.2f} ± {baseline['psnr_std']:.2f}  median {baseline['psnr_median']:.2f}"
    )
    print(
        f"  SSIM  mean {baseline['ssim_mean']:.4f} ± {baseline['ssim_std']:.4f}  median {baseline['ssim_median']:.4f}"
    )
    print(f"=== 2mm toy (n={toy['n']}) ===")
    print(f"  latent shape: {toy['latent_shape']}")
    print(
        f"  PSNR  mean {toy['psnr_mean']:.2f} ± {toy['psnr_std']:.2f}  median {toy['psnr_median']:.2f}"
    )
    print(
        f"  SSIM  mean {toy['ssim_mean']:.4f} ± {toy['ssim_std']:.4f}  median {toy['ssim_median']:.4f}"
    )
    print(f"=== Delta (2mm - 1mm, median) ===")
    print(f"  PSNR: {delta_psnr:+.2f} dB   SSIM: {delta_ssim:+.4f}")
    print(f"=== Gate (drop <= 1.5 dB) ===")
    print(f"  viable: {result['gate_decision']['viable']}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
