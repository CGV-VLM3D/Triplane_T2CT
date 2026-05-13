"""Full evaluation script for the triplane autoencoder.

Loads a checkpoint, runs validation against the MAISI upper-bound baseline,
and writes metrics + figures to output_dir.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from monai.inferers import SlidingWindowInferer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT / "reference" / "scripts"))

BUNDLE_DIR = ROOT / "maisi_bundle"
UPPER_BOUND_JSON = ROOT / "results" / "upper_bound.json"

LATENT_ROI = (20, 20, 20)
SW_OVERLAP = 0.4


def _build_maisi_decoder(device: torch.device):
    sys.path.insert(0, str(BUNDLE_DIR))
    from extract_maisi_latent import (
        add_bundle_to_syspath,
        build_autoencoder,
        find_ae_ckpt,
        load_state,
    )

    add_bundle_to_syspath(BUNDLE_DIR)
    net = build_autoencoder(BUNDLE_DIR, "autoencoder_def")
    ckpt_path = find_ae_ckpt(BUNDLE_DIR, None)
    load_state(net, ckpt_path)
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def run_full_evaluation(
    model_ckpt_path: str,
    config_path: str,
    output_dir: str,
    sw_batch_size: int = 16,
) -> None:
    """Run full validation and write results to output_dir.

    Parameters
    ----------
    model_ckpt_path:
        Path to a checkpoint saved by scripts/train.py (dict with "model" key).
    config_path:
        Path to the Hydra YAML config used to train.
    output_dir:
        Directory to write metrics.json, per_sample_metrics.csv, and figs/.
    """
    from scripts.train import build_model
    from src.data.maisi_latent_dataset import MAISILatentDataset
    from src.evaluation.validate import run_validation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(exist_ok=True)

    # --- Load model ---
    cfg = OmegaConf.load(config_path)
    model = build_model(cfg).to(device)
    ckpt = torch.load(model_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # --- Load MAISI decoder ---
    print("Loading MAISI autoencoder...")
    net = _build_maisi_decoder(device)
    maisi_decoder = torch.compile(net.decode_stage_2_outputs, mode="reduce-overhead")

    # Warm up the compiled decoder so the first sample doesn't pay the cost.
    print("Warming up torch.compile (3 passes)...")
    _inferer_warm = SlidingWindowInferer(
        roi_size=(20, 20, 20),
        sw_batch_size=16,
        mode="gaussian",
        overlap=0.4,
        device=device,
        sw_device=device,
        progress=False,
    )
    _dummy = torch.zeros(1, 4, 120, 120, 64, device=device)
    for _ in range(3):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = _inferer_warm(_dummy, maisi_decoder)
        torch.cuda.synchronize()
    del _dummy, _inferer_warm
    torch.cuda.empty_cache()

    # --- Dataset ---
    ds = MAISILatentDataset(split="valid", load_ct_recon=True)
    val_loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # --- Aggregate validation ---
    print("Running full validation...")
    with torch.no_grad():
        agg = run_validation(
            model=model,
            maisi_decoder=maisi_decoder,
            val_loader=val_loader,
            n_samples=None,
            device=str(device),
            compute_image_metrics=True,
            sw_batch_size=sw_batch_size,
        )

    # --- Per-sample pass (collect per-sample rows) ---
    print("Collecting per-sample metrics...")
    per_sample_rows = _collect_per_sample(
        model, maisi_decoder, val_loader, device, sw_batch_size
    )

    # --- Load upper bound ---
    with open(UPPER_BOUND_JSON) as f:
        ub = json.load(f)
    ub_psnr_mean = ub["psnr"]["mean"]
    ub_ssim_mean = ub["ssim"]["mean"]
    ub_psnr_per = np.array(ub["per_sample"]["psnr"])
    ub_ssim_per = np.array(ub["per_sample"]["ssim"])
    ub_sample_ids = ub.get("sample_ids", [])

    # --- Compute per-sample std for aggregate metrics ---
    our_psnr_per = np.array([r["image_psnr_3d"] for r in per_sample_rows])
    our_ssim_per = np.array([r["image_ssim_3d"] for r in per_sample_rows])

    metrics_out = {
        "n_samples": len(per_sample_rows),
        "latent_l1": {"mean": agg.get("latent_l1"), "std": None},
        "latent_l2": {"mean": agg.get("latent_l2"), "std": None},
        "latent_psnr": {"mean": agg.get("latent_psnr"), "std": None},
        "image_psnr_3d": {
            "mean": float(our_psnr_per.mean()),
            "std": float(our_psnr_per.std()),
        },
        "image_ssim_3d": {
            "mean": float(our_ssim_per.mean()),
            "std": float(our_ssim_per.std()),
        },
        "gap_to_upper_psnr": float(ub_psnr_mean - our_psnr_per.mean()),
        "gap_to_upper_ssim_3d": float(ub_ssim_mean - our_ssim_per.mean()),
        "upper_bound": {"psnr_mean": ub_psnr_mean, "ssim_mean": ub_ssim_mean},
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Saved metrics to {out_dir / 'metrics.json'}")

    # --- Per-sample CSV ---
    csv_path = out_dir / "per_sample_metrics.csv"
    fieldnames = [
        "sample_id",
        "latent_l1",
        "latent_l2",
        "latent_psnr",
        "image_psnr_3d",
        "image_ssim_3d",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_sample_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Saved per-sample CSV to {csv_path}")

    # --- Figures ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. PSNR histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        our_psnr_per,
        bins=40,
        alpha=0.7,
        label=f"Triplane-AE (mean={our_psnr_per.mean():.2f})",
    )
    if len(ub_psnr_per) == len(our_psnr_per):
        ax.hist(
            ub_psnr_per,
            bins=40,
            alpha=0.5,
            label=f"MAISI upper bound (mean={ub_psnr_per.mean():.2f})",
        )
    ax.set_xlabel("Image PSNR (dB)")
    ax.set_ylabel("Count")
    ax.set_title("Per-sample PSNR distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs_dir / "psnr_histogram.png", dpi=150)
    plt.close(fig)

    # 2. Scatter: our PSNR vs upper-bound PSNR per sample
    if len(ub_psnr_per) == len(our_psnr_per):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(ub_psnr_per, our_psnr_per, s=4, alpha=0.4)
        lims = [
            min(ub_psnr_per.min(), our_psnr_per.min()) - 1,
            max(ub_psnr_per.max(), our_psnr_per.max()) + 1,
        ]
        ax.plot(lims, lims, "r--", linewidth=1, label="y = x")
        ax.set_xlabel("Upper-bound PSNR (dB)")
        ax.set_ylabel("Triplane-AE PSNR (dB)")
        ax.set_title("Per-sample PSNR: Triplane-AE vs MAISI upper bound")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figs_dir / "psnr_scatter.png", dpi=150)
        plt.close(fig)

    # 3. Worst-case visualisation: 3 lowest-PSNR samples
    _worst_case_viz(per_sample_rows, ds, net, device, figs_dir)

    print(f"Figures saved to {figs_dir}")
    print(f"\n=== Summary ===")
    print(
        f"Image PSNR: {our_psnr_per.mean():.3f} dB  (gap to UB: {metrics_out['gap_to_upper_psnr']:.3f} dB)"
    )
    print(
        f"Image SSIM: {our_ssim_per.mean():.4f}     (gap to UB: {metrics_out['gap_to_upper_ssim_3d']:.4f})"
    )


def _collect_per_sample(
    model, maisi_decoder, val_loader, device: torch.device, sw_batch_size: int
) -> list[dict]:
    """Second pass over val_loader to collect per-sample metric rows."""
    from monai.inferers import SlidingWindowInferer
    from monai.metrics import PSNRMetric, SSIMMetric

    inferer = SlidingWindowInferer(
        roi_size=LATENT_ROI,
        sw_batch_size=sw_batch_size,
        mode="gaussian",
        overlap=SW_OVERLAP,
        device=device,
        sw_device=device,
        progress=False,
    )
    psnr_metric = PSNRMetric(max_val=1.0, reduction="none")
    ssim_metric = SSIMMetric(
        spatial_dims=3, data_range=1.0, win_size=11, reduction="none"
    )

    rows = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            mu = batch["mu"].to(device)
            ct_recon = batch["ct_recon"].to(device)
            sample_id = (
                batch["sample_id"][0]
                if isinstance(batch["sample_id"], (list, tuple))
                else batch["sample_id"]
            )

            mu_hat, _ = model(mu)

            diff = mu_hat - mu
            lat_l1 = diff.abs().mean().item()
            lat_l2 = diff.pow(2).mean().sqrt().item()
            mse = diff.pow(2).mean().item()
            import math

            drange = (mu.max() - mu.min()).item() or 1.0
            lat_psnr = (
                20.0 * math.log10(drange / math.sqrt(mse)) if mse > 0 else float("inf")
            )

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                our_recon = inferer(mu_hat.float(), maisi_decoder).clamp(0.0, 1.0)
            our_recon_f32 = our_recon.float()

            img_psnr = float(psnr_metric(our_recon_f32, ct_recon).mean().item())
            img_ssim = float(ssim_metric(our_recon_f32, ct_recon).mean().item())

            rows.append(
                {
                    "sample_id": sample_id,
                    "latent_l1": lat_l1,
                    "latent_l2": lat_l2,
                    "latent_psnr": lat_psnr,
                    "image_psnr_3d": img_psnr,
                    "image_ssim_3d": img_ssim,
                    # Keep the decoded recon tensor path-accessible for worst-case viz
                    "_our_recon_cpu": our_recon_f32.cpu(),
                    "_ct_recon_cpu": ct_recon.cpu(),
                }
            )

            del mu, mu_hat, ct_recon, our_recon, our_recon_f32
            torch.cuda.empty_cache()

    return rows


def _worst_case_viz(
    per_sample_rows: list[dict], ds, net, device: torch.device, figs_dir: Path
) -> None:
    """Save a 3-row PNG for the 3 worst-PSNR samples."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sorted_rows = sorted(per_sample_rows, key=lambda r: r["image_psnr_3d"])
    worst = sorted_rows[:3]

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    col_titles = ["MAISI recon (reference)", "Our recon", "Abs diff"]

    for row_idx, row in enumerate(worst):
        ct_vol = row["_ct_recon_cpu"][0, 0]  # [480, 480, 256]
        our_vol = row["_our_recon_cpu"][0, 0]
        diff_vol = (ct_vol - our_vol).abs()

        mid_z = ct_vol.shape[2] // 2

        ct_slice = ct_vol[:, :, mid_z].numpy()
        our_slice = our_vol[:, :, mid_z].numpy()
        diff_slice = diff_vol[:, :, mid_z].numpy()

        axes[row_idx, 0].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 1].imshow(our_slice, cmap="gray", vmin=0, vmax=1)
        im = axes[row_idx, 2].imshow(diff_slice, cmap="hot", vmin=0, vmax=0.5)
        fig.colorbar(im, ax=axes[row_idx, 2], fraction=0.046)

        axes[row_idx, 0].set_ylabel(
            f"{row['sample_id']}\nPSNR={row['image_psnr_3d']:.2f}", fontsize=7
        )

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=9)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("3 worst-PSNR samples (mid-axial slice)", fontsize=11)
    fig.tight_layout()
    fig.savefig(figs_dir / "worst_case.png", dpi=150)
    plt.close(fig)
