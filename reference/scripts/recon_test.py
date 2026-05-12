"""
Encode + decode a single CT volume with the MAISI autoencoder and
compute SSIM / PSNR between input and reconstruction.

Saves GT and recon as .nii.gz under /workspace/recon_test/.

Example:
    python scripts/recon_test.py
    python scripts/recon_test.py --patient_id train_42
    python scripts/recon_test.py --image_path /abs/path/to/foo.nii.gz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.dataloader import CTVolumeDataset  # noqa: E402
from monai.inferers import sliding_window_inference  # noqa: E402
from monai.metrics import PSNRMetric, SSIMMetric  # noqa: E402

from extract_maisi_latent import (  # noqa: E402
    add_bundle_to_syspath,
    build_autoencoder,
    find_ae_ckpt,
    load_state,
)


@torch.no_grad()
def encode_mu(net, x, roi, overlap, sw_batch):
    def predictor(window):
        out = net.encode(window)
        if isinstance(out, (tuple, list)):
            return out[0]
        if hasattr(out, "mean"):
            return out.mean
        return out

    return sliding_window_inference(
        inputs=x, roi_size=roi, sw_batch_size=sw_batch,
        predictor=predictor, overlap=overlap, mode="gaussian", progress=False,
    )


@torch.no_grad()
def encode_mu_sigma(net, x, roi, overlap, sw_batch):
    """
    Sliding-window encode returning mu AND sigma, stacked along channel.

    The predictor stacks (mu, sigma) into [B, 2C, D', H', W'] so that
    MONAI's sliding_window_inference stitches both with the same gaussian
    blending. We split them back into separate tensors after stitching.
    """
    def predictor(window):
        mu, sigma = net.encode(window)
        return torch.cat([mu, sigma], dim=1)

    out = sliding_window_inference(
        inputs=x, roi_size=roi, sw_batch_size=sw_batch,
        predictor=predictor, overlap=overlap, mode="gaussian", progress=False,
    )
    c = out.shape[1] // 2
    return out[:, :c], out[:, c:]


@torch.no_grad()
def decode_sw(net, z, image_roi, overlap, sw_batch):
    """
    Sliding-window decode in latent space.

    image_roi is the image-side window size (e.g. 80). Encoder downsamples 4x,
    so latent_roi = image_roi // 4. The decoder upsamples 4x, so each latent
    window of 20^3 produces an 80^3 image patch, stitched by MONAI.

    Full-volume decode breaks MAISI's internal num_splits behavior at large
    spatial sizes; sliding-window decode keeps the network in its trained regime.
    """
    latent_roi = tuple(max(1, r // 4) for r in image_roi)
    return sliding_window_inference(
        inputs=z, roi_size=latent_roi, sw_batch_size=sw_batch,
        predictor=net.decode_stage_2_outputs,
        overlap=overlap, mode="gaussian", progress=False,
    )


def save_nifti(volume_chw: torch.Tensor, path: Path, spacing=(1.0, 1.0, 1.0)) -> None:
    """volume_chw: [1, D, H, W] or [D, H, W]; saved as float32 nii.gz."""
    if volume_chw.dim() == 4:
        volume_chw = volume_chw[0]
    arr = volume_chw.detach().cpu().to(torch.float32).numpy()
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    nib.save(nib.Nifti1Image(arr, affine), str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle_dir", type=str, default="/workspace/maisi_bundle")
    parser.add_argument("--ae_ckpt", type=str, default=None)
    parser.add_argument("--network_key", type=str, default="autoencoder_def")

    parser.add_argument("--split_json", type=str, default="/workspace/datasets/split.json")
    parser.add_argument("--patient_id", type=str, default=None,
                        help="Pick a specific patient (e.g. train_42) from split.json.")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Override: encode/decode this exact .nii.gz file instead of split lookup.")

    parser.add_argument("--spatial_size", type=int, nargs=3, default=[480, 480, 256])
    parser.add_argument("--out_dir", type=str, default="/workspace/recon_test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--sw_roi", type=int, nargs=3, default=None)
    parser.add_argument("--sw_overlap", type=float, default=None)
    parser.add_argument("--sw_batch_size", type=int, default=4)
    parser.add_argument("--sample_z", action="store_true",
                        help="Encode mu AND sigma, sample z = mu + sigma*eps once, decode z. "
                             "Default (off) decodes the deterministic mu.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for sampling eps (only used with --sample_z).")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- pick image ----------------------------------------------------------
    if args.image_path:
        image_path = Path(args.image_path).resolve()
        patient_id = image_path.name.replace(".nii.gz", "")
    else:
        with open(args.split_json) as f:
            split = json.load(f)
        root = Path(split["root"])
        pool = split["train"] + split["valid"]
        if args.patient_id:
            entry = next((e for e in pool if e["patient_id"] == args.patient_id), None)
            if entry is None:
                raise ValueError(f"{args.patient_id} not found in split.json")
        else:
            entry = pool[0]
        image_path = (root / entry["path"]).resolve()
        patient_id = entry["patient_id"]
    print(f"[recon] patient: {patient_id}")
    print(f"[recon] image:   {image_path}")

    # ---- preprocess ----------------------------------------------------------
    ds = CTVolumeDataset(
        spatial_size=tuple(args.spatial_size),
        train=False,
        files=[str(image_path)],
    )
    sample = ds[0]
    gt = sample["image"]
    if hasattr(gt, "as_tensor"):
        gt = gt.as_tensor()
    gt = gt.unsqueeze(0)  # [1, 1, D, H, W]
    print(f"[recon] preprocessed gt shape: {tuple(gt.shape)}  range=[{gt.min():.3f}, {gt.max():.3f}]")

    # ---- build network -------------------------------------------------------
    add_bundle_to_syspath(bundle_dir)
    net = build_autoencoder(bundle_dir, args.network_key)
    ckpt_path = find_ae_ckpt(bundle_dir, args.ae_ckpt)
    print(f"[recon] weights: {ckpt_path}")
    load_state(net, ckpt_path)

    # sliding-window defaults from bundle
    with open(bundle_dir / "configs" / "inference.json") as f:
        bundle_cfg = json.load(f)
    sw_roi = tuple(args.sw_roi) if args.sw_roi else tuple(
        bundle_cfg.get("autoencoder_sliding_window_infer_size", [96, 96, 96])
    )
    sw_overlap = args.sw_overlap if args.sw_overlap is not None else float(
        bundle_cfg.get("autoencoder_sliding_window_infer_overlap", 0.4)
    )
    print(f"[recon] sw_roi={sw_roi}  overlap={sw_overlap}  sw_batch={args.sw_batch_size}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "float32": torch.float32,
             "bfloat16": torch.bfloat16}[args.dtype]
    net = net.to(device).eval()
    x = gt.to(device=device, dtype=dtype)

    # ---- encode / decode -----------------------------------------------------
    with torch.autocast(device_type=device.type,
                        dtype=dtype if dtype != torch.float32 else torch.float32,
                        enabled=(dtype != torch.float32)):
        if args.sample_z:
            print("[recon] encoding mu + sigma (sliding window)...")
            mu, sigma = encode_mu_sigma(net, x, sw_roi, sw_overlap, args.sw_batch_size)
            print(f"[recon] mu shape: {tuple(mu.shape)}  "
                  f"mu range=[{mu.float().min():.3f},{mu.float().max():.3f}]  "
                  f"sigma range=[{sigma.float().min():.4f},{sigma.float().max():.4f}]")
            g = torch.Generator(device=device).manual_seed(args.seed)
            eps = torch.randn(mu.shape, generator=g, device=device, dtype=mu.dtype)
            z = mu + sigma * eps
            print(f"[recon] sampled z range=[{z.float().min():.3f},{z.float().max():.3f}]")
        else:
            print("[recon] encoding mu (sliding window)...")
            z = encode_mu(net, x, sw_roi, sw_overlap, args.sw_batch_size)
            print(f"[recon] latent shape: {tuple(z.shape)}")

        print("[recon] decoding (sliding window in latent space)...")
        recon = decode_sw(net, z, sw_roi, sw_overlap, args.sw_batch_size)
        recon = recon.clamp(0.0, 1.0)
        print(f"[recon] recon shape: {tuple(recon.shape)}  range=[{recon.min():.3f}, {recon.max():.3f}]")

    # ---- metrics -------------------------------------------------------------
    gt_f32 = gt.to(torch.float32).to(device)
    recon_f32 = recon.to(torch.float32)

    psnr_metric = PSNRMetric(max_val=1.0)
    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0)

    psnr = psnr_metric(recon_f32, gt_f32).mean().item()
    ssim = ssim_metric(recon_f32, gt_f32).mean().item()
    mae = (recon_f32 - gt_f32).abs().mean().item()
    mse = ((recon_f32 - gt_f32) ** 2).mean().item()

    print(f"[recon] PSNR: {psnr:.3f} dB")
    print(f"[recon] SSIM: {ssim:.4f}")
    print(f"[recon] MAE : {mae:.5f}")
    print(f"[recon] MSE : {mse:.6f}")

    # ---- save ----------------------------------------------------------------
    gt_path = out_dir / f"{patient_id}_gt.nii.gz"
    recon_path = out_dir / f"{patient_id}_recon.nii.gz"
    save_nifti(gt[0], gt_path)
    save_nifti(recon[0].cpu(), recon_path)
    print(f"[recon] wrote {gt_path}")
    print(f"[recon] wrote {recon_path}")

    metrics = {
        "patient_id": patient_id,
        "source": str(image_path),
        "spatial_size": list(args.spatial_size),
        "latent_shape": list(z.shape),
        "psnr_db": psnr,
        "ssim": ssim,
        "mae": mae,
        "mse": mse,
        "intensity_range": [0.0, 1.0],
        "dtype": args.dtype,
        "sw_roi": list(sw_roi),
        "sw_overlap": sw_overlap,
    }
    metrics_path = out_dir / f"{patient_id}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[recon] wrote {metrics_path}")


if __name__ == "__main__":
    main()
