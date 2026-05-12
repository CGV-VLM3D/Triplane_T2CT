from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.dataloader import CTVolumeDataset  # noqa: E402
from models.trivqvae_conv import TriVQAEConv  # noqa: E402


def build_trivqae_from_cfg(cfg_dict) -> TriVQAEConv:
    m_cfg = cfg_dict["model"]
    t_cfg = cfg_dict["training"]
    decoder_channels = m_cfg.get("decoder_channels", None)
    if decoder_channels is not None:
        decoder_channels = tuple(decoder_channels)
    return TriVQAEConv(
        in_channels=int(m_cfg["in_channels"]),
        out_channels=int(m_cfg["in_channels"]),
        latent_shape=tuple(t_cfg["spatial_size"]),
        encoder_channels=tuple(m_cfg["encoder_channels"]),
        blocks_per_stage=int(m_cfg.get("blocks_per_stage", 2)),
        plane_channels=int(m_cfg["plane_channels"]),
        decoder_channels=decoder_channels,
        final_hidden_channels=int(m_cfg.get("final_hidden_channels", m_cfg["plane_channels"])),
        final_depth=int(m_cfg.get("final_depth", 2)),
        output_act=m_cfg.get("output_act", None),
    )


def build_maisi(bundle_dir: Path, network_key: str):
    if str(bundle_dir) not in sys.path:
        sys.path.insert(0, str(bundle_dir))
    from monai.bundle import ConfigParser

    cfg_path = bundle_dir / "configs" / "inference.json"
    p = ConfigParser()
    p.read_config(str(cfg_path))
    p.parse()
    return p.get_parsed_content(network_key)


def find_ae_ckpt(bundle_dir: Path, override: str | None) -> Path:
    if override:
        return Path(override)
    cands = sorted((bundle_dir / "models").glob("autoencoder*.pt"))
    if not cands:
        raise FileNotFoundError(bundle_dir / "models")
    return cands[0]


def maisi_decode_predictor(net):
    def _fn(window):
        if hasattr(net, "decode_stage_2_outputs"):
            return net.decode_stage_2_outputs(window)
        out = net.decode(window)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out
    return _fn


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="TriVQAEConv ckpt from train_latent_conv.py")
    ap.add_argument("--latent", default=None,
                    help="Specific latent .pt path. Default: first under latent_dir.")
    ap.add_argument("--latent_dir", default="/workspace/dataset/maisi_latents")
    ap.add_argument("--out_dir", default="/workspace/outputs/trivqae_conv_recon_check")
    ap.add_argument("--decode_image", action="store_true",
                    help="Also decode through MAISI to image space (saves nii.gz).")
    ap.add_argument("--bundle_dir", default="/workspace/maisi_bundle")
    ap.add_argument("--network_key", default="autoencoder_def")
    ap.add_argument("--ae_ckpt", default=None)
    ap.add_argument("--sw_overlap", type=float, default=None)
    ap.add_argument("--sw_batch_size", type=int, default=1)
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--spatial_size", type=int, nargs=3, default=[480, 480, 256])
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    # 1. TriVQAEConv
    print(f"[recon] loading TriVQAEConv ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg_dict = ckpt["cfg"]
    model = build_trivqae_from_cfg(cfg_dict)
    miss, unexp = model.load_state_dict(ckpt["generator"], strict=False)
    print(f"  missing={len(miss)} unexpected={len(unexp)}")
    model = model.to(device).eval()

    # 2. Norm stats
    stats_path = Path(args.latent_dir) / "stats.json"
    with open(stats_path, "r") as f:
        stats = json.load(f)
    mean = torch.tensor(stats["channel_mean"], dtype=torch.float32,
                        device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(stats["channel_std"], dtype=torch.float32,
                       device=device).clamp(min=1e-6).view(1, -1, 1, 1, 1)

    # 3. One latent
    if args.latent:
        latent_path = Path(args.latent)
    else:
        cands = sorted(Path(args.latent_dir).glob("*.pt"))
        if not cands:
            raise FileNotFoundError(args.latent_dir)
        latent_path = cands[0]
    print(f"[recon] using latent: {latent_path}")
    blob = torch.load(str(latent_path), map_location="cpu")
    mu_real = blob["mu"].to(torch.float32).unsqueeze(0).to(device)  # [1, C, D, H, W]
    src = blob.get("src", "?")
    print(f"  src: {src}")
    print(f"  mu shape: {tuple(mu_real.shape)}, range "
          f"[{float(mu_real.min()):.3f}, {float(mu_real.max()):.3f}]")

    # 4. TriVQAEConv roundtrip in normalized latent space
    z = (mu_real - mean) / std
    print(f"  z (normalized) range: [{float(z.min()):.3f}, {float(z.max()):.3f}]")

    with torch.autocast(device_type=device.type, dtype=dtype,
                        enabled=(dtype != torch.float32)):
        out = model(z.to(dtype))
        z_hat = out[0]
    z_hat = z_hat.float()

    latent_l1 = (z_hat - z).abs().mean().item()
    latent_mse = ((z_hat - z) ** 2).mean().item()
    print(f"[recon] latent L1={latent_l1:.6f}  MSE={latent_mse:.6f}")

    mu_hat = z_hat * std + mean

    # Per-channel breakdown (informative for z-score data)
    per_ch_l1 = (z_hat - z).abs().mean(dim=(0, 2, 3, 4)).cpu().tolist()
    print(f"  per-channel latent L1: {[f'{v:.4f}' for v in per_ch_l1]}")

    # 5. Save latent recon
    np.save(out_dir / "mu_real.npy", mu_real[0].float().cpu().numpy())
    np.save(out_dir / "mu_hat.npy", mu_hat[0].float().cpu().numpy())
    np.save(out_dir / "z.npy", z[0].float().cpu().numpy())
    np.save(out_dir / "z_hat.npy", z_hat[0].cpu().numpy())

    summary: dict = {
        "ckpt": args.ckpt,
        "latent_path": str(latent_path),
        "src": src,
        "mu_shape": list(mu_real.shape),
        "latent_L1": latent_l1,
        "latent_MSE": latent_mse,
        "per_channel_latent_L1": per_ch_l1,
    }

    # 6. Optional: decode through MAISI to image space
    if args.decode_image:
        bundle_dir = Path(args.bundle_dir).resolve()

        with open(bundle_dir / "configs" / "inference.json", "r") as f:
            bcfg = json.load(f)
        sw_input_roi = tuple(bcfg.get("autoencoder_sliding_window_infer_size", [80, 80, 80]))
        sw_overlap = args.sw_overlap if args.sw_overlap is not None else float(
            bcfg.get("autoencoder_sliding_window_infer_overlap", 0.4)
        )
        latent_roi = tuple(max(1, r // 4) for r in sw_input_roi)
        print(f"[recon] decode latent_roi={latent_roi} overlap={sw_overlap}")

        maisi = build_maisi(bundle_dir, args.network_key)
        ae_ckpt_path = find_ae_ckpt(bundle_dir, args.ae_ckpt)
        sd = torch.load(str(ae_ckpt_path), map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and not any(
            k.startswith(("encoder", "decoder", "quant")) for k in sd
        ):
            sd = sd["state_dict"]
        maisi.load_state_dict(sd, strict=False)
        maisi = maisi.to(device).eval()
        for p in maisi.parameters():
            p.requires_grad_(False)

        from monai.inferers import sliding_window_inference

        def _decode(latent):
            with torch.autocast(device_type=device.type, dtype=dtype,
                                enabled=(dtype != torch.float32)):
                return sliding_window_inference(
                    inputs=latent,
                    roi_size=latent_roi,
                    sw_batch_size=args.sw_batch_size,
                    predictor=maisi_decode_predictor(maisi),
                    overlap=sw_overlap,
                    mode="gaussian",
                    progress=False,
                )

        print("[recon] decoding x_target = MAISI(mu_real)...")
        x_target = _decode(mu_real.to(dtype)).float()
        print("[recon] decoding x_hat = MAISI(mu_hat)...")
        x_hat = _decode(mu_hat.to(dtype)).float()

        x_t = x_target.clamp(0.0, 1.0)
        x_h = x_hat.clamp(0.0, 1.0)
        x_t_np = x_t[0, 0].cpu().numpy()
        x_h_np = x_h[0, 0].cpu().numpy()

        # Load the original CT (GT) by re-running the same preprocessing on
        # the source file recorded in the latent blob. Skipped if src is
        # unknown.
        x_gt_np = None
        if src and src != "?" and Path(src).exists():
            print(f"[recon] loading GT image from src={src}")
            ds = CTVolumeDataset(
                data_root=None,
                spatial_size=tuple(args.spatial_size),
                files=[str(src)],
                train=False,
            )
            sample = ds[0]
            x_gt = sample["image"]
            if hasattr(x_gt, "as_tensor"):
                x_gt = x_gt.as_tensor()
            x_gt = x_gt.float().numpy()  # [1, D, H, W]
            x_gt_np = x_gt[0]
            print(f"  x_gt shape: {x_gt_np.shape}, range "
                  f"[{x_gt_np.min():.4f}, {x_gt_np.max():.4f}]")
        else:
            print("[recon] GT not available (src missing); skipping vs-gt metrics.")

        # Crop x_t/x_h to GT shape if any rounding mismatch.
        if x_gt_np is not None and x_t_np.shape != x_gt_np.shape:
            slc = tuple(slice(0, s) for s in x_gt_np.shape)
            x_t_np = x_t_np[slc]
            x_h_np = x_h_np[slc]

        from skimage.metrics import structural_similarity as _ssim_3d

        def _mae(a, b): return float(np.abs(a - b).mean())
        def _mse(a, b): return float(((a - b) ** 2).mean())
        def _psnr(mse): return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
        def _ssim(a, b): return float(_ssim_3d(a, b, data_range=1.0))

        # Always available: hat vs MAISI ceiling
        mae_hat_vs_target = _mae(x_h_np, x_t_np)
        mse_hat_vs_target = _mse(x_h_np, x_t_np)
        psnr_hat_vs_target = _psnr(mse_hat_vs_target)
        ssim_hat_vs_target = _ssim(x_h_np, x_t_np)
        print(f"[recon] hat vs target:  MAE={mae_hat_vs_target:.6f}  "
              f"PSNR={psnr_hat_vs_target:.2f}dB  SSIM={ssim_hat_vs_target:.4f}")

        summary.update({
            "hat_vs_target": {
                "MAE": mae_hat_vs_target,
                "MSE": mse_hat_vs_target,
                "PSNR_dB": psnr_hat_vs_target,
                "SSIM": ssim_hat_vs_target,
            },
            "sw_latent_roi": list(latent_roi),
            "sw_overlap": sw_overlap,
        })

        # If GT available: also compare both reconstructions against GT.
        if x_gt_np is not None:
            mae_target_vs_gt = _mae(x_t_np, x_gt_np)
            mse_target_vs_gt = _mse(x_t_np, x_gt_np)
            psnr_target_vs_gt = _psnr(mse_target_vs_gt)
            ssim_target_vs_gt = _ssim(x_t_np, x_gt_np)

            mae_hat_vs_gt = _mae(x_h_np, x_gt_np)
            mse_hat_vs_gt = _mse(x_h_np, x_gt_np)
            psnr_hat_vs_gt = _psnr(mse_hat_vs_gt)
            ssim_hat_vs_gt = _ssim(x_h_np, x_gt_np)

            print(f"[recon] target vs gt :  MAE={mae_target_vs_gt:.6f}  "
                  f"PSNR={psnr_target_vs_gt:.2f}dB  SSIM={ssim_target_vs_gt:.4f}")
            print(f"[recon] hat    vs gt :  MAE={mae_hat_vs_gt:.6f}  "
                  f"PSNR={psnr_hat_vs_gt:.2f}dB  SSIM={ssim_hat_vs_gt:.4f}")

            summary.update({
                "target_vs_gt": {
                    "MAE": mae_target_vs_gt,
                    "MSE": mse_target_vs_gt,
                    "PSNR_dB": psnr_target_vs_gt,
                    "SSIM": ssim_target_vs_gt,
                },
                "hat_vs_gt": {
                    "MAE": mae_hat_vs_gt,
                    "MSE": mse_hat_vs_gt,
                    "PSNR_dB": psnr_hat_vs_gt,
                    "SSIM": ssim_hat_vs_gt,
                },
            })

        # Save volumes (and err vs gt if available, otherwise vs target).
        np.save(out_dir / "x_target.npy", x_t_np)
        np.save(out_dir / "x_hat.npy", x_h_np)
        if x_gt_np is not None:
            np.save(out_dir / "x_gt.npy", x_gt_np)
            np.save(out_dir / "err_target_vs_gt.npy", np.abs(x_t_np - x_gt_np))
            np.save(out_dir / "err_hat_vs_gt.npy", np.abs(x_h_np - x_gt_np))
        np.save(out_dir / "err_hat_vs_target.npy", np.abs(x_h_np - x_t_np))

        try:
            import nibabel as nib
            affine = np.eye(4)
            nib.save(nib.Nifti1Image(x_t_np.astype(np.float32), affine),
                     str(out_dir / "x_target.nii.gz"))
            nib.save(nib.Nifti1Image(x_h_np.astype(np.float32), affine),
                     str(out_dir / "x_hat.nii.gz"))
            nib.save(nib.Nifti1Image(np.abs(x_h_np - x_t_np).astype(np.float32), affine),
                     str(out_dir / "err_hat_vs_target.nii.gz"))
            if x_gt_np is not None:
                nib.save(nib.Nifti1Image(x_gt_np.astype(np.float32), affine),
                         str(out_dir / "x_gt.nii.gz"))
                nib.save(nib.Nifti1Image(np.abs(x_t_np - x_gt_np).astype(np.float32), affine),
                         str(out_dir / "err_target_vs_gt.nii.gz"))
                nib.save(nib.Nifti1Image(np.abs(x_h_np - x_gt_np).astype(np.float32), affine),
                         str(out_dir / "err_hat_vs_gt.nii.gz"))
        except ImportError:
            pass

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[recon] saved to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
