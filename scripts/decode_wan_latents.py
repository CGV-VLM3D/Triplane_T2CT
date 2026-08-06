"""Decode report2ct_wan latents (.npy) → HU .mha predictions — WAN conda env only.

Step 2 of the 3-step Wan eval flow (see docs/wan_latent_runbook.md): reads the Wan latents
written by scripts/generate_wan_latents.py and decodes each through the Wan2.1 VAE into a HU
volume saved as <out>/predictions/<scan_id>.mha (the layout scripts/run_eval.py expects).

Runs in the `wan` env (diffusers ≥ 0.34), GPU 1:
  CUDA_VISIBLE_DEVICES=1 /opt/conda/envs/wan/bin/python scripts/decode_wan_latents.py \\
      --latent-dir /workspace/data/vlm3d_eval/ctgen/report2ct_wan/latents \\
      --out /workspace/data/vlm3d_eval/ctgen/report2ct_wan/predictions --spacing 0.73 0.73 1.34

Axis chain (baseline-clone #4): WanVAE.decode returns (T=Z, H=X, W=Y); SimpleITK wants (Z, Y, X),
so we permute (0,2,1). Spacing stamped [0.73,0.73,1.34] MUST MATCH what generate_wan_latents.py
conditioned the UNet with (a spacing-conditional model generates at the spacing it is told).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.baselines.wan_vae import DEFAULT_WAN_HF_ID, WanVAE  # noqa: E402
from src.eval.samplers._orient import ras_to_lps  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("decode_wan_latents")


def _save_mha(hu_zxy: np.ndarray, spacing_mm: list[float], out_path: Path) -> None:
    """Save a decoded HU volume (Z, X, Y) as .mha. SimpleITK wants (Z, Y, X) → permute (0,2,1)."""
    arr_zyx = np.ascontiguousarray(hu_zxy.transpose(0, 2, 1))  # (Z, Y, X)
    # WanVAE decodes RAS content (Orientationd(RAS) precompute); flip in-plane to LPS so it
    # matches the LPS GT / real CT-RATE for eval + submission (plan: eval-wise-catmull).
    arr_zyx = ras_to_lps(arr_zyx)  # (Z, Y, X), X/Y reversed
    img = sitk.GetImageFromArray(arr_zyx)
    img.SetSpacing([float(s) for s in spacing_mm])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path))


def main() -> None:
    ap = argparse.ArgumentParser(description="Decode Wan latents → HU .mha (wan env)")
    ap.add_argument(
        "--latent-dir", required=True, help="dir of <scan_id>.npy Wan latents"
    )
    ap.add_argument(
        "--out", required=True, help="output dir for <scan_id>.mha predictions"
    )
    ap.add_argument(
        "--hf-id", default=DEFAULT_WAN_HF_ID, help="Wan VAE HuggingFace repo id"
    )
    ap.add_argument(
        "--device", default="cuda:0", help="Torch device (cuda:0 under CVD=2)"
    )
    ap.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        required=True,
        help="spacing (mm) for the .mha affine, REQUIRED (no default); MUST MATCH the "
        "--spacing passed to generate_wan_latents.py (spacing-conditional model).",
    )
    args = ap.parse_args()

    latent_dir = Path(args.latent_dir)
    out_dir = Path(args.out)
    npys = sorted(latent_dir.glob("*.npy"))
    if not npys:
        log.error("No .npy latents in %s", latent_dir)
        return

    log.info("Loading Wan VAE %s on %s …", args.hf_id, args.device)
    vae = WanVAE(hf_id=args.hf_id, device=args.device)

    processed = skipped = 0
    for npy in tqdm(npys, desc="Wan decode"):
        out_path = out_dir / f"{npy.stem}.mha"
        if out_path.exists():
            skipped += 1
            continue
        z = torch.from_numpy(np.load(npy))  # (16, 64, 64, 64) = (C, T_lat, H, W)
        hu = vae.decode(z).numpy()  # (T=Z, H=X, W=Y) HU float
        _save_mha(hu.astype(np.int16), list(args.spacing), out_path)
        processed += 1

    log.info("Done. decoded=%d, skipped=%d → %s", processed, skipped, out_dir)


if __name__ == "__main__":
    main()
