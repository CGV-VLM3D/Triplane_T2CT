"""Stage 2 (`wan` conda env): decode Wan latents for the selected patients -> per-patient npz.

Reads manifest.json (written by stage 1), decodes each scan's stored Wan latent through the
Wan2.1 VAE (that IS the Wan encode->decode round-trip; no scale_factor), and extracts the
SAME Z_FRACTIONS axial slices (oriented, 512x512, HU) as stage 1.

Run (wan env, GPU):
  CUDA_VISIBLE_DEVICES=0 HF_HOME=/workspace/data/checkpoints/hf_cache \
    /opt/conda/envs/wan/bin/python tests/vae_recon_compare/run_wan.py
"""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import torch
from einops import rearrange

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.baselines.wan_vae import WanVAE  # noqa: E402
from tests.vae_recon_compare import common as C  # noqa: E402

DEVICE = "cuda:0"


def wan_recon_hu(sid: str, vae: WanVAE) -> np.ndarray:
    """Decode the stored Wan latent -> (X,Y,Z) HU (RAS), matching stage-1's GT/MAISI frame."""
    hwdc = nib.load(
        str(C.WAN_LAT_DIR / f"{sid}_emb.nii.gz")
    ).get_fdata()  # (64,64,64,16)=HWDC
    z = torch.from_numpy(
        rearrange(hwdc, "h w d c -> c d h w")
    ).float()  # (16,T_lat=64,64,64)
    hu_zxy = vae.decode(z).numpy()  # (T=Z=253, H=X=512, W=Y=512) HU
    return np.ascontiguousarray(hu_zxy.transpose(1, 2, 0))  # -> (X, Y, Z)


def main() -> None:
    patients = json.loads((C.OUT_DIR / "manifest.json").read_text())
    vae = WanVAE(device=DEVICE)
    for p in patients:
        sid = p["id"]
        out_npz = C.OUT_DIR / f"{sid}_wan.npz"
        if out_npz.exists():
            print(f"  skip {sid} (exists)")
            continue
        wan = C.slices_hu(wan_recon_hu(sid, vae))  # (n_frac,512,512) HU
        np.savez_compressed(out_npz, wan=wan)
        print(f"  saved {sid}: wan{wan.shape}")


if __name__ == "__main__":
    main()
