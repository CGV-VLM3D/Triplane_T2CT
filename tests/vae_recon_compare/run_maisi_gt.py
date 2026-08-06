"""Stage 1 (main env): select patients, decode MAISI latents, load GT -> per-patient npz.

Selects the 5 most detail-heavy abnormal valid_v2 scans, writes manifest.json, then for
each scan:
  * GT: load native _fixed CT (RAS, no resize) -> (X,Y,Z) HU.
  * MAISI recon: decode the stored _emb latent through the frozen MAISI VAE (tiled decode,
    scale_factor=1 => pure round-trip) -> (480,480,256) [0,1] -> HU.
Extracts the Z_FRACTIONS axial slices (oriented, 512x512, HU) and saves gt/maisi to npz.

Run:
  CUDA_VISIBLE_DEVICES=0 python tests/vae_recon_compare/run_maisi_gt.py
"""

from __future__ import annotations

import json

import monai
import nibabel as nib
import numpy as np
import torch
from monai.inferers.inferer import SlidingWindowInferer
from monai.transforms import Compose
from torch.amp import autocast

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.baselines.maisi import load_frozen  # noqa: E402
from src.eval.samplers.report2ct import _ReconModel, _dynamic_infer  # noqa: E402
from tests.vae_recon_compare import common as C  # noqa: E402

DEVICE = "cuda:0"


def load_gt_hu(sid: str) -> np.ndarray:
    """Load a native _fixed CT to (X,Y,Z) HU in RAS (matches the recon frame, no resize)."""
    tf = Compose(
        [
            monai.transforms.LoadImaged(keys="image"),
            monai.transforms.EnsureChannelFirstd(keys="image"),
            monai.transforms.Orientationd(keys="image", axcodes="RAS"),
            monai.transforms.EnsureTyped(keys="image", dtype=torch.float32),
        ]
    )
    out = tf({"image": str(C.gt_path(sid))})
    return out["image"].numpy().squeeze()  # (X, Y, Z) HU


def maisi_recon_hu(
    sid: str, recon: _ReconModel, inferer: SlidingWindowInferer
) -> np.ndarray:
    """Decode the stored MAISI latent -> (X,Y,Z) HU via the tiled eval decode path."""
    hwdc = nib.load(
        str(C.MAISI_LAT_DIR / f"{sid}_emb.nii.gz")
    ).get_fdata()  # (120,120,64,4)
    z = torch.from_numpy(hwdc).permute(3, 0, 1, 2).contiguous()  # (C=4,H,W,D)
    z = z.float()[None].to(DEVICE)  # (1,4,120,120,64)
    with torch.inference_mode(), autocast("cuda", dtype=torch.float16):
        vol = _dynamic_infer(inferer, recon, z)  # (1,1,480,480,256) ~[0,1]
    v01 = vol.squeeze().float().cpu().numpy()  # (480,480,256) = (X,Y,Z)
    return (
        v01 * 2000.0 - 1000.0
    )  # [0,1] -> HU (inverse of ScaleIntensityRange[-1000,1000])


def main() -> None:
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    patients = C.select_detail_patients(n=5) + C.select_healthy_patients(n=1)
    (C.OUT_DIR / "manifest.json").write_text(json.dumps(patients, indent=2))
    print("Selected patients:")
    for p in patients:
        print(
            f"  [{p['cohort']:8s}] {p['id']}  detail={p['detail_score']} total={p['total']}  {p['positive']}"
        )

    vae = load_frozen(device=DEVICE)
    recon = _ReconModel(vae, scale_factor=1.0).to(DEVICE)  # sf=1 => pure encode->decode
    inferer = SlidingWindowInferer(
        roi_size=(80, 80, 80), sw_batch_size=1, overlap=0.4, progress=False
    )

    for p in patients:
        sid = p["id"]
        out_npz = C.OUT_DIR / f"{sid}_maisi_gt.npz"
        if out_npz.exists():
            print(f"  skip {sid} (exists)")
            continue
        gt = C.slices_hu(load_gt_hu(sid))  # (n_frac,512,512) HU
        maisi = C.slices_hu(maisi_recon_hu(sid, recon, inferer))  # (n_frac,512,512) HU
        np.savez_compressed(out_npz, gt=gt, maisi=maisi)
        print(f"  saved {sid}: gt{gt.shape} maisi{maisi.shape}")


if __name__ == "__main__":
    main()
