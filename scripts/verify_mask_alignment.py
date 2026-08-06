"""Alignment gate (baseline-clone #4): is the precomputed top-K mask voxel-aligned with the
text2ct latent it will be concatenated to?

Decode a real toy_v2 latent through the text2ct VAE -> CT (512,512,128) HU -> avg-pool to the
latent grid (128,128,32). Then test whether the mask's dominant class per cell PREDICTS the cell
mean HU (class-map-free eta^2 / one-way ANOVA): if the mask is aligned, lung cells are very low HU
and bone cells very high, so class explains HU strongly. Compare eta^2 for the correct arrangement
vs axis flips/transposes — the correct (identity) arrangement must win, else there is an axis bug.

Run: CUDA_VISIBLE_DEVICES=1 python scripts/verify_mask_alignment.py
"""

from __future__ import annotations

import glob
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda:0"
EMB = "/workspace/third_party/text2ct/embeddings/toy_v2"
MASK_DIR = (
    "/tmp/claude-0/-workspace/ad35a113-c96a-4537-83c8-9173982d26e5/scratchpad/mask_test"
)


def eta2(dom: torch.Tensor, hu: torch.Tensor) -> float:
    """Fraction of HU variance explained by the dominant-class label (one-way ANOVA eta^2)."""
    dom, hu = dom.reshape(-1), hu.reshape(-1)
    total = ((hu - hu.mean()) ** 2).sum()
    within = torch.zeros((), device=hu.device)
    for c in torch.unique(dom):
        h = hu[dom == c]
        within = within + ((h - h.mean()) ** 2).sum()
    return float(1 - within / total)


def main() -> None:
    from src.baselines.text2ct_adapter import Text2CTAdapter

    adapter = Text2CTAdapter(device_str="cuda:0", load_weights=True)
    adapter._ensure_built()
    vae = adapter._autoencoder

    vols = [
        os.path.basename(p)[:-3]
        for p in sorted(glob.glob(os.path.join(MASK_DIR, "*.pt")))
    ]
    print(f"checking {len(vols)} vols")

    for vol in vols:
        z = nib.load(
            os.path.join(EMB, f"{vol}.nii.gz")
        ).get_fdata()  # (128,128,32,4) HWDC
        z = (
            torch.from_numpy(np.ascontiguousarray(z.transpose(3, 0, 1, 2)))[None]
            .float()
            .to(DEV)
        )
        # on-disk latent is RAW encoder output -> decode directly (no scale_factor; A6)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            ct = vae.decode_stage_2_outputs(z)  # (1,1,512,512,128) ~[0,1]
        hu = (ct.float() * 2000 - 1000).clamp(-1000, 1000)  # (1,1,512,512,128) HU
        ct_cell = F.avg_pool3d(hu, kernel_size=4)[
            0, 0
        ]  # (128,128,32) mean HU per latent cell

        md = torch.load(os.path.join(MASK_DIR, f"{vol}.pt"))
        dom = md["classes"][0].long().to(DEV)  # (128,128,32) top-1 dominant class

        # eta^2 for correct arrangement vs axis flips/transposes (flip the CT, keep mask fixed)
        variants = {
            "identity": ct_cell,
            "flipX": ct_cell.flip(0),
            "flipY": ct_cell.flip(1),
            "flipZ": ct_cell.flip(2),
            "swapXY": ct_cell.permute(1, 0, 2),
        }
        scores = {k: eta2(dom, v) for k, v in variants.items()}
        best = max(scores, key=scores.get)
        print(
            f"\n{vol}: eta2 "
            + " ".join(f"{k}={scores[k]:.3f}" for k in scores)
            + f"  -> BEST={best}"
        )

        # eyeball anatomy: mean HU of the most common dominant classes (expect lung<<0, bone>>0)
        uq, cnt = dom.reshape(-1).unique(return_counts=True)
        top = uq[cnt.argsort(descending=True)[:8]]
        pairs = [
            (int(c), round(float(ct_cell[dom == c].mean()), 0), int((dom == c).sum()))
            for c in top
        ]
        print("   top dominant classes (id, meanHU, ncells):", pairs)


if __name__ == "__main__":
    main()
