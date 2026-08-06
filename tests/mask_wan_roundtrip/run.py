"""Input mask vs WanVAE-recon montage for report2ct_wan_mask, from the REAL trained mask latents.

Sanity check that the frozen Wan VAE preserves the painted 4-organ geometry that report2ct_wan_mask
conditions on. For each id in `data/report2ct_wan/mask_latents_512x512x253`:
  INPUT = painted 4-organ GT mask (recomputed exactly as scripts/precompute_wan_mask_latents.py):
          {bg:-1000, lung:-500, heart:0, aorta:+500, esoph:+1000} HU, shape (X,Y,Z)
  RECON = WanVAE.decode(<id>_mask_emb.nii.gz) -> (X,Y,Z) HU
  quantize RECON -> nearest of the 5 painted levels -> label {0..4}
  montage: [INPUT label | RECON HU | RECON label] at 3 axial levels + per-organ Dice -> png/<id>.png

Runs in the `wan` conda env (WanVAE needs diffusers >= 0.34), GPU. Decode-only, ~few s/vol:
  CUDA_VISIBLE_DEVICES=2 HF_HOME=/workspace/data/checkpoints/hf_cache \
    /opt/conda/envs/wan/bin/python tests/mask_wan_roundtrip/run.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
import torch
from einops import rearrange

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

import rootutils  # noqa: E402

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.baselines.wan_vae import WanVAE  # noqa: E402
from src.data.organ_groups import GROUP_NAMES, apply_grouping  # noqa: E402
import scripts.precompute_wan_mask_latents as P  # noqa: E402

LAT_DIR = Path("/workspace/data/report2ct_wan/mask_latents_512x512x253")
OUT = Path(__file__).parent / "png"
IDS = ["valid_1000_a_1", "valid_1001_a_1", "valid_1002_a_1", "valid_1005_a_1"]
LEVELS = np.array(
    [P.HU_MIN + P.HU_STEP * g for g in range(5)]
)  # [-1000,-500,0,500,1000]
DEVICE = "cuda:0"
# bg, lung, heart, aorta, esophagus
CMAP = ListedColormap(["black", "orange", "green", "red", "deepskyblue"])


def painted_gt(vid: str) -> np.ndarray:
    """Painted 4-organ GT mask (X,Y,Z) HU — identical to the precompute pre-encode step."""
    tf = P.build_resample_transforms(512, 253)
    mp = P.ts_mask_path(vid, Path(P.DEFAULT_TS_DIR), "valid_fixed")
    m = tf({"mask": str(mp)})["mask"].numpy().squeeze()
    return P.HU_MIN + P.HU_STEP * apply_grouping(np.rint(m).astype(np.int16)).astype(
        np.float32
    )


def decode(vid: str, vae: WanVAE) -> np.ndarray:
    """Decode <id>_mask_emb.nii.gz -> (X,Y,Z) HU."""
    hwdc = nib.load(
        str(LAT_DIR / f"{vid}_mask_emb.nii.gz")
    ).get_fdata()  # (64,64,64,16)
    z = torch.from_numpy(rearrange(hwdc, "h w d c -> c d h w")).float()  # (16,64,64,64)
    return np.ascontiguousarray(vae.decode(z).numpy().transpose(1, 2, 0))  # (X,Y,Z)


def to_label(hu: np.ndarray) -> np.ndarray:
    return np.abs(hu[..., None] - LEVELS).argmin(-1).astype(np.int8)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / denom) if denom else float("nan")


def main() -> None:
    vae = WanVAE(device=DEVICE)
    OUT.mkdir(parents=True, exist_ok=True)
    rot = lambda s: np.flipud(np.rot90(s, 1))  # noqa: E731
    for vid in IDS:
        if not (LAT_DIR / f"{vid}_mask_emb.nii.gz").is_file():
            print(f"skip {vid} (no latent)")
            continue
        gt_l, rec_hu = to_label(painted_gt(vid)), decode(vid, vae)
        rec_l = to_label(rec_hu)
        dices = {GROUP_NAMES[g]: dice(gt_l == g, rec_l == g) for g in (1, 2, 3, 4)}
        fracs = (0.35, 0.5, 0.65)
        fig, ax = plt.subplots(len(fracs), 3, figsize=(9, 9))
        for r, fr in enumerate(fracs):
            z = int(fr * (gt_l.shape[2] - 1))
            ax[r, 0].imshow(rot(gt_l[:, :, z]), cmap=CMAP, vmin=0, vmax=4)
            ax[r, 1].imshow(rot(rec_hu[:, :, z]), cmap="gray", vmin=-1000, vmax=1000)
            ax[r, 2].imshow(rot(rec_l[:, :, z]), cmap=CMAP, vmin=0, vmax=4)
            for c in range(3):
                ax[r, c].axis("off")
        ax[0, 0].set_title("INPUT mask (4-organ)", fontsize=11)
        ax[0, 1].set_title("WanVAE RECON (HU)", fontsize=11)
        ax[0, 2].set_title("RECON label", fontsize=11)
        fig.suptitle(
            f"{vid}   Dice: " + "  ".join(f"{k}={v:.3f}" for k, v in dices.items()),
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(OUT / f"{vid}.png", dpi=95, bbox_inches="tight")
        plt.close(fig)
        print(f"{vid}: " + "  ".join(f"{k}={v:.3f}" for k, v in dices.items()))
    print(f"PNGs -> {OUT}")


if __name__ == "__main__":
    main()
