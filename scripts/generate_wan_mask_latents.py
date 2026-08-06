"""Generate report2ct_wan_mask predictions as Wan latents (.npy) on valid_v2 — MAIN env.

Twin of `scripts/generate_wan_latents.py` for the mask-conditioned variant: each RFlow step
concatenates the per-case precomputed Wan mask latent at the UNet input (32-ch UNet). Step 1 of
the 3-step Wan eval flow (see docs/wan_latent_runbook.md):
  1. THIS (main env): RFlow-denoise → <out>/latents/<scan_id>.npy  (16-ch image latent; the mask
     is conditioning only, never saved/predicted)
  2. scripts/decode_wan_latents.py (wan env): latents → <out>/predictions/<scan_id>.mha
  3. scripts/run_eval.py model=report2ct_wan_mask out_dir=<out>: score the pre-decoded .mha

Conditioning is report2ct's 2560-d findings+impression (encoded live by the sampler) PLUS the
GT 4-organ Wan mask latent from --mask-dir (produced by scripts/precompute_wan_mask_latents.py).

Resumable — existing .npy are skipped. GPU 1:
  CUDA_VISIBLE_DEVICES=1 python scripts/generate_wan_mask_latents.py \\
      --ckpt /workspace/outputs/report2ct_wan_mask/<ts>/checkpoints/epoch_099.ckpt \\
      --out /workspace/data/vlm3d_eval/ctgen/report2ct_wan_mask \\
      --mask-dir /workspace/data/report2ct_wan/mask_latents_512x512x253 \\
      --n 1304 --spacing 0.73 0.73 1.34
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch  # noqa: E402

from src.eval.ct_rate_cases import load_eval_cases  # noqa: E402
from src.eval.samplers.report2ct_wan import Report2CTWanMaskLatentSampler  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_wan_mask_latents")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="report2ct_wan_mask latent generation (valid_v2)"
    )
    ap.add_argument(
        "--ckpt", required=True, help="Lightning .ckpt (experiment=report2ct_wan_mask)"
    )
    ap.add_argument(
        "--out",
        default="/workspace/data/vlm3d_eval/ctgen/report2ct_wan_mask",
        help="eval out dir; latents land in <out>/latents/",
    )
    ap.add_argument(
        "--mask-dir",
        required=True,
        help="dir of <vol_id>_mask_emb.nii.gz Wan mask latents (precompute --out-dir)",
    )
    ap.add_argument(
        "--n", type=int, default=1304, help="number of valid_v2 cases (head)"
    )
    ap.add_argument("--n-steps", type=int, default=100, help="RFlow sampling steps")
    ap.add_argument(
        "--cfg-scale",
        type=float,
        required=True,
        help="TEXT CFG scale, REQUIRED (no default; report2ct eval uses 5.0, 1.0 = off). Mask is always on.",
    )
    ap.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        required=True,
        help="spacing (mm) stamped for UNet conditioning, REQUIRED (no default); "
        "sweep {0.70,0.73,0.80} post-hoc.",
    )
    args = ap.parse_args()

    latent_dir = Path(args.out) / "latents"
    cases = load_eval_cases(n_samples=args.n)
    done = len(list(latent_dir.glob("*.npy"))) if latent_dir.is_dir() else 0
    log.info("Loaded %d cases; %d latents already in %s", len(cases), done, latent_dir)

    sampler = Report2CTWanMaskLatentSampler(
        ckpt_path=args.ckpt,
        mask_dir=args.mask_dir,
        n_steps=args.n_steps,
        cfg_scale=args.cfg_scale,
        spacing_mm=list(args.spacing),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    written = sampler.generate(cases, latent_dir, device)
    log.info("Done: %d/%d latents in %s", len(written), len(cases), latent_dir)


if __name__ == "__main__":
    main()
