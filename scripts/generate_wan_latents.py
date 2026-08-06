"""Generate report2ct_wan predictions as Wan latents (.npy) on the valid_v2 split — MAIN env.

Step 1 of the 3-step Wan eval flow (see docs/wan_latent_runbook.md):
  1. THIS (main env): RFlow-denoise a Wan latent per valid_v2 case → <out>/latents/<scan_id>.npy
  2. scripts/decode_wan_latents.py (wan env): latents → <out>/predictions/<scan_id>.mha
  3. scripts/run_eval.py model=report2ct_wan out_dir=<out>: score the pre-decoded .mha

Conditioning is report2ct's 2560-d findings+impression (encoded live from each EvalCase by the
sampler's Report2CTTextEncoder) — no precomputed cond dir needed.

Resumable — existing .npy are skipped. GPU 1:
  CUDA_VISIBLE_DEVICES=1 python scripts/generate_wan_latents.py \\
      --ckpt /workspace/outputs/report2ct_wan/<ts>/checkpoints/epoch_099.ckpt \\
      --out /workspace/data/vlm3d_eval/ctgen/report2ct_wan --n 1304 --spacing 0.73 0.73 1.34
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch  # noqa: E402

from src.eval.ct_rate_cases import load_eval_cases  # noqa: E402
from src.eval.samplers.report2ct_wan import Report2CTWanLatentSampler  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_wan_latents")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="report2ct_wan latent generation (valid_v2)"
    )
    ap.add_argument(
        "--ckpt", required=True, help="Lightning .ckpt (experiment=report2ct_wan)"
    )
    ap.add_argument(
        "--out",
        default="/workspace/data/vlm3d_eval/ctgen/report2ct_wan",
        help="eval out dir; latents land in <out>/latents/",
    )
    ap.add_argument(
        "--n", type=int, default=1304, help="number of valid_v2 cases (head)"
    )
    ap.add_argument("--n-steps", type=int, default=100, help="RFlow sampling steps")
    ap.add_argument(
        "--cfg-scale",
        type=float,
        required=True,
        help="CFG scale, REQUIRED (no default; report2ct eval uses 5.0, 1.0 = off)",
    )
    ap.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        required=True,
        help="spacing (mm) stamped for UNet conditioning, REQUIRED (no default); "
        "n=300 joint (in-plane,z) spacing_fov sweep (tests/spacing_fov/RESULTS.md §9, "
        "epoch_299) found (0.75,0.75,1.3) optimal: FID 1.445, CLIP-T2I 66.07 — near-best "
        "on both metrics, no real trade needed. Supersedes the earlier 0.73/0.73/1.34 "
        "coarse-sweep estimate.",
    )
    args = ap.parse_args()

    latent_dir = Path(args.out) / "latents"
    cases = load_eval_cases(n_samples=args.n)
    done = len(list(latent_dir.glob("*.npy"))) if latent_dir.is_dir() else 0
    log.info("Loaded %d cases; %d latents already in %s", len(cases), done, latent_dir)

    sampler = Report2CTWanLatentSampler(
        ckpt_path=args.ckpt,
        n_steps=args.n_steps,
        cfg_scale=args.cfg_scale,
        spacing_mm=list(args.spacing),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    written = sampler.generate(cases, latent_dir, device)
    log.info("Done: %d/%d latents in %s", len(written), len(cases), latent_dir)


if __name__ == "__main__":
    main()
