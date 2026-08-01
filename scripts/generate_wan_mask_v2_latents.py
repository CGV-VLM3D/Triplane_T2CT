"""Generate report2ct_wan_mask_v2 predictions as Wan latents (.npy) on valid_v2 — MAIN env.

Twin of `scripts/generate_wan_mask_latents.py` for the dual-condition-CFG variant. Each RFlow step
runs the text-inner / mask-outer 3-pass CFG (Report2CTWanMaskV2LatentSampler):
    pred = e(∅m,∅t) + s_t·[e(∅m,t)−e(∅m,∅t)] + s_m·[e(m,t)−e(∅m,t)]
where ∅m is the LEARNED null mask embedding from the ckpt. **s_m is a clean mask-effect dial**:
s_m=0 ignores the mask (pure text→CT), s_m=1 uses it naturally, s_m>1 amplifies organ geometry.

Step 1 of the 3-step Wan eval flow (see docs/wan_latent_runbook.md):
  1. THIS (main env): RFlow-denoise → <out>/latents/<scan_id>.npy  (16-ch image latent; the mask is
     conditioning only, never saved/predicted)
  2. scripts/decode_wan_latents.py (wan env): latents → <out>/predictions/<scan_id>.mha
  3. scripts/run_eval.py model=report2ct_wan_mask_v2 out_dir=<out>: score the pre-decoded .mha

Both --cfg-scale-text and --cfg-scale-mask are REQUIRED (no defaults) — the loud-knob convention.
Sweep s_m ∈ {0,1,1.5,2} at s_t=5 to quantify the mask's effect. Encode BOTH scales in the eval dir
name: eval_ep<NNN>_sp<in>_<z>_cfgt<T>_m<M>  (e.g. eval_ep099_sp0.73_1.34_cfgt5_m1.5).

Mask-intervention mode (`--manifest`, docs/mask_intervention_manifest.md): one latent per manifest
ROW instead of one per scan — `<sample_id>.npy`, text from the TARGET's report, conditioning mask
from that row's `cond_mask_source_id` (`null` -> the learned `no_mask_embed`), and the initial
noise pinned per (target, seed) so every condition of a target starts from the same noise. The run
is DIAGNOSTIC-ONLY (CLIPScore-T2I / Dice); `run_eval.py` refuses FID/FVD when `task.manifest` is
set. `--manifest` and `--n` are mutually exclusive.

Resumable — existing .npy are skipped. ⚠ Because existing files are skipped, a run whose settings
changed (spacing / cfg / ckpt / manifest) MUST use a fresh --out, or it silently keeps the old
volumes. GPU 1:
  CUDA_VISIBLE_DEVICES=1 python scripts/generate_wan_mask_v2_latents.py \\
      --ckpt /workspace/outputs/report2ct_wan_mask_v2/<ts>/checkpoints/epoch_099.ckpt \\
      --out /workspace/data/vlm3d_eval/ctgen/report2ct_wan_mask_v2 \\
      --mask-dir /workspace/data/report2ct_wan/mask_latents_512x512x253 \\
      --n 1304 --cfg-scale-text 5 --cfg-scale-mask 1.0 --spacing 0.73 0.73 1.34
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch  # noqa: E402

from src.eval.ct_rate_cases import load_eval_cases, load_manifest_cases  # noqa: E402
from src.eval.manifest import (  # noqa: E402
    check_generation_provenance,
    read_manifest_rows,
)
from src.eval.samplers.report2ct_wan import Report2CTWanMaskV2LatentSampler  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_wan_mask_v2_latents")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="report2ct_wan_mask_v2 latent generation (valid_v2, dual-condition CFG)"
    )
    ap.add_argument(
        "--ckpt",
        required=True,
        help="Lightning .ckpt (experiment=report2ct_wan_mask_v2; must have no_mask_embed)",
    )
    ap.add_argument(
        "--out",
        default="/workspace/data/vlm3d_eval/ctgen/report2ct_wan_mask_v2",
        help="eval out dir; latents land in <out>/latents/",
    )
    ap.add_argument(
        "--mask-dir",
        required=True,
        help="dir of <vol_id>_mask_emb.nii.gz Wan mask latents (precompute --out-dir)",
    )
    cases_from = ap.add_mutually_exclusive_group()
    cases_from.add_argument(
        "--n", type=int, default=1304, help="number of valid_v2 cases (head)"
    )
    cases_from.add_argument(
        "--manifest",
        default=None,
        help="mask-intervention JSONL (docs/mask_intervention_manifest.md): generate one latent "
        "PER ROW — <sample_id>.npy, text from the target's report, mask from the row's "
        "cond_mask_source_id (null = the learned no_mask_embed). Diagnostic-only run: "
        "run_eval.py refuses FID/FVD for it.",
    )
    ap.add_argument("--n-steps", type=int, default=100, help="RFlow sampling steps")
    ap.add_argument(
        "--cfg-scale-text",
        type=float,
        required=True,
        help="s_t: TEXT guidance scale, REQUIRED (no default; report2ct eval uses 5.0).",
    )
    ap.add_argument(
        "--cfg-scale-mask",
        type=float,
        required=True,
        help="s_m: MASK guidance scale, REQUIRED (no default; 0=mask off, 1=natural, >1=amplified).",
    )
    ap.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        required=True,
        help="spacing (mm) stamped for UNet conditioning, REQUIRED (no default); "
        "Wan best {0.73 0.73 1.34}.",
    )
    args = ap.parse_args()

    latent_dir = Path(args.out) / "latents"
    if args.manifest:
        # Provenance cross-check BEFORE any GPU time: sample_id encodes s_m, and the rows record
        # the ckpt + both scales, so a mismatched command would write files whose names claim
        # settings the volumes do not have.
        rows = read_manifest_rows(args.manifest)
        check_generation_provenance(
            rows, args.ckpt, args.cfg_scale_text, args.cfg_scale_mask
        )
        cases = load_manifest_cases(args.manifest)
    else:
        cases = load_eval_cases(n_samples=args.n)
    done = len(list(latent_dir.glob("*.npy"))) if latent_dir.is_dir() else 0
    log.info("Loaded %d cases; %d latents already in %s", len(cases), done, latent_dir)
    log.info(
        "Dual CFG: s_t=%.2f  s_m=%.2f  (s_m=0 → mask off / pure text→CT)",
        args.cfg_scale_text,
        args.cfg_scale_mask,
    )

    sampler = Report2CTWanMaskV2LatentSampler(
        ckpt_path=args.ckpt,
        mask_dir=args.mask_dir,
        cfg_scale_text=args.cfg_scale_text,
        cfg_scale_mask=args.cfg_scale_mask,
        n_steps=args.n_steps,
        spacing_mm=list(args.spacing),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    written = sampler.generate(cases, latent_dir, device)
    log.info("Done: %d/%d latents in %s", len(written), len(cases), latent_dir)


if __name__ == "__main__":
    main()
