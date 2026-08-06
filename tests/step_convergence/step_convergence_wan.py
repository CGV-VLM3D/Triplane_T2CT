"""RFlow sampling-step convergence test for report2ct_wan — latent generation (MAIN env).

Question (user): report2ct (MAISI) showed no convergence knee — its output keeps moving
through 100 steps. Does report2ct_wan behave the same? The n=150 metric calibration
(`data/vlm3d_eval/ctgen/wan_stepcal/`) hints it does NOT: FID/CLIP/FVD are flat between
50 and 100 steps. This is the per-sample version of that question.

Method mirrors `step_convergence.py` exactly (same 4 cases via `load_eval_cases(n_samples=4)`,
same per-case seeds, same 250-step reference), with two forced differences:
  * the sampler is `Report2CTWanLatentSampler`, so the latent is (1,16,64,64,64), not MAISI's
    (1,4,120,120,64) — latent rel-L2 is therefore comparable ACROSS STEPS but not across models;
  * Wan decode lives in a separate conda env, so this script stops at the latent and writes
    one `.npy` per (case, n_steps). Decode + scoring are steps 2 and 3 of the flow:
      2. `CUDA_VISIBLE_DEVICES=N /opt/conda/envs/wan/bin/python scripts/decode_wan_latents.py \
             --latent-dir <OUT>/latents --out <OUT>/predictions --spacing 0.75 0.75 1.3`
      3. `python tests/step_convergence/step_convergence_wan_report.py`

Regime = the production/stepcal regime (epoch_299, spacing 0.75/0.75/1.3, cfg 5.0) so the
numbers sit next to the eval dirs already scored at those settings.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from src.eval.ct_rate_cases import load_eval_cases
from src.eval.samplers.report2ct_wan import Report2CTWanLatentSampler

CKPT = "/workspace/outputs/report2ct_wan/2026-07-16_3/checkpoints/epoch_299.ckpt"
STEPS = [30, 50, 75, 100, 250]
REF = 250  # high-step reference (~converged ODE solution)
N_CASES = 4
SPACING = [0.75, 0.75, 1.3]
CFG = 5.0
OUT = Path("/workspace/data/vlm3d_eval/ctgen/wan_stepconv")

DEVICE = torch.device("cuda")


def main() -> None:
    cases = load_eval_cases(n_samples=N_CASES)
    print(f"Loaded {len(cases)} cases: {[c.scan_id for c in cases]}", flush=True)

    latent_dir = OUT / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)

    sampler = Report2CTWanLatentSampler(
        ckpt_path=CKPT, spacing_mm=SPACING, cfg_scale=CFG, n_steps=100
    )
    sampler._init(DEVICE)
    spacing_tensor = sampler._make_spacing_tensor(sampler.spacing_mm)  # (1, 3)

    manifest = []
    for ci, case in enumerate(cases):
        seed = (
            1000 + ci
        )  # per-case seed; identical z_T across all n_steps for this case
        context = sampler._case_to_context(case)  # (1, 2, 2560)

        for ns in STEPS:
            out_path = latent_dir / f"{case.scan_id}__s{ns:03d}.npy"
            if out_path.exists():
                print(f"  [{case.scan_id}] steps={ns:3d}  skip (exists)", flush=True)
                continue
            sampler.n_steps = ns
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            t0 = time.time()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z = sampler._denoise(context, spacing_tensor)  # (1, 16, 64, 64, 64)
            # UNet (C,H,W,D) -> Wan (C,T_lat,H,W) for decode  (report2ct_wan.py:281)
            z_wan = z.squeeze(0).permute(0, 3, 1, 2).contiguous()  # (16, 64, 64, 64)
            np.save(out_path, z_wan.cpu().float().numpy())
            print(
                f"  [{case.scan_id}] steps={ns:3d}  {time.time() - t0:5.1f}s  "
                f"latent std={z_wan.std().item():.4f}",
                flush=True,
            )
        manifest.append({"scan_id": case.scan_id, "seed": seed})

    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "ckpt": CKPT,
                "steps": STEPS,
                "ref": REF,
                "spacing": SPACING,
                "cfg": CFG,
                "cases": manifest,
            },
            indent=2,
        )
    )
    print(f"\nwrote {len(list(latent_dir.glob('*.npy')))} latents → {latent_dir}")
    print(
        f"next: decode in the wan env, then step_convergence_wan_report.py", flush=True
    )


if __name__ == "__main__":
    main()
