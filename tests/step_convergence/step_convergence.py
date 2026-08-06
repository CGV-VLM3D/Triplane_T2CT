"""RFlow sampling-step convergence test for Report2CT.

Question: is n_steps=100 necessary, or does 30 (text2ct default) already converge?

Method: for a few valid_v2 cases, fix the initial noise (seed) so ONLY the RFlow
discretization changes, sweep n_steps, and measure how much the output moves as
steps increase. If 30 ≈ 250 (high PSNR/SSIM, tiny latent L2), 30 has converged and
a full 1304 metric eval at 30 would match 100.

Regime = real eval regime (spacing 0.8/0.8/1.5, cfg 5.0) — CFG stiffens the ODE
trajectory, so this is the conservative setting where few steps is most likely to hurt.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

from src.eval.ct_rate_cases import load_eval_cases
from src.eval.samplers.report2ct import Report2CTSampler

CKPT = (
    "/workspace/outputs/report2ct/report2ct_full_2026-06-30/checkpoints/epoch_074.ckpt"
)
STEPS = [30, 50, 75, 100, 250]
REF = 250  # high-step reference (~converged ODE solution)
N_CASES = 4
SPACING = [0.8, 0.8, 1.5]
CFG = 5.0
OUT = Path(__file__).parent / "step_convergence_results.json"

DEVICE = torch.device("cuda")


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 2000.0) -> float:
    """PSNR between two HU int16 volumes, data_range = 2000 HU ([-1000,1000])."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((data_range**2) / mse))


def main() -> None:
    cases = load_eval_cases(n_samples=N_CASES)
    print(f"Loaded {len(cases)} cases: {[c.scan_id for c in cases]}", flush=True)

    sampler = Report2CTSampler(
        ckpt_path=CKPT, spacing_mm=SPACING, cfg_scale=CFG, n_steps=100
    )
    sampler._init(DEVICE)
    spacing_tensor = sampler._make_spacing_tensor(sampler.spacing_mm)

    results = []
    for ci, case in enumerate(cases):
        seed = 1000 + ci  # per-case seed; identical z0 across all n_steps for this case
        context = sampler._case_to_context(case)

        latents: dict[int, torch.Tensor] = {}
        vols: dict[int, np.ndarray] = {}
        for ns in STEPS:
            sampler.n_steps = ns
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            t0 = time.time()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z = sampler._denoise(context, spacing_tensor)  # (1,4,120,120,64)
                hu = sampler._decode_to_hu(z)  # (480,480,256) int16
            latents[ns] = z.detach().float().cpu()
            vols[ns] = hu
            print(
                f"  [{case.scan_id}] steps={ns:3d}  {time.time() - t0:5.1f}s  "
                f"HU[{hu.min()},{hu.max()}]",
                flush=True,
            )

        z_ref = latents[REF]
        ref_norm = z_ref.norm().item()
        per_case = {"scan_id": case.scan_id, "seed": seed, "vs_ref": {}}
        for ns in STEPS:
            lat_rel_l2 = (latents[ns] - z_ref).norm().item() / ref_norm
            p = psnr(vols[ns], vols[REF])
            s = float(
                structural_similarity(
                    vols[ns].astype(np.float32),
                    vols[REF].astype(np.float32),
                    data_range=2000.0,
                )
            )
            per_case["vs_ref"][ns] = {
                "latent_rel_l2": round(lat_rel_l2, 4),
                "psnr": round(p, 2),
                "ssim": round(s, 4),
            }
        # explicit 30 vs 100
        p_30_100 = psnr(vols[30], vols[100])
        s_30_100 = float(
            structural_similarity(
                vols[30].astype(np.float32),
                vols[100].astype(np.float32),
                data_range=2000.0,
            )
        )
        l_30_100 = (latents[30] - latents[100]).norm().item() / latents[
            100
        ].norm().item()
        per_case["p30_vs_100"] = {
            "latent_rel_l2": round(l_30_100, 4),
            "psnr": round(p_30_100, 2),
            "ssim": round(s_30_100, 4),
        }
        results.append(per_case)
        print(
            f"  -> {case.scan_id} 30-vs-100: PSNR {p_30_100:.2f} SSIM {s_30_100:.4f} "
            f"latentL2 {l_30_100:.4f}",
            flush=True,
        )

    OUT.write_text(
        json.dumps(
            {
                "ckpt": CKPT,
                "steps": STEPS,
                "ref": REF,
                "spacing": SPACING,
                "cfg": CFG,
                "results": results,
            },
            indent=2,
        )
    )

    # summary table
    print("\n===== SUMMARY (each vs steps=250 reference) =====", flush=True)
    print(
        f"{'steps':>6} | {'lat_relL2':>9} | {'PSNR(dB)':>9} | {'SSIM':>7}", flush=True
    )
    for ns in STEPS:
        ll = np.mean([r["vs_ref"][ns]["latent_rel_l2"] for r in results])
        pp = np.mean([r["vs_ref"][ns]["psnr"] for r in results])
        ss = np.mean([r["vs_ref"][ns]["ssim"] for r in results])
        print(f"{ns:>6} | {ll:>9.4f} | {pp:>9.2f} | {ss:>7.4f}", flush=True)
    print("\n30-vs-100 (mean):", flush=True)
    pp = np.mean([r["p30_vs_100"]["psnr"] for r in results])
    ss = np.mean([r["p30_vs_100"]["ssim"] for r in results])
    ll = np.mean([r["p30_vs_100"]["latent_rel_l2"] for r in results])
    print(f"  PSNR {pp:.2f} dB  SSIM {ss:.4f}  latent_relL2 {ll:.4f}", flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
