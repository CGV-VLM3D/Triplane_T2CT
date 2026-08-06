"""Score the report2ct_wan step-convergence sweep against its 250-step reference (step 3 of 3).

Consumes what `step_convergence_wan.py` (latents) and `scripts/decode_wan_latents.py`
(predictions) wrote, and applies the SAME three distances `step_convergence.py` used for
report2ct, so the two models' curves can be laid side by side:
  * ``latent_rel_l2`` — ``‖z_ns − z_250‖ / ‖z_250‖`` in the model's own latent space,
  * ``psnr`` / ``ssim`` — on the decoded HU volume, ``data_range=2000`` (HU ∈ [-1000, 1000]).

These measure DISTANCE TO THE 250-STEP OUTPUT, not quality against GT: a low PSNR here means
"a different picture than 250 steps would have drawn", not "a worse picture".

⚠ ``latent_rel_l2`` is comparable across steps but NOT across models — MAISI's latent is
(4,120,120,64) and Wan's is (16,64,64,64). Only the decoded PSNR/SSIM and the SHAPE of the
curve carry across.
"""

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage.metrics import structural_similarity

WORK = Path("/workspace/data/vlm3d_eval/ctgen/wan_stepconv")
HERE = Path(__file__).parent
REPORT2CT_JSON = HERE / "step_convergence_results.json"
OUT = HERE / "step_convergence_wan_results.json"
PNG = HERE / "step_convergence_wan.png"


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 2000.0) -> float:
    """PSNR between two HU int16 volumes, data_range = 2000 HU ([-1000,1000])."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((data_range**2) / mse))


def _load_vol(p: Path) -> np.ndarray:
    """Decoded prediction → HU ndarray ``(Z, Y, X)``."""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p)))


def main() -> None:
    meta = json.loads((WORK / "manifest.json").read_text())
    steps, ref = meta["steps"], meta["ref"]

    results = []
    for case in meta["cases"]:
        sid = case["scan_id"]
        lats = {
            ns: np.load(WORK / "latents" / f"{sid}__s{ns:03d}.npy") for ns in steps
        }  # (16, 64, 64, 64)
        vols = {
            ns: _load_vol(WORK / "predictions" / f"{sid}__s{ns:03d}.mha")
            for ns in steps
        }  # (Z, Y, X) HU

        z_ref, v_ref = lats[ref], vols[ref]
        ref_norm = float(np.linalg.norm(z_ref))
        per_case = {"scan_id": sid, "seed": case["seed"], "vs_ref": {}}
        for ns in steps:
            per_case["vs_ref"][ns] = {
                "latent_rel_l2": round(
                    float(np.linalg.norm(lats[ns] - z_ref)) / ref_norm, 4
                ),
                "psnr": round(psnr(vols[ns], v_ref), 2),
                "ssim": round(
                    float(
                        structural_similarity(
                            vols[ns].astype(np.float32),
                            v_ref.astype(np.float32),
                            data_range=2000.0,
                        )
                    ),
                    4,
                ),
            }
        per_case["p30_vs_100"] = {
            "latent_rel_l2": round(
                float(np.linalg.norm(lats[30] - lats[100]))
                / float(np.linalg.norm(lats[100])),
                4,
            ),
            "psnr": round(psnr(vols[30], vols[100]), 2),
            "ssim": round(
                float(
                    structural_similarity(
                        vols[30].astype(np.float32),
                        vols[100].astype(np.float32),
                        data_range=2000.0,
                    )
                ),
                4,
            ),
        }
        results.append(per_case)
        print(
            f"  {sid}  30-vs-100: PSNR {per_case['p30_vs_100']['psnr']:.2f} "
            f"SSIM {per_case['p30_vs_100']['ssim']:.4f} "
            f"latentL2 {per_case['p30_vs_100']['latent_rel_l2']:.4f}",
            flush=True,
        )

    OUT.write_text(json.dumps({**meta, "results": results}, indent=2))

    print("\n===== SUMMARY (each vs steps=250 reference) =====", flush=True)
    print(
        f"{'steps':>6} | {'lat_relL2':>9} | {'PSNR(dB)':>9} | {'SSIM':>7}", flush=True
    )
    for ns in steps:
        m = {
            k: np.mean([r["vs_ref"][ns][k] for r in results])
            for k in ("latent_rel_l2", "psnr", "ssim")
        }
        print(
            f"{ns:>6} | {m['latent_rel_l2']:>9.4f} | {m['psnr']:>9.2f} | {m['ssim']:>7.4f}",
            flush=True,
        )

    _plot(steps, ref, results)
    print(f"\nwrote {OUT} and {PNG}", flush=True)


def _mean_curve(results: list, steps: list, key: str) -> np.ndarray:
    """Case-averaged metric across `steps`.

    ``vs_ref`` keys are ints in the in-memory wan results but strings once round-tripped
    through JSON (report2ct's file), so both are looked up by ``str(ns)``.
    """
    return np.array(
        [
            np.mean(
                [
                    {str(k): v for k, v in r["vs_ref"].items()}[str(ns)][key]
                    for r in results
                ]
            )
            for ns in steps
        ]
    )


def _plot(steps: list, ref: int, wan_results: list) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r2ct = json.loads(REPORT2CT_JSON.read_text())
    non_ref = [s for s in steps if s != ref]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    specs = [
        ("latent_rel_l2", "latent rel-L2 vs 250-step", "lower = closer to converged"),
        ("psnr", "PSNR vs 250-step (dB)", "higher = closer"),
        ("ssim", "SSIM vs 250-step", "higher = closer"),
    ]
    for ax, (key, ylabel, sub) in zip(axes, specs):
        for tag, res, c in (
            ("report2ct (MAISI, ep074)", r2ct["results"], "#d62728"),
            ("report2ct_wan (ep299)", wan_results, "#1f77b4"),
        ):
            y = _mean_curve(res, non_ref, key)
            ax.plot(non_ref, y, "-o", color=c, label=tag)
        ax.set_xlabel("n_steps")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}\n({sub})", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Stage 1: per-sample step convergence, seed fixed, 4 cases — distance to the "
        "250-step output (NOT quality vs GT)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(PNG, dpi=120)


if __name__ == "__main__":
    main()
