# Sampling-step convergence + metric reliability (report2ct / report2ct_wan)

Two questions, asked of two model families:
1. **How many denoising steps** does a ctgen eval need?
2. **How many generated samples (n)** make the metric estimate reliable enough to call a gap real?

Everything here is **analysis-only** — `src/eval/` is untouched, so vlm3d-docker-faithful
numbers are unaffected. Stages 3–4 run off cached features, not fresh generation.

---

## Stage 1 — per-sample step convergence, report2ct (`step_convergence.py`)

Model `epoch_074.ckpt` (clean, `datalist_full_v2`), spacing 0.8/0.8/1.5, cfg 5.0.

4 valid_v2 cases, **seed fixed per case** (`torch.manual_seed(1000+i)` immediately before
`_denoise`, so the initial noise z_T is identical across step counts and ONLY the RFlow
discretization changes). Each case is generated at `n_steps ∈ {30,50,75,100,250}`; the
250-step output is the reference (~converged ODE solution).

The three metrics are **distance to the 250-step output, not quality vs GT** — a low PSNR
means "a different picture than 250 steps would have drawn", not "a worse picture".

| steps | lat_relL2 | PSNR(dB) | SSIM |
|---|---|---|---|
| 30 | 0.3039 | 21.62 | 0.7721 |
| 50 | 0.2014 | 26.44 | 0.8838 |
| 75 | 0.1514 | 28.54 | 0.9222 |
| 100 | 0.0979 | 33.23 | 0.9685 |

**No convergence knee** — the output keeps moving through 100 (first-order O(1/N) Euler error),
so `30 ≠ 100` and a cheap "30 is a drop-in for 100" was ruled out; needed real metric eval
(stage 2). Case spread is large: 30-vs-100 PSNR ranges 17.18–27.04 dB (`valid_885_a_2` is
nearly converged at 30, `valid_598_a_1` is not at all).
cfg 5.0 is deliberate — CFG stiffens the ODE trajectory, the conservative setting where few
steps is most likely to hurt. Results: `step_convergence_results.json`.

## Stage 2 — metric eval, report2ct (numbers in the parent analysis / chat)
FID + CLIPScore-T2I over 300 cases at n_steps 30/50/100. FID improves monotonically with
steps; CLIP-T2I is flat (30 even marginally highest).

## Stage 3 — FID-vs-n reliability, bootstrap (`fid_bootstrap_ncurve.py`, `.png`, `.json`)
Reference fixed at the 3001-valid GT (μ,Σ) — the **`research` FID profile**. For each
n ∈ {50..300}, 20 volume-level bootstraps of the cached prediction features → FID mean ± std.
Note the JSON's top-level keys `"30"/"50"/"100"` are **step counts, not model names**
(one model, three step settings) — the opposite of stage 4's per-model keys.

| steps | full300 FID_Avg | XY | YZ | XZ | boot n=300 |
|---|---|---|---|---|---|
| 30 | 1.8188 | 1.7168 | 1.3900 | 2.3495 | 1.8295 ± 0.0366 |
| 50 | 1.6909 | 1.6298 | 1.2893 | 2.1535 | 1.7048 ± 0.0371 |
| 100 | 1.5554 | 1.6038 | 1.1661 | 1.8961 | 1.5728 ± 0.0280 |

- FID curve flattens by **n ≈ 150–200**; std ~0.08→**~0.03–0.04 at n=300**. So 300 samples is a
  reliable eval size; n>300 not needed.
- The 30-vs-100 gap (0.26) is **~5.6σ** at n=300 → the step effect is real, not estimator noise.
  Most of the gain is in the **XZ plane** (2.35 → 1.90).

The FID here uses a GPU eigvalsh route to `Tr(sqrtm(Σx Σy)) = Σ√eigvals(Σx Σy)`, verified
identical to `monai.metrics.fid.compute_frechet_distance` to <1e-6. This lives ONLY in the
analysis script — the eval pipeline (`src/eval/`) is unchanged and still uses MONAI.

## Stage 4 — FVD-vs-n reliability, wan vs wan_mask (`fvd_profiles_dump.py`, `fvd_bootstrap_ncurve.py`)

Same bootstrap treatment for `FVD_CTCLIP`, but applied to a **model comparison** at fixed
steps=100 (`report2ct_wan` vs `report2ct_wan_mask`, ep299, sp 0.75/0.75/1.3, cfg 5.0, n=300).

FVD is structurally different from FID: it is not one distance over n samples but the **mean of
per-chunk (CHUNK=4) Frechet distances**, paired to GT by stem. So a bootstrap draw resamples
(gen, stem-matched GT) **pairs**, then re-chunks. `fvd_profiles_dump.py` persists the 18-d
CT-CLIP profiles once (~5 s/vol) so the curve itself is pure numpy.

**Gate**: recomputing over the full 300 in file order reproduces each model's published
`fvd_ctclip.json` **exactly** (`gate_abs_diff: 0.0` for both) — every point on the curve is
therefore on the same scale as the numbers in the eval dirs.

Production metrics at those settings (n=300, `research` FID profile):

| | FID_Avg | XY | YZ | XZ | CLIP-T2I | CLIP-I2I | FVD_CTCLIP |
|---|---|---|---|---|---|---|---|
| wan | 1.6137 | 0.8423 | 1.3475 | 2.6513 | **65.17** | 46.99 | 0.3687 |
| wan_mask | **1.4992** | 0.7312 | 1.2373 | 2.5292 | 57.58 | **55.70** | **0.2841** |

Mask conditioning improves FID/FVD/I2I and costs 7.6 points of T2I.

Bootstrap std falls 0.09 (n=50) → 0.033/0.028 (n=300). Significance of the FVD gap:

| n | gap | unpaired σ (script output) | paired σ | P(wan > wan_mask) |
|---|---|---|---|---|
| 100 | +0.085 | 1.13 | 1.8 | 0.950 |
| 300 | +0.084 | 1.94 | **2.9** | **1.000** |

`_report_gap()` prints the **unpaired** figure (pooled `hypot(σ_a, σ_b)`), which is conservative:
both models' npz hold the **same stems in the same order** and the bootstrap uses the same seed,
so the draws are index-paired and the paired difference is the correct statistic. At n=300
wan_mask was lower in **all 200 resamples** → the FVD advantage is real, though not as
overwhelming as stage 3's step effect.

## Stage 5 — per-sample step convergence, report2ct_wan (`step_convergence_wan*.py`)

Stage 1 repeated for `report2ct_wan` — same 4 cases (`load_eval_cases(n_samples=4)`), same
per-case seeds, same 250-step reference, only the sampler swapped. Regime is the production one
(`epoch_299`, spacing 0.75/0.75/1.3, cfg 5.0). Three steps because Wan decode is a separate env:

```bash
CUDA_VISIBLE_DEVICES=3 python tests/step_convergence/step_convergence_wan.py
CUDA_VISIBLE_DEVICES=3 /opt/conda/envs/wan/bin/python scripts/decode_wan_latents.py \
    --latent-dir /workspace/data/vlm3d_eval/ctgen/wan_stepconv/latents \
    --out /workspace/data/vlm3d_eval/ctgen/wan_stepconv/predictions --spacing 0.75 0.75 1.3
python tests/step_convergence/step_convergence_wan_report.py
```

| steps | report2ct lat/PSNR/SSIM | wan lat/PSNR/SSIM |
|---|---|---|
| 30 | 0.304 / 21.62 / 0.772 | 0.391 / 17.21 / 0.631 |
| 50 | 0.201 / 26.44 / 0.884 | 0.316 / 18.92 / 0.713 |
| 75 | 0.151 / 28.54 / 0.922 | 0.219 / 21.93 / 0.806 |
| 100 | 0.098 / 33.23 / 0.969 | 0.166 / 24.27 / 0.856 |

**wan has no knee either**, and sits farther from its own 250-step reference at every step count.
Its case spread is much tighter than report2ct's (30-vs-100 PSNR 17.6–19.7 vs 17.2–27.0) —
uniformly non-converged rather than case-dependent.

⚠ Cross-model reading limits: `latent_rel_l2` is **not** comparable across models (MAISI
`(4,120,120,64)` vs Wan `(16,64,64,64)` are different latent spaces), and PSNR/SSIM are partly
confounded by different decoders and output sizes (480×480×256 vs 253×512×512). Only "neither
converges" and the **shape** of the curves carry across. 4 cases, no error bars — directional only.

## The Stage-5 vs step-calibration split (the practically useful finding)

The wan step calibration (`scripts/run_wan_stepcal.sh` → `data/vlm3d_eval/ctgen/wan_stepcal/`,
n=150, same regime) says the opposite of stage 5 at the distribution level:

| steps | FID_Avg | XY | YZ | XZ | CLIP-T2I | CLIP-I2I | FVD_CTCLIP |
|---|---|---|---|---|---|---|---|
| 50 | 1.5584 | 0.7983 | 1.3195 | 2.5574 | (66.77 ⚠) | (45.82 ⚠) | 0.3532 |
| 100 | 1.5537 | 0.8136 | 1.2824 | 2.5651 | 65.24 | 46.91 | 0.3578 |

So for the SAME model over 50 → 100 steps:
- **per-sample** (stage 5): SSIM 0.713 → 0.856 — the picture changes substantially;
- **distribution** (above): FID Δ+0.005, FVD Δ−0.005, CLIP Δ−1.5 — all flat, all inside the
  n=150 bootstrap noise (~0.04–0.05).

The extra steps move samples **within** the learned distribution rather than moving the
distribution toward the data. report2ct's extra steps did move the distribution (FID 1.69→1.55).
Practical split:
- reporting **FID/FVD/CLIP only** → wan at 50 steps is indistinguishable from 100; halve the cost;
- anything needing **per-sample reproducibility** (mask-following Dice, saliency, per-volume
  ablation, qualitative comparison) → 50 is **not** a drop-in for 100.

Consistency check across independent generation runs (no seed is fixed in the production flow):
wan steps=100 at n=150 (stepcal) vs n=300 (production) gives CLIP-T2I 65.24 vs 65.17,
FID 1.5537 vs 1.6137, FVD 0.3578 vs 0.3687 — a useful empirical bound on run-to-run drift.

## Known gaps / hygiene

- **wan s50 CLIP was never saved to json** — `wan_stepcal/s50/metrics_clip/` is empty. The
  66.77/45.82 above appear only in `logs/wan_stepcal_s50score_20260725_210034.log`, which does
  not record its `--pred-dir`, so the attribution is **unverified**. Recompute before citing.
- **docker-profile FID is NaN** for both wan and wan_mask production dirs
  (`fid_docker/metrics.json`) — there is no leaderboard-comparable FID for these runs.
- **`scripts/generate_wan_latents.py` has no seed control** — no `manual_seed`/`Generator`, so
  every production run draws fresh noise and step/config comparisons made from separate runs are
  unpaired w.r.t. sampling noise. Stage 5 works around this by seeding in its own script.
- **wan step 30 and all wan_mask step variants were never generated** — the step axis for
  wan_mask is empty, and wan has no steps=30 metric point (stage 5 covers steps=30 per-sample only).
- **`fvd_bootstrap_ncurve.py`'s hardcoded eval path is stale.** `outputs/report2ct_wan/`'s
  eval dirs were reorganized on 2026-07-29 into `cfg_sweep/` and `ep_sweep/`, so the wan entry
  in its `MODELS` dict (`outputs/report2ct_wan/eval_ep299_n300_sp0.75_1.3_cfg5`) now points at
  nothing and the published-value gate would fail. New path:
  `outputs/report2ct_wan/cfg_sweep/eval_ep299_n300_sp0.75_1.3_cfg5` (wan_mask is unmoved).
  The stage-4 numbers above were re-verified against the relocated dirs and are unchanged.
