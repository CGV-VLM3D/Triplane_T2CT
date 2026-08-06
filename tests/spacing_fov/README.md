# spacing → FOV → eval-preprocessing study

Why does report2ct's 2.5D-FID improve so much when the *declared* voxel spacing is changed
from `(1.0, 1.0, 1.5)` to `(0.8, 0.8, 1.5)` — same checkpoint, same guidance scale?

Observed in production (LPS-corrected, n=100, ckpt `report2ct_full_2026-06-30/epoch_069`):
**FID Avg 4.68 → 2.01**, with the out-of-plane planes carrying the change (YZ 9.15 → 2.79 on
the full-1304 RAS run).

## The three candidate explanations

| | hypothesis | discriminating signal |
|---|---|---|
| **A** | **declared scale** — voxels unchanged; the eval resamples by the header, so the body ends up the wrong size inside the metric's fixed frame | relabelling alone reproduces the gap; FID(s) is U-shaped with its minimum where the declared FOV matches the GT FOV |
| **B** | **conditioning** — `s=1.0` is outside the training distribution, so the UNet draws worse/different content | same-seed volumes differ across `s`; `gen@1.0` relabelled to 0.8 is still worse than `gen@0.8` |
| **C** | **resampling smoothing** — declaring a smaller `s` downsamples 480→384 px, blurring generator artefacts (GT is downsampled too) | FID(s) decreases monotonically instead of forming a U |

**H0** (the intuitive version): *"it works best at the spacing the model saw most during
training."* This predicts the right optimum — but it cannot by itself distinguish A from B,
because the two are numerically degenerate here: the training conditioning value **is**
`FOV / 480`, so the training median (0.765) and the GT-FOV-matching value (366 mm / 480 =
0.762) are the same number drawn from the same population. Only the mechanism separates them,
and only header-only relabelling can do that.

## Why the mechanism matters in practice

If the effect is **A**, spacing is a free post-hoc knob: rewrite the header, keep the voxels,
done. If it is **B**, changing spacing means regenerating everything (~23 h for the full 1304).

## Key facts established from the code (see the plan for file:line citations)

* The output grid is **fixed at 480×480×256** (the latent is `(4,120,120,64)`), so
  `FOV = grid × declared spacing` is a pure declaration. The model can only change how much
  of the grid the body occupies.
* **FID does not crop** — after the 1 mm resample every volume (250–500 px) fits inside the
  512³ frame, so the chain is *pad-only*. What differs between arms is the body-to-frame
  ratio. **CLIPScore does crop**: at `s=1.0` the in-plane 640 px is cut to 480 (25% lost),
  at `s=0.8` 512 → 480 (6%).
* Training never normalised to a common mm scale: `Resized(480,480,256)` stretches each
  scan's full FOV into the grid and the *post-resize* effective spacing (`FOV/480`) is what
  the UNet is conditioned on. Self-consistent at training time; at inference it means the
  model has no intrinsic mm scale to fall back on.
* Measured distributions: training in-plane spacing median **0.765** (p5–p95 0.646–0.925),
  z median **1.318**; GT in-plane FOV median **366 mm**, z FOV median **339 mm**.
  `s=1.0` declares 480 mm — larger than **98.2%** of the GT volumes.

## Scripts

| script | what it does |
|---|---|
| `gen_seeded.py` | generates predictions with a per-case fixed noise seed, so two arms differ only by spacing |
| `geometry_stats.py` | **E1** GT FOV + training-spacing distributions → `figs/F1`, `figs/F2` |
| `eval_preproc_dump.py` | **E2** pushes real / gen@1.0 / gen@0.8 through both eval chains, dumps NIfTI + `figs/F3_*` |
| `body_metrics.py` | **E3** body/lung geometry, n=100, in the FID's 1 mm space → `figs/F4` |
| `relabel_sweep.py` | **E4** FID vs declared spacing with voxels held fixed → `figs/F5` |
| `content_invariance.py` | **E0b** same seed, spacing swept → does the conditioning change content? → `figs/F6` |
| `decompose_2x2.py` | **E5** the 2×2 (generated-at × labelled-as) via the production evaluator |
| `_common.py` | replicas of the upstream FID / CLIP preprocessing chains + body/lung masks |

`RESULTS.md` holds the conclusions.

## Reproducing

```bash
# 1. two arms, fixed seed, n=100  (~1.8 h/arm on an idle GPU)
CUDA_VISIBLE_DEVICES=2 python tests/spacing_fov/gen_seeded.py \
    --spacing 1.0 1.0 1.5 --out-dir tests/spacing_fov/preds/gen_sp1.0
CUDA_VISIBLE_DEVICES=3 python tests/spacing_fov/gen_seeded.py \
    --spacing 0.8 0.8 1.5 --out-dir tests/spacing_fov/preds/gen_sp0.8

# 2. CPU analyses
python tests/spacing_fov/geometry_stats.py
python tests/spacing_fov/eval_preproc_dump.py
python tests/spacing_fov/body_metrics.py

# 3. production 2x2, then the sweep gated against it
CUDA_VISIBLE_DEVICES=2 python tests/spacing_fov/decompose_2x2.py
CUDA_VISIBLE_DEVICES=2 python tests/spacing_fov/relabel_sweep.py --content gen_sp0.8 \
    --gate <C_gen0.8_label0.8 FID_2p5D_Avg> --gate-config 0.8 1.5
CUDA_VISIBLE_DEVICES=2 python tests/spacing_fov/relabel_sweep.py --content gen_sp1.0 \
    --gate <A_gen1.0_label1.0 FID_2p5D_Avg> --gate-config 1.0 1.5

# 4. conditioning probe (reuses the two arms; generates 0.7 / 1.5 / 3.0)
CUDA_VISIBLE_DEVICES=2 python tests/spacing_fov/content_invariance.py --generate
```

Evaluation reuses the frozen assets the production numbers are built on:
GT `data/vlm3d_eval/_lps_reeval/gt_lps_seed100`, prompts `prompts_seed100.xlsx`, and the
shared GT FID-feature cache (so no GT features are recomputed).

`preds/`, `dumps/`, `_cells/`, `_sweep_featcache/`, `_invariance/`, `runs/` are bulk
artefacts and are not committed.
