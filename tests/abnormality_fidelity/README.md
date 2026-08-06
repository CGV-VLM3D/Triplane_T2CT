# Abnormality-fidelity spot check — GT vs 4 ctgen models

Qualitative "does the generated CT follow the abnormality described in the report" check on
hand-picked **valid_v2** cases, plus a quantitative 4-organ mask-following Dice for the
mask-conditioned model. Everything here is CPU-only.

## ⚠ Orientation — fixed at the source (2026-07-25); read this before trusting old dirs
- **Now**: every sampler writes **LPS content at save time** (`report2ct._save_mha` + subclasses,
  `text2ct._save_mha` (added 2026-07-24), `decode_wan_latents._save_mha` all apply `ras_to_lps`;
  `generatect` is natively LPS). So **top-level `predictions/` are now the canonical, correct-orientation
  output** and are scored directly against the LPS GT. Do **not** re-flip them.
- **Reversed history**: this README used to say "top-level is RAS-wrong, only `eval_LPS_n100` is valid".
  That was true *only in the pre-fix regime*, when top-level predictions were RAS and a read-side script
  (`scripts/reeval_lps_n100.py`) flipped them once to LPS. After the save-side fix that guidance is
  **inverted** — a read-side re-flip now *double-flips* the (already-LPS) predictions back to RAS. The
  read-side flow (`reeval_lps_n100.py`, `aggregate_lps_reeval.py`, `run_ts_n100.sh`, `compute_dice_n100.py`,
  `run_ts_cpu.sh`, `compute_dice.py`) was **removed 2026-07-25**.
- **The `eval_LPS_n100/` dirs on disk are retained** as historical artifacts: they hold the pre-fix
  RAS-top-level predictions flipped once → LPS, so their recorded numbers (e.g. `report2ct_wan` CLIPScore
  **63.0** vs the RAS-broken top-level **24.3**, paper anchor ≈59.9) are still the correct-orientation
  results **for those old predictions**. The tables below are kept from that run. Any *newly generated*
  predictions should be scored at top-level, not via `eval_LPS_n100`.
- **`make_figures.py` caveat**: it flips per-model via a hardcoded `is_lps` flag (all model rows =
  `False`). Those flags assume RAS top-level predictions — **stale after the save-side fix**. If a
  prediction dir is regenerated with the current sampler (now LPS), `make_figures.py` will double-flip it.
  Verify on-disk orientation before regenerating figures.

## Models compared (eval prediction dirs)

| label in figures | eval dir (`predictions/*.mha`) | geometry |
|---|---|---|
| `GT` | `data/vlm3d_eval/_valid_full_3001` (raw CT-RATE, LPS) | 1024²×194 @ 0.39×0.39×1.5 |
| `report2ct_toy_v2` | `outputs/report2ct/eval_cfg5_spacing0.8_toy_v2` | 480²×256 @ 0.8×0.8×1.5 |
| `report2ct_wan` | `outputs/report2ct_wan/eval_ep299_sp0.73_1.34_cfg5` | 512²×253 @ 0.73×0.73×1.34 |
| `report2ct_wan_mask` | `outputs/report2ct_wan_mask/eval_ep299_sp0.73_1.34_cfg5` | 512²×253 @ 0.73×0.73×1.34 |
| `text2ct_v2` | `outputs/text2ct_toy_v2/eval_2026-06-28` (UNet `unet_rflow_200ep_toy_v2.pt`) | 512²×128 @ 0.75×0.75×3.0 |

> **`text2ct_v2` = the 200-epoch toy_v2 UNet.** There is also `text2ct_100ep_toy_v2`
> (101-epoch). If you meant the 101ep one, swap the dir and re-run `make_figures.py`.

Generations are **text-conditioned, not spatially registered** to the GT (and not to each other).
Slices are chosen per-volume from that volume's own body center-of-mass; they will **not** show
the same anatomical point across a row. The figures answer "does the described finding appear /
look right", not "is it in the same voxel".

## The 4 cases (full reports in `case_reports.txt`)

| case | pick | key report finding |
|---|---|---|
| `valid_27_a_1`  | **Cardiomegaly (심비대)** | "cardiothoracic index increased in favor of the heart", pericardial effusion 2.7 cm, bilateral pleural effusions, ascending-aorta ectasia 4.7 cm |
| `valid_144_a_1` | **Severe** | **giant right mediastinal/hilar mass (~15 cm CC)** narrowing the right main bronchus + right pleural effusion |
| `valid_155_a_1` | **Small** | focal **lung nodule 15×13 mm** (RUL, horizontal fissure). (The 25 mm nodule in the report is *thyroid*, out of chest FOV.) |
| `valid_322_a_1` | **Small** | focal **coronary-artery (LAD) calcification**, otherwise normal chest |

Figures: `figures/<case>.png` — rows = {GT, 4 models}, cols = {axial, coronal, sagittal},
window chosen per finding (mediastinal for heart/mass/calcium, lung window for the nodule).

### What the figures show (my read — verify yourself)
- **Cardiomegaly / giant mass**: clearly reproduced. The report2ct models fill the right hemithorax
  with a large soft-tissue mass in `valid_144`, and show an enlarged cardiac silhouette in `valid_27`.
- **Nodule / coronary calcium**: these are millimetric — do not expect them at a fixed location.
  Page through the volumes yourself; the figure just fixes a representative level + the right window.

## Mask-following Dice (bonus: does `report2ct_wan_mask` obey the mask?)

`report2ct_wan_mask` is conditioned on a painted **4-organ** mask (lung / heart / aorta / esophagus;
`src/data/organ_groups.py`). We compare, **voxel-index-aligned on the prediction grid**:

- **GT conditioning mask** — ts_seg (valid_fixed) → Orientation(RAS) → Resize NEAREST (512,512,253)
  → `apply_grouping` {0..4}, i.e. literally the mask latent that conditioned the model.
- **PRED organ mask** — **TotalSegmentator** (`--fast`, 3 mm `total`, **CPU**) on the decoded `.mha`
  → `apply_grouping` {0..4}.

`report2ct_wan` (no mask conditioning) is scored against the **same** GT mask as a baseline — the
`wan → wan_mask` jump is the evidence that conditioning actually pins the anatomy.

Dice per organ (recorded in `mask_dice/dice_results.json`; producer script `compute_dice.py` removed 2026-07-25 — table retained as the result):

| case | lung (wan→mask) | heart | aorta | esophagus |
|---|---|---|---|---|
| valid_27  | 0.47 → **0.82** | 0.18 → 0.32 | 0.00 → **0.47** | 0.00 → 0.00 |
| valid_144 | 0.38 → **0.82** | 0.00 → 0.20 | 0.00 → 0.05 | 0.00 → 0.00 |
| valid_155 | 0.65 → **0.97** | 0.09 → **0.51** | 0.10 → 0.37 | 0.11 → 0.13 |
| valid_322 | 0.86 → **0.96** | 0.54 → 0.37 | 0.31 → 0.41 | 0.05 → 0.13 |
| **MEAN**  | **0.59 → 0.89** | **0.20 → 0.35** | **0.10 → 0.33** | 0.04 → 0.06 |

**Conclusion: yes — the mask-conditioned model follows the mask.** Lung Dice ≈ **0.89** (up to 0.97)
and mask conditioning raises every organ on average (lung **+0.30**, aorta **+0.22**, heart **+0.15**),
almost always per-case. The generated anatomy lands where the mask said.

### Caveats
- **Alignment was verified, not assumed.** The (now-removed) `compute_dice.py` baked an orientation
  self-check (identity vs axis flips on the lung channel); **identity won for all wan_mask cases** and all
  four organs co-registered together — that's how the correct index alignment was fixed (the first pass had
  an extra in-plane double-flip). Any future re-implementation must keep such a self-check.
- **Esophagus Dice is unreliable** (tiny, low-contrast; TS-3 mm barely finds it) — treat as noise.
- **Case 144** (giant mass) distorts the mediastinum, so heart/aorta segmentation of the *generated*
  volume is poor for both models — expected.
- PRED masks come from **TotalSegmentator on generated CT**; TS error is folded into every number, so
  read Dice as a *lower bound on fidelity* and only the wan↔wan_mask *contrast* as clean signal.
- 4 cases, CPU TS 3 mm — a spot check, not a full-set metric.

## Abnormality subfolders (`figures/<abnormality>/`)
Beyond the 4 showcase cases, `make_figures.py` auto-picks the **2 cleanest (lowest-n_pos) valid_v2
cases per abnormality** + 2 all-normal cases, each rendered as the same 5-row×3-plane figure in the
finding-appropriate window. Selected ids in `figures/abnormality_cases.json`.

Each folder also has a **`reports.txt`** (`make_case_reports.py`) giving, per case: the positive
abnormality labels, **★ the report sentences relevant to that abnormality cropped to the top**
(regex match — note CTO/CTI need `\b` or they falsely hit "se*cti*ons"/"obstru*cti*ve"), then the full
FINDINGS + IMPRESSION. Folders:
`cardiomegaly`, `pleural_effusion`, `emphysema`, `lung_nodule`, `consolidation`,
`arterial_calcification`, `normal_all_zero`. Model-family read: `MODEL_FAMILY_READ.md`.

## n100 mask-following Dice — all 7 models (eval_LPS_n100, same 100 ids)
TotalSegmentator (fast/3mm, CPU) on each model's `eval_LPS_n100` predictions vs the GT 4-organ mask on
that model's grid. Per-model orientation frame chosen empirically (heart+lung max) — **all 7 picked
`flipYX`**; margin vs identity is large for mask models (+0.62) and small for non-mask (~+0.20),
which is itself the alignment audit. (Producer scripts `run_ts_n100.sh` + `compute_dice_n100.py`
removed 2026-07-25 — results retained in `mask_dice_n100/dice_n100_results.json` and the table below.)

| model | mask? | lung | heart | aorta | esophagus |
|---|---|---|---|---|---|
| **wan_mask** | **YES** | **0.969** | **0.923** | **0.881** | **0.748** |
| **report2ct_text2ct_mask** | **YES** | **0.967** | **0.918** | **0.876** | **0.772** |
| wan | no | 0.621 | 0.450 | 0.225 | 0.087 |
| report2ct_text2ct | no | 0.604 | 0.425 | 0.224 | 0.070 |
| text2ct | no | 0.612 | 0.432 | 0.230 | 0.086 |
| report2ct_toy_v2 | no | 0.609 | 0.393 | 0.203 | 0.060 |
| report2ct_full | no | 0.612 | 0.421 | 0.245 | 0.093 |

**Conclusion**: both mask-conditioned models follow the mask **near-perfectly** (lung ≈0.97, heart ≈0.92,
aorta ≈0.88, even esophagus ≈0.76) and beat their non-mask sibling by huge, consistent margins
(Δ lung +0.35, heart +0.48, aorta +0.65, esophagus +0.68 for BOTH the Wan and text2ct backbones). The
five non-mask models cluster at the "unconditioned prior" level (lung ≈0.61, heart ≈0.42) regardless of
backbone — so wan's texture noise does NOT hurt organ geometry (it's texture-only). Mask conditioning is
what pins the anatomy.

## Reproduce
```bash
python  tests/abnormality_fidelity/make_figures.py      # figures/ (+ abnormality subfolders)
```
> Mask-following Dice scripts (`run_ts_cpu.sh` + `compute_dice.py`, and the n100 pair
> `run_ts_n100.sh` + `compute_dice_n100.py`) were removed 2026-07-25 along with the read-side
> `reeval_lps_n100.py` flow — they were coupled to the pre-fix RAS→LPS re-flip regime. Their
> results are preserved in `mask_dice/dice_results.json` / `mask_dice_n100/dice_n100_results.json`
> and the tables above. To recompute mask-Dice on current (already-LPS) predictions, re-implement
> against the top-level `predictions/` with no read-side flip.
