# Mask-intervention manifest (requirement F)

Schema for evaluating mask-conditioned models under interventions on the input conditioning
mask: `gt` (target's own GT mask), `label_matched_swap` (another patient's mask with the same
18-label vector), `label_mismatched_swap` (a random patient's mask), and `null` (no mask —
supported today only by `report2ct_wan_mask_v2`'s learned `no_mask_embed`).

**Status**: schema + eval-side consumption are implemented ([src/eval/analysis/persample.py](../src/eval/analysis/persample.py)
reads this manifest when `task.manifest` is set), and so is the **builder**
([scripts/build_mask_intervention_manifest.py](../scripts/build_mask_intervention_manifest.py) —
deterministic donor pairing with self-pair/patient-leakage/round-trip checks enforced in code and
fixed by [tests/eval_analysis/test_manifest_builder.py](../tests/eval_analysis/test_manifest_builder.py)).
Generation-side *consumption* of the manifest (the sampler generating one volume per row) is the
remaining piece.

⚠ **A manifest run is for DIAGNOSTIC metrics only** — CLIPScore-T2I, `dice_to_input_mask`,
`dice_to_gt_mask`. Do **not** score it for FID/FVD as leaderboard metrics: one target appears in
several rows, which breaks the "1 volume = 1 independent sample" premise those set-level metrics
rest on (and the GT reference would be counted once per repeat of the same target).

## Format

JSONL — one line per generated sample. Array-like label fields are explicit per-label dict
entries (not packed arrays), so the format stays flat and typo-checkable.

```jsonc
{
  "sample_id": "valid_10_a_1__label_matched_swap__src-valid_88_a_1__sm-1.0__seed-0",
  "target_id": "valid_10_a_1",
  "condition": "label_matched_swap",
  "cond_mask_source_id": "valid_88_a_1",
  "seed": 0,
  "cfg_scale_text": 5.0,
  "cfg_scale_mask": 1.0,
  "run_id": "wanmaskv2_ep099",
  "ckpt": "/workspace/outputs/report2ct_wan_mask_v2/2026-.../checkpoints/epoch_099.ckpt",
  "target_labels": {"Medical material": 0, "Cardiomegaly": 1, "...": 0},
  "source_labels": {"Medical material": 0, "Cardiomegaly": 1, "...": 0},
  "label_overlap": 3,
  "report_note": "label vector identical to target, but the source patient's report findings are not — matched label vector is not the same as matched findings.",
  "gen_path": "/workspace/outputs/.../predictions/valid_10_a_1__label_matched_swap__src-valid_88_a_1__sm-1.0__seed-0.mha"
}
```

## Field reference

| field | meaning |
|---|---|
| `sample_id` | primary key = the generated file's stem. Plain (non-intervention) runs never need a manifest: they default to `sample_id == target_id == scan_id`. |
| `target_id` | the scan whose report/labels/GT this sample is generated *for*. Label and GT-volume/GT-mask joins always key on this. |
| `condition` | `gt` \| `label_matched_swap` \| `label_mismatched_swap` \| `null`. |
| `cond_mask_source_id` | the scan whose mask was actually fed to the model. `= target_id` for `gt`; a donor scan for the two swap conditions; `null` for `condition=null`. Input-mask joins (`dice_to_input_mask`) key on this. |
| `seed` / `cfg_scale_text` / `cfg_scale_mask` / `run_id` / `ckpt` | generation provenance. |
| `target_labels` / `source_labels` | explicit 18-entry dicts (label name → 0/1). `source_labels` is `null` for `condition in {gt, null}`. |
| `label_overlap` | `\|target_labels ∩ source_labels\|` positive count — always recorded for swap conditions, even exact matches, so an approximate "matched" pair is distinguishable from an exact one. |
| `label_exact_match` | whether the donor's 18-vector is *identical* to the target's. `label_matched_swap` rows are a MIX (see below), so this is the field to filter on before pooling them. `null` for `gt`/`null`. |
| `label_hamming` | number of differing labels — what the matched-swap fallback minimizes. `0` ⟺ `label_exact_match=true`. |
| `report_note` | free-text limitation flag — label-vector equality is not report equality; kept as data, not just documentation. |
| `gen_path` | the generated `.mha` (stem = `sample_id`). Written as `null` by the builder: the consumer derives it as `pred_dir / f"{sample_id}.mha"` ([persample.py](../src/eval/analysis/persample.py)), so the manifest never carries a path that can go stale relative to the run that used it. |

## `condition` vs `cfg_scale_mask` — not the same axis

`condition` says *which* mask was designated as input; `cfg_scale_mask` (`s_m`) says *how strongly*
the model attends to whatever mask was designated. They are orthogonal manifest fields. In
`report2ct_wan_mask_v2`'s dual-CFG sampler
([Report2CTWanMaskV2LatentSampler._predict](../src/eval/samplers/report2ct_wan.py)),
`s_m=0` zeroes the designated mask's contribution to the prediction — so the *output* of
`condition=gt, cfg_scale_mask=0` is mathematically identical to `condition=null` — but a
`gt`/swap row at `s_m=0` still has a *designated* input mask (so `dice_to_input_mask` is a
defined, informative "how much did suppressing s_m break mask-following" value), whereas
`condition=null` has no designated mask at all (`cond_mask_source_id=null` →
`dice_to_input_mask` = NA, only `dice_to_gt_mask` is meaningful). Do not conflate the two when
analyzing an `s_m` sweep — each `s_m` value is a **separate generation pass** (a fresh sampling
run, not a free re-derivation), so sweeping `s_m` is "same model/checkpoint, different sampling
runs," not "one generation, many post-hoc readings."

## Building one

One manifest per `s_m` value (see the orthogonality note above), targets from
`load_eval_cases(--n)`:

```bash
python scripts/build_mask_intervention_manifest.py \
    --n 50 --seed 0 --conditions gt label_matched_swap label_mismatched_swap null \
    --out /workspace/data/mask_intervention/manifest_n50_sm1.0_seed0.jsonl \
    --ckpt /workspace/outputs/report2ct_wan_mask_v2/2026-07-26_2/checkpoints/epoch_299.ckpt \
    --run-id wanmaskv2_ep299 --cfg-text 5.0 --cfg-mask 1.0
```

It self-verifies before exiting (self-pairs, patient leakage, coverage/uniqueness, a
`persample._read_manifest` round-trip counting both records **and** warnings, and a from-CSV
recomputation of every `label_overlap`), printing the table and failing loudly on any violation.

## Running one (report2ct_wan_mask_v2)

Same 3-step Wan flow as a plain run, with the manifest replacing "n cases":

```bash
# 1. main env — one latent per manifest ROW, named <sample_id>.npy
CUDA_VISIBLE_DEVICES=3 python scripts/generate_wan_mask_v2_latents.py \
    --ckpt /workspace/outputs/report2ct_wan_mask_v2/2026-07-26_2/checkpoints/epoch_299.ckpt \
    --manifest /workspace/data/mask_intervention/manifest_n50_sm1.0_seed0.jsonl \
    --out <OUT> --mask-dir /workspace/data/report2ct_wan/mask_latents_512x512x253 \
    --cfg-scale-text 5 --cfg-scale-mask 1.0 --spacing 0.75 0.75 1.3

# 2. wan env — decode (unchanged: it is driven by the latent's stem)
CUDA_VISIBLE_DEVICES=3 /opt/conda/envs/wan/bin/python scripts/decode_wan_latents.py \
    --latent-dir <OUT>/latents --out <OUT>/predictions --spacing 0.75 0.75 1.3

# 3. score the DIAGNOSTICS only (FID/FVD are refused for a manifest run)
python scripts/run_eval.py task=ctgen model=report2ct_wan_mask_v2 out_dir=<OUT> \
    task.manifest=<manifest> task.is_mask_model=true \
    task.metrics.fid_2p5d=false task.metrics.fvd=false task.metrics.fvd_ctclip=false \
    task.metrics.per_sample=true task.metrics.dice=true task.metrics.hd95=true
```

- Step 1 cross-checks `--ckpt` / `--cfg-scale-text` / `--cfg-scale-mask` against the manifest's
  recorded provenance **before** any GPU time, and pins each row's initial noise per
  `(target_id, seed)` — so a target's `gt`, swap and `null` volumes differ only by the mask.
- One manifest per `s_m`. Existing `.npy`/`.mha` are skipped for resumability, so a run whose
  settings changed needs a fresh `--out` / `out_dir`.
- Step 3 keys prompts.xlsx by `sample_id` (text = the target's report) and builds
  `<OUT>/gt_view/` — symlinks `<sample_id>.mha -> <gt>/<target_id>.mha`, because upstream CLIP
  finds a prediction's GT by filename stem.

## Generation-side rules (enforced by the builder)

- **Self-pair forbidden**: `cond_mask_source_id != target_id` for both swap conditions.
- **Patient-leakage forbidden — check by `patient_id`, not `scan_id`**: donor pools drawn from
  the full valid census (3001, multiple scans/patient) can otherwise pick a *different scan of
  the same patient* as the target — assert `patient_id(source) != patient_id(target)`. Donor
  pools drawn only from `valid_v2` (1 scan/patient) satisfy this automatically, but the check
  should still be explicit rather than relied upon implicitly.
- **Deterministic pairing**: seed the donor selection so re-running the manifest builder with
  the same seed reproduces the same pairs.
- **No exact label match found** (mismatched-swap, or matched-swap with a rare label
  combination): fall back to nearest-overlap donor and record the true `label_overlap` — do not
  silently substitute a worse match without recording how much it differs.
- **`null` condition**: `report2ct_wan_mask_v2` supports this natively via its learned
  `no_mask_embed` (see the module docstring cited above). Older mask models
  (`report2ct_wan_mask`, `report2ct_text2ct_mask`) have no learned null — approximating with an
  empty/zero mask is a known risk (the model was never trained to see this input) and should be
  flagged as such if used. The builder gates this on the **checkpoint's own `state_dict`**
  (`no_mask_embed` present or not), not on a model name.

### Donor pool and what the numbers actually look like

The pool is the full clean valid census (3001) **intersected with the scans that already have a
precomputed Wan mask latent** (`data/report2ct_wan/mask_latents_512x512x253`) — the generation
side cannot condition on a mask that does not exist. Measured 2026-08-01: **1907 of 3001**
qualify, covering **all 1304 patients**, with a mild label shift (max per-label prevalence
difference 0.030; mean label burden 3.47 → 3.28). The builder prints the exclusion count, and the
sidecar `<out>.meta.json` records it together with the build's git sha / argv / verification
table (kept out of the JSONL so that two same-seed builds stay byte-identical).

Two properties of this dataset drive the rules above and should be read before interpreting a
matched/mismatched contrast:

- **Exact label-vector matches do not exist for every target.** Only **798 of the 1304** valid_v2
  targets have at least one cross-patient donor with an identical 18-vector (816 even if the pool
  were the whole 3001) — so roughly **40 % of `label_matched_swap` rows are nearest-Hamming
  fallbacks**, not exact matches. On the shipped `n=50` manifest: 32 exact / 18 fallback (mean
  Hamming 1.83, max 4). Always split on `label_exact_match` before averaging.
- **A uniformly random donor is an exact match ~3 % of the time** (mostly normal↔normal, where
  the vector is all-zero), which would quietly turn part of the "mismatched" arm into a second
  matched arm. The builder therefore excludes exact matches from `label_mismatched_swap` and
  applies **no overlap cap** — the realized overlap is reported instead (`n=50` manifest: mean
  0.88, median 1, p90 2, max 4).
