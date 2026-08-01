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
| `label_overlap` | `\|target_labels ∩ source_labels\|` positive count, recorded for both swap conditions. Informative for `label_mismatched_swap` (how much the random donor happens to share); for `label_matched_swap` it is by construction the target's own positive count. |
| `label_exact_match` | whether the donor's 18-vector is *identical* to the target's. **Always `true` on a `label_matched_swap` row** (no near-matches are emitted) and `false` on `label_mismatched_swap`; `null` for `gt`/`null`. |
| `label_hamming` | number of differing labels. `0` ⟺ `label_exact_match=true`, so it is `0` on every matched row and `>0` on every mismatched one. |
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
    --n 300 --seed 0 --conditions gt label_matched_swap label_mismatched_swap null \
    --out /workspace/data/mask_intervention/manifest_n300_sm1.0_seed0.jsonl \
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
    --manifest /workspace/data/mask_intervention/manifest_n300_sm1.0_seed0.jsonl \
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
- **Exact match or no row** (matched-swap, superseding the earlier "nearest-overlap fallback"
  rule, 2026-08-02): a `label_matched_swap` donor must carry *exactly* the same 18-label vector.
  When no such scan exists anywhere in the census, the target simply gets **no matched row** and
  is listed in `<manifest>.uncovered_targets.json` — a near-match is never substituted, so every
  matched row in a manifest is a true twin (`label_exact_match` is always `true` there).
- **`null` condition**: `report2ct_wan_mask_v2` supports this natively via its learned
  `no_mask_embed` (see the module docstring cited above). Older mask models
  (`report2ct_wan_mask`, `report2ct_text2ct_mask`) have no learned null — approximating with an
  empty/zero mask is a known risk (the model was never trained to see this input) and should be
  flagged as such if used. The builder gates this on the **checkpoint's own `state_dict`**
  (`no_mask_embed` present or not), not on a model name.

### Donor pool and what the numbers actually look like

**Two pools, because the two swap arms have opposite problems** (`--donor-ids`, default = the
clean valid 3001 **and** train 46,393 censuses):

| arm | searched in | why |
|---|---|---|
| `label_matched_swap` | the whole census (49,394 scans) | an exact 18-label twin is scarce, so the search must be as wide as possible; among twins, one whose mask latent already exists is preferred, and any donor still missing one is listed for precompute |
| `label_mismatched_swap` | only scans that already have a Wan mask latent (6,907) | any differently-labelled donor works, so this arm never needs extra precompute |

Two properties of this dataset drive the rules above and should be read before interpreting a
matched/mismatched contrast:

- **An exact label twin does not exist for every target.** Only 770 distinct 18-label
  combinations occur among the 3001 valid scans, and a high-burden combination is usually unique.
  Measured on the shipped `n=300` manifest (2026-08-02): searching valid alone covers ~193/300;
  adding train raises it to **253/300**, and the remaining **47 targets have no twin anywhere in
  CT-RATE's 49,394 scans** — they are all high-burden (3+ positive labels, mostly 6–10). Those
  targets get **no matched row** (listed in `<manifest>.uncovered_targets.json`), so the matched
  arm covers a *subset* of the targets while `gt`/`label_mismatched_swap`/`null` cover all of
  them. Any matched-vs-other comparison must be made on that subset, not on the full set.
- **A uniformly random donor is an exact match ~3 % of the time** (mostly normal↔normal, where
  the vector is all-zero), which would quietly turn part of the "mismatched" arm into a second
  matched arm. The builder therefore excludes same-vector donors from `label_mismatched_swap` and
  applies **no overlap cap** — the realized overlap is reported instead (`n=300` manifest: mean
  0.75, median 0, p90 2, max 7).

**Train-split donors** are ordinary CT-RATE chest scans and their masks are precomputed by the
same script, but two things follow from using them: the donor's `ts_seg` lives under
`ts_total/train_fixed/`, which `seg_metrics._ts_seg_path` resolves from the id prefix (a
hardcoded `valid_fixed` would have left `dice_to_input_mask`/HD95/ASSD silently NaN — the very
metrics this experiment reads); and their mask latents may not exist yet, which is what the
`<manifest>.needs_mask_latent.{valid,train}.json` sidecars are for (the builder prints the exact
`precompute_wan_mask_latents.py` command for each).
