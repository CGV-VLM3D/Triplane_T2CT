---
paths:
  - "src/baselines/fvlm_*.py"
---

# fVLM adapter & organ-mask preprocessing

- **Adapter** ([src/baselines/fvlm_adapter.py](../../src/baselines/fvlm_adapter.py)) — `FVLMBackbone` exposes `.model` (the `BlipPretrain` instance) plus passthroughs `.prepare_text_feat(...)` and `.forward_test_win(...)`. fVLM is **anatomy-aware** — its forward requires `(image, organ_segmentation_mask)` and returns per-organ ROI features, so the encode_image contract does not apply. Build path mirrors `third_party/fvlm/eval.py` (NOT upstream's `BlipPretrain.from_config`, which hardcodes a developer MAE-pretrain path).
- **Organ-mask prerequisite**: fVLM expects `(volume, mask)` pairs (mask covers lung / heart / esophagus / aorta). The TotalSegmentator outputs already exist at `/workspace/datasets/datasets/CT-RATE/dataset/ts_seg/` (`ts_total/` is one multilabel NIfTI per scan, labels 0–117, split into `train_fixed/`+`valid_fixed/`) — read from there; no separate precompute job is needed (the earlier `/workspace/data/preprocessed/ctrate_*_masks/` staging dir has been removed). The in-memory preprocessing mirroring upstream's offline 4-script pipeline (`fix_data→generate_mask→resize→preprocess`) is [src/baselines/fvlm_preprocess.py](../../src/baselines/fvlm_preprocess.py) `load_ct_and_mask_for_local` (resample to 1×1×3 mm → ScaleIntensityRange(-1150,350)→[0,1] → bbox-crop +5/+20 → pad to 112×256×352). It remaps TS labels to `{lung:1, heart:2, esophagus:3, aorta:4}` by deriving from a **verbatim copy of `resize.py`'s `class_map` + `merged_organ_id`** (single source of truth — never hand-list the int→int map; that once silently dropped `atrial_appendage_left`/TS-61). **fVLM pad-only**: `load_ct_and_mask_for_local(..., pad_only=True)` reproduces upstream's pad-ONLY step 4 (no crop) + patch-aligned padding, so downstream code can derive the ViT token grid from the actual shape — use it for fVLM so no chest voxel is clipped. **Upstream is NOT sliding-window** (corrected 2026-06-10): `eval.py:301-307` builds `dense_patch_slices`/`num_win` but marks it `############# dead code #############`; the real path (`eval.py:309-338`) loops organs and does `center_crop(image, mask==organ, crop_size=(112,288,352))` **once per organ** → `DivisiblePadd(method="end", k=(16,16,32))` → one `forward_test_win(..., skip_organ=organ_id)`. The window is centered on the target organ (so it is never boundary-clipped), and `skip_organ` excludes the target from the boundary-completeness check. The faithful eval path is therefore: `load_ct_and_mask_for_local` (whole preprocessed volume, multilabel mask, no margin-zero) → per-organ `center_crop_organ` + `divisible_pad_end` → `forward_test_win`. The old "single whole-volume forward + 2-voxel margin-zeroing" was a workaround (now removed) that mischaracterized upstream as sliding-window.
- **"thoracic-organ-absent" masks are NOT a label-scheme bug** (verified 2026-06-10, full `valid_fixed` 3039-mask scan): exactly **one consistent TotalSegmentator v2 "total" 117-class scheme** across the whole dataset — `max_label_anywhere=117`, `out_of_range(>117)=0`. CT-RATE ran TS v2 once and shipped it; fVLM authors use that same CT-RATE TS (no re-run). **Do not re-run TotalSegmentator** — a head/neck scan has no lung/heart/esophagus/aorta regardless of TS version. **0.99% (30/3039, 10 patients)** of masks have *no* thoracic organ; their labels are all valid v2 head/neck ids (47-50 C-spine, 79 spinal_cord, 86/87 autochthon, 90 brain, 91 skull, occasional 16 trachea / 20 colon) — out-of-FOV scans, not data errors. The earlier "different label scheme overlap=0" diagnosis was a misread: cause is "non-thoracic FOV", not scheme mismatch.
- **all-zero merged mask = silent no-op in fVLM, learns nothing.** Traced `blip_pretrain.py:119-203`: the visual encoder still runs and produces `image_embeds` for a zero-mask sample, but with no `organ_ids>0` its `organ_mask_flags` row is all-False, so it is excluded from ROI pooling (155-158 `continue`), from every organ's contrastive set (`cl_patient_ids = where(organ_mask_flags[:,organ])`, 190), and even as a negative (negatives come only from patients *with* that organ). Net: **0 loss, 0 gradient** from that sample — just a wasted forward + batch slot. Encoder is ViT (LayerNorm, no cross-sample batch stats), so **skipping zero-mask samples in the dataloader is numerically equivalent to keeping them AND strictly cheaper** — the organ-absent guard (return `None` + skip + log) is the right policy for both train and inference paths.

## ⚠ Axis convention — fVLM is the ONE exception in this repo

`fvlm_preprocess.py::load_ct_and_mask_for_local` differs from every other encoder path in **two**
places (`:290`, `:324`):

1. **No `Orientationd(axcodes="RAS")`** — it uses `nib.get_fdata()`'s raw voxel order, which for
   CT-RATE is **LPS content** (same reason `prepare_valid_gt` yields LPS GT). Every other path,
   including `tests/repa_probe/_spectre.py::load_volume`, reorients to RAS first.
2. **`transpose(2, 1, 0)`** to feed the ViT's `(D, H, W)` — so arrays are `(Z, Y, X)`, not `(X, Y, Z)`.

To use fVLM features **alongside anything else** (Wan latents, ts_seg masks pooled by other code,
cross-encoder figures), convert back: `grid.permute(2, 1, 0, 3).flip(0).flip(1)` and reverse
`voxel_shape` too. Reference impl: `tests/repa_probe/u6_teachers/run.py::fvlm_grid_to_ras`.
The two flips are the same operation as `src/eval/samplers/_orient.py::ras_to_lps` (involutive).

Metrics computed **entirely inside** fVLM's own frame (its image and its mask come out of the same
function) are unaffected — the silent breakage is figures and any cross-encoder alignment.

**Whole-volume dense path**: `load_ct_and_mask_for_local(..., pad_only=True)` → one
`model.visual_encoder(img)` → `(x, outs)`; `x` is `(B, T, C)` dense tokens. `_VIT_KWARGS` sets no
`classification`, so **no CLS token is prepended** — still assert `T == prod(grid)`.
Grid is coarse: patch `(16,16,32)` at 1×1×3 mm ⇒ one z-token spans **48 mm**; a whole volume gives
`(X, Y, Z) = (11, 16, 7)` ≈ 1232 tokens (lung 186 / heart 17 / aorta 5 / esophagus 0, **no liver**).
