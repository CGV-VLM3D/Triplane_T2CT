# CT-RATE EDA bundle — valid_fixed sample

Source: `/workspace/datasets/datasets/CT-RATE/dataset/valid_fixed/` (READ-ONLY).
Files copied byte-exact (`shutil.copy2`); **no resample/crop/normalize, no DICOM slope/intercept reapplied**. Fixed NIfTI store HU directly (nifti `scl_slope/inter` = nan/identity), so intensities are read as-is from `dataobj`.

## 1. Totals
- **Files copied:** 30 (25 primary + 5 recon-2 comparison)
- **Total size:** 5.29 GB (5289.91 MB)
- **no_chest exclusion hits:** 0 — none of the requested files appear in `no_chest_valid.txt` / `no_chest_train.txt`.

## 2. Per-group counts & size (incl. recon-2 comparison file)
| group | files | size (MB) |
|---|---|---|
| all-zero | 6 | 1147.49 |
| lung-nodule-only | 6 | 828.86 |
| diffuse-low-burden | 6 | 1193.43 |
| multi-abnormality | 6 | 862.21 |
| medical-material | 6 | 1257.92 |

## 3. Found files (30)
**all-zero**
- `files/all-zero/valid_1000_a_1.nii.gz` — 1024x1024x194 @ 0.39355x0.39355x1.5 mm, int16, LPS
- `files/all-zero/valid_1010_a_1.nii.gz` — 1024x1024x250 @ 0.34863x0.34863x1.5 mm, int16, LPS
- `files/all-zero/valid_1012_a_1.nii.gz` — 512x512x203 @ 0.57422x0.57422x1.5 mm, int16, LPS
- `files/all-zero/valid_1020_a_1.nii.gz` — 1024x1024x218 @ 0.30859x0.30859x1.5 mm, int16, LPS
- `files/all-zero/valid_1025_b_1.nii.gz` — 512x512x375 @ 0.74249x0.74249x1 mm, int16, LPS
- `files/all-zero/valid_1000_a_2.nii.gz` — 512x512x388 @ 0.78711x0.78711x0.75 mm, int16, LPS  _(recon-2 comparison)_

**lung-nodule-only**
- `files/lung-nodule-only/valid_1001_a_1.nii.gz` — 512x512x213 @ 0.68359x0.68359x1.5 mm, int16, LPS
- `files/lung-nodule-only/valid_1009_a_1.nii.gz` — 768x768x218 @ 0.53385x0.53385x1.5 mm, int16, LPS
- `files/lung-nodule-only/valid_1061_a_1.nii.gz` — 512x512x229 @ 0.61434x0.61434x1.25 mm, int16, LPS
- `files/lung-nodule-only/valid_1077_a_1.nii.gz` — 1024x1024x222 @ 0.31543x0.31543x1.5 mm, int16, LPS
- `files/lung-nodule-only/valid_1085_a_1.nii.gz` — 512x512x232 @ 0.72656x0.72656x1.5 mm, int16, LPS
- `files/lung-nodule-only/valid_1001_a_2.nii.gz` — 1024x1024x106 @ 0.3418x0.3418x3 mm, int16, LPS  _(recon-2 comparison)_

**diffuse-low-burden**
- `files/diffuse-low-burden/valid_1022_a_1.nii.gz` — 512x512x232 @ 0.76562x0.76562x1.5 mm, int16, LPS
- `files/diffuse-low-burden/valid_1039_a_1.nii.gz` — 768x768x246 @ 0.44531x0.44531x1.5 mm, int16, LPS
- `files/diffuse-low-burden/valid_1068_a_1.nii.gz` — 1024x1024x209 @ 0.34961x0.34961x1.5 mm, int16, LPS
- `files/diffuse-low-burden/valid_1073_a_1.nii.gz` — 768x768x226 @ 0.39062x0.39062x1.5 mm, int16, LPS
- `files/diffuse-low-burden/valid_1153_a_1.nii.gz` — 1024x1024x242 @ 0.38574x0.38574x1.5 mm, int16, LPS
- `files/diffuse-low-burden/valid_1022_a_2.nii.gz` — 1024x1024x116 @ 0.38281x0.38281x3 mm, int16, LPS  _(recon-2 comparison)_

**multi-abnormality**
- `files/multi-abnormality/valid_1016_b_1.nii.gz` — 512x512x209 @ 0.65625x0.65625x1.5 mm, int16, LPS
- `files/multi-abnormality/valid_1016_d_1.nii.gz` — 512x512x295 @ 0.68771x0.68771x1 mm, int16, LPS
- `files/multi-abnormality/valid_103_a_1.nii.gz` — 1024x1024x236 @ 0.45312x0.45312x1.5 mm, int16, LPS
- `files/multi-abnormality/valid_1041_c_1.nii.gz` — 512x512x225 @ 0.67188x0.67188x1.25 mm, int16, LPS
- `files/multi-abnormality/valid_1078_a_1.nii.gz` — 1024x1024x226 @ 0.37012x0.37012x1.5 mm, int16, LPS
- `files/multi-abnormality/valid_1016_b_2.nii.gz` — 512x512x209 @ 0.65625x0.65625x1.5 mm, int16, LPS  _(recon-2 comparison)_

**medical-material**
- `files/medical-material/valid_1288_a_1.nii.gz` — 1024x1024x277 @ 0.38379x0.38379x1.5 mm, int16, LPS
- `files/medical-material/valid_225_b_1.nii.gz` — 1024x1024x242 @ 0.38574x0.38574x1.5 mm, int16, LPS
- `files/medical-material/valid_366_a_1.nii.gz` — 1024x1024x229 @ 0.3418x0.3418x1.5 mm, int16, LPS
- `files/medical-material/valid_1103_b_1.nii.gz` — 512x512x200 @ 0.74805x0.74805x1.5 mm, int16, LPS
- `files/medical-material/valid_114_b_1.nii.gz` — 512x512x206 @ 0.54883x0.54883x1.5 mm, int16, LPS
- `files/medical-material/valid_1288_a_2.nii.gz` — 512x512x554 @ 0.76758x0.76758x0.75 mm, int16, LPS  _(recon-2 comparison)_

## 4. Missing files
| group | requested | role | reason |
|---|---|---|---|
| all-zero | `valid_1000_a_1 (raw)` | comparison_raw | no raw/original valid NIfTI on server; only valid_fixed exists |
| lung-nodule-only | `valid_1001_a_1 (raw)` | comparison_raw | no raw/original valid NIfTI on server; only valid_fixed exists |
| diffuse-low-burden | `valid_1022_a_1 (raw)` | comparison_raw | no raw/original valid NIfTI on server; only valid_fixed exists |
| multi-abnormality | `valid_1016_b_1 (raw)` | comparison_raw | no raw/original valid NIfTI on server; only valid_fixed exists |
| medical-material | `valid_1288_a_1 (raw)` | comparison_raw | no raw/original valid NIfTI on server; only valid_fixed exists |

> **Raw NIfTI comparison could not be satisfied:** this server holds only `*_fixed` volumes. There is no raw/original `valid/` NIfTI tree on disk (the `no_chest_valid.txt` `valid/...` prefix is the upstream HF naming, not an on-disk directory here; `ctrate_full/.../latents/*_emb.nii.gz` are MAISI latents, and `ts_seg/` holds segmentation masks — neither is a raw CT volume). So the raw-vs-fixed intensity/spacing/affine comparison (goal #5) cannot be run from this bundle.

## 5. Comparison set (recon-1 vs recon-2 of the same scan)
recon-1 is the primary group file; recon-2 is the `comparison_recon2` file. raw = not available (see §4).
| case (group) | recon-1 (primary) | recon-2 (comparison) | raw |
|---|---|---|---|
| valid_1000_a (all-zero) | 1024x1024x194 @ 0.39355x0.39355x1.5 | 512x512x388 @ 0.78711x0.78711x0.75 | — (none on server) |
| valid_1001_a (lung-nodule-only) | 512x512x213 @ 0.68359x0.68359x1.5 | 1024x1024x106 @ 0.3418x0.3418x3 | — (none on server) |
| valid_1022_a (diffuse-low-burden) | 512x512x232 @ 0.76562x0.76562x1.5 | 1024x1024x116 @ 0.38281x0.38281x3 | — (none on server) |
| valid_1016_b (multi-abnormality) | 512x512x209 @ 0.65625x0.65625x1.5 | 512x512x209 @ 0.65625x0.65625x1.5 | — (none on server) |
| valid_1288_a (medical-material) | 1024x1024x277 @ 0.38379x0.38379x1.5 | 512x512x554 @ 0.76758x0.76758x0.75 | — (none on server) |

## 6. EDA notes / data-quality flags
- **-8192 padding sentinel:** `valid_1025_b_1` and `valid_1016_d_1` have `intensity_min = -8192` (median still ~-860 HU). This is edge/out-of-FOV padding, not corruption — clip/window to e.g. [-1000, 1000] HU for EDA. Do **not** treat -8192 as a real HU value.
- **High-HU outliers:** `valid_1061_a_1` max = 10286 (metal/contrast); several files max at 3071 (12-bit CT ceiling).
- **Orientation:** all 30 volumes are `LPS` — consistent.
- **Recon differences** are large (in-plane matrix + slice thickness), e.g. `valid_1000_a`: 1024²×194 @1.5mm (recon-1) vs 512²×388 @0.75mm (recon-2); `valid_1001_a`: 512²×213 @1.5mm vs 1024²×106 @3mm. Good for goal #4.
- **all-zero group** = label-negative (all 18 labels 0), normal-like controls.

## Files in this bundle
- `manifest.csv` — full per-file metadata (all requested columns + `role`, `bundle_path`)
- `summary.json` — machine-readable totals / per-group / missing
- `files/<group>/*.nii.gz` — the 30 copied NIfTI volumes