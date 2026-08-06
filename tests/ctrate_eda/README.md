# CT-RATE EDA — Folder Map & Reproduction Guide

Exploratory data analysis of **CT-RATE** for the VLM3D 2026 submission (ctgen / reportgen / abnclass).
Every number in the reports is read back from the files in `tables/` and `figures/`; nothing is hand-typed.

**Start here:**
- [`final_report_ko.md`](final_report_ko.md) — 종합 보고서 (Korean): all findings + 12 research-question answers + modeling recommendations.
- [`discrepancy_report.md`](discrepancy_report.md) — every prior (GPT) claim vs our recomputation (Verified / Minor / Major / Unable) + adopted values.

---

## Folder map

```
/workspace/tests/ctrate_eda/
├── final_report_ko.md        # SYNTHESIS (Korean) — read first
├── discrepancy_report.md     # prior-vs-recomputed reconciliation
├── README.md                 # this file
├── run_all_eda.py            # sequential module runner (subprocess, --root, GPU-aware)
│
├── scripts/                  # one NN_*.py per EDA module (self-contained, re-runnable)
│   ├── 01_validate_structure.py   # counts, CSV joins, no_chest, orphans
│   ├── 02_metadata.py             # demographics, scanner, spacing, missingness, kernels
│   ├── 03_label.py                # 18 silver-label prevalence, co-occurrence, scanner skew
│   ├── 04_report.py               # report length, boilerplate, tokenizer fit, NegEx term↔label
│   ├── 05_nifti_header.py         # header-only geometry QC (dtype/orientation/spacing)
│   ├── 06_nifti_voxel.py          # voxel-level intensity QC (hands-on companion)
│   ├── 07_multimodal.py           # per-case CT+report+label sheets + montages + mismatch pointers
│   ├── 08_recon.py                # reconstruction/kernel sharpness natural experiments
│   ├── A_latent.py                # MAISI latent stats + UMAP + GPU decode (needs CUDA)
│   ├── B_spectrum.py              # RAPSD power-spectrum slope (pixel vs latent)
│   ├── viewer.py                  # interactive ipywidgets viewer used by hands_on notebooks
│   ├── build_notebooks.py         # regenerates hands_on/*.ipynb
│   └── verify_*.py / _verify_*.py # independent verification scripts (cross-check pass)
│
├── tables/                   # *.csv (rounded) + *_raw.json (RAW pre-rounding values)
├── figures/                  # *.png (>=200 dpi, titled, unit stated); figures/montages/ 25 case montages
├── samples/                  # example reports (normal/abnormal/random) + label_disagreements.md
├── case_sheets/              # 25 per-case *.md + INDEX.md + mismatch_suspects.md
└── hands_on/                 # interactive Jupyter notebooks (00_explore, 06_voxel, 07_multimodal, 08_recon)
```

### Key tables (unit noted)
| File | Unit | Contents |
|---|---|---|
| `dataset_counts.csv` | vol/scan/patient | population counts + no_chest quarantine |
| `demographics.csv` | scan | age / sex |
| `scanner_protocol.csv`, `spacing_shape.csv` | scan / volume | manufacturer, spacing, matrix |
| `metadata_missingness.csv` | volume | per-column missing % (+ vendor fingerprint) |
| `label_prevalence.csv`, `label_cooccurrence.csv`, `label_by_scanner.csv` | scan | 18 labels |
| `report_statistics.csv`, `report_boilerplate.csv`, `term_label_agreement.csv` | scan | text |
| `nifti_qc.csv`, `nifti_qc_flags.csv` | volume | geometry QC |
| `recon_compare.csv` | reconstruction | kernel sharpness |
| `latent_stats.csv`, `rapsd.csv` | volume / slice | MAISI latent + spectrum |

---

## How to reproduce

All scripts self-insert `/workspace` on `sys.path`, are **read-only** on the dataset, and write only under `/workspace/tests/ctrate_eda/`.

**Run the whole suite in order:**
```bash
cd /workspace
python /workspace/tests/ctrate_eda/run_all_eda.py                 # auto-skips GPU module if no CUDA
python /workspace/tests/ctrate_eda/run_all_eda.py --root /workspace/tests/ctrate_eda
python /workspace/tests/ctrate_eda/run_all_eda.py --only 03_label.py 04_report.py   # subset
python /workspace/tests/ctrate_eda/run_all_eda.py --force-gpu     # force A_latent even if probe fails
```
- `run_all_eda.py` runs `01 → 02 → … → 08 → A_latent → B_spectrum` via subprocess, prints per-module status/time, and continues past a failure (records it). It **skips `A_latent.py`** when no CUDA device is visible.
- **GPU note**: `A_latent.py` does a MAISI decode — use `CUDA_VISIBLE_DEVICES=1` (or `2`), never device 0 (user's). It wraps encode/decode in `inference_mode()`+autocast and uses the tiled `SlidingWindowInferer` (full-volume 480³ decode OOMs).

**Run a single module directly:**
```bash
cd /workspace
python /workspace/tests/ctrate_eda/scripts/03_label.py
CUDA_VISIBLE_DEVICES=1 python /workspace/tests/ctrate_eda/scripts/A_latent.py
```

**Re-verify (independent cross-check):** the `verify_*.py` / `_verify_*.py` scripts recompute headline numbers from raw CSV/NIfTI and compare to the emitted tables.

---

## How to use the hands-on notebooks

Interactive exploration lives in [`hands_on/`](hands_on/) (`00_explore`, `06_voxel`, `07_multimodal`, `08_recon`). They drive [`scripts/viewer.py`](scripts/viewer.py) (ipywidgets + ipympl).

1. Launch Jupyter with the `ipympl` backend available.
2. First cell of any notebook:
   ```python
   %matplotlib widget
   import sys; sys.path.insert(0, '/workspace/tests/ctrate_eda/scripts')
   from viewer import *
   ```
3. Then explore any volume / recon pair / report by absolute path:
   ```python
   view_volume('/workspace/datasets/datasets/CT-RATE/dataset/valid_fixed/valid_1000/valid_1000_a/valid_1000_a_1.nii.gz')
   view_recon_pair(path_recon1, path_recon2)   # kernel A/B slider (see §6 of the report)
   load_report('valid_1000_a_1')               # joined report + 18 labels for the scan
   ```
- Everything is path-swappable (any NIfTI / TotalSegmentator mask / report id).
- **Analysis unit reminder**: one `.nii.gz` = one *reconstruction* (volume); reports and the 18 labels are constant per *scan*.
- To regenerate the notebooks: `python /workspace/tests/ctrate_eda/scripts/build_notebooks.py`.

---

## Conventions (load-bearing)

- **Population**: default CLEAN = exclude `no_chest` (volume-level) + unencodable. Clean = train 46,393 / valid 3,001.
- **Unit**: dedup to **scan** for text/label stats (report & 18 labels are scan constants); **volume** for geometry/intensity; **patient** only where noted.
- **`_fixed` NIfTI**: HU already baked — **never** reapply metadata RescaleSlope/Intercept (-8192 is an out-of-FOV padding sentinel; clip to [-1000, 1000]).
- **Latent scale**: canonical `_emb` std ≈ 0.98; never mix `mu.pt` (std ≈ 0.67).
- Figures: ≥200 dpi, titled, axis-labeled, analysis unit in the caption, train=blue / valid=orange.
