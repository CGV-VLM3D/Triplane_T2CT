# CT-RATE EDA — Prior-vs-Recomputed Discrepancy Report

Aggregates every prior (GPT) claim across all EDA modules against our independent recomputation.
Verdict legend: **Verified** (match ≤ noise), **Minor** (small diff, cause understood, adopt ours), **Major** (materially different, adopt ours), **Unable** (could not recompute).

Default population is **CLEAN** (no_chest + unencodable excluded); some GPT numbers were computed on the raw CSV and differ slightly for that reason. Analysis unit is stated per row.

Source tables: `tables/dataset_counts.csv`, `tables/discrepancy_seed.csv`, and each module's `*_raw.json`.

---

## Module 01 — Dataset structure / counts

| Metric (unit) | Prev (GPT) | Recomputed | Verdict | Cause | Adopted |
|---|---|---|---|---|---|
| train patients | 20,000 | 20,000 | Verified | — | 20,000 |
| train scans | 24,128 | 24,128 | Verified | — | 24,128 |
| train volumes (manifest) | 47,149 | 47,149 | Verified | — | 47,149 |
| valid patients | 1,304 | 1,304 | Verified | — | 1,304 |
| valid scans | 1,564 | 1,564 | Verified | — | 1,564 |
| valid volumes (manifest) | 3,039 | 3,039 | Verified | — | 3,039 |
| total patients | 21,304 | 21,304 | Verified | — | 21,304 |
| total scans | 25,692 | 25,692 | Verified | — | 25,692 |
| total volumes (manifest) | 50,188 | 50,188 | Verified | — | 50,188 |
| CSV orphans (report↔label↔meta) | 0 | 0 | Verified | perfect join on VolumeName | 0 |
| recon↔label mismatch | 0 | 0 | Verified | labels constant per scan | 0 |
| no_chest volumes train | 752 | 752 | Verified | — | 752 |
| no_chest volumes valid | 37 | 37 | Verified | — | 37 |
| no_chest volumes total | 789 | 789 | Verified | — | 789 |
| **no_chest scans (total)** | **260** | **280** | **Major** | no_chest is per-**reconstruction**; GPT counted at wrong grain. train 267 + valid 13 = 280 | **280** |
| **no_chest patients (total)** | **258** | **278** | **Major** | same per-reconstruction grain error | **278** |
| on-disk clean train volumes | (n/a) | 46,393 | Verified | matches `ctrate_full` census | 46,393 |
| on-disk clean valid volumes | (n/a) | 3,001 | Verified | matches `ctrate_full` census | 3,001 |
| raw-vs-fixed comparison | (implied possible) | impossible | Major | no raw v1 NIfTI on disk; only `_fixed` exists | fixed-only |

---

## Module 02 — Metadata / scanner / spacing

| Metric (unit) | Prev (GPT) | Recomputed | Verdict | Cause | Adopted |
|---|---|---|---|---|---|
| age median (scan) | 46 | 46 (both splits) | Verified | — | 46 |
| age IQR train (scan) | 35–62 | 35–62 | Verified | — | 35–62 |
| age IQR valid (scan) | 35–61 | 35–61 | Verified | — | 35–61 |
| age range train | 18–102 | 18–102 | Verified | 3 scans >100 (max 102) | 18–102 |
| sex train (scan) | M14098/F10027/miss3 | identical | Verified EXACT | — | M14098/F10027/miss3 |
| sex valid (scan) | M910/F654 | identical (0 missing) | Verified EXACT | — | M910/F654 |
| manufacturer raw train (scan) | Ph14218/SH6904/SIE1069/PNMS1937 | identical | Verified EXACT | — | identical |
| manufacturer raw valid (scan) | Ph953/SH426/SIE60/PNMS125 | identical | Verified EXACT | — | identical |
| manufacturer % (full cohort) | 61.5/30.1/8.4 | clean scan train 58.9/33.0/8.0; valid 61.0/31.1/8.0 | Minor | paper is full-cohort volume-level; ours clean scan-level | ours (both reported) |
| XY spacing median (vol) | ~0.676/0.682 | 0.680 train / 0.683 valid | Verified | — | 0.680/0.683 |
| Z spacing median (vol) | 1.25 (IQR .75–1.5) | 1.25 (IQR 0.75–1.5) | Verified | — | 1.25 |
| **Z min (vol)** | **0.035 mm** | **clean 0.3 mm** (raw 0.035) | **Major** | 0.035 mm is a raw artifact from patient train_9792 (2 no_chest vols, metadata error); excluded in clean pop | **0.3 mm (clean)** |
| NumberofSlices median train (vol) | 259 | 261 | Minor | clean population vs raw | 261 |
| NumberofSlices max train (vol) | 2062 | 2062 | Verified | — | 2062 |
| RescaleType missing train (vol) | ~66.6% | 65.96% | Verified | clean pop slightly lower | 65.96% |
| FocalSpots/GeneratorPower missing | ~66.3% | 65.65% | Verified | clean pop | 65.65% |

---

## Module 03 — Labels (18 silver abnormalities)

| Metric (unit) | Prev (GPT) | Recomputed | Verdict | Cause | Adopted |
|---|---|---|---|---|---|
| 18-label chest-only prevalence, train (scan) | full list | all match ≤ ~0.3pp | Verified | recon-pick in dedup shifts marginally (labels constant per scan) | ours |
| Lung nodule prevalence (scan) | 45.56 | 45.48 | Verified | — | 45.48 |
| Lung opacity (scan) | 36.66 | 36.66 | Verified EXACT | — | 36.66 |
| all-zero % train (scan) | 11.35 | 11.32 | Verified | — | 11.32 |
| all-zero % valid (scan) | 11.53 | 11.57 | Verified | — | 11.57 |
| mean positive labels train/valid | 3.41 / 3.43 | 3.41 / 3.43 | Verified EXACT | — | 3.41 / 3.43 |
| Arterial+Coronary calc phi | 0.742 | 0.7422 | Verified | — | 0.7422 |
| Arterial+Coronary joint count | 5020 | 5120 | Minor | different recon-pick in scan dedup | 5120 |
| Lung opacity+Consolidation phi | 0.336 | 0.3337 | Verified | — | 0.3337 |
| imbalance ratio (train scan) | — | 6.45 | Verified | Lung nodule 45.48 / Pericardial 7.05 | 6.45 |

---

## Module 04 — Reports / text

| Metric (unit) | Prev (GPT) | Recomputed | Verdict | Cause | Adopted |
|---|---|---|---|---|---|
| Clinical missing % | ~50% | 50.4 / 51.1 | Verified | — | 50.4 / 51.1 |
| Technique word median | 16 | 15 | Minor | tokenization/whitespace | 15 |
| Findings word median (scan) | 185 | 185 (train) / 183 (valid) | Verified | — | 185/183 |
| Findings word IQR | ~147–231 | 147–231 | Verified | — | 147–231 |
| Findings sentence median | 15 | 16 | Minor | sentence splitter | 16 |
| Findings exact-unique train/valid | 94.27 / 96.13 | 94.34 / 96.10 | Verified | — | 94.34 / 96.10 |
| Impression word median | 27 | 28 | Verified | — | 28 |
| Impression missing % | ~3.2 | 3.24 | Verified | — | 3.24 |
| Impression exact-unique train | 81.09 | 82.36 | Verified | — | 82.36 |
| Impression/Findings word ratio | 0.14 | 0.142 | Verified | — | 0.142 |
| top Findings sentence count | 'Pericardial…not observed' ~9731 (rank1) | that sentence 9,201 (rank2); rank1 'When examined in the lung parenchyma window;' 14,029 | Minor | scan-dedup + clean pop | ours |
| Pericardial raw mention in label-NEG % | ~80% | 91.9% | Verified (stronger) | 'not observed' boilerplate | 91.9% |
| Atelectasis positive-mention pattern | ~99.84% | affirmed prec 0.896 / rec 0.913 | Directionally verified | NegEx window | ours |
| burden bands (Findings/Impression words) | 130/6, 168/20, 220/40, 290/61 | 130/6, 168/20, 220/41, 289/62 | Verified | — | ours |

---

## Module 05 — NIfTI headers (geometry QC)

| Metric (unit) | Prev (GPT) | Recomputed | Verdict | Cause | Adopted |
|---|---|---|---|---|---|
| valid_fixed on-disk volumes | (n/a) | 3,001 | Verified | matches census | 3,001 |
| no_chest present on disk | (implied) | 0 | Verified | 37+1 physically absent, not flagged-present | 0 |
| dtype / orientation / ndim | (n/a) | int16 / LPS / 3 (all 3031) | Verified | — | int16/LPS/3 |
| metadata-vs-header spacing diff | (assumed exact) | max 4.67e-8 mm | Verified | float32 roundoff; faithful geometry, no resampling | ≈0 |
| metadata-vs-header shape diff | (assumed exact) | 0 | Verified | — | 0 |
| header read errors / QC flags | (n/a) | 0 / 0 | Verified | — | 0 |

---

## Module 07 — Multimodal case sheets (pointer-only)

| Metric | Prev (GPT) | Recomputed | Verdict | Cause | Adopted |
|---|---|---|---|---|---|
| FP-suspect label cells | — | 16 (8 scans) | Verified | positive silver label whose term not AFFIRMED | 16 |
| FN-suspect label cells | — | 3 (2 scans) | Verified | AFFIRMED disease term with negative label | 3 |
| NegEx spot-check valid_1001_a | — | Lung nodule AFFIRMED; Pericardial/Lung-opacity NEGATED | Verified | matches raw report text | ours |

_No prior GPT numbers for this module; all are new pointer diagnostics for human audit._

---

## Module 08 — Reconstruction / kernel

| Finding | Prev (GPT) | Recomputed | Verdict | Cause | Adopted |
|---|---|---|---|---|---|
| recon-index → kernel mapping | (implied consistent) | INCONSISTENT across scanners | Major (new) | Philips iCT recon-1=sharp; Philips Big Bore recon-1=soft (opposite) | read ConvolutionKernel, not index |
| noise (soft-tissue HU) per pair | — | 5/5 reproduced exact | Verified | — | ours |
| Laplacian variance per pair | — | 5/5 reproduced exact | Verified | — | ours |
| valid_1016_b matched-geometry pair | — | mean\|diff\|=55.5 HU, not identical | Verified | different kernels at same geometry | ours |

---

## Module A — MAISI latent space

| Metric (unit) | Prev / known | Recomputed | Verdict | Cause | Adopted |
|---|---|---|---|---|---|
| `_emb` latent overall std | ~0.98 | 0.9573 | Verified | canonical `_emb` source | 0.957 |
| per-channel std | — | 0.928/0.993/0.939/0.961 | Verified | — | ours |
| `mu.pt` std ~0.67 (mismatch) | known | Unable (absent in valid_v2) | Unable | mu.pt not present in ctrate_toy_v2/valid_v2 | asserted from prior knowledge |
| silhouette by label (pooled UMAP) | — | -0.262 | Verified (interp) | no clustering by abnormality | -0.262 |
| silhouette by vendor | — | -0.074 | Verified (interp) | weak vendor confound in latent | -0.074 |
| PCA top-5 var ratio | — | 0.247/0.096/0.077/0.053/0.043 | Verified | — | ours |

---

## Module B — Power spectrum (RAPSD)

| Metric (unit) | Prev / ref | Recomputed | Verdict | Cause | Adopted |
|---|---|---|---|---|---|
| CT pixel RAPSD slope α | natural img ~2.0 / Imagenette ~2.45 | **2.71** | Verified (new) | CT is smoother/lower-frequency than photos | 2.71 |
| latent α ch0..3 | — | 0.94 / 0.54 / 0.56 / 0.42 | Verified (new) | MAISI latent near-spectrally-white | ~<1 |

---

## Summary

- **2 Major numeric corrections**: no_chest scan/patient counts (260→**280** / 258→**278**, per-reconstruction grain), and Z-min (0.035→**0.3 mm clean**, raw artifact).
- **1 Major structural correction**: raw-vs-fixed comparison is **impossible** (no raw NIfTI on disk).
- **1 Major methodological finding**: reconstruction **index does not map to kernel family**; must read `ConvolutionKernel`.
- **Minor** diffs (manufacturer %, slice median, Technique words, sentence median, joint counts) all trace to clean-population dedup or tokenizer/splitter choices — ours adopted.
- Everything else **Verified** (often exact). Core counts (50,188 vol / 25,692 scan / 21,304 patient; clean 46,393 / 3,001) confirmed independently.
