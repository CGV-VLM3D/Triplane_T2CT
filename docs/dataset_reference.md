# Dataset reference (full detail)

Summary + decision-driving paths live in [CLAUDE.md](../CLAUDE.md) `## Dataset reference`; this file
is the full reference (sizes, dtypes, split provenance, deprecation notes).

**Note the doubled path**: the data root is `/workspace/datasets/datasets/` (not `/workspace/datasets/`). The outer `/workspace/datasets/` only holds download scripts + `split.json`, which is why it looks "empty".

- **Raw CT (CT-RATE)** at `/workspace/datasets/datasets/CT-RATE/dataset/`
  - `train_fixed/`: 20,000 patient dirs, 47,148 scans (NIfTI); `valid_fixed/`: 1,304 patient dirs, 3,038 scans
  - `metadata/{train,validation}_metadata.csv`: spacing, kernel, manufacturer, etc.
  - `multi_abnormality_labels/`: 18 binary labels per scan
  - `radiology_text_reports/`: free-text reports (findings + impression)
  - `ts_seg/`: TotalSegmentator outputs (`ts_total/`, `ts_lung_nodules/`, `ts_pleural_pericard_effusion/`, each split into `train_fixed/`+`valid_fixed/`) — use this for fVLM organ masks; no need to re-run TotalSegmentator.

- **Split definition**: `/workspace/datasets/split.json` (seed 42, `root` = CT-RATE dataset dir) — train 5000 / valid 1000 / unused 625 / missing_a_1 4. **Note**: this `valid` 1000 is a *train-drawn dev-valid* (all `train_*`, same distribution as train) — **DEPRECATED** and unused by the ctgen train/eval path. The evaluation proxy is now `valid_fixed`-based (see the canonical toy dataset v2 below + Win condition).

- **MAISI VAE latents — two distinct sets** (latent shape is MAISI VAE `[4, 120, 120, 64]`):
  - **Small `.pt` set** — `/workspace/datasets/datasets/latents/{train,valid}/<scan_id>/{mu.pt,sigma.pt,src.txt}` (~83 GB). 5000 train + 1000 valid, matching `split.json`. `mu`/`sigma` are `[4,120,120,64]` **fp16**. Channel mean/std for normalization in `latents/stats.json`. The `valid/` 1000 here is the **deprecated train-drawn dev-valid** (all `train_*`); do not use it for model selection.
  - **Canonical toy dataset (v2)** — `/workspace/data/ctrate_toy_v2/` (built by [scripts/make_toy_dataset.py](../scripts/make_toy_dataset.py)): `train/` = the same 5000 `train_fixed` scans with latents **symlinked** (no 614 GB copy); `valid_v2/` = `valid_fixed` **one scan per patient = 1304** (frozen `valid_v2/ids.json`), the evaluation proxy for the (hidden) challenge test set. `report/REPORT.md` documents representativeness (toy-train vs train_fixed gated; valid_v2 is a patient-level census). Use this for split-based training/sweeps; the lab dev-valid is superseded.
  - **Full 48k `.nii.gz` set** — `/workspace/data/report2ct_work_dir/image_embeddings/*_emb.nii.gz` (~614 GB). **48,145 scans** = essentially all of train; NIfTI `(120,120,64,4)` **fp32**, latent-space spacing stamped in the header. Paired `*_emb.nii.gzmulti_2560.json` carry spacing/text-context for Report2CT. Indexed by `/workspace/data/report2ct_work_dir/datalist_5k.json`; matching text embeddings under `report2ct_work_dir/text_embeddings/`. This is the full set used for Report2CT training.

- **Storage convention**: `/workspace/datasets/` is collaborator's read-only area. New artifacts → `/workspace/data/`.
- **GPU convention**: prefix scripts with `CUDA_VISIBLE_DEVICES=0` for single-GPU; explicit `CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 ...` for multi-GPU.
