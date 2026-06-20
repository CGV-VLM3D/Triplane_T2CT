# Report2CT Training Runbook

Step-by-step guide from raw CT-RATE data → Report2CT smoke run → full 100-epoch training.

> **⚠️ 2026-06-20 update — ids/datalist canonicalization (no_chest correction).**
> The `ids_train.txt` / `ids_valid.txt` / `datalist.json` (`datalist_5k.json`) referenced
> in the steps below are **deprecated**: they predate the no_chest (brain) correction and
> reintroduce 47 train + 20 valid brain volumes (and the old 1000-scan valid split).
> Use the corrected canonical files instead:
> `ids_train_v2.txt` (5000, from `ctrate_toy_v2/train/ids.json`),
> `ids_valid_v2.txt` (1304 valid_v2, from `ctrate_toy_v2/valid_v2/ids.json`),
> merged into **`datalist_v2.json`** (now the default in `configs/data/report2ct.yaml`).
> Brain embeddings are quarantined under `data/no_chest_quarantine/`. See plan
> `ticklish-snuggling-gem` and `data/no_chest_quarantine/no_chest_ids.json`.

---

## 1. Overview (pipeline in 5 steps)

```
Step 0  Setup working directory
Step 1  Precompute text embeddings (3 HF BERT models, ~hours for 6K samples)
Step 2  Precompute image embeddings (MAISI VAE encode, ~hours for 6K samples)
Step 3  Build datalist.json
Step 4  (optional) Generate parity references + run parity tests
Step 5  Launch training
```

GPU policy: **all steps use GPU 1 (or 1,2 for training)**. GPU 0 is reserved for other work.

---

## 2. Step-by-step

### Step 0 — Setup

```bash
bash scripts/setup_report2ct_workdir.sh
# Creates /workspace/data/report2ct_work_dir/ subdirectories
# and extracts ids_train.txt (5000 IDs) + ids_valid.txt (1000 IDs)
```

### Step 1 — Text embeddings

First run downloads ~10 GB of HF models (MedEmbed-large-v0.1, ClinicalBERT, BiomedVLP-CXR-BERT-specialized). Cached thereafter.

**Smoke (100 samples, ~5 min after first download):**
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/precompute_report2ct_text_embeddings.py \
    --ids-file  /workspace/data/report2ct_work_dir/ids_train.txt \
    --reports-csv /workspace/datasets/datasets/CT-RATE/dataset/radiology_text_reports/train_reports.csv \
    --out-dir   /workspace/data/report2ct_work_dir/text_embeddings \
    --limit 100 --device cuda:0
```

**Full (5000 train + 1000 valid, ~1–3 hours):**
```bash
# Train
CUDA_VISIBLE_DEVICES=1 python scripts/precompute_report2ct_text_embeddings.py \
    --ids-file  /workspace/data/report2ct_work_dir/ids_train.txt \
    --reports-csv /workspace/datasets/datasets/CT-RATE/dataset/radiology_text_reports/train_reports.csv \
    --out-dir   /workspace/data/report2ct_work_dir/text_embeddings \
    --device cuda:0
# Valid
CUDA_VISIBLE_DEVICES=1 python scripts/precompute_report2ct_text_embeddings.py \
    --ids-file  /workspace/data/report2ct_work_dir/ids_valid.txt \
    --reports-csv /workspace/datasets/datasets/CT-RATE/dataset/radiology_text_reports/validation_reports.csv \
    --out-dir   /workspace/data/report2ct_work_dir/text_embeddings \
    --device cuda:0
```

Expected output: `<id>_emb.nii.gzmulti_2560.json` per sample in `text_embeddings/`.

### Step 2 — Image embeddings

**Smoke (100 samples, ~10–30 min):**
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/precompute_report2ct_image_embeddings.py \
    --ids-file  /workspace/data/report2ct_work_dir/ids_train.txt \
    --ct-rate-dir /workspace/datasets/datasets/CT-RATE/dataset/train_fixed \
    --out-dir   /workspace/data/report2ct_work_dir/image_embeddings \
    --vae-ckpt  /workspace/third_party/maisi_bundle/models/autoencoder.pt \
    --limit 100 --device cuda:0
```

**Full (5000 train + 1000 valid, ~4–8 hours):**
```bash
# Train
CUDA_VISIBLE_DEVICES=1 python scripts/precompute_report2ct_image_embeddings.py \
    --ids-file  /workspace/data/report2ct_work_dir/ids_train.txt \
    --ct-rate-dir /workspace/datasets/datasets/CT-RATE/dataset/train_fixed \
    --out-dir   /workspace/data/report2ct_work_dir/image_embeddings \
    --vae-ckpt  /workspace/third_party/maisi_bundle/models/autoencoder.pt --device cuda:0
# Valid
CUDA_VISIBLE_DEVICES=1 python scripts/precompute_report2ct_image_embeddings.py \
    --ids-file  /workspace/data/report2ct_work_dir/ids_valid.txt \
    --ct-rate-dir /workspace/datasets/datasets/CT-RATE/dataset/train_fixed \
    --out-dir   /workspace/data/report2ct_work_dir/image_embeddings \
    --vae-ckpt  /workspace/third_party/maisi_bundle/models/autoencoder.pt --device cuda:0
```

Expected output: `<id>_emb.nii.gz` per sample in `image_embeddings/`.

### Step 3 — Build datalist

```bash
python scripts/build_report2ct_datalist.py \
    --image-dir /workspace/data/report2ct_work_dir/image_embeddings \
    --text-dir  /workspace/data/report2ct_work_dir/text_embeddings \
    --ids-train /workspace/data/report2ct_work_dir/ids_train.txt \
    --ids-valid /workspace/data/report2ct_work_dir/ids_valid.txt \
    --out /workspace/data/report2ct_work_dir/datalist.json
```

Also writes merged `*_emb.nii.gzmulti_2560.json` files (dim + spacing + embeddings) into `image_embeddings/`.

### Step 4 — (Optional) Parity reference generation + tests

#### 4a — Text encoder reference

```bash
python - <<'PY'
import json
from src.baselines.report2ct_text_encoder import Report2CTTextEncoder
enc = Report2CTTextEncoder(device="cuda:1")
samples = [
    {"text": "Trachea is open. No consolidation.", "type": "findings"},
    {"text": "Normal chest CT.", "type": "impression"},
]
for s in samples:
    s["embedding"] = enc.encode(s["text"]).tolist()
import pathlib
pathlib.Path("/workspace/data/report2ct_work_dir/parity_refs").mkdir(parents=True, exist_ok=True)
json.dump({"samples": samples},
          open("/workspace/data/report2ct_work_dir/parity_refs/text_ref.json","w"))
print("text_ref.json written")
PY
```

#### 4b — Image encoder reference

Choose any 1 volume from ids_train.txt and run both upstream + ours:

```bash
# Upstream (uses submodule training script's encode logic):
# --- OR ---
# Ours (simpler):
python - <<'PY'
from src.baselines.report2ct_image_encoder import Report2CTImageEncoder
import pathlib
enc = Report2CTImageEncoder(device="cuda:1")
nifti = "/workspace/datasets/datasets/CT-RATE/dataset/train_fixed/train_10000/train_10000_a/train_10000_a_1.nii.gz"
out = "/workspace/data/report2ct_work_dir/parity_refs/train_10000_a_1_emb_ref.nii.gz"
enc.encode_to_file(nifti, out)
print(f"Saved {out}")
PY
# To test parity against upstream, run upstream's encode on the same volume and save as ref.
```

#### 4c — Run all parity tests

```bash
pytest tests/test_report2ct_parity.py -v
```

Expected:
- `test_config_parity_vs_upstream_json`  PASSED ✓
- `test_unet_forward_parity`             PASSED ✓ (bit-exact)
- `test_scheduler_parity`               PASSED ✓
- `test_text_encoder_parity`            PASSED / SKIPPED (need text_ref.json)
- `test_image_encoder_parity`           PASSED / SKIPPED (need *_emb_ref.nii.gz)
- `test_training_step_smoke`            PASSED ✓

### Step 5 — Training

**Smoke (1 epoch, 50 batches, ~10–20 min, 2 GPU):**
```bash
CUDA_VISIBLE_DEVICES=1,2 python src/train.py \
    experiment=report2ct_repro \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=50 \
    logger=wandb
```

**Full (100 epochs, 2 GPU, multi-day):**
```bash
CUDA_VISIBLE_DEVICES=1,2 python src/train.py experiment=report2ct_repro
```

Checkpoints are saved to `/workspace/outputs/report2ct/<timestamp>/checkpoints/` (one
timestamped dir per run; older runs from before the reorg live under
`/workspace/outputs/report2ct/legacy_work_dir/checkpoints/`).

---

## 3. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: state_dict` when loading VAE | `autoencoder.pt` key mismatch | The checkpoint may be a plain `state_dict` dict; `src/baselines/maisi.py:load_frozen` handles both formats |
| `RuntimeError: Expected all tensors on same device` | spacing/context not moved to device | Ensure DataModule's transforms produce plain tensors (not MetaTensor) |
| `CUDA out of memory` during image encode | Full 480×480×256 → VAE at fp16 needs ~10 GB | Use `CUDA_VISIBLE_DEVICES=1` (96 GB A6000) |
| `val/loss not found` in callbacks | val_dataloader empty | Check datalist has `validation` entries; run `build_report2ct_datalist.py` with both splits |
| First training step loss = NaN | scale_factor = 0 or inf | Image embeddings may be all-zero; verify a few `.nii.gz` files with `nibabel.load(p).get_fdata().std()` |

---

## 4. Key file locations

| File | Purpose |
|---|---|
| `data/report2ct_work_dir/ids_{train,valid}.txt` | Volume ID lists |
| `data/report2ct_work_dir/image_embeddings/<id>_emb.nii.gz` | MAISI VAE latents (H×W×D×C) |
| `data/report2ct_work_dir/image_embeddings/<id>_emb.nii.gzmulti_2560.json` | Merged dim/spacing/embeddings |
| `data/report2ct_work_dir/datalist.json` | Training datalist |
| `data/report2ct_work_dir/parity_refs/` | Parity reference files |
| `outputs/report2ct/<timestamp>/checkpoints/` | Saved model checkpoints (per run) |
| `configs/experiment/report2ct_repro.yaml` | Main experiment config |
| `configs/model/report2ct.yaml` | UNet + scheduler kwargs |
| `src/models/report2ct_module.py` | Lightning training loop |
| `src/data/report2ct_datamodule.py` | MONAI DataModule |
| `src/baselines/report2ct_text_encoder.py` | 3-encoder text pipeline |
| `src/baselines/report2ct_image_encoder.py` | MAISI VAE image pipeline |
