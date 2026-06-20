# Report2CT Small-Scale Training — Implementation Plan

**Plan ID**: `report2ct-smallrun-plan`
**Source spec**: `/workspace/.omc/specs/deep-interview-report2ct-small-scale-train.md` (5 rounds, 20% ambiguity, PASSED)
**Phase**: B Day 1 (small-scale code-path verification before full Phase B run)
**Generated**: 2026-05-27
**Revised**: 2026-05-27 (iter 2, addressing Architect REVISE + Critic REJECT)
**Mode**: RALPLAN-DR SHORT
**Status**: pending approval (iter 2)

---

## 1. Requirements Summary

### Goal
Run a **small-scale Report2CT training** (1000 train + 500 valid samples drawn from collaborator's `latents/{train,valid}/` directories — see split-semantics note below; paper hyperparams except `n_epochs: 100 → 10`; ~4 h wall-clock on a single A6000) using freshly re-encoded MAISI latents (via submodule `third_party/report2ct/src/maisi/scripts/vlm3d_image_embedding.py`) and precomputed 3-text-encoder embeddings. **Success = train + valid loss decreasing over 10 epochs with no NaN/divergence**, proving the data + model + launcher pipeline is end-to-end correct before committing to the full Phase B run (20k+ scans).

### CRITICAL split-semantics note (resolves Critic C1)
Collaborator's `/workspace/datasets/datasets/latents/valid/` directory contains 1000 samples ALL prefixed `train_*` (verified empirically: `ls .../latents/valid/ | head` → `train_10001_a_1`, ...). Each `src.txt` resolves to `train_fixed/...`. CT-RATE's official `validation_reports.csv` uses `valid_*` VolumeNames — joining against it returns 0 hits.

**Therefore**: collaborator's "valid" subdir is a held-out random portion of CT-RATE's TRAIN split, NOT CT-RATE's official valid split. Both `latents/train/` and `latents/valid/` join against **`train_reports.csv` only**. We preserve which subdir each sample came from by writing a `split` field with values `"smallrun_train"` / `"smallrun_valid"` per entry, and we emit two separate datalist JSONs.

### Constraints
- **Subset size**: exactly 1000 train + 500 valid, alphabetical-first deterministic.
- **Latent precompute path**: re-encode via submodule. Do NOT bridge `mu.pt`.
- **GPU resources**: 2× A6000 for latent precompute, 1× A6000 for training.
- **Wall-clock budget**: ~1 h Stage 0 + ~30 min text + ~4-8 h latent + ~4 h training.
- **Paper hyperparams locked**: `batch_size=2`, `lr=1e-4`, RFlow (1000 steps, scale 1.4), CFG p=0.15, 3-TE. Only `n_epochs` deviates.
- **External code untouched**: `third_party/report2ct/` read-only (P2).
- **MAISI VAE**: frozen via `src.baselines.maisi.load_frozen`.
- **Documented P2 deviation**: submodule's `PolynomialLR(total_iters = n_epochs × N/batch_size, power=2.0)` (training script L200,L515) compresses the schedule 10× when `n_epochs=10`. Sub-option chosen: **document as P2 deviation** (lr decays to ~0 by epoch ~6; treat epochs 7-10 as low-lr fine-tune). Loss-monotonicity AC (AC-5.1) evaluates epoch-10 vs epoch-2 to absorb this effect.

### Non-Goals
- NOT matching paper FID/CLIP envelope.
- NOT VLM3D-Docker eval.
- NOT stratified sampling by labels.
- NOT bridging `mu.pt`.
- NOT 1-TE simplification.
- NOT super-resolution.

---

## 2. RALPLAN-DR (SHORT mode)

### Principles (5)
1. **Submodule read-only** — `third_party/report2ct/` and `third_party/maisi_bundle/` never edited.
2. **Paper-faithful schedule** — only `n_epochs` deviates (PolynomialLR compression acknowledged downstream).
3. **Loss-only sanity** — no FID/CLIP/FVD. PASS = epoch-10 avg ≤ 0.7 × epoch-2 avg + no NaN.
4. **Deterministic alphabetical selection** — no silent backfill on shortfall.
5. **GPU budgeted** — Stage 0 gate fails fast on 15 samples.

### Decision Drivers (top 3)
1. **Code-path verification** — every interface executes end-to-end.
2. **Fail-fast staging** — Stage 0 catches R1/R3/R6 in ~1 h before 6 h Stage 1.
3. **Format fidelity over speed** — submodule's own encode driver guarantees compatibility.

### Viable Options

#### Option A (initially proposed) — single-shot 1500 encode
- Pros: format guaranteed-correct.
- Cons: 10 h committed before any verification.

#### Option B (REJECTED) — Bridge existing `mu.pt`
- Pros: no encoding wall-clock.
- Cons / invalidation: triplane-purpose `mu.pt` has unknown normalization; scale_factor masks mismatch.

#### Option C (REJECTED) — GenerateCT smoke
- Pros: faster.
- Cons / invalidation: different scheduler/encoder; doesn't validate Report2CT path.

#### Option D (CHOSEN) — Staged: Stage 0 (15 samples) → Stage 1 (1500 samples) → Stage 2 (training)
- Pros: fail-fast on R1/R3/R6 in ~1 h on 15 samples. Same submodule code paths exercised end-to-end at both scales. If Stage 0 fails, no encode budget wasted.
- Cons: +1 h overhead vs A. Stage 0 cannot detect data-scale-dependent issues (acceptable).
- Why chosen over A: net negative wall-clock under realistic failure; ≤10% overhead best-case for substantial risk reduction. A dominated by D without prior empirical evidence the full path works.

---

## 3. Acceptance Criteria (25 total)

### Phase 0: Stage-0 gate (15-sample smoke)
- [ ] **AC-0.1** `data/report2ct_smallrun_stage0/datalist_{train,valid}.json` (10 train + 5 valid) exist; bundled under `"training"` per submodule contract. First 10/5 of `sorted(os.listdir('/workspace/datasets/datasets/latents/{train,valid}/'))`, both joined against `train_reports.csv`.
- [ ] **AC-0.2** All 15 `_emb.nii.gz` written under `data/report2ct_smallrun_stage0/image_emb/` via **stock** `vlm3d_image_embedding.py` (single-GPU). **Precondition (resolves Architect N2)**: BEFORE invoking the stock driver, force-symlink `third_party/maisi_bundle/models/autoencoder.pt` → driver-expected location `third_party/report2ct/vlm3D_work_dir/models/autoencoder_epoch273.pt` AND override `trained_autoencoder_path` in the driver's env_config so Stage 0 uses the EXACT SAME ckpt as Stage 1. Else Stage 0 PASS gives false R1 confidence (driver would otherwise download a different NVIDIA-hosted checkpoint at `vlm3d_image_embedding.py:156`). Assert symlink exists before driver launch.
- [ ] **AC-0.3** All 15 text JSONs `<vol>_emb.nii.gzmulti_2560.json` **co-located** in the same `image_emb/` directory.
- [ ] **AC-0.4** Stage-0 training runs 10 epochs single-GPU and emits `diff_unet_smallrun_stage0_best.pt`.
- [ ] **AC-0.5** Stage-0 satisfies AC-5.1 + AC-5.2 + AC-5.3 (Stage 0 variant). **GATE**: PASS is hard prerequisite for Stage 1.

### Phase 1: Sample-report join (owner: `src/data/report2ct_smallrun.py`)
- [ ] **AC-1.1** `data/report2ct_smallrun/datalist_train.json` (1000 entries) + `datalist_valid.json` (500); each `{"training": [...]}`.
- [ ] **AC-1.2** Each entry has `volume_name, nifti_path, latent_emb_path, text_emb_path, findings, impression, spacing, split`. `split ∈ {"smallrun_train", "smallrun_valid"}`.
- [ ] **AC-1.3** 100% entries have non-empty `findings` and `impression`.
- [ ] **AC-1.4** `latent_emb_path` and `text_emb_path` BOTH under `/workspace/data/report2ct_smallrun/image_emb/` (per Critic C7).
- [ ] **AC-1.5** First 1000 of `sorted(os.listdir(latents/train))` + first 500 of `sorted(os.listdir(latents/valid))`, both joined against `train_reports.csv` only.
- [ ] **AC-1.6** (R4 hard abort) If intersection < 1500, writes `.dropped_volumes.json` + exits non-zero. **No silent backfill.**

### Phase 2: Latent precompute (Stage 1)
- [ ] **AC-2.0** (MAISI ckpt key-diff guard) Before encode: `strict=True load_state_dict` on a fresh `AutoencoderKlMaisi` instance **built from `third_party/report2ct/vlm3D_work_dir/config_maisi_2560.json` `autoencoder_def` block (NOT `maisi_bundle/configs/inference.json`)** so the instance matches Stage 1's actual encoder schema (resolves Architect N3 — `num_splits: 4` in Report2CT config vs `num_splits: 2` in maisi_bundle would cause false-FAIL on conv-split tensors). Assert `set(saved_keys) ^ set(model_keys) == set()`. Abort on diff with diagnostic.
- [ ] **AC-2.1** All 1500 `_emb.nii.gz` under `data/report2ct_smallrun/image_emb/`.
- [ ] **AC-2.2** Spot-check 5: 4-channel, no NaN/Inf.
- [ ] **AC-2.3** Wall-clock ≤ 8 h on 2 GPUs (logged to `.encode_walltime.json`).
- [ ] **AC-2.4** No CUDA OOM in `.encode_stderr.log`.

### Phase 2.5: Post-encode launcher smoke
- [ ] **AC-2.5** After Phase 2: `bash scripts/run_report2ct_smallrun.sh --dry-run` reaches `calculate_scale_factor` block + stdout contains `scale_factor ->`. Runs AFTER Phase 2 because the call requires `_emb.nii.gz` to exist.

### Phase 3: Text embedding precompute
- [ ] **AC-3.1** 1500 `<vol>_emb.nii.gzmulti_2560.json` files in `data/report2ct_smallrun/image_emb/` (SAME dir).
- [ ] **AC-3.2** Spot-check 5: `findings_embeddings` shape `(L_f, 2560)`, `impression_embeddings` shape `(L_i, 2560)`, `spacing` 3-vector.
- [ ] **AC-3.3** 3-TE commit hashes in `text_emb_pins.json`.

### Phase 4: Training launcher
- [ ] **AC-4.1** Per-run `config_maisi_diff_model_vlm3D_smallrun.json` sets `n_epochs == 10`. Hydra meta mirrors `training.n_epochs == 10`.
- [ ] **AC-4.2** `env_config.json` overrides submodule paths; **NO separate `text_emb_dir`** (Critic C7).
- [ ] **AC-4.3** `bash -n scripts/run_report2ct_smallrun.sh` exits 0.
- [ ] **AC-4.4** Hydra config does NOT contain `text_emb_dir`.

### Phase 5: Validation / sanity
- [ ] **AC-5.1** (per-epoch avg) Parse training script L562 `epoch {E} average loss: {L:.4f}`. PASS = `epoch_10_avg ≤ 0.7 × epoch_2_avg`. Skip epoch 1 (scale_factor warm-up).
- [ ] **AC-5.2** No NaN/Inf in per-iter or per-epoch loss lines.
- [ ] **AC-5.3** Final `data/checkpoints/report2ct/smallrun/diff_unet_smallrun_best.pt` exists (submodule writes `_best.pt` at L568, NOT `_last.pt`). Launcher post-step also creates `last.ckpt` symlink → `_best.pt` to satisfy spec wording.
- [ ] **AC-5.4** (REQUIRED `eval_val_smallrun.py`) Loads `_best.pt`, runs forward-only pass with identical scale_factor + RFlow + `loss_pt` over `datalist_valid.json`, writes `runs/.../val_loss.csv`. **`scale_factor` is loaded directly from the `_best.pt` checkpoint dict's `"scale_factor"` key** (training script saves it at L397; resume loads it at L497-506). Do NOT recompute via `first(train_loader)` — `partition_dataset(shuffle=True)` at L485-487 + `DataLoader(shuffle=True)` at L111 make the "first batch" non-reproducible across runs. Defensive: if `"scale_factor"` key missing in ckpt dict, abort with diagnostic (Architect N1 resolution). PASS = epoch-10 valid loss finite + ≤ epoch-2 train loss.
- [ ] **AC-5.5** (Optional) Mid-train 1-sample decode at epochs 5, 10.
- [ ] **AC-5.6** One-line verdict: AC-5.1 + 5.2 + 5.3 + 5.4 PASS → smallrun PASS.

---

## 4. Implementation Steps

### Step 0 — Stage-0 (15-sample) gate
**Files**: `scripts/run_report2ct_smallrun.sh --stage 0` + `src/data/report2ct_smallrun.py --stage 0`

- Build datalists (10 train + 5 valid), join against `train_reports.csv` only.
- Run stock `vlm3d_image_embedding.py` single-GPU on all 15.
- Run text precompute, JSONs into `image_emb/`.
- Full 10-epoch training. Stage 0 dir: `model_filename=diff_unet_smallrun_stage0.pt`, `embedding_base_dir=data/report2ct_smallrun_stage0/image_emb/`.
- Evaluate AC-5.1 + 5.2 + 5.3 (Stage 0). PASS → Stage 1.

### Step 1 — Sample-report join
**File**: `src/data/report2ct_smallrun.py` (~180 lines)

- `select_volume_names(subdir, n)` — `sorted(os.listdir('latents/{subdir}/'))[:n]`.
- `load_reports()` — `train_reports.csv` only (NOT validation).
- `read_src_nifti_path(subdir, vol)` — `latents/<subdir>/<vol>/src.txt` first line.
- `read_spacing(nifti_path)` — `nibabel.load(p).header.get_zooms()[:3]`.
- `build_datalist(stage)` — emits `datalist_train.json` (split="smallrun_train") + `datalist_valid.json` (split="smallrun_valid"). On shortfall < target: write `.dropped_volumes.json`, exit non-zero.

### Step 2 — Latent precompute (2-GPU Stage 1)
**Files**: `scripts/precompute_latents_smallrun.sh` + `scripts/precompute_latents_worker.py`

(Skipped at Stage 0 — Stage 0 uses stock driver.)

Bypasses `vlm3d_image_embedding.py`'s hard-coded `num_gpus=1`. **Mechanism**: per-rank slicing at `diff_model_create_training_data_vlm3D_all.py:378` via `if _iter % world_size != local_rank: continue` (modulo-stride). `partition_dataset` is used at training script L485 for the DataLoader, NOT in encode.

R1 mitigation: symlink + env_config override + AC-2.0 strict-load guard.

Worker modes: `--stage-csv`, `--verify --sample N`, `--record-walltime`, `--ckpt-keydiff` (AC-2.0).

### Step 3 — Text embedding precompute (3-TE)
**File**: `scripts/precompute_text_emb_smallrun.py` (~250 lines)

1. Load 3 HF encoders on `cuda:0`: MedEmbed-large-v0.1 / ClinicalBERT / BiomedVLP-CXR-BERT-specialized.
2. Per sample: tokenize findings + impression → mask-weighted mean pool (1024+768+768=2560) → concat → write JSON.
3. Filename: `<embedding_base_dir>/<volume_name>_emb.nii.gzmulti_2560.json` (SAME dir as `_emb.nii.gz`).
4. R6 pre-launch check: forward `(1, 2, 2560)` through `report2ct_adapter.build_unet()`.

### Step 4 — Submodule env config override
**File**: `data/report2ct_smallrun/env_config.json`

```json
{
  "data_base_dir":            "/workspace/datasets/datasets/CT-RATE/dataset/train_fixed",
  "embedding_base_dir":       "/workspace/data/report2ct_smallrun/image_emb",
  "json_data_list":           "/workspace/data/report2ct_smallrun/datalist_train.json",
  "json_data_list_val":       "/workspace/data/report2ct_smallrun/datalist_valid.json",
  "model_dir":                "/workspace/data/checkpoints/report2ct/smallrun",
  "model_filename":           "diff_unet_smallrun.pt",
  "output_dir":               "/workspace/runs/report2ct_smallrun",
  "output_prefix":            "unet_3d_smallrun",
  "trained_autoencoder_path": "/workspace/third_party/maisi_bundle/models/autoencoder.pt",
  "existing_ckpt_filepath":   null
}
```

**Removed** vs iter-1: standalone `text_emb_dir` (Critic C7).

### Step 5 — Training launcher
**File**: `scripts/run_report2ct_smallrun.sh`

- Generates per-run schedule config (`n_epochs: 10`, `n_epochs: 1` for `--dry-run`).
- `CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 -m ...`.
- Wall-clock watcher: epoch-boundary SIGTERM only, after ≥2 epochs (tails log for `epoch {E} average loss:` markers).
- Post-step: `ln -sfn diff_unet_smallrun_best.pt last.ckpt`.

### Step 6 — Hydra meta config
`configs/experiment/report2ct_smallrun.yaml` for run logging; no `text_emb_dir`.

### Step 7 — Sanity checker + valid eval
**Files**: `scripts/check_smallrun_sanity.py` + `scripts/eval_val_smallrun.py` (REQUIRED)

`check_smallrun_sanity.py`:
- `--check loss-monotonic` (AC-5.1): per-epoch avg, `epoch_10 ≤ 0.7 × epoch_2`.
- `--check no-nan` (AC-5.2): scan all loss values.
- `--check valid-loss` (AC-5.4): assert `val_loss.csv` exists, finite, ≤ epoch-2 train.
- `--final-verdict`: AND of AC-5.1+5.2+5.3+5.4.

`eval_val_smallrun.py`:
- Load `_best.pt` + UNet from `config_maisi_2560.json`.
- Same `RFlowScheduler` + `loss_pt` (MSE).
- **`scale_factor` is loaded from `_best.pt["scale_factor"]` ckpt-dict key** (training script L397 persists it). NOT recomputed (Architect N1 — `partition_dataset(shuffle=True)` makes first-batch non-deterministic).
- **Datalist for valid**: invoke `load_filenames(json_data_list_val)` MANUALLY — training script only reads `json_data_list` at L461; `json_data_list_val` in env_config is informational only (Architect N6).
- Forward-only over `datalist_valid.json`. Append to `val_loss.csv`.

---

## 5. Risks and Mitigations

| ID | Risk | Mitigation |
|---|---|---|
| **R1** | MAISI ckpt filename mismatch | env_config override + **AC-2.0 strict-load key-diff guard** + symlink fallback |
| **R2** | 2-GPU parallel encoding (driver hard-codes num_gpus=1) | Bypass driver, invoke inner module via torchrun. **Per-rank slicing via modulo-stride at L378 (`_iter % world_size != local_rank`), NOT `partition_dataset` (L485 is training-time only)** |
| **R3** | `numpy<2.0` vs submodule `numpy==2.3.4` | Stage 0 smoke + `python -c "from ...import diff_model_train_..."` |
| **R4** | Sample-report join intersection < 1500 | **Abort with `.dropped_volumes.json`**. No silent backfill |
| **R5** | Wall-clock blowout | **Epoch-boundary SIGTERM only, after ≥2 epochs.** Watcher tails log for boundary markers |
| **R6** | 3-TE concat shape `(B,2,2560)` | Stage 0 + R6 pre-launch forward check |
| **R7** | Split-prefix collision (collaborator's valid/ has train_* IDs) | Both subdirs join `train_reports.csv`; explicit `split` field; two separate datalists |
| **R8** | PolynomialLR compression (lr→0 by epoch ~6) | **Documented P2 deviation.** AC-5.1 uses epoch-10-vs-epoch-2 to absorb. Alternative override deferred to follow-up |

---

## 6. Verification Steps

```bash
# Phase 0 — Stage-0 gate (must PASS before Phase 1+)
python -m src.data.report2ct_smallrun --stage 0
bash scripts/run_report2ct_smallrun.sh --stage 0
python scripts/check_smallrun_sanity.py --run-dir runs/report2ct_smallrun_stage0/ --final-verdict

# Phase 1
python -m src.data.report2ct_smallrun
python -c "import json; tr=json.load(open('data/report2ct_smallrun/datalist_train.json'))['training']; va=json.load(open('data/report2ct_smallrun/datalist_valid.json'))['training']; assert len(tr)==1000 and len(va)==500; assert all(e.get('split') in ('smallrun_train','smallrun_valid') for e in tr+va)"

# Phase 2
python scripts/precompute_latents_worker.py --ckpt-keydiff   # AC-2.0
bash scripts/precompute_latents_smallrun.sh
find data/report2ct_smallrun/image_emb -name '*_emb.nii.gz' | wc -l   # → 1500
python scripts/precompute_latents_worker.py --verify --sample 5

# Phase 2.5
bash -n scripts/run_report2ct_smallrun.sh
bash scripts/run_report2ct_smallrun.sh --dry-run   # → log contains 'scale_factor ->'

# Phase 3
python scripts/precompute_text_emb_smallrun.py \
    --datalist-train data/report2ct_smallrun/datalist_train.json \
    --datalist-valid data/report2ct_smallrun/datalist_valid.json \
    --out-dir data/report2ct_smallrun/image_emb   # SAME dir
find data/report2ct_smallrun/image_emb -name '*multi_2560.json' | wc -l   # → 1500

# Phase 5
bash scripts/run_report2ct_smallrun.sh
ls data/checkpoints/report2ct/smallrun/diff_unet_smallrun_best.pt
ls data/checkpoints/report2ct/smallrun/last.ckpt   # symlink
python scripts/eval_val_smallrun.py --ckpt .../diff_unet_smallrun_best.pt --datalist data/report2ct_smallrun/datalist_valid.json --out runs/report2ct_smallrun/val_loss.csv
python scripts/check_smallrun_sanity.py --run-dir runs/report2ct_smallrun/ --final-verdict
```

---

## 7. Dependencies on External Repos

| Dependency | Path | Mode |
|---|---|---|
| Stage-0 encode driver | `third_party/report2ct/src/maisi/scripts/vlm3d_image_embedding.py` | invoked as-is at Stage 0 |
| Stage-1 encode inner module | `third_party/report2ct/src/maisi/scripts/diff_model_create_training_data_vlm3D_all.py` | `torchrun -m ...` (modulo-stride at L378) |
| Training script | `third_party/report2ct/src/maisi/scripts/diff_model_train_vlm3D_2560_multi_text.py` | `torchrun -m ...` |
| Model arch config | `third_party/report2ct/vlm3D_work_dir/config_maisi_2560.json` | read-as-is |
| Training schedule template | `third_party/report2ct/vlm3D_work_dir/config_maisi_diff_model_vlm3D.json` | copied + `n_epochs` overridden |
| Env config template | `third_party/report2ct/vlm3D_work_dir/environment_maisi_diff_model_vlm3D_FI_2560_multi.json` | template only |
| MAISI VAE ckpt (frozen) | `third_party/maisi_bundle/models/autoencoder.pt` | path-injected + AC-2.0 key-diff |
| 3 HF text encoders | `abhinand/MedEmbed-large-v0.1`, `medicalai/ClinicalBERT`, `microsoft/BiomedVLP-CXR-BERT-specialized` | `AutoModel.from_pretrained` |
| CT-RATE NIfTI volumes | `/workspace/datasets/datasets/CT-RATE/dataset/train_fixed/.../*.nii.gz` | read-only (all samples resolve here, incl. `smallrun_valid`) |
| Reports CSV | `/workspace/datasets/datasets/CT-RATE/dataset/radiology_text_reports/train_reports.csv` | read-only; **`validation_reports.csv` NOT used** |
| Phase A — MAISI loader | `src/baselines/maisi.py::load_frozen` | optional + AC-2.0 helper |
| Phase A — Report2CT adapter | `src/baselines/report2ct_adapter.py::build_unet` | R6 forward check |
| Phase A — CT-RATE DataModule | `src/data/ct_rate_datamodule.py::load_records` | optional helper |

---

## ADR — Architecture Decision Record

**Decision**: Re-encode 1500 selected CT-RATE volumes in two stages: Stage 0 (15 samples, stock single-GPU driver, ~1 h) as hard pass-gate; Stage 1 (1485 remaining, 2-GPU torchrun over inner module modulo-stride at L378, ~6 h) only if Stage 0 PASSES. Text embeddings co-located in same `image_emb/` dir as `_emb.nii.gz` (Critic C7). Invoke `diff_model_train_vlm3D_2560_multi_text` via custom env_config + per-run schedule (only `n_epochs: 100 → 10`). Final ckpt is `_best.pt`; `last.ckpt` symlink satisfies spec wording. Required `eval_val_smallrun.py` produces `val_loss.csv`. Sanity = epoch-10 avg train loss ≤ 0.7 × epoch-2 avg + no NaN + `_best.pt` exists + valid loss finite.

**Decision Drivers**: (1) code-path verification, (2) fail-fast staging, (3) format fidelity over speed.

**Alternatives considered**:
- **A — Single-shot 1500 (no Stage 0)**. Rejected — dominated by D. Commits 10 h before any verification; saves only ~1 h best-case.
- **B — Bridge `mu.pt`**. Rejected — silent normalization mismatch.
- **C — GenerateCT smoke**. Rejected — different scheduler/encoder.
- **D — Staged (Stage 0 → Stage 1) (CHOSEN)**. See Option D analysis in §2.

**Why chosen** (D over A): worst-case (any R1/R3/R6 hit) saves ~6 h; best-case adds ~1 h to a 10-14 h run (≤10% overhead). Asymmetry sharply in D's favor without prior empirical evidence.

**Consequences**:
- ~6 GB disk under `image_emb/` (+ 60 MB Stage 0).
- ~7 h GPU encode total (1 h Stage 0 + 6 h Stage 1).
- `eval_val_smallrun.py` now REQUIRED.
- PolynomialLR compresses 10×; accepted as P2 deviation (R8). AC-5.1 absorbs.
- Spec `last.ckpt` reconciled to `_best.pt` via symlink (Critic C4/C5). Spec text effectively updates.

**Follow-ups**:
1. PASS → scale to full Phase B (`n_epochs: 100`, paper PolynomialLR shape applies unmodified).
2. FAIL → root-cause via `smallrun_failure_diagnosis.md`. Likely: R3, R6, R1.
3. Promote `eval_val_smallrun.py` to periodic-eval hook (every-epoch checkpointing).
4. Add `report2ct_smallrun.yaml` to `.omc/wiki`.
5. `text_emb_pins.json` source-of-truth for HF revisions.
6. Update spec to reflect `_best.pt` reality so future readers don't hit C4/C5 again.

---

## Iteration 2 Changelog

### Critic-found (CRITICAL + MAJOR)
| ID | Item | Status | Where |
|---|---|---|---|
| **C1** | train/valid prefix collision (verified empirically) | **FIXED** | §1 split-semantics note + AC-1.1/1.2/1.5 + R7 + §7 |
| **C2** | R4 abort, not silent backfill | **FIXED** | §5 R4 + AC-1.6 |
| **C3** | AC-4.4 phase-ordering bug (dry-run requires Phase 2) | **FIXED** | Moved to AC-2.5 (Phase 2.5) |
| **C4** | `_best.pt` vs `_last.pt` filename | **FIXED** | AC-5.3 + launcher symlink |
| **C5** | spec/plan path reconcile | **FIXED** | AC-5.3 + ADR Consequences + follow-up #6 |
| **C7** | text_emb_dir co-location | **FIXED** | AC-1.4, AC-3.1, AC-4.2, AC-4.4, env_config (removed text_emb_dir), Step 3, Step 4 note |

### Architect-found (9 items, all ACCEPTED by Critic)
| ID | Item | Status | Where |
|---|---|---|---|
| **A1** | Stage 0 (15-sample) gate | **FIXED** | New Phase 0 + AC-0.1–0.5 + Step 0 + ADR Option D |
| **A2** | Fix R2 description (modulo stride) | **FIXED** | R2 + Step 2 cite L378 (vs L485 training-time) |
| **A3** | MAISI ckpt key-diff guard | **FIXED** | AC-2.0 + worker `--ckpt-keydiff` |
| **A5** | `eval_val_smallrun.py` REQUIRED | **FIXED** | AC-5.4 + Step 7 second script + follow-up #3 |
| **A6** | PolynomialLR shape compression | **FIXED** (documented as P2 deviation) | §1 + R8 + Consequences |
| **A7** | R5 SIGTERM at epoch boundary only | **FIXED** | R5 + Step 5 watcher (≥2 epochs gate) |
| **A8** | AC-5.1 per-epoch avg | **FIXED** | AC-5.1 (epoch_10 ≤ 0.7 × epoch_2) + Step 7 parser switch |
| **A9** | Option D in ADR | **FIXED** | §2 Option D + ADR Alternatives + Why chosen D over A |

## Iteration 3 Changelog (Critic APPROVE)

Three surgical fixes from Architect/Critic iter-2 light REVISE applied directly:

| ID | Item | Status | Where |
|---|---|---|---|
| **N1** | Load `scale_factor` from `_best.pt["scale_factor"]` (training L397 saves; L497-506 resumes) — NOT recompute on shuffled train loader | **FIXED** | AC-5.4 + Step 7 `eval_val_smallrun.py` |
| **N2** | Force-symlink `maisi_bundle/autoencoder.pt` → driver-expected `autoencoder_epoch273.pt` BEFORE Stage 0 stock driver (else Stage 0 uses different ckpt than Stage 1) | **FIXED** | AC-0.2 |
| **N3** | AC-2.0 builds fresh `AutoencoderKlMaisi` from `config_maisi_2560.json` `autoencoder_def` (NOT `maisi_bundle/inference.json`) — `num_splits: 4` vs `2` schema match | **FIXED** | AC-2.0 |
| N6 | `eval_val_smallrun.py` invokes `load_filenames(json_data_list_val)` manually (training script only reads `json_data_list` at L461) | **APPLIED** | Step 7 |

**Final verdict path**: REJECT (iter 1) → REVISE (iter 2) → **APPROVE (iter 3)**. Plan status: **pending approval** (user execution gate).
