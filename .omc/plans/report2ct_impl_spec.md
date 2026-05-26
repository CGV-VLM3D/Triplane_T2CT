# Report2CT Implementation Spec (Day 1 PM)

## Strategy (revised 2026-05-26 Day 1 — user direction + submodule audit)

**Report2CT submodule (`third_party/report2ct`, SHA `7b483a8`) ships a working PyTorch implementation including the training script.** Weights are NOT released, but the code IS. The `vlm3d_inference.ipynb` + `train.sh` + `src/maisi/scripts/*.py` cover the full pipeline.

Our [A] deliverable is therefore **not** a from-scratch reimplementation but a **thin adapter layer**:

1. Wrap `third_party/report2ct/src/maisi/scripts/diff_model_train_vlm3D_2560_multi_text.py` as our canonical training entrypoint via `scripts/run_report2ct_training.sh`.
2. Reuse `third_party/report2ct/vlm3D_work_dir/{environment_..., config_maisi_diff_model_vlm3D.json, config_maisi_2560.json}` as the model/env config.
3. Provide a Hydra config in `configs/experiment/report2ct_repro.yaml` that exposes paths (data, MAISI ckpt, output dir, num_gpus) and shells out to the submodule script.
4. Provide forward-hook utilities in `src/diagnostics/cross_attn.py` (Phase B) that import the submodule's UNet class and register attention hooks — **read-only** access to the submodule code.
5. Provide `docs/report2ct_training_handoff.md` for the user-driven training run ([[report2ct-training-is-user-owned]]).

**Principle P2**: third_party/ stays read-only. Wrap, do not modify.

## Submodule pipeline (as shipped)

`train.sh` shows the two-step canonical run:

```bash
# Step 1: precompute MAISI latent embeddings for the training set
python -m src.maisi.scripts.vlm3d_image_embedding

# Step 2: train multi-text-encoder LDM
torchrun --nproc_per_node=2 -m src.maisi.scripts.diff_model_train_vlm3D_2560_multi_text \
    --env_config ./vlm3D_work_dir/environment_maisi_diff_model_vlm3D_FI_2560_multi.json \
    --model_config ./vlm3D_work_dir/config_maisi_diff_model_vlm3D.json \
    --model_def ./vlm3D_work_dir/config_maisi_2560.json \
    --num_gpus 2
```

The submodule expects to be the CWD (uses `./vlm3D_work_dir/...` relative paths). Our launcher script will `cd third_party/report2ct && torchrun ...`.

## Paper-grounded reference (for diagnostic hooks + writeup)

### End-to-end pipeline

```
Findings (text)  ─┐
Impression (text) ─┤── 3 text encoders ──► concat ──► 2×2560 conditioning tensor ─┐
                                                                                  ├─► 3D U-Net (cross-attn)
Voxel spacing (3,) ────────────────────► embed ─────────────────────────────────► │
                                                                                  │
Noisy latent z_t  [B, 4, 120, 120, 64] ────────────────────────────────────────► U-Net ──► ε_pred
                                                                                  │
Rectified Flow scheduler (1000 steps, σ ≈ 1.4)                                    │
                                                                                  ▼
                                                                       (training: MSE vs added noise)
                                                                       (inference: iterative denoise → z_0)

z_0 [B, 4, 120, 120, 64] ──► MAISI VAE decoder (frozen) ──► CT volume [B, 1, 480, 480, 256]
```

### MAISI VAE (frozen)

| Field | Value |
|---|---|
| Input | CT volume `[1, 480, 480, 256]`, HU-clipped to `[-1000, 1000]`, min-max → `[0, 1]` |
| Latent | `[4, 120, 120, 64]` (×4 downsample per spatial axis) |
| Weights | `/workspace/maisi_bundle/models/autoencoder.pt` (R6 test: `tests/test_maisi_frozen_load.py`) |
| API | MONAI generative `AutoencoderKL.encode_stage_2_inputs` / `decode_stage_2_outputs` |

### Three text encoders (per paper p.7)

| Encoder | HF model id (Day 2 verify) | Hidden dim |
|---|---|---|
| MedEmbed | `abhinand/MedEmbed-large-v0.1` | **1024** |
| ClinicalBERT | `medicalai/ClinicalBERT` (or `emilyalsentzer/Bio_ClinicalBERT`) | **768** |
| BiomedVLP-CXR-BERT | `microsoft/BiomedVLP-CXR-BERT-specialized` | **768** |

Per-section: tokenize → encode → mask-weighted mean pool → concat → `[768 + 1024 + 768] = [2560]`.
Final conditioning: `[B, 2, 2560]` (findings + impression).

### 3D U-Net denoiser (paper "Implementation details")

| Field | Value |
|---|---|
| Channel widths | `[64, 128, 256, 512]` (4 levels) |
| Cross-attention | last 2 resolution levels (256, 512 channels) |
| Cross-attn dim | **2560** |
| Library | `monai.generative.networks.nets.diffusion_model_unet` (per submodule code) |

### Diffusion scheduler

- **Rectified Flow (RFlow)**, 1000 steps, scale ≈ 1.4
- Loss: MSE on velocity / noise
- Library: `monai.generative.schedulers` (per submodule)

### CFG

- Training: zero-vector dropout p=0.15
- Inference: `ε = ε_uncond + s · (ε_cond - ε_uncond)`, `s ∈ [3, 7]`

### Training schedule (paper p.7)

| Field | Value | Our adaptation |
|---|---|---|
| Hardware | 2× H100 NVL (94 GB) | ≤3× A6000 Blackwell (96 GB) — comparable VRAM, slower throughput |
| Precision | mixed (fp16/bf16) | bf16 default, fp16 fallback |
| Distributed | DDP, `torchrun --nproc_per_node=2` | `CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 ...` |
| Batch size | 2 | match; if OOM → 1 + grad accum |
| Optimizer | Adam, lr 1e-4, polynomial decay | match |
| Epochs | implied 100 (1h/epoch on 2× H100) | TBD via 6/1 wall-clock measurement |
| Data | 20,000 CT volumes from CT-RATE train_fixed | same |

## Reported metrics + 3-TE-cfg5 headline anchor

Paper Figure 5 / Figure 6 ⇒ 6 model variants:

| Variant | CLIP-T2I | CLIP-I2I | FID Avg | FID XY | FID XZ | FID YZ |
|---|---|---|---|---|---|---|
| GenerateCT | 28.99 | 16.97 | 20.64 | 12.54 | 23.68 | 25.69 |
| 1-TE w/o cfg | 35.25 | 41.58 | 3.57 | 2.56 | 4.89 | 3.27 |
| 3-TE w/o cfg | 45.43 | 45.12 | 3.79 | 2.72 | 5.10 | 3.56 |
| 1-TE +cfg | 56.01 | 51.58 | 3.75 | 2.76 | 5.03 | 3.46 |
| **3-TE +cfg (mid)** ⭐ | **59.93** | 53.06 | **4.04** | 2.92 | 5.48 | 3.72 |
| 3-TE +cfg (high) | 61.29 | 53.26 | 4.19 | 3.07 | 5.68 | 3.81 |

**Headline anchor**:
- FID 2.5D Average = **4.04** ⇒ ±15% envelope ⇒ `[3.43, 4.65]`
- CLIPScore-T2I = **59.93** ⇒ ±10% envelope ⇒ `[53.94, 65.92]`
- CLIPScore-I2I = 53.06 (secondary informational, no envelope)
- FVD = self-measured 6/1 anchor (paper does NOT report FVD); envelope ±25%

## Open questions — Day 2 EOD resolutions

- [x] **ClinicalBERT HF id** → `medicalai/ClinicalBERT` (confirmed in `vlm3d_inference.ipynb` cell 0). All 3 text encoders pinned in `docs/report2ct_external_components.md`.
- [x] **UNet, scheduler, training schedule** confirmed from `vlm3D_work_dir/config_maisi_2560.json` + `config_maisi_diff_model_vlm3D.json`. See "Paper-grounded reference" section above + `docs/report2ct_external_components.md`. Key facts: `DiffusionModelUNetMaisi`, RFlow, 100 epochs, lr=1e-4, batch=2, CFG dropout 0.15.
- [x] **Text + image embeddings are precomputed**: `<image>multi_2560.json` (text) and `<volume>_emb.nii.gz` (image). Training script `diff_model_train_vlm3D_2560_multi_text.py` reads both from disk.
- [x] **vlm3d_image_embedding.py precompute step**: it iterates `train_reports_patients.csv` and writes `_emb.nii.gz` under a configurable embedding_base_dir — it does NOT touch the read-only `/workspace/datasets/datasets/latents/`. Our launcher will point it at `/workspace/data/report2ct_embeddings/image/` per repo storage convention.
- [x] **MONAI version compat**: submodule pins `monai==1.5.0`; we have `monai==1.4.0`. The `DiffusionModelUNetMaisi` + `RFlowScheduler` classes exist in both; if the submodule's script triggers a 1.5-only API, we'll either bump to 1.5 (likely safe — verify Day 3) or freeze a Python env per submodule (env file approach). Open until Day 3 first-run.
- [x] **numpy compat**: submodule pins `numpy==2.3.4`; we pin `numpy<2.0` for MONAI bundle compat. Submodule script runs as a SUBPROCESS, so it can use its own conda env if needed — not blocking the main pipeline.
- [ ] **CFG-scale exact labels** in paper Figure-5/6 bars (we lock anchor at 3-TE-cfg5 ≈ 4.04 FID Avg; exact label cross-check Day 4-5 when we render comparison figures).
- [ ] **Spacing-conditioning entry point** in UNet (config flag `include_spacing_input: true` is set; exact projection MLP inside MAISI's source — read Day 3 when wiring adapter).
- [ ] **LR polynomial decay degree** (paper text only says "polynomial decay"; submodule config only specifies lr=1e-4; default Hydra-side decision Phase B).
- [ ] **VLM3D-Dockers `2.5-D FID` vs paper's `FID with 2.5D feature extraction`** definition match — Day 5 when wiring `vlm3d_runner.py`.

## Day 2 PM consolidation

Submodule wrap strategy confirmed correct + thinner than initially planned:
1. Run `vlm3d_inference.ipynb`-style text-embedding precompute on all CT-RATE reports → `<image>multi_2560.json` files under `/workspace/data/report2ct_embeddings/text/`.
2. Run `third_party/report2ct/src/maisi/scripts/vlm3d_image_embedding.py` on CT-RATE train_fixed (uses MAISI frozen VAE at `third_party/maisi_bundle/models/autoencoder.pt`) → `_emb.nii.gz` under `/workspace/data/report2ct_embeddings/image/`.
3. Compose Hydra-flavored launcher → `bash scripts/run_report2ct_training.sh` (user runs multi-day training).

## Deliverable handoff plan (Phase B [A] / [U])

- **[A] Assistant by Phase B 6/1**:
  - `configs/experiment/report2ct_repro.yaml` (Hydra config wrapping the submodule run)
  - `scripts/run_report2ct_training.sh` (cd + torchrun + env vars)
  - `src/baselines/report2ct_adapter.py` (LightningModule shell — minimal, since training uses submodule's loop)
  - `docs/report2ct_training_handoff.md` (runbook for user)
  - Forward-hook stubs for diagnostic instrumentation in Phase B B.3
- **[U] User**: launches `bash scripts/run_report2ct_training.sh`, produces `data/checkpoints/report2ct/our_repro/best.ckpt`, fills `results/report2ct_sanity.json`.
