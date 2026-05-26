# VLM3D 2026 Task 4 — Report2CT-Beating Text→3D CT Generation

## Project goal
Submit to **MICCAI VLM3D 2026 Task 4 (Text-Conditional CT Generation)** by **2026-08-20** and beat the published winner **Report2CT** on FVD / 2.5D-FID / CLIPScore as measured by VLM3D-Dockers.

Pipeline (target): radiology report (findings + impression) + voxel spacing →
multi-encoder text + spacing conditioning → latent diffusion in MAISI VAE latent space
(`[B, 4, 120, 120, 64]`) → MAISI decoder → CT volume (`[B, 1, 480, 480, 256]`).

Plan: `/workspace/.omc/plans/vlm3d-pivot-plan.md` (Critic APPROVED, iter 3 consensus).

## Phase plan
- **Phase A (5/26 → 5/31, 5d)**: repo restructure on lightning-hydra-template + EDA + GenerateCT pretrained inference + Report2CT paper read + stub submission docker.
- **Phase B (6/1 → 6/30, 4w)**: Report2CT training-ready code via submodule adapter + 4 diagnostic modules + VLM3D-Dockers eval. **User runs the multi-day Report2CT training** ([[report2ct-training-is-user-owned]]).
- **Phase C (7/1 → 7/31, 4w)**: our v1 model + ablations.
- **Phase D (8/1 → 8/20, 3w)**: final + submission docker.

## Environment
DeepCGV-Mk7, up to **3× A6000 Blackwell (96 GB each)**. Docker dev container. Python 3.11, PyTorch 2.x, MONAI 1.4, lightning 2.4, Hydra 1.3, transformers 4.46, diffusers 0.31, wandb.

## Repo layout (post Phase A restructure)
```
src/                      # lightning-hydra-template base + our additions
  data/                   # LightningDataModule (CT-RATE, MAISI latents)
  models/                 # ours_v1+ LightningModule
  baselines/              # adapter LightningModules for Report2CT, GenerateCT
  diagnostics/            # cross_attn, retrieval, counterfactual, token_region
  train.py, eval.py       # Hydra entrypoints

configs/                  # Hydra hierarchy (data/model/trainer/logger/callbacks/experiment/...)

third_party/              # READ-ONLY submodules (P2)
  report2ct/              # SHA 7b483a8 — paper code + train.sh + JSON configs (weights NOT released)
  generatect/             # SHA 2a81135 — has 3 pretrained .pt ckpts on HF
  vlm3d_dockers/          # SHA c73fe07 — official VLM3D-Dockers eval containers (Task 4 = FVD + CLIPScore + 2.5-D FID)

tests/                    # pytest scaffold (test_hydra_compose, test_maisi_frozen_load, +placeholders)
notebooks/                # eda.ipynb (lands Day 4-5)
docs/                     # submodule_pins.md, ct_clip_check.md, report2ct_training_handoff.md (Day 2+)
submission/               # Phase A Day 5 stub submission docker → Phase D production
results/                  # results/upper_bound.json (1mm MAISI VAE PSNR 30.94 baseline)
data/                     # new artifacts only (data/checkpoints/...)
maisi_bundle/             # FROZEN MAISI VAE (autoencoder.pt) — R6 mitigation: test_maisi_frozen_load.py
paper_pdf/                # Report2CT.pdf + others (reference)
deprecated/               # all triplane-era work (do not import from here)
```

## Dataset reference
- **Raw CT (CT-RATE)** at `/workspace/datasets/datasets/CT-RATE/dataset/`
  - `train_fixed/`: 20,000 patient dirs, 47,148 scans (NIfTI)
  - `valid_fixed/`: 1,304 patient dirs, 3,038 scans
  - `metadata/{train,validation}_metadata.csv`: spacing, kernel, manufacturer, etc.
  - `multi_abnormality_labels/`: 18 binary labels per scan
  - `radiology_text_reports/`: free-text reports (findings + impression)
- **Storage convention**: `/workspace/datasets/` is collaborator's read-only area. New artifacts → `/workspace/data/`.
- **GPU convention**: prefix scripts with `CUDA_VISIBLE_DEVICES=0` for single-GPU; explicit `CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 ...` for multi-GPU.

## Win condition (frozen)
On CT-RATE valid 1000-split, via VLM3D-Dockers: `ours_final` beats `report2ct_our_repro` in ≥2 of {2.5D-FID, CLIPScore-T2I, FVD}. Metric priority for headline: **2.5D-FID > CLIPScore-T2I > FVD**.

## Envelope (Report2CT 3-TE-cfg5 anchor)
- 2.5D-FID anchor = 4.04 (Fig 6 FID Avg) ⇒ ±15%
- CLIPScore-T2I anchor = 59.93 (Fig 5) ⇒ ±10%
- FVD anchor = self-measured on 6/1 (paper does NOT report FVD) ⇒ ±25%

## Upper bound (preserved from earlier work)
MAISI VAE encode→decode round-trip on 1000 CT-RATE valid: **PSNR 30.94 ± 2.97 dB, SSIM 0.7195 ± 0.1084**. Intensity: HU clipped to `[-1000, 1000]`, scaled to `[0, 1]`; spatial 480×480×256. Full: `results/upper_bound.json`.

## Compute / I/O notes
3D MAISI latent streaming (`mu.pt`, 7.2 MB/sample × 6,000 ≈ 43 GB) **dominates runtime** for any per-sample sweep. Defaults: `--device cpu --num-workers 16` for cheap arithmetic; GPU only when inner ops justify it. See [[3d-latent-i-o-bottleneck]] memory.

## Non-goals
- Triplane autoencoder research ([[triplane-deprecated-2026-05]]).
- Report2CT full training (user runs it directly — [[report2ct-training-is-user-owned]]).
- New dataset preprocessing beyond what VLM3D-Dockers expects.
