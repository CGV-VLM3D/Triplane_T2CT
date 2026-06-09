# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# VLM3D 2026 — Three-Task Submission Built on a CT Vision-Language Backbone

Companion: [GUIDE.md](GUIDE.md) is the developer tour (module-by-module walkthrough, EDA figures, GenerateCT inference recipes, Phase B next-actions). Read it before any non-trivial code change.

## Project goal
Submit to **MICCAI VLM3D 2026** by **2026-08-20** across the **three open tasks** (abnloc is **not open this year** — its `abnloc_*` dirs under `third_party/vlm3d_dockers/` are last-year artifacts):

1. **ctgen** — text → 3D CT generation (Task 4 headline; beat Report2CT on FVD / 2.5D-FID / CLIPScore via VLM3D-Dockers)
2. **reportgen** — CT → radiology report
3. **abnclass** — CT → 18 binary abnormality labels

**Unifying technical bet**: a strong **CT vision-language backbone** (CT-CLIP / fVLM) that powers (a) text-conditioning for ctgen, (b) image features for reportgen + abnclass, and (c) retrieval-based diagnostics. CT-CLIP is the de-facto encoder inside all three official eval dockers, so backbone quality directly moves the official metrics.

**Execution order**: VLM backbones → downstream task heads (cls / reportgen / retrieval) → generator pivot. We do *not* design the generator until at least one VLM baseline produces measured zero-shot numbers — those numbers are the conditioning signal the generator would otherwise be tuned blind against.

ctgen pipeline (target): radiology report (findings + impression) + voxel spacing →
multi-encoder text + spacing conditioning → latent diffusion in MAISI VAE latent space
(`[B, 4, 120, 120, 64]`) → MAISI decoder → CT volume (`[B, 1, 480, 480, 256]`).

Plan: `/workspace/.omc/plans/vlm3d-pivot-plan.md` (Critic APPROVED, iter 3 consensus). VLM-backbone plan: `/root/.claude/plans/baseline-models-are-recursive-manatee.md`.

## Environment
DeepCGV-Mk7, up to **3× A6000 Blackwell (96 GB each)**. Docker dev container. Python 3.11, PyTorch 2.x, MONAI 1.4, lightning 2.4, Hydra 1.3, transformers 4.46, diffusers 0.31, wandb.

## Repo layout (post Phase A restructure)
```
src/                      # lightning-hydra-template base + our additions
  data/                   # LightningDataModule (CT-RATE, MAISI latents)
  models/                 # report2ct_module.py + components/; src/models/vlm/ for VLM task heads (deferred to follow-up plan)
  baselines/              # adapters: Report2CT, GenerateCT (generators);
                          #           CT-CLIP, fVLM (VLM backbones — plain nn.Module, NOT LightningModule)
  diagnostics/            # (placeholder — future: cross-attn / retrieval / counterfactual diagnostics)
  train.py, eval.py       # Hydra entrypoints

configs/                  # Hydra hierarchy (data/model/trainer/logger/callbacks/experiment/...)
  model/vlm_backbone/     # NEW group: ctclip.yaml / fvlm.yaml — backbone-only configs

third_party/              # READ-ONLY external code (P2)
  report2ct/              # submodule SHA 7b483a8 — paper code + train.sh + JSON configs (weights NOT released)
  generatect/             # submodule SHA 2a81135 — 3 pretrained .pt ckpts on HF
  vlm3d_dockers/          # submodule SHA c73fe07 — official VLM3D-Dockers eval containers (ctgen/reportgen/abnclass active; abnloc_* dirs are last-year artifacts)
  maisi_bundle/           # vendored MONAI MAISI bundle (FROZEN VAE: autoencoder.pt + configs/inference.json)
  ct_clip/                # submodule — ibrahimhamamci/CT-CLIP, train-side canonical (eval-side copies live inside vlm3d_dockers/*/CT-CLIP/)
  fvlm/                   # submodule — alibaba-damo-academy/fvlm (ViT + BiomedVLP-CXR-BERT-specialized, CT-RATE-trained)

tests/                    # pytest suite (hydra_compose, maisi_frozen_load, data_module, adapters, spacing, parity)
notebooks/                # eda.ipynb + 3D_viewer.ipynb + test_generateCT.ipynb
docs/                     # runbooks + submodule_pins.md, report2ct_*.md, fvlm_report2ct_guidebook.md
results/                  # results/upper_bound.json (1mm MAISI VAE PSNR 30.94 baseline)
data/                     # new artifacts only
  checkpoints/            # per-model: generatect/, ctclip/, fvlm/, hf_cache/ (HF_HOME)
paper_pdf/                # Report2CT.pdf + others (reference)
deprecated/               # all triplane-era work (do not import from here)
```

## Dataset reference
**Doubled-path trap**: data root is `/workspace/datasets/datasets/` (not `/workspace/datasets/`, which only holds download scripts + `split.json`). Full detail (sizes, dtypes, `split.json` provenance, dev-valid deprecation) in [docs/dataset_reference.md](docs/dataset_reference.md).

- **Raw CT (CT-RATE)**: `/workspace/datasets/datasets/CT-RATE/dataset/` — `train_fixed/` (47,148 scans), `valid_fixed/` (3,038); plus `metadata/`, `multi_abnormality_labels/`, `radiology_text_reports/`, `ts_seg/` (TotalSegmentator masks for fVLM).
- **MAISI VAE latents** (shape `[4,120,120,64]`): small `.pt` set under `…/latents/{train,valid}/`; **canonical toy v2** at `/workspace/data/ctrate_toy_v2/` (`train/` = 5000 symlinked, `proxy_test/` = `valid_fixed` 1304 one-per-patient = **the eval proxy**); full 48k `.nii.gz` set under `report2ct_work_dir/` (Report2CT training).

## Win condition (frozen)
On the **CT-RATE `valid_fixed` 1304 one-scan-per-patient proxy-test** (frozen `ctrate_toy_v2/proxy_test/ids.json`; `load_eval_cases()` restricts to it, any `n_samples<1304` is a seeded-shuffle nested subset), via VLM3D-Dockers: `ours_final` beats `report2ct_our_repro` in ≥2 of {2.5D-FID, CLIPScore-T2I, FVD}. Metric priority for headline: **2.5D-FID > CLIPScore-T2I > FVD**. *(Self-measured baseline numbers are pending re-measurement by the user against this 1304 set; the paper envelope below is unaffected.)*

## Envelope (Report2CT 3-TE-cfg5 anchor)
- 2.5D-FID anchor = 4.04 (Fig 6 FID Avg) ⇒ ±15%
- CLIPScore-T2I anchor = 59.93 (Fig 5) ⇒ ±10%
- FVD anchor = self-measured (paper does NOT report FVD) ⇒ ±25%

## Upper bound
MAISI VAE encode→decode round-trip on an earlier **1000-volume CT-RATE valid sample** (pre-v2 — not directly comparable to the 1304 proxy-test): **PSNR 30.94 ± 2.97 dB, SSIM 0.7195 ± 0.1084**. Intensity: HU clipped to `[-1000, 1000]`, scaled to `[0, 1]`; spatial 480×480×256. Full: `results/upper_bound.json`.

## Compute / I/O notes
3D MAISI latent streaming (`mu.pt`, 7.2 MB/sample × 6,000 ≈ 43 GB) **dominates runtime** for any per-sample sweep. Defaults: `--device cpu --num-workers 16` for cheap arithmetic; GPU only when inner ops justify it. See [[3d-latent-i-o-bottleneck]] memory.

## Non-goals
- Triplane autoencoder research ([[triplane-deprecated-2026-05]]).
- Report2CT full training (user runs it directly — [[report2ct-training-is-user-owned]]).
- New dataset preprocessing beyond what VLM3D-Dockers expects.
- abnloc (abnormality localization) — not open in 2026 ([[vlm3d-2026-three-tasks]]).
- From-scratch CT-CLIP pretraining — we use the released `CT-CLIP_v2.pt`.
- Report-generation LLM training from scratch — we adapt frozen VLM encoders + a small text decoder. Task heads themselves are deferred to a follow-up plan after the backbone adapters land.

## Architecture (big picture across files)
- **lightning-hydra-template** is the skeleton: `src/train.py` and `src/eval.py` are Hydra `@main` entrypoints; `configs/train.yaml` / `configs/eval.yaml` compose `data/ + model/ + trainer/ + callbacks/ + logger/ + experiment/`. `configs/data/` (report2ct.yaml) and `configs/model/` (report2ct/generatect/text2ct + vlm_backbone/) are populated; select via CLI override or experiment yaml.
- **Frozen MAISI VAE** is the latent space. Always load via [src/baselines/maisi.py](src/baselines/maisi.py) `load_frozen(device=...)` (uses `monai.bundle.ConfigParser` on `third_party/maisi_bundle/configs/inference.json` — do **not** re-declare architecture kwargs). `tests/test_maisi_frozen_load.py` enforces all params `requires_grad=False`.
- **Report2CT (LIVE)** is [src/models/report2ct_module.py](src/models/report2ct_module.py) `Report2CTModule` (LightningModule, training loop only — the `DiffusionModelUNetMaisi` 233M UNet + RFlow scheduler are **injected** via Hydra `instantiate`). UNet/scheduler kwargs are transcribed 1:1 from the submodule's `config_maisi_2560.json` into [configs/model/report2ct.yaml](configs/model/report2ct.yaml); that parity (and a bit-exact forward) is enforced by [tests/test_report2ct_parity.py](tests/test_report2ct_parity.py). Actual multi-day training is user-owned.
- **GenerateCT adapter** ([src/baselines/generatect_adapter.py](src/baselines/generatect_adapter.py)) is the one place that intentionally duplicates kwargs (submodule has no JSON config); look for the `DUPLICATION INTENTIONAL` annotations before editing. Uses `sys.path.insert` instead of installing the submodule.
- **Data path** is metadata-only today: [src/data/ct_rate_datamodule.py](src/data/ct_rate_datamodule.py) joins 3 CSVs (reports + metadata + labels) into `CTRateRecord` dataclasses; `mode="volume"` is `NotImplementedError` until Phase C.
- **Evaluation** is delegated to VLM3D-Dockers via [src/vlm3d_runner.py](src/vlm3d_runner.py); the runner auto-falls-back to `--dry-run` (NaN-filled 8-key schema) when no docker daemon is available, so the downstream compare pipeline can be exercised inside the dev container.
- **VLM backbones** are plain `nn.Module` (NOT LightningModule), wrapped one-per-baseline, with the same `sys.path.insert` + lazy `_ensure_built()` pattern as `generatect_adapter.py`. Each backbone is one YAML file under [configs/model/vlm_backbone/](configs/model/vlm_backbone/) and is swappable via `override /model/backbone: <name>`. Weights live under `data/checkpoints/<name>/`; see [docs/vlm_baselines_runbook.md](docs/vlm_baselines_runbook.md) for the user-owned download recipes. The two adapters do **not** share a uniform forward contract — each mirrors its upstream's actual API:
    - [src/baselines/ctclip_adapter.py](src/baselines/ctclip_adapter.py) — `CTCLIPBackbone` exposes `encode_image(vol) -> (B, 512)`, `encode_text(ids, mask) -> (B, 512)`, `tokenize(text)`. Mirrors CT-CLIP's contrastive forward path (`visual_transformer(..., return_encoded_tokens=True)` → temporal mean → flatten → `to_visual_latent` + L2 norm; text uses `last_hidden_state[:, 0]` → `to_text_latent` + L2 norm).
    - **fVLM** is anatomy-aware — its forward needs `(volume, organ_mask)` and returns per-organ ROI features (no `encode_image` contract). Adapter contract + TotalSegmentator-mask preprocessing details live in [.claude/rules/fvlm.md](.claude/rules/fvlm.md) (auto-loads when editing `src/baselines/fvlm_*.py`).

## Common commands
Use `make help` to discover targets. Most useful:

```bash
# Tests
pytest tests/ -q
# Single test:
pytest tests/test_maisi_frozen_load.py -k frozen -v

# Hydra config compose sanity (no training):
python src/train.py --cfg job --resolve
python src/eval.py --cfg job --resolve ckpt_path=/tmp/dummy.ckpt

# Train / eval (Phase B+ — needs an experiment config):
python src/train.py experiment=report2ct_repro     # lands Phase B
python src/eval.py  ckpt_path=path/to/best.ckpt

# EDA regenerate (writes to figs/eda/):
python scripts/run_eda.py --split valid --hu-sample 50

# VLM3D-Dockers wrapper (works without docker daemon via --dry-run):
python -m src.vlm3d_runner --dry-run --out /tmp/smoke.json

# Environment sanity:
python -c "import lightning, hydra, monai, diffusers, transformers; print('OK')"

# VLM backbone smoke (no weights required — gates 2 & 3 of the backbone plan):
pytest tests/test_ctclip_adapter.py tests/test_fvlm_adapter.py \
  -k "not requires_weights" -v
# (config-compose checks per backbone: see docs/vlm_baselines_runbook.md)

# VLM backbone with weights (after running docs/vlm_baselines_runbook.md downloads):
CUDA_VISIBLE_DEVICES=0 pytest tests/test_ctclip_adapter.py -k requires_weights -v
```

## Conventions
- **Always work on the `main` branch — never in a git worktree.** Do NOT create worktrees, do NOT spawn agents with `isolation: "worktree"`, and do NOT use worktree-first flows (e.g. project-session-manager). All edits, commits, and agent work happen directly in `/workspace` on `main`. If a task genuinely seems to need isolation, ask first.
- **`third_party/` is READ-ONLY** (Principle P2 in the plan). Pin via submodule SHA in [docs/submodule_pins.md](docs/submodule_pins.md); don't patch the trees in-place.
- **Reuse external code, don't rewrite it** — wrap submodule configs via `monai.bundle.ConfigParser` or `sys.path.insert` rather than copying class definitions. Duplications must be tagged `DUPLICATION INTENTIONAL` with file:line citation to the upstream source.
- **`deprecated/` is import-forbidden** — it preserves the triplane-era work for history only.
- **GPU prefix**: `CUDA_VISIBLE_DEVICES=0 python ...` for single-GPU; explicit `CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 ...` for multi-GPU.
- **Artifacts go under `/workspace/data/`**, never under `/workspace/datasets/` (collaborator's read-only area).
- **Decision trail**: design decisions live in `.omc/specs/` (deep-interview transcripts) and `.omc/plans/` (consensus plans). Quote them when justifying non-obvious choices.
- **Cloning a baseline?** Run the 8-point silent-bug audit in [.claude/rules/baseline-clone.md](.claude/rules/baseline-clone.md) (auto-loads when editing `src/baselines/**` or `src/eval/**`).

### Coding principles
- **Think before coding.** State assumptions explicitly; if uncertain or ambiguous, stop and ask rather than guessing silently. If a simpler approach exists, say so and push back.
- **Simplicity first (load-bearing).** Minimum code that solves the *stated* problem — nothing speculative. No unrequested features, no abstractions for single-use code, no configurability or error handling for impossible cases. If 200 lines could be 50, rewrite it. Senior-engineer test: if it reads as overcomplicated, simplify.
- **Surgical changes.** Touch only what the request requires. Don't improve/refactor/reformat adjacent working code; match existing style. Remove only the orphans your own change creates; flag pre-existing dead code, don't delete it.
- **Goal-driven execution.** Turn the task into a verifiable success criterion (e.g. "write a failing test that reproduces the bug, then make it pass") and loop until it's met.

### Implementation workflow (incremental + review-gated)
- **Plan the split first.** Before writing any code, present how the feature will be broken into small units and in what order.
- **Build one unit at a time.** Large features MUST be decomposed and implemented **one unit per step** — never several units in one pass.
- **Stop and explain after each unit.** When a unit is done, halt and explain the design intent and the core logic, then wait for the user's review/approval.
- **No advancing without approval.** Do not start the next unit until the current one is explicitly approved.
- **Shape comments on every core tensor op.** Annotate all key tensor operations with their shapes (e.g. `# (B, 4, 120, 120, 64)`).
