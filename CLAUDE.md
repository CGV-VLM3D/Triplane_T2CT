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

Plan: `/workspace/.omc/plans/vlm3d-pivot-plan.md` (Critic APPROVED, iter 3 consensus).

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
  vlm3d_dockers/          # submodule SHA a945900 — official VLM3D-Dockers eval containers (ct_challenges/ reorg; ctgen/reportgen/abnclass active; abnloc_* dirs are last-year artifacts)
  maisi_bundle/           # vendored MONAI MAISI bundle (FROZEN VAE: autoencoder.pt + configs/inference.json)
  ct_clip/                # submodule — ibrahimhamamci/CT-CLIP, train-side canonical (eval-side copies live inside vlm3d_dockers/*/CT-CLIP/)
  text2ct/                # submodule — MAISI-latent text→CT LDM baseline (adapter: src/baselines/text2ct_adapter.py; live generator)
  fvlm/                   # alibaba-damo-academy/fvlm (ViT + BiomedVLP-CXR-BERT-specialized, CT-RATE-trained) — vendored dir, NOT a git submodule (absent from .gitmodules)

tests/                    # pytest suite (hydra_compose, maisi_frozen_load, data_module, adapters, spacing, parity)
notebooks/                # 3D_viewer.ipynb + test_generateCT.ipynb
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
  - **⚠ Quarantined scans — always exclude**: the census is cleaned by moving bad scans into `dataset/no_chest_data/` (789 scans, chest out-of-FOV) and `dataset/error_ctrate_data/` (3 scans), mirroring the `{train_fixed,valid_fixed}/<patient>/<scan>/*.nii.gz` layout. **`ts_seg/` is NOT cleaned** — its masks still include these quarantined scans (raw glob = 47,146 train / 3,039 valid ts_seg masks). Any dataset built from `ts_seg/` (or raw CT) **must filter out these scans by basename** → clean counts **train 46,393 / valid 3,001** (matches [[no-chest-correction-report2ct]] / [[ctrate-full-dataset]]). Reusable exclusion helper: `tests/mask_vae/dataset.py::excluded_scan_ids()`.
- **MAISI VAE latents** (shape `[4,120,120,64]`): small `.pt` set under `…/latents/{train,valid}/`; **canonical toy v2** at `/workspace/data/ctrate_toy_v2/` (`train/` = 5000 symlinked, `valid_v2/` = `valid_fixed` 1304 one-per-patient (one **scan-`a`** recon per patient, drawn **uniform-random** over that patient's clean recons, seed=42 — kernel-de-biased; the legacy `_a_1`-preferred rule skewed `ConvolutionKernel` (B 28%→5%, YA 26%→51%), fixed 2026-07 — see [[proxy-test-is-valid-v2]]) = **the eval proxy AND the canonical validation set "valid_v2"** — used for both training val/loss and headline FID/CLIP; the older 1000 `valid_*` split is **deprecated** (its `datalist_5k.json` deleted 2026-06-20; canonical training datalist is now `datalist_v2.json` (5k) / `datalist_full_v2.json` (47k), both no_chest+unencodable-corrected — see [[no-chest-correction-report2ct]]); the real test set is held by the challenge organizers); full 48k `.nii.gz` set under `report2ct_work_dir/` (Report2CT training). Image latents + text sidecar (`*_emb.nii.gzmulti_2560.json`) for all 1304 valid_v2 scans are **complete** (images via frozen MAISI `scripts/precompute_report2ct_image_embeddings.py`, **no `--metadata-csv`** — `_fixed` HU already baked; text via `scripts/precompute_report2ct_text_embeddings.py` → merged by `build_report2ct_datalist.py`). **⚠ Canonical train latent = `report2ct_work_dir/image_embeddings/*_emb.nii.gz`** (fp32, std≈0.98, same space as valid_v2 `_emb.nii.gz`). As of **2026-06-18** the toy_v2 `train/` symlinks point at these `_emb.nii.gz` (previously raw VAE **`mu.pt`**, std≈0.67 — never use `mu.pt`, mismatched scale corrupts cross-split work); `train/stats.json` recomputed on `_emb` (std≈0.93–0.99). See [[maisi-latent-source-scale-mismatch]].

## Win condition (frozen)
Models **generate** on the **CT-RATE `valid_fixed` 1304 one-scan-per-patient valid_v2** (frozen `ctrate_toy_v2/valid_v2/ids.json`; `load_eval_cases()` restricts to it, any `n_samples<1304` is a seeded-shuffle nested subset). The **FID reference distribution** is the **full clean valid census 3001** (`valid_fixed` 3038 − no_chest 37 − unencodable 1; `data/ctrate_full/valid/ids.json`; `task.gt_dir=_valid_full_3001`) — FID is asymmetric so the model-independent GT reference uses all 3001 real volumes (precomputed per-plane (μ, Σ), [src/eval/tasks/_fid_refstats.py](src/eval/tasks/_fid_refstats.py)); FVD/CLIP pair each generated volume to its own GT by stem (extra GT ignored). Via VLM3D-Dockers: `ours_final` beats `report2ct_our_repro` in ≥2 of {2.5D-FID, CLIPScore-T2I, FVD}. **`ours_final` is currently `report2ct_wan_mask_v2`** (set 2026-07-29 — see "Headline model" below for the measurement that picked it). Metric priority for headline (**revised 2026-07-29**): **FVD & CLIPScore-T2I are the primary metrics; 2.5D-FID is a secondary/supporting metric.** ⚠ The official `FVD_CTNet` cannot be computed locally (shipped CT-Net checkpoint is a corrupt stub → always NaN, real weights are server-side only — [[fvd-ctnet-corrupt-weight]]); locally we track the `FVD_CTCLIP` proxy ([src/eval/tasks/_fvd_ctclip.py](src/eval/tasks/_fvd_ctclip.py), not leaderboard-comparable) and confirm true FVD only on the leaderboard. *(All baseline numbers — ours + report2ct_repro — are pending re-measurement by the user against this 3001-reference setup; the paper envelope below is unaffected.)*

⚠ **2.5D-FID profile (2026-07-29, defaults revised 2026-08-01)**: the "3001 reference / all predictions" description above is the **`research` profile**, now opt-in only. The default is **`[docker, docker_n300]`** — both squeezenet1_1, `docker` over the **first 100** predictions vs the **same 100 GT stems** ([evaluation.py:229-230](third_party/vlm3d_dockers/ct_challenges/ctgen_evaluation/evaluation.py)), `docker_n300` the same network at 300 volumes (added so subgroup axes — which need more than 100 to avoid single-digit/zero per-label matches — always have it cached). `docker`/`docker_n300` share a scale but different n (check `fid_num_images` before combining); `research` is a different metric family entirely on an incomparable scale (measured on one prediction set: docker **57.87** vs research **1.61**) — `metrics.json` carries `fid_profile`/`fid_num_images` and `fid.json` adds `model_name`/`scored_stems_sha1`. Full detail in [docs/ctgen_local_eval.md](docs/ctgen_local_eval.md). CLIP/FVD are unaffected by the profile and still use every generated volume.

⚠ **Local numbers are for model-vs-model comparison only — the leaderboard scores a different dataset.** The server evaluates a **held-out test split we do not have**; every local metric (both FID profiles, CLIPScore, the FVD proxy) is computed on our own CT-RATE `valid_v2` predictions. So no local configuration can reproduce a leaderboard value, and there is deliberately no option to "score the 100 the leaderboard will score" — the scored 100 is simply the first 100 of a run's own sorted predictions. Absolute leaderboard standing is confirmed only by submitting; locally we compare models against each other under one fixed profile.

## Headline model (set 2026-07-29)
**`report2ct_wan_mask_v2`** — ep299, spacing `0.75/0.75/1.3`, `cfg_text=5 / cfg_mask=1.0`
(`outputs/report2ct_wan_mask_v2/eval_ep299_n300_sp0.75_1.3_cfgt5_cfgm1.0`). Chosen on the
**docker** FID profile + the existing 300-volume CLIP/FVD, measured across all 12 scored runs:

| | wan_mask_v2 | wan_mask | best non-mask |
|---|---|---|---|
| CLIPScore-T2I (primary) | **59.42** | 57.58 | 65.70 (wan/cfg9) |
| FVD_CTCLIP (primary proxy) | **0.269** | 0.284 | 0.369 (wan/cfg5) |
| 2.5D-FID docker (secondary) | 47.19 | **46.93** | 55.68 (wan/cfg3) |

The two mask models beat every non-mask run on FID by ~18 % and on FVD_CTCLIP by a wide margin;
`wan_mask_v2` wins 2 of 3 against `wan_mask`, and the metric priority above makes CLIP/FVD the
deciders. A 300-iteration paired bootstrap confirms this is the right way to break the tie: the
mask-vs-field gap is significant (**P = 0.993**) but `wan_mask_v2` vs `wan_mask` on FID is a coin
flip (**P = 0.43**, diff 0.26 ± 1.71). **This is a 2.5D-FID-only caveat** — the docker profile's
n=100 gives it a bootstrap std of 2.5–3.1 (~5 %), so small FID gaps (a few points) don't mean
much on their own. CLIP and FVD_CTCLIP are measured on all 300 generated volumes and are
unaffected by the profile, which is why they carry the tie-break here. ⚠ Under the **research** profile the ranking is different (`report2ct` cfg7/cfg5 lead,
mask sits 3rd/6th) — Spearman ρ = 0.748 between the profiles, so they are correlated but not
interchangeable. Details + the full 12-run table in [docs/ctgen_local_eval.md](docs/ctgen_local_eval.md);
paired-bootstrap significance in [tests/fid_profile_bootstrap/](tests/fid_profile_bootstrap/run.py).

## Envelope (Report2CT 3-TE-cfg5 anchor) — listed in priority order
- FVD anchor = self-measured (paper does NOT report FVD) ⇒ ±25%   *(primary)*
- CLIPScore-T2I anchor = 59.93 (Fig 5) ⇒ ±10%   *(primary)*
- 2.5D-FID anchor = 4.04 (Fig 6 FID Avg) ⇒ ±15%   *(secondary; **`fid_profile=research` only** — the paper's number is a RadImageNet-family FID, so it says nothing about a `docker`-profile value. `research` is opt-in since 2026-08-01 (not in the default `[docker, docker_n300]` list) — pass `task.fid_profile=research` explicitly to get a number comparable to this anchor)*

## Upper bound
MAISI VAE encode→decode round-trip on an earlier **1000-volume CT-RATE valid sample** (pre-v2 — not directly comparable to the 1304 valid_v2): **PSNR 30.94 ± 2.97 dB, SSIM 0.7195 ± 0.1084**. Intensity: HU clipped to `[-1000, 1000]`, scaled to `[0, 1]`; spatial 480×480×256. Full: `results/upper_bound.json`.

## Compute / I/O notes
3D MAISI latent streaming (`mu.pt`, 7.2 MB/sample × 6,000 ≈ 43 GB) **dominates runtime** for any per-sample sweep. Defaults: `--device cpu --num-workers 16` for cheap arithmetic; GPU only when inner ops justify it. See [[3d-latent-i-o-bottleneck]] memory.

**⚠ Filesystem layout — put ALL heavy work on `/workspace`, never on the session scratchpad.** `/workspace` is `/dev/md0` (**28 TB**, TBs free); the container root `/` (which backs `/tmp`, incl. the `/tmp/claude-*/…/scratchpad` session scratchpad) is a **small ~938 GB nvme at ~95%**. The scratchpad is for tiny files (scripts, filelists) only. Any GB-scale intermediate (feature dumps, extraction work dirs, temp volume copies) and every **eval `out_dir`** MUST live under `/workspace/…` — not `/tmp`. Two concrete traps (both hit 2026-07-25, see [[heavy-intermediates-on-workspace-not-tmp]]): (1) writing 28 GB of extracted features to the `/tmp` scratchpad filled root to 100 %; (2) an eval `out_dir` on `/tmp` makes `_run_fid` **copy** the whole GT feature set cross-device (`/workspace`→`/tmp`, 36 GB) instead of hardlinking — always pass `out_dir` on the same `/workspace` device as `data/vlm3d_eval/` so GT features hardlink for free. Also: when judging free space, check `df -h /workspace` (md0), not `df /` (root).

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
- **Evaluation** runs the VLM3D-Dockers ctgen-evaluation *scripts directly* inside our container (no docker daemon): [scripts/run_eval.py](scripts/run_eval.py) (Hydra) generates predictions via a sampler then delegates to [src/eval/tasks/ctgen.py](src/eval/tasks/ctgen.py) `CTGenEvaluator`, which subprocess-invokes the upstream `evaluate_fvd.py` / `evaluate_clip.py` / `compute_fid_2-5d_ct.py` — recreating the docker environment via `/opt/app/*` symlinks, PYTHONPATH injection, a numpy-2.x shim ([src/eval/tasks/_fid_runner.py](src/eval/tasks/_fid_runner.py)), and a CPU-side FID computation (the upstream GPU FID OOMs at scale). Path resolution is centralized in [src/eval/_vlm3d_paths.py](src/eval/_vlm3d_paths.py).
- **VLM backbones** are plain `nn.Module` (NOT LightningModule), wrapped one-per-baseline, with the same `sys.path.insert` + lazy `_ensure_built()` pattern as `generatect_adapter.py`. Each backbone is one YAML file under [configs/model/vlm_backbone/](configs/model/vlm_backbone/) and is swappable via `override /model/backbone: <name>`. Weights live under `data/checkpoints/<name>/`; see [docs/vlm_baselines_runbook.md](docs/vlm_baselines_runbook.md) for the user-owned download recipes. The two adapters do **not** share a uniform forward contract — each mirrors its upstream's actual API:
    - [src/baselines/spectre_adapter.py](src/baselines/spectre_adapter.py) — `SpectreBackbone`, a frozen CT 3D-ViT (SPECTRE ViT-L, patch 16×16×8) used as the **REPA alignment teacher** for `report2ct_wan`. Image-only, so it does **not** share the `encode_image`/`encode_text` contract: it returns a dense token grid, `encode_dense(vol_hu) -> (32, 32, 32, 1080)`. Alignment head is [src/models/components/repa.py](src/models/components/repa.py) `RepaAligner` (opt-in — `repa: null` keeps the loss path bit-identical to the baseline). Runbook [docs/repa_runbook.md](docs/repa_runbook.md); the measurements that set its defaults (they differ from the papers) are in [tests/repa_probe/](tests/repa_probe/README.md).
    - [src/baselines/ctclip_adapter.py](src/baselines/ctclip_adapter.py) — `CTCLIPBackbone` exposes `encode_image(vol) -> (B, 512)`, `encode_text(ids, mask) -> (B, 512)`, `tokenize(text)`. Mirrors CT-CLIP's contrastive forward path (`visual_transformer(..., return_encoded_tokens=True)` → temporal mean → flatten → `to_visual_latent` + L2 norm; text uses `last_hidden_state[:, 0]` → `to_text_latent` + L2 norm).
    - **fVLM** is anatomy-aware — its forward needs `(volume, organ_mask)` and returns per-organ ROI features (no `encode_image` contract). **fVLM does NOT do sliding-window inference** — upstream `eval.py:301-307` computes `dense_patch_slices`/`num_win` but labels it `############# dead code #############` and never uses it; the real eval path (`eval.py:309-338`) loops organs and calls `center_crop(image, mask==organ, crop_size=(112,288,352))` **once per organ** (window centered on that organ, organ never boundary-clipped) → `DivisiblePadd(method="end")` → a single `forward_test_win(..., skip_organ=organ_id)`. The "win" in `forward_test_win` is that per-organ centered crop, not a sliding sweep. Adapter contract + TotalSegmentator-mask preprocessing details live in [.claude/rules/fvlm.md](.claude/rules/fvlm.md) (auto-loads when editing `src/baselines/fvlm_*.py`).

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

# VLM3D ctgen eval (no docker daemon — runs upstream scripts directly):
CUDA_VISIBLE_DEVICES=0 python scripts/run_eval.py task=ctgen model=report2ct model.ckpt_path=<ckpt> \
  model.spacing_mm=[0.8,0.8,1.5] model.cfg_scale=5.0 task.n_samples=1   # spacing_mm + cfg_scale are REQUIRED (no default) for report2ct-family models

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
- **Run logs go under `/workspace/logs/`, never under `data/`.** When auto-launching train/eval as a background/detached process, redirect stdout+stderr to `logs/<model>_<timestamp>.log` (gitignored; `make clean-logs`). The python `logging` output is already saved by Hydra to `outputs/<model>/<KST-date>[_N]/<task>.log` — `data/` stays log-free (artifacts only).
- **Decision trail**: design decisions live in `.omc/specs/` (deep-interview transcripts) and `.omc/plans/` (consensus plans). Quote them when justifying non-obvious choices.
- **Cloning a baseline?** Run the 8-point silent-bug audit in [.claude/rules/baseline-clone.md](.claude/rules/baseline-clone.md) (auto-loads when editing `src/baselines/**` or `src/eval/**`).
- **ctgen eval reproducibility — spacing/cfg are always explicit, never defaulted**:
  - **Wan flow** (argparse): `--spacing` + `--cfg-scale` are **required args (no defaults)** on [scripts/generate_wan_latents.py](scripts/generate_wan_latents.py), [scripts/generate_wan_mask_latents.py](scripts/generate_wan_mask_latents.py), and [scripts/decode_wan_latents.py](scripts/decode_wan_latents.py) (`--spacing` only). The `--spacing` given to generate MUST MATCH the one given to decode (spacing-conditional model; image + mask flows share the decode script).
  - **run_eval flow** (Hydra): for the spacing-conditional **report2ct-family** eval configs (`report2ct`, `report2ct_clip3d`, `report2ct_fvlm`, `report2ct_text2ct`, `report2ct_text2ct_mask`, `report2ct_text2ct_mask_g4`), `model.spacing_mm` and `model.cfg_scale` are `???` (Hydra MISSING) — you MUST pass `model.spacing_mm=[..] model.cfg_scale=..` on the CLI or instantiate fails loudly (recommended values kept in each config's comment). `text2ct`/`generatect` are left as-is (spacing there is a fixed native output tag, not a conditioning knob).
  - Rationale: a silent default once let a run drift geometry unnoticed → quietly-wrong metrics. Make the knob loud. **Every eval output dir name must encode epoch + spacing + cfg**: `eval_ep<NNN>_sp<inplane>_<z>_cfg<C>` (e.g. `eval_ep299_sp0.73_1.34_cfg5`).
  - ⚠ **That naming convention is the ONLY thing guarding generation settings — no code checks them.** The guards below catch a *cross-profile* re-run, not a same-profile one, and every sampler skips an `.mha` that already exists. So pointing `out_dir=` at an existing eval dir with a different `model.spacing_mm` / `model.cfg_scale` / `model.ckpt_path` **regenerates nothing**: at the same `task.n_samples` the volumes stay from the ORIGINAL settings while that dir's `.hydra/` + `run_eval.log` now document the NEW ones, and at a larger `n_samples` (subsets are nested) old- and new-setting volumes end up **mixed in one prediction set**. Any run whose generation settings changed needs a fresh `out_dir` — which its own `eval_ep…_sp…_cfg…` name already implies.
  - **The FID profile names the folder, so it never needs to be in the dir name** (2026-07-31). Every scoring pass — `run_eval.py` and `scripts/rescore_predictions.py` alike — writes its artifacts to `<eval_dir>/fid_<profile>/` (`fid_docker` / `fid_docker_n300` / `fid_research`); **the only top-level metric file is the combined `summary.json`, which labels every FID by profile.** `predictions/`, `latents/`, `analysis/`, `prompts.xlsx`, `.hydra/`, `run_eval.log` are profile-independent and stay at the top. One eval dir routinely holds `fid_research/` next to `fid_docker/`; that is normal, not a clobber. The older `_fidresearch` dir-name suffix is obsolete — don't add it to new runs.
    ```
    eval_ep299_n300_sp0.75_1.3_cfg1/
    ├── fid_research/   metrics.json summary.json clip.json fid.json fvd_ctclip.json fid_features/
    ├── fid_docker/     metrics.json summary.json fid.json fid_features_squeezenet1_1/
    ├── _shared_pred_fidfeat/  <model>/<hash>/  ← per-pred-dir cache (2026-08-01), see below
    ├── summary.json    ← 전 프로파일 통합 (metrics = 공통 지표, fid.<profile> = 프로파일별 FID)
    └── predictions/ latents/ analysis/ prompts.xlsx .hydra/ run_eval.log
    ```
    `_shared_pred_fidfeat/` lets `docker`/`docker_n300` reuse each other's prediction FID
    features within one run (docker's 100 scored stems are always the first 100 of
    docker_n300's 300) — keyed on feature model + a hash of the actual `pred_dir` path, not
    just its parent, so harnesses that score several different prediction sets under one shared
    parent (`tests/orientation_quant/`, `tests/spacing_fov/`) can't collide on it.
  - **`run_eval.py` scores BOTH squeezenet1_1 FID profiles by default** (changed 2026-08-01): `task.fid_profile` takes a name **or a list**, and defaults to `[docker, docker_n300]`. `research` (radimagenet_resnet50 vs the full 3001-volume GT census) was dropped from the default that day — it's still the only profile comparable to the paper envelope (below) or to anything recorded before 2026-07-29, just no longer produced automatically; get it with an explicit `task.fid_profile=research` or `task.fid_profile=[docker,docker_n300,research]` override. The extra profile only re-runs FID — FVD / FVD_CTCLIP / CLIP are computed in the **first** profile's pass and copied into the others, so each `fid_<profile>/metrics.json` stays self-contained (raw `clip.json` / `fvd_ctclip.json` sidecars stay with the pass that wrote them). Measured cost at n=300 with a warm GT cache: docker FID **~7 min**, research FID **~31 min** (now opt-in), CLIP ~12, FVD_CTCLIP ~12 — against a 2.5–11 h generation stage. `docker_n300`'s own cost is n-dependent and cache-dependent: **~49 min** cold (first-ever docker_n300 run against a given prediction set, no shared GT or pred features to reuse) down to **~12 min** with a partially warm cache (GT already shared cross-run from another model's docker_n300 run + ~1/3 of predictions already shared from that run's own `docker` pass) — see the partial-cache optimization below. Set `task.fid_profile=docker` for a single-family run; `scripts/rescore_predictions.py` stays one-profile-per-invocation. `docker_n300` is also now the default `subgroup_fid_profile` (was `research`) — subgroup breakdowns need more than 100 volumes to avoid single-digit/zero per-label matches, and this way every routine run already has it cached.

  **FID feature caching (added 2026-08-01)**: `docker`'s 100 scored stems are always the first 100 (by sorted filename) of `docker_n300`'s 300 — same `predictions/` dir, same sort — so whichever profile runs first leaves the other's overlapping work already done. GT features were already shared cross-run (`_shared_gt_feat_dir`, keyed by GT-set+model, permanent root under `data/vlm3d_eval/_shared_gt_fidfeat/`); predictions now get an analogous **per-pred-dir** cache (`_shared_pred_feat_dir`, keyed by model + a hash of the actual `pred_dir` path — not just its parent, since some harnesses score several different prediction sets under one shared parent with reused filenames, e.g. `tests/orientation_quant/`, `tests/spacing_fov/`; hashing the full path keeps those collision-proof). On top of the sharing, `_run_fid` also skips the upstream loop's load+resample (not just its CNN forward) for any GT/pred volume whose feature file is already present locally, whether via a hardlink or a prior partial run — previously that shortcut only existed for the all-or-nothing case where a complete GT ref-stats npz already existed. Both mechanisms only affect *how fast* a value is computed, never *what* value — verified by rescoring two different models (`report2ct_wan_mask_v2`, `text2ct_toy_v2`) and reproducing bit-identical FID_2p5D values pre/post.
  - **`metrics.json` is merged, never overwritten** ([src/eval/tasks/ctgen.py](src/eval/tasks/ctgen.py) `_merge_metrics`). A pass computes only the metrics it was asked for; dumping that dict straight out used to drop everything already recorded (scoring FVD into a dir holding CLIP+FID left a one-key file — the sole reason a separate `fvd/` subdir ever existed). Metric keys stay **flat at the top level** because every aggregator reads `metrics.get("FID_2p5D_Avg")`; the audit trail goes in `_history`, one dated row per pass listing the metric set, the keys added, and any key replaced together with its previous value. `summary.py` skips `_`-prefixed keys.
  - **Aggregators take the profile explicitly** (`aggregate_wan_epoch_sweep.py --profile research|docker|docker_n300`) — with several profiles legitimately side by side, guessing from the tree is not possible.
  - ⚠ **`run_eval.py` was the 2026-07-29 clobber path — now closed structurally.** Two `report2ct_wan_mask_v2` runs lost their research FID to `run_eval.py … out_dir=<existing eval dir> task.fid_profile=docker` (proven by `outputs/report2ct_wan_mask_v2/eval_2026-07-29_{2,3}/.hydra/overrides.yaml`): samplers skip existing `.mha`, so the re-run regenerated nothing yet rewrote top-level `metrics.json` + `summary.json` with the other metric family. The profile-folder layout above makes that unreachable (a docker pass cannot touch a research pass's file) and the merge makes a same-folder re-run additive, so the runtime guard `_refuse_cross_profile_overwrite` and its `task.allow_overwrite` escape hatch were **deleted on 2026-07-31** as redundant.
    - **`out_dir` is the single run-dir knob** — [configs/eval/default.yaml](configs/eval/default.yaml) wires `hydra.run.dir: ${out_dir}`, so `out_dir=<dir>` moves `.hydra/` + `run_eval.log` with the results, and `run_eval.py` refuses a split (`_refuse_split_run_dir`). Under the OLD wiring the incident's `out_dir=` override left Hydra logging to its own dated dir, so the clobbered dirs kept a `.hydra/` from the ORIGINAL run and hid the cause for a day. ⚠ Any `.hydra/` dated before 2026-07-30 is still not a reliable witness for its own dir. Do not pass `hydra.run.dir=`.
  - **Reading older runs**: three pre-2026-07-31 layouts survive on disk — `<dir>/metrics/fid_<profile>/` (sweep), `<dir>/metrics/` (rescore before 2026-07-29), `<dir>/metrics.json` (run_eval). The last two predate the `fid_profile` key, so a missing key means `research`. `_find_metrics` in the epoch aggregator resolves all four in recency order.
- **ctgen prediction orientation — RAS-decoding models must flip to LPS before saving**: the eval GT ([src/eval/ct_rate_cases.py](src/eval/ct_rate_cases.py) `prepare_valid_gt`) is written straight from raw CT-RATE NIfTI (`nib.get_fdata`, no reorient) = **LPS content**. What decides a model's output orientation is **whether its training preprocessing reorients to RAS**, not whether it uses MAISI. Helper: [src/eval/samplers/_orient.py](src/eval/samplers/_orient.py) `ras_to_lps` (flips the two in-plane X/Y axes; involutive).
  - **RAS-decoding → MUST apply `ras_to_lps` in `_save_mha`**: `report2ct` (+ all its subclasses: `fvlm`/`clip3d`/`text2ct_*`/`wan_*`, [src/eval/samplers/report2ct.py:400](src/eval/samplers/report2ct.py#L400)); the Wan eval decode path [scripts/decode_wan_latents.py:46](scripts/decode_wan_latents.py#L46); and **`text2ct`** ([src/eval/samplers/text2ct.py](src/eval/samplers/text2ct.py) — added 2026-07-24; upstream `preprocess_ctrate.py` + `diff_model_create_training_data.py` reorient to RAS).
  - **Native-LPS → NO flip**: `generatect` ([src/eval/samplers/generatect.py](src/eval/samplers/generatect.py)) reads raw voxels (`nib.get_fdata`, no reorient) like the GT, so it already matches — flipping it would be wrong.
  - ⚠ A missing/extra flip compares RAS-pred vs LPS-GT and quietly tanks metrics (CLIPScore ~24 vs ~63). Any text2ct predictions generated before 2026-07-24 need **re-generation** with the current sampler (which now flips at save). The old read-side re-flip flow (`reeval_lps_n100.py` + `aggregate_lps_reeval.py`) was **removed 2026-07-25**: with every sampler now writing LPS at save time, a read-side re-flip would *double-flip* predictions back to RAS. Top-level `predictions/` are now the canonical LPS output; do NOT re-flip them.

### Coding principles
- **Think before coding.** State assumptions explicitly; if uncertain or ambiguous, stop and ask rather than guessing silently. If a simpler approach exists, say so and push back.
- **Simplicity first (load-bearing).** Minimum code that solves the *stated* problem — nothing speculative. No unrequested features, no abstractions for single-use code, no configurability or error handling for impossible cases. If 200 lines could be 50, rewrite it. Senior-engineer test: if it reads as overcomplicated, simplify.
- **Surgical changes.** Touch only what the request requires. Don't improve/refactor/reformat adjacent working code; match existing style. Remove only the orphans your own change creates; flag pre-existing dead code, don't delete it.
- **Goal-driven execution.** Turn the task into a verifiable success criterion (e.g. "write a failing test that reproduces the bug, then make it pass") and loop until it's met.
- **Docstrings: Google style + shapes.** Every public function/method gets a one-line summary; add `Args:` / `Returns:` only when they add information (trivial lifecycle/dunder methods can be a single line). For any tensor/array arg or return, put the shape in the relevant `Args:`/`Returns:` entry as `` ``(B, 4, 120, 120, 64)`` `` — but only a shape you can read from the code (an inline comment or op); describe in prose rather than invent one. Keep step-by-step `# upstream :NNN` / `# (B, ...)` inline comments in the body — docstring states the I/O contract, inline comments trace the steps.

### Implementation workflow (incremental + review-gated)
- **Plan the split first.** Before writing any code, present how the feature will be broken into small units and in what order.
- **Build one unit at a time.** Large features MUST be decomposed and implemented **one unit per step** — never several units in one pass.
- **Stop and explain after each unit.** When a unit is done, halt and explain the design intent and the core logic, then wait for the user's review/approval.
- **No advancing without approval.** Do not start the next unit until the current one is explicitly approved.
- **Shape comments on every core tensor op.** Annotate all key tensor operations with their shapes (e.g. `# (B, 4, 120, 120, 64)`).
