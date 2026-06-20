# Deep Interview Spec: Report2CT Small-Scale Training

## Metadata
- Interview ID: report2ct-small-scale-train-2026-05-27
- Rounds: 5 (Round 0 topology + Rounds 1-5)
- Final Ambiguity: 20%
- Type: brownfield
- Generated: 2026-05-27
- Threshold: 20%
- Status: PASSED (at threshold)
- Challenge Modes Used: Contrarian (Round 4)

## Clarity Breakdown (brownfield weights: goal .35 / constraints .25 / criteria .25 / context .15)

| Component | Goal | Cons | Crit | Ctx | Weighted | Amb |
|---|---|---|---|---|---|---|
| 1. Sample-report join | 0.85 | 0.85 | 0.90 | 0.85 | 0.86 | 14% |
| 2. Latent precompute | 0.85 | 0.75 | 0.85 | 0.80 | 0.82 | 18% |
| 3. Text emb precompute | 0.85 | 0.85 | 0.85 | 0.80 | 0.84 | 16% |
| 4. Training launcher | 0.92 | 0.85 | 0.85 | 0.70 | 0.85 | 15% |
| 5. Validation/sanity | 0.85 | 0.70 | 0.85 | 0.75 | 0.80 | 20% |
| **Overall (min)** | | | | | **0.80** | **20%** |

## Topology

| Component | Status | Description | Coverage |
|---|---|---|---|
| 1. Sample-report join | active | 1000 train + 500 valid (alphabetical first) — `latents/<split>/<sample>/src.txt` ↔ `radiology_text_reports/{train,validation}_reports.csv` VolumeName join | Round 5 |
| 2. Latent precompute | active | Re-encode 1500 selected samples via submodule `vlm3d_image_embedding.py` on 2 GPUs in parallel | Round 0 (Option B), Round 3 |
| 3. Text emb precompute | active | 3 HF encoders (MedEmbed-large-v0.1 + ClinicalBERT + BiomedVLP-CXR-BERT-specialized) → `*multi_2560.json` per sample | Round 4 (3-TE paper-fidelity confirmed) |
| 4. Training launcher | active | Paper hyperparams unchanged (batch=2, lr=1e-4, RFlow, CFG 0.15) + epochs=10 (paper=100) on 1 GPU, ~4h wall-clock | Round 2 |
| 5. Validation/sanity | active | Loss-only sanity: train+valid loss decreasing for first N epochs, no NaN/diverge. Optional 1-sample decode mid-train | Round 1 |

## Goal

Run a **small-scale Report2CT training** (1000 train + 500 valid CT-RATE samples; paper hyperparams except epochs=10; ~4h wall-clock on a single A6000) using freshly re-encoded MAISI latents (via the submodule's own `vlm3d_image_embedding.py`) and precomputed 3-text-encoder embeddings. **Success = training+valid loss decreasing over 10 epochs with no NaN/divergence**, proving the data + model + launcher pipeline is end-to-end correct before committing to the full Phase B run on all of CT-RATE train_fixed.

## Constraints

- **Subset size**: exactly 1000 train + 500 valid, alphabetical-first deterministic selection from `/workspace/datasets/datasets/latents/{train,valid}/`.
- **Latent precompute path**: re-encode via submodule (`third_party/report2ct/src/maisi/scripts/vlm3d_image_embedding.py`); do NOT bridge the existing collaborator-provided `mu.pt` files. Reason: avoids silent normalization mismatch (Round 0 Option-B decision).
- **GPU resources**: 2× A6000 for latent precompute (parallel disjoint subsets); 1× A6000 for training (paper batch=2, single-GPU is correct paper spec).
- **Wall-clock budget**: ~30 min text precompute + ~4-8h latent precompute (2 GPU parallel) + ~4h training. Single overnight run end-to-end.
- **Paper hyperparams locked**: batch_size=2, lr=1e-4, RFlowScheduler (1000 steps, scale 1.4), CFG dropout p=0.15, 3-TE multi-encoder. Only `n_epochs` deviates (100 → 10).
- **External code untouched**: `third_party/report2ct/` is read-only (Principle P2). We add our own launcher / env-config / precompute wrappers in `scripts/` and `configs/`.
- **MAISI VAE**: frozen, loaded via `src.baselines.maisi.load_frozen` (already R6-verified Phase A Day 2).

## Non-Goals

- **NOT** matching paper FID/CLIP envelope numbers — this run is for *code-path verification*, not for headline metric. Full envelope check is Phase B B.2 sanity (separate `results/report2ct_envelope.json`).
- **NOT** evaluating with VLM3D-Dockers (FVD/CLIPScore/2.5D-FID). Loss-only sanity. Quantitative eval happens after full training in Phase B B.2.
- **NOT** stratified sampling by 18 abnormality labels — deterministic alphabetical selection is sufficient for sanity (Round 5).
- **NOT** bridging the existing `mu.pt` files — too much silent-failure risk (Round 0 Option-A rejected).
- **NOT** using 1-TE / 768-d simplification — Contrarian probe rejected by user (Round 4), keep paper 3-TE for downstream reuse.
- **NOT** super-resolution / final 512×512×256 output. Latent-space training only.

## Acceptance Criteria

### Phase: Sample-report join
- [ ] `data/report2ct_smallrun/datalist.json` exists with 1500 entries (1000 train + 500 valid).
- [ ] Each entry has fields: `volume_name`, `nifti_path`, `latent_emb_path` (target), `text_emb_path` (target), `findings`, `impression`, `spacing` (xyz).
- [ ] 100% of entries have non-empty `findings` and `impression`.
- [ ] `latent_emb_path` and `text_emb_path` are computed deterministic functions of `volume_name`.

### Phase: Latent precompute
- [ ] `/workspace/data/report2ct_smallrun/image_emb/<volume_name>_emb.nii.gz` exists for all 1500 entries.
- [ ] `ls /workspace/data/report2ct_smallrun/image_emb/ | grep '_emb.nii.gz' | wc -l == 1500`.
- [ ] Spot-check 5 files: shape == `(4, 120, 120, 64)` (or whatever submodule writes), no NaN/Inf.
- [ ] Encoding wall-clock ≤ 8 hours total on 2 GPUs (4 hours per GPU on 750 disjoint samples).

### Phase: Text embedding precompute
- [ ] `/workspace/data/report2ct_smallrun/text_emb/<volume_name>multi_2560.json` exists for all 1500 entries.
- [ ] Spot-check 5 files: `findings_embeddings` shape `(L_f, 2560)`, `impression_embeddings` shape `(L_i, 2560)`, plus `spacing` field with 3-vector.
- [ ] 3 HF encoders pinned by commit hash (`config._commit_hash`) recorded in `data/report2ct_smallrun/text_emb_pins.json`.

### Phase: Training launcher
- [ ] `configs/experiment/report2ct_smallrun.yaml` Hydra config exists, defines paths + epochs=10.
- [ ] `scripts/run_report2ct_smallrun.sh` invokes submodule's training script with custom env config + our datalist.json.
- [ ] `data/report2ct_smallrun/env_config.json` overrides submodule defaults (paths point at `/workspace/data/report2ct_smallrun/...` and `/workspace/third_party/maisi_bundle/models/autoencoder.pt`).
- [ ] First batch reaches the optimizer step (no shape/dtype/CFG-dropout errors).

### Phase: Validation / sanity (the success criterion)
- [ ] Train loss curve monotonically decreasing (or roughly so — allow noisy) over epochs 1-10. Reported via WandB / TB.
- [ ] Valid loss recorded at epoch end, decreasing or stable.
- [ ] No NaN/Inf in any loss value. No CUDA OOM during 10 epochs.
- [ ] Final checkpoint at `data/checkpoints/report2ct/smallrun/last.ckpt` exists.
- [ ] Optional: at epoch 5 and epoch 10, sample 1 prompt → latent → MAISI decode → save .nii.gz to `data/report2ct_smallrun/mid_train_samples/`. Visual sanity (3D Slicer / ITK-SNAP) shows lung-like shape (no hard pass/fail).
- [ ] **One-line decision**: if loss + no NaN, declare smallrun PASS. Otherwise diagnose with `.omc/plans/smallrun_failure_diagnosis.md`.

## Assumptions Exposed & Resolved

| Assumption | Challenge (round) | Resolution |
|---|---|---|
| Existing `mu.pt` latents can be plugged into submodule training script | Format mismatch + collaborator-prepared (triplane purpose) → silent normalization risk | Round 0: Re-encode (Option B) to guarantee paper-spec format |
| Re-encoding all 5,000 train + 1,000 valid is needed | Small scale = 1500 is enough for code-path sanity | Round 0 user clarification: 1000 + 500 |
| One GPU is enough for encoding | 2 GPUs available, parallel encode halves wall-clock | Round 0 user clarification: 2-GPU parallel |
| Need quantitative metric (FID/CLIP) to declare success | First run is code-path validation; quant metric is Phase B | Round 1: Loss-only sanity (no quant eval) |
| Paper hyperparam set (batch=2, lr=1e-4, 100 epochs) used verbatim | 100 epochs × 1k samples = ~42h wall-clock, blows the budget | Round 2: epochs 100 → 10 (truncate); everything else paper-faithful |
| Latent precompute needs sanity checks (round-trip PSNR, stats match) | Existence + shape + non-NaN is enough for code-path check | Round 3: existence-only criteria |
| 3-TE precompute is required (Contrarian) | Could simplify to 1-TE / 768-d for code-path validation | Round 4: keep 3-TE (paper fidelity + reusable for full run) |
| Sample selection method matters for sanity | Alphabetical-first is reproducible enough; stratification is overkill | Round 5: alphabetical-first 1000/500 |

## Technical Context

### Existing scaffolding (Phase A artifacts we leverage)

| Artifact | Location | Use |
|---|---|---|
| MAISI VAE frozen loader | `src/baselines/maisi.py::load_frozen` | Encoder + decoder for latent precompute and (optional) mid-train sample decode |
| Report2CT UNet skeleton | `src/baselines/report2ct_adapter.py::build_unet` | Sanity check that the UNet config still instantiates the same 233 M model (not actually used during training — training uses submodule script's own UNet builder) |
| CT-RATE DataModule (metadata mode) | `src/data/ct_rate_datamodule.py` | Provides `load_records(split)` for the sample-report join |
| 3-text-encoder HF pins | `docs/report2ct_external_components.md` | Source-of-truth for precompute |
| MAISI bundle | `third_party/maisi_bundle/models/autoencoder.pt` + `configs/inference.json` | Override submodule's `trained_autoencoder_path` (which expects `autoencoder_epoch273.pt`) |

### To-build (this spec's deliverables)

| Artifact | Location | Owner |
|---|---|---|
| Sample-report join + datalist | `data/report2ct_smallrun/datalist.json` + helper in `src/data/report2ct_smallrun.py` | assistant |
| Latent precompute launcher | `scripts/precompute_latents_smallrun.sh` + `scripts/precompute_latents_worker.py` | assistant |
| Text embedding precompute | `scripts/precompute_text_emb_smallrun.py` | assistant |
| Training launcher | `scripts/run_report2ct_smallrun.sh` | assistant |
| Custom env config | `data/report2ct_smallrun/env_config.json` | assistant |
| Hydra experiment config | `configs/experiment/report2ct_smallrun.yaml` | assistant |
| Loss curve sanity checker | `scripts/check_smallrun_sanity.py` | assistant |

### Submodule entry points (READ-ONLY usage)

- `third_party/report2ct/src/maisi/scripts/vlm3d_image_embedding.py` — image latent precompute (invoke as subprocess with our env config)
- `third_party/report2ct/src/maisi/scripts/diff_model_train_vlm3D_2560_multi_text.py` — training loop (invoke via `torchrun` with our env config)
- `third_party/report2ct/vlm3D_work_dir/config_maisi_2560.json` — model arch (use as-is)
- `third_party/report2ct/vlm3D_work_dir/config_maisi_diff_model_vlm3D.json` — training schedule (override only `n_epochs` to 10)

## Ontology

| Entity | Type | Fields | Relationships |
|---|---|---|---|
| CT-RATE sample | core domain | volume_name, patient_id, study_id, nifti_path, findings, impression, spacing_xy, spacing_z, 18-labels | Joined to MAISI latent + text embedding |
| MAISI latent (mu.pt — existing) | data artifact | shape (4, 120, 120, 64), mean/std per channel | Existing collaborator-prepared, **not used** in this run |
| MAISI latent (_emb.nii.gz — to produce) | data artifact | NIfTI, shape (4, 120, 120, 64) (or submodule-determined), fp16/fp32 | Re-encoded by submodule via MAISI VAE |
| Text embedding JSON | data artifact | findings_embeddings (L, 2560), impression_embeddings (L, 2560), spacing (3,) | Per-sample multi_2560.json |
| Datalist JSON | data artifact | list[{volume_name, nifti_path, latent_emb_path, text_emb_path, findings, impression, spacing}] | The "manifest" the training script reads |
| Submodule training script | external runtime | torchrun + 3 config args | Invoked as subprocess by our launcher |
| Hydra experiment config | code artifact | epochs=10, paths, ... | Hydra side; meta-tracking only |
| Loss curve | sanity artifact | per-step train/loss, per-epoch val/loss; from WandB / TB | Inspected to declare smallrun PASS/FAIL |

## Ontology Convergence

| Round | Entities | New | Stable | Stability |
|---|---|---|---|---|
| 1 | 8 | 8 | — | N/A |
| 2 | 8 | 0 | 8 | 100% |
| 3 | 8 | 0 | 8 | 100% |
| 4 | 8 | 0 | 8 | 100% |
| 5 | 8 | 0 | 8 | 100% |

수렴 — 도메인 entity는 Round 1부터 안정.

## Interview Transcript

<details>
<summary>Full Q&A (5 rounds + Round 0)</summary>

### Round 0 (Topology)
**Q**: 5 component (sample-report join / latent format bridge / text emb precompute / training launcher / validation sanity) topology 맞나요?
**A**: 사용자 추가 의견 → "기존 mu.pt 그대로 쓸지 새로 인코딩할지 재고"가 더 중요. 5-component 확정 + (2)가 "format bridge"에서 "re-encode"로 의미 변경.
**Ambiguity**: 100% → 65% (estimated post-confirmation).

### Round 1 (Validation/sanity Criteria)
**Q**: 성공/실패 판단 기준?
**A**: Loss-only sanity (train+valid loss decreasing, no NaN/diverge, ~3-6h budget).
**Ambiguity**: 65% → 35%.

### Round 2 (Training launcher Constraints)
**Q**: Hyperparam — paper 그대로 vs 변형?
**A**: Paper 그대로 + epochs 100→10 (~4h).
**Ambiguity**: 35% → 32%.

### Round 3 (Latent precompute Criteria)
**Q**: Encoding 완료 검증 어디까지?
**A**: Existence-only (1500 files + shape + non-NaN).
**Ambiguity**: 32% → 26%.

### Round 4 (Contrarian — Text emb precompute)
**Q**: 3-TE 그대로 vs 1-TE simplification?
**A**: 3-TE 그대로 (paper fidelity, 본 학습 reuse).
**Ambiguity**: 26% → 25%.

### Round 5 (Sample-report join Criteria)
**Q**: 1000/500 선정 방식?
**A**: Alphabetical first 1000/500 (deterministic, reproducible).
**Ambiguity**: 25% → **20%** — PASSED.

</details>
