# VLM3D 2026 Pivot — Implementation Plan

**Spec source:** `/workspace/.omc/specs/deep-interview-vlm3d-pivot.md`
**Generated:** 2026-05-26 (Iteration 2 revision)
**Mode:** Consensus (RALPLAN-DR, short)
**Hard deadline:** 2026-08-20 (MICCAI VLM3D 2026 Task 4)

---

## 1. Requirements Summary

### Goal
Beat **Report2CT** on MICCAI VLM3D 2026 Task 4 (Text-Conditional CT Generation) by 2026-08-20, measured on CT-RATE valid 1000-split via the VLM3D-Dockers harness (FVD + CLIPScore + 2.5D-FID). Stack: lightning-hydra-template repo, MAISI VAE latent diffusion with VLM-focused conditioning improvements, baseline diagnostic suite (4 modules, applied per architectural feasibility) to pinpoint Report2CT's weaknesses, then exploit them in our model.

### Constraints (binding)
- **Deadline**: 2026-08-20 — Phase A 5 working days (5/26–5/31, 5/30–5/31 weekend bundled as Day 5), Phase B 4 wk, Phase C 4 wk, Phase D 3 wk.
- **Hardware**: ≤3× A6000 Blackwell (DeepCGV-Mk7), Python 3.11, PyTorch 2.x, MONAI, wandb. GPU convention: `CUDA_VISIBLE_DEVICES=0` prefix for every script (`/workspace/CLAUDE.md` GPU convention).
- **Baseline policy**:
  - GenerateCT: pretrained-only inference via 3 checkpoints (`ctvit.pt`, `transformer.pt`, `superres.pt`) from `huggingface.co/generatect/GenerateCT/pretrained_models/`.
  - Report2CT: weights NOT public → reimplement from paper (`/workspace/paper_pdf/Report2CT.pdf`), train on CT-RATE `train_fixed/`, keep MAISI VAE-GAN frozen (note: `/workspace/maisi_bundle/models/` ships only `autoencoder.pt`; no discriminator weights — see Open Question #6 / R6).
- **External code unmodified**: third-party repos live under `third_party/` as git submodules with SHA-pinned `git checkout`; bridged by adapter LightningModules in `src/baselines/`.
- **Hard deliverable**: VLM3D-Dockers-compliant submission docker (Dockerfile + `test.sh` + `export.sh`). A *stub* version is required by **Phase A Day 5** (P3 enforcement).
- **Triplane retired**: Existing triplane code/configs/runs move to `/workspace/deprecated/`. The `/workspace/reference/` directory (trivae adjacent variants `trivae.py`, `trivae2.py`, `trivae3.py`, `trivae4.py`) is also triplane-adjacent — move `reference/models/trivae*.py` to `/workspace/deprecated/reference_trivae/` on Phase A Day 1. Keep `reference/configs/` and `reference/scripts/` in place if VLM3D pipelines reference them (verify Day 1 EOD).
- **Datasets**: read-only at `/workspace/datasets/datasets/{CT-RATE,latents}/`; new artifacts under `/workspace/data/`.

### Non-Goals
Additional triplane work; baseline retraining beyond Report2CT-from-paper; clinical-grade QC; new dataset preprocessing pipelines; TRACE quantitative comparison; non-text modalities.

---

## 2. RALPLAN-DR (short mode)

### Principles (5)
1. **Clean codebase before scale.** Full lightning-hydra migration in Phase A; downstream phases can't tolerate a messy base when timelines compress.
2. **External code is read-only.** Submodules + adapter layer (`src/baselines/<repo>_adapter.py`). No vendored copies, no inline edits. Submodule SHAs pinned on Day 1 via `git -C third_party/<repo> checkout <SHA>`.
3. **Submission docker is a hard contract.** VLM3D-Dockers `test.sh`/`export.sh` must pass; eval pipeline must be the *same* harness as the leaderboard **from Phase A Day 5** — a *stub* docker wired to GenerateCT outputs is required at end of Phase A, not deferred.
4. **Diagnose, then improve.** Phase B closes with diagnostic findings on both baselines applied **where architecturally feasible** (cross-attn diagnostics may not apply to GenerateCT's CT-ViT + MaskGIT pipeline; see `docs/diagnostic_baseline_compatibility.md`). **Report2CT envelope clause**: our v1 design in Phase C is justified by diagnostic findings *and* must point at concrete weaknesses Report2CT exhibits *within its paper-reported envelope*; if our repro misses envelope, downgrade the claim (see Win Condition branch). A v1→C bridge gate (6/28–6/30) blocks Phase C kickoff until v1 design has Critic review + ≥2 diagnostic findings mapped to architectural choices.
5. **Compute is the silent constraint.** Wall-clock budget is reassessed at the 6/1 measurement gate; if Report2CT reimpl is too slow, we cut scope (subset, simplification) rather than slip the docker deadline.

### Top-3 Decision Drivers
1. **Hard deadline 2026-08-20** — drives the order: docker harness, submission scaffold, and eval-runner must exist before the model is "good"; nothing else can recover slip.
2. **Report2CT weight unavailability** — forces a re-implementation (added ~3-4 weeks of compute risk), moving the bar from "inference-only baseline" to "Phase-B trained baseline + sanity check vs paper FVD/CLIPScore *within numerical envelope*".
3. **DeepCGV-Mk7 compute reality (3× A6000)** — small enough that a 1mm full-res training run cannot be sized in one cycle; Phase B kickoff must do a measured 1-epoch wall-clock on 100 samples *before* committing to full-data Report2CT training. A 1-step correctness gate (loss monotone-decreasing over 10 steps on a single fixed batch) is required on Day 4 *before* the 6/1 wall-clock gate.

### Viable Options (3)

#### Option A: Full lightning-hydra migration in Phase A + Report2CT reimpl (CHOSEN)
- **Pros**: Phases B/C/D all benefit from a stable Hydra config tree, LightningModule contracts, and pytest scaffolding. Aligns with Round-6 user preference (clean structure > schedule risk). No mid-project refactor. Win-condition narrative ("beat Report2CT") fully supported.
- **Cons**: 5-day Phase A is tight. If migration slips, GenerateCT first-inference also slips and downstream phases bleed. Report2CT reimpl adds 3–4 weeks of compute risk in Phase B.
- **Mitigation**: explicit cut-list in Risk section (eval-runner can carry to Day 6 — but NOT past Day 5 for the stub docker; EDA scope deflated to 4 PNGs). Day-1 deliverable is `git mv` + submodule add (SHA-pinned) + pytest discovery green — no LightningModule logic yet.

#### Option B: Skeleton + lazy migration
- **Pros**: Phase A finishes early, GenerateCT inference lands Day 2-3.
- **Cons**: Half-migrated repo = config drift, two ways to instantiate models, every Phase B/C/D PR pays interest. Round 6 already rejected this on user preference. Submission docker (Phase D) becomes harder because configs and entrypoints aren't standardized.
- **Status**: NOT CHOSEN. Documented for ADR completeness.

#### Option C: GenerateCT-only baselines (no Report2CT reimpl)
- **Pros**: Phase A unchanged but Phase B compresses dramatically — no 3–4 wk Report2CT training arc, no compute gate risk. Phase B reduces to diagnostics + VLM3D-Dockers eval on GenerateCT only. Frees ~3 weeks for our model exploration.
- **Cons**: Weakens the publishable claim to "competitive with GenerateCT" only. VLM3D 2025 winner (Report2CT) absent from comparison set → reviewers will ask "why not Report2CT?" → answer becomes "weights unavailable" which is weaker than "reimplemented and beaten".
- **Status**: NOT CHOSEN. **Invalidation rationale**: spec Section "Win condition" explicitly names Report2CT as the bar; user (Round 8) accepted reimpl risk to keep the narrative. Retained as the contingency landing zone if R2 fires and "best-effort repro" is also impractical (e.g., compute gate shows >6 wk training requirement). If Option C activates mid-Phase-B, requires explicit user approval.

#### Sub-decision: Report2CT reimpl now (Phase B) vs scout substitute baseline
- **Reimpl (CHOSEN)**: matches Round-8 decision; ensures fair comparison with the actual 2025 winner. Compute risk is real but bounded by the 6/1 measurement gate.
- **Substitute baseline (REJECTED)**: no published alternative occupies the same "MAISI VAE-GAN + multi-encoder LDM on CT-RATE" niche; substituting (e.g., TRACE 2.5D) would invalidate the win-condition narrative ("beat Report2CT"). Invalidation rationale: spec Section "Win condition" requires comparison vs Report2CT specifically.

### ADR (final, recorded for Phase A handoff)
- **Decision**: Full lightning-hydra Phase A + Report2CT reimpl in Phase B + adapter pattern for third-party code + VLM3D-Dockers as eval-from-day-one (stub docker at Phase A Day 5).
- **Drivers**: deadline, weight availability, compute budget.
- **Alternatives considered**: Option B (skeleton-only migration); Option C (GenerateCT-only baselines); Report2CT substitution by TRACE; vendoring external code.
- **Why chosen**: leaderboard requires Report2CT comparison + clean base accelerates 3 downstream phases + docker harness is the literal deliverable. Stub-docker-by-Day-5 closes the P3 gap; metric-specific numerical envelope (±15% 2.5D-FID paper-anchored Fig 6 3-TE-cfg5, ±10% CLIPScore-T2I paper-anchored Fig 5 3-TE-cfg5, ±25% FVD self-anchored — FVD not reported in paper) protects against silent claim weakening.
- **Consequences**: 5-day Phase A is the most compressed window of the project; we accept slip risk on EDA polish (now 4 PNGs not 6) and KS-drift (deferred to Phase B nice-to-have) but not on `deprecated/` move, submodule wiring, stub submission docker, or Report2CT paper read.
- **Follow-ups**: (a) Phase B compute gate on 6/1 with 1-step correctness pre-gate on 5/29 (Day 4); (b) Phase B→C bridge gate 6/28–6/30 — Critic review of `our_model_v1_design.md` is mandatory before Phase C training begins; (c) diagnostic shortlist gate 6/14: if B.2 progress is behind, drop counterfactual first (most expensive).

---

## 3. Acceptance Criteria (testable, by phase)

### Phase A — 5/26 → 5/31 (5 working days; 5/30–5/31 weekend bundled as Day 5)
- [ ] `/workspace/deprecated/` exists and contains: `triplane_src/` (= former `/workspace/src/`), `triplane_configs/` (= former `/workspace/configs/trial_*.yaml` + `configs/{model,train,data,eval,loss}/`), `triplane_runs/` (= former `/workspace/runs/`), `triplane_tests/` (= former `/workspace/tests/test_{cross_attn_3d,d3t,tier0_overfit,tri_conv,triplane_ae_*,forward_backward,pos_embed_gradient,shapes,z_init_gradient,resume,dataset}.py`), and `reference_trivae/` (= moved `/workspace/reference/models/trivae{,2,3,4}.py`). Verifiable: `ls /workspace/deprecated/triplane_src/models/triplane_ae.py` succeeds; `ls /workspace/deprecated/reference_trivae/trivae2.py` succeeds.
- [ ] New `/workspace/src/` has subdirs `data/`, `models/`, `baselines/`, `eval/`, `utils/`, plus `train.py`, `eval.py`. Verifiable: `python -c "import src.train; import src.eval"` from `/workspace` exits 0.
- [ ] New `/workspace/configs/` has Hydra hierarchy: `data/`, `model/`, `trainer/`, `logger/`, `callbacks/`, `experiment/`, `hparams_search/`, plus root `train.yaml`, `eval.yaml`. Verifiable: `python /workspace/src/train.py --cfg job --resolve` prints a fully resolved Hydra config with no resolver errors.
- [ ] `/workspace/third_party/{report2ct,generatect,vlm3d_dockers}/` exist as git submodules with **explicit SHA pin** recorded in `/workspace/.omc/plans/submodule_pins.md`. Verifiable: `git submodule status` shows three entries at the recorded SHAs.
- [ ] `/workspace/requirements.txt` has 6 added packages pinned with versions (`pytorch-lightning==2.x.y`, `hydra-core==1.3.x`, `hydra-colorlog==1.2.0`, `rich==13.x.y`, `transformers==4.x.y`, `huggingface_hub==0.x.y`). Verifiable: `grep -E "^(pytorch-lightning|hydra-core|hydra-colorlog|rich|transformers|huggingface_hub)==" /workspace/requirements.txt | wc -l` ≥ 6.
- [ ] `/workspace/src/baselines/generatect_adapter.py` exposes a `GenerateCTAdapter(LightningModule)` that loads `ctvit.pt`/`transformer.pt`/`superres.pt` from `/workspace/data/checkpoints/generatect/` and runs text→volume on 1 sample. Verifiable: `pytest tests/test_generatect_adapter.py::test_one_sample_inference` passes.
- [ ] `/workspace/.omc/plans/report2ct_impl_spec.md` written by **Day 2 EOD**, sourced from `/workspace/paper_pdf/Report2CT.pdf` (text encoder, conditioning, UNet, training schedule, loss, page citations for each baseline number used in envelope locking).
- [ ] **Day 4 1-step correctness gate**: Report2CT paper-spec UNet skeleton forward-passes one batch; loss curve over 10 optimizer steps on one fixed batch is **monotone-decreasing**. Verifiable: `/workspace/runs/phase_a_day4/report2ct_correctness.log` exists; tail shows `step 9 loss < step 0 loss` and no NaN.
- [ ] **`docs/diagnostic_baseline_compatibility.md`** at `/workspace/docs/diagnostic_baseline_compatibility.md` maps each of {cross_attn, retrieval, counterfactual, token_region} × each baseline {GenerateCT (CT-ViT + MaskGIT + SR diffusion), Report2CT_our_repro (multi-encoder cross-attn UNet LDM)}; flags infeasible combinations (e.g., GenerateCT cross-attn heatmap is infeasible as-defined — UNet cross-attn substitute is GenerateCT's MaskGIT-transformer self-attention, must use a different hook). Verifiable: `test -f /workspace/docs/diagnostic_baseline_compatibility.md` and content includes a 4×2 table.
- [ ] **`cross_attn.py` smoke against GenerateCT** (architect-modified suggestion #2): a stripped-down attention-hook smoke test under `tests/test_diagnostic_cross_attn_generatect_smoke.py` runs against GenerateCT's transformer self-attention as a poor-but-informative analog, to catch architectural mismatch early. Verifiable: `CUDA_VISIBLE_DEVICES=0 pytest tests/test_diagnostic_cross_attn_generatect_smoke.py -q` exits 0.
- [ ] **MAISI frozen-load test (R6)**: `pytest tests/test_maisi_frozen_load.py` exists and passes by Phase A Day 2 EOD; asserts `/workspace/maisi_bundle/models/autoencoder.pt` loads and `requires_grad=False` on all VAE params; documents missing-discriminator-weight implication in test docstring.
- [ ] **Phase-A end-of-phase Lightning-wiring assertion**: `pl.Trainer.fit(model=GenerateCTAdapter | DummyLightningModule, datamodule=MAISILatentDataModule, max_steps=1)` runs ≥1 step without error. Verifiable: `tests/test_lightning_fit_smoke.py` passes.
- [ ] `/workspace/notebooks/eda.ipynb` plus `/workspace/figs/eda/` PNGs cover: HU histogram per kernel + manufacturer (`figs/eda/hu_by_kernel.png`, `figs/eda/hu_by_manufacturer.png`), spacing/dim violin (`figs/eda/spacing_violin.png`), 18-label frequency (`figs/eda/label_freq.png`). (Deflated scope: KS-drift CSV + co-occurrence + report-length plot moved to Phase B nice-to-have.) Verifiable: `find /workspace/figs/eda -name '*.png' | wc -l` ≥ 4.
- [ ] ≥50 valid samples generated by GenerateCT, stored at `/workspace/data/baselines/generatect/valid_50/<sample_id>/volume.npy` (or `.nii.gz`), plus one human-review markdown note at `/workspace/data/baselines/generatect/valid_50/review.md`. Verifiable: `find /workspace/data/baselines/generatect/valid_50 -name volume.npy | wc -l` ≥ 50.
- [ ] **Stub submission docker at `/workspace/submission/`** (P3 enforcement): `Dockerfile`, `test.sh`, `export.sh`, `fixtures/` (5-sample). The stub wires GenerateCT outputs through the VLM3D-Dockers metric pipeline. Verifiable: `bash /workspace/submission/test.sh` exits 0 on the 5-sample fixture; `/workspace/submission/output/metrics.json` written.
- [ ] **Envelope locked end-of-Phase-A (5/31)** with **metric-specific anchors** (Critic iter-3 verification of `/workspace/paper_pdf/Report2CT.pdf` confirmed FVD is NOT reported in the paper; only FID with 2.5D feature extraction + CLIPScore-T2I + CLIPScore-I2I are reported across Figures 5–6):
   - **2.5D-FID anchor**: cite `Report2CT.pdf` Figure 6 (FID Average) for the **3-TE-cfg5** configuration (FID Average ≈ 4.04). Envelope = **±15%**. (3-TE-cfg5 chosen as Report2CT's headline best configuration — 3 text encoders + classifier-free-guidance scale 5, matching the paper's main result row; document rationale in `/workspace/.omc/plans/report2ct_envelope.md`.)
   - **CLIPScore anchor**: use **CLIPScore-T2I from Figure 5** for the same 3-TE-cfg5 configuration as the **PRIMARY** anchor. Envelope = **±10%**. Report **CLIPScore-I2I** as **SECONDARY** (informational only, no envelope bound).
   - **FVD anchor**: FVD is **NOT reported** in the Report2CT paper. Anchor will be **self-measured** from the 6/1 paper-spec-skeleton 1-epoch run on CT-RATE valid 1000. Envelope = **±25%** (widened to absorb the unanchored baseline). Document explicitly in `report2ct_envelope.md`: "FVD not reported in Report2CT paper; anchor is our 6/1 paper-spec measurement (looser bound than other metrics)."
   - **Citations required only for the two paper-anchored metrics** (2.5D-FID Fig 6 cell, CLIPScore-T2I Fig 5 cell, both for 3-TE-cfg5). FVD line in `report2ct_envelope.md` carries the explicit "self-measured, unanchored" disclaimer in lieu of a paper citation.
   - Proposal drafted Day 3, finalized Day 5.
- [ ] `pytest /workspace/tests/` passes end-to-end with at least: `test_generatect_adapter.py`, `test_hydra_compose.py`, `test_data_module.py`, `test_maisi_frozen_load.py`, `test_lightning_fit_smoke.py`, `test_diagnostic_cross_attn_generatect_smoke.py`. Verifiable: `pytest /workspace/tests/ -q` exits 0.

### Phase B — 6/1 → 6/30 (4 weeks)

> **Scope Boundary (user clarification 2026-05-26)**: Assistant deliverable for Report2CT in Phase B is **training-ready code preparation**, not the full training run. User executes the actual multi-day training and reports back the checkpoint + sanity numbers. Acceptance criteria below split into **A: assistant-owned (code + smoke)** and **U: user-owned (training execution)**. Phase B exit gate fires when A items are complete; U items become prerequisites for Phase C diagnostics that consume the Report2CT checkpoint.

> **Metric scope clarification (user 2026-05-26)**: Since this is a challenge submission, **ALL VLM3D-Dockers metrics are measured for every model** (GenerateCT inference, Report2CT_our_repro, ours_v1, ours_final), regardless of whether the source paper reports each metric. The envelope's per-metric anchor strategy (paper-anchored vs self-anchored) determines only the *Phase B sanity comparison target*; it does NOT reduce the measurement set. FVD is always measured and reported alongside 2.5D-FID and CLIPScore-T2I/I2I on every model run.

- [A] **6/1 compute gate (code-side)**: `/workspace/runs/phase_b_gate/report2ct_1epoch_100samples_smoke.log` produced by **assistant-run** 1-epoch-on-100-samples smoke (built on the Day-2 paper-spec skeleton, not a stub) — confirms code runs and produces a wall-clock-per-step number; `/workspace/.omc/plans/phase_b_budget.md` written with the assistant-measured rate + projected full-train GPU-hours and one of decisions (a) subset size, (b) overlap with Phase C, (c) Report2CT simplification. If GPU-hours forecast > 6 weeks for full training, R2 fallback escalates to Option C in §2.
- [U] **Full Report2CT training run** (user-driven): user launches multi-day training using the prepared `scripts/run_report2ct_training.sh` + Hydra config; resulting checkpoint lands at `/workspace/data/checkpoints/report2ct/our_repro/best.ckpt`. Wall-clock timing logged by user to `/workspace/runs/report2ct_our_repro/training.log`.
- [ ] **6/14 diagnostic shortlist gate**: `/workspace/.omc/plans/diagnostic_shortlist.md` records whether all 4 diagnostics ship by 6/30 or whether counterfactual is dropped first (most expensive). Decision recorded with B.2 progress evidence.
- [A] `/workspace/src/baselines/report2ct/` contains paper-based reimpl with files `text_encoders.py`, `unet_ldm.py`, `report2ct_module.py` (LightningModule), `config.yaml`. **Reuse-first policy** (user clarification 2026-05-26): text encoders use pretrained HF checkpoints (e.g., `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` or whatever the Report2CT paper specifies); UNet backbone reuses `diffusers.UNet2DConditionModel`/`UNet3DConditionModel` or `monai.generative`'s diffusion UNet where structurally compatible; LDM scheduler from `diffusers.schedulers`. Custom code written ONLY for paper-specific glue (multi-encoder fusion, spacing conditioning, MAISI VAE-GAN adapter). Each adopted external component documented in `docs/report2ct_external_components.md` with HF/GitHub URL + version pin. MAISI VAE-GAN is loaded frozen via `/workspace/maisi_bundle/models/autoencoder.pt`. Verifiable: `pytest tests/test_report2ct_module.py::test_forward_one_batch` passes AND `pytest tests/test_report2ct_module.py::test_overfit_one_batch` passes (Tier-0 overfit on single batch ⇒ confirms training pipeline is correct).
- [A] Training launch artifacts ready for user handoff: `/workspace/configs/experiment/report2ct_repro.yaml` (full Hydra experiment config) + `/workspace/scripts/run_report2ct_training.sh` (torchrun launcher with documented `CUDA_VISIBLE_DEVICES`, multi-GPU, resume-from-checkpoint flags) + `/workspace/docs/report2ct_training_handoff.md` (user-facing runbook: expected wall-clock from 6/1 smoke, monitoring tips, checkpoint naming convention, sanity-metric reporting template).
- [U] Report2CT trained checkpoint at `/workspace/data/checkpoints/report2ct/our_repro/best.ckpt` produced by **user-driven training run**; `/workspace/results/report2ct_sanity.json` filled in by user using the schema template the assistant prepared, records 2.5D-FID, CLIPScore-T2I (primary), CLIPScore-I2I (secondary informational), and FVD, judged against the locked metric-specific envelope (±15% 2.5D-FID vs Fig 6 3-TE-cfg5 ≈ 4.04, ±10% CLIPScore-T2I vs Fig 5 3-TE-cfg5, ±25% FVD vs self-measured 6/1 anchor — FVD not reported in paper) with paper page/figure citations for the two paper-anchored metrics.
- [ ] Diagnostic modules implemented under `/workspace/src/eval/diagnostic/`, applied **per `docs/diagnostic_baseline_compatibility.md`**:
  - [ ] `cross_attn.py` — hooks cross-attn layers on Report2CT UNet (and self-attn analog on GenerateCT MaskGIT-transformer where compatibility table marks "partial"); dumps per-token spatial maps + sagittal/coronal/axial PNG grids to `/workspace/figs/diagnostic/cross_attn/<model>/<sample>/`.
  - [ ] `retrieval.py` — CT-CLIP-based R@1/R@5/R@10 scatter for both baselines, output `/workspace/results/diagnostic/retrieval_<model>.json` + `/workspace/figs/diagnostic/retrieval_<model>.png`.
  - [ ] `counterfactual.py` (subject to 6/14 shortlist gate) — report_A vs report_A_minus_finding diff map, output PNG grid `/workspace/figs/diagnostic/counterfactual/<model>/<finding>/<sample>.png`.
  - [ ] `token_region.py` — applied only where cross-attn hookable per compatibility table; cross-attn × TotalSegmentator/VISTA-3D anatomy seg IoU/Dice table at `/workspace/results/diagnostic/token_region_<model>.csv`.
- [ ] EDA carryover (deflated from Phase A): KS-drift CSV `/workspace/figs/eda/ks_drift.csv` + co-occurrence + report-length PNGs delivered as Phase B nice-to-have.
- [ ] `/workspace/src/eval/vlm3d_runner.py` invokes VLM3D-Dockers via subprocess for both `generatect` and `report2ct_our_repro`, full CT-RATE valid set, with outputs at `/workspace/results/vlm3d/<model>/metrics.json`. Verifiable: `python -m src.eval.vlm3d_runner --baseline=generatect --split=valid` runs to completion and writes metrics.json.
- [ ] **6/27 v1 design draft**: `/workspace/.omc/plans/our_model_v1_design.md` first complete draft, documenting which diagnostic finding each architectural choice targets (≥2 concrete mappings required).
- [A] **6/28–6/30 Phase B→C bridge gate** (P4 enforcement): (a) Critic review of `our_model_v1_design.md` recorded at `/workspace/.omc/reviews/our_model_v1_design_critic.md`; (b) ≥2 diagnostic findings concretely mapped to architectural choices verified by Critic. Phase C kickoff on 7/1 is **conditional** on both items being signed off. **NOTE on user-owned training**: diagnostics that consume `report2ct_our_repro` checkpoint (cross_attn, retrieval, counterfactual, token_region) are gated on user completing the [U] training. If user training is incomplete by 6/27, the v1 design draft uses GenerateCT-only diagnostic findings + a contingency mapping clause for Report2CT findings to be incorporated when checkpoint arrives.

### Phase C — 7/1 → 7/31 (4 weeks)
- [ ] Our model v1 LightningModule at `/workspace/src/models/ours_v1_module.py` + Hydra experiment `/workspace/configs/experiment/ours_v1.yaml`. Verifiable: `pytest tests/test_ours_v1_module.py::test_overfit_one_batch` passes (Tier-0 style overfit).
- [ ] Trained checkpoint `/workspace/data/checkpoints/ours/v1/best.ckpt`; applicable diagnostics applied per compatibility table → outputs under `/workspace/figs/diagnostic/<diag>/ours_v1/` and `/workspace/results/diagnostic/<diag>_ours_v1.*`.
- [ ] ≥3 ablations under `/workspace/configs/experiment/ours_v1_ablation_{text_encoder,cond_depth,attn_layer}.yaml` with results aggregated at `/workspace/results/ablations/ours_v1.csv`.
- [ ] VLM3D-Dockers metrics on `ours_v1` at `/workspace/results/vlm3d/ours_v1/metrics.json`; ≥2/3 metrics improve vs `report2ct_our_repro`. Verifiable: `python -m src.eval.compare_vlm3d --a=ours_v1 --b=report2ct_our_repro` prints PASS/FAIL.

### Phase D — 8/1 → 8/20 (3 weeks)
- [ ] Final training run logged at `/workspace/runs/ours_final/`; best checkpoint copied to `/workspace/data/checkpoints/ours/final/best.ckpt`.
- [ ] Production submission docker at `/workspace/submission/` (Phase A stub graduated to production): `Dockerfile`, `test.sh`, `export.sh`, conforming to VLM3D-Dockers spec, baking in `ours/final/best.ckpt`. Verifiable: `bash /workspace/submission/test.sh` exits 0 and emits `/workspace/submission/output/metrics.json` matching VLM3D-Dockers schema.
- [ ] Method writeup at `/workspace/submission/writeup.md` (≤4 pages).
- [ ] Submitted; URL/receipt logged in `/workspace/.omc/plans/submission_receipt.md`.

### Global win condition (with R2 fallback branch)
- [ ] **Primary**: On CT-RATE valid 1000-split, via VLM3D-Dockers: `ours_final` beats `report2ct_our_repro` in ≥2 of {2.5D-FID, CLIPScore-T2I, FVD} (metric priority order for the headline claim: **2.5D-FID > CLIPScore-T2I > FVD**, since FVD has only a self-measured anchor; a 2/3 win that **excludes** 2.5D-FID requires the writeup to note the weaker anchor situation), **and** `report2ct_our_repro` is within the locked envelope (**±15% 2.5D-FID** paper-anchored, **±10% CLIPScore-T2I** paper-anchored, **±25% FVD** self-anchored) per the Phase A `report2ct_envelope.md`.
- [ ] **Branch (envelope miss)**: If `report2ct_our_repro` is **outside** envelope on either paper-anchored metric (2.5D-FID or CLIPScore-T2I), the publishable claim **downgrades** to "competitive with our reimplementation of Report2CT" (NOT "beats Report2CT"). A FVD-only envelope miss is treated as expected variance given the unanchored baseline but still recorded. Envelope miss does **not** silently pass: (a) `/workspace/.omc/plans/envelope_miss_addendum.md` written documenting the miss + likely cause; (b) explicit **user approval** is required before submission proceeds; (c) writeup footnote required; (d) Option C in §2 (GenerateCT-only) may be activated retroactively if user prefers a cleaner contingency narrative.

---

## 4. Implementation Steps

### Phase A — Day-by-day

**Day 1 (5/26 Mon): repo skeleton + deprecation move + submodule SHA pin + paper read kickoff**
- `mkdir -p /workspace/deprecated/{triplane_src,triplane_configs,triplane_runs,triplane_tests,reference_trivae}`.
- `git mv /workspace/src /workspace/deprecated/triplane_src`.
- `git mv /workspace/configs /workspace/deprecated/triplane_configs` (preserve `/workspace/results/`, `/workspace/maisi_bundle/` in place).
- `git mv /workspace/runs /workspace/deprecated/triplane_runs`.
- `git mv /workspace/tests /workspace/deprecated/triplane_tests`.
- `git mv /workspace/reference/models/trivae*.py /workspace/deprecated/reference_trivae/` (preserve `reference/configs/`, `reference/scripts/`, `reference/models/{__init__,dataloader,latent_dataset}.py` — verify Day 1 EOD that no new code imports them; if unused, also move to deprecated).
- `mkdir -p /workspace/src/{data,models,baselines,eval,utils} /workspace/configs/{data,model,trainer,logger,callbacks,experiment,hparams_search} /workspace/third_party /workspace/tests /workspace/notebooks /workspace/figs/eda /workspace/data/checkpoints /workspace/docs`.
- `touch /workspace/src/__init__.py /workspace/src/{data,models,baselines,eval,utils}/__init__.py`.
- Submodule registration **with SHA pin**:
  ```
  git submodule add https://github.com/sinaamirrajab/report2ct           third_party/report2ct
  git submodule add https://github.com/ibrahimethemhamamci/GenerateCT    third_party/generatect
  git submodule add https://github.com/forithmus/VLM3D-Dockers           third_party/vlm3d_dockers
  # Pin each to known-good SHA (latest at clone time)
  git -C third_party/report2ct      checkout <SHA>
  git -C third_party/generatect     checkout <SHA>
  git -C third_party/vlm3d_dockers  checkout <SHA>
  # Record SHAs
  printf 'report2ct: %s\ngeneratect: %s\nvlm3d_dockers: %s\n' \
    "$(git -C third_party/report2ct rev-parse HEAD)" \
    "$(git -C third_party/generatect rev-parse HEAD)" \
    "$(git -C third_party/vlm3d_dockers rev-parse HEAD)" \
    > /workspace/.omc/plans/submodule_pins.md
  ```
- Append 6 packages to `/workspace/requirements.txt` with explicit version pins (`pytorch-lightning`, `hydra-core`, `hydra-colorlog`, `rich`, `transformers`, `huggingface_hub`).
- Create empty Hydra root configs: `/workspace/configs/train.yaml`, `/workspace/configs/eval.yaml` with `defaults: [data: default, model: null, trainer: default, logger: default, callbacks: default, experiment: null, _self_]`.
- Add `pytest.ini` (rootdir = `/workspace`, testpaths = `tests`). Smoke test: `pytest tests/ -q` should exit 0 (empty discovery is OK).
- **Day 1 afternoon: start reading `/workspace/paper_pdf/Report2CT.pdf`** — outline text encoder choices, UNet conditioning, training schedule. Stub `/workspace/.omc/plans/report2ct_impl_spec.md` with headings.
- **R5 verification on CT-CLIP availability** logged to `/workspace/.omc/plans/ct_clip_availability.md` by EOD.

**Day 2 (5/27 Tue): Hydra entrypoints + DataModule + paper-spec skeleton + MAISI frozen test**
- `/workspace/src/train.py`: `@hydra.main(config_path="../configs", config_name="train")` → `pl.Trainer.fit(model, datamodule)`.
- `/workspace/src/eval.py`: mirrors `train.py` but `Trainer.validate`/`predict`.
- `/workspace/src/data/maisi_latent_datamodule.py`: LightningDataModule wrapping `/workspace/deprecated/triplane_src/data/maisi_latent_dataset.py` semantics (port dataset class into `src/data/maisi_latent_dataset.py`, then wrap).
- `/workspace/configs/data/maisi_latents_1mm.yaml`, `/workspace/configs/data/maisi_latents_2mm.yaml`.
- Port metrics: copy `/workspace/deprecated/triplane_src/metrics/image_metrics.py` → `/workspace/src/eval/image_metrics.py`.
- Tests: `tests/test_hydra_compose.py` (asserts `train.yaml` composes via `--cfg job --resolve`), `tests/test_data_module.py` (one batch loads from `/workspace/datasets/datasets/latents/`).
- **`tests/test_maisi_frozen_load.py`** (R6): asserts `/workspace/maisi_bundle/models/autoencoder.pt` loads via the MAISI loader; all params have `requires_grad=False` after the frozen helper runs; documents in the test docstring that `maisi_bundle/models/` does *not* ship discriminator weights (the GAN's discriminator was discarded at distribution time — confirms our pipeline only needs the VAE encoder/decoder).
- **`/workspace/.omc/plans/report2ct_impl_spec.md` finalized end of Day 2**: text encoder choice (BiomedCLIP vs Bio-ClinicalBERT — primary), UNet block diagram, conditioning entry points, training schedule, loss, page-cited paper numbers for envelope locking.

**Day 3 (5/28 Wed): GenerateCT adapter + checkpoint download + envelope proposal**
- Download to `/workspace/data/checkpoints/generatect/`:
  - `https://huggingface.co/generatect/GenerateCT/resolve/main/pretrained_models/ctvit.pt`
  - `https://huggingface.co/generatect/GenerateCT/resolve/main/pretrained_models/transformer.pt`
  - `https://huggingface.co/generatect/GenerateCT/resolve/main/pretrained_models/superres.pt`
- `/workspace/src/baselines/generatect_adapter.py`: imports from `third_party.generatect.<...>`, wraps in `GenerateCTAdapter(LightningModule)` with `forward(text: str) -> Tensor[1, D, H, W]`.
- `/workspace/configs/model/generatect.yaml` instantiates the adapter with checkpoint paths.
- Test: `tests/test_generatect_adapter.py::test_one_sample_inference` runs with `CUDA_VISIBLE_DEVICES=0` and saves output to `/workspace/data/baselines/generatect/smoke/sample0.npy`.
- **Envelope proposal draft**: `/workspace/.omc/plans/report2ct_envelope.md` v1 with metric-specific anchors: **2.5D-FID ±15%** (anchor: Report2CT Figure 6 FID Average, 3-TE-cfg5 ≈ 4.04), **CLIPScore-T2I ±10%** (anchor: Report2CT Figure 5, 3-TE-cfg5, primary; CLIPScore-I2I secondary informational), **FVD ±25%** (self-measured anchor on 6/1 paper-spec run — FVD not reported in Report2CT paper). To be finalized 5/31.

**Day 4 (5/29 Thu): batch inference + Report2CT 1-step correctness gate + EDA notebook draft**
- Script `/workspace/scripts/run_generatect_valid50.py` reads 50 report strings from `/workspace/datasets/datasets/CT-RATE/dataset/radiology_text_reports/valid/`, runs adapter, writes `/workspace/data/baselines/generatect/valid_50/<id>/volume.npy`. Prefix command with `CUDA_VISIBLE_DEVICES=0`.
- **1-step correctness gate** (architect-suggestion #3 split): `scripts/phase_a_day4_report2ct_correctness.py` builds the paper-spec multi-encoder UNet (using Day 2's `report2ct_impl_spec.md`), forward-passes one batch from MAISI latents, runs 10 optimizer steps on one fixed batch, asserts loss monotone-decreasing and no NaN. Log to `/workspace/runs/phase_a_day4/report2ct_correctness.log`. This is the **pre-condition** for 6/1's wall-clock gate.
- `/workspace/notebooks/eda.ipynb` cells (deflated to 4 figs):
  1. HU histogram per kernel (`metadata/validation_metadata.csv` column `kernel`).
  2. HU histogram per manufacturer.
  3. Spacing + dim violin.
  4. 18-label frequency.
- (Co-occurrence, KS-drift, report-length plots deferred to Phase B nice-to-have.)

**Day 5 (5/30 Sat – 5/31 Sun): stub submission docker + VLM3D-Dockers wrap + envelope lock + diagnostic compat doc + cross-attn smoke**

Weekend stance: Day 5 is the 5/30–5/31 weekend bundle. We accept weekend work for this single day; subsequent weekends are not pre-committed. If individual sub-items overrun, see cut order below.

- `/workspace/src/eval/vlm3d_runner.py`: subprocess wrapper around `third_party/vlm3d_dockers/task4/` containers; args: `--baseline`, `--split`, `--n`; writes `/workspace/results/vlm3d/<baseline>/metrics.json`.
- Smoke run: `python -m src.eval.vlm3d_runner --baseline=generatect --split=valid --n=50` writes partial `metrics.json`.
- **Stub submission docker** (P3 enforcement, NEW): `/workspace/submission/Dockerfile`, `test.sh`, `export.sh`, `fixtures/` (5 samples). Wires GenerateCT outputs through the VLM3D-Dockers metric pipeline. Acceptance: `bash /workspace/submission/test.sh` exits 0; `/workspace/submission/output/metrics.json` written.
- **`/workspace/docs/diagnostic_baseline_compatibility.md`**: 4 diagnostics × 2 baselines table. Mark cross_attn = "feasible (Report2CT)" / "partial via self-attn analog (GenerateCT)"; retrieval = "feasible (both)"; counterfactual = "feasible (both, but expensive)"; token_region = "feasible (Report2CT only) / N/A (GenerateCT — no spatial cross-attn)".
- **`tests/test_diagnostic_cross_attn_generatect_smoke.py`** (architect-modified suggestion #2): a stripped attention-hook smoke against GenerateCT transformer self-attention; verifies hook registers and dumps a map even if the analog is imperfect — surfaces architectural mismatch before Phase B blocks on it.
- **Envelope locked**: finalize `/workspace/.omc/plans/report2ct_envelope.md` with metric-specific anchors — **2.5D-FID ±15%** cited to Figure 6 FID Average 3-TE-cfg5 (≈4.04), **CLIPScore-T2I ±10%** cited to Figure 5 3-TE-cfg5 (primary), CLIPScore-I2I logged as secondary informational, **FVD ±25%** with explicit "self-measured on 6/1, FVD not reported in Report2CT paper" disclaimer.
- **Lightning fit smoke**: `tests/test_lightning_fit_smoke.py` — `pl.Trainer.fit(GenerateCTAdapter, datamodule, max_steps=1)` runs end-to-end.
- Human-review note: `/workspace/data/baselines/generatect/valid_50/review.md`.

**Cut order if Day 5 overruns** (Phase A risk mitigation; revised iter-3 — restored real slack by demoting cross-attn smoke from "NEVER cut" to slip-eligible):
1. Co-occurrence / KS-drift / report-length EDA cells → already deferred to Phase B (no further cut needed).
2. **Cross-attn GenerateCT smoke test** (`tests/test_diagnostic_cross_attn_generatect_smoke.py`) **may slip to Phase B Day 1 (6/1) if time permits**. Rationale: it's an early-warning architectural-compatibility probe, not a Phase B gate. The structural cover for this insight is `docs/diagnostic_baseline_compatibility.md` (still NEVER-cut), which captures the same architectural finding documentally; the smoke test is "would be nice to verify early in code" but Phase B Day 1 is an acceptable slip target.
3. Envelope citation-completeness for the **two paper-anchored metrics** (2.5D-FID, CLIPScore-T2I) can finalize Phase B Day 1 *if* the percentages and the 3-TE-cfg5 row selection are locked by 5/31 EOD. (The FVD line is self-anchored, so it has no paper-citation completion to defer.)
4. **KS-drift / non-PNG EDA outputs** (already deferred to Phase B as nice-to-have).
5. **NEVER cut (9 items)**: `deprecated/` move (incl. `reference_trivae/`), submodule add + SHA pin, GenerateCT adapter test, root Hydra compose `--resolve` test, **stub submission docker**, MAISI frozen-load test, paper read + impl spec, Day 4 1-step correctness gate, **numerical envelope lock (percentages + 3-TE-cfg5 selection)**, Lightning fit smoke, **diagnostic baseline compatibility doc** (`docs/diagnostic_baseline_compatibility.md`).

Result: **9 items "NEVER cut", 4 items cuttable** (EDA secondary cells, KS-drift, cross-attn smoke, envelope citation completeness for non-FVD metrics) — restored real slack.

### Phase B — Steps

**B.1 (6/1) Compute-measurement gate**
- Script `/workspace/scripts/phase_b_gate.py`: builds Report2CT module from Day-2 paper-spec skeleton (NOT a stub), runs 1 epoch on 100 samples from `/workspace/datasets/datasets/latents/train/` subset, logs per-step wall-clock to `/workspace/runs/phase_b_gate/report2ct_1epoch_100samples.log`.
- Decision doc `/workspace/.omc/plans/phase_b_budget.md`: extrapolated full-train GPU-hours, chosen path among (a) shrink subset, (b) eat into Phase C, (c) simplify Report2CT, (d) activate Option C (GenerateCT-only) with user approval.

**B.2 Report2CT reimpl + train**
- Files under `/workspace/src/baselines/report2ct/`:
  - `text_encoders.py` — paper-specified encoders per `report2ct_impl_spec.md`.
  - `unet_ldm.py` — multi-encoder cross-attention UNet over MAISI latent shape `[B, 4, 120, 120, 64]`.
  - `report2ct_module.py` — `Report2CTModule(LightningModule)`; uses frozen MAISI VAE from `/workspace/maisi_bundle/models/autoencoder.pt`.
  - `config.yaml` referenced by `/workspace/configs/model/report2ct_our_repro.yaml`.
- Experiment config `/workspace/configs/experiment/report2ct_train.yaml`; entrypoint `CUDA_VISIBLE_DEVICES=0 python src/train.py experiment=report2ct_train`.
- Sanity check at half-train: `/workspace/results/report2ct_sanity.json` records 2.5D-FID, CLIPScore-T2I (primary), CLIPScore-I2I (secondary informational), and FVD on a 200-sample valid subset, judged against the locked metric-specific envelope (±15% 2.5D-FID vs Fig 6 3-TE-cfg5 ≈ 4.04, ±10% CLIPScore-T2I vs Fig 5 3-TE-cfg5, ±25% FVD self-anchored — FVD not reported in paper).

**B.3 Four diagnostic modules (per compatibility table)**
- `/workspace/src/eval/diagnostic/cross_attn.py` — register forward hooks on Report2CT UNet cross-attn blocks; for GenerateCT apply the self-attn analog only where compatibility table marks "partial".
- `/workspace/src/eval/diagnostic/retrieval.py` — load CT-CLIP (R5); compute R@1/R@5/R@10 across CT-RATE valid; scatter plot.
- `/workspace/src/eval/diagnostic/counterfactual.py` — generate `report_A` and `report_A_minus_finding`; diff map; one row per (finding, sample). **Subject to 6/14 shortlist gate — dropped first if B.2 is behind.**
- `/workspace/src/eval/diagnostic/token_region.py` — pull TotalSegmentator masks from `/workspace/datasets/datasets/CT-RATE/dataset/ts_seg/` and VISTA-3D from `/workspace/datasets/datasets/CT-RATE/dataset/anatomy_segmentation_labels/`; compute IoU/Dice between thresholded cross-attn maps and anatomy masks per finding-token. Report2CT only per compat table.
- Test entrypoints: `pytest tests/test_diagnostic_{cross_attn,retrieval,counterfactual,token_region}.py`.

**B.4 Full VLM3D-Dockers eval + v1 design + bridge gate**
- `python -m src.eval.vlm3d_runner --baseline=generatect --split=valid` (full 1000).
- `python -m src.eval.vlm3d_runner --baseline=report2ct_our_repro --split=valid`.
- **6/27 draft**: `/workspace/.omc/plans/our_model_v1_design.md` first complete draft, mapping each weakness from diagnostics to a concrete architectural countermeasure.
- **6/28–6/30 Phase B→C bridge gate**: Critic review at `/workspace/.omc/reviews/our_model_v1_design_critic.md`; verify ≥2 diagnostic findings → architectural choices; Phase C kickoff 7/1 **only if both signed off**.

### Phase C — Our model v1

- `/workspace/src/models/ours_v1_module.py` extends MAISI VAE frozen latent diffusion with the design from `our_model_v1_design.md` (post-Critic).
- `/workspace/configs/experiment/ours_v1.yaml` + ablation siblings.
- Training entrypoint: `CUDA_VISIBLE_DEVICES=0 python src/train.py experiment=ours_v1`.
- Tier-0 overfit test in `/workspace/tests/test_ours_v1_module.py::test_overfit_one_batch` (pattern: see `/workspace/deprecated/triplane_tests/test_tier0_overfit.py`).
- Re-run applicable diagnostics on `ours_v1` checkpoint per compatibility table.
- Run VLM3D-Dockers on `ours_v1`; aggregate ablations into `/workspace/results/ablations/ours_v1.csv`.

### Phase D — Submission docker (production)

- Graduate Phase A stub docker → production: rebuild `/workspace/submission/Dockerfile` (FROM nvidia/cuda:12.x-runtime; install pinned deps from `/workspace/requirements.txt`; COPY `src/`, `third_party/`, `data/checkpoints/ours/final/`).
- `/workspace/submission/test.sh` matches VLM3D-Dockers task4 contract (reads `input/`, writes `output/metrics.json`).
- `/workspace/submission/export.sh` packages a single tar.
- Verify locally: `bash /workspace/submission/test.sh` on the 5-sample fixture under `/workspace/submission/fixtures/`.
- Final training already complete by Phase D start; this phase is integration + writeup only.
- Submit to MICCAI VLM3D 2026 portal; receipt logged.

---

## 5. Risks and Mitigations

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|------------|
| R1 | Phase A 5 days slips (full lightning-hydra migration + stub docker + paper read is aggressive) | High | High | Day-1 deliverable is `git mv` + submodule add + SHA pin only; **9 items never cut** (deprecation move, GenerateCT adapter test, stub submission docker, MAISI frozen-load test, paper read + impl spec, Day-4 correctness gate, envelope lock, Lightning fit smoke, diagnostic compat doc). **4 items cuttable** to restore slack: EDA secondary cells (already deferred), KS-drift, **cross-attn GenerateCT smoke test (slip-eligible to Phase B Day 1)**, envelope citation-completeness for non-FVD metrics. |
| R2 | Report2CT reimpl doesn't match paper metrics (outside locked envelope: ±15% 2.5D-FID paper-anchored to Fig 6 3-TE-cfg5, ±10% CLIPScore-T2I paper-anchored to Fig 5 3-TE-cfg5, ±25% FVD self-anchored — **FVD is not reported in `Report2CT.pdf`** so its anchor is our 6/1 paper-spec measurement, hence widened to ±25%) | High | High | 6/1 compute gate (`scripts/phase_b_gate.py`) measures wall-clock; 5/29 1-step correctness gate ensures the skeleton is right before scale-out. Half-train sanity check at `/workspace/results/report2ct_sanity.json` judged against locked envelope. If outside envelope on a paper-anchored metric (2.5D-FID or CLIPScore-T2I): (a) inspect text encoder vs paper, (b) drop conditioning depth one level, (c) escalate to Option C (GenerateCT-only) with user approval — Win Condition downgrade clause activates, writeup footnote required, **no silent pass**. FVD-only miss is recorded but not auto-escalated (unanchored baseline). |
| R3 | GPU contention on 3× A6000 between Report2CT train (Phase B), diagnostics (Phase B), our v1 train (Phase C) | Med | High | Strict `CUDA_VISIBLE_DEVICES=0` discipline per `/workspace/CLAUDE.md`; one training job per GPU; diagnostics run on GPU 1/2 in parallel. Document GPU assignment in `/workspace/.omc/plans/phase_b_budget.md`. |
| R4 | VLM3D-Dockers docker subprocess fails inside our dev container (docker-in-docker) | Med | High | Test `third_party/vlm3d_dockers/task4/test.sh` on Phase A Day 5 with the 50-sample GenerateCT output through the stub submission docker. If docker-in-docker is blocked, fall back to running the docker's metric scripts directly in-process; flag as Phase B Day 1 priority if it fails. |
| R5 | CT-CLIP weights for retrieval diagnostic may be unavailable/restricted | Med | Med | Verify availability on Day 1 by visiting `huggingface.co/ibrahimethemhamamci/CT-CLIP`. If gated/missing: (a) request access, (b) substitute with BioBERT text + 2.5D image encoder, (c) drop retrieval and replace with a clip-similarity heatmap diagnostic. Decision in `/workspace/.omc/plans/ct_clip_availability.md` by 5/27 EOD. |
| R6 | MAISI VAE checkpoint path drift or discriminator-weights mismatch (no `discriminator.pt` in `maisi_bundle/models/`) | Low | High | Pin commit of `/workspace/maisi_bundle/` and reference `/workspace/maisi_bundle/models/autoencoder.pt` absolute path in `src/baselines/report2ct/report2ct_module.py` + `src/models/ours_v1_module.py`. **`pytest tests/test_maisi_frozen_load.py`** is on the Phase A Day 2 required-test list; the test docstring confirms our pipeline only needs the VAE encoder/decoder (no GAN discriminator needed for inference-time encode/decode). |
| R7 | 3D MAISI latent streaming I/O (~43 GB) dominates training time | High | Med | Per `/workspace/CLAUDE.md` and `3d_latent_io_bottleneck.md`: `num_workers≥8`, `prefetch_factor≥4`, `pin_memory=True`, `non_blocking=True`. Cache summary stats on first pass. Use `/workspace/data/latents_2mm/` for Tier-1 / quick iteration. |
| R8 | Submission docker (Phase D) fails VLM3D-Dockers `test.sh` at the wire | Low | Catastrophic | **Stub submission docker required on Phase A Day 5** (P3 enforcement) using GenerateCT outputs; ensures full contract works end-to-end before the real model is final. Phase D = graduation of the already-working stub. |
| R9 | Diagnostic suite assumed UNet-cross-attn but GenerateCT is CT-ViT + MaskGIT + SR diffusion (no UNet cross-attn) | Med | Med | `docs/diagnostic_baseline_compatibility.md` mandated on Phase A Day 5 (NEVER-cut) maps each diagnostic × baseline; cross-attn + token-region marked "Report2CT only" or "partial-via-self-attn-analog". This compat doc is the structural cover. Cross-attn GenerateCT-smoke test on Day 5 catches architectural mismatch early **in code** (slip-eligible to Phase B Day 1 if Day 5 overruns; compat doc still lands on Day 5). |
| R10 | Phase B→C handoff is too fast (zero buffer between 6/30 and 7/1) | Med | High | 6/27 v1 design draft → 6/28–6/30 Critic review at `/workspace/.omc/reviews/our_model_v1_design_critic.md` → 7/1 conditional kickoff with ≥2 diagnostic→architecture mappings verified. P4 enforcement. |
| R11 | Diagnostic shortlist overload at 6/30 | Med | Med | 6/14 shortlist gate (`/workspace/.omc/plans/diagnostic_shortlist.md`); counterfactual drops first if B.2 is behind. |

---

## 6. Verification Steps

### Phase A verification
```bash
# repo structure
ls /workspace/deprecated/triplane_src/models/triplane_ae.py
ls /workspace/deprecated/triplane_configs/trial_toy.yaml
ls /workspace/deprecated/triplane_runs/trial_toy
ls /workspace/deprecated/triplane_tests/test_tier0_overfit.py
ls /workspace/deprecated/reference_trivae/trivae2.py

# Hydra compose (resolved)
python /workspace/src/train.py --cfg job --resolve   # must print fully resolved config

# submodules pinned
git -C /workspace submodule status                   # 3 entries
test -f /workspace/.omc/plans/submodule_pins.md
test -f /workspace/third_party/generatect/README.md
test -f /workspace/third_party/report2ct/README.md
test -f /workspace/third_party/vlm3d_dockers/README.md

# requirements pinned
test "$(grep -E '^(pytorch-lightning|hydra-core|hydra-colorlog|rich|transformers|huggingface_hub)==' /workspace/requirements.txt | wc -l)" -ge 6

# Paper read + impl spec
test -f /workspace/.omc/plans/report2ct_impl_spec.md

# MAISI frozen-load test
CUDA_VISIBLE_DEVICES=0 pytest /workspace/tests/test_maisi_frozen_load.py -q

# Day-4 1-step correctness gate
test -f /workspace/runs/phase_a_day4/report2ct_correctness.log

# Diagnostic compat doc
test -f /workspace/docs/diagnostic_baseline_compatibility.md

# Cross-attn GenerateCT smoke
CUDA_VISIBLE_DEVICES=0 pytest /workspace/tests/test_diagnostic_cross_attn_generatect_smoke.py -q

# GenerateCT adapter smoke
CUDA_VISIBLE_DEVICES=0 pytest /workspace/tests/test_generatect_adapter.py -q

# Lightning fit smoke
CUDA_VISIBLE_DEVICES=0 pytest /workspace/tests/test_lightning_fit_smoke.py -q

# EDA artifacts (deflated to ≥4)
test "$(find /workspace/figs/eda -name '*.png' | wc -l)" -ge 4

# Batch inference
test "$(find /workspace/data/baselines/generatect/valid_50 -name volume.npy | wc -l)" -ge 50

# Stub submission docker (P3 enforcement)
bash /workspace/submission/test.sh
test -f /workspace/submission/output/metrics.json

# Envelope locked
test -f /workspace/.omc/plans/report2ct_envelope.md

# Overall test suite green
cd /workspace && pytest tests/ -q
```

### Phase B verification
```bash
# 6/1 compute gate
test -f /workspace/runs/phase_b_gate/report2ct_1epoch_100samples.log
test -f /workspace/.omc/plans/phase_b_budget.md

# 6/14 shortlist gate
test -f /workspace/.omc/plans/diagnostic_shortlist.md

# Report2CT module
CUDA_VISIBLE_DEVICES=0 pytest /workspace/tests/test_report2ct_module.py -q

# Report2CT trained
test -f /workspace/data/checkpoints/report2ct/our_repro/best.ckpt
python -c "import json; j=json.load(open('/workspace/results/report2ct_sanity.json')); assert 'fvd' in j and 'clip_score' in j and 'fid_2p5d' in j"

# Diagnostics (per compat table)
CUDA_VISIBLE_DEVICES=0 pytest /workspace/tests/test_diagnostic_cross_attn.py /workspace/tests/test_diagnostic_retrieval.py /workspace/tests/test_diagnostic_counterfactual.py /workspace/tests/test_diagnostic_token_region.py -q
test -f /workspace/results/diagnostic/retrieval_generatect.json
test -f /workspace/results/diagnostic/retrieval_report2ct_our_repro.json

# VLM3D-Dockers eval
CUDA_VISIBLE_DEVICES=0 python -m src.eval.vlm3d_runner --baseline=generatect --split=valid
CUDA_VISIBLE_DEVICES=0 python -m src.eval.vlm3d_runner --baseline=report2ct_our_repro --split=valid
test -f /workspace/results/vlm3d/generatect/metrics.json
test -f /workspace/results/vlm3d/report2ct_our_repro/metrics.json

# v1 design + bridge gate
test -f /workspace/.omc/plans/our_model_v1_design.md
test -f /workspace/.omc/reviews/our_model_v1_design_critic.md
```

### Phase C verification
```bash
# Tier-0 overfit
CUDA_VISIBLE_DEVICES=0 pytest /workspace/tests/test_ours_v1_module.py::test_overfit_one_batch -q

# Trained + diagnostics
test -f /workspace/data/checkpoints/ours/v1/best.ckpt
ls /workspace/figs/diagnostic/cross_attn/ours_v1/
test -f /workspace/results/diagnostic/token_region_ours_v1.csv

# Ablations
test "$(find /workspace/configs/experiment -name 'ours_v1_ablation_*.yaml' | wc -l)" -ge 3
test -f /workspace/results/ablations/ours_v1.csv

# Win-condition compare
CUDA_VISIBLE_DEVICES=0 python -m src.eval.vlm3d_runner --baseline=ours_v1 --split=valid
python -m src.eval.compare_vlm3d --a=ours_v1 --b=report2ct_our_repro  # must print PASS, OR
# downgrade branch
test -f /workspace/.omc/plans/envelope_miss_addendum.md   # only if envelope missed
```

### Phase D verification
```bash
# Submission docker (production)
docker build -t vlm3d-ours /workspace/submission/
bash /workspace/submission/test.sh
test -f /workspace/submission/output/metrics.json

# Writeup + receipt
test -f /workspace/submission/writeup.md
test -f /workspace/.omc/plans/submission_receipt.md
```

---

## 7. Dependencies on External Repos

### Git submodules (Phase A Day 1, SHA-pinned)
| Submodule path | Upstream | Purpose | Pin strategy |
|----------------|----------|---------|--------------|
| `third_party/report2ct` | `https://github.com/sinaamirrajab/report2ct` | Reference code/paper artifacts for reimpl. Code itself NOT executed; reading only. | `git -C third_party/report2ct checkout <SHA>`; SHA recorded in `/workspace/.omc/plans/submodule_pins.md`. |
| `third_party/generatect` | `https://github.com/ibrahimethemhamamci/GenerateCT` | Pretrained inference for Phase A baseline. Wrapped by `src/baselines/generatect_adapter.py`; no in-tree edits. | `git -C third_party/generatect checkout <SHA>`; SHA recorded. |
| `third_party/vlm3d_dockers` | `https://github.com/forithmus/VLM3D-Dockers` | Evaluation containers (FVD, CLIPScore, 2.5D-FID) + submission contract reference. | `git -C third_party/vlm3d_dockers checkout <SHA>`; SHA recorded; documented in `/workspace/submission/Dockerfile`. |

### External pretrained checkpoints (manual download, `*.pt` is gitignored per `/workspace/.gitignore`)
- `/workspace/data/checkpoints/generatect/ctvit.pt` ← `https://huggingface.co/generatect/GenerateCT/resolve/main/pretrained_models/ctvit.pt`
- `/workspace/data/checkpoints/generatect/transformer.pt` ← `https://huggingface.co/generatect/GenerateCT/resolve/main/pretrained_models/transformer.pt`
- `/workspace/data/checkpoints/generatect/superres.pt` ← `https://huggingface.co/generatect/GenerateCT/resolve/main/pretrained_models/superres.pt`
- `/workspace/maisi_bundle/models/autoencoder.pt` already present (verified Day 2 via `test_maisi_frozen_load.py`); **no `discriminator.pt` exists** — confirmed by `ls /workspace/maisi_bundle/models/`; our pipeline only consumes the VAE encoder/decoder.
- `/workspace/data/checkpoints/ct_clip/` ← request from `huggingface.co/ibrahimethemhamamci/CT-CLIP`; pending verification per R5 by 5/27 EOD.
- TotalSegmentator / VISTA-3D masks: already present under `/workspace/datasets/datasets/CT-RATE/dataset/{ts_seg,anatomy_segmentation_labels}/`.

### Python deps additions (pinned Phase A Day 1 in `/workspace/requirements.txt`)
Phase A: `pytorch-lightning`, `hydra-core`, `hydra-colorlog`, `rich`, `transformers`, `huggingface_hub` — all pinned with explicit versions. Phase B may add (also pinned at addition time): `open-clip-torch` (or CT-CLIP loader), `torchmetrics-fid`, `scipy>=1.11`. `/workspace/Dockerfile` (Phase D) reuses the same pinned `requirements.txt`.

---

## 8. Open Questions (deferred to user / future rounds)

Written into `/workspace/.omc/plans/open-questions.md` per planner protocol:
1. CT-CLIP weight access (R5) — confirm by 5/27 EOD.
2. Phase B envelope — **RESOLVED in iter-3**: ±15% 2.5D-FID anchored to Figure 6 FID Average 3-TE-cfg5 (≈4.04); ±10% CLIPScore-T2I anchored to Figure 5 3-TE-cfg5 as primary (CLIPScore-I2I secondary informational); ±25% FVD self-anchored to our 6/1 paper-spec measurement (FVD not reported in Report2CT paper, hence widened envelope). Locked end-of-Phase-A.
3. Which of Report2CT's two referenced text encoders (BiomedCLIP vs Bio-ClinicalBERT) is the *primary* — resolved from paper read by 5/27 EOD (Day 2), not Day 5.
4. Docker-in-docker capability inside the dev container (R4) — tested 5/30 via stub submission docker.
5. Whether `our_model_v1` should be one architecture or branched against the top-2 diagnostic findings — decided end of Phase B at 6/27 design draft, signed off 6/28–6/30 by Critic.
6. MAISI VAE-GAN discriminator-weights status — confirmed Day 2: `/workspace/maisi_bundle/models/` ships `autoencoder.pt` only; no `discriminator.pt`. Our pipeline does not need it for frozen-VAE encode/decode. Question marked **RESOLVED** at Day 2; if Report2CT paper specifies discriminator-conditioned training (highly unlikely), escalate to user.
7. `reference/configs/`, `reference/scripts/`, and remaining `reference/models/` files: confirmed by Day 1 EOD whether any are imported by new code; if not, move to `/workspace/deprecated/reference_misc/`.
8. Option C activation (GenerateCT-only baselines) — requires explicit user approval if R2 compute gate forces it (>6 wk Report2CT train forecast).
9. **2.5D-FID definition match (NEW iter-3)**: Verify that VLM3D-Dockers' `2.5-D FID` metric matches Report2CT paper's FID-with-2.5D-feature-extraction definition (paper uses RadImageNet ResNet-50 across XY/XZ/YZ planes averaged, per Figure 6). If definitions differ (e.g., different backbone, different plane-averaging scheme), document the difference in `report2ct_envelope.md` and either: (a) re-run our `report2ct_our_repro` through both definitions if cheap, or (b) note the methodological discrepancy explicitly in the writeup. Owner: Day 5 envelope-lock step; due 5/31 EOD.

---

## Plan Summary

**Plan saved to:** `/workspace/.omc/plans/vlm3d-pivot-plan.md`

**Scope**: 4 phases over ~12 weeks; ~40 acceptance checkboxes across phases; brownfield repo restructure + 1 baseline reimpl + our model + submission docker (stub at Phase A, production at Phase D).
**Estimated complexity**: HIGH (deadline-bound, multi-baseline reimpl, hard docker deliverable, P3/P4 enforcement gates).

**Key deliverables**:
1. lightning-hydra restructured repo with triplane code retired to `/workspace/deprecated/` (incl. `reference/models/trivae*`).
2. GenerateCT inference adapter + 4-diagnostic suite (applied per architectural feasibility) + VLM3D-Dockers eval runner.
3. Report2CT paper-based reimpl trained on CT-RATE with sanity check against locked metric-specific envelope (±15% 2.5D-FID paper-anchored to Fig 6 3-TE-cfg5, ±10% CLIPScore-T2I paper-anchored to Fig 5 3-TE-cfg5 as primary, ±25% FVD self-anchored — FVD not reported in paper); no silent passes.
4. `ours_v1` architecture designed against diagnostic findings, Critic-reviewed in 6/28–6/30 bridge gate, ablated.
5. VLM3D-Dockers-compliant submission docker — **stub at Phase A Day 5**, production at 2026-08-20.

---

## Changelog (Iteration 2)

### Blocking items (Critic REJECT)
- **#1 FIXED** — Replaced `/workspace/paper_pdf/report2ct_vlm3d2025.pdf` with actual file `/workspace/paper_pdf/Report2CT.pdf` (verified via `ls /workspace/paper_pdf/`).
- **#2 FIXED** — Stub submission docker now Phase A Day 5 deliverable with explicit acceptance criterion (`bash /workspace/submission/test.sh` exits 0 on 5-sample fixture); removed slip-to-Phase-B-Day-1 allowance from cut list; stub-docker added to "NEVER cut" list.
- **#3 FIXED** — Phase B→C bridge gate added: v1 design draft 6/27, Critic review 6/28–6/30 at `/workspace/.omc/reviews/our_model_v1_design_critic.md`, ≥2 diagnostic→architecture mappings verified, Phase C kickoff 7/1 conditional. R10 added.
- **#4 FIXED** — Win condition now has explicit branch: envelope miss downgrades claim to "competitive with our reimpl", requires user approval + writeup footnote + `envelope_miss_addendum.md`; explicitly **no silent pass**.
- **#5 FIXED** — `test_maisi_frozen_load.py` added to Phase A acceptance criteria list and Phase A verification block; R6 mitigation now matches acceptance criteria.
- **#6 FIXED** — `docs/diagnostic_baseline_compatibility.md` added as Phase A Day 5 deliverable (4×2 table); Phase B B.3 explicitly applies diagnostics per compat table, not blindly to both baselines. R9 added.
- **#7 FIXED** — `paper_pdf/Report2CT.pdf` read promoted to Day 1 afternoon; `report2ct_impl_spec.md` finalized Day 2 EOD; Day 5 paper-read entry removed.
- **#8 FIXED** — Day 4 1-step correctness gate added (`scripts/phase_a_day4_report2ct_correctness.py`, monotone-decreasing loss over 10 steps on fixed batch); precedes 6/1 wall-clock gate; B.1 now explicitly uses paper-spec skeleton not stub.
- **#9 FIXED** — Envelope pinned to ±15% FVD, ±10% CLIPScore, ±15% 2.5D-FID with paper page/table citations required in `report2ct_envelope.md`; Day 3 proposal → Day 5 lock.

### Non-blocking items
- **#10 APPLIED** — 6/14 diagnostic shortlist gate added (`/workspace/.omc/plans/diagnostic_shortlist.md`); counterfactual drops first. R11 added.
- **#11 APPLIED** — EDA acceptance criterion deflated from ≥6 PNGs to ≥4 PNGs; KS-drift + co-occurrence + report-length moved to Phase B nice-to-have.
- **#12 APPLIED** — P4 amended with Report2CT envelope clause (no 6th principle added).
- **#13 APPLIED** — Phase A header renamed "(5 working days; 5/30–5/31 weekend bundled as Day 5)" with explicit weekend stance.
- **#14 APPLIED** — Submodule SHA pinning added to Day 1 commands + `/workspace/.omc/plans/submodule_pins.md`.
- **#15 APPLIED** — 6 packages pinned in `/workspace/requirements.txt` on Day 1; acceptance criterion added.
- **#16 APPLIED** — `python /workspace/src/train.py --cfg job` → `--cfg job --resolve` in all locations.
- **#17 APPLIED** — Sample-count assertions changed from `ls | wc -l` to `find -name volume.npy | wc -l` (and analogous corrections elsewhere).
- **#18 APPLIED** — Phase A end-of-phase Lightning-wiring assertion added (`tests/test_lightning_fit_smoke.py`).
- **#19 APPLIED** — `reference/models/trivae*.py` explicitly moved to `/workspace/deprecated/reference_trivae/` Day 1; other `reference/` subtrees verified Day 1 EOD (Open Question #7).
- **#20 APPLIED** — MAISI VAE-GAN discriminator question resolved: `/workspace/maisi_bundle/models/` ships only `autoencoder.pt` (verified via `ls`); test docstring + Open Question #6 record. Pipeline does not need discriminator weights.
- **#21 APPLIED** — Option C (GenerateCT-only) added to RALPLAN-DR with invalidation rationale; activatable as R2 contingency with user approval.

### Architect's improvement suggestions (adjudicated)
- **A#1 — covered by blocking #7** (paper read front-loaded).
- **A#2 — APPLIED (modified)** — Cross-attn GenerateCT-smoke test on Phase A Day 5 (`test_diagnostic_cross_attn_generatect_smoke.py`).
- **A#3 — covered by blocking #8** (Day 4 1-step + 6/1 wall-clock split).
- **A#4 — covered by blocking #2** (stub docker Day 5).
- **A#5 — APPLIED (modified)** — Day 1-2 paper read → Day 3 envelope proposal → Day 5 lock.
- **A#6 — covered by non-blocking #10** (6/14 shortlist gate).
- **A#7 — APPLIED (modified)** — P4 amended with envelope clause rather than adding P6.
- **A#8 — covered by non-blocking #11** (EDA deflated to 4 PNGs).
- **A#9 — covered by blocking #9** (±15%/±10%/±15% pinned).
- **A#10 — covered by blocking #3** (B→C bridge gate).

---

## Changelog (Iteration 3)

### Blocking items (Critic REJECT — narrow scope)

- **#9-extension FIXED — Report2CT paper does NOT report FVD.** Critic verified by direct cover-to-cover read of `/workspace/paper_pdf/Report2CT.pdf` (14 pages). Paper reports FID with 2.5D feature extraction (Figure 6 FID Average across XY/XZ/YZ planes) and CLIPScore-T2I + CLIPScore-I2I (Figure 5), with multiple configurations (1-TE w/o cfg = 3.57, 3-TE w/o cfg = 3.79, 1-TE cfg5 = 3.75, **3-TE cfg5 = 4.04**, 3-TE cfg7 = 4.19). FVD is NOT reported anywhere; VLM3D-Dockers added FVD on top of paper. Original envelope criterion (`paper page/table citations for each of three metrics`) was structurally unsatisfiable for FVD.
  - **Envelope reworded with metric-specific anchors**: 2.5D-FID = ±15% paper-anchored to Fig 6 3-TE-cfg5 ≈ 4.04 (headline best config — 3 text encoders + cfg scale 5); CLIPScore-T2I = ±10% paper-anchored to Fig 5 3-TE-cfg5 as PRIMARY; CLIPScore-I2I = SECONDARY informational only; FVD = **±25% self-anchored** to our 6/1 paper-spec measurement, with explicit "FVD not reported in Report2CT paper" disclaimer required in `report2ct_envelope.md`.
  - **Win condition reordered**: metric priority for headline claim is 2.5D-FID > CLIPScore-T2I > FVD (since FVD has weaker anchor). A 2/3 win that excludes 2.5D-FID requires writeup to note the weaker anchor situation. Envelope-miss escalation triggers only on a paper-anchored metric miss (2.5D-FID or CLIPScore-T2I); FVD-only miss is recorded but not auto-escalated.
  - **R2 mitigation rewritten** to include the FVD-unanchored fact and the metric-specific envelope.
  - **ADR consequences updated** to use metric-specific envelope language.
  - **Open Question #2 marked RESOLVED** with the 3-TE-cfg5 + T2I-primary + FVD-widened resolution.
  - **NEW Open Question #9 added**: verify VLM3D-Dockers `2.5-D FID` metric definition matches Report2CT paper's FID-with-2.5D-feature-extraction (RadImageNet ResNet-50, three planes averaged) or document the difference.

- **#2 FIXED — Phase A cut order rebalanced (cross-attn smoke demoted).** Previous cut order had only 3 slack items vs 10 NEVER-cut → functionally empty slack. Demoted exactly one item:
  - **Cross-attn GenerateCT smoke test (`tests/test_diagnostic_cross_attn_generatect_smoke.py`) moved from "NEVER cut" to "may slip to Phase B Day 1 (6/1) if time permits."** Rationale: it is an early-warning architectural-compatibility probe, not a Phase B gate; `docs/diagnostic_baseline_compatibility.md` (still NEVER-cut) is the structural cover for the same architectural insight.
  - **Resulting balance: 9 items NEVER-cut, 4 items cuttable** (EDA secondary cells, KS-drift, cross-attn smoke, envelope citation completeness for non-FVD metrics) → real slack restored.
  - R1 mitigation updated to reflect new 9/4 split; R9 mitigation updated to mark smoke test as slip-eligible while compat doc remains Day 5 NEVER-cut.

### Text-cleanup edits made

- Disambiguated CLIPScore throughout: paper reports both **T2I and I2I**; plan now uses **CLIPScore-T2I as the primary anchor** for envelope and win condition; CLIPScore-I2I recorded as secondary informational.
- Pinned envelope anchor to **3-TE-cfg5** (Report2CT's headline best configuration: 3 text encoders + classifier-free-guidance scale 5, FID Average ≈ 4.04 per Figure 6) — propagated across acceptance criterion, Day 3 proposal, Day 5 lock, Phase B sanity-check criterion, ADR consequences, R2 mitigation, and win condition.
- Removed legacy "±15% FVD / ±10% CLIPScore / ±15% 2.5D-FID" symmetric envelope phrasing in all locations where it appeared; replaced with metric-specific language.
- Renamed metric set in win condition from `{FVD, CLIPScore, 2.5D-FID}` to `{2.5D-FID, CLIPScore-T2I, FVD}` (priority order, headline-anchored metric first).
