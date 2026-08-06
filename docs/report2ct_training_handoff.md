# Report2CT Training Handoff (Phase B kickoff, 6/1)

Per [[report2ct-training-is-user-owned]]: **assistant prepares training-ready code,
user runs the multi-day training**. This document is the runbook.

---

## TL;DR

```bash
# 0. Make sure 3 text encoders downloaded + cached (one-time, ~3 GB)
python scripts/precompute_report2ct_text_embeddings.py   # Phase B Day 1 deliverable

# 1. Precompute MAISI image embeddings on CT-RATE train_fixed
python scripts/precompute_report2ct_image_embeddings.py  # Phase B Day 1 deliverable

# 2. Train (multi-day; user runs)
bash scripts/run_report2ct_training.sh

# 3. After training, fill sanity metrics (run_eval.py generates + evaluates in one pass)
CUDA_VISIBLE_DEVICES=0 python scripts/run_eval.py \
    task=ctgen model=report2ct model.ckpt_path=<best.ckpt> \
    model.spacing_mm=[0.8,0.8,1.5] model.cfg_scale=5.0   # spacing_mm/cfg_scale REQUIRED (no default)
python scripts/fill_report2ct_sanity.py \
    --metrics /workspace/results/vlm3d/report2ct_our_repro/metrics.json \
    --out /workspace/results/report2ct_sanity.json
```

> The `precompute_*.py` and `fill_report2ct_sanity.py` scripts are Phase B Day 1
> deliverables wrapping `third_party/report2ct/src/maisi/scripts/vlm3d_image_embedding.py`
> and the `vlm3d_inference.ipynb` cell-0 text-encoder loop respectively. This doc
> exists now to lock the interface — implementation lands 6/1.

---

## What lives where

| Artifact | Path | Owner |
|---|---|---|
| Submodule training script | `third_party/report2ct/src/maisi/scripts/diff_model_train_vlm3D_2560_multi_text.py` | upstream (P2, read-only) |
| Submodule model config | `third_party/report2ct/vlm3D_work_dir/config_maisi_2560.json` | upstream |
| Submodule training config | `third_party/report2ct/vlm3D_work_dir/config_maisi_diff_model_vlm3D.json` | upstream |
| Hydra launcher config | `configs/experiment/report2ct_repro.yaml` | **assistant** (Phase B Day 1) |
| Bash launcher | `scripts/run_report2ct_training.sh` | **assistant** (Phase B Day 1) |
| Sanity schema | `results/report2ct_sanity.json` | **user** (after training) |
| Envelope (target) | `results/report2ct_envelope.json` | already locked Phase A Day 5 |
| MAISI VAE weights (frozen) | `third_party/maisi_bundle/models/autoencoder.pt` | upstream |
| MAISI VAE loader util | `src/baselines/maisi.py::load_frozen` | assistant |

---

## Hardware reality

Paper says: 2× NVIDIA H100 NVL (94 GB), ~1 hour/epoch on 20,000 CT-RATE samples,
100 epochs ⇒ ~100 GPU-hours.

Our lab: ≤3× A6000 Blackwell (96 GB). FP16/BF16 throughput is ~30-50% of H100
TFLOPS depending on op, so expect **~1.5-2.5× wall-clock penalty** ⇒
**150-250 GPU-hours** on 2× A6000 (DDP `torchrun --nproc_per_node=2`).
At 24h × 7d = 168 hours/week per GPU pair, 100-epoch training is ~1-1.5 weeks.

**Phase B 6/1 wall-clock gate**: run 1-epoch wall-clock on a 100-sample subset
to refine this estimate before committing to a 100-epoch run.

---

## Disk space estimate

| Artifact | Per sample | × 20,000 train | × 1,000 valid | Total |
|---|---|---|---|---|
| Original NIfTI | ~7 MB | 140 GB | 7 GB | ~150 GB |
| MAISI latent (`_emb.nii.gz`) | ~3 MB | 60 GB | 3 GB | ~63 GB |
| Text embeddings JSON | ~25 KB | 0.5 GB | 25 MB | ~0.5 GB |
| Checkpoints (best + last) | — | — | — | ~3 GB |

Plan for `/workspace/data/report2ct_embeddings/` to need **~65 GB**; checkpoints
under `/workspace/data/checkpoints/report2ct/` need **~3-5 GB**.

---

## Sanity reporting template (`results/report2ct_sanity.json`)

After training, the user runs VLM3D-Dockers eval on the trained model's predictions
and copies the metrics into the sanity schema. The schema mirrors the envelope:

```jsonc
{
  "trained_at": "2026-MM-DD",
  "checkpoint": "/workspace/data/checkpoints/report2ct/our_repro/best.ckpt",
  "training_epochs": 100,
  "gpu_hours_actual": 0.0,
  "metrics": {
    "FID_2p5D_Avg": {"value": null, "envelope_bounds": [3.434, 4.646], "within": null},
    "CLIPScore":    {"value": null, "envelope_bounds": [53.937, 65.923], "within": null},
    "CLIPScore_I2I": {"value": null, "informational": true},
    "FVD_CTNet":    {"value": null, "envelope_anchor": null, "envelope_pct": 25, "informational_anchor": "self-measured 6/1 run"}
  },
  "envelope_status": "within | outside_FID | outside_CLIP | outside_both"
}
```

If `envelope_status != "within"`, the **win condition downgrade branch** activates
(see `results/report2ct_envelope.json::win_condition.envelope_miss_branch`). User
must explicitly approve before Phase D submission proceeds.

---

## Common pitfalls (collected from upstream issues + paper)

1. **CFG dropout p=0.15** during training — submodule sets this; do not override unless ablating.
2. **MAISI VAE must stay frozen** — `src/baselines/maisi.py::load_frozen` enforces this.
   If you set any param `requires_grad=True` by mistake, the latent space drifts and FID
   collapses.
3. **DDP rank-0 only logging** — the submodule's logger setup uses `rank_zero_only`. When
   you launch with `torchrun --nproc_per_node=2`, only rank 0 writes to `runs/`.
4. **batch_size=2 PER GPU** by default. With 2 GPUs total batch is 4. Don't conflate.
5. **Resume from checkpoint** — submodule script supports `--resume_from` arg; use it
   if your job is interrupted, do NOT restart from scratch.

---

## Decision gates (user-driven)

- **After 1-epoch wall-clock (6/1)**: if projected total > 6 weeks, choose between
  (a) subset training, (b) Phase C overlap, (c) Option C (drop Report2CT). Record
  decision in `.omc/plans/phase_b_budget.md`.
- **After half-train sanity (~6/15 estimated)**: if FID or CLIPScore is already
  drifting outside envelope, decide between continuing or restarting with adjusted
  hyperparameters. Record in `.omc/plans/phase_b_sanity_check.md`.
- **After full training**: fill `results/report2ct_sanity.json` + decide envelope
  status. If `outside_*`, escalate to deep-interview branch.
