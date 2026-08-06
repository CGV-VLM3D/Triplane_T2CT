#!/usr/bin/env bash
# Stage-2 sampling-step metric sweep for Report2CT (30 vs 100 vs 50).
# Same 300 valid_v2 cases (seeded, nested), real eval regime (spacing 0.8/0.8/1.5, cfg 5).
# Dedicated isolated GT dir → no collision with the concurrent 3001-GT recompute.
# FID + CLIPScore-T2I only (FVD off). Sequential on one GPU. See step_convergence stage 1.
set -u
cd /workspace

CKPT=/workspace/outputs/report2ct/report2ct_full_2026-06-30/checkpoints/epoch_074.ckpt
GT=/workspace/data/vlm3d_eval/_stepconv_v2_gt300
N=300

for S in 30 100 50; do
  echo "===================== n_steps=$S  ($(date)) ====================="
  CUDA_VISIBLE_DEVICES=${GPU:-3} python scripts/run_eval.py \
    task=ctgen model=report2ct \
    model.ckpt_path="$CKPT" \
    model.n_steps="$S" \
    model.spacing_mm=[0.8,0.8,1.5] model.cfg_scale=5.0 \
    task.n_samples="$N" \
    task.gt_dir="$GT" \
    task.metrics.fvd=false task.metrics.fvd_ctclip=false \
    task.metrics.clip_score=true task.metrics.fid_2p5d=true \
    out_dir=/workspace/outputs/report2ct/stepconv_ep074/steps${S}
  echo "----- steps=$S summary -----"
  cat /workspace/outputs/report2ct/stepconv_ep074/steps${S}/summary.json 2>/dev/null
  echo ""
done
echo "===================== ALL DONE ($(date)) ====================="
