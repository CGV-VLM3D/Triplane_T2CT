#!/usr/bin/env bash
# Wan step-sensitivity calibration: 50 vs 100 RFlow steps, SAME n head cases, fixed regime
# (spacing 0.75/0.75/1.3, cfg 5.0, epoch_299), FID+CLIP vs the canonical 3001 GT.
# Answers: does report2ct_wan's step-convergence differ from report2ct (MAISI)?
# generate=main env (RFlow->latents), decode=wan env (AutoencoderKLWan), score=main env.
set -u
cd /workspace

WAN_CKPT=/workspace/outputs/report2ct_wan/2026-07-16_3/checkpoints/epoch_299.ckpt
WANPY=/opt/conda/envs/wan/bin/python
BASE=/workspace/data/vlm3d_eval/ctgen/wan_stepcal
N=${N:-150}
SP="0.75 0.75 1.3"
CFG=5.0
GPU=${GPU:-2}

for S in 50 100; do
  D="$BASE/s${S}"
  echo "===================== steps=$S  generate ($(date)) ====================="
  CUDA_VISIBLE_DEVICES=$GPU python scripts/generate_wan_latents.py \
    --ckpt "$WAN_CKPT" --out "$D" --n "$N" --n-steps "$S" --cfg-scale "$CFG" --spacing $SP
  echo "===================== steps=$S  decode (wan env) ($(date)) ====================="
  CUDA_VISIBLE_DEVICES=$GPU $WANPY scripts/decode_wan_latents.py \
    --latent-dir "$D/latents" --out "$D/predictions" --spacing $SP
  echo "===================== steps=$S  score ($(date)) ====================="
  CUDA_VISIBLE_DEVICES=$GPU python scripts/rescore_predictions.py \
    --pred-dir "$D/predictions" --out "$D" --n "$N"
  echo "----- steps=$S summary -----"; cat "$D"/fid_*/metrics.json 2>/dev/null; echo ""
done
echo "===================== ALL DONE ($(date)) ====================="
