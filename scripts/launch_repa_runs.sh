#!/usr/bin/env bash
# Wait for the SPECTRE teacher precompute to finish, then start the two REPA training arms.
#
#   Track A (GPU 3) — from-scratch 100-epoch pilot, U-REPA regime (hard cosine + relational).
#                     Compared against the baseline run outputs/report2ct_wan/2026-07-16_3 at
#                     matched epochs. ~12 h.
#   Track B (GPU 2) — finetune the trained baseline relational-only (VideoREPA regime), then the
#                     SAME number of steps with REPA off as the control. Without that control the
#                     finetune is not an A/B: it has simply seen more steps than the baseline.
#                     ~1.5 h each, run sequentially so they do not share a card.
#
# Usage:  nohup bash scripts/launch_repa_runs.sh > logs/repa_launcher_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -uo pipefail
cd /workspace

TEACHER_DIR=data/report2ct_wan/spectre_ssl_16
DATALIST=data/report2ct_wan/datalist_wan_2560.json
BASELINE_CKPT=outputs/report2ct_wan/2026-07-16_3/checkpoints/epoch_299.ckpt
FINETUNE_EPOCHS=10   # 625 steps/epoch -> 6250 steps, VideoREPA finetunes for 4k
TS() { date +%Y%m%d_%H%M%S; }

# Readiness is checked against the DATALIST, which is what the datamodule actually reads — not
# against data/ctrate_toy_v2/*/ids.json. The two have drifted apart before (a stale wan datalist
# disagreed with the canonical valid_v2 list on 603 of 1304 scans, [[proxy-test-is-valid-v2]]),
# and counting the wrong list let training start against 603 missing teacher files.
missing_count() {
  python - "$TEACHER_DIR" "$DATALIST" <<'PY'
import json, pathlib, sys
teacher, datalist = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
entries = json.loads(datalist.read_text())
ids = {
    pathlib.Path(e["image"]).name[: -len("_emb.nii.gz")]
    for split in ("training", "validation")
    for e in entries[split]
}
print(sum(not (teacher / f"{i}.npy").is_file() for i in ids), len(ids))
PY
}

echo "[launcher] waiting for every $DATALIST entry to have a grid in $TEACHER_DIR"
while :; do
  read -r missing total <<<"$(missing_count)"
  echo "[launcher] $(date +%H:%M:%S) $((total - missing))/$total ready"
  [ "$missing" -eq 0 ] && break
  sleep 300
done
echo "[launcher] precompute complete — starting training"

# --- Track A: from-scratch pilot ---------------------------------------------
CUDA_VISIBLE_DEVICES=3 nohup python src/train.py experiment=report2ct_wan_repa \
    trainer.max_epochs=100 \
    callbacks.model_checkpoint.every_n_epochs=10 \
  > "logs/report2ct_wan_repa_trackA_$(TS).log" 2>&1 &
echo "[launcher] Track A started on GPU 3 (pid $!)"

# --- Track B: relational-only finetune, then its REPA-off control -------------
# init_from_ckpt (not Lightning's ckpt_path): weights only, fresh optimizer. Resuming would also
# restore a PolynomialLR that has already decayed to ~0 after 300 epochs.
CUDA_VISIBLE_DEVICES=2 nohup bash -c "
  python src/train.py experiment=report2ct_wan_repa \
      task_name=report2ct_wan_repa_ftB \
      model.init_from_ckpt=$BASELINE_CKPT \
      model.lr=2e-6 model.warmup_steps=0 \
      model.repa.hard_weight=0 model.repa.rel_loss=videorepa_l1_st \
      trainer.max_epochs=$FINETUNE_EPOCHS callbacks.model_checkpoint.every_n_epochs=2
  python src/train.py experiment=report2ct_wan_repa \
      task_name=report2ct_wan_ft_control \
      model.init_from_ckpt=$BASELINE_CKPT \
      model.lr=2e-6 model.warmup_steps=0 \
      model.repa_weight=0 \
      trainer.max_epochs=$FINETUNE_EPOCHS callbacks.model_checkpoint.every_n_epochs=2
" > "logs/report2ct_wan_repa_trackB_$(TS).log" 2>&1 &
echo "[launcher] Track B + control started on GPU 2 (pid $!)"
wait
echo "[launcher] all runs exited"
