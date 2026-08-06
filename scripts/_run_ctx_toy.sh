#!/bin/bash
set -e
export HF_HOME=/workspace/data/checkpoints/hf_cache
export CUDA_VISIBLE_DEVICES=3
cd /workspace

echo "=== [1/2] train (5000) ==="
python scripts/precompute_report2ct_text_embeddings.py \
  --ids-file /workspace/data/report2ct_work_dir/ids_train_v2.txt \
  --reports-csv /workspace/datasets/datasets/CT-RATE/dataset/radiology_text_reports/train_reports.csv \
  --out-dir /workspace/data/report2ct_wan/latents_512x512x253 \
  --save-tokens --device cuda:0

echo "=== [2/2] valid (1304) ==="
python scripts/precompute_report2ct_text_embeddings.py \
  --ids-file /workspace/data/report2ct_work_dir/ids_valid_v2.txt \
  --reports-csv /workspace/datasets/datasets/CT-RATE/dataset/radiology_text_reports/validation_reports.csv \
  --out-dir /workspace/data/report2ct_wan/latents_512x512x253 \
  --save-tokens --device cuda:0

echo "=== DONE ==="
