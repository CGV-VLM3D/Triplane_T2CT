#!/usr/bin/env bash
# ⚠️ DEPRECATED (2026-06-20) — DO NOT USE TO REGENERATE ids_train.txt / ids_valid.txt.
# The ids produced here predate the no_chest (brain) correction and the valid_v2
# migration, so they reintroduce 47 train + 20 valid brain volumes and the old
# deprecated 1000-scan valid split. Canonical id sources now:
#   train : ctrate_toy_v2/train/ids.json   (5000, no_chest-corrected)
#   valid : ctrate_toy_v2/valid_v2/ids.json (1304 one-per-patient, corrected)
# -> exported as report2ct_work_dir/ids_train_v2.txt / ids_valid_v2.txt, merged by
#    scripts/build_report2ct_datalist.py into datalist_v2.json. See plan
#    ticklish-snuggling-gem + data/no_chest_quarantine/no_chest_ids.json.
#
# Bootstrap Report2CT working directory under /workspace/data/report2ct_work_dir/
#
# Train IDs  : 5000 volumes from /workspace/datasets/datasets/latents/train/
#              (CT-RATE train split, pre-cached latents)
# Valid IDs  : 1000 volumes sampled from CT-RATE official validation split
#              (validation_metadata.csv, 3039 total) — separate patients from train
set -euo pipefail

WORKDIR=/workspace/data/report2ct_work_dir
LATENTS_DIR=/workspace/datasets/datasets/latents
CTRATE_META=/workspace/datasets/datasets/CT-RATE/dataset/metadata/validation_metadata.csv
VALID_SAMPLE=1000
SEED=42

mkdir -p "$WORKDIR"/{text_embeddings,image_embeddings}

# Train IDs from cached latents
ls "$LATENTS_DIR/train" > "$WORKDIR/ids_train.txt"

# Valid IDs: random sample from CT-RATE official validation set
python - <<PY
import pandas as pd, pathlib, sys
meta = pd.read_csv("$CTRATE_META")
ids = meta["VolumeName"].str.replace(".nii.gz", "", regex=False)
sampled = ids.sample(n=$VALID_SAMPLE, random_state=$SEED).sort_values()
pathlib.Path("$WORKDIR/ids_valid.txt").write_text("\n".join(sampled) + "\n")
print(f"  sampled {len(sampled)} valid IDs from CT-RATE validation split", flush=True)
PY

echo "[setup_report2ct_workdir] created:"
echo "  $WORKDIR/{text_embeddings,image_embeddings}/"
echo "  $WORKDIR/ids_train.txt ($(wc -l < "$WORKDIR/ids_train.txt") IDs)"
echo "  $WORKDIR/ids_valid.txt ($(wc -l < "$WORKDIR/ids_valid.txt") IDs)"
