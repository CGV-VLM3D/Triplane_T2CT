#!/usr/bin/env bash
# Build weights.zip for the ctgen submission (mounted at /weights by the platform).
#
# Layout inside the zip (files at the ROOT, no parent dir — README §7):
#   report2ct_wan_ep299.ckpt   the trained UNet (optimizer state stripped)
#   hf_cache/hub/models--*/    the 3 report2ct text encoders + the Wan2.1 VAE
#
# process.py reads the ckpt from $FORITHMUS_WEIGHTS and the Dockerfile sets
# HF_HOME=/weights/hf_cache + HF_HUB_OFFLINE=1, so nothing has to be symlinked at runtime and no
# repo code changes for the offline environment.
#
# Two size reductions, both verified not to change what loads:
#   * the Lightning .ckpt is 2.8 GB but _load_wan_checkpoint only reads ckpt["state_dict"];
#     re-saving just that drops the optimizer/scheduler state.
#   * an HF cache stores each file once in blobs/ with snapshots/ symlinking to it. zip
#     dereferences symlinks by default, so zipping both would store every weight twice —
#     we stage snapshots/ (dereferenced) and skip blobs/.
set -euo pipefail

SRC_CKPT=${SRC_CKPT:-/workspace/outputs/report2ct_wan/2026-07-16_3/checkpoints/epoch_299.ckpt}
SRC_HF=${SRC_HF:-/workspace/data/checkpoints/hf_cache/hub}
DEST=${DEST:-/workspace/data/vlm3d_submission}          # md0, never /tmp (CLAUDE.md)
STAGE="$DEST/weights_stage"
ZIP="$DEST/weights.zip"

REPOS=(
    models--abhinand--MedEmbed-large-v0.1
    models--medicalai--ClinicalBERT
    models--microsoft--BiomedVLP-CXR-BERT-specialized
    models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers
)

rm -rf "$STAGE" "$ZIP"
mkdir -p "$STAGE/hf_cache/hub"

echo "[stage] stripping optimizer state from $SRC_CKPT"
python - "$SRC_CKPT" "$STAGE/report2ct_wan_ep299.ckpt" <<'PY'
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
ckpt = torch.load(src, map_location="cpu", weights_only=False)
torch.save({"state_dict": ckpt["state_dict"]}, dst)
print(f"[stage]   kept {len(ckpt['state_dict'])} state_dict tensors")
PY

for r in "${REPOS[@]}"; do
    echo "[stage] $r"
    mkdir -p "$STAGE/hf_cache/hub/$r"
    # -L dereferences the snapshots/ symlinks so blobs/ can be dropped.
    cp -rL "$SRC_HF/$r/refs" "$SRC_HF/$r/snapshots" "$STAGE/hf_cache/hub/$r/"
    [ -d "$SRC_HF/$r/.no_exist" ] && cp -r "$SRC_HF/$r/.no_exist" "$STAGE/hf_cache/hub/$r/"
done

# Python's zipfile instead of the `zip` binary (not installed here); ZIP_STORED == `zip -0`.
echo "[stage] zipping (store-only, files at zip root)"
python - "$STAGE" "$ZIP" <<'PY'
import sys, zipfile
from pathlib import Path
stage, zip_path = Path(sys.argv[1]), sys.argv[2]
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
    for p in sorted(stage.rglob("*")):
        if p.is_file():
            zf.write(p, p.relative_to(stage))
    print(f"[stage]   {len(zf.namelist())} files")
PY

echo "[stage] done"
du -sh "$STAGE" "$ZIP"
python -c "
import zipfile,sys
n=zipfile.ZipFile(sys.argv[1]).namelist()
print('[stage] zip roots:', sorted({x.split('/')[0] for x in n}))
" "$ZIP"
