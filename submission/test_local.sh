#!/usr/bin/env bash
# Phase A acceptance path: runs process.py directly via Python (no docker daemon
# required). Verifies the submission contract end-to-end on the 5-prompt fixture.
#
# Acceptance: exits 0 and writes /tmp/submission_test_out/predictions.zip
# matching the VLM3D-Dockers ctgen schema.

set -euo pipefail

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
OUT="${OUT:-/tmp/submission_test_out}"

rm -rf "$OUT"
mkdir -p "$OUT"

INPUT_DIR="$SCRIPTPATH/test" OUTPUT_DIR="$OUT" python "$SCRIPTPATH/process.py"

echo "---"
ls -lh "$OUT"
echo "---"
# Verify predictions.zip exists and is non-empty.
test -s "$OUT/predictions.zip" || { echo "predictions.zip missing or empty"; exit 1; }
# Verify zip contains the expected number of .mha entries (via Python; busybox/alpine envs lack unzip).
N=$(python -c "import zipfile; z=zipfile.ZipFile('$OUT/predictions.zip'); print(sum(1 for n in z.namelist() if n.endswith('.mha')))")
echo "predictions.zip contains $N .mha files"
test "$N" -eq 5 || { echo "expected 5 .mha entries, got $N"; exit 1; }
echo "submission stub contract OK (local path)."
