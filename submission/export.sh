#!/usr/bin/env bash
# Save the built submission image as a .tar.gz for upload.
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-vlm3d-submission-stub}"
OUT="${OUT:-submission.tar.gz}"

docker save "$IMAGE_NAME" | gzip -c > "$OUT"
echo "Saved $IMAGE_NAME -> $OUT"
ls -lh "$OUT"
