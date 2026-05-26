#!/usr/bin/env bash
set -euo pipefail
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
IMAGE_NAME="${IMAGE_NAME:-vlm3d-submission-stub}"
docker build -t "$IMAGE_NAME" "$SCRIPTPATH"
