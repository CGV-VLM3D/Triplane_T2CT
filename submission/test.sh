#!/usr/bin/env bash
# Docker-based submission test (Phase D production path).
# Builds the image, runs against the 5-prompt fixture, copies outputs to
# `exported_images/`, then cleans up. Requires Docker daemon access.
#
# Phase A acceptance uses test_local.sh instead (no docker required).

set -euo pipefail

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
IMAGE_NAME="${IMAGE_NAME:-vlm3d-submission-stub}"

# Build (idempotent).
docker build -t "$IMAGE_NAME" "$SCRIPTPATH"

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 2>/dev/null | md5sum | cut -d' ' -f1)
EXPORT_DIR="$SCRIPTPATH/exported_images"
mkdir -p "$EXPORT_DIR"
rm -f "$EXPORT_DIR"/*

docker volume create "submission-out-$VOLUME_SUFFIX" >/dev/null

docker run --rm \
    --memory="8g" \
    --network="bridge" \
    --cap-drop="ALL" \
    --security-opt="no-new-privileges" \
    --shm-size="128m" \
    --pids-limit="256" \
    -v "$SCRIPTPATH/test":/input \
    -v "submission-out-$VOLUME_SUFFIX":/output \
    "$IMAGE_NAME"

docker run --rm \
    -v "submission-out-$VOLUME_SUFFIX":/data \
    -v "$EXPORT_DIR":/dst \
    alpine:latest \
    sh -c 'cp -a /data/. /dst/'

docker volume rm "submission-out-$VOLUME_SUFFIX" >/dev/null

ls -lh "$EXPORT_DIR"
test -s "$EXPORT_DIR/predictions.zip" || { echo "predictions.zip missing"; exit 1; }
echo "✅ submission docker contract OK."
