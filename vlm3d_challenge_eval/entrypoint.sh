#!/bin/sh
# ctgen thin-image entrypoint.
#
# `exec` keeps process.py as PID 1, which is what lets Vertex flush the gcsfuse-backed /output
# on shutdown (README §6) and delivers SIGTERM straight to our handler on preemption/timeout.
set -eu

# The trampoline guarantees the /output SYMLINK, not the leaf directory (README §9).
mkdir -p /output

exec python /opt/app/process.py
