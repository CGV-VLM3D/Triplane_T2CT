#!/usr/bin/env bash
set -euo pipefail

cd /workspace 2>/dev/null || exit 0

command -v git >/dev/null 2>&1 || exit 0
[[ -d .git ]] || exit 0

branch=$(git branch --show-current 2>/dev/null || true)
[[ -z "$branch" ]] && exit 0
git config --get "branch.${branch}.remote" >/dev/null 2>&1 || exit 0

timeout 5s git fetch origin --quiet 2>/dev/null || exit 0

local_sha=$(git rev-parse HEAD 2>/dev/null || true)
remote_sha=$(git rev-parse "origin/${branch}" 2>/dev/null || true)
[[ -z "$remote_sha" || -z "$local_sha" ]] && exit 0

if [[ "$local_sha" != "$remote_sha" ]]; then
  behind=$(git rev-list --count "HEAD..origin/${branch}" 2>/dev/null || echo "?")
  msg="origin/${branch} has ${behind} new commit(s). Run 'git pull' before working."
  printf '{"systemMessage": "%s"}\n' "$msg"
fi
exit 0
