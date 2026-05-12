#!/usr/bin/env bash
set -euo pipefail

input=$(cat)
file_path=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')
[[ -z "$file_path" ]] && exit 0

reason=""
case "$file_path" in
  */checkpoints/*|checkpoints/*)             reason="checkpoints/ is protected (training artifacts).";;
  */wandb/*|wandb/*)                          reason="wandb/ is protected (experiment logs).";;
  */configs/credentials.yaml|configs/credentials.yaml) reason="configs/credentials.yaml is protected (credentials).";;
esac

if [[ -z "$reason" ]]; then
  case "$(basename -- "$file_path")" in
    .env|.env.*) reason=".env files are protected (secrets).";;
  esac
fi

if [[ -n "$reason" ]]; then
  {
    printf 'Edit blocked by protect-files hook.\n'
    printf '  path:   %s\n' "$file_path"
    printf '  reason: %s\n' "$reason"
  } >&2
  exit 2
fi
exit 0
