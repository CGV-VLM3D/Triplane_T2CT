#!/usr/bin/env bash
set -euo pipefail

input=$(cat)
agent_type=$(printf '%s' "$input" | jq -r '.agent_type // "experiment-runner"')
ts=$(date '+%Y-%m-%d %H:%M:%S')

title="Claude Code — $agent_type finished"
body="Training subagent completed at $ts"

if command -v notify-send >/dev/null 2>&1; then
  notify-send -u normal "$title" "$body" || true
else
  printf '\a' >&2
  printf '[NOTIFY] %s — %s\n' "$title" "$body" >&2
fi
exit 0
