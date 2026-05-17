#!/usr/bin/env bash
# SubagentStop hook: posts a Discord message when the experiment-runner
# subagent finishes (success or failure). The in-script notifier in
# scripts/train.py covers normal start/best/end/exception events; this hook
# is the catch-all for cases where train.py itself never runs (pytest gate
# failure, OOM retries, etc.) or for the agent's own status.
set -euo pipefail

input=$(cat)
agent_type=$(printf '%s' "$input" | jq -r '.agent_type // "experiment-runner"')
ts=$(date '+%Y-%m-%d %H:%M:%S')

# Load .env.local if present (gitignored — holds DISCORD_WEBHOOK_URL).
if [[ -f /workspace/.env.local ]]; then
  set -a
  # shellcheck disable=SC1091
  . /workspace/.env.local
  set +a
fi

title="Claude Code — ${agent_type} subagent finished"
body="${title} at ${ts}"

# Local desktop notification (kept as a fallback).
if command -v notify-send >/dev/null 2>&1; then
  notify-send -u normal "$title" "$body" || true
else
  printf '\a' >&2
  printf '[NOTIFY] %s\n' "$body" >&2
fi

# Discord webhook (best-effort: never block hook exit).
if [[ -n "${DISCORD_WEBHOOK_URL:-}" ]]; then
  payload=$(jq -n \
    --arg title "$title" \
    --arg desc  "Subagent \`${agent_type}\` finished at ${ts}." \
    '{embeds: [{title: $title, description: $desc, color: 3447003}]}')
  curl -fsS -m 5 -H 'Content-Type: application/json' \
       -X POST -d "$payload" "$DISCORD_WEBHOOK_URL" >/dev/null 2>&1 \
    || printf '[notify-training-done] discord post failed\n' >&2
fi

exit 0
