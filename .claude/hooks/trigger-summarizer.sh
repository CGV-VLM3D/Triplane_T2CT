#!/usr/bin/env bash
# Fires on SubagentStop for the result-analyzer.
# Hooks cannot directly invoke another agent, but exit code 2 surfaces
# stderr back to the parent Claude session as a blocking message, which
# makes the parent very likely to invoke research-summarizer next.
set -euo pipefail

input=$(cat)
agent_type=$(printf '%s' "$input" | jq -r '.agent_type // ""')

if [[ "$agent_type" != "result-analyzer" ]]; then
  exit 0
fi

# Best-effort: try to find the most recent exp_name from result-analyzer memory.
memfile="/workspace/.claude/agents/memory/result-analyzer/MEMORY.md"
exp_name="(check result-analyzer output)"
if [[ -f "$memfile" ]]; then
  last_line=$(tail -n 1 "$memfile" 2>/dev/null || true)
  # Format: <date> <exp_name> PSNR=... SSIM=... figs=figs/<exp>/
  candidate=$(printf '%s' "$last_line" | awk '{print $2}')
  if [[ -n "$candidate" ]]; then
    exp_name="$candidate"
  fi
fi

# Send a clear directive to the parent Claude via stderr + exit 2.
cat >&2 <<EOF
[auto-summarizer trigger]
result-analyzer for exp_name="${exp_name}" just finished.
NEXT ACTION: invoke the research-summarizer agent (Agent tool, subagent_type="research-summarizer")
with the exp_name above so it can update /workspace/research_summary/summary.md.
EOF

exit 2
