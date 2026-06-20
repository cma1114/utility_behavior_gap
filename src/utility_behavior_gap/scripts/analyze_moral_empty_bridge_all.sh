#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

actors=(
  deepseek-v3.2-or
  gpt-5.4-mini-or
  glm-5.1-or
  kimi-k2.5-or
  mimo-v25-pro-or
  qwen3.5-9b-or
  qwen3.6-plus-or
)

args=()
for actor in "${actors[@]}"; do
  manifest_list="outputs/api/runs/moral_empty_bridge_manifests__${actor}.tsv"
  if [ ! -f "$manifest_list" ]; then
    echo "missing manifest list: $manifest_list" >&2
    exit 1
  fi
  while IFS=$'\t' read -r _ manifest; do
    args+=(--run-dir "${manifest%/generation_jobs.jsonl}")
  done < "$manifest_list"
done

.venv/bin/python -m utility_behavior_gap.scripts.analyze_moral_empty_bridge_judging "${args[@]}"
.venv/bin/python -m utility_behavior_gap.scripts.analyze_moral_baseline_bridges --comparison moral_low_vs_framed_empty
