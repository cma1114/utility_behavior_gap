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
  manifest_list="outputs/api/runs/high_utility_r0_bridge_manifests__${actor}.tsv"
  if [ ! -f "$manifest_list" ]; then
    echo "missing manifest list: $manifest_list" >&2
    echo "run: bash src/utility_behavior_gap/scripts/run_high_utility_r0_bridge_actor.sh $actor all" >&2
    exit 1
  fi
  while IFS=$'\t' read -r _ manifest; do
    run_dir="${manifest%/generation_jobs.jsonl}"
    args+=(--run-dir "$run_dir")
  done < "$manifest_list"
done

.venv/bin/python -m utility_behavior_gap.scripts.analyze_highlow_r0_bridge_judging "${args[@]}"
.venv/bin/python -m utility_behavior_gap.scripts.plot_highlow_r0_bridge --from-high-utility-manifests --paper-ready
