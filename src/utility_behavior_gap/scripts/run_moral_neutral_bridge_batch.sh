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

max_parallel="${MAX_PARALLEL_ACTORS:-2}"
cycles="${CYCLES:-1}"
cycle_sleep_s="${CYCLE_SLEEP_S:-900}"
export JUDGING_WORKERS="${JUDGING_WORKERS:-1}"
export JUDGING_PASSES="${JUDGING_PASSES:-1}"
export JUDGING_SLEEP_S="${JUDGING_SLEEP_S:-300}"

if ! [[ "$max_parallel" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL_ACTORS must be a positive integer" >&2
  exit 1
fi
if ! [[ "$cycles" =~ ^[1-9][0-9]*$ ]]; then
  echo "CYCLES must be a positive integer" >&2
  exit 1
fi

manifest_is_current() {
  local actor="$1"
  local manifest_list="outputs/api/runs/moral_neutral_bridge_manifests__${actor}.tsv"
  if [ ! -f "$manifest_list" ]; then
    return 1
  fi
  .venv/bin/python - "$manifest_list" <<'PY'
import json
import sys
from pathlib import Path

manifest_list = Path(sys.argv[1])
try:
    lines = [line.strip().split("\t") for line in manifest_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("empty manifest list")
    manifest = Path(lines[0][1])
    with manifest.open(encoding="utf-8") as fh:
        jobs = [json.loads(line) for line in fh if line.strip()]
    if not jobs:
        raise RuntimeError("empty generation_jobs")
    bad = [job.get("pair_uid", "") for job in jobs if "axis" not in job or "axis_definition" not in job]
    if bad:
        raise RuntimeError(f"missing required judging fields in {len(bad)} jobs")
except Exception as exc:
    print(f"stale or invalid manifest: {exc}", file=sys.stderr)
    sys.exit(1)
PY
}

prepare_if_needed() {
  local actor="$1"
  if manifest_is_current "$actor"; then
    echo "manifest current: $actor"
  else
    echo "preparing fresh moral-neutral bridge manifest: $actor"
    .venv/bin/python -m utility_behavior_gap.scripts.prepare_moral_neutral_bridge_judging \
      --actors "$actor"
  fi
}

judge_actor() {
  local actor="$1"
  echo "=== moral bad vs framed neutral: $actor ==="
  bash src/utility_behavior_gap/scripts/run_moral_neutral_bridge_actor.sh "$actor" judge
  bash src/utility_behavior_gap/scripts/run_moral_neutral_bridge_actor.sh "$actor" analyze
  bash src/utility_behavior_gap/scripts/run_moral_neutral_bridge_actor.sh "$actor" status
}

echo "moral-neutral bridge batch"
echo "MAX_PARALLEL_ACTORS=$max_parallel JUDGING_WORKERS=$JUDGING_WORKERS CYCLES=$cycles"

for actor in "${actors[@]}"; do
  prepare_if_needed "$actor"
done

for cycle in $(seq 1 "$cycles"); do
  echo "=== cycle $cycle/$cycles ==="
  pids=()
  labels=()
  failed=0
  for actor in "${actors[@]}"; do
    (
      judge_actor "$actor"
    ) &
    pids+=("$!")
    labels+=("$actor")
    if [ "${#pids[@]}" -ge "$max_parallel" ]; then
      if ! wait "${pids[0]}"; then
        echo "actor failed: ${labels[0]}" >&2
        failed=1
      fi
      pids=("${pids[@]:1}")
      labels=("${labels[@]:1}")
    fi
  done
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "actor failed: ${labels[$idx]}" >&2
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    exit 1
  fi
  if [ "$cycle" != "$cycles" ]; then
    echo "sleeping ${cycle_sleep_s}s before next resumable judging cycle"
    sleep "$cycle_sleep_s"
  fi
done

bash src/utility_behavior_gap/scripts/analyze_moral_neutral_bridge_all.sh
