#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

actor="${1:-}"
mode="${2:-all}"

if [ -z "$actor" ]; then
  echo "usage: bash src/utility_behavior_gap/scripts/run_framed_empty_actor.sh ACTOR [all|prepare|generate|status]" >&2
  exit 1
fi

case "$actor" in
  deepseek-v3.2-or|gpt-5.4-mini-or|glm-5.1-or|kimi-k2.5-or|mimo-v25-pro-or|qwen3.5-9b-or|qwen3.6-plus-or)
    ;;
  *)
    echo "unknown actor: $actor" >&2
    exit 1
    ;;
esac

case "$mode" in
  all|prepare|generate|status)
    ;;
  *)
    echo "unknown mode: $mode" >&2
    echo "usage: bash src/utility_behavior_gap/scripts/run_framed_empty_actor.sh ACTOR [all|prepare|generate|status]" >&2
    exit 1
    ;;
esac

manifest_list="outputs/api/runs/framed_empty_manifests__${actor}.tsv"
generation_passes="${GENERATION_PASSES:-4}"
generation_workers="${GENERATION_WORKERS:-10}"
generation_sleep_s="${GENERATION_SLEEP_S:-60}"
total_repeats="${TOTAL_REPEATS:-10}"
repeat_start="${REPEAT_START:-0}"

tasks=(
  essay
  translation
  grant_proposal_abstract
  incident_postmortem
)

comparison_for_task() {
  case "$1" in
    essay) echo "modgrid_essay_framed_empty" ;;
    translation) echo "modgrid_translation_framed_empty" ;;
    grant_proposal_abstract) echo "modgrid_grant_framed_empty" ;;
    incident_postmortem) echo "modgrid_incident_framed_empty" ;;
    *)
      echo "unknown task: $1" >&2
      exit 1
      ;;
  esac
}

max_tokens_for_task() {
  case "$1" in
    essay) echo 900 ;;
    translation) echo 600 ;;
    grant_proposal_abstract) echo 1000 ;;
    incident_postmortem) echo 3000 ;;
    *)
      echo "unknown task: $1" >&2
      exit 1
      ;;
  esac
}

prepare() {
  : > "$manifest_list"
  for task in "${tasks[@]}"; do
    comparison="$(comparison_for_task "$task")"
    echo "=== prepare framed_empty $actor / $task ==="
    out=$(.venv/bin/python -m utility_behavior_gap.scripts.prepare_generation_jobs \
      --comparisons "$comparison" \
      --tasks "$task" \
      --actors "$actor" \
      --system-repeats "$total_repeats" \
      --modgrid-repeat-start "$repeat_start")
    printf '%s\n' "$out"
    manifest=$(printf '%s\n' "$out" | awk '/immutable generation job manifest/{print $NF}')
    printf "%s\t%s\t%s\n" "$actor" "$task" "$manifest" >> "$manifest_list"
  done
  echo "wrote manifest list to $manifest_list"
}

require_manifest_list() {
  if [ ! -f "$manifest_list" ]; then
    echo "missing manifest list: $manifest_list" >&2
    echo "run: bash src/utility_behavior_gap/scripts/run_framed_empty_actor.sh $actor prepare" >&2
    exit 1
  fi
}

generate() {
  require_manifest_list
  for pass in $(seq 1 "$generation_passes"); do
    echo "=== framed_empty generation pass $pass/$generation_passes ($actor; workers=$generation_workers) ==="
    while IFS=$'\t' read -r _ task manifest; do
      max_tokens="$(max_tokens_for_task "$task")"
      echo "=== generate framed_empty $actor / $task ==="
      .venv/bin/python -m utility_behavior_gap.scripts.run_generation \
        --workers "$generation_workers" \
        --max-tokens "$max_tokens" \
        --conditions framed_empty \
        --jobs "$manifest"
    done < "$manifest_list"
    if [ "$pass" != "$generation_passes" ]; then
      sleep "$generation_sleep_s"
    fi
  done
}

status() {
  require_manifest_list
  while IFS=$'\t' read -r _ task manifest; do
    dir="${manifest%/generation_jobs.jsonl}"
    jobs=$(wc -l < "$manifest" | tr -d ' ')
    read -r gens gen_rows failures <<< "$(PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - "$dir" <<'PY'
from pathlib import Path
import json
import sys

run_dir = Path(sys.argv[1])
generation_path = run_dir / "generations.jsonl"
failure_path = run_dir / "generation_failures.jsonl"

rows = 0
ids = set()
if generation_path.exists():
    with generation_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            if row.get("condition") == "framed_empty" and row.get("success") is True:
                ids.add(str(row.get("output_id", "")))

failures = 0
if failure_path.exists():
    with failure_path.open(encoding="utf-8") as handle:
        failures = sum(1 for line in handle if line.strip())

print(len(ids), rows, failures)
PY
)"
    printf "%-18s %-24s framed_empty unique gens %4s/%4s  rows %4s  failures %3s  %s\n" \
      "$actor" "$task" "$gens" "$jobs" "$gen_rows" "$failures" "$dir"
  done < "$manifest_list"
}

case "$mode" in
  all)
    prepare
    generate
    status
    ;;
  prepare)
    prepare
    ;;
  generate)
    generate
    ;;
  status)
    status
    ;;
esac
