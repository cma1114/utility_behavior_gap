#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

actor="${1:-}"
mode="${2:-all}"

if [ -z "$actor" ]; then
  echo "usage: bash src/utility_behavior_gap/scripts/run_moral_neutral_bridge_actor.sh ACTOR [all|prepare|judge|analyze|status]" >&2
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
  all|prepare|judge|analyze|status)
    ;;
  *)
    echo "unknown mode: $mode" >&2
    echo "usage: bash src/utility_behavior_gap/scripts/run_moral_neutral_bridge_actor.sh ACTOR [all|prepare|judge|analyze|status]" >&2
    exit 1
    ;;
esac

manifest_list="outputs/api/runs/moral_neutral_bridge_manifests__${actor}.tsv"
judging_passes="${JUDGING_PASSES:-1}"
judging_workers="${JUDGING_WORKERS:-1}"
judging_sleep_s="${JUDGING_SLEEP_S:-300}"

prepare() {
  echo "=== prepare moral bad vs framed neutral: $actor ==="
  .venv/bin/python -m utility_behavior_gap.scripts.prepare_moral_neutral_bridge_judging \
    --actors "$actor"
}

ensure_manifest_list() {
  if [ ! -f "$manifest_list" ]; then
    prepare
  else
    echo "using existing manifest list: $manifest_list"
  fi
}

require_manifest_list() {
  if [ ! -f "$manifest_list" ]; then
    echo "missing manifest list: $manifest_list" >&2
    echo "run: bash src/utility_behavior_gap/scripts/run_moral_neutral_bridge_actor.sh $actor prepare" >&2
    exit 1
  fi
}

judge() {
  require_manifest_list
  for pass in $(seq 1 "$judging_passes"); do
    echo "=== judging pass $pass/$judging_passes ($actor; moral bad vs framed neutral) ==="
    while IFS=$'\t' read -r _ manifest; do
      dir="${manifest%/generation_jobs.jsonl}"
      remaining="$(pending_votes "$dir")"
      if [ "$remaining" = "0" ]; then
        echo "already complete: $dir"
        continue
      fi
      echo "pending judge votes before pass: $remaining"
      .venv/bin/python -m utility_behavior_gap.scripts.run_judging \
        --run-dir "$dir" \
        --orders both \
        --workers "$judging_workers"
      remaining="$(pending_votes "$dir")"
      echo "pending judge votes after pass: $remaining"
    done < "$manifest_list"
    if all_complete; then
      echo "all moral bad vs framed neutral judging complete for $actor"
      break
    fi
    if [ "$pass" != "$judging_passes" ]; then
      sleep "$judging_sleep_s"
    fi
  done
}

pending_votes() {
  local dir="$1"
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - "$dir" <<'PY'
from pathlib import Path
import sys

from utility_behavior_gap.io_utils import read_jsonl
from utility_behavior_gap.openrouter import judge_model_ids
from utility_behavior_gap.paths import OUTPUT_API
from utility_behavior_gap.scripts.run_judging import (
    current_pair_hashes,
    existing_vote_keys,
    generation_map,
    pending_judge_requests,
    run_log_path,
)

run_dir = Path(sys.argv[1])
jobs = read_jsonl(run_dir / "generation_jobs.jsonl")
generations = generation_map([run_dir / "generations.jsonl"])
pair_hashes = current_pair_hashes(jobs, generations)
run_votes = run_log_path(jobs, "run_judge_votes_path", OUTPUT_API / "judge_votes.jsonl")
done = existing_vote_keys(pair_hashes, paths=[run_votes])
pending = pending_judge_requests(
    jobs=jobs,
    generations=generations,
    done=done,
    judges=judge_model_ids(),
    seed=12345,
    limit=None,
    orders="both",
)
print(len(pending))
PY
}

all_complete() {
  while IFS=$'\t' read -r _ manifest; do
    dir="${manifest%/generation_jobs.jsonl}"
    if [ "$(pending_votes "$dir")" != "0" ]; then
      return 1
    fi
  done < "$manifest_list"
  return 0
}

analyze() {
  require_manifest_list
  while IFS=$'\t' read -r _ manifest; do
    dir="${manifest%/generation_jobs.jsonl}"
    .venv/bin/python -m utility_behavior_gap.scripts.analyze_moral_neutral_bridge_judging \
      --run-dir "$dir"
  done < "$manifest_list"
}

status() {
  require_manifest_list
  while IFS=$'\t' read -r _ manifest; do
    dir="${manifest%/generation_jobs.jsonl}"
    .venv/bin/python -m utility_behavior_gap.scripts.status \
      --run-dir "$dir" \
      --orders both
  done < "$manifest_list"
}

case "$mode" in
  all)
    prepare
    judge
    analyze
    status
    ;;
  prepare)
    prepare
    ;;
  judge)
    judge
    ;;
  analyze)
    analyze
    ;;
  status)
    status
    ;;
esac
