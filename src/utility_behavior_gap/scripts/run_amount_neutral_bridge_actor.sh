#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

actor="${1:-}"
mode="${2:-all}"

if [ -z "$actor" ]; then
  echo "usage: bash src/utility_behavior_gap/scripts/run_amount_neutral_bridge_actor.sh ACTOR [all|prepare|judge|analyze|status]" >&2
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
    echo "usage: bash src/utility_behavior_gap/scripts/run_amount_neutral_bridge_actor.sh ACTOR [all|prepare|judge|analyze|status]" >&2
    exit 1
    ;;
esac

manifest_list="outputs/api/runs/amount_neutral_bridge_manifests__${actor}.tsv"
judging_passes="${JUDGING_PASSES:-6}"
judging_workers="${JUDGING_WORKERS:-10}"
judging_sleep_s="${JUDGING_SLEEP_S:-60}"

prepare() {
  echo "=== prepare amount high vs framed neutral: $actor ==="
  .venv/bin/python -m utility_behavior_gap.scripts.prepare_amount_neutral_bridge_judging \
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
    echo "run: bash src/utility_behavior_gap/scripts/run_amount_neutral_bridge_actor.sh $actor prepare" >&2
    exit 1
  fi
}

judge() {
  require_manifest_list
  for pass in $(seq 1 "$judging_passes"); do
    echo "=== judging pass $pass/$judging_passes ($actor; amount high vs framed neutral) ==="
    while IFS=$'\t' read -r _ manifest; do
      dir="${manifest%/generation_jobs.jsonl}"
      .venv/bin/python -m utility_behavior_gap.scripts.run_judging \
        --run-dir "$dir" \
        --orders both \
        --workers "$judging_workers"
    done < "$manifest_list"
    if [ "$pass" != "$judging_passes" ]; then
      sleep "$judging_sleep_s"
    fi
  done
}

analyze() {
  require_manifest_list
  while IFS=$'\t' read -r _ manifest; do
    dir="${manifest%/generation_jobs.jsonl}"
    .venv/bin/python -m utility_behavior_gap.scripts.analyze_amount_neutral_bridge_judging \
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
    ensure_manifest_list
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
