#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

actor="${1:-}"
mode="${2:-all}"

if [ -z "$actor" ]; then
  echo "usage: bash src/utility_behavior_gap/scripts/run_framed_user_mid_effort_essay_actor.sh ACTOR [all|prepare|generate|judge|analyze|status]" >&2
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
  all|prepare|generate|judge|analyze|status)
    ;;
  *)
    echo "unknown mode: $mode" >&2
    echo "usage: bash src/utility_behavior_gap/scripts/run_framed_user_mid_effort_essay_actor.sh ACTOR [all|prepare|generate|judge|analyze|status]" >&2
    exit 1
    ;;
esac

manifest_list="outputs/api/runs/framed_user_mid_effort_essay_manifests__${actor}.tsv"
generation_passes="${GENERATION_PASSES:-3}"
judging_passes="${JUDGING_PASSES:-12}"
generation_workers="${GENERATION_WORKERS:-10}"
judging_workers="${JUDGING_WORKERS:-5}"

prepare() {
  : > "$manifest_list"
  echo "=== prepare $actor / essay ==="
  out=$(.venv/bin/python -m utility_behavior_gap.scripts.prepare_framed_user_mid_effort_essay_jobs \
    --actors "$actor")
  printf '%s\n' "$out"
  manifest=$(printf '%s\n' "$out" | awk '/wrote .*generation_jobs.jsonl/{print $NF}')
  printf "%s\t%s\t%s\n" "$actor" "essay" "$manifest" >> "$manifest_list"
  echo "wrote manifest list to $manifest_list"
}

require_manifest_list() {
  if [ ! -f "$manifest_list" ]; then
    echo "missing manifest list: $manifest_list" >&2
    echo "run: bash src/utility_behavior_gap/scripts/run_framed_user_mid_effort_essay_actor.sh $actor prepare" >&2
    exit 1
  fi
}

generate() {
  require_manifest_list
  for pass in $(seq 1 "$generation_passes"); do
    echo "=== generation pass $pass/$generation_passes ($actor; essay mid-effort) ==="
    while IFS=$'\t' read -r _ _ manifest; do
      .venv/bin/python -m utility_behavior_gap.scripts.run_generation \
        --workers "$generation_workers" \
        --max-tokens 900 \
        --jobs "$manifest"
    done < "$manifest_list"
    sleep 60
  done
}

judge() {
  require_manifest_list
  for pass in $(seq 1 "$judging_passes"); do
    echo "=== judging pass $pass/$judging_passes ($actor; essay mid-effort) ==="
    while IFS=$'\t' read -r _ _ manifest; do
      .venv/bin/python -m utility_behavior_gap.scripts.run_judging \
        --manifest "$manifest" \
        --orders both \
        --workers "$judging_workers"
    done < "$manifest_list"
    sleep 60
  done
}

analyze() {
  require_manifest_list
  while IFS=$'\t' read -r _ _ manifest; do
    dir="${manifest%/generation_jobs.jsonl}"
    .venv/bin/python -m utility_behavior_gap.scripts.analyze_framed_user_strong_judging \
      --run-dir "$dir" \
      --strong-condition framed_user_mid_effort \
      --neutral-condition framed_neutral \
      --stem-prefix framed_user_mid_effort_essay \
      --title "Framed User-Prompt Mid-Effort Essay" \
      --treatment-label "mid-effort" \
      --control-label "framed-neutral"
  done < "$manifest_list"
}

status() {
  require_manifest_list
  while IFS=$'\t' read -r _ _ manifest; do
    dir="${manifest%/generation_jobs.jsonl}"
    .venv/bin/python -m utility_behavior_gap.scripts.status \
      --run-dir "$dir" \
      --orders both
  done < "$manifest_list"
}

case "$mode" in
  all)
    prepare
    generate
    judge
    analyze
    status
    ;;
  prepare)
    prepare
    ;;
  generate)
    generate
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
