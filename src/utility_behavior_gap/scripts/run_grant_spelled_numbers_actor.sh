#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

actor="${1:-}"
mode="${2:-all}"

if [ -z "$actor" ]; then
  echo "usage: bash src/utility_behavior_gap/scripts/run_grant_spelled_numbers_actor.sh ACTOR [all|prepare|generate|judge|status]" >&2
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
  all|prepare|generate|judge|status)
    ;;
  *)
    echo "unknown mode: $mode" >&2
    echo "usage: bash src/utility_behavior_gap/scripts/run_grant_spelled_numbers_actor.sh ACTOR [all|prepare|generate|judge|status]" >&2
    exit 1
    ;;
esac

manifest_list="outputs/api/runs/grant_spelled_numbers_manifests__${actor}.tsv"
generation_passes="${GENERATION_PASSES:-3}"
judging_passes="${JUDGING_PASSES:-12}"
generation_workers="${GENERATION_WORKERS:-10}"
judging_workers="${JUDGING_WORKERS:-5}"

prepare() {
  echo "=== prepare $actor / grant_proposal_abstract / spelled-number high-low ==="
  out=$(.venv/bin/python -m utility_behavior_gap.scripts.prepare_generation_jobs \
    --comparisons modgrid_grant_highlow_spelled_numbers \
    --tasks grant_proposal_abstract \
    --actors "$actor")
  printf '%s\n' "$out"
  manifest=$(printf '%s\n' "$out" | awk '/immutable generation job manifest/{print $NF}')
  printf "%s\t%s\t%s\n" "$actor" "grant_proposal_abstract" "$manifest" > "$manifest_list"
  echo "wrote manifest list to $manifest_list"
}

require_manifest_list() {
  if [ ! -f "$manifest_list" ]; then
    echo "missing manifest list: $manifest_list" >&2
    echo "run: bash src/utility_behavior_gap/scripts/run_grant_spelled_numbers_actor.sh $actor prepare" >&2
    exit 1
  fi
}

generate() {
  require_manifest_list
  for pass in $(seq 1 "$generation_passes"); do
    echo "=== generation pass $pass/$generation_passes ($actor) ==="
    while IFS=$'\t' read -r _ task manifest; do
      echo "=== generate $actor / $task ==="
      .venv/bin/python -m utility_behavior_gap.scripts.run_generation \
        --workers "$generation_workers" \
        --max-tokens 1000 \
        --jobs "$manifest"
    done < "$manifest_list"
    sleep 60
  done
}

judge() {
  require_manifest_list
  for pass in $(seq 1 "$judging_passes"); do
    echo "=== judging pass $pass/$judging_passes ($actor) ==="
    while IFS=$'\t' read -r _ task manifest; do
      echo "=== judge $actor / $task ==="
      .venv/bin/python -m utility_behavior_gap.scripts.run_judging \
        --manifest "$manifest" \
        --orders both \
        --workers "$judging_workers"
    done < "$manifest_list"
    sleep 60
  done
}

status() {
  require_manifest_list
  while IFS=$'\t' read -r _ task manifest; do
    dir="${manifest%/generation_jobs.jsonl}"
    jobs=$(wc -l < "$manifest" | tr -d ' ')
    if [ -f "$dir/generations.jsonl" ]; then
      gens=$(wc -l < "$dir/generations.jsonl" | tr -d ' ')
    else
      gens=0
    fi
    if [ -f "$dir/judge_votes.jsonl" ]; then
      votes=$(wc -l < "$dir/judge_votes.jsonl" | tr -d ' ')
    else
      votes=0
    fi
    printf "%-18s %-24s gens %4s/%4s  votes %5s/%5s  %s\n" \
      "$actor" "$task" "$gens" "$((2 * jobs))" "$votes" "$((6 * jobs))" "$dir"
  done < "$manifest_list"
}

case "$mode" in
  all)
    prepare
    generate
    judge
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
  status)
    status
    ;;
esac
