#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

actor="${1:-}"
mode="${2:-all}"

if [ -z "$actor" ]; then
  echo "usage: bash src/utility_behavior_gap/scripts/run_highlow_reasoning_actor.sh ACTOR [all|prepare|generate|judge|status|smoke|analyze]" >&2
  echo "default task scope: essay only. Set TASKS=all to include all four tasks." >&2
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
  all|prepare|generate|judge|status|smoke|analyze)
    ;;
  *)
    echo "unknown mode: $mode" >&2
    echo "usage: bash src/utility_behavior_gap/scripts/run_highlow_reasoning_actor.sh ACTOR [all|prepare|generate|judge|status|smoke|analyze]" >&2
    exit 1
    ;;
esac

manifest_list="outputs/api/runs/highlow_reasoning_medium_manifests__${actor}.tsv"
generation_passes="${GENERATION_PASSES:-3}"
judging_passes="${JUDGING_PASSES:-12}"
generation_workers="${GENERATION_WORKERS:-4}"
judging_workers="${JUDGING_WORKERS:-3}"
total_repeats="${TOTAL_REPEATS:-10}"
repeat_start="${REPEAT_START:-0}"
tasks_arg="${TASKS:-essay}"

if [ "$tasks_arg" = "all" ]; then
  tasks=(essay translation grant_proposal_abstract incident_postmortem)
else
  IFS=',' read -r -a tasks <<< "$tasks_arg"
fi

for task in "${tasks[@]}"; do
  case "$task" in
    essay|translation|grant_proposal_abstract|incident_postmortem)
      ;;
    *)
      echo "unknown task in TASKS: $task" >&2
      exit 1
      ;;
  esac
done

task_enabled() {
  local wanted="$1"
  for task in "${tasks[@]}"; do
    if [ "$task" = "$wanted" ]; then
      return 0
    fi
  done
  return 1
}

comparison_for_task() {
  case "$1" in
    essay) echo "modgrid_essay_highlow_reasoning_medium" ;;
    translation) echo "modgrid_translation_highlow_reasoning_medium" ;;
    grant_proposal_abstract) echo "modgrid_grant_highlow_reasoning_medium" ;;
    incident_postmortem) echo "modgrid_incident_highlow_reasoning_medium" ;;
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
    comp="$(comparison_for_task "$task")"
    echo "=== prepare high-low reasoning-medium $actor / $task ==="
    out=$(.venv/bin/python -m utility_behavior_gap.scripts.prepare_generation_jobs \
      --comparisons "$comp" \
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
    echo "run: bash src/utility_behavior_gap/scripts/run_highlow_reasoning_actor.sh $actor prepare" >&2
    exit 1
  fi
}

selected_generations_complete() {
  require_manifest_list
  local saw_selected=0
  while IFS=$'\t' read -r _ task manifest; do
    if ! task_enabled "$task"; then
      continue
    fi
    saw_selected=1
    local dir jobs gens
    dir="${manifest%/generation_jobs.jsonl}"
    jobs=$(wc -l < "$manifest" | tr -d ' ')
    if [ -f "$dir/generations.jsonl" ]; then
      gens=$(wc -l < "$dir/generations.jsonl" | tr -d ' ')
    else
      gens=0
    fi
    if [ "$gens" -lt "$((2 * jobs))" ]; then
      return 1
    fi
  done < "$manifest_list"
  [ "$saw_selected" -eq 1 ]
}

selected_judging_complete() {
  require_manifest_list
  local saw_selected=0
  while IFS=$'\t' read -r _ task manifest; do
    if ! task_enabled "$task"; then
      continue
    fi
    saw_selected=1
    local dir jobs votes
    dir="${manifest%/generation_jobs.jsonl}"
    jobs=$(wc -l < "$manifest" | tr -d ' ')
    if [ -f "$dir/judge_votes.jsonl" ]; then
      votes=$(wc -l < "$dir/judge_votes.jsonl" | tr -d ' ')
    else
      votes=0
    fi
    if [ "$votes" -lt "$((6 * jobs))" ]; then
      return 1
    fi
  done < "$manifest_list"
  [ "$saw_selected" -eq 1 ]
}

generate() {
  require_manifest_list
  if selected_generations_complete; then
    echo "selected generation tasks already complete for $actor (TASKS=$tasks_arg)"
    return
  fi
  for pass in $(seq 1 "$generation_passes"); do
    echo "=== generation pass $pass/$generation_passes ($actor high-low reasoning-medium) ==="
    while IFS=$'\t' read -r _ task manifest; do
      if ! task_enabled "$task"; then
        continue
      fi
      max_tokens="$(max_tokens_for_task "$task")"
      echo "=== generate $actor / $task ==="
      .venv/bin/python -m utility_behavior_gap.scripts.run_generation \
        --workers "$generation_workers" \
        --max-tokens "$max_tokens" \
        --reasoning-effort medium \
        --reasoning-exclude \
        --jobs "$manifest"
    done < "$manifest_list"
    if selected_generations_complete; then
      echo "selected generation tasks complete for $actor (TASKS=$tasks_arg)"
      break
    fi
    if [ "$pass" -lt "$generation_passes" ]; then
      sleep 60
    fi
  done
}

judge() {
  require_manifest_list
  if selected_judging_complete; then
    echo "selected judging tasks already complete for $actor (TASKS=$tasks_arg)"
    return
  fi
  for pass in $(seq 1 "$judging_passes"); do
    echo "=== judging pass $pass/$judging_passes ($actor high-low reasoning-medium) ==="
    while IFS=$'\t' read -r _ task manifest; do
      if ! task_enabled "$task"; then
        continue
      fi
      echo "=== judge $actor / $task ==="
      .venv/bin/python -m utility_behavior_gap.scripts.run_judging \
        --manifest "$manifest" \
        --orders both \
        --workers "$judging_workers"
    done < "$manifest_list"
    if selected_judging_complete; then
      echo "selected judging tasks complete for $actor (TASKS=$tasks_arg)"
      break
    fi
    if [ "$pass" -lt "$judging_passes" ]; then
      sleep 60
    fi
  done
}

smoke() {
  if [ ! -f "$manifest_list" ]; then
    prepare
  fi
  IFS=$'\t' read -r _ task manifest < "$manifest_list"
  if ! task_enabled "$task"; then
    echo "first manifest task is $task, but TASKS=$tasks_arg; run prepare with the desired TASKS first" >&2
    exit 1
  fi
  max_tokens="$(max_tokens_for_task "$task")"
  echo "=== smoke generation: $actor / $task, one output only ==="
  .venv/bin/python -m utility_behavior_gap.scripts.run_generation \
    --workers 1 \
    --limit 1 \
    --max-tokens "$max_tokens" \
    --reasoning-effort medium \
    --reasoning-exclude \
    --jobs "$manifest"
  status
}

status() {
  require_manifest_list
  while IFS=$'\t' read -r _ task manifest; do
    if ! task_enabled "$task"; then
      continue
    fi
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
    printf "%-18s %-24s jobs %4s  gens %4s/%4s  votes %5s/%5s  %s\n" \
      "$actor" "$task" "$jobs" "$gens" "$((2 * jobs))" "$votes" "$((6 * jobs))" "$dir"
  done < "$manifest_list"
}

analyze() {
  require_manifest_list
  .venv/bin/python -m utility_behavior_gap.scripts.analyze_highlow_reasoning \
    --manifest-list "$manifest_list" \
    --tasks "$tasks_arg"
}

case "$mode" in
  all)
    if [ ! -f "$manifest_list" ]; then
      prepare
    else
      echo "using existing manifest list: $manifest_list"
    fi
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
  status)
    status
    ;;
  smoke)
    smoke
    ;;
  analyze)
    analyze
    ;;
esac
