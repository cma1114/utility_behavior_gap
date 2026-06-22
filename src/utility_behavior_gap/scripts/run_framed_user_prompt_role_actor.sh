#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

actor="${1:-}"
task_arg="${2:-essay}"
mode="${3:-all}"

if [ "$task_arg" = "all" ] || [ "$task_arg" = "prepare" ] || [ "$task_arg" = "generate" ] || [ "$task_arg" = "judge" ] || [ "$task_arg" = "analyze" ] || [ "$task_arg" = "status" ]; then
  mode="$task_arg"
  task_arg="essay"
fi

if [ -z "$actor" ]; then
  echo "usage: bash src/utility_behavior_gap/scripts/run_framed_user_prompt_role_actor.sh ACTOR [TASK] [all|prepare|generate|judge|analyze|status]" >&2
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

case "$task_arg" in
  essay)
    task="essay"
    task_label="essay"
    max_tokens="${GENERATION_MAX_TOKENS:-900}"
    treatment_label="world-class essayist role cue"
    control_label="skilled essay writer role cue"
    ;;
  grant|grant_proposal_abstract)
    task="grant_proposal_abstract"
    task_label="grant abstract"
    max_tokens="${GENERATION_MAX_TOKENS:-1400}"
    treatment_label="world-class grant writer role cue"
    control_label="skilled grant writer role cue"
    ;;
  translation)
    task="translation"
    task_label="translation"
    max_tokens="${GENERATION_MAX_TOKENS:-600}"
    treatment_label="world-class translator role cue"
    control_label="skilled translator role cue"
    ;;
  incident|incident_postmortem)
    task="incident_postmortem"
    task_label="incident postmortem"
    max_tokens="${GENERATION_MAX_TOKENS:-3000}"
    treatment_label="world-class SRE role cue"
    control_label="skilled SRE role cue"
    ;;
  *)
    echo "unknown task: $task_arg" >&2
    exit 1
    ;;
esac

case "$mode" in
  all|prepare|generate|judge|analyze|status)
    ;;
  *)
    echo "unknown mode: $mode" >&2
    echo "usage: bash src/utility_behavior_gap/scripts/run_framed_user_prompt_role_actor.sh ACTOR [TASK] [all|prepare|generate|judge|analyze|status]" >&2
    exit 1
    ;;
esac

system_repeats="${SYSTEM_REPEATS:-5}"
modgrid_repeat_start="${MODGRID_REPEAT_START:-0}"
if [ "$modgrid_repeat_start" -gt "$system_repeats" ]; then
  echo "MODGRID_REPEAT_START cannot exceed SYSTEM_REPEATS" >&2
  exit 1
fi
repeat_end=$((system_repeats - 1))
repeat_suffix=""
if [ "$system_repeats" != "5" ] || [ "$modgrid_repeat_start" != "0" ]; then
  repeat_suffix="__r${modgrid_repeat_start}_to_r${repeat_end}"
fi
manifest_list="outputs/api/runs/framed_user_prompt_role_manifests__${actor}__${task}${repeat_suffix}.tsv"
generation_passes="${GENERATION_PASSES:-1}"
judging_passes="${JUDGING_PASSES:-1}"
generation_workers="${GENERATION_WORKERS:-10}"
judging_workers="${JUDGING_WORKERS:-5}"

prepare() {
  : > "$manifest_list"
  echo "=== prepare framed user-prompt role cue $actor / $task_label ==="
  out=$(.venv/bin/python -m utility_behavior_gap.scripts.prepare_generation_jobs \
    --comparisons framed_user_prompt_role \
    --tasks "$task" \
    --actors "$actor" \
    --system-repeats "$system_repeats" \
    --modgrid-repeat-start "$modgrid_repeat_start")
  printf '%s\n' "$out"
  manifest=$(printf '%s\n' "$out" | awk '/wrote immutable generation job manifest to/{print $NF}')
  if [ -z "$manifest" ]; then
    echo "failed to find immutable generation manifest in prepare output" >&2
    exit 1
  fi
  printf "%s\t%s\t%s\n" "$actor" "$task" "$manifest" >> "$manifest_list"
  echo "wrote manifest list to $manifest_list"
}

require_manifest_list() {
  if [ ! -f "$manifest_list" ]; then
    echo "missing manifest list: $manifest_list" >&2
    echo "run: bash src/utility_behavior_gap/scripts/run_framed_user_prompt_role_actor.sh $actor $task prepare" >&2
    exit 1
  fi
}

generate() {
  require_manifest_list
  for pass in $(seq 1 "$generation_passes"); do
    echo "=== generation pass $pass/$generation_passes ($actor; $task_label framed role cue) ==="
    while IFS=$'\t' read -r _ _ manifest; do
      .venv/bin/python -m utility_behavior_gap.scripts.run_generation \
        --workers "$generation_workers" \
        --max-tokens "$max_tokens" \
        --compact-log \
        --jobs "$manifest"
    done < "$manifest_list"
    if [ "$pass" -lt "$generation_passes" ]; then
      sleep 60
    fi
  done
}

judge() {
  require_manifest_list
  for pass in $(seq 1 "$judging_passes"); do
    echo "=== judging pass $pass/$judging_passes ($actor; $task_label framed role cue) ==="
    while IFS=$'\t' read -r _ _ manifest; do
      .venv/bin/python -m utility_behavior_gap.scripts.run_judging \
        --manifest "$manifest" \
        --orders both \
        --workers "$judging_workers"
    done < "$manifest_list"
    if [ "$pass" -lt "$judging_passes" ]; then
      sleep 60
    fi
  done
}

analyze() {
  require_manifest_list
  while IFS=$'\t' read -r _ _ manifest; do
    dir="${manifest%/generation_jobs.jsonl}"
    .venv/bin/python -m utility_behavior_gap.scripts.analyze_framed_user_strong_judging \
      --run-dir "$dir" \
      --strong-condition user_strong \
      --neutral-condition user_normal \
      --stem-prefix framed_user_prompt_role \
      --title "Framed User-Prompt Role Cue" \
      --treatment-label "$treatment_label" \
      --control-label "$control_label"
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
    if [ -f "$manifest_list" ]; then
      echo "using existing manifest list: $manifest_list"
    else
      prepare
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
  analyze)
    analyze
    ;;
  status)
    status
    ;;
esac
