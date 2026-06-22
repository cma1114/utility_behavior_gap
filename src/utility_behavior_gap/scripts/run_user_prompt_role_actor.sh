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
  echo "usage: bash src/utility_behavior_gap/scripts/run_user_prompt_role_actor.sh ACTOR [TASK] [all|prepare|generate|judge|analyze|status]" >&2
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
    echo "usage: bash src/utility_behavior_gap/scripts/run_user_prompt_role_actor.sh ACTOR [TASK] [all|prepare|generate|judge|analyze|status]" >&2
    exit 1
    ;;
esac

if [ "$task" = "essay" ]; then
  manifest_list="outputs/api/runs/user_prompt_role_manifests__${actor}.tsv"
else
  manifest_list="outputs/api/runs/user_prompt_role_manifests__${actor}__${task}.tsv"
fi
generation_passes="${GENERATION_PASSES:-1}"
judging_passes="${JUDGING_PASSES:-1}"
generation_workers="${GENERATION_WORKERS:-10}"
judging_workers="${JUDGING_WORKERS:-5}"

prepare() {
  : > "$manifest_list"
  echo "=== prepare user-prompt role cue $actor / $task_label ==="
  out=$(.venv/bin/python -m utility_behavior_gap.scripts.prepare_generation_jobs \
    --comparisons user_prompt_role \
    --tasks "$task" \
    --actors "$actor")
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
    echo "run: bash src/utility_behavior_gap/scripts/run_user_prompt_role_actor.sh $actor $task prepare" >&2
    exit 1
  fi
}

ensure_complete() {
  manifest="$1"
  .venv/bin/python - "$manifest" <<'PY'
import json
import sys
from pathlib import Path

from utility_behavior_gap.openrouter import judge_model_ids
from utility_behavior_gap.fingerprints import output_text_fingerprint

manifest = Path(sys.argv[1])
run_dir = manifest.parent

def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

jobs = read_jsonl(manifest)
pair_uids = {str(job["pair_uid"]) for job in jobs}
expected_outputs = {f"{pair_uid}::{suffix}" for pair_uid in pair_uids for suffix in ("a", "b")}
generations = read_jsonl(run_dir / "generations.jsonl")
valid_outputs = {
    str(row.get("output_id", ""))
    for row in generations
    if str(row.get("output_id", "")) in expected_outputs
    and row.get("success") is not False
    and (not str(row.get("finish_reason") or "") or str(row.get("finish_reason") or "") == "stop")
    and str(row.get("output_text", "")).strip()
}
current_hashes = {}
by_output_id = {str(row.get("output_id", "")): row for row in generations}
for pair_uid in pair_uids:
    out_a = by_output_id.get(f"{pair_uid}::a")
    out_b = by_output_id.get(f"{pair_uid}::b")
    if out_a is not None and out_b is not None:
        current_hashes[pair_uid] = (output_text_fingerprint(out_a), output_text_fingerprint(out_b))

judges = set(judge_model_ids())
votes = read_jsonl(run_dir / "judge_votes.jsonl")
successful_votes = {
    (str(row.get("pair_uid", "")), str(row.get("judge_model", "")), bool(row.get("flipped")))
    for row in votes
    if str(row.get("pair_uid", "")) in pair_uids
    and str(row.get("judge_model", "")) in judges
    and row.get("success") is not False
    and (
        row.get("source_output_a_hash"),
        row.get("source_output_b_hash"),
    ) == current_hashes.get(str(row.get("pair_uid", "")))
}
expected_votes = {
    (pair_uid, judge, flipped)
    for pair_uid in pair_uids
    for judge in judges
    for flipped in (False, True)
}

missing_outputs = len(expected_outputs - valid_outputs)
missing_votes = len(expected_votes - successful_votes)
if missing_outputs or missing_votes:
    print(
        f"incomplete run: {missing_outputs} missing outputs, {missing_votes} missing judge votes; "
        "rerun generate and/or judge before analyze",
        file=sys.stderr,
    )
    sys.exit(1)
PY
}

generate() {
  require_manifest_list
  for pass in $(seq 1 "$generation_passes"); do
    echo "=== generation pass $pass/$generation_passes ($actor; $task_label user-prompt role cue) ==="
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
    echo "=== judging pass $pass/$judging_passes ($actor; $task_label user-prompt role cue) ==="
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
    ensure_complete "$manifest"
    dir="${manifest%/generation_jobs.jsonl}"
    .venv/bin/python -m utility_behavior_gap.scripts.analyze_framed_user_strong_judging \
      --run-dir "$dir" \
      --strong-condition user_strong \
      --neutral-condition user_normal \
      --stem-prefix user_prompt_role \
      --title "User-Prompt Role Cue" \
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
