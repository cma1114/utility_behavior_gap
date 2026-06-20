#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

mode="${1:-all}"
pos_workers="${POS_WORKERS:-1}"
enable_pos="${ENABLE_POS:-0}"

task_codes=(
  outputs/analysis/task_rubric_pilot__task-essay__contrast-all__sample-proportional__n-0__seed-20260614__coder-google-gemini-2.5-flash/task_rubric_codes.jsonl
  outputs/analysis/task_rubric_pilot__task-translation__contrast-all__sample-proportional__n-0__seed-20260614__coder-google-gemini-2.5-flash/task_rubric_codes.jsonl
  outputs/analysis/task_rubric_pilot__task-grant_proposal_abstract__contrast-all__sample-proportional__n-0__seed-20260614__coder-google-gemini-2.5-flash/task_rubric_codes.jsonl
  outputs/analysis/task_rubric_pilot__task-incident_postmortem__contrast-all__sample-proportional__n-0__seed-20260614__coder-google-gemini-2.5-flash/task_rubric_codes.jsonl
)

baseline_codes=(
  outputs/analysis/baseline_rubric__task-essay__baseline-direct_low__arms-all-default__sample-arm_balanced__n-0__seed-20260614__coder-google-gemini-2.5-flash/baseline_rubric_codes.jsonl
  outputs/analysis/baseline_rubric__task-essay__baseline-r0__arms-all-default__sample-arm_balanced__n-0__seed-20260614__coder-google-gemini-2.5-flash/baseline_rubric_codes.jsonl
  outputs/analysis/baseline_rubric__task-translation__baseline-direct_low__arms-all-default__sample-arm_balanced__n-0__seed-20260614__coder-google-gemini-2.5-flash/baseline_rubric_codes.jsonl
  outputs/analysis/baseline_rubric__task-translation__baseline-r0__arms-all-default__sample-arm_balanced__n-0__seed-20260614__coder-google-gemini-2.5-flash/baseline_rubric_codes.jsonl
  outputs/analysis/baseline_rubric__task-grant_proposal_abstract__baseline-direct_low__arms-all-default__sample-arm_balanced__n-0__seed-20260614__coder-google-gemini-2.5-flash/baseline_rubric_codes.jsonl
  outputs/analysis/baseline_rubric__task-grant_proposal_abstract__baseline-r0__arms-all-default__sample-arm_balanced__n-0__seed-20260614__coder-google-gemini-2.5-flash/baseline_rubric_codes.jsonl
  outputs/analysis/baseline_rubric__task-incident_postmortem__baseline-direct_low__arms-all-default__sample-arm_balanced__n-0__seed-20260614__coder-google-gemini-2.5-flash/baseline_rubric_codes.jsonl
  outputs/analysis/baseline_rubric__task-incident_postmortem__baseline-r0__arms-all-default__sample-arm_balanced__n-0__seed-20260614__coder-google-gemini-2.5-flash/baseline_rubric_codes.jsonl
)

usage() {
  echo "usage: bash src/utility_behavior_gap/scripts/run_local_analysis_refresh.sh [all|audit|text|framed-r0|panel|rubric-models]" >&2
}

run_audit() {
  .venv/bin/python -m utility_behavior_gap.scripts.audit_canonical_readiness
}

run_text() {
  if [ "$enable_pos" = "1" ]; then
    .venv/bin/python -m utility_behavior_gap.scripts.analyze_final_text_features --pos-workers "$pos_workers"
  else
    .venv/bin/python -m utility_behavior_gap.scripts.analyze_final_text_features --no-pos
  fi
}

run_framed_r0() {
  .venv/bin/python -m utility_behavior_gap.scripts.analyze_framed_neutral_vs_r0_features
}

run_panel() {
  .venv/bin/python -m utility_behavior_gap.scripts.analyze_panel_feature_lasso
}

existing_paths() {
  for path in "$@"; do
    if [ -f "$path" ]; then
      printf '%s\n' "$path"
    else
      printf 'missing expected code file: %s\n' "$path" >&2
      return 1
    fi
  done
}

run_rubric_models() {
  mapfile -t existing_task_codes < <(existing_paths "${task_codes[@]}")
  mapfile -t existing_baseline_codes < <(existing_paths "${baseline_codes[@]}")
  .venv/bin/python -m utility_behavior_gap.scripts.analyze_task_rubric_feature_models \
    "${existing_task_codes[@]}" \
    --out-prefix outputs/analysis/task_rubric_feature_models
  .venv/bin/python -m utility_behavior_gap.scripts.analyze_baseline_feature_models \
    "${existing_baseline_codes[@]}" \
    --out-prefix outputs/analysis/baseline_feature_models
}

case "$mode" in
  all)
    run_audit
    run_text
    run_framed_r0
    run_panel
    run_rubric_models
    ;;
  audit)
    run_audit
    ;;
  text)
    run_text
    ;;
  framed-r0)
    run_framed_r0
    ;;
  panel)
    run_panel
    ;;
  rubric-models)
    run_rubric_models
    ;;
  *)
    usage
    exit 1
    ;;
esac
