#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

actor="${1:-}"
mode="${2:-all}"

if [ -z "$actor" ]; then
  echo "usage: bash src/utility_behavior_gap/scripts/run_framed_user_prompt_role_all_tasks_actor.sh ACTOR [all|prepare|generate|judge|analyze|status]" >&2
  exit 1
fi

case "$mode" in
  all|prepare|generate|judge|analyze|status)
    ;;
  *)
    echo "unknown mode: $mode" >&2
    echo "usage: bash src/utility_behavior_gap/scripts/run_framed_user_prompt_role_all_tasks_actor.sh ACTOR [all|prepare|generate|judge|analyze|status]" >&2
    exit 1
    ;;
esac

tasks=(essay grant_proposal_abstract incident_postmortem translation)

for task in "${tasks[@]}"; do
  echo
  echo "=== framed user-prompt role: $actor / $task / $mode ==="
  bash src/utility_behavior_gap/scripts/run_framed_user_prompt_role_actor.sh "$actor" "$task" "$mode"
done
