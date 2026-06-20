# Script Map

This directory contains both current paper workflow scripts and older
diagnostic/exploratory scripts. Do not infer canonical status from a script
being present here.

For the current paper-facing workflow, start with:

- `docs/canonical_artifact_index.md`
- `docs/analysis_workflow.md`
- `outputs/paper_ready/README.md`

## Current Workflow Scripts

Generation and judging:

- `prepare_generation_jobs.py`
- `run_generation.py`
- `run_judging.py`
- `status.py`

Run wrappers:

- `run_framed_user_strong_actor.sh`
- `run_fund_wording_actor.sh`
- `run_canonical_amount_base_actor.sh`
- `run_canonical_highn_actor.sh`

Canonical analyses:

- `export_canonical_prompt_book.py`
- `analyze_framed_user_strong_panel.py`
- `plot_direct_instruction_main.py`
- `classify_canonical_highn_moral_refusals.py`
- `analyze_canonical_highn_conditions.py`
- `plot_canonical_highn_highlow.py`
- `plot_canonical_highn_amount.py`
- `plot_canonical_highn_moral.py`
- `analyze_canonical_utility_gap_trend.py`
- `analyze_high_utility_neutral_trend.py`

Bridge analyses:

- `prepare_highlow_neutral_bridge_judging.py`
- `analyze_highlow_neutral_bridge_judging.py`
- `plot_highlow_neutral_bridge.py`
- `prepare_amount_neutral_bridge_judging.py`
- `analyze_amount_neutral_bridge_judging.py`
- `prepare_moral_neutral_bridge_judging.py`
- `analyze_moral_neutral_bridge_judging.py`
- `run_moral_neutral_bridge_batch.sh`
- `prepare_framed_neutral_r0_bridge_judging.py`
- `analyze_framed_neutral_r0_bridge_judging.py`

Feature analyses:

- `analyze_final_text_features.py`
- `analyze_standard_feature_deltas.py`
- `run_task_rubric_feature_coding.py`
- `analyze_task_rubric_feature_coding.py`
- `make_feature_appendix_table.py`
- `make_direct_instruction_feature_appendix_table.py`

## Diagnostic, Exploratory, Or Historical Scripts

Scripts not listed above should be treated as diagnostic, exploratory, or
historical unless promoted into `docs/canonical_artifact_index.md`.

Current exploratory scripts:

- `prepare_framed_user_mid_effort_essay_jobs.py`
- `run_framed_user_mid_effort_essay_actor.sh`
- `run_highlow_reasoning_actor.sh`
- `analyze_highlow_reasoning.py`
- `run_framed_empty_actor.sh`
- `prepare_highlow_framed_empty_bridge_judging.py`
- `run_high_utility_framed_empty_bridge_actor.sh`
- `analyze_highlow_framed_empty_bridge_judging.py`
- `plot_highlow_framed_empty_bridge.py`
