# Utility-Behavior Gap

Start with [START_HERE.md](START_HERE.md).

For the current paper line, the important human-facing artifacts are in:

`CURRENT_PAPER/`

That directory contains real copied files, not symlinks:

- `CURRENT_PAPER/figures/`: current paper-facing figures.
- `CURRENT_PAPER/results/`: current result summaries and model/task CSVs.
- `CURRENT_PAPER/features/`: current feature-analysis outputs.
- `CURRENT_PAPER/docs/`: current workflow/spec documentation.

The same current artifacts are also copied under `outputs/paper_ready/`.

## Current Primary Figures

- `CURRENT_PAPER/figures/direct_instruction_main.png`
- `CURRENT_PAPER/figures/highlow_model_task_lollipop.png`
- `CURRENT_PAPER/figures/amount_model_task_lollipop.png`
- `CURRENT_PAPER/figures/moral_model_task_lollipop.png`

## Current Primary Results

- `CURRENT_PAPER/results/direct_instruction_panel_summary.md`
- `CURRENT_PAPER/results/highn_condition_results_summary.md`
- `CURRENT_PAPER/results/highlow_summary.md`
- `CURRENT_PAPER/results/amount_summary.md`
- `CURRENT_PAPER/results/high_utility_vs_neutral_summary.md`
- `CURRENT_PAPER/results/amount_high_vs_neutral_summary.md`
- `CURRENT_PAPER/results/moral_low_vs_neutral_summary.md`

## Current Documentation

- `CURRENT_PAPER/docs/canonical_prompt_book.md`
- `docs/canonical_artifact_index.md`
- `docs/analysis_workflow.md`
- `analysis_specs/feature_definitions.yaml`
- `src/utility_behavior_gap/scripts/README.md`

## Important Code Entry Points

Generation and judging:

- `src/utility_behavior_gap/scripts/prepare_generation_jobs.py`
- `src/utility_behavior_gap/scripts/run_generation.py`
- `src/utility_behavior_gap/scripts/run_judging.py`
- `src/utility_behavior_gap/scripts/status.py`

Current run wrappers:

- `src/utility_behavior_gap/scripts/run_framed_user_strong_actor.sh`
- `src/utility_behavior_gap/scripts/run_fund_wording_actor.sh`
- `src/utility_behavior_gap/scripts/run_canonical_amount_base_actor.sh`
- `src/utility_behavior_gap/scripts/run_canonical_highn_actor.sh`

Current analyses:

- `src/utility_behavior_gap/scripts/export_canonical_prompt_book.py`
- `src/utility_behavior_gap/scripts/analyze_framed_user_strong_panel.py`
- `src/utility_behavior_gap/scripts/analyze_canonical_highn_conditions.py`
- `src/utility_behavior_gap/scripts/analyze_canonical_utility_gap_trend.py`
- `src/utility_behavior_gap/scripts/analyze_final_text_features.py`
- `src/utility_behavior_gap/scripts/analyze_standard_feature_deltas.py`
- `src/utility_behavior_gap/scripts/run_task_rubric_feature_coding.py`
- `src/utility_behavior_gap/scripts/analyze_task_rubric_feature_coding.py`

## Archive / Not Current

Nothing was deleted. Old or misleading artifacts were moved or documented as
archival.

- `outputs/archive/stale_paper_ready__2026-06-16/`: old stale paper-ready copy.
- `outputs/archive/symlink_paper_ready__2026-06-16/`: symlink-based paper-ready index, replaced with real copies.
- `outputs/archive/legacy_figures/`: old non-current figures.
- `outputs/analysis/`: mixed current intermediate outputs plus exploratory and historical analyses.

Do not use old `modgrid_*`, `highlow_intervention_*`, or unindexed analysis
artifacts as paper-ready without re-auditing them.
