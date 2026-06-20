# Canonical Artifact Index

This document records the current intended structure for the paper analyses.
It is an index, not an analysis script.

## Canonical Output Entry Point

Use `CURRENT_PAPER/` or `outputs/paper_ready/` first. These directories contain
real copied files for the current figures, tables, prompt book, and feature
summaries.

The stale older `outputs/paper_ready` directory has been moved to
`outputs/archive/stale_paper_ready__2026-06-16/`.

## Canonical Prompt Book

- `CURRENT_PAPER/docs/canonical_prompt_book.md`
- `outputs/paper_ready/docs/canonical_prompt_book.md`
- source analysis copy: `outputs/analysis/canonical_prompt_book.md`
- exporter: `src/utility_behavior_gap/scripts/export_canonical_prompt_book.py`

Do not use `outputs/analysis/modgrid_prompt_book.md` as current. It was
generated from older `4-comparisons` manifests and is retained only as an
archival/diagnostic artifact.

## Primary Conditions In The Paper Outline

1. Direct instruction: framed neutral versus user-prompt max effort.
2. High-low utility: high-utility versus low-utility intervention.
3. High utility versus framed neutral.
4. High utility versus framed empty.
5. High utility versus R0.
6. Moral: good-cause versus bad-cause intervention.
7. Moral low versus framed neutral.
8. Amount: larger versus smaller dollar amount.
9. Amount high versus framed neutral.
10. Feature analysis: standardized generic features plus task-specific LLM rubric coding.

## Primary Scripts

### Generation And Judging Primitives

- `src/utility_behavior_gap/scripts/prepare_generation_jobs.py`
- `src/utility_behavior_gap/scripts/run_generation.py`
- `src/utility_behavior_gap/scripts/run_judging.py`
- `src/utility_behavior_gap/scripts/status.py`

### Canonical Run Wrappers

- `src/utility_behavior_gap/scripts/run_framed_user_strong_actor.sh`
- `src/utility_behavior_gap/scripts/run_fund_wording_actor.sh`
- `src/utility_behavior_gap/scripts/run_canonical_amount_base_actor.sh`
- `src/utility_behavior_gap/scripts/run_canonical_highn_actor.sh`

### Canonical Analyses And Figures

- `src/utility_behavior_gap/scripts/export_canonical_prompt_book.py`
- `src/utility_behavior_gap/scripts/analyze_framed_user_strong_panel.py`
- `src/utility_behavior_gap/scripts/plot_direct_instruction_main.py`
- `src/utility_behavior_gap/scripts/classify_canonical_highn_moral_refusals.py`
- `src/utility_behavior_gap/scripts/analyze_canonical_highn_conditions.py`
- `src/utility_behavior_gap/scripts/plot_canonical_highn_highlow.py`
- `src/utility_behavior_gap/scripts/plot_canonical_highn_amount.py`
- `src/utility_behavior_gap/scripts/plot_canonical_highn_moral.py`
- `src/utility_behavior_gap/scripts/analyze_canonical_utility_gap_trend.py`
- `src/utility_behavior_gap/scripts/analyze_high_utility_neutral_trend.py`

### Neutral/R0 Bridge Analyses

- `src/utility_behavior_gap/scripts/prepare_highlow_neutral_bridge_judging.py`
- `src/utility_behavior_gap/scripts/analyze_highlow_neutral_bridge_judging.py`
- `src/utility_behavior_gap/scripts/plot_highlow_neutral_bridge.py`
- `src/utility_behavior_gap/scripts/prepare_amount_neutral_bridge_judging.py`
- `src/utility_behavior_gap/scripts/analyze_amount_neutral_bridge_judging.py`
- `src/utility_behavior_gap/scripts/prepare_moral_neutral_bridge_judging.py`
- `src/utility_behavior_gap/scripts/analyze_moral_neutral_bridge_judging.py`
- `src/utility_behavior_gap/scripts/run_moral_neutral_bridge_batch.sh`
- `src/utility_behavior_gap/scripts/prepare_framed_neutral_r0_bridge_judging.py`
- `src/utility_behavior_gap/scripts/analyze_framed_neutral_r0_bridge_judging.py`
- `src/utility_behavior_gap/scripts/prepare_highlow_r0_bridge_judging.py`
- `src/utility_behavior_gap/scripts/run_high_utility_r0_bridge_actor.sh`
- `src/utility_behavior_gap/scripts/analyze_high_utility_r0_bridge_all.sh`
- `src/utility_behavior_gap/scripts/plot_highlow_r0_bridge.py`
- `src/utility_behavior_gap/scripts/prepare_highlow_framed_empty_bridge_judging.py`
- `src/utility_behavior_gap/scripts/run_high_utility_framed_empty_bridge_actor.sh`
- `src/utility_behavior_gap/scripts/analyze_highlow_framed_empty_bridge_judging.py`
- `src/utility_behavior_gap/scripts/plot_highlow_framed_empty_bridge.py`

### Feature Analysis

- `src/utility_behavior_gap/scripts/analyze_final_text_features.py`
- `src/utility_behavior_gap/scripts/analyze_standard_feature_deltas.py`
- `src/utility_behavior_gap/scripts/run_task_rubric_feature_coding.py`
- `src/utility_behavior_gap/scripts/analyze_task_rubric_feature_coding.py`
- `src/utility_behavior_gap/scripts/make_feature_appendix_table.py`
- `src/utility_behavior_gap/scripts/make_direct_instruction_feature_appendix_table.py`

`make_feature_appendix_table.py` is the reusable table builder for all combined
generic-plus-rubric feature tables, including direct instruction. It also adds
`Panel preference (r)`, the within-comparison association between having more of
a feature and being preferred by the judging panel. The older direct-only table
builder is retained only for backwards compatibility.

## Definition Files

- `analysis_specs/feature_definitions.yaml`
- `docs/analysis_workflow.md`
- `src/utility_behavior_gap/prompts.py`
- `src/utility_behavior_gap/job_builder.py`
- `src/utility_behavior_gap/constants.py`
- `src/utility_behavior_gap/judging.py`
- `src/utility_behavior_gap/stats.py`

## Current Primary Figures

- `outputs/paper_ready/figures/direct_instruction_main.png`
- `outputs/paper_ready/figures/highlow_model_task_lollipop.png`
- `outputs/paper_ready/figures/high_utility_vs_framed_empty_model_task_lollipop.png`
- `outputs/paper_ready/figures/high_utility_vs_r0_model_task_lollipop.png`
- `outputs/paper_ready/figures/amount_model_task_lollipop.png`
- `outputs/paper_ready/figures/moral_model_task_lollipop.png`

## Current Primary Result Files

- `outputs/paper_ready/results/direct_instruction_panel_summary.md`
- `outputs/paper_ready/results/highn_condition_results_summary.md`
- `outputs/paper_ready/results/highn_condition_results_model_task.csv`
- `outputs/paper_ready/results/highn_condition_results_pair_outcomes.csv`
- `outputs/paper_ready/results/highlow_summary.md`
- `outputs/paper_ready/results/amount_summary.md`
- `outputs/paper_ready/results/high_utility_vs_neutral_summary.md`
- `outputs/paper_ready/results/high_utility_vs_framed_empty_summary.md`
- `outputs/paper_ready/results/high_utility_vs_framed_empty_model_task_cells.csv`
- `outputs/paper_ready/results/high_utility_vs_framed_empty_pair_outcomes.csv`
- `outputs/paper_ready/results/high_utility_vs_r0_summary.md`
- `outputs/paper_ready/results/high_utility_vs_r0_model_task_cells.csv`
- `outputs/paper_ready/results/high_utility_vs_r0_pair_outcomes.csv`
- `outputs/paper_ready/results/amount_high_vs_neutral_summary.md`
- `outputs/paper_ready/results/moral_low_vs_neutral_summary.md`

## Current Primary Feature Files

- `outputs/paper_ready/features/final_text_analysis_summary.md`
- `outputs/paper_ready/features/final_text_analysis_by_output.csv`
- `outputs/paper_ready/features/final_text_analysis_pair_deltas.csv`

Full generic-plus-rubric feature appendices:

- `outputs/paper_ready/features/direct_instruction_combined_feature_appendix_summary.md`
- `outputs/paper_ready/features/utility_combined_feature_appendix_summary.md`
- `outputs/paper_ready/features/amount_combined_feature_appendix_summary.md`
- `outputs/paper_ready/features/moral_combined_feature_appendix_summary.md`
- `outputs/paper_ready/features/utility_high_vs_r0_combined_feature_appendix_summary.md`
- `outputs/paper_ready/features/utility_high_vs_framed_empty_combined_feature_appendix_summary.md`
- `outputs/paper_ready/features/moral_low_vs_r0_combined_feature_appendix_summary.md`
- `outputs/paper_ready/features/moral_low_vs_framed_empty_combined_feature_appendix_summary.md`

Generic-only bridge feature appendices:

- `outputs/paper_ready/features/utility_high_vs_framed_neutral_generic_feature_appendix_summary.md`
- `outputs/paper_ready/features/amount_high_vs_framed_neutral_generic_feature_appendix_summary.md`
- `outputs/paper_ready/features/moral_low_vs_framed_neutral_generic_feature_appendix_summary.md`

Rubric audit data currently mirrored into paper-ready outputs:

- `outputs/paper_ready/features/direct_instruction_rubric_coding/`

## Current Primary Documentation Files

- `outputs/paper_ready/docs/canonical_prompt_book.md`
- `outputs/paper_ready/docs/canonical_artifact_index.md`
- `outputs/paper_ready/docs/analysis_workflow.md`
- `outputs/paper_ready/docs/feature_definitions.yaml`

## Public Repository Planning

- `docs/public_repo_manifest.md`
- `docs/public_repo_audit_checklist.md`

These are planning documents for a future clean public repository. They are
not a staging area and do not imply that files have been copied.

## Appendix / Exploratory Paper-Relevant Files

These are not primary main-text results, but are current paper-relevant
appendix candidates.

- `docs/appendix_findings_to_carry_forward.md`
- `outputs/paper_ready/docs/appendix_findings_to_carry_forward.md`
- `docs/appendix_effort_dose_test.md`
- `outputs/paper_ready/docs/appendix_effort_dose_test.md`
- `CURRENT_PAPER/docs/appendix_reasoning_trace_incentive_mentions.md`
- `outputs/paper_ready/docs/appendix_reasoning_trace_incentive_mentions.md`
- `outputs/paper_ready/figures/appendix_effort_dose_mid_vs_neutral.png`
- `outputs/paper_ready/figures/appendix_effort_dose_combined.png`
- `outputs/paper_ready/results/appendix_effort_dose_summary.md`
- `outputs/paper_ready/results/appendix_effort_dose_summary.csv`
- `outputs/paper_ready/results/appendix_effort_dose_lollipop_data.csv`
- `outputs/analysis/figures/framed_user_mid_vs_neutral_lollipop.png`
- `outputs/analysis/figures/framed_user_effort_dose_lollipops.png`
- `outputs/analysis/framed_user_mid_effort_essay_dose_summary.md`
- `outputs/analysis/framed_user_mid_effort_essay_dose_summary.csv`
- `outputs/analysis/framed_user_effort_dose_lollipop_data.csv`
- `outputs/paper_ready/figures/highlow_reasoning_traces_medium_essay_lollipop.png`
- `outputs/analysis/highlow_reasoning_traces_medium__tasks-essay__7-actors_summary.md`
- `outputs/analysis/reasoning_incentive_mentions__success_only_summary.md`
- `outputs/analysis/reasoning_mentions_vs_performance__both_traces_required_summary.md`
- `outputs/analysis/moral_r0_bridge__combined_7runs__summary.md`

## Archival Or Diagnostic Areas

- `outputs/archive/stale_paper_ready__2026-06-16/`
- `outputs/archive/legacy_figures/`
- `outputs/analysis/` files not linked from `outputs/paper_ready/`
- old `modgrid_*` summaries
- old `highlow_intervention_*` figures and summaries
