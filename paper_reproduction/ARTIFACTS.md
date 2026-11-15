# Paper Artifact Map

Experiments are selected explicitly, not by timestamps or loose filename searches.

| Result | Accepted artifact under CURRENT_PAPER | Offline step |
| --- | --- | --- |
| Direct exhortation | figures/direct_instruction_main.png; features/direct_instruction_main_feature_table.tex | outcomes; features |
| High-low utility | figures/highlow_model_task_lollipop.png; figures/highlow_domain_*; features/highlow_main_feature_table.tex | outcomes; features |
| Framed role cue | results/framed_user_prompt_role_model_task_plot_data.csv; features/framed_user_prompt_role_feature_table.tex | outcomes; features |
| Harmful vs framed empty | results/moral_low_vs_framed_empty_model_task_cells.csv; features/moral_low_vs_framed_empty_feature_table.tex | outcomes; features |
| High utility vs framed neutral | results/high_utility_vs_framed_neutral_pair_outcomes.csv | outcomes |
| High utility vs framed empty / R0 | results/high_utility_vs_framed_empty_pair_outcomes.csv; results/high_utility_vs_r0_pair_outcomes.csv | outcomes |
| Utility magnitude | figures/utility_trend_two_panel_gap_and_absolute_framed_empty.png | trends |
| Feature directionality | features/feature_directionality_by_contrast_table.tex | features |
| Reasoning / suspicion exclusion | docs/appendix_reasoning_trace_analysis.md; results/reasoning_trace/ | reasoning |

Reviewer follow-up (separate from `CURRENT_PAPER`):
`docs/reviewer_response/policy_framed_empty/` contains the low-policy/framed-empty
quality results and the parallel harmful/low-policy feature table. Reproduce
with `python paper_reproduction/reproduce.py run --step policy`. The self-contained
evidence is `data/processed/policy_framed_empty.jsonl.gz`; this supplement does
not silently replace the original high-low or harmful paper snapshots.

The unified aggregate reference is
`paper_reproduction/reference/aggregate_winrate_bootstrap_cis.csv`. The runner
recomputes it from pair outcomes, not from figure annotations.

## Retained, Not Main-Paper Inputs

- Amount results, early moral high-low studies, unframed `user_prompt_role`
  experiments, system-prompt studies, and old `highlow_intervention` runs are
  retained history. Do not substitute them for the explicitly selected inputs.
- Mid-effort appendix artifacts remain in `results/appendix_effort_dose_*`.
  The four-step runner does not currently rebuild that historical experiment
  from its original votes.
- Source utility fits/outcome tables are in `data/inputs/` and the specifications.
  Re-estimating utilities requires model calls and the cited Utility Engineering
  pipeline; it is not part of offline statistical reproduction.
- Frontier-model, religion-beneficiary, and next-task follow-ups are separate
  explorations, not the seven-model paper roster. Private preservation does not
  imply public publication.

## Definitions

- `CURRENT_PAPER/docs/canonical_prompt_book.md`: approved actor paradigm.
- `CURRENT_PAPER/docs/judge_prompts.md`: literal judging prompts.
- `analysis_specs/feature_definitions.yaml`: editable feature definitions.
- `analysis_specs/reasoning_trace_stance_classifier.md`: recovered coding instructions.
- `data/processed/provenance.json`: processed-input lineage.

`outputs/paper_ready/` is not another canonical location. Regenerated outputs
belong in `outputs/reproduced/`; accepted snapshots belong in `CURRENT_PAPER/`.
