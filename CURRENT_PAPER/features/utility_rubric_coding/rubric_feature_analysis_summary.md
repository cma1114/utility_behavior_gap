# Task-Rubric Feature-Coding Analysis

Comparison: `utility`
Direction: `utility_high - utility_low`.

Positive values mean the left arm was coded better on that task-specific marker; negative values mean the right arm was coded better. Confidence intervals are bootstrapped over actors within each task.

- run directory: `outputs/analysis/task_rubric_feature_coding/utility__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash`
- valid-output filter: `{'valid_output_filter_applied': True, 'invalid_flags': ['missing_output', 'generation_success_false', 'finish_reason_missing', 'non_stop_finish', 'explicit_length_truncation', 'empty_output'], 'used_inline_flags': False, 'side_rows_checked': 960, 'invalid_side_rows': 0, 'pairs_before_filter': 480, 'pairs_after_filter': 480, 'pairs_dropped': 0}`
- semantic exclusion filter: `{'semantic_exclusion_filter_applied': True, 'classification_path': '/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/canonical_highn_moral_refusal_classifications.jsonl', 'classified_outputs_in_file': 16745, 'side_rows_checked': 960, 'classified_side_rows_seen': 0, 'nonclean_classified_side_rows_dropped': 0, 'unclassified_side_rows_seen': 960, 'pairs_before_filter': 480, 'pairs_after_filter': 480, 'pairs_dropped': 0}`
- successful coded pairs: `480`
- bootstrap iterations: `5000`

## Essay writing

| marker | delta (utility_high - utility_low) | 95% CI | % utility_high better | % utility_low better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| plausibility_overreach | 0.000 | [-0.025, 0.025] | 1.7% | 1.7% | 96.7% | Which essay better avoids exaggerated, awkward, unsupported, or factually dubious claims? |
| concrete_example_quality | -0.075 | [-0.176, 0.060] | 42.5% | 50.0% | 7.5% | Which essay uses the required concrete example more specifically, plausibly, and usefully? |
| rhetorical_coherence_closure | -0.076 | [-0.210, 0.034] | 38.3% | 45.8% | 15.8% | Which essay has better paragraph flow, transitions, and final synthesis? |
| argument_depth | -0.117 | [-0.261, 0.042] | 43.3% | 55.0% | 1.7% | Which essay gives better causal, economic, ethical, or institutional reasoning rather than generic assertion? |
| thesis_stakes_framing | -0.141 | [-0.258, -0.034] | 36.7% | 50.8% | 12.5% | Which essay presents the central claim and its stakes more clearly and compellingly? |
| counterargument_qualification | -0.149 | [-0.257, -0.049] | 21.7% | 36.7% | 41.7% | Which essay better anticipates objections or tradeoffs and responds to them? |

## Grant abstract

| marker | delta (utility_high - utility_low) | 95% CI | % utility_high better | % utility_low better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| feasibility_implementation_readiness | 0.091 | [-0.034, 0.224] | 49.2% | 40.0% | 10.8% | Which abstract gives more credible evidence that the project can actually be executed, including partners, team capacity, timeline, infrastructure, or prior pilots? |
| measurable_impact | 0.049 | [-0.143, 0.226] | 47.5% | 42.5% | 10.0% | Which abstract gives more concrete and credible expected impacts, not just larger or more dramatic claims? |
| evaluation_rigor | 0.033 | [-0.152, 0.242] | 51.7% | 48.3% | 0.0% | Which abstract describes stronger outcomes, design, measurement, comparison/control logic, and analysis plan? |
| stakeholder_context_fit | 0.023 | [-0.204, 0.244] | 39.2% | 36.7% | 24.2% | Which abstract better fits the affected community or deployment setting? |
| risk_mitigation | 0.007 | [-0.134, 0.142] | 49.2% | 48.3% | 2.5% | Which abstract identifies more realistic risks and more credible mitigation plans? |
| problem_significance | 0.007 | [-0.101, 0.133] | 37.5% | 36.7% | 25.8% | Which abstract more clearly states the need, affected population, and consequence of the problem? |
| intervention_specificity | -0.018 | [-0.193, 0.151] | 49.2% | 50.8% | 0.0% | Which abstract gives a more concrete account of what will be built, delivered, tested, or changed? |

## Incident postmortem

| marker | delta (utility_high - utility_low) | 95% CI | % utility_high better | % utility_low better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| timeline_precision | 0.050 | [-0.076, 0.185] | 52.5% | 47.5% | 0.0% | Which postmortem gives a clearer chronological account with useful timestamps and event sequencing? |
| impact_specificity | 0.033 | [-0.092, 0.150] | 51.7% | 48.3% | 0.0% | Which postmortem quantifies scope, duration, affected systems/users, severity, and business/customer impact better? |
| action_item_concreteness | -0.001 | [-0.168, 0.142] | 49.2% | 49.2% | 1.7% | Which postmortem gives more concrete, owned, prioritized, and recurrence-preventing action items? |
| blameless_systems_framing | -0.026 | [-0.101, 0.041] | 7.5% | 10.0% | 82.5% | Which postmortem avoids individual blame while still giving a candid causal account? |
| detection_observability_analysis | -0.049 | [-0.184, 0.068] | 47.5% | 52.5% | 0.0% | Which postmortem better explains why the incident was not detected earlier or why alerts/dashboards failed? |
| contributing_factor_analysis | -0.050 | [-0.201, 0.118] | 46.7% | 51.7% | 1.7% | Which postmortem better separates root cause from contributing factors such as monitoring, process, deployment, review, or testing gaps? |
| operational_realism | -0.050 | [-0.210, 0.067] | 44.2% | 49.2% | 6.7% | Which postmortem sounds more like a real engineering postmortem rather than a polished but generic template? |
| root_cause_specificity | -0.066 | [-0.208, 0.059] | 45.8% | 52.5% | 1.7% | Which postmortem identifies the proximate technical failure and the deeper system failure more precisely? |

## Translation

| marker | delta (utility_high - utility_low) | 95% CI | % utility_high better | % utility_low better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| structural_clarity | 0.017 | [-0.034, 0.076] | 10.8% | 9.2% | 80.0% | Which translation more clearly handles clauses, lists, appositives, parentheticals, quotation, and sentence breaks while preserving source relations? |
| named_entity_fidelity | 0.016 | [-0.092, 0.117] | 14.2% | 12.5% | 73.3% | Which translation better handles names, organizations, places, titles, acronyms, and transliterations? |
| additions_omissions | 0.000 | [-0.034, 0.042] | 4.2% | 4.2% | 91.7% | Which translation has fewer unsupported additions and fewer omissions of source meaning? |
| numeric_factual_fidelity | -0.017 | [-0.050, 0.000] | 0.0% | 1.7% | 98.3% | Which translation better preserves numbers, dates, currencies, percentages, units, and factual details? |
| fluency_idiomaticity | -0.092 | [-0.184, 0.000] | 21.7% | 30.8% | 47.5% | Which translation reads more naturally and idiomatically in English? |
| terminology_precision | -0.109 | [-0.176, -0.050] | 10.8% | 21.7% | 67.5% | Which translation uses more domain-appropriate terminology? |

## Outputs

- by task/dimension: `outputs/analysis/task_rubric_feature_coding/utility__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash/rubric_feature_analysis_by_task_dimension.csv`
- by actor/task/dimension: `outputs/analysis/task_rubric_feature_coding/utility__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash/rubric_feature_analysis_by_actor_task_dimension.csv`
- flat codes: `outputs/analysis/task_rubric_feature_coding/utility__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash/rubric_feature_analysis_flat_codes.csv`