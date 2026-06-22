# Task-Rubric Feature-Coding Analysis

Comparison: `framed_user_prompt_role`
Direction: `role_strong - role_weak`.

Positive values mean the left arm was coded better on that task-specific marker; negative values mean the right arm was coded better. Confidence intervals are bootstrapped over actors within each task.

- run directory: `outputs/analysis/task_rubric_feature_coding/framed_user_prompt_role__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash`
- valid-output filter: `{'valid_output_filter_applied': True, 'invalid_flags': ['missing_output', 'generation_success_false', 'finish_reason_missing', 'non_stop_finish', 'explicit_length_truncation', 'empty_output'], 'used_inline_flags': False, 'side_rows_checked': 960, 'invalid_side_rows': 0, 'pairs_before_filter': 480, 'pairs_after_filter': 480, 'pairs_dropped': 0}`
- semantic exclusion filter: `{'semantic_exclusion_filter_applied': True, 'classification_path': '/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/canonical_highn_moral_refusal_classifications.jsonl', 'classified_outputs_in_file': 16745, 'side_rows_checked': 960, 'classified_side_rows_seen': 0, 'nonclean_classified_side_rows_dropped': 0, 'unclassified_side_rows_seen': 960, 'pairs_before_filter': 480, 'pairs_after_filter': 480, 'pairs_dropped': 0}`
- successful coded pairs: `479`
- bootstrap iterations: `5000`

## Essay writing

| marker | delta (role_strong - role_weak) | 95% CI | % role_strong better | % role_weak better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| argument_depth | 0.282 | [0.134, 0.400] | 62.5% | 34.2% | 3.3% | Which essay gives better causal, economic, ethical, or institutional reasoning rather than generic assertion? |
| concrete_example_quality | 0.257 | [0.089, 0.388] | 60.0% | 34.2% | 5.8% | Which essay uses the required concrete example more specifically, plausibly, and usefully? |
| counterargument_qualification | 0.249 | [0.165, 0.328] | 45.0% | 20.0% | 35.0% | Which essay better anticipates objections or tradeoffs and responds to them? |
| rhetorical_coherence_closure | 0.249 | [0.098, 0.387] | 53.3% | 28.3% | 18.3% | Which essay has better paragraph flow, transitions, and final synthesis? |
| thesis_stakes_framing | 0.224 | [-0.006, 0.412] | 55.8% | 33.3% | 10.8% | Which essay presents the central claim and its stakes more clearly and compellingly? |
| plausibility_overreach | 0.000 | [-0.025, 0.025] | 2.5% | 2.5% | 95.0% | Which essay better avoids exaggerated, awkward, unsupported, or factually dubious claims? |

## Grant abstract

| marker | delta (role_strong - role_weak) | 95% CI | % role_strong better | % role_weak better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| measurable_impact | 0.093 | [-0.051, 0.238] | 50.4% | 41.2% | 8.4% | Which abstract gives more concrete and credible expected impacts, not just larger or more dramatic claims? |
| evaluation_rigor | 0.092 | [-0.023, 0.200] | 54.6% | 45.4% | 0.0% | Which abstract describes stronger outcomes, design, measurement, comparison/control logic, and analysis plan? |
| risk_mitigation | 0.091 | [-0.008, 0.198] | 53.8% | 44.5% | 1.7% | Which abstract identifies more realistic risks and more credible mitigation plans? |
| stakeholder_context_fit | 0.074 | [-0.026, 0.183] | 43.7% | 36.1% | 20.2% | Which abstract better fits the affected community or deployment setting? |
| problem_significance | 0.073 | [-0.042, 0.199] | 40.3% | 32.8% | 26.9% | Which abstract more clearly states the need, affected population, and consequence of the problem? |
| intervention_specificity | 0.073 | [-0.035, 0.180] | 53.8% | 46.2% | 0.0% | Which abstract gives a more concrete account of what will be built, delivered, tested, or changed? |
| feasibility_implementation_readiness | 0.058 | [-0.035, 0.158] | 47.1% | 41.2% | 11.8% | Which abstract gives more credible evidence that the project can actually be executed, including partners, team capacity, timeline, infrastructure, or prior pilots? |

## Incident postmortem

| marker | delta (role_strong - role_weak) | 95% CI | % role_strong better | % role_weak better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| timeline_precision | 0.075 | [-0.087, 0.227] | 53.3% | 45.8% | 0.8% | Which postmortem gives a clearer chronological account with useful timestamps and event sequencing? |
| action_item_concreteness | 0.008 | [-0.176, 0.202] | 50.0% | 49.2% | 0.8% | Which postmortem gives more concrete, owned, prioritized, and recurrence-preventing action items? |
| blameless_systems_framing | -0.000 | [-0.042, 0.041] | 5.8% | 5.8% | 88.3% | Which postmortem avoids individual blame while still giving a candid causal account? |
| operational_realism | -0.008 | [-0.143, 0.118] | 44.2% | 45.0% | 10.8% | Which postmortem sounds more like a real engineering postmortem rather than a polished but generic template? |
| contributing_factor_analysis | -0.049 | [-0.218, 0.109] | 47.5% | 52.5% | 0.0% | Which postmortem better separates root cause from contributing factors such as monitoring, process, deployment, review, or testing gaps? |
| detection_observability_analysis | -0.050 | [-0.261, 0.134] | 46.7% | 51.7% | 1.7% | Which postmortem better explains why the incident was not detected earlier or why alerts/dashboards failed? |
| root_cause_specificity | -0.124 | [-0.292, 0.019] | 43.3% | 55.8% | 0.8% | Which postmortem identifies the proximate technical failure and the deeper system failure more precisely? |
| impact_specificity | -0.167 | [-0.302, -0.047] | 41.7% | 58.3% | 0.0% | Which postmortem quantifies scope, duration, affected systems/users, severity, and business/customer impact better? |

## Translation

| marker | delta (role_strong - role_weak) | 95% CI | % role_strong better | % role_weak better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| additions_omissions | 0.026 | [-0.024, 0.076] | 7.5% | 5.0% | 87.5% | Which translation has fewer unsupported additions and fewer omissions of source meaning? |
| named_entity_fidelity | 0.009 | [-0.065, 0.092] | 13.3% | 12.5% | 74.2% | Which translation better handles names, organizations, places, titles, acronyms, and transliterations? |
| structural_clarity | 0.009 | [-0.050, 0.076] | 12.5% | 11.7% | 75.8% | Which translation more clearly handles clauses, lists, appositives, parentheticals, quotation, and sentence breaks while preserving source relations? |
| numeric_factual_fidelity | 0.008 | [0.000, 0.024] | 1.7% | 0.8% | 97.5% | Which translation better preserves numbers, dates, currencies, percentages, units, and factual details? |
| fluency_idiomaticity | 0.001 | [-0.098, 0.111] | 29.2% | 29.2% | 41.7% | Which translation reads more naturally and idiomatically in English? |
| terminology_precision | -0.033 | [-0.074, 0.025] | 16.7% | 20.0% | 63.3% | Which translation uses more domain-appropriate terminology? |

## Outputs

- by task/dimension: `outputs/analysis/task_rubric_feature_coding/framed_user_prompt_role__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash/rubric_feature_analysis_by_task_dimension.csv`
- by actor/task/dimension: `outputs/analysis/task_rubric_feature_coding/framed_user_prompt_role__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash/rubric_feature_analysis_by_actor_task_dimension.csv`
- flat codes: `outputs/analysis/task_rubric_feature_coding/framed_user_prompt_role__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash/rubric_feature_analysis_flat_codes.csv`