# Task-Rubric Feature-Coding Analysis

Comparison: `utility_high_vs_framed_empty`
Direction: `utility_high - framed_empty`.

Positive values mean the left arm was coded better on that task-specific marker; negative values mean the right arm was coded better. Confidence intervals are bootstrapped over actors within each task.

- run directory: `outputs/analysis/task_rubric_feature_coding/utility_high_vs_framed_empty`
- valid-output filter: `{'valid_output_filter_applied': True, 'invalid_flags': ['missing_output', 'generation_success_false', 'finish_reason_missing', 'non_stop_finish', 'explicit_length_truncation', 'empty_output'], 'used_inline_flags': False, 'side_rows_checked': 960, 'invalid_side_rows': 1, 'pairs_before_filter': 480, 'pairs_after_filter': 479, 'pairs_dropped': 1}`
- semantic exclusion filter: `{'semantic_exclusion_filter_applied': True, 'classification_path': '/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/canonical_highn_moral_refusal_classifications.jsonl', 'classified_outputs_in_file': 16745, 'side_rows_checked': 958, 'classified_side_rows_seen': 0, 'nonclean_classified_side_rows_dropped': 0, 'unclassified_side_rows_seen': 958, 'pairs_before_filter': 479, 'pairs_after_filter': 479, 'pairs_dropped': 0}`
- successful coded pairs: `474`
- bootstrap iterations: `5000`

## Essay writing

| marker | delta (utility_high - framed_empty) | 95% CI | % utility_high better | % framed_empty better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| plausibility_overreach | 0.008 | [0.000, 0.025] | 1.7% | 0.8% | 97.5% | Which essay better avoids exaggerated, awkward, unsupported, or factually dubious claims? |
| counterargument_qualification | -0.079 | [-0.335, 0.202] | 28.6% | 37.0% | 34.5% | Which essay better anticipates objections or tradeoffs and responds to them? |
| concrete_example_quality | -0.154 | [-0.403, 0.106] | 39.5% | 55.5% | 5.0% | Which essay uses the required concrete example more specifically, plausibly, and usefully? |
| argument_depth | -0.181 | [-0.412, 0.090] | 39.5% | 58.0% | 2.5% | Which essay gives better causal, economic, ethical, or institutional reasoning rather than generic assertion? |
| rhetorical_coherence_closure | -0.207 | [-0.412, 0.011] | 24.4% | 45.4% | 30.3% | Which essay has better paragraph flow, transitions, and final synthesis? |
| thesis_stakes_framing | -0.212 | [-0.412, 0.005] | 29.4% | 50.4% | 20.2% | Which essay presents the central claim and its stakes more clearly and compellingly? |

## Grant abstract

| marker | delta (utility_high - framed_empty) | 95% CI | % utility_high better | % framed_empty better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| measurable_impact | -0.062 | [-0.218, 0.095] | 42.9% | 48.7% | 8.4% | Which abstract gives more concrete and credible expected impacts, not just larger or more dramatic claims? |
| problem_significance | -0.111 | [-0.202, -0.019] | 25.2% | 36.1% | 38.7% | Which abstract more clearly states the need, affected population, and consequence of the problem? |
| stakeholder_context_fit | -0.128 | [-0.277, 0.022] | 27.7% | 40.3% | 31.9% | Which abstract better fits the affected community or deployment setting? |
| evaluation_rigor | -0.161 | [-0.402, 0.081] | 42.0% | 58.0% | 0.0% | Which abstract describes stronger outcomes, design, measurement, comparison/control logic, and analysis plan? |
| feasibility_implementation_readiness | -0.213 | [-0.345, -0.079] | 36.1% | 57.1% | 6.7% | Which abstract gives more credible evidence that the project can actually be executed, including partners, team capacity, timeline, infrastructure, or prior pilots? |
| risk_mitigation | -0.238 | [-0.412, -0.056] | 37.8% | 61.3% | 0.8% | Which abstract identifies more realistic risks and more credible mitigation plans? |
| intervention_specificity | -0.271 | [-0.464, -0.070] | 36.1% | 63.0% | 0.8% | Which abstract gives a more concrete account of what will be built, delivered, tested, or changed? |

## Incident postmortem

| marker | delta (utility_high - framed_empty) | 95% CI | % utility_high better | % framed_empty better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| impact_specificity | 0.285 | [0.074, 0.496] | 63.9% | 35.3% | 0.8% | Which postmortem quantifies scope, duration, affected systems/users, severity, and business/customer impact better? |
| detection_observability_analysis | 0.190 | [-0.032, 0.411] | 58.8% | 39.5% | 1.7% | Which postmortem better explains why the incident was not detected earlier or why alerts/dashboards failed? |
| contributing_factor_analysis | 0.185 | [-0.016, 0.396] | 58.8% | 40.3% | 0.8% | Which postmortem better separates root cause from contributing factors such as monitoring, process, deployment, review, or testing gaps? |
| operational_realism | 0.185 | [-0.038, 0.403] | 54.6% | 36.1% | 9.2% | Which postmortem sounds more like a real engineering postmortem rather than a polished but generic template? |
| root_cause_specificity | 0.167 | [-0.031, 0.362] | 58.0% | 41.2% | 0.8% | Which postmortem identifies the proximate technical failure and the deeper system failure more precisely? |
| timeline_precision | 0.151 | [-0.030, 0.344] | 57.1% | 42.0% | 0.8% | Which postmortem gives a clearer chronological account with useful timestamps and event sequencing? |
| action_item_concreteness | 0.109 | [-0.076, 0.294] | 55.5% | 44.5% | 0.0% | Which postmortem gives more concrete, owned, prioritized, and recurrence-preventing action items? |
| blameless_systems_framing | -0.039 | [-0.180, 0.084] | 7.6% | 11.8% | 80.7% | Which postmortem avoids individual blame while still giving a candid causal account? |

## Translation

| marker | delta (utility_high - framed_empty) | 95% CI | % utility_high better | % framed_empty better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| named_entity_fidelity | 0.003 | [-0.108, 0.097] | 17.9% | 17.9% | 64.1% | Which translation better handles names, organizations, places, titles, acronyms, and transliterations? |
| additions_omissions | -0.007 | [-0.044, 0.043] | 6.8% | 7.7% | 85.5% | Which translation has fewer unsupported additions and fewer omissions of source meaning? |
| numeric_factual_fidelity | -0.009 | [-0.036, 0.017] | 0.9% | 1.7% | 97.4% | Which translation better preserves numbers, dates, currencies, percentages, units, and factual details? |
| structural_clarity | -0.034 | [-0.093, 0.018] | 6.8% | 10.3% | 82.9% | Which translation more clearly handles clauses, lists, appositives, parentheticals, quotation, and sentence breaks while preserving source relations? |
| fluency_idiomaticity | -0.107 | [-0.208, 0.002] | 20.5% | 30.8% | 48.7% | Which translation reads more naturally and idiomatically in English? |
| terminology_precision | -0.121 | [-0.190, -0.046] | 12.8% | 24.8% | 62.4% | Which translation uses more domain-appropriate terminology? |

## Outputs

- by task/dimension: `outputs/analysis/task_rubric_feature_coding/utility_high_vs_framed_empty/rubric_feature_analysis_by_task_dimension.csv`
- by actor/task/dimension: `outputs/analysis/task_rubric_feature_coding/utility_high_vs_framed_empty/rubric_feature_analysis_by_actor_task_dimension.csv`
- flat codes: `outputs/analysis/task_rubric_feature_coding/utility_high_vs_framed_empty/rubric_feature_analysis_flat_codes.csv`