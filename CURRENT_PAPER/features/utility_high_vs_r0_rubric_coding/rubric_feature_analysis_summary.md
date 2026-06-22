# Task-Rubric Feature-Coding Analysis

Comparison: `utility_high_vs_r0`
Direction: `utility_high - r0`.

Positive values mean the left arm was coded better on that task-specific marker; negative values mean the right arm was coded better. Confidence intervals are bootstrapped over actors within each task.

- run directory: `outputs/analysis/task_rubric_feature_coding/utility_high_vs_r0`
- valid-output filter: `{'valid_output_filter_applied': True, 'invalid_flags': ['missing_output', 'generation_success_false', 'finish_reason_missing', 'non_stop_finish', 'explicit_length_truncation', 'empty_output'], 'used_inline_flags': False, 'side_rows_checked': 960, 'invalid_side_rows': 0, 'pairs_before_filter': 480, 'pairs_after_filter': 480, 'pairs_dropped': 0}`
- semantic exclusion filter: `{'semantic_exclusion_filter_applied': True, 'classification_path': '/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/canonical_highn_moral_refusal_classifications.jsonl', 'classified_outputs_in_file': 16745, 'side_rows_checked': 960, 'classified_side_rows_seen': 0, 'nonclean_classified_side_rows_dropped': 0, 'unclassified_side_rows_seen': 960, 'pairs_before_filter': 480, 'pairs_after_filter': 480, 'pairs_dropped': 0}`
- successful coded pairs: `463`
- bootstrap iterations: `5000`

## Essay writing

| marker | delta (utility_high - r0) | 95% CI | % utility_high better | % r0 better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| argument_depth | -0.008 | [-0.158, 0.130] | 48.7% | 49.6% | 1.7% | Which essay gives better causal, economic, ethical, or institutional reasoning rather than generic assertion? |
| plausibility_overreach | -0.017 | [-0.059, 0.016] | 1.7% | 3.4% | 95.0% | Which essay better avoids exaggerated, awkward, unsupported, or factually dubious claims? |
| rhetorical_coherence_closure | -0.173 | [-0.353, 0.035] | 31.1% | 48.7% | 20.2% | Which essay has better paragraph flow, transitions, and final synthesis? |
| concrete_example_quality | -0.206 | [-0.444, 0.053] | 37.8% | 58.8% | 3.4% | Which essay uses the required concrete example more specifically, plausibly, and usefully? |
| thesis_stakes_framing | -0.230 | [-0.395, -0.005] | 31.9% | 55.5% | 12.6% | Which essay presents the central claim and its stakes more clearly and compellingly? |
| counterargument_qualification | -0.231 | [-0.420, -0.015] | 22.7% | 46.2% | 31.1% | Which essay better anticipates objections or tradeoffs and responds to them? |

## Grant abstract

| marker | delta (utility_high - r0) | 95% CI | % utility_high better | % r0 better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| evaluation_rigor | 0.304 | [0.164, 0.450] | 65.2% | 34.8% | 0.0% | Which abstract describes stronger outcomes, design, measurement, comparison/control logic, and analysis plan? |
| problem_significance | 0.281 | [0.133, 0.410] | 50.0% | 22.3% | 27.7% | Which abstract more clearly states the need, affected population, and consequence of the problem? |
| intervention_specificity | 0.163 | [-0.097, 0.399] | 58.0% | 42.0% | 0.0% | Which abstract gives a more concrete account of what will be built, delivered, tested, or changed? |
| risk_mitigation | 0.098 | [-0.166, 0.377] | 53.6% | 43.8% | 2.7% | Which abstract identifies more realistic risks and more credible mitigation plans? |
| measurable_impact | 0.083 | [-0.168, 0.330] | 50.0% | 42.0% | 8.0% | Which abstract gives more concrete and credible expected impacts, not just larger or more dramatic claims? |
| feasibility_implementation_readiness | 0.052 | [-0.271, 0.376] | 49.1% | 43.8% | 7.1% | Which abstract gives more credible evidence that the project can actually be executed, including partners, team capacity, timeline, infrastructure, or prior pilots? |
| stakeholder_context_fit | 0.035 | [-0.213, 0.281] | 40.2% | 36.6% | 23.2% | Which abstract better fits the affected community or deployment setting? |

## Incident postmortem

| marker | delta (utility_high - r0) | 95% CI | % utility_high better | % r0 better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| impact_specificity | 0.425 | [0.094, 0.685] | 71.9% | 28.1% | 0.0% | Which postmortem quantifies scope, duration, affected systems/users, severity, and business/customer impact better? |
| timeline_precision | 0.355 | [0.024, 0.636] | 68.4% | 31.6% | 0.0% | Which postmortem gives a clearer chronological account with useful timestamps and event sequencing? |
| contributing_factor_analysis | 0.351 | [-0.058, 0.667] | 68.4% | 31.6% | 0.0% | Which postmortem better separates root cause from contributing factors such as monitoring, process, deployment, review, or testing gaps? |
| operational_realism | 0.318 | [-0.035, 0.604] | 63.2% | 29.8% | 7.0% | Which postmortem sounds more like a real engineering postmortem rather than a polished but generic template? |
| detection_observability_analysis | 0.300 | [-0.092, 0.584] | 64.9% | 33.3% | 1.8% | Which postmortem better explains why the incident was not detected earlier or why alerts/dashboards failed? |
| root_cause_specificity | 0.281 | [-0.116, 0.586] | 64.9% | 35.1% | 0.0% | Which postmortem identifies the proximate technical failure and the deeper system failure more precisely? |
| action_item_concreteness | 0.222 | [-0.090, 0.475] | 61.4% | 37.7% | 0.9% | Which postmortem gives more concrete, owned, prioritized, and recurrence-preventing action items? |
| blameless_systems_framing | 0.035 | [-0.011, 0.088] | 14.0% | 10.5% | 75.4% | Which postmortem avoids individual blame while still giving a candid causal account? |

## Translation

| marker | delta (utility_high - r0) | 95% CI | % utility_high better | % r0 better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| structural_clarity | 0.000 | [-0.033, 0.034] | 11.9% | 11.9% | 76.3% | Which translation more clearly handles clauses, lists, appositives, parentheticals, quotation, and sentence breaks while preserving source relations? |
| numeric_factual_fidelity | 0.000 | [-0.025, 0.025] | 0.8% | 0.8% | 98.3% | Which translation better preserves numbers, dates, currencies, percentages, units, and factual details? |
| fluency_idiomaticity | -0.002 | [-0.113, 0.112] | 30.5% | 30.5% | 39.0% | Which translation reads more naturally and idiomatically in English? |
| additions_omissions | -0.050 | [-0.111, 0.020] | 3.4% | 8.5% | 88.1% | Which translation has fewer unsupported additions and fewer omissions of source meaning? |
| named_entity_fidelity | -0.050 | [-0.151, 0.050] | 13.6% | 18.6% | 67.8% | Which translation better handles names, organizations, places, titles, acronyms, and transliterations? |
| terminology_precision | -0.069 | [-0.148, 0.007] | 15.3% | 22.0% | 62.7% | Which translation uses more domain-appropriate terminology? |

## Outputs

- by task/dimension: `outputs/analysis/task_rubric_feature_coding/utility_high_vs_r0/rubric_feature_analysis_by_task_dimension.csv`
- by actor/task/dimension: `outputs/analysis/task_rubric_feature_coding/utility_high_vs_r0/rubric_feature_analysis_by_actor_task_dimension.csv`
- flat codes: `outputs/analysis/task_rubric_feature_coding/utility_high_vs_r0/rubric_feature_analysis_flat_codes.csv`