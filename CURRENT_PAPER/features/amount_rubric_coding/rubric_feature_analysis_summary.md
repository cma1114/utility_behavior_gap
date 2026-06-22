# Task-Rubric Feature-Coding Analysis

Comparison: `amount`
Direction: `amount_high - amount_low`.

Positive values mean the left arm was coded better on that task-specific marker; negative values mean the right arm was coded better. Confidence intervals are bootstrapped over actors within each task.

- run directory: `outputs/analysis/task_rubric_feature_coding/amount__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash`
- valid-output filter: `{'valid_output_filter_applied': True, 'invalid_flags': ['missing_output', 'generation_success_false', 'finish_reason_missing', 'non_stop_finish', 'explicit_length_truncation', 'empty_output'], 'used_inline_flags': False, 'side_rows_checked': 960, 'invalid_side_rows': 0, 'pairs_before_filter': 480, 'pairs_after_filter': 480, 'pairs_dropped': 0}`
- semantic exclusion filter: `{'semantic_exclusion_filter_applied': True, 'classification_path': '/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/canonical_highn_moral_refusal_classifications.jsonl', 'classified_outputs_in_file': 16745, 'side_rows_checked': 960, 'classified_side_rows_seen': 0, 'nonclean_classified_side_rows_dropped': 0, 'unclassified_side_rows_seen': 960, 'pairs_before_filter': 480, 'pairs_after_filter': 480, 'pairs_dropped': 0}`
- successful coded pairs: `479`
- bootstrap iterations: `5000`

## Essay writing

| marker | delta (amount_high - amount_low) | 95% CI | % amount_high better | % amount_low better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| counterargument_qualification | 0.000 | [-0.059, 0.067] | 25.0% | 25.0% | 50.0% | Which essay better anticipates objections or tradeoffs and responds to them? |
| plausibility_overreach | -0.008 | [-0.025, 0.000] | 0.8% | 1.7% | 97.5% | Which essay better avoids exaggerated, awkward, unsupported, or factually dubious claims? |
| thesis_stakes_framing | -0.083 | [-0.234, 0.102] | 39.2% | 47.5% | 13.3% | Which essay presents the central claim and its stakes more clearly and compellingly? |
| rhetorical_coherence_closure | -0.084 | [-0.193, 0.035] | 32.5% | 40.8% | 26.7% | Which essay has better paragraph flow, transitions, and final synthesis? |
| argument_depth | -0.133 | [-0.233, -0.025] | 41.7% | 55.0% | 3.3% | Which essay gives better causal, economic, ethical, or institutional reasoning rather than generic assertion? |
| concrete_example_quality | -0.142 | [-0.311, 0.009] | 41.7% | 55.8% | 2.5% | Which essay uses the required concrete example more specifically, plausibly, and usefully? |

## Grant abstract

| marker | delta (amount_high - amount_low) | 95% CI | % amount_high better | % amount_low better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| evaluation_rigor | 0.392 | [0.266, 0.513] | 69.2% | 30.0% | 0.8% | Which abstract describes stronger outcomes, design, measurement, comparison/control logic, and analysis plan? |
| risk_mitigation | 0.392 | [0.258, 0.538] | 69.2% | 30.0% | 0.8% | Which abstract identifies more realistic risks and more credible mitigation plans? |
| intervention_specificity | 0.374 | [0.244, 0.525] | 68.3% | 30.8% | 0.8% | Which abstract gives a more concrete account of what will be built, delivered, tested, or changed? |
| problem_significance | 0.367 | [0.249, 0.459] | 55.8% | 19.2% | 25.0% | Which abstract more clearly states the need, affected population, and consequence of the problem? |
| feasibility_implementation_readiness | 0.350 | [0.168, 0.523] | 65.0% | 30.0% | 5.0% | Which abstract gives more credible evidence that the project can actually be executed, including partners, team capacity, timeline, infrastructure, or prior pilots? |
| measurable_impact | 0.341 | [0.193, 0.526] | 63.3% | 29.2% | 7.5% | Which abstract gives more concrete and credible expected impacts, not just larger or more dramatic claims? |
| stakeholder_context_fit | 0.332 | [0.194, 0.447] | 56.7% | 23.3% | 20.0% | Which abstract better fits the affected community or deployment setting? |

## Incident postmortem

| marker | delta (amount_high - amount_low) | 95% CI | % amount_high better | % amount_low better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| impact_specificity | 0.149 | [0.032, 0.230] | 57.5% | 42.5% | 0.0% | Which postmortem quantifies scope, duration, affected systems/users, severity, and business/customer impact better? |
| action_item_concreteness | 0.117 | [-0.077, 0.268] | 55.8% | 44.2% | 0.0% | Which postmortem gives more concrete, owned, prioritized, and recurrence-preventing action items? |
| detection_observability_analysis | 0.100 | [-0.001, 0.167] | 54.2% | 44.2% | 1.7% | Which postmortem better explains why the incident was not detected earlier or why alerts/dashboards failed? |
| operational_realism | 0.074 | [-0.059, 0.158] | 49.2% | 41.7% | 9.2% | Which postmortem sounds more like a real engineering postmortem rather than a polished but generic template? |
| root_cause_specificity | 0.050 | [-0.050, 0.151] | 51.7% | 46.7% | 1.7% | Which postmortem identifies the proximate technical failure and the deeper system failure more precisely? |
| timeline_precision | 0.032 | [-0.126, 0.146] | 51.7% | 48.3% | 0.0% | Which postmortem gives a clearer chronological account with useful timestamps and event sequencing? |
| contributing_factor_analysis | 0.024 | [-0.094, 0.117] | 50.8% | 48.3% | 0.8% | Which postmortem better separates root cause from contributing factors such as monitoring, process, deployment, review, or testing gaps? |
| blameless_systems_framing | 0.009 | [-0.033, 0.068] | 7.5% | 6.7% | 85.8% | Which postmortem avoids individual blame while still giving a candid causal account? |

## Translation

| marker | delta (amount_high - amount_low) | 95% CI | % amount_high better | % amount_low better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| named_entity_fidelity | 0.049 | [-0.008, 0.108] | 13.4% | 8.4% | 78.2% | Which translation better handles names, organizations, places, titles, acronyms, and transliterations? |
| numeric_factual_fidelity | 0.008 | [-0.017, 0.033] | 1.7% | 0.8% | 97.5% | Which translation better preserves numbers, dates, currencies, percentages, units, and factual details? |
| additions_omissions | -0.009 | [-0.051, 0.033] | 7.6% | 8.4% | 84.0% | Which translation has fewer unsupported additions and fewer omissions of source meaning? |
| structural_clarity | -0.108 | [-0.147, -0.066] | 5.9% | 16.8% | 77.3% | Which translation more clearly handles clauses, lists, appositives, parentheticals, quotation, and sentence breaks while preserving source relations? |
| terminology_precision | -0.108 | [-0.167, -0.049] | 10.9% | 21.8% | 67.2% | Which translation uses more domain-appropriate terminology? |
| fluency_idiomaticity | -0.125 | [-0.193, -0.057] | 24.4% | 37.0% | 38.7% | Which translation reads more naturally and idiomatically in English? |

## Outputs

- by task/dimension: `outputs/analysis/task_rubric_feature_coding/amount__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash/rubric_feature_analysis_by_task_dimension.csv`
- by actor/task/dimension: `outputs/analysis/task_rubric_feature_coding/amount__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash/rubric_feature_analysis_by_actor_task_dimension.csv`
- flat codes: `outputs/analysis/task_rubric_feature_coding/amount__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash/rubric_feature_analysis_flat_codes.csv`