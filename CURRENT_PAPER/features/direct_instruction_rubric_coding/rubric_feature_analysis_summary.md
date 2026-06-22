# Task-Rubric Feature-Coding Analysis

Comparison: `direct_instruction`
Direction: `direct_high - direct_low`.

Positive values mean the left arm was coded better on that task-specific marker; negative values mean the right arm was coded better. Confidence intervals are bootstrapped over actors within each task.

- run directory: `outputs/analysis/task_rubric_feature_coding/direct_instruction__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash`
- successful coded pairs: `479`
- bootstrap iterations: `5000`

## Essay writing

| marker | delta (direct_high - direct_low) | 95% CI | % direct_high better | % direct_low better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| argument_depth | 0.447 | [0.262, 0.617] | 71.7% | 26.7% | 1.7% | Which essay gives better causal, economic, ethical, or institutional reasoning rather than generic assertion? |
| rhetorical_coherence_closure | 0.339 | [0.193, 0.501] | 57.5% | 23.3% | 19.2% | Which essay has better paragraph flow, transitions, and final synthesis? |
| thesis_stakes_framing | 0.255 | [0.042, 0.473] | 56.7% | 30.8% | 12.5% | Which essay presents the central claim and its stakes more clearly and compellingly? |
| concrete_example_quality | 0.248 | [0.017, 0.455] | 60.8% | 35.8% | 3.3% | Which essay uses the required concrete example more specifically, plausibly, and usefully? |
| counterargument_qualification | 0.148 | [0.014, 0.294] | 41.7% | 26.7% | 31.7% | Which essay better anticipates objections or tradeoffs and responds to them? |
| plausibility_overreach | -0.008 | [-0.025, 0.000] | 0.0% | 0.8% | 99.2% | Which essay better avoids exaggerated, awkward, unsupported, or factually dubious claims? |

## Grant abstract

| marker | delta (direct_high - direct_low) | 95% CI | % direct_high better | % direct_low better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| intervention_specificity | 0.570 | [0.442, 0.681] | 78.3% | 21.7% | 0.0% | Which abstract gives a more concrete account of what will be built, delivered, tested, or changed? |
| evaluation_rigor | 0.553 | [0.401, 0.697] | 77.5% | 22.5% | 0.0% | Which abstract describes stronger outcomes, design, measurement, comparison/control logic, and analysis plan? |
| risk_mitigation | 0.528 | [0.343, 0.697] | 75.8% | 23.3% | 0.8% | Which abstract identifies more realistic risks and more credible mitigation plans? |
| feasibility_implementation_readiness | 0.501 | [0.375, 0.639] | 71.7% | 21.7% | 6.7% | Which abstract gives more credible evidence that the project can actually be executed, including partners, team capacity, timeline, infrastructure, or prior pilots? |
| stakeholder_context_fit | 0.500 | [0.395, 0.610] | 65.0% | 15.0% | 20.0% | Which abstract better fits the affected community or deployment setting? |
| measurable_impact | 0.485 | [0.378, 0.588] | 70.8% | 22.5% | 6.7% | Which abstract gives more concrete and credible expected impacts, not just larger or more dramatic claims? |
| problem_significance | 0.468 | [0.325, 0.611] | 61.7% | 15.0% | 23.3% | Which abstract more clearly states the need, affected population, and consequence of the problem? |

## Incident postmortem

| marker | delta (direct_high - direct_low) | 95% CI | % direct_high better | % direct_low better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| action_item_concreteness | 0.513 | [0.395, 0.647] | 75.6% | 24.4% | 0.0% | Which postmortem gives more concrete, owned, prioritized, and recurrence-preventing action items? |
| impact_specificity | 0.479 | [0.311, 0.630] | 73.9% | 26.1% | 0.0% | Which postmortem quantifies scope, duration, affected systems/users, severity, and business/customer impact better? |
| contributing_factor_analysis | 0.462 | [0.311, 0.597] | 73.1% | 26.9% | 0.0% | Which postmortem better separates root cause from contributing factors such as monitoring, process, deployment, review, or testing gaps? |
| operational_realism | 0.445 | [0.303, 0.588] | 69.7% | 25.2% | 5.0% | Which postmortem sounds more like a real engineering postmortem rather than a polished but generic template? |
| root_cause_specificity | 0.429 | [0.277, 0.546] | 71.4% | 28.6% | 0.0% | Which postmortem identifies the proximate technical failure and the deeper system failure more precisely? |
| detection_observability_analysis | 0.420 | [0.235, 0.597] | 70.6% | 28.6% | 0.8% | Which postmortem better explains why the incident was not detected earlier or why alerts/dashboards failed? |
| timeline_precision | 0.353 | [0.185, 0.521] | 67.2% | 31.9% | 0.8% | Which postmortem gives a clearer chronological account with useful timestamps and event sequencing? |
| blameless_systems_framing | 0.134 | [0.034, 0.235] | 17.6% | 4.2% | 78.2% | Which postmortem avoids individual blame while still giving a candid causal account? |

## Translation

| marker | delta (direct_high - direct_low) | 95% CI | % direct_high better | % direct_low better | % tie/NA | description |
| --- | --- | --- | --- | --- | --- | --- |
| fluency_idiomaticity | 0.208 | [0.118, 0.294] | 44.2% | 23.3% | 32.5% | Which translation reads more naturally and idiomatically in English? |
| structural_clarity | 0.118 | [0.042, 0.202] | 17.5% | 5.8% | 76.7% | Which translation more clearly handles clauses, lists, appositives, parentheticals, quotation, and sentence breaks while preserving source relations? |
| terminology_precision | 0.102 | [-0.024, 0.227] | 23.3% | 13.3% | 63.3% | Which translation uses more domain-appropriate terminology? |
| named_entity_fidelity | 0.034 | [-0.066, 0.160] | 22.5% | 19.2% | 58.3% | Which translation better handles names, organizations, places, titles, acronyms, and transliterations? |
| numeric_factual_fidelity | 0.008 | [0.000, 0.025] | 1.7% | 0.8% | 97.5% | Which translation better preserves numbers, dates, currencies, percentages, units, and factual details? |
| additions_omissions | -0.007 | [-0.055, 0.034] | 5.0% | 5.8% | 89.2% | Which translation has fewer unsupported additions and fewer omissions of source meaning? |

## Outputs

- by task/dimension: `outputs/analysis/task_rubric_feature_coding/direct_instruction__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash/rubric_feature_analysis_by_task_dimension.csv`
- by actor/task/dimension: `outputs/analysis/task_rubric_feature_coding/direct_instruction__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash/rubric_feature_analysis_by_actor_task_dimension.csv`
- flat codes: `outputs/analysis/task_rubric_feature_coding/direct_instruction__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash/rubric_feature_analysis_flat_codes.csv`