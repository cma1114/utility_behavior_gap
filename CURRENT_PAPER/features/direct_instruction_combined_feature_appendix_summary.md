# Direct Instruction Feature Appendix

Comparison: direct_instruction.

The table combines two feature families:

- Generic text features are computed on the full paired dataset for this contrast.
- LLM rubric markers are task-specific comparative codes from the random actor-balanced A/B-coded sample.

Editable feature labels, definitions, and display rounding come from `/Users/christopherackerman/repos/Utility-Behavior-Gap/analysis_specs/feature_definitions.yaml`.

The generic text rows use the standard paper-facing feature set: Words, Paragraphs, Unique-word ratio, Quantitative detail, Flesch-Kincaid grade, Positive-word rate per 1k words, Negative-word rate per 1k words. Quantitative detail is `z(numbers + percentages)`, standardized within task before paired differencing.

Positive deltas mean the direct high output had more of the generic feature or was favored by the rubric coder on the task-specific marker. Confidence intervals are over actor/task cells for the generic full-sample rows and over actors within task for the rubric rows.

`Panel preference (r)` is the within-comparison correlation between left-minus-right feature score and left-minus-right panel score. Positive values mean the judging panel tended to prefer the side with more of that feature; negative values mean the panel tended to prefer the side with less.

- total rows: `55`
- clear-difference rows: `39`

## Clear Differences Only

### Essay writing

| Feature | Delta | 95% CI | Panel preference (r) | % direct high higher/better | % direct low higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Words | 17 | [9, 26] | +0.36 (n=2,100) | 67.1% | 31.7% | Word count. |
| Positive-word rate per 1k words | -2.87 | [-3.92, -2.01] | -0.11 (n=2,100) | 43.6% | 56.4% | VADER positive-lexicon tokens per 1,000 words. |
| Quantitative detail | 0.053 | [0.016, 0.109] |  | 27.0% | 22.6% | Within-task z-score of numeric tokens plus percentage expressions. Computed separately for each output before paired differencing. |
| Unique-word ratio | -0.0042 | [-0.0079, -0.0004] | +0.04 (n=2,100) | 46.2% | 53.8% | Unique lowercased word tokens divided by total word tokens. |
| Argument depth | 0.45 | [0.26, 0.62] | +0.24 (n=120) | 71.7% | 26.7% | Which essay gives better causal, economic, ethical, or institutional reasoning rather than generic assertion? |
| Rhetorical coherence and closure | 0.34 | [0.19, 0.50] | +0.27 (n=120) | 57.5% | 23.3% | Which essay has better paragraph flow, transitions, and final synthesis? |
| Thesis and stakes framing | 0.25 | [0.04, 0.47] | +0.24 (n=120) | 56.7% | 30.8% | Which essay presents the central claim and its stakes more clearly and compellingly? |
| Concrete example quality | 0.25 | [0.02, 0.45] | +0.26 (n=120) | 60.8% | 35.8% | Which essay uses the required concrete example more specifically, plausibly, and usefully? |
| Counterargument/qualification | 0.15 | [0.01, 0.29] | +0.40 (n=120) | 41.7% | 26.7% | Which essay better anticipates objections or tradeoffs and responds to them? |

### Grant abstract

| Feature | Delta | 95% CI | Panel preference (r) | % direct high higher/better | % direct low higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Words | 14 | [4, 27] | +0.30 (n=2,095) | 59.6% | 39.6% | Word count. |
| Negative-word rate per 1k words | 1.06 | [0.62, 1.63] | +0.03 (n=2,095) | 55.6% | 44.2% | VADER negative-lexicon tokens per 1,000 words. |
| Flesch-Kincaid grade | 0.33 | [0.22, 0.44] | +0.14 (n=2,095) | 60.5% | 39.5% | Flesch-Kincaid grade level. |
| Quantitative detail | 0.168 | [0.050, 0.314] |  | 48.1% | 34.3% | Within-task z-score of numeric tokens plus percentage expressions. Computed separately for each output before paired differencing. |
| Unique-word ratio | 0.0076 | [0.0057, 0.0096] | +0.10 (n=2,095) | 59.2% | 40.8% | Unique lowercased word tokens divided by total word tokens. |
| Intervention specificity | 0.57 | [0.44, 0.68] | +0.32 (n=120) | 78.3% | 21.7% | Which abstract gives a more concrete account of what will be built, delivered, tested, or changed? |
| Evaluation rigor | 0.55 | [0.40, 0.70] | +0.32 (n=120) | 77.5% | 22.5% | Which abstract describes stronger outcomes, design, measurement, comparison/control logic, and analysis plan? |
| Risk mitigation | 0.53 | [0.34, 0.70] | +0.40 (n=120) | 75.8% | 23.3% | Which abstract identifies more realistic risks and more credible mitigation plans? |
| Feasibility/readiness | 0.50 | [0.38, 0.64] | +0.41 (n=120) | 71.7% | 21.7% | Which abstract gives more credible evidence that the project can actually be executed, including partners, team capacity, timeline, infrastructure, or prior pilots? |
| Stakeholder/context fit | 0.50 | [0.39, 0.61] | +0.32 (n=120) | 65.0% | 15.0% | Which abstract better fits the affected community or deployment setting? |
| Measurable impact | 0.48 | [0.38, 0.59] | +0.39 (n=120) | 70.8% | 22.5% | Which abstract gives more concrete and credible expected impacts, not just larger or more dramatic claims? |
| Problem significance | 0.47 | [0.33, 0.61] | +0.28 (n=120) | 61.7% | 15.0% | Which abstract more clearly states the need, affected population, and consequence of the problem? |

### Incident postmortem

| Feature | Delta | 95% CI | Panel preference (r) | % direct high higher/better | % direct low higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Words | 66 | [26, 107] | +0.37 (n=2,100) | 70.9% | 28.8% | Word count. |
| Paragraphs | 1.1 | [0.1, 2.4] | +0.25 (n=2,100) | 50.5% | 31.1% | Paragraph count split on blank lines. |
| Negative-word rate per 1k words | 0.91 | [0.47, 1.28] | -0.02 (n=2,100) | 54.0% | 46.0% | VADER negative-lexicon tokens per 1,000 words. |
| Positive-word rate per 1k words | 0.63 | [0.04, 1.31] | -0.05 (n=2,100) | 53.8% | 46.2% | VADER positive-lexicon tokens per 1,000 words. |
| Quantitative detail | 0.261 | [0.128, 0.439] |  | 61.7% | 35.2% | Within-task z-score of numeric tokens plus percentage expressions. Computed separately for each output before paired differencing. |
| Action-item concreteness | 0.51 | [0.39, 0.65] | +0.38 (n=119) | 75.6% | 24.4% | Which postmortem gives more concrete, owned, prioritized, and recurrence-preventing action items? |
| Impact specificity | 0.48 | [0.31, 0.63] | +0.34 (n=119) | 73.9% | 26.1% | Which postmortem quantifies scope, duration, affected systems/users, severity, and business/customer impact better? |
| Contributing-factor analysis | 0.46 | [0.31, 0.60] | +0.29 (n=119) | 73.1% | 26.9% | Which postmortem better separates root cause from contributing factors such as monitoring, process, deployment, review, or testing gaps? |
| Operational realism | 0.45 | [0.30, 0.59] | +0.41 (n=119) | 69.7% | 25.2% | Which postmortem sounds more like a real engineering postmortem rather than a polished but generic template? |
| Root-cause specificity | 0.43 | [0.28, 0.55] | +0.34 (n=119) | 71.4% | 28.6% | Which postmortem identifies the proximate technical failure and the deeper system failure more precisely? |
| Detection/observability analysis | 0.42 | [0.24, 0.60] | +0.37 (n=119) | 70.6% | 28.6% | Which postmortem better explains why the incident was not detected earlier or why alerts/dashboards failed? |
| Timeline precision | 0.35 | [0.18, 0.52] | +0.31 (n=119) | 67.2% | 31.9% | Which postmortem gives a clearer chronological account with useful timestamps and event sequencing? |
| Blameless systems framing | 0.13 | [0.03, 0.24] | +0.33 (n=119) | 17.6% | 4.2% | Which postmortem avoids individual blame while still giving a candid causal account? |

### Translation

| Feature | Delta | 95% CI | Panel preference (r) | % direct high higher/better | % direct low higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Words | 0.5 | [0.2, 0.8] | +0.05 (n=3,150) | 45.9% | 37.0% | Word count. |
| Negative-word rate per 1k words | 0.29 | [0.14, 0.45] | -0.02 (n=3,150) | 17.6% | 17.4% | VADER negative-lexicon tokens per 1,000 words. |
| Quantitative detail | -0.005 | [-0.009, -0.001] |  | 1.5% | 2.3% | Within-task z-score of numeric tokens plus percentage expressions. Computed separately for each output before paired differencing. |
| Fluency/idiomaticity | 0.21 | [0.12, 0.29] | +0.43 (n=120) | 44.2% | 23.3% | Which translation reads more naturally and idiomatically in English? |
| Structural clarity | 0.12 | [0.04, 0.20] | +0.23 (n=120) | 17.5% | 5.8% | Which translation more clearly handles clauses, lists, appositives, parentheticals, quotation, and sentence breaks while preserving source relations? |

## Output Files

- full CSV: `outputs/analysis/direct_instruction_combined_feature_appendix_all.csv`
- clear-differences CSV: `outputs/analysis/direct_instruction_combined_feature_appendix_clear_differences.csv`
- clear-differences Markdown: `outputs/analysis/direct_instruction_combined_feature_appendix_clear_differences.md`
- LaTeX longtable: `outputs/analysis/direct_instruction_combined_feature_appendix_clear_differences.tex`

The full CSV includes all rows, including features whose confidence interval includes zero.