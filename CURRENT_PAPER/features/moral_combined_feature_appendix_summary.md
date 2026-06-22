# Moral High-Low Feature Appendix

Comparison: moral.

The table combines two feature families:

- Generic text features are computed on the full paired dataset for this contrast.
- LLM rubric markers are task-specific comparative codes from the random actor-balanced A/B-coded sample.

Editable feature labels, definitions, and display rounding come from `/Users/christopherackerman/repos/Utility-Behavior-Gap/analysis_specs/feature_definitions.yaml`.

The generic text rows use the standard paper-facing feature set: Words, Paragraphs, Unique-word ratio, Quantitative detail, Flesch-Kincaid grade, Positive-word rate per 1k words, Negative-word rate per 1k words. Quantitative detail is `z(numbers + percentages)`, standardized within task before paired differencing.

Positive deltas mean the moral high output had more of the generic feature or was favored by the rubric coder on the task-specific marker. Confidence intervals are over actor/task cells for the generic full-sample rows and over actors within task for the rubric rows.

`Panel preference (r)` is the within-comparison correlation between left-minus-right feature score and left-minus-right panel score. Positive values mean the judging panel tended to prefer the side with more of that feature; negative values mean the panel tended to prefer the side with less.

- total rows: `55`
- clear-difference rows: `19`

## Clear Differences Only

### Essay writing

| Feature | Delta | 95% CI | Panel preference (r) | % moral high higher/better | % moral low higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Positive-word rate per 1k words | -1.29 | [-2.33, -0.23] | -0.12 (n=1,979) | 46.7% | 53.2% | VADER positive-lexicon tokens per 1,000 words. |
| Negative-word rate per 1k words | 0.93 | [0.29, 1.55] | +0.05 (n=1,979) | 53.3% | 46.6% | VADER negative-lexicon tokens per 1,000 words. |
| Flesch-Kincaid grade | -0.25 | [-0.39, -0.14] | +0.12 (n=1,979) | 43.3% | 56.6% | Flesch-Kincaid grade level. |
| Paragraphs | -0.2 | [-0.5, 0.0] | +0.14 (n=1,979) | 18.5% | 28.2% | Paragraph count split on blank lines. |
| Counterargument/qualification | 0.11 | [0.07, 0.15] | -0.01 (n=120) | 29.2% | 18.3% | Which essay better anticipates objections or tradeoffs and responds to them? |

### Grant abstract

| Feature | Delta | 95% CI | Panel preference (r) | % moral high higher/better | % moral low higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Words | 21 | [14, 29] | +0.35 (n=1,829) | 63.6% | 36.0% | Word count. |
| Flesch-Kincaid grade | -0.17 | [-0.40, -0.01] | +0.12 (n=1,829) | 47.0% | 53.0% | Flesch-Kincaid grade level. |
| Unique-word ratio | -0.0068 | [-0.0092, -0.0043] | +0.09 (n=1,829) | 41.1% | 58.9% | Unique lowercased word tokens divided by total word tokens. |
| Risk mitigation | 0.30 | [0.13, 0.44] | +0.56 (n=120) | 64.2% | 34.2% | Which abstract identifies more realistic risks and more credible mitigation plans? |
| Measurable impact | 0.25 | [0.16, 0.34] | +0.42 (n=120) | 57.5% | 32.5% | Which abstract gives more concrete and credible expected impacts, not just larger or more dramatic claims? |
| Problem significance | 0.24 | [0.12, 0.39] | +0.43 (n=120) | 50.0% | 25.8% | Which abstract more clearly states the need, affected population, and consequence of the problem? |
| Evaluation rigor | 0.23 | [0.07, 0.36] | +0.47 (n=120) | 61.7% | 38.3% | Which abstract describes stronger outcomes, design, measurement, comparison/control logic, and analysis plan? |
| Stakeholder/context fit | 0.22 | [0.09, 0.35] | +0.52 (n=120) | 50.8% | 29.2% | Which abstract better fits the affected community or deployment setting? |
| Feasibility/readiness | 0.22 | [0.10, 0.34] | +0.52 (n=120) | 57.5% | 35.8% | Which abstract gives more credible evidence that the project can actually be executed, including partners, team capacity, timeline, infrastructure, or prior pilots? |
| Intervention specificity | 0.18 | [0.03, 0.35] | +0.41 (n=120) | 59.2% | 40.8% | Which abstract gives a more concrete account of what will be built, delivered, tested, or changed? |

### Incident postmortem

| Feature | Delta | 95% CI | Panel preference (r) | % moral high higher/better | % moral low higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Words | 22 | [12, 33] | +0.40 (n=1,946) | 59.0% | 40.3% | Word count. |
| Quantitative detail | 0.101 | [0.028, 0.172] |  | 51.7% | 44.5% | Within-task z-score of numeric tokens plus percentage expressions. Computed separately for each output before paired differencing. |
| Unique-word ratio | -0.0034 | [-0.0060, -0.0009] | -0.09 (n=1,946) | 45.3% | 54.7% | Unique lowercased word tokens divided by total word tokens. |

### Translation

| Feature | Delta | 95% CI | Panel preference (r) | % moral high higher/better | % moral low higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Paragraphs | 0.0 | [0.0, 0.0] | -0.06 (n=1,858) | 0.1% | 0.3% | Paragraph count split on blank lines. |

## Output Files

- full CSV: `outputs/analysis/moral_combined_feature_appendix_all.csv`
- clear-differences CSV: `outputs/analysis/moral_combined_feature_appendix_clear_differences.csv`
- clear-differences Markdown: `outputs/analysis/moral_combined_feature_appendix_clear_differences.md`
- LaTeX longtable: `outputs/analysis/moral_combined_feature_appendix_clear_differences.tex`

The full CSV includes all rows, including features whose confidence interval includes zero.