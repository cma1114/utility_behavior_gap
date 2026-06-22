# Utility High vs R0 Feature Appendix

Comparison: utility_high_vs_r0.

The table combines two feature families:

- Generic text features are computed on the full paired dataset for this contrast.
- LLM rubric markers are task-specific comparative codes from the random actor-balanced A/B-coded sample.

Editable feature labels, definitions, and display rounding come from `/Users/christopherackerman/repos/Utility-Behavior-Gap/analysis_specs/feature_definitions.yaml`.

The generic text rows use the standard paper-facing feature set: Words, Paragraphs, Unique-word ratio, Quantitative detail, Flesch-Kincaid grade, Positive-word rate per 1k words, Negative-word rate per 1k words. Quantitative detail is `z(numbers + percentages)`, standardized within task before paired differencing.

Positive deltas mean the utility high output had more of the generic feature or was favored by the rubric coder on the task-specific marker. Confidence intervals are over actor/task cells for the generic full-sample rows and over actors within task for the rubric rows.

`Panel preference (r)` is the within-comparison correlation between left-minus-right feature score and left-minus-right panel score. Positive values mean the judging panel tended to prefer the side with more of that feature; negative values mean the panel tended to prefer the side with less.

- total rows: `55`
- clear-difference rows: `15`

## Clear Differences Only

### Essay writing

| Feature | Delta | 95% CI | Panel preference (r) | % utility high higher/better | % R0 higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Quantitative detail | -0.124 | [-0.220, -0.049] | +0.21 (n=1,050) | 22.6% | 36.4% | Within-task z-score of numeric tokens plus percentage expressions. Computed separately for each output before paired differencing. |
| Counterargument/qualification | -0.23 | [-0.42, -0.01] | +0.27 (n=119) | 22.7% | 46.2% | Which essay better anticipates objections or tradeoffs and responds to them? |
| Thesis and stakes framing | -0.23 | [-0.39, -0.01] | +0.38 (n=119) | 31.9% | 55.5% | Which essay presents the central claim and its stakes more clearly and compellingly? |

### Grant abstract

| Feature | Delta | 95% CI | Panel preference (r) | % utility high higher/better | % R0 higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Words | 25 | [15, 36] | +0.28 (n=1,050) | 66.7% | 32.5% | Word count. |
| Positive-word rate per 1k words | -1.66 | [-2.33, -1.18] | -0.18 (n=1,050) | 44.0% | 55.8% | VADER positive-lexicon tokens per 1,000 words. |
| Negative-word rate per 1k words | 0.88 | [0.22, 1.43] | +0.00 (n=1,050) | 53.8% | 46.1% | VADER negative-lexicon tokens per 1,000 words. |
| Paragraphs | 0.4 | [0.1, 0.6] | +0.15 (n=1,050) | 36.8% | 22.5% | Paragraph count split on blank lines. |
| Unique-word ratio | -0.0079 | [-0.0125, -0.0034] | +0.10 (n=1,050) | 39.9% | 60.0% | Unique lowercased word tokens divided by total word tokens. |
| Evaluation rigor | 0.30 | [0.16, 0.45] | +0.27 (n=112) | 65.2% | 34.8% | Which abstract describes stronger outcomes, design, measurement, comparison/control logic, and analysis plan? |
| Problem significance | 0.28 | [0.13, 0.41] | +0.29 (n=112) | 50.0% | 22.3% | Which abstract more clearly states the need, affected population, and consequence of the problem? |

### Incident postmortem

| Feature | Delta | 95% CI | Panel preference (r) | % utility high higher/better | % R0 higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Positive-word rate per 1k words | -1.11 | [-2.40, -0.08] | -0.08 (n=1,050) | 43.9% | 56.1% | VADER positive-lexicon tokens per 1,000 words. |
| Impact specificity | 0.42 | [0.09, 0.68] | +0.45 (n=114) | 71.9% | 28.1% | Which postmortem quantifies scope, duration, affected systems/users, severity, and business/customer impact better? |
| Timeline precision | 0.35 | [0.02, 0.64] | +0.41 (n=114) | 68.4% | 31.6% | Which postmortem gives a clearer chronological account with useful timestamps and event sequencing? |

### Translation

| Feature | Delta | 95% CI | Panel preference (r) | % utility high higher/better | % R0 higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Positive-word rate per 1k words | -1.32 | [-2.39, -0.11] | -0.03 (n=1,050) | 29.5% | 39.9% | VADER positive-lexicon tokens per 1,000 words. |
| Flesch-Kincaid grade | -0.34 | [-0.62, -0.04] | -0.03 (n=1,050) | 44.0% | 50.8% | Flesch-Kincaid grade level. |

## Output Files

- full CSV: `outputs/analysis/utility_high_vs_r0_combined_feature_appendix_all.csv`
- clear-differences CSV: `outputs/analysis/utility_high_vs_r0_combined_feature_appendix_clear_differences.csv`
- clear-differences Markdown: `outputs/analysis/utility_high_vs_r0_combined_feature_appendix_clear_differences.md`
- LaTeX longtable: `outputs/analysis/utility_high_vs_r0_combined_feature_appendix_clear_differences.tex`

The full CSV includes all rows, including features whose confidence interval includes zero.