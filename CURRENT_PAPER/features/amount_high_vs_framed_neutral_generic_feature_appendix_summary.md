# Amount High vs Framed Neutral Generic Feature Appendix

Comparison: amount_high_vs_framed_neutral.

The table combines the available feature families:

- Generic text features are computed on the full paired dataset for this contrast.
- LLM rubric markers are task-specific comparative codes from the random actor-balanced A/B-coded sample when that coding exists.

Editable feature labels, definitions, and display rounding come from `/Users/christopherackerman/repos/Utility-Behavior-Gap/analysis_specs/feature_definitions.yaml`.

The generic text rows use the standard paper-facing feature set: Words, Paragraphs, Unique-word ratio, Quantitative detail, Flesch-Kincaid grade, Positive-word rate per 1k words, Negative-word rate per 1k words. Quantitative detail is `z(numbers + percentages)`, standardized within task before paired differencing.

Positive deltas mean the amount high output had more of the generic feature or was favored by the rubric coder on the task-specific marker. Confidence intervals are over actor/task cells for the generic full-sample rows and over actors within task for the rubric rows.

`Panel preference (r)` is the within-comparison correlation between left-minus-right feature score and left-minus-right panel score. Positive values mean the judging panel tended to prefer the side with more of that feature; negative values mean the panel tended to prefer the side with less.

- total rows: `28`
- clear-difference rows: `7`

## Clear Differences Only

### Grant abstract

| Feature | Delta | 95% CI | Panel preference (r) | % amount high higher/better | % framed neutral higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Negative-word rate per 1k words | 0.33 | [0.05, 0.62] | +0.00 (n=2,100) | 51.6% | 48.3% | VADER negative-lexicon tokens per 1,000 words. |
| Flesch-Kincaid grade | 0.13 | [0.02, 0.25] | +0.12 (n=2,100) | 53.6% | 46.4% | Flesch-Kincaid grade level. |
| Paragraphs | 0.1 | [0.0, 0.2] | +0.10 (n=2,100) | 29.3% | 24.7% | Paragraph count split on blank lines. |
| Unique-word ratio | 0.0049 | [0.0029, 0.0072] | +0.13 (n=2,100) | 55.6% | 44.4% | Unique lowercased word tokens divided by total word tokens. |

### Incident postmortem

| Feature | Delta | 95% CI | Panel preference (r) | % amount high higher/better | % framed neutral higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Quantitative detail | 0.067 | [0.010, 0.123] | +0.40 (n=2,100) | 51.8% | 44.5% | Within-task z-score of numeric tokens plus percentage expressions. Computed separately for each output before paired differencing. |
| Unique-word ratio | 0.0016 | [0.0001, 0.0031] | -0.12 (n=2,100) | 53.4% | 46.5% | Unique lowercased word tokens divided by total word tokens. |

### Translation

| Feature | Delta | 95% CI | Panel preference (r) | % amount high higher/better | % framed neutral higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Quantitative detail | 0.003 | [0.000, 0.006] | +0.03 (n=2,100) | 2.0% | 1.4% | Within-task z-score of numeric tokens plus percentage expressions. Computed separately for each output before paired differencing. |

## Output Files

- full CSV: `outputs/analysis/amount_high_vs_framed_neutral_generic_feature_appendix_all.csv`
- clear-differences CSV: `outputs/analysis/amount_high_vs_framed_neutral_generic_feature_appendix_clear_differences.csv`
- clear-differences Markdown: `outputs/analysis/amount_high_vs_framed_neutral_generic_feature_appendix_clear_differences.md`
- LaTeX longtable: `outputs/analysis/amount_high_vs_framed_neutral_generic_feature_appendix_clear_differences.tex`

The full CSV includes all rows, including features whose confidence interval includes zero.