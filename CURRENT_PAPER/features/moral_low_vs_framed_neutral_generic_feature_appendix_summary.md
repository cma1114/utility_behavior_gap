# Moral Low vs Framed Neutral Generic Feature Appendix

Comparison: moral_low_vs_framed_neutral.

The table combines the available feature families:

- Generic text features are computed on the full paired dataset for this contrast.
- LLM rubric markers are task-specific comparative codes from the random actor-balanced A/B-coded sample when that coding exists.

Editable feature labels, definitions, and display rounding come from `/Users/christopherackerman/repos/Utility-Behavior-Gap/analysis_specs/feature_definitions.yaml`.

The generic text rows use the standard paper-facing feature set: Words, Paragraphs, Unique-word ratio, Quantitative detail, Flesch-Kincaid grade, Positive-word rate per 1k words, Negative-word rate per 1k words. Quantitative detail is `z(numbers + percentages)`, standardized within task before paired differencing.

Positive deltas mean the moral low output had more of the generic feature or was favored by the rubric coder on the task-specific marker. Confidence intervals are over actor/task cells for the generic full-sample rows and over actors within task for the rubric rows.

`Panel preference (r)` is the within-comparison correlation between left-minus-right feature score and left-minus-right panel score. Positive values mean the judging panel tended to prefer the side with more of that feature; negative values mean the panel tended to prefer the side with less.

- total rows: `28`
- clear-difference rows: `14`

## Clear Differences Only

### Essay writing

| Feature | Delta | 95% CI | Panel preference (r) | % moral low higher/better | % framed neutral higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Words | -9.9 | [-19, -2] | +0.30 (n=1,518) | 38.2% | 60.7% | Word count. |
| Positive-word rate per 1k words | 2.28 | [1.21, 3.51] | -0.12 (n=1,518) | 56.0% | 44.0% | VADER positive-lexicon tokens per 1,000 words. |
| Negative-word rate per 1k words | -1.80 | [-2.64, -1.03] | +0.04 (n=1,518) | 43.3% | 56.6% | VADER negative-lexicon tokens per 1,000 words. |
| Flesch-Kincaid grade | 0.38 | [0.24, 0.57] | +0.13 (n=1,518) | 63.0% | 37.0% | Flesch-Kincaid grade level. |
| Paragraphs | 0.2 | [0.0, 0.5] | +0.14 (n=1,518) | 30.7% | 18.1% | Paragraph count split on blank lines. |

### Grant abstract

| Feature | Delta | 95% CI | Panel preference (r) | % moral low higher/better | % framed neutral higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Words | -31 | [-47, -18] | +0.24 (n=947) | 28.6% | 70.7% | Word count. |
| Positive-word rate per 1k words | -0.97 | [-1.61, -0.29] | -0.14 (n=947) | 46.7% | 53.1% | VADER positive-lexicon tokens per 1,000 words. |
| Paragraphs | 0.5 | [0.0, 1.0] | +0.18 (n=947) | 33.8% | 21.8% | Paragraph count split on blank lines. |
| Unique-word ratio | 0.0119 | [0.0074, 0.0165] | +0.13 (n=947) | 65.3% | 34.6% | Unique lowercased word tokens divided by total word tokens. |

### Incident postmortem

| Feature | Delta | 95% CI | Panel preference (r) | % moral low higher/better | % framed neutral higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Words | -23 | [-30, -16] | +0.39 (n=1,065) | 39.7% | 59.8% | Word count. |
| Paragraphs | -0.6 | [-1.2, -0.1] | +0.26 (n=1,065) | 33.1% | 47.2% | Paragraph count split on blank lines. |
| Unique-word ratio | 0.0061 | [0.0031, 0.0094] | -0.11 (n=1,065) | 58.3% | 41.7% | Unique lowercased word tokens divided by total word tokens. |

### Translation

| Feature | Delta | 95% CI | Panel preference (r) | % moral low higher/better | % framed neutral higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Words | 0.2 | [0.1, 0.4] | -0.03 (n=1,239) | 40.7% | 40.4% | Word count. |
| Paragraphs | 0.0 | [0.0, 0.0] | -0.08 (n=1,239) | 0.4% | 0.1% | Paragraph count split on blank lines. |

## Output Files

- full CSV: `outputs/analysis/moral_low_vs_framed_neutral_generic_feature_appendix_all.csv`
- clear-differences CSV: `outputs/analysis/moral_low_vs_framed_neutral_generic_feature_appendix_clear_differences.csv`
- clear-differences Markdown: `outputs/analysis/moral_low_vs_framed_neutral_generic_feature_appendix_clear_differences.md`
- LaTeX longtable: `outputs/analysis/moral_low_vs_framed_neutral_generic_feature_appendix_clear_differences.tex`

The full CSV includes all rows, including features whose confidence interval includes zero.