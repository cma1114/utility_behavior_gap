# Utility High vs Framed Neutral Generic Feature Appendix

Comparison: utility_high_vs_framed_neutral.

The table combines the available feature families:

- Generic text features are computed on the full paired dataset for this contrast.
- LLM rubric markers are task-specific comparative codes from the random actor-balanced A/B-coded sample when that coding exists.

Editable feature labels, definitions, and display rounding come from `/Users/christopherackerman/repos/Utility-Behavior-Gap/analysis_specs/feature_definitions.yaml`.

The generic text rows use the standard paper-facing feature set: Words, Paragraphs, Unique-word ratio, Quantitative detail, Flesch-Kincaid grade, Positive-word rate per 1k words, Negative-word rate per 1k words. Quantitative detail is `z(numbers + percentages)`, standardized within task before paired differencing.

Positive deltas mean the utility high output had more of the generic feature or was favored by the rubric coder on the task-specific marker. Confidence intervals are over actor/task cells for the generic full-sample rows and over actors within task for the rubric rows.

`Panel preference (r)` is the within-comparison correlation between left-minus-right feature score and left-minus-right panel score. Positive values mean the judging panel tended to prefer the side with more of that feature; negative values mean the panel tended to prefer the side with less.

- total rows: `28`
- clear-difference rows: `7`

## Clear Differences Only

### Essay writing

| Feature | Delta | 95% CI | Panel preference (r) | % utility high higher/better | % framed neutral higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Paragraphs | 0.1 | [0.0, 0.2] | +0.13 (n=1,050) | 26.9% | 19.6% | Paragraph count split on blank lines. |

### Grant abstract

| Feature | Delta | 95% CI | Panel preference (r) | % utility high higher/better | % framed neutral higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Positive-word rate per 1k words | -1.12 | [-1.70, -0.45] | -0.15 (n=1,050) | 46.4% | 53.6% | VADER positive-lexicon tokens per 1,000 words. |
| Negative-word rate per 1k words | 0.69 | [0.21, 1.19] | -0.05 (n=1,050) | 52.7% | 47.3% | VADER negative-lexicon tokens per 1,000 words. |
| Paragraphs | 0.3 | [0.0, 0.6] | +0.09 (n=1,050) | 33.0% | 23.0% | Paragraph count split on blank lines. |
| Unique-word ratio | 0.0055 | [0.0038, 0.0070] | +0.07 (n=1,050) | 55.3% | 44.7% | Unique lowercased word tokens divided by total word tokens. |

### Incident postmortem

| Feature | Delta | 95% CI | Panel preference (r) | % utility high higher/better | % framed neutral higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Words | -5.4 | [-11, 0] | +0.36 (n=1,050) | 45.0% | 54.3% | Word count. |
| Negative-word rate per 1k words | 0.62 | [0.23, 1.02] | +0.01 (n=1,050) | 52.9% | 47.1% | VADER negative-lexicon tokens per 1,000 words. |

## Output Files

- full CSV: `outputs/analysis/utility_high_vs_framed_neutral_generic_feature_appendix_all.csv`
- clear-differences CSV: `outputs/analysis/utility_high_vs_framed_neutral_generic_feature_appendix_clear_differences.csv`
- clear-differences Markdown: `outputs/analysis/utility_high_vs_framed_neutral_generic_feature_appendix_clear_differences.md`
- LaTeX longtable: `outputs/analysis/utility_high_vs_framed_neutral_generic_feature_appendix_clear_differences.tex`

The full CSV includes all rows, including features whose confidence interval includes zero.