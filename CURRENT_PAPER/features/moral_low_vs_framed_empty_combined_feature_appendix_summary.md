# Moral Low vs Framed Empty Feature Appendix

Comparison: moral_low_vs_framed_empty.

The table combines two feature families:

- Generic text features are computed on the full paired dataset for this contrast.
- LLM rubric markers are task-specific comparative codes from the random actor-balanced A/B-coded sample.

Editable feature labels, definitions, and display rounding come from `/Users/christopherackerman/repos/Utility-Behavior-Gap/analysis_specs/feature_definitions.yaml`.

The generic text rows use the standard paper-facing feature set: Words, Paragraphs, Unique-word ratio, Quantitative detail, Flesch-Kincaid grade, Positive-word rate per 1k words, Negative-word rate per 1k words. Quantitative detail is `z(numbers + percentages)`, standardized within task before paired differencing.

Positive deltas mean the moral low output had more of the generic feature or was favored by the rubric coder on the task-specific marker. Confidence intervals are over actor/task cells for the generic full-sample rows and over actors within task for the rubric rows.

`Panel preference (r)` is the within-comparison correlation between left-minus-right feature score and left-minus-right panel score. Positive values mean the judging panel tended to prefer the side with more of that feature; negative values mean the panel tended to prefer the side with less.

- total rows: `55`
- clear-difference rows: `23`

## Clear Differences Only

### Essay writing

| Feature | Delta | 95% CI | Panel preference (r) | % moral low higher/better | % framed empty higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Words | -15 | [-30, -3] | +0.33 (n=1,979) | 34.3% | 64.6% | Word count. |
| Positive-word rate per 1k words | 1.60 | [0.99, 2.32] | -0.07 (n=1,979) | 54.2% | 45.7% | VADER positive-lexicon tokens per 1,000 words. |
| Negative-word rate per 1k words | -1.23 | [-1.95, -0.57] | +0.01 (n=1,979) | 44.1% | 55.7% | VADER negative-lexicon tokens per 1,000 words. |
| Flesch-Kincaid grade | 0.36 | [0.19, 0.51] | +0.11 (n=1,979) | 62.3% | 37.7% | Flesch-Kincaid grade level. |
| Thesis and stakes framing | -0.37 | [-0.55, -0.20] | +0.27 (n=120) | 27.5% | 64.2% | Which essay presents the central claim and its stakes more clearly and compellingly? |
| Argument depth | -0.31 | [-0.55, -0.08] | +0.50 (n=120) | 32.5% | 64.2% | Which essay gives better causal, economic, ethical, or institutional reasoning rather than generic assertion? |
| Rhetorical coherence and closure | -0.28 | [-0.55, -0.06] | +0.42 (n=120) | 27.5% | 55.8% | Which essay has better paragraph flow, transitions, and final synthesis? |

### Grant abstract

| Feature | Delta | 95% CI | Panel preference (r) | % moral low higher/better | % framed empty higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Words | -18 | [-34, -4] | +0.21 (n=1,831) | 35.9% | 63.4% | Word count. |
| Negative-word rate per 1k words | 0.77 | [0.24, 1.32] | +0.01 (n=1,831) | 54.3% | 45.4% | VADER negative-lexicon tokens per 1,000 words. |
| Paragraphs | 0.6 | [0.1, 1.3] | +0.11 (n=1,831) | 36.9% | 20.5% | Paragraph count split on blank lines. |
| Flesch-Kincaid grade | 0.33 | [0.07, 0.65] | +0.12 (n=1,831) | 57.3% | 42.7% | Flesch-Kincaid grade level. |
| Measurable impact | -0.33 | [-0.53, -0.10] | +0.46 (n=118) | 29.7% | 61.9% | Which abstract gives more concrete and credible expected impacts, not just larger or more dramatic claims? |
| Risk mitigation | -0.32 | [-0.55, -0.01] | +0.50 (n=118) | 33.1% | 65.3% | Which abstract identifies more realistic risks and more credible mitigation plans? |
| Stakeholder/context fit | -0.32 | [-0.53, -0.04] | +0.42 (n=118) | 26.3% | 57.6% | Which abstract better fits the affected community or deployment setting? |
| Intervention specificity | -0.30 | [-0.50, -0.07] | +0.40 (n=118) | 34.7% | 64.4% | Which abstract gives a more concrete account of what will be built, delivered, tested, or changed? |
| Evaluation rigor | -0.27 | [-0.44, -0.09] | +0.32 (n=118) | 36.4% | 63.6% | Which abstract describes stronger outcomes, design, measurement, comparison/control logic, and analysis plan? |

### Incident postmortem

| Feature | Delta | 95% CI | Panel preference (r) | % moral low higher/better | % framed empty higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Words | -25 | [-40, -13] | +0.32 (n=1,944) | 37.7% | 61.9% | Word count. |
| Unique-word ratio | 0.0083 | [0.0040, 0.0124] | -0.10 (n=1,944) | 60.2% | 39.8% | Unique lowercased word tokens divided by total word tokens. |

### Translation

| Feature | Delta | 95% CI | Panel preference (r) | % moral low higher/better | % framed empty higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Negative-word rate per 1k words | -0.48 | [-0.81, -0.14] | -0.01 (n=1,933) | 14.8% | 16.2% | VADER negative-lexicon tokens per 1,000 words. |
| Words | 0.2 | [0.1, 0.4] | +0.00 (n=1,933) | 41.9% | 38.3% | Word count. |
| Paragraphs | 0.0 | [0.0, 0.0] | -0.04 (n=1,933) | 0.4% | 0.2% | Paragraph count split on blank lines. |
| Quantitative detail | -0.004 | [-0.008, -0.001] | -0.04 (n=1,933) | 1.7% | 2.2% | Within-task z-score of numeric tokens plus percentage expressions. Computed separately for each output before paired differencing. |
| Avoids additions/omissions | -0.06 | [-0.11, -0.01] | -0.01 (n=120) | 1.7% | 7.5% | Which translation has fewer unsupported additions and fewer omissions of source meaning? |

## Output Files

- full CSV: `outputs/analysis/moral_low_vs_framed_empty_combined_feature_appendix_all.csv`
- clear-differences CSV: `outputs/analysis/moral_low_vs_framed_empty_combined_feature_appendix_clear_differences.csv`
- clear-differences Markdown: `outputs/analysis/moral_low_vs_framed_empty_combined_feature_appendix_clear_differences.md`
- LaTeX longtable: `outputs/analysis/moral_low_vs_framed_empty_combined_feature_appendix_clear_differences.tex`

The full CSV includes all rows, including features whose confidence interval includes zero.