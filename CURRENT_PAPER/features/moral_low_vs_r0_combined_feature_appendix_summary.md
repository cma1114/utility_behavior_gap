# Moral Low vs R0 Feature Appendix

Comparison: moral_low_vs_r0.

The table combines two feature families:

- Generic text features are computed on the full paired dataset for this contrast.
- LLM rubric markers are task-specific comparative codes from the random actor-balanced A/B-coded sample.

Editable feature labels, definitions, and display rounding come from `/Users/christopherackerman/repos/Utility-Behavior-Gap/analysis_specs/feature_definitions.yaml`.

The generic text rows use the standard paper-facing feature set: Words, Paragraphs, Unique-word ratio, Quantitative detail, Flesch-Kincaid grade, Positive-word rate per 1k words, Negative-word rate per 1k words. Quantitative detail is `z(numbers + percentages)`, standardized within task before paired differencing.

Positive deltas mean the moral low output had more of the generic feature or was favored by the rubric coder on the task-specific marker. Confidence intervals are over actor/task cells for the generic full-sample rows and over actors within task for the rubric rows.

`Panel preference (r)` is the within-comparison correlation between left-minus-right feature score and left-minus-right panel score. Positive values mean the judging panel tended to prefer the side with more of that feature; negative values mean the panel tended to prefer the side with less.

- total rows: `55`
- clear-difference rows: `16`

## Clear Differences Only

### Essay writing

| Feature | Delta | 95% CI | Panel preference (r) | % moral low higher/better | % R0 higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Positive-word rate per 1k words | 1.62 | [0.56, 2.47] | -0.05 (n=992) | 55.9% | 44.0% | VADER positive-lexicon tokens per 1,000 words. |
| Negative-word rate per 1k words | -1.58 | [-2.19, -0.90] | -0.06 (n=992) | 43.9% | 56.1% | VADER negative-lexicon tokens per 1,000 words. |
| Paragraphs | 0.3 | [0.0, 0.5] | +0.19 (n=992) | 39.2% | 16.9% | Paragraph count split on blank lines. |
| Concrete example quality | -0.28 | [-0.39, -0.18] | +0.43 (n=120) | 33.3% | 61.7% | Which essay uses the required concrete example more specifically, plausibly, and usefully? |
| Counterargument/qualification | -0.26 | [-0.42, -0.10] | +0.09 (n=120) | 18.3% | 44.2% | Which essay better anticipates objections or tradeoffs and responds to them? |
| Rhetorical coherence and closure | -0.22 | [-0.34, -0.10] | +0.40 (n=120) | 29.2% | 51.7% | Which essay has better paragraph flow, transitions, and final synthesis? |
| Thesis and stakes framing | -0.19 | [-0.30, -0.08] | +0.37 (n=120) | 35.8% | 55.0% | Which essay presents the central claim and its stakes more clearly and compellingly? |

### Grant abstract

| Feature | Delta | 95% CI | Panel preference (r) | % moral low higher/better | % R0 higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Positive-word rate per 1k words | -1.53 | [-3.26, -0.16] | -0.09 (n=914) | 43.5% | 56.3% | VADER positive-lexicon tokens per 1,000 words. |
| Paragraphs | 0.6 | [0.1, 1.1] | +0.16 (n=914) | 39.9% | 19.6% | Paragraph count split on blank lines. |
| Risk mitigation | -0.26 | [-0.45, -0.03] | +0.56 (n=120) | 36.7% | 62.5% | Which abstract identifies more realistic risks and more credible mitigation plans? |

### Incident postmortem

| Feature | Delta | 95% CI | Panel preference (r) | % moral low higher/better | % R0 higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Words | -39 | [-66, -14] | +0.40 (n=974) | 34.4% | 65.3% | Word count. |
| Positive-word rate per 1k words | -1.88 | [-3.21, -0.78] | -0.05 (n=974) | 40.7% | 59.2% | VADER positive-lexicon tokens per 1,000 words. |
| Paragraphs | -1.1 | [-1.9, -0.3] | +0.16 (n=974) | 29.2% | 55.0% | Paragraph count split on blank lines. |
| Unique-word ratio | 0.0087 | [0.0003, 0.0169] | -0.14 (n=974) | 61.6% | 38.4% | Unique lowercased word tokens divided by total word tokens. |

### Translation

| Feature | Delta | 95% CI | Panel preference (r) | % moral low higher/better | % R0 higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Positive-word rate per 1k words | -0.90 | [-1.64, -0.08] | -0.07 (n=970) | 32.2% | 35.3% | VADER positive-lexicon tokens per 1,000 words. |
| Negative-word rate per 1k words | -0.79 | [-1.31, -0.26] | +0.08 (n=970) | 14.9% | 16.3% | VADER negative-lexicon tokens per 1,000 words. |

## Output Files

- full CSV: `outputs/analysis/moral_low_vs_r0_combined_feature_appendix_all.csv`
- clear-differences CSV: `outputs/analysis/moral_low_vs_r0_combined_feature_appendix_clear_differences.csv`
- clear-differences Markdown: `outputs/analysis/moral_low_vs_r0_combined_feature_appendix_clear_differences.md`
- LaTeX longtable: `outputs/analysis/moral_low_vs_r0_combined_feature_appendix_clear_differences.tex`

The full CSV includes all rows, including features whose confidence interval includes zero.