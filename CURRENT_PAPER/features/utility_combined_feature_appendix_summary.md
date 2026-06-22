# Utility High-Low Feature Appendix

Comparison: utility.

The table combines two feature families:

- Generic text features are computed on the full paired dataset for this contrast.
- LLM rubric markers are task-specific comparative codes from the random actor-balanced A/B-coded sample.

Editable feature labels, definitions, and display rounding come from `/Users/christopherackerman/repos/Utility-Behavior-Gap/analysis_specs/feature_definitions.yaml`.

The generic text rows use the standard paper-facing feature set: Words, Paragraphs, Unique-word ratio, Quantitative detail, Flesch-Kincaid grade, Positive-word rate per 1k words, Negative-word rate per 1k words. Quantitative detail is `z(numbers + percentages)`, standardized within task before paired differencing.

Positive deltas mean the utility high output had more of the generic feature or was favored by the rubric coder on the task-specific marker. Confidence intervals are over actor/task cells for the generic full-sample rows and over actors within task for the rubric rows.

`Panel preference (r)` is the within-comparison correlation between left-minus-right feature score and left-minus-right panel score. Positive values mean the judging panel tended to prefer the side with more of that feature; negative values mean the panel tended to prefer the side with less.

- total rows: `55`
- clear-difference rows: `7`

## Clear Differences Only

### Essay writing

| Feature | Delta | 95% CI | Panel preference (r) | % utility high higher/better | % utility low higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Counterargument/qualification | -0.15 | [-0.26, -0.05] | +0.24 (n=120) | 21.7% | 36.7% | Which essay better anticipates objections or tradeoffs and responds to them? |
| Thesis and stakes framing | -0.14 | [-0.26, -0.03] | +0.34 (n=120) | 36.7% | 50.8% | Which essay presents the central claim and its stakes more clearly and compellingly? |

### Grant abstract

| Feature | Delta | 95% CI | Panel preference (r) | % utility high higher/better | % utility low higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Words | 4.7 | [0.7, 9.4] | +0.34 (n=2,100) | 51.2% | 48.1% | Word count. |
| Quantitative detail | 0.064 | [0.023, 0.108] |  | 42.4% | 37.3% | Within-task z-score of numeric tokens plus percentage expressions. Computed separately for each output before paired differencing. |

### Incident postmortem

| Feature | Delta | 95% CI | Panel preference (r) | % utility high higher/better | % utility low higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Positive-word rate per 1k words | 0.46 | [0.19, 0.76] | -0.07 (n=2,100) | 52.7% | 47.2% | VADER positive-lexicon tokens per 1,000 words. |
| Negative-word rate per 1k words | 0.42 | [0.18, 0.66] | +0.00 (n=2,100) | 50.8% | 49.1% | VADER negative-lexicon tokens per 1,000 words. |

### Translation

| Feature | Delta | 95% CI | Panel preference (r) | % utility high higher/better | % utility low higher/better | Definition |
| --- | --- | --- | --- | --- | --- | --- |
| Terminology precision | -0.11 | [-0.18, -0.05] | +0.31 (n=120) | 10.8% | 21.7% | Which translation uses more domain-appropriate terminology? |

## Output Files

- full CSV: `outputs/analysis/utility_combined_feature_appendix_all.csv`
- clear-differences CSV: `outputs/analysis/utility_combined_feature_appendix_clear_differences.csv`
- clear-differences Markdown: `outputs/analysis/utility_combined_feature_appendix_clear_differences.md`
- LaTeX longtable: `outputs/analysis/utility_combined_feature_appendix_clear_differences.tex`

The full CSV includes all rows, including features whose confidence interval includes zero.