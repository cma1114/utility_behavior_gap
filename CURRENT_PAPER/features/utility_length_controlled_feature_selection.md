# Length-Controlled Feature Selection: utility

Selection rule: keep features whose left-minus-right arm gap excludes zero, whose panel association excludes zero, and whose signs align. Non-word features control for word-count difference in both tests. Word count itself is included when its raw arm gap and panel association are both clear.

Gap model for non-word features: `feature_delta ~ delta_words + actor fixed effects`.
Panel model for non-word features: `panel_score ~ standardized(feature_delta) + standardized(delta_words) + actor fixed effects`.
Word-count models omit the word-count control because word count is the feature.
The standardized arm gap divides the adjusted arm gap by the observed SD of the paired feature deltas within the same task. It is descriptive scale information, not a separate significance test.

Pair deltas: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/final_text_analysis_pair_deltas.csv`
Rubric run: `outputs/analysis/task_rubric_feature_coding/utility__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash`

## Essay writing

| family | feature | n gap | arm gap | arm gap, SD | n panel | panel assoc. |
| --- | --- | --- | --- | --- | --- | --- |
| Generic text feature | MATTR-50 | 2100 | +0.00 [+0.00, +0.00] | +0.04 [+0.00, +0.08] | 2100 | +0.17 [+0.14, +0.19] |

## Grant abstract

| family | feature | n gap | arm gap | arm gap, SD | n panel | panel assoc. |
| --- | --- | --- | --- | --- | --- | --- |
| Generic text feature | Words | 2100 | +4.66 [+1.80, +7.52] | +0.07 [+0.03, +0.11] | 2100 | +0.24 [+0.20, +0.28] |
| Generic text feature | Positive-word rate per 1k words | 2100 | -0.46 [-0.93, +0.00] | -0.04 [-0.09, +0.00] | 2100 | -0.14 [-0.16, -0.11] |

## Output Files

- selected/full CSV: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/utility_length_controlled_feature_selection.csv`

Selected rows: `3` of `59` tested feature-task rows.
