# Length-Controlled Feature Selection: framed_user_prompt_role

Selection rule: keep features whose left-minus-right arm gap excludes zero, whose panel association excludes zero, and whose signs align. Non-word features control for word-count difference in both tests. Word count itself is included when its raw arm gap and panel association are both clear.

Gap model for non-word features: `feature_delta ~ delta_words + actor fixed effects`.
Panel model for non-word features: `panel_score ~ standardized(feature_delta) + standardized(delta_words) + actor fixed effects`.
Word-count models omit the word-count control because word count is the feature.
The standardized arm gap divides the adjusted arm gap by the observed SD of the paired feature deltas within the same task. It is descriptive scale information, not a separate significance test.

Pair deltas: `outputs/analysis/framed_user_prompt_role_pair_deltas.csv`
Rubric run: `outputs/analysis/task_rubric_feature_coding/framed_user_prompt_role__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash`

## Essay writing

| family | feature | n gap | arm gap | arm gap, SD | n panel | panel assoc. |
| --- | --- | --- | --- | --- | --- | --- |
| Generic text feature | Rare-word rate per 1k words | 2097 | +6.86 [+5.68, +8.05] | +0.30 [+0.25, +0.35] | 2097 | +0.08 [+0.05, +0.11] |
| Generic text feature | Words | 2097 | +5.43 [+3.99, +6.87] | +0.16 [+0.12, +0.20] | 2097 | +0.19 [+0.17, +0.22] |
| Generic text feature | Positive-word rate per 1k words | 2097 | -2.77 [-3.44, -2.11] | -0.18 [-0.23, -0.14] | 2097 | -0.04 [-0.06, -0.01] |

## Grant abstract

| family | feature | n gap | arm gap | arm gap, SD | n panel | panel assoc. |
| --- | --- | --- | --- | --- | --- | --- |
| Generic text feature | Rare-word rate per 1k words | 2089 | +2.51 [+1.74, +3.28] | +0.14 [+0.10, +0.18] | 2089 | +0.17 [+0.15, +0.20] |
| Generic text feature | Quantitative detail rate | 2089 | +0.32 [+0.01, +0.63] | +0.04 [+0.00, +0.09] | 2089 | +0.24 [+0.22, +0.27] |
| Generic text feature | Flesch-Kincaid grade | 2089 | +0.10 [+0.03, +0.16] | +0.06 [+0.02, +0.11] | 2089 | +0.10 [+0.08, +0.13] |
| Generic text feature | MATTR-50 | 2089 | +0.00 [+0.00, +0.00] | +0.09 [+0.05, +0.13] | 2089 | +0.15 [+0.12, +0.17] |

## Incident postmortem

| family | feature | n gap | arm gap | arm gap, SD | n panel | panel assoc. |
| --- | --- | --- | --- | --- | --- | --- |
| Generic text feature | Rare-word rate per 1k words | 2088 | +1.30 [+0.43, +2.17] | +0.06 [+0.02, +0.11] | 2088 | +0.13 [+0.11, +0.16] |
| Generic text feature | Paragraphs | 2088 | +0.23 [+0.02, +0.44] | +0.05 [+0.00, +0.09] | 2088 | +0.05 [+0.01, +0.08] |
| Generic text feature | MATTR-50 | 2088 | +0.00 [+0.00, +0.00] | +0.07 [+0.03, +0.12] | 2088 | +0.11 [+0.08, +0.14] |

## Output Files

- selected/full CSV: `outputs/analysis/framed_user_prompt_role_length_controlled_feature_selection.csv`

Selected rows: `10` of `59` tested feature-task rows.
