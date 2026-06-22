# Length-Controlled Feature Selection: moral_low_vs_framed_empty

Selection rule: keep features whose left-minus-right arm gap excludes zero, whose panel association excludes zero, and whose signs align. Non-word features control for word-count difference in both tests. Word count itself is included when its raw arm gap and panel association are both clear.

Gap model for non-word features: `feature_delta ~ delta_words + actor fixed effects`.
Panel model for non-word features: `panel_score ~ standardized(feature_delta) + standardized(delta_words) + actor fixed effects`.
Word-count models omit the word-count control because word count is the feature.
The standardized arm gap divides the adjusted arm gap by the observed SD of the paired feature deltas within the same task. It is descriptive scale information, not a separate significance test.

Pair deltas: `outputs/analysis/moral_low_vs_framed_empty_current_pair_deltas_for_selection.csv`
Rubric run: `outputs/analysis/task_rubric_feature_coding/moral_low_vs_framed_empty`

## Essay writing

| family | feature | n gap | arm gap | arm gap, SD | n panel | panel assoc. |
| --- | --- | --- | --- | --- | --- | --- |
| Generic text feature | Paragraphs | 1979 | +0.34 [+0.29, +0.39] | +0.32 [+0.27, +0.36] | 1979 | +0.08 [+0.04, +0.13] |
| Generic text feature | Flesch-Kincaid grade | 1979 | +0.32 [+0.26, +0.38] | +0.26 [+0.21, +0.31] | 1979 | +0.08 [+0.04, +0.12] |
| Generic text feature | MATTR-50 | 1979 | +0.00 [+0.00, +0.00] | +0.07 [+0.02, +0.12] | 1979 | +0.17 [+0.13, +0.21] |

## Grant abstract

| family | feature | n gap | arm gap | arm gap, SD | n panel | panel assoc. |
| --- | --- | --- | --- | --- | --- | --- |
| Generic text feature | Flesch-Kincaid grade | 1831 | +0.30 [+0.23, +0.37] | +0.21 [+0.16, +0.26] | 1831 | +0.09 [+0.05, +0.13] |

## Incident postmortem

| family | feature | n gap | arm gap | arm gap, SD | n panel | panel assoc. |
| --- | --- | --- | --- | --- | --- | --- |
| Generic text feature | Quantitative detail rate | 1944 | +1.37 [+0.52, +2.21] | +0.07 [+0.03, +0.12] | 1944 | +0.27 [+0.23, +0.31] |
| Generic text feature | Rare-word rate per 1k words | 1944 | +0.98 [+0.06, +1.91] | +0.05 [+0.00, +0.10] | 1944 | +0.12 [+0.08, +0.16] |
| LLM rubric marker | Operational realism | 119 | +0.20 [+0.03, +0.37] | +0.21 [+0.03, +0.39] | 119 | +0.44 [+0.26, +0.62] |

## Output Files

- selected/full CSV: `outputs/analysis/moral_low_vs_framed_empty_length_controlled_feature_selection.csv`

Selected rows: `7` of `59` tested feature-task rows.
