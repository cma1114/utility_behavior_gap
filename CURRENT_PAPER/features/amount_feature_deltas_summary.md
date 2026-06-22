# Amount Standard Generic-Feature Deltas

Comparison: `amount_high - amount_low` (larger-amount arm minus smaller-amount arm).

The primary overall estimate is the equal actor-task-cell mean. Its confidence interval uses a nonparametric bootstrap over actor and task cells.

Generic features use the standard paper-facing set from `analysis_specs/feature_definitions.yaml`: words, paragraphs, unique-word ratio, quantitative detail, Flesch-Kincaid grade, positive-word rate, and negative-word rate. Quantitative detail is `z(numbers + percentages)` standardized within task before paired differencing.

- input pair catalog: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/final_text_analysis_pair_deltas.csv`
- feature definitions CSV: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/final_text_analysis_feature_definitions.csv`
- editable feature spec: `/Users/christopherackerman/repos/Utility-Behavior-Gap/analysis_specs/feature_definitions.yaml`
- bootstrap iterations: `5000`
- valid-output filter: `{'valid_output_filter_applied': True, 'invalid_flags': ['missing_output', 'generation_success_false', 'finish_reason_missing', 'non_stop_finish', 'explicit_length_truncation', 'empty_output'], 'used_inline_flags': True, 'side_rows_checked': 16800, 'invalid_side_rows': 0, 'pairs_before_filter': 8400, 'pairs_after_filter': 8400, 'pairs_dropped': 0}`
- semantic exclusion filter: `{'semantic_exclusion_filter_applied': True, 'classification_path': '/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/canonical_highn_moral_refusal_classifications.jsonl', 'classified_outputs_in_file': 16745, 'side_rows_checked': 16800, 'classified_side_rows_seen': 0, 'nonclean_classified_side_rows_dropped': 0, 'unclassified_side_rows_seen': 16800, 'pairs_before_filter': 8400, 'pairs_after_filter': 8400, 'pairs_dropped': 0}`
- pairs: `8400`
- actor-task cells: `28`
- standard generic features analyzed: `7`
- clear overall differences: `2`
- skipped feature columns: `none`

## Compact Overall Table

| feature | high mean | low mean | delta | 95% CI | std delta |
| --- | --- | --- | --- | --- | --- |
| words | 465.886 | 472.631 | -6.745 | [-14.812, -1.189] | -0.124 |
| unique_word_ratio | 0.636 | 0.633 | 0.003 | [0.001, 0.005] | 0.089 |
| quantitative_detail | 0.013 | -0.013 | 0.026 | [-0.006, 0.075] | 0.034 |
| negative_words_per_1k | 28.157 | 27.843 | 0.314 | [-0.017, 0.689] | 0.033 |
| textstat_flesch_kincaid_grade | 16.839 | 16.761 | 0.078 | [-0.043, 0.196] | 0.033 |
| paragraphs | 8.238 | 8.325 | -0.087 | [-0.299, 0.028] | -0.031 |
| positive_words_per_1k | 46.014 | 46.023 | -0.008 | [-0.418, 0.362] | -0.001 |

## Largest High-Minus-Low Deltas

| feature | delta | 95% CI | std delta |
| --- | --- | --- | --- |
| unique_word_ratio | 0.003 | [0.001, 0.005] | 0.089 |
| quantitative_detail | 0.026 | [-0.006, 0.075] | 0.034 |
| negative_words_per_1k | 0.314 | [-0.017, 0.689] | 0.033 |
| textstat_flesch_kincaid_grade | 0.078 | [-0.043, 0.196] | 0.033 |
| positive_words_per_1k | -0.008 | [-0.418, 0.362] | -0.001 |
| paragraphs | -0.087 | [-0.299, 0.028] | -0.031 |
| words | -6.745 | [-14.812, -1.189] | -0.124 |

## Largest Low-Minus-High Deltas

| feature | delta | 95% CI | std delta |
| --- | --- | --- | --- |
| words | -6.745 | [-14.812, -1.189] | -0.124 |
| paragraphs | -0.087 | [-0.299, 0.028] | -0.031 |
| positive_words_per_1k | -0.008 | [-0.418, 0.362] | -0.001 |
| textstat_flesch_kincaid_grade | 0.078 | [-0.043, 0.196] | 0.033 |
| negative_words_per_1k | 0.314 | [-0.017, 0.689] | 0.033 |
| quantitative_detail | 0.026 | [-0.006, 0.075] | 0.034 |
| unique_word_ratio | 0.003 | [0.001, 0.005] | 0.089 |

## Task Summary

| task | clear high > low features | clear low > high features | largest high > low | largest low > high |
| --- | --- | --- | --- | --- |
| Essay writing | 1 | 1 | unique_word_ratio | words |
| Grant abstract | 4 | 1 | textstat_flesch_kincaid_grade | words |
| Incident postmortem | 3 | 1 | unique_word_ratio | words |
| Translation | 0 | 0 |  |  |

## Outputs

- overall: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/amount_feature_deltas_overall.csv`
- by task: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/amount_feature_deltas_by_task.csv`
- by actor: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/amount_feature_deltas_by_actor.csv`
- by actor-task: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/amount_feature_deltas_by_actor_task.csv`
- significant overall features: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/amount_feature_deltas_overall_significant.csv`
