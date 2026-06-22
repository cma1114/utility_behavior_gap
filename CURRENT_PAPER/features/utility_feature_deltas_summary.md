# Utility Standard Generic-Feature Deltas

Comparison: `utility_high - utility_low` (high-utility arm minus low-utility arm).

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
- clear overall differences: `0`
- skipped feature columns: `none`

## Compact Overall Table

| feature | high mean | low mean | delta | 95% CI | std delta |
| --- | --- | --- | --- | --- | --- |
| words | 464.036 | 462.442 | 1.594 | [-0.289, 4.815] | 0.028 |
| quantitative_detail | 0.009 | -0.009 | 0.018 | [-0.016, 0.061] | 0.024 |
| negative_words_per_1k | 28.334 | 28.220 | 0.114 | [-0.197, 0.454] | 0.012 |
| paragraphs | 8.221 | 8.252 | -0.030 | [-0.185, 0.064] | -0.011 |
| unique_word_ratio | 0.636 | 0.636 | 0.000 | [-0.001, 0.001] | 0.006 |
| positive_words_per_1k | 46.007 | 46.036 | -0.029 | [-0.547, 0.428] | -0.002 |
| textstat_flesch_kincaid_grade | 16.800 | 16.796 | 0.004 | [-0.055, 0.079] | 0.002 |

## Largest High-Minus-Low Deltas

| feature | delta | 95% CI | std delta |
| --- | --- | --- | --- |
| words | 1.594 | [-0.289, 4.815] | 0.028 |
| quantitative_detail | 0.018 | [-0.016, 0.061] | 0.024 |
| negative_words_per_1k | 0.114 | [-0.197, 0.454] | 0.012 |
| unique_word_ratio | 0.000 | [-0.001, 0.001] | 0.006 |
| textstat_flesch_kincaid_grade | 0.004 | [-0.055, 0.079] | 0.002 |
| positive_words_per_1k | -0.029 | [-0.547, 0.428] | -0.002 |
| paragraphs | -0.030 | [-0.185, 0.064] | -0.011 |

## Largest Low-Minus-High Deltas

| feature | delta | 95% CI | std delta |
| --- | --- | --- | --- |
| paragraphs | -0.030 | [-0.185, 0.064] | -0.011 |
| positive_words_per_1k | -0.029 | [-0.547, 0.428] | -0.002 |
| textstat_flesch_kincaid_grade | 0.004 | [-0.055, 0.079] | 0.002 |
| unique_word_ratio | 0.000 | [-0.001, 0.001] | 0.006 |
| negative_words_per_1k | 0.114 | [-0.197, 0.454] | 0.012 |
| quantitative_detail | 0.018 | [-0.016, 0.061] | 0.024 |
| words | 1.594 | [-0.289, 4.815] | 0.028 |

## Task Summary

| task | clear high > low features | clear low > high features | largest high > low | largest low > high |
| --- | --- | --- | --- | --- |
| Essay writing | 0 | 0 |  |  |
| Grant abstract | 2 | 0 | words |  |
| Incident postmortem | 2 | 0 | positive_words_per_1k |  |
| Translation | 0 | 0 |  |  |

## Outputs

- overall: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/utility_feature_deltas_overall.csv`
- by task: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/utility_feature_deltas_by_task.csv`
- by actor: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/utility_feature_deltas_by_actor.csv`
- by actor-task: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/utility_feature_deltas_by_actor_task.csv`
- significant overall features: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/utility_feature_deltas_overall_significant.csv`
