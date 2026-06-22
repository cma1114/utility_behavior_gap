# Direct-Instruction Feature Deltas

Comparison: `direct_high - direct_low`, where `direct_high` is the finalized exhortative user-prompt arm and `direct_low` is framed neutral.

The primary overall estimate is the equal actor-task-cell mean. Its confidence interval uses a nonparametric bootstrap over actor and task cells, so translation's larger sample does not dominate the aggregate.

Generic features use the standard paper-facing set: words, paragraphs, unique-word ratio, quantitative detail, Flesch-Kincaid grade, positive-word rate, and negative-word rate. Quantitative detail is `z(numbers + percentages)` standardized within task before paired differencing.

- input pair catalog: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/final_text_analysis_pair_deltas.csv`
- feature definitions: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/final_text_analysis_feature_definitions.csv`
- bootstrap iterations: `5000`
- valid-output filter: `{'valid_output_filter_applied': True, 'invalid_flags': ['missing_output', 'generation_success_false', 'finish_reason_missing', 'non_stop_finish', 'explicit_length_truncation', 'empty_output'], 'used_inline_flags': True, 'side_rows_checked': 18900, 'invalid_side_rows': 5, 'pairs_before_filter': 9450, 'pairs_after_filter': 9445, 'pairs_dropped': 5}`
- semantic exclusion filter: `{'semantic_exclusion_filter_applied': True, 'classification_path': '/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/canonical_highn_moral_refusal_classifications.jsonl', 'classified_outputs_in_file': 16745, 'side_rows_checked': 18890, 'classified_side_rows_seen': 0, 'nonclean_classified_side_rows_dropped': 0, 'unclassified_side_rows_seen': 18890, 'pairs_before_filter': 9445, 'pairs_after_filter': 9445, 'pairs_dropped': 0}`
- direct-instruction pairs: `9445`
- actor-task cells: `28`
- standard generic features analyzed: `7`
- skipped feature columns: `none`

## Compact Overall Table

| feature | strong mean | neutral mean | delta | 95% CI | std delta |
| --- | --- | --- | --- | --- | --- |
| words | 492.347 | 467.760 | 24.588 | [4.139, 61.171] | 0.363 |
| quantitative_detail | 0.060 | -0.059 | 0.119 | [0.016, 0.286] | 0.169 |
| paragraphs | 8.480 | 8.169 | 0.312 | [-0.071, 1.138] | 0.105 |
| negative_words_per_1k | 28.776 | 28.135 | 0.641 | [0.227, 1.112] | 0.066 |
| positive_words_per_1k | 45.384 | 46.108 | -0.724 | [-2.311, 0.495] | -0.051 |
| unique_word_ratio | 0.635 | 0.634 | 0.001 | [-0.004, 0.006] | 0.034 |
| textstat_flesch_kincaid_grade | 16.851 | 16.801 | 0.050 | [-0.274, 0.291] | 0.019 |

## Largest Increases

| feature | delta | 95% CI | std delta |
| --- | --- | --- | --- |
| words | 24.588 | [4.139, 61.171] | 0.363 |
| quantitative_detail | 0.119 | [0.016, 0.286] | 0.169 |
| paragraphs | 0.312 | [-0.071, 1.138] | 0.105 |
| negative_words_per_1k | 0.641 | [0.227, 1.112] | 0.066 |
| unique_word_ratio | 0.001 | [-0.004, 0.006] | 0.034 |
| textstat_flesch_kincaid_grade | 0.050 | [-0.274, 0.291] | 0.019 |
| positive_words_per_1k | -0.724 | [-2.311, 0.495] | -0.051 |

## Largest Decreases

| feature | delta | 95% CI | std delta |
| --- | --- | --- | --- |
| positive_words_per_1k | -0.724 | [-2.311, 0.495] | -0.051 |
| textstat_flesch_kincaid_grade | 0.050 | [-0.274, 0.291] | 0.019 |
| unique_word_ratio | 0.001 | [-0.004, 0.006] | 0.034 |
| negative_words_per_1k | 0.641 | [0.227, 1.112] | 0.066 |
| paragraphs | 0.312 | [-0.071, 1.138] | 0.105 |
| quantitative_detail | 0.119 | [0.016, 0.286] | 0.169 |
| words | 24.588 | [4.139, 61.171] | 0.363 |

## Task Summary

| task | significant increases | significant decreases | largest increase | largest decrease |
| --- | --- | --- | --- | --- |
| Essay writing | 2 | 2 | words | positive_words_per_1k |
| Grant abstract | 5 | 0 | unique_word_ratio |  |
| Incident postmortem | 5 | 0 | words |  |
| Translation | 2 | 1 | words | quantitative_detail |

## Outputs

- overall: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/direct_instruction_feature_deltas_overall.csv`
- by task: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/direct_instruction_feature_deltas_by_task.csv`
- by actor: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/direct_instruction_feature_deltas_by_actor.csv`
- by actor-task: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/direct_instruction_feature_deltas_by_actor_task.csv`
- significant overall features: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/direct_instruction_feature_deltas_overall_significant.csv`
