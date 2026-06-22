# Moral Standard Generic-Feature Deltas

Comparison: `moral_high - moral_low` (morally good arm minus morally bad arm).

The primary overall estimate is the equal actor-task-cell mean. Its confidence interval uses a nonparametric bootstrap over actor and task cells.

Generic features use the standard paper-facing set from `analysis_specs/feature_definitions.yaml`: words, paragraphs, unique-word ratio, quantitative detail, Flesch-Kincaid grade, positive-word rate, and negative-word rate. Quantitative detail is `z(numbers + percentages)` standardized within task before paired differencing.

- input pair catalog: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/final_text_analysis_pair_deltas.csv`
- feature definitions CSV: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/final_text_analysis_feature_definitions.csv`
- editable feature spec: `/Users/christopherackerman/repos/Utility-Behavior-Gap/analysis_specs/feature_definitions.yaml`
- bootstrap iterations: `5000`
- valid-output filter: `{'valid_output_filter_applied': True, 'invalid_flags': ['missing_output', 'generation_success_false', 'finish_reason_missing', 'non_stop_finish', 'explicit_length_truncation', 'empty_output'], 'used_inline_flags': True, 'side_rows_checked': 16800, 'invalid_side_rows': 55, 'pairs_before_filter': 8400, 'pairs_after_filter': 8345, 'pairs_dropped': 55}`
- semantic exclusion filter: `{'semantic_exclusion_filter_applied': True, 'classification_path': '/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/canonical_highn_moral_refusal_classifications.jsonl', 'classified_outputs_in_file': 16745, 'side_rows_checked': 16690, 'classified_side_rows_seen': 16690, 'nonclean_classified_side_rows_dropped': 772, 'unclassified_side_rows_seen': 0, 'pairs_before_filter': 8345, 'pairs_after_filter': 7612, 'pairs_dropped': 733}`
- pairs: `7612`
- actor-task cells: `28`
- standard generic features analyzed: `7`
- clear overall differences: `2`
- skipped feature columns: `none`

## Compact Overall Table

| feature | high mean | low mean | delta | 95% CI | std delta |
| --- | --- | --- | --- | --- | --- |
| words | 463.908 | 451.828 | 12.080 | [1.628, 24.195] | 0.211 |
| unique_word_ratio | 0.637 | 0.639 | -0.003 | [-0.006, 0.000] | -0.082 |
| textstat_flesch_kincaid_grade | 16.834 | 16.952 | -0.117 | [-0.281, 0.026] | -0.049 |
| negative_words_per_1k | 27.751 | 27.352 | 0.399 | [0.021, 0.924] | 0.042 |
| positive_words_per_1k | 46.179 | 46.467 | -0.287 | [-1.182, 0.267] | -0.022 |
| quantitative_detail | 0.009 | -0.005 | 0.014 | [-0.057, 0.078] | 0.019 |
| paragraphs | 8.182 | 8.190 | -0.008 | [-0.326, 0.372] | -0.003 |

## Largest High-Minus-Low Deltas

| feature | delta | 95% CI | std delta |
| --- | --- | --- | --- |
| words | 12.080 | [1.628, 24.195] | 0.211 |
| negative_words_per_1k | 0.399 | [0.021, 0.924] | 0.042 |
| quantitative_detail | 0.014 | [-0.057, 0.078] | 0.019 |
| paragraphs | -0.008 | [-0.326, 0.372] | -0.003 |
| positive_words_per_1k | -0.287 | [-1.182, 0.267] | -0.022 |
| textstat_flesch_kincaid_grade | -0.117 | [-0.281, 0.026] | -0.049 |
| unique_word_ratio | -0.003 | [-0.006, 0.000] | -0.082 |

## Largest Low-Minus-High Deltas

| feature | delta | 95% CI | std delta |
| --- | --- | --- | --- |
| unique_word_ratio | -0.003 | [-0.006, 0.000] | -0.082 |
| textstat_flesch_kincaid_grade | -0.117 | [-0.281, 0.026] | -0.049 |
| positive_words_per_1k | -0.287 | [-1.182, 0.267] | -0.022 |
| paragraphs | -0.008 | [-0.326, 0.372] | -0.003 |
| quantitative_detail | 0.014 | [-0.057, 0.078] | 0.019 |
| negative_words_per_1k | 0.399 | [0.021, 0.924] | 0.042 |
| words | 12.080 | [1.628, 24.195] | 0.211 |

## Task Summary

| task | clear high > low features | clear low > high features | largest high > low | largest low > high |
| --- | --- | --- | --- | --- |
| Essay writing | 1 | 3 | negative_words_per_1k | textstat_flesch_kincaid_grade |
| Grant abstract | 1 | 2 | words | unique_word_ratio |
| Incident postmortem | 2 | 1 | words | unique_word_ratio |
| Translation | 0 | 1 |  | paragraphs |

## Outputs

- overall: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/moral_feature_deltas_overall.csv`
- by task: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/moral_feature_deltas_by_task.csv`
- by actor: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/moral_feature_deltas_by_actor.csv`
- by actor-task: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/moral_feature_deltas_by_actor_task.csv`
- significant overall features: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/moral_feature_deltas_overall_significant.csv`
