# moral_low Versus framed_empty Standard Generic-Feature Deltas

Comparison: `moral_low - framed_empty` on matched actor/task/item/repeat outputs.

The primary overall estimate is the equal actor-task-cell mean. Confidence intervals use the same bootstrap convention as the other standard generic-feature tables.

- input output catalog: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/final_text_analysis_by_output.csv`
- feature definitions CSV: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/final_text_analysis_feature_definitions.csv`
- bootstrap iterations: `5000`
- valid-output filter: `{'valid_output_filter_applied': True, 'invalid_flags': ['missing_output', 'generation_success_false', 'finish_reason_missing', 'non_stop_finish', 'explicit_length_truncation', 'empty_output'], 'rows_before_filter': 81900, 'rows_after_filter': 81838, 'rows_dropped': 62}`
- semantic exclusion filter: `{'semantic_exclusion_filter_applied': True, 'classification_path': '/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/canonical_highn_moral_refusal_classifications.jsonl', 'classified_outputs_in_file': 16745, 'rows_before_filter': 81838, 'classified_rows_seen': 16745, 'nonclean_classified_rows_dropped': 774, 'unclassified_rows_seen': 65093, 'rows_after_filter': 81064}`
- matched pairs: `7687`
- standard generic features analyzed: `7`
- clear overall differences: `1`
- skipped feature columns: `none`

## Compact Overall Table

| feature | delta | 95% CI | std delta |
| --- | --- | --- | --- |
| words | -14.227 | [-27.594, -2.877] | -0.256 |
| unique_word_ratio | 0.004 | [-0.002, 0.010] | 0.110 |
| textstat_flesch_kincaid_grade | 0.184 | [-0.031, 0.444] | 0.080 |
| paragraphs | 0.109 | [-0.393, 0.629] | 0.038 |
| negative_words_per_1k | -0.299 | [-1.129, 0.474] | -0.031 |
| quantitative_detail | -0.018 | [-0.112, 0.086] | -0.025 |
| positive_words_per_1k | 0.293 | [-0.423, 1.236] | 0.023 |

## Task Summary

| task | clear moral_low > framed_empty features | clear framed_empty > moral_low features |
| --- | --- | --- |
| Essay writing | 2 | 2 |
| Grant abstract | 3 | 1 |
| Incident postmortem | 1 | 1 |
| Translation | 2 | 2 |

## Outputs

- overall: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/moral_low_vs_framed_empty_feature_deltas_overall.csv`
- by task: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/moral_low_vs_framed_empty_feature_deltas_by_task.csv`
- by actor: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/moral_low_vs_framed_empty_feature_deltas_by_actor.csv`
- by actor-task: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/moral_low_vs_framed_empty_feature_deltas_by_actor_task.csv`
- matched pairs: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/moral_low_vs_framed_empty_feature_deltas_pairs.csv`
