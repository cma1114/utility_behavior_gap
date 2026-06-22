# moral_low Versus r0 Standard Generic-Feature Deltas

Comparison: `moral_low - r0` on matched actor/task/item/repeat outputs.

The primary overall estimate is the equal actor-task-cell mean. Confidence intervals use the same bootstrap convention as the other standard generic-feature tables.

- input output catalog: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/final_text_analysis_by_output.csv`
- feature definitions CSV: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/final_text_analysis_feature_definitions.csv`
- bootstrap iterations: `5000`
- valid-output filter: `{'valid_output_filter_applied': True, 'invalid_flags': ['missing_output', 'generation_success_false', 'finish_reason_missing', 'non_stop_finish', 'explicit_length_truncation', 'empty_output'], 'rows_before_filter': 81900, 'rows_after_filter': 81838, 'rows_dropped': 62}`
- semantic exclusion filter: `{'semantic_exclusion_filter_applied': True, 'classification_path': '/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/canonical_highn_moral_refusal_classifications.jsonl', 'classified_outputs_in_file': 16745, 'rows_before_filter': 81838, 'classified_rows_seen': 16745, 'nonclean_classified_rows_dropped': 774, 'unclassified_rows_seen': 65093, 'rows_after_filter': 81064}`
- matched pairs: `3853`
- standard generic features analyzed: `7`
- clear overall differences: `0`
- skipped feature columns: `none`

## Compact Overall Table

| feature | delta | 95% CI | std delta |
| --- | --- | --- | --- |
| words | -11.527 | [-35.554, 3.414] | -0.190 |
| unique_word_ratio | 0.003 | [-0.004, 0.010] | 0.073 |
| quantitative_detail | -0.050 | [-0.207, 0.098] | -0.067 |
| positive_words_per_1k | -0.672 | [-2.118, 0.950] | -0.054 |
| textstat_flesch_kincaid_grade | 0.126 | [-0.174, 0.562] | 0.052 |
| negative_words_per_1k | -0.435 | [-1.345, 0.475] | -0.046 |
| paragraphs | -0.057 | [-0.871, 0.544] | -0.019 |

## Task Summary

| task | clear moral_low > r0 features | clear r0 > moral_low features |
| --- | --- | --- |
| Essay writing | 2 | 1 |
| Grant abstract | 1 | 1 |
| Incident postmortem | 1 | 3 |
| Translation | 0 | 2 |

## Outputs

- overall: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/moral_low_vs_r0_feature_deltas_overall.csv`
- by task: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/moral_low_vs_r0_feature_deltas_by_task.csv`
- by actor: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/moral_low_vs_r0_feature_deltas_by_actor.csv`
- by actor-task: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/moral_low_vs_r0_feature_deltas_by_actor_task.csv`
- matched pairs: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/moral_low_vs_r0_feature_deltas_pairs.csv`
