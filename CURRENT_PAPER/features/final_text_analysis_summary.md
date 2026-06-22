# Final Text-Feature Catalog

This analysis is manifest-driven. It includes the current direct-instruction, amount, moral, utility, and R0 arms.

spaCy status: Skipped by --no-pos; POS feature columns set to NaN.

## Included Outputs

| source_family | condition | outputs |
| --- | --- | --- |
| amount | amount_high | 8400 |
| amount | amount_low | 8400 |
| direct_instruction | direct_high | 9450 |
| direct_instruction | direct_low | 9450 |
| moral | moral_high | 8400 |
| moral | moral_low | 8400 |
| r0 | r0 | 4197 |
| utility | utility_high | 8400 |
| utility | utility_low | 8400 |

## Included Pair Contrasts

| contrast | pairs | resolved | high_wins | low_wins | ties |
| --- | --- | --- | --- | --- | --- |
| amount | 8400 | 8400 | 1914 | 1779 | 4707 |
| direct_instruction | 9450 | 9445 | 3650 | 1156 | 4639 |
| moral | 8400 | 8345 | 2660 | 1486 | 4199 |
| utility | 8400 | 8400 | 1983 | 1882 | 4535 |

## Artifact Checks

| artifact | count | rate |
| --- | --- | --- |
| non_stop_finish | 0 | 0.000 |
| explicit_length_truncation | 0 | 0.000 |
| near_token_cap_95 | 385 | 0.005 |
| near_token_cap_98 | 94 | 0.001 |
| strict_hidden_scaffold_leak | 266 | 0.004 |
| refusal_or_meta | 1091 | 0.015 |

## Output Files

- `response_features`: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/final_text_analysis_by_output.csv`
- `pair_deltas`: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/final_text_analysis_pair_deltas.csv`
- `response_feature_summary`: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/final_text_analysis_response_feature_summary.csv`
- `pair_delta_summary`: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/final_text_analysis_pair_delta_summary.csv`
- `artifact_summary`: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/final_text_analysis_artifact_summary.csv`
- `feature_definitions`: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/final_text_analysis_feature_definitions.csv`
- `source_manifest`: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/final_text_analysis_source_manifest.json`
- `source_run_audit`: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/final_text_analysis_source_run_audit.csv`
- `summary_markdown`: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/final_text_analysis_summary.md`

## Source Summary

- direct run dirs: 63
- fund-wording run dirs: 28
- amount run dirs: 28
- R0 included rows: 4197

Confidence intervals in the text-feature summary files are descriptive normal-approximation intervals over the rows in each summary group. They are for mechanism screening, not the primary outcome inference.
