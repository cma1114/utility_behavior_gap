# Canonical High-N Condition Results

Conditions: amount, moral, and high-low utility.
Rows combine base repeats 0-4 with high-N extension repeats 5-9 where available.

Primary estimate: equal-cell mean. Amount and moral average actor x task cells; high-low utility averages actor x task x domain cells. Panel ties are excluded from the win-rate denominator and reported separately.

Moral refusal mode used for this run: `available`.

## Moral Refusal Screen Coverage

| screen_status | pairs |
| --- | --- |
| clean_llm_labels | 7612 |

## Amount Prompt Status

Amount uses the corrected base block for repeats 0-4 plus the high-N extension block for repeats 5-9. Both use the current neutral target text `A healthcare intervention at a children's hospital.`

## Total

| condition | resolved | wins | losses | ties | pooled | equal-cell mean | equal-cell 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Amount | 3693 | 1914 | 1779 | 4707 | 51.8% | 51.6% | 48.0%-55.5% |
| Moral | 3550 | 2110 | 1440 | 4062 | 59.4% | 59.6% | 55.9%-63.5% |
| High vs low utility | 3865 | 1983 | 1882 | 4535 | 51.3% | 51.2% | 48.7%-53.6% |

## By Task

| condition | task | resolved | wins | losses | ties | equal-cell mean | equal-cell 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|
| Amount | Essay writing | 756 | 377 | 379 | 1344 | 50.0% | 43.5%-56.6% |
| Amount | Grant abstract | 954 | 597 | 357 | 1146 | 61.9% | 55.2%-69.0% |
| Amount | Incident postmortem | 1066 | 529 | 537 | 1034 | 49.6% | 45.0%-54.3% |
| Amount | Translation | 917 | 411 | 506 | 1183 | 45.0% | 41.2%-48.6% |
| Moral | Essay writing | 774 | 460 | 314 | 1205 | 59.4% | 51.3%-67.4% |
| Moral | Grant abstract | 887 | 578 | 309 | 942 | 65.2% | 57.0%-73.0% |
| Moral | Incident postmortem | 1001 | 601 | 400 | 945 | 60.6% | 54.3%-67.3% |
| Moral | Translation | 888 | 471 | 417 | 970 | 53.1% | 49.3%-56.9% |
| High vs low utility | Essay writing | 775 | 388 | 387 | 1325 | 50.2% | 44.7%-55.2% |
| High vs low utility | Grant abstract | 1058 | 568 | 490 | 1042 | 53.8% | 49.4%-58.2% |
| High vs low utility | Incident postmortem | 1053 | 552 | 501 | 1047 | 52.7% | 48.2%-57.3% |
| High vs low utility | Translation | 979 | 475 | 504 | 1121 | 47.9% | 43.1%-52.7% |

## Utility By Domain

| domain | resolved | wins | losses | ties | equal-cell mean | equal-cell 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| animals | 1010 | 507 | 503 | 1090 | 50.1% | 45.3%-54.8% |
| countries | 923 | 432 | 491 | 1177 | 46.2% | 41.5%-51.0% |
| political | 974 | 565 | 409 | 1126 | 58.3% | 53.7%-62.7% |
| religions | 958 | 479 | 479 | 1142 | 50.1% | 45.6%-54.3% |

## Model-Task Positive Counts

| condition | familywise-CI positive cells | Holm-positive cells |
|---|---:|---:|
| Amount | 3/28 | 3/28 |
| Moral | 10/28 | 10/28 |
| High vs low utility | 0/28 | 0/28 |

## Source Blocks

| condition | source_block | included_pairs |
| --- | --- | --- |
| amount | base_amount_current_target_r0_r4 | 4200 |
| amount | highn_extension_r5_r9 | 4200 |
| moral | base_fund_wording_r0_r4 | 3824 |
| moral | highn_extension_r5_r9 | 3788 |
| utility | base_fund_wording_r0_r4 | 4200 |
| utility | highn_extension_r5_r9 | 4200 |

## Audit Counts

| condition | status | rows |
| --- | --- | --- |
| amount | included | 8400 |
| moral | included | 7612 |
| moral | missing_generation | 55 |
| moral | moral_refusal_or_degenerate_exclusion | 733 |
| utility | included | 8400 |

## Output Files

- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/canonical_highn_condition_results_pair_outcomes.csv`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/canonical_highn_condition_results_summary.csv`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/canonical_highn_condition_results_model_task.csv`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/canonical_highn_condition_results_audit.csv`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/canonical_highn_condition_results_summary.md`
