# Framed User Prompt Role Length Controlled Feature Selection Selected Feature Table

Rows shown are exactly the selected rows from `outputs/analysis/framed_user_prompt_role_length_controlled_feature_selection.csv`.

Arm gap is left-minus-right at equal word count for non-word features. Panel association is the standardized feature coefficient in `panel_score ~ feature + words + actor`; for Words, it is the word-count coefficient without word-count control.

Arm gap in SD units divides the adjusted arm gap by the observed SD of the paired feature deltas within the same task. Use it to compare effect sizes across features with different raw units.

## Essay writing

| Task | Family | Feature | N gap | Arm gap (95% CI) | Arm gap, SD units (95% CI) | Gap p | N panel | Panel assoc. per SD (95% CI) | Panel p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Essay writing | Generic text feature | Rare-word rate per 1k words | 2097 | +6.86 [+5.68, +8.05] | +0.30 [+0.25, +0.35] | <0.001 | 2097 | +0.08 [+0.05, +0.11] | <0.001 |
| Essay writing | Generic text feature | Words | 2097 | +5.43 [+3.99, +6.87] | +0.16 [+0.12, +0.20] | <0.001 | 2097 | +0.19 [+0.17, +0.22] | <0.001 |
| Essay writing | Generic text feature | Positive-word rate per 1k words | 2097 | -2.77 [-3.44, -2.11] | -0.18 [-0.23, -0.14] | <0.001 | 2097 | -0.04 [-0.06, -0.01] | 0.003 |

## Grant abstract

| Task | Family | Feature | N gap | Arm gap (95% CI) | Arm gap, SD units (95% CI) | Gap p | N panel | Panel assoc. per SD (95% CI) | Panel p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Grant abstract | Generic text feature | Rare-word rate per 1k words | 2089 | +2.51 [+1.74, +3.28] | +0.14 [+0.10, +0.18] | <0.001 | 2089 | +0.17 [+0.15, +0.20] | <0.001 |
| Grant abstract | Generic text feature | Quantitative detail rate | 2089 | +0.32 [+0.01, +0.63] | +0.04 [+0.00, +0.09] | 0.043 | 2089 | +0.24 [+0.22, +0.27] | <0.001 |
| Grant abstract | Generic text feature | Flesch-Kincaid grade | 2089 | +0.10 [+0.03, +0.16] | +0.06 [+0.02, +0.11] | 0.004 | 2089 | +0.10 [+0.08, +0.13] | <0.001 |
| Grant abstract | Generic text feature | MATTR-50 | 2089 | +0.00 [+0.00, +0.00] | +0.09 [+0.05, +0.13] | <0.001 | 2089 | +0.15 [+0.12, +0.17] | <0.001 |

## Incident postmortem

| Task | Family | Feature | N gap | Arm gap (95% CI) | Arm gap, SD units (95% CI) | Gap p | N panel | Panel assoc. per SD (95% CI) | Panel p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Incident postmortem | Generic text feature | Rare-word rate per 1k words | 2088 | +1.30 [+0.43, +2.17] | +0.06 [+0.02, +0.11] | 0.003 | 2088 | +0.13 [+0.11, +0.16] | <0.001 |
| Incident postmortem | Generic text feature | Paragraphs | 2088 | +0.23 [+0.02, +0.44] | +0.05 [+0.00, +0.09] | 0.033 | 2088 | +0.05 [+0.01, +0.08] | 0.005 |
| Incident postmortem | Generic text feature | MATTR-50 | 2088 | +0.00 [+0.00, +0.00] | +0.07 [+0.03, +0.12] | <0.001 | 2088 | +0.11 [+0.08, +0.14] | <0.001 |
