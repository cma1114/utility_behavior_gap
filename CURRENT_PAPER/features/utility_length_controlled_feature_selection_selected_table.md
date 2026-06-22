# Utility Length Controlled Feature Selection Selected Feature Table

Rows shown are exactly the selected rows from `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/utility_length_controlled_feature_selection.csv`.

Arm gap is left-minus-right at equal word count for non-word features. Panel association is the standardized feature coefficient in `panel_score ~ feature + words + actor`; for Words, it is the word-count coefficient without word-count control.

Arm gap in SD units divides the adjusted arm gap by the observed SD of the paired feature deltas within the same task. Use it to compare effect sizes across features with different raw units.

## Essay writing

| Task | Family | Feature | N gap | Arm gap (95% CI) | Arm gap, SD units (95% CI) | Gap p | N panel | Panel assoc. per SD (95% CI) | Panel p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Essay writing | Generic text feature | MATTR-50 | 2100 | +0.00 [+0.00, +0.00] | +0.04 [+0.00, +0.08] | 0.047 | 2100 | +0.17 [+0.14, +0.19] | <0.001 |

## Grant abstract

| Task | Family | Feature | N gap | Arm gap (95% CI) | Arm gap, SD units (95% CI) | Gap p | N panel | Panel assoc. per SD (95% CI) | Panel p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Grant abstract | Generic text feature | Words | 2100 | +4.66 [+1.80, +7.52] | +0.07 [+0.03, +0.11] | 0.001 | 2100 | +0.24 [+0.20, +0.28] | <0.001 |
| Grant abstract | Generic text feature | Positive-word rate per 1k words | 2100 | -0.46 [-0.93, +0.00] | -0.04 [-0.09, +0.00] | 0.050 | 2100 | -0.14 [-0.16, -0.11] | <0.001 |
