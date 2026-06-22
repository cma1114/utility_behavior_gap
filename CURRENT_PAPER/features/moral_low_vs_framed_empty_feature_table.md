# Moral Low Vs Framed Empty Feature Table

Rows combine current generic text features with existing task-specific LLM rubric coding.
Rows are selected symmetrically: clear arm gaps, clear panel associations, and absolute standardized arm gaps above the display threshold.
Main-text display threshold: standardized arm gap approximately `0.25 SD` or larger.

| Task | Dimension | Harmful-minus-empty gap (SD units, 95% CI) | Raw arm gap (95% CI) | Panel association (95% CI) |
| --- | --- | --- | --- | --- |
| Essay writing | Words | -0.41 [-0.45, -0.37] | -15.0 [-16.4, -13.6] | +0.23 [+0.19, +0.28] |
|  | Paragraphs | +0.32 [+0.27, +0.36] | +0.34 [+0.29, +0.39] | +0.08 [+0.04, +0.13] |
|  | Flesch-Kincaid grade | +0.26 [+0.21, +0.31] | +0.32 [+0.26, +0.38] | +0.08 [+0.04, +0.12] |
| Grant abstract | Words | -0.34 [-0.38, -0.30] | -20.0 [-22.5, -17.5] | +0.22 [+0.18, +0.26] |
|  | Stakeholder/context fit | -0.26 [-0.45, -0.07] | -0.22 [-0.39, -0.06] | +0.29 [+0.10, +0.48] |
|  | Evaluation rigor | -0.26 [-0.46, -0.07] | -0.25 [-0.44, -0.07] | +0.24 [+0.06, +0.42] |
|  | Risk mitigation | -0.28 [-0.47, -0.08] | -0.26 [-0.44, -0.08] | +0.39 [+0.19, +0.58] |
|  | Measurable impact | -0.29 [-0.48, -0.09] | -0.26 [-0.44, -0.09] | +0.35 [+0.16, +0.54] |
| Incident postmortem | Words | -0.29 [-0.33, -0.25] | -24.6 [-28.3, -20.9] | +0.31 [+0.27, +0.35] |

Outputs:
- CSV: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/moral_low_vs_framed_empty_feature_table.csv`
- LaTeX: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/moral_low_vs_framed_empty_feature_table.tex`
- selected detail: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/moral_low_vs_framed_empty_feature_table_selected_detail.csv`
