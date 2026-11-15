# Low Policy and Harmful: Comparison Using the Paper's Feature Analysis

Both comparisons below use **framed empty**: competition/judging framing without an outcome. They do not use R0. The new low-policy analysis uses the same eight generic features, task-specific rubric, coder, length-adjustment models, and symmetric 0.25-SD display criterion as the harmful paper analysis.

## Are They Worse in the Same Way?

**The profiles overlap substantially, especially in grant abstracts, but the shared pattern is not uniformly worse writing on every dimension.** The table below describes the full feature estimates, including effects below the paper's display threshold. The [parallel numerical table](harmful_and_low_policy_feature_table.md) shows the union of paper-selected rows, with both contrasts' estimates and panel associations on every row. A [LaTeX version](harmful_and_low_policy_feature_table.tex) is available too.

| Task | Harmful versus framed empty | Low policy versus framed empty | What is shared, and what differs? |
| --- | --- | --- | --- |
| Essay | Shorter, with more paragraphs and higher Flesch-Kincaid grade. Little clear evidence of a broad decline across the argument-quality rubric, although thesis/stakes framing is lower. | Also shorter, with more paragraphs and higher grade. No task-specific rubric gap is clearly different from zero. | The clearest common change is shorter, more segmented prose, not a demonstrated shared collapse in argument quality. More paragraphs is positively associated with panel preference in both, so some changes counteract rather than explain the quality decline. |
| Grant abstract | Shorter proposals, weaker evaluation rigor, risk mitigation, measurable impact, and stakeholder/context fit. All seven grant-rubric point estimates are negative. | Not shorter, but weaker feasibility/readiness, risk mitigation, and stakeholder/context fit meet the full display rule. Measurable impact and intervention specificity also decline; all seven rubric point estimates are negative. | Strong overlap in practical proposal quality: explaining how the project would work, manage risks, fit its setting, and deliver results. Risk mitigation and stakeholder/context fit are especially clear in both. The important difference is length: the low-policy deficit occurs without a shorter abstract. |
| Incident postmortem | Shorter, but not broadly worse on the task-specific rubric. Quantitative detail increases, and operational realism has a small positive shift. | Also shorter, with more quantitative detail and lexical diversity. No broad decline across root-cause analysis, timelines, action items, or other task-specific dimensions. | Shared shortening, accompanied by some favorable changes. Neither resembles the broad grant-quality deficit. The harmful shortening is larger; the low-policy shortening falls below the 0.25-SD display threshold. |
| Translation | No dimension meets the full paper display rule. Some small generic changes and a nominal additions/omissions deficit appear in the full estimates. | No feature gap has a nominal 95% interval excluding zero. | No established common degradation pattern. |

The qualitative overlap is broader than a comparison of selected-row counts suggests. For example, low-policy measurable impact narrowly misses the 0.25-SD threshold but its estimate is close to harmful's. Conversely, a feature being significant in only one contrast does not establish that the two effects differ. These are comparisons of observed profiles, not formal tests of mechanism or between-contrast differences.

The clearest resemblance is in **grant abstracts**, not uniform deterioration across all four tasks. Low-policy grants show length-independent deficits in feasibility/readiness, risk mitigation, and stakeholder/context fit, all positively associated with panel preference. Risk mitigation and stakeholder/context fit also met the paper's display criterion for harmful grants.

## Grant Abstracts

Adjusted feature gaps in paired-difference SD units, with nominal 95% confidence intervals. Negative means worse rubric-rated quality on the outcome side relative to framed empty. All seven grant dimensions are shown, not only the selected ones.

| Dimension | Low policy minus framed empty | Harmful minus framed empty |
| --- | --- | --- |
| Problem significance | -0.14 [-0.32, +0.05] | -0.07 [-0.26, +0.11] |
| Intervention specificity | -0.31 [-0.49, -0.13] | -0.23 [-0.42, -0.04] |
| Evaluation rigor | -0.18 [-0.37, 0.00] | -0.26 [-0.46, -0.07] |
| Feasibility/readiness | -0.27 [-0.45, -0.08] | -0.15 [-0.34, +0.04] |
| Risk mitigation | -0.41 [-0.58, -0.23] | -0.28 [-0.47, -0.08] |
| Measurable impact | -0.25 [-0.43, -0.06] | -0.29 [-0.48, -0.09] |
| Stakeholder/context fit | -0.27 [-0.45, -0.09] | -0.26 [-0.45, -0.07] |

All seven point estimates are negative in both comparisons. This is descriptive convergence, not seven independent replications: the rubric dimensions are correlated. The comparisons use different sampled pairs, and this juxtaposition does not formally test equality of effects.

Three low-policy rows meet the full paper display rule: feasibility/readiness, risk mitigation, and stakeholder/context fit. Intervention specificity has a clear negative gap, but its panel-association interval crosses zero. Measurable impact has a clear negative gap and positive panel association, but its unrounded standardized gap is approximately -0.248, just below the unchanged 0.25 threshold. Evaluation rigor's upper bound is slightly below zero before rounding, but its gap does not survive Holm correction within the seven grant dimensions. These rows are retained in the full table rather than treated as evidence of no effect.

**This is not merely a length deficit.** The low-policy grant word-count difference is -0.6 words [95% CI: -6.0, +4.7], compared with -20.0 [-22.5, -17.5] for harmful. The rubric estimates above additionally adjust for paired word-count difference and actor. Five of the seven low-policy grant gaps survive Holm correction within the grant rubric family: feasibility, intervention specificity, measurable impact, risk mitigation, and stakeholder/context fit. The compact paper-style table uses the original nominal-CI plus effect-size selection rule, not a newly substituted multiple-testing rule.

## Other Tasks

- **Essay:** low-policy essays are shorter by 10.4 words [95% CI: 7.6, 13.2 fewer], and length is positively associated with panel preference. This matches the direction in harmful, where essays are shorter by 15.0 words [13.6, 16.4 fewer]. No low-policy task-specific rubric dimension has a gap clearly different from zero. There are countervailing generic shifts: more paragraphs (+0.24 SD [+0.16, +0.32]) and quantitative detail (+0.13 SD [+0.03, +0.22]), both positively associated with panel preference; their gaps are below the 0.25-SD display threshold.
- **Incident postmortem:** low-policy outputs are shorter by 12.6 words [5.9, 19.4 fewer], but the standardized difference is only -0.16 SD. Lexical diversity and quantitative detail instead increase, each by about +0.14 SD with intervals excluding zero, and both are positively associated with panel preference. No feature meets the full paper display rule. This is a mixed pattern, not the broad grant-quality deficit.
- **Translation:** none of the feature gaps has a 95% interval excluding zero. No feature meets the display rule. This is not a claim that all possible quality differences are zero.

## Scope and Verification

The generic analysis covers all 2,090 clean pairs: 523 essay, 518 grant, 525 postmortem, and 524 translation. Rubric coding covers 120 randomly sampled, actor-balanced pairs per task. Ten refusal/degenerate pairs were removed before sampling, using the completed source-aware screening. All 480 codings completed; one malformed JSON response was retried, for 481 total attempts and $0.90085607 in provider-reported cost. No responses were truncated, and no token-cap increases were needed. Original requests, unsuccessful responses, and successful responses remain saved.

The integration test `test_reproduce_all_59_harmful_paper_feature_estimates` recomputes the harmful paper's 32 generic and 27 rubric estimates, including gap and panel-association confidence intervals. They match the saved paper estimates to floating-point precision. This verifies methodological parity rather than merely giving the analyses similar labels.

As in the paper, the task-specific features are automatic comparative ratings, not independently verified factual measures. Manual checking of the first four calls found plausible task-focused explanations but also an erroneous translation rationale: the coder treated incorrect personal-name renderings as standard English alternatives (`policy_low_framed_empty:1f39db0d882c75c4`). The recorded rating was not selectively edited or replaced. This illustrates coder fallibility and cautions against treating any one rating or rationale as ground truth; no error-rate estimate is inferred from four checks.

The observed grant deficits are compatible with the harmful-condition quality pattern. They do not establish deliberate sandbagging, the same internal mechanism, or complete mediation of the panel effect. They also do not, by themselves, explain the full high-minus-low policy effect; that question requires the high-side baseline comparison as well.

## Files

- [Selected paper-style table and methods](README.md)
- [All generic and task-specific feature estimates, by task](all_features.md)
- [Full numerical results](feature_results.csv)
- [LaTeX table](feature_table.tex)
- Harmful reference: `CURRENT_PAPER/features/moral_low_vs_framed_empty_length_controlled_feature_selection.csv` and `CURRENT_PAPER/features/moral_low_vs_framed_empty_feature_table.md`.
