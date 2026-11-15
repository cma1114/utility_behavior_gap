# Harmful and Low policy Versus Framed Empty

Both contrasts are against framed empty. Gaps are outcome-minus-control differences in SD units, with nominal 95% confidence intervals. Negative gaps mean fewer words or weaker rubric-rated quality; for other generic features, they indicate less of that feature, not necessarily worse quality. A positive panel association means that more of the feature predicts greater panel preference for the outcome-side output. Non-word features adjust for paired word-count difference and actor in both models; Words adjusts for actor only. SDs are calculated within each task and contrast.

| Task | Dimension | Harmful: gap | Harmful: panel association | Low policy: gap | Low policy: panel association |
| --- | --- | --- | --- | --- | --- |
| Essay writing | Words | **-0.41 [-0.45, -0.37]** | +0.23 [+0.19, +0.28] | **-0.29 [-0.37, -0.21]** | +0.31 [+0.23, +0.38] |
| Essay writing | Paragraphs | **+0.32 [+0.27, +0.36]** | +0.08 [+0.04, +0.13] | +0.24 [+0.16, +0.32] | +0.10 [+0.02, +0.18] |
| Essay writing | Flesch-Kincaid grade | **+0.26 [+0.21, +0.31]** | +0.08 [+0.04, +0.12] | +0.13 [+0.04, +0.22] | +0.06 [-0.02, +0.13] |
| Grant abstract | Words | **-0.34 [-0.38, -0.30]** | +0.22 [+0.18, +0.26] | -0.01 [-0.09, +0.07] | +0.19 [+0.10, +0.28] |
| Grant abstract | Feasibility/readiness | -0.15 [-0.34, +0.04] | +0.32 [+0.13, +0.50] | **-0.27 [-0.45, -0.08]** | +0.33 [+0.15, +0.51] |
| Grant abstract | Stakeholder/context fit | **-0.26 [-0.45, -0.07]** | +0.29 [+0.10, +0.48] | **-0.27 [-0.45, -0.09]** | +0.25 [+0.06, +0.43] |
| Grant abstract | Evaluation rigor | **-0.26 [-0.46, -0.07]** | +0.24 [+0.06, +0.42] | -0.18 [-0.37, +0.00] | +0.30 [+0.12, +0.48] |
| Grant abstract | Risk mitigation | **-0.28 [-0.47, -0.08]** | +0.39 [+0.19, +0.58] | **-0.41 [-0.58, -0.23]** | +0.31 [+0.13, +0.49] |
| Grant abstract | Measurable impact | **-0.29 [-0.48, -0.09]** | +0.35 [+0.16, +0.54] | -0.25 [-0.43, -0.06] | +0.27 [+0.09, +0.46] |
| Incident postmortem | Words | **-0.29 [-0.33, -0.25]** | +0.31 [+0.27, +0.35] | -0.16 [-0.25, -0.07] | +0.26 [+0.18, +0.34] |

Rows meet the paper's criteria in at least one contrast: absolute gap >= 0.25 SD, and both gap and panel-association 95% intervals exclude zero. Both estimates are shown for every selected row, including uncertain and smaller counterparts. Bold gaps mark the contrasts meeting all three criteria. Selection uses unrounded values; rounding to 0.25 does not itself mean a row qualifies. These are the paper's feature intervals, not familywise-adjusted intervals. This is a descriptive comparison, not a test of the difference between contrasts. No dimensions qualify in either contrast for: Translation.

## Reproduction

Reconstruct the recorded panels and rubric ratings, refit both contrasts, and
verify this table without API calls or a private checkout:

```sh
python paper_reproduction/reproduce.py run --step policy
```

Regenerated files are written to `outputs/reproduced/policy/`, not over this
reference table. See [verification](../PUBLIC_VERIFICATION.md).
