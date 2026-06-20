# Appendix Findings To Carry Forward

Status: paper planning note. This file records appendix-relevant analyses that should not be lost in the exploratory output directory. It is not a replacement for the underlying result files.

## Confirmed Paper-Relevant Findings

### 1. Effort-Dose Test

The essay-only mid-effort prompt produced an intermediate effect between framed neutral and the full max-effort prompt.

- Mid-effort prompt: `This essay is extremely important - give your absolute best effort!`
- Control: framed-neutral essay prompt.
- Max-effort comparison: paper-ready user-prompt direct-instruction condition.
- System prompts: blank in all three arms.

Main result: mid effort improved essay quality relative to framed neutral in the pooled analysis, but was weaker and less universal than the full max-effort instruction. This supports an effort-dose interpretation, especially as appendix evidence: stronger exhortation produces larger and more consistent quality gains.

Primary records:

- `docs/appendix_effort_dose_test.md`
- `outputs/paper_ready/results/appendix_effort_dose_summary.md`
- `outputs/paper_ready/figures/appendix_effort_dose_mid_vs_neutral.png`
- `outputs/paper_ready/figures/appendix_effort_dose_combined.png`

Paper placement: appendix, with a short main-text mention if discussing whether the direct-instruction effect is graded rather than binary.

### 2. Reasoning-On High-Low Utility Test

The essay high-low utility condition was rerun with actor reasoning enabled and visible reasoning traces requested.

Main result: enabling reasoning did not make the high-utility side perform better. The essay high-low effect remained near chance overall, with no model-level cell showing a corrected positive effect.

Primary record:

- `outputs/analysis/highlow_reasoning_traces_medium__tasks-essay__7-actors_summary.md`

Paper placement: appendix. This is useful negative evidence against the explanation that the high-low null result occurs only because actors lacked enough deliberation.

### 3. Reasoning Trace Incentive-Mention Analysis

The visible reasoning traces were audited for whether models mentioned the incentive scaffold, generic target terms, the exact assigned consequence, or meta-recognition of the experimental setup.

Main result: models often noticed the incentive wrapper, but this did not translate into better performance for the side that noticed or explicitly reasoned about it. The trace analysis therefore does not support a mechanism in which reasoning about the incentive amplifies utility motivation.

Primary records:

- `CURRENT_PAPER/docs/appendix_reasoning_trace_incentive_mentions.md`
- `outputs/analysis/reasoning_incentive_mentions__success_only_summary.md`
- `outputs/analysis/reasoning_mentions_vs_performance__both_traces_required_summary.md`

Paper placement: appendix, with a short main-text pointer in the section interpreting the high-low utility null.

### 4. DeepSeek Essay Scaffold-Length Diagnostic

DeepSeek V3.2 essay outputs showed an unusually large gap in the bridge comparisons: high-utility essays lost badly to both framed-empty and R0 baselines, while high- and low-utility essays were near parity with each other.

Follow-up checks indicate this is not a source-record mismatch. The high-utility records in the bridge analyses are the same `hl_high` outputs used in the canonical high-low condition. Instead, the pattern appears to be a scaffold effect: the hidden sponsor/intervention prompt wrapper makes DeepSeek essays shorter and more conservative. In DeepSeek essays, utility, moral, amount, and framed-neutral arms were around 298-315 words, while framed-empty and R0 baselines were around 345-353 words. The max-effort direct-instruction arm also reached about 346 words, suggesting that the explicit effort cue can override this compression.

Interpretation: high versus low utility stays near chance because both sides share the same hidden sponsor/intervention scaffold. High utility loses to framed empty and R0 because those baselines remove some or all of that scaffold and elicit longer, more developed essays.

Primary records:

- `CURRENT_PAPER/results/high_utility_vs_framed_empty_pair_outcomes.csv`
- `CURRENT_PAPER/results/high_utility_vs_r0_pair_outcomes.csv`
- `CURRENT_PAPER/results/highn_condition_results_pair_outcomes.csv`

Paper placement: appendix diagnostic for the high-utility baseline comparisons. This should be described as evidence that the hidden sponsorship/intervention scaffold can itself depress quality for some model-task cells, separate from utility value.

## Sandbagging Diagnostic

### Moral Low Versus R0

Moral-low outputs were compared against a bare R0 task baseline.

Purpose: this comparison tests whether bad-cause moral framing pushes outputs below a bare-task baseline. That is the relevant sandbagging question: not merely whether good causes beat bad causes, but whether bad causes depress performance relative to an unincentivized task.

Main result: pooled data favored R0 over moral low, but the equal-cell aggregate was weaker and crossed zero. This is therefore not strong evidence that moral-low framing reliably induces sandbagging across models and tasks. It is still an important result to report because it directly constrains the sandbagging interpretation.

Primary record:

- `outputs/analysis/moral_r0_bridge__combined_7runs__summary.md`

Paper placement: report as a planned sandbagging diagnostic, probably in the moral-results section or appendix depending on space. The correct interpretation is not to hide the result, but to state that this comparison gives at most mixed evidence for bad-cause sandbagging under the primary equal-cell analysis.
