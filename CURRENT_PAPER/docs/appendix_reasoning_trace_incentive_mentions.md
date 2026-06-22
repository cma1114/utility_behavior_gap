# Appendix Finding: Reasoning About Incentives Does Not Amplify the Utility Effect

## Question

The high-low utility experiment asks whether a model produces better output when success would fund an outcome it has assigned higher utility. In the reasoning-trace follow-up, we tested a more specific mechanism: whether explicitly reasoning about the success-contingent incentive amplifies the utility contrast.

The prediction is directional. If incentive reasoning strengthens the utility effect, then:

- when the high-utility side reasons about its assigned consequence, the high-utility output should be more likely to win;
- when the low-utility side reasons about its assigned consequence, the low-utility output should be less likely to win, so the high-utility output should be more likely to win.

## Data and coding

We reran the essay high-low utility condition with actor reasoning enabled and requested visible reasoning traces. The primary analysis used pairs where both the high- and low-utility outputs had readable returned reasoning text.

For each output, a lexical audit coded whether the readable reasoning trace mentioned:

- the incentive scaffold: competition, judging, sponsor, funding, intervention, reward, outcome, or consequence;
- target terms: generic consequence terms such as medical intervention, terminal illness, conservation intervention, policy change, or saving;
- the exact assigned consequence for that side;
- meta-recognition of the setup, such as hidden context, jailbreak/prompt-injection framing, alignment-test language, or moral-manipulation language.

The most direct test is the exact assigned consequence code, because it captures cases where the model explicitly reasons about the specific high- or low-utility outcome assigned to that output.

Scripts and outputs:

- script: `src/utility_behavior_gap/scripts/analyze_reasoning_incentive_mentions.py`
- script: `src/utility_behavior_gap/scripts/analyze_reasoning_mentions_vs_performance.py`
- mention audit: `outputs/analysis/reasoning_incentive_mentions__success_only_per_output.csv`
- pair-level performance analysis: `outputs/analysis/reasoning_mentions_vs_performance__both_traces_required_summary.md`
- exact-target examples: `outputs/analysis/reasoning_mentions_vs_performance__both_traces_required_examples_mentions_exact_assigned_target.md`

## Results

Across trace-bearing pairs available at the time of analysis, the high-low utility contrast remained null: 361 high-side wins, 375 low-side wins, and 1118 ties across 1854 included pairs.

Readable reasoning traces often noticed the incentive wrapper. Among trace-bearing successful outputs, 90.9% of high-side traces and 92.7% of low-side traces mentioned some part of the incentive scaffold. This shows that the models often represented the success-contingent setup internally, even though the final essays were instructed not to mention it.

However, explicit reasoning about the assigned consequence did not amplify the utility effect. For exact assigned-consequence mentions:

| Mention pattern | Pairs | High wins | Low wins | Ties | High win rate excluding ties | Net high-minus-low score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| High side only | 30 | 1 | 8 | 21 | 11.1% | -23.3 pp |
| Low side only | 89 | 21 | 12 | 56 | 63.6% | +10.1 pp |
| Both sides | 10 | 0 | 0 | 10 | n/a | 0.0 pp |
| Neither side | 1725 | 339 | 355 | 1031 | 48.8% | -0.9 pp |

The low-side-only cell is in the predicted direction: when only the low-utility output explicitly reasoned about its assigned consequence, the high-utility output won 63.6% of resolved comparisons. But the high-side-only cell goes sharply in the opposite direction: when only the high-utility output explicitly reasoned about its assigned consequence, the high-utility output won only 11.1% of resolved comparisons. The high-side-only cell is small, but it rules out the simple mechanism that explicit reasoning about the high-utility incentive reliably makes the high side try harder and win.

Broader mention categories also failed to support the amplification account. For generic target-term mentions, the mentioning side won 67 and lost 63 resolved comparisons. For meta-recognition of the setup, the mentioning side won 71 and lost 68 resolved comparisons. These are effectively null.

## Interpretation

The reasoning traces show that the incentive wrapper was often noticed. That is not surprising, because the incentive was explicitly present in the prompt. The important point is that noticing or explicitly reasoning about the incentive did not translate into higher-quality output for the incentivized side.

Thus, the trace analysis strengthens the main high-low result. The null high-low effect is not easily explained by models failing to perceive the success-contingent incentive. In many traces, models did perceive it, sometimes explicitly describing the sponsor, funding consequence, or hidden incentive structure. But this perception did not produce the pattern expected if stated utilities were acting as effective motivators.

The most defensible conclusion is narrow: visible incentive reasoning does not provide evidence for a mechanism in which models try harder for higher-utility consequences or sandbag lower-utility consequences. If anything, explicit fixation on the assigned consequence may sometimes reflect distraction or meta-reasoning rather than improved task performance, but that interpretation should remain secondary because the exact-target cells are small and exploratory.

## Caveat

This is an exploratory appendix analysis of returned reasoning summaries/traces, not a canonical main-text result. Some providers return summaries or partial trace text rather than full private chain-of-thought. The analysis therefore concerns the readable reasoning information returned by the API, not necessarily the model's complete internal computation.
