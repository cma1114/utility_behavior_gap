# GPT-5.4 Mini High-Utility-vs-R0 Outlier Diagnostic

Bridge run: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/highlow-r0-bridge__2026-06-16_22-48-56Z__hash-934fbfa8fc5d`

This is a local diagnostic of the GPT-5.4-mini cells in the high-utility versus R0 bridge. No API calls are made.

## Main Cell Results

| task | resolved | high wins | R0 wins | ties | win rate | FWER-CI positive |
|---|---:|---:|---:|---:|---:|---|
| Essay writing | 127 | 114 | 13 | 23 | 89.8% | True |
| Grant abstract | 148 | 81 | 67 | 2 | 54.7% | False |
| Incident postmortem | 148 | 134 | 14 | 2 | 90.5% | True |
| Translation | 126 | 62 | 64 | 24 | 49.2% | False |

## Interpretation Check

- The outlier is concentrated in essay and incident postmortem.
- The effect is present in every domain for those two tasks, including political, which argues against a utility-content explanation.
- All GPT-5.4-mini high and R0 outputs in this bridge ended with `finish_reason=stop`; none were empty.
- R0 is not reused: each task has 150 distinct high outputs and 150 distinct R0 outputs.
- The high side includes the blind evaluation/sponsor/funding wrapper; R0 is the bare task prompt. This bridge therefore tests high-utility framing against a bare baseline, not utility content alone.
- For postmortems, the high side has many fewer bracket placeholders and many more concrete time/number details; judge rationales explicitly reward specificity, timelines, owners, due dates, and technical detail.

## Domain Breakout For Outlier Tasks

| task | domain | resolved | high wins | R0 wins | ties | win rate |
|---|---|---:|---:|---:|---:|---:|
| essay | animals | 29 | 26 | 3 | 9 | 89.7% |
| essay | countries | 34 | 32 | 2 | 4 | 94.1% |
| essay | political | 31 | 26 | 5 | 6 | 83.9% |
| essay | religions | 33 | 30 | 3 | 4 | 90.9% |
| incident_postmortem | animals | 38 | 33 | 5 | 0 | 86.8% |
| incident_postmortem | countries | 37 | 34 | 3 | 1 | 91.9% |
| incident_postmortem | political | 37 | 32 | 5 | 0 | 86.5% |
| incident_postmortem | religions | 36 | 35 | 1 | 1 | 97.2% |

## Output Artifact Means

| task | condition | n | non_stop | empty | words | number_tokens | bracket_placeholders | date_placeholders | specific_dates | time_mentions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| essay | hl_high | 150 | 0 | 0 | 349.88 | 0.43 | 0.00 | 0.00 | 0.81 | 0.08 |
| essay | r0 | 150 | 0 | 0 | 339.24 | 0.50 | 0.00 | 0.00 | 1.06 | 0.08 |
| grant_proposal_abstract | hl_high | 150 | 0 | 0 | 659.82 | 0.57 | 0.00 | 0.00 | 0.73 | 0.01 |
| grant_proposal_abstract | r0 | 150 | 0 | 0 | 649.27 | 0.25 | 0.00 | 0.00 | 1.07 | 0.00 |
| incident_postmortem | hl_high | 150 | 0 | 0 | 822.89 | 33.09 | 0.97 | 0.71 | 1.01 | 9.30 |
| incident_postmortem | r0 | 150 | 0 | 0 | 838.30 | 17.62 | 3.06 | 2.28 | 0.25 | 3.21 |
| translation | hl_high | 150 | 0 | 0 | 48.87 | 1.65 | 0.00 | 0.00 | 0.17 | 0.00 |
| translation | r0 | 150 | 0 | 0 | 48.41 | 1.63 | 0.00 | 0.00 | 0.17 | 0.00 |

## Paired Feature Deltas

Deltas are high minus R0.

| task | target_win | words_delta_high_minus_r0 | number_tokens_delta_high_minus_r0 | bracket_placeholders_delta_high_minus_r0 | specific_dates_delta_high_minus_r0 | time_mentions_delta_high_minus_r0 |
| --- | --- | --- | --- | --- | --- | --- |
| essay | False | 0.92 | -0.19 | 0.00 | -0.56 | -0.06 |
| essay | True | 13.71 | -0.04 | 0.00 | -0.16 | 0.02 |
| grant_proposal_abstract | False | -3.04 | 0.22 | 0.00 | -0.38 | 0.03 |
| grant_proposal_abstract | True | 22.12 | 0.41 | 0.00 | -0.31 | 0.00 |
| incident_postmortem | False | -30.94 | 2.75 | -1.44 | -0.06 | 2.94 |
| incident_postmortem | True | -13.55 | 16.99 | -2.16 | 0.85 | 6.47 |
| translation | False | 0.18 | 0.01 | 0.00 | 0.00 | 0.00 |
| translation | True | 0.85 | 0.02 | 0.00 | 0.00 | 0.00 |

## Judge Vote Split

| task | judge_model | hl_high | r0 | tie |
| --- | --- | --- | --- | --- |
| essay | anthropic/claude-haiku-4.5 | 122.00 | 23.00 | 5.00 |
| essay | google/gemini-3.1-flash-lite-preview | 118.00 | 28.00 | 4.00 |
| essay | openai/gpt-5-mini | 68.00 | 13.00 | 69.00 |
| grant_proposal_abstract | anthropic/claude-haiku-4.5 | 74.00 | 76.00 | 0.00 |
| grant_proposal_abstract | google/gemini-3.1-flash-lite-preview | 87.00 | 63.00 | 0.00 |
| grant_proposal_abstract | openai/gpt-5-mini | 86.00 | 60.00 | 4.00 |
| incident_postmortem | anthropic/claude-haiku-4.5 | 119.00 | 31.00 | 0.00 |
| incident_postmortem | google/gemini-3.1-flash-lite-preview | 138.00 | 12.00 | 0.00 |
| incident_postmortem | openai/gpt-5-mini | 127.00 | 20.00 | 3.00 |
| translation | anthropic/claude-haiku-4.5 | 63.00 | 83.00 | 4.00 |
| translation | google/gemini-3.1-flash-lite-preview | 71.00 | 70.00 | 9.00 |
| translation | openai/gpt-5-mini | 55.00 | 52.00 | 43.00 |

## Comparison Against Related GPT-5.4-Mini Results

| comparison | task | resolved_n | target_wins | target_losses | ties | target_win_rate_excluding_ties | familywise_ci_positive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| high_utility_vs_r0 | essay | 127 | 114 | 13 | 23 | 0.898 | True |
| high_utility_vs_r0 | grant_proposal_abstract | 148 | 81 | 67 | 2 | 0.547 | False |
| high_utility_vs_r0 | incident_postmortem | 148 | 134 | 14 | 2 | 0.905 | True |
| high_utility_vs_r0 | translation | 126 | 62 | 64 | 24 | 0.492 | False |
| high_low | essay | 110 | 57 | 53 | 190 | 0.518 | False |
| high_low | grant_proposal_abstract | 127 | 68 | 59 | 173 | 0.535 | False |
| high_low | incident_postmortem | 164 | 85 | 79 | 136 | 0.518 | False |
| high_low | translation | 124 | 58 | 66 | 176 | 0.468 | False |
| high_utility_vs_framed_neutral | essay | 106 | 51 | 55 | 194 | 0.481 | False |
| high_utility_vs_framed_neutral | grant_proposal_abstract | 119 | 67 | 52 | 181 | 0.563 | False |
| high_utility_vs_framed_neutral | incident_postmortem | 165 | 133 | 32 | 135 | 0.806 | True |
| high_utility_vs_framed_neutral | translation | 111 | 50 | 61 | 189 | 0.450 | False |
| direct_instruction | essay | 141 | 133 | 8 | 159 | 0.943 | True |
| direct_instruction | translation | 172 | 81 | 91 | 278 | 0.471 | False |
| direct_instruction | incident_postmortem | 158 | 123 | 35 | 142 | 0.778 | True |
| direct_instruction | grant_proposal_abstract | 126 | 105 | 21 | 174 | 0.833 | True |

