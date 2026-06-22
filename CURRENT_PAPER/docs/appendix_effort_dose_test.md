# Appendix Note: Essay Effort-Dose Test

Status: appendix/exploratory. This is not part of the canonical main direct-instruction condition unless explicitly promoted.

## Question

Does a milder exhortation produce an intermediate quality effect between framed neutral and the full max-effort user prompt?

## Design

- Task: essay writing only.
- Control: framed-neutral essay prompt.
- Mid-effort treatment: the same framed-neutral prompt with the sentence `This essay is extremely important - give your absolute best effort!` appended at the end of the user prompt.
- Strong treatment for comparison: the existing paper-ready max-effort user prompt.
- System prompts: blank for neutral, mid-effort, and max-effort conditions.
- Judges: the same three-judge panel, with both presentation orders judged.
- Sample: 150 judged essay pairs per actor for mid versus neutral.

The mid-effort run reuses the same framed-neutral outputs as the current headroom/direct-instruction runs. Therefore, mid versus neutral and max-effort versus neutral are direct judged contrasts against the same neutral baseline.

## Main Result

The mid-effort prompt produced a moderate aggregate effect, smaller than the max-effort prompt.

| contrast | pairs | treatment wins | neutral wins | ties | win rate excluding ties | net score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mid effort vs framed neutral | 1050 | 236 | 178 | 636 | 0.570 | 0.055 |
| Max effort vs framed neutral | 1050 | 410 | 72 | 568 | 0.851 | 0.322 |

At the model level, the mid-effort effect is clearly positive for GPT-5.4 mini and Kimi K2.5 after Holm correction, positive but not correction-significant for GLM-5.1, and near zero for the other four models. This is consistent with a graded effect overall, but it is not as broad or reliable as the full max-effort prompt.

## Interpretation

This supports an effort-dose interpretation for essay writing: a short, moderate exhortation can improve judged quality relative to framed neutral, but the larger max-effort instruction produces a much stronger and more consistently positive effect across models.

The appropriate appendix framing is that the main direct-instruction result is not merely a binary artifact of adding any effort-related text. The strength and specificity of the exhortation matter.

## Outputs

- Paper-ready figure: `CURRENT_PAPER/figures/appendix_effort_dose_mid_vs_neutral.png`
- Paper-ready combined dose figure: `CURRENT_PAPER/figures/appendix_effort_dose_combined.png`
- Paper-ready summary: `CURRENT_PAPER/results/appendix_effort_dose_summary.md`
- Paper-ready summary CSV: `CURRENT_PAPER/results/appendix_effort_dose_summary.csv`
- Paper-ready plot data: `CURRENT_PAPER/results/appendix_effort_dose_lollipop_data.csv`
- Figure: `outputs/analysis/figures/framed_user_mid_vs_neutral_lollipop.png`
- Combined dose figure: `outputs/analysis/figures/framed_user_effort_dose_lollipops.png`
- Summary Markdown: `outputs/analysis/framed_user_mid_effort_essay_dose_summary.md`
- Summary CSV: `outputs/analysis/framed_user_mid_effort_essay_dose_summary.csv`
- Plot data: `outputs/analysis/framed_user_effort_dose_lollipop_data.csv`

## Scripts

- Prepare jobs: `src/utility_behavior_gap/scripts/prepare_framed_user_mid_effort_essay_jobs.py`
- Run wrapper: `src/utility_behavior_gap/scripts/run_framed_user_mid_effort_essay_actor.sh`
- Plot dose figures: `src/utility_behavior_gap/scripts/plot_framed_user_effort_dose.py`
