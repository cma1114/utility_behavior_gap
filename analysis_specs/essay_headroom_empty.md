# Essay Headroom Versus Framed Empty

Authorized scope: essays only, both strong effort and world-class role, all seven
paper actors. Reuse existing generations; no new actor calls and no new rubric
or refusal-classifier batch. Keep original experiments and results unchanged.

Two planned comparisons, each 7 actors x 30 topics x 10 saved repetitions =
2,100 pairs. Both use the same 2,100 matched framed-empty control outputs.
The treatment outputs retain the neutral children's-hospital intervention;
the control has competition framing but no intervention. This is a test of
headroom above that baseline, not a cue-only manipulation on a fixed empty
prompt. No actor prompt, generation parameter, or essay text is edited.

Select source runs using the paper's frozen pair tables, not timestamps or
their outcomes. Match actor, exact topic, and repetition. Effort has two blocks
of five repetitions: map base 0-4 to empty 0-4 and block01/0-4 to empty 5-9.
Keep both original and normalized repetition identifiers. Role already uses
0-9. Missing outputs remain in the accounting and are not regenerated.

Use the paper's framed-empty bridge panel and exact essay judge template:
Claude Haiku 4.5, Gemini 3.1 Flash Lite Preview, GPT-5-mini; one randomized
orientation per judge, copied from the corresponding archived high-utility
comparison. Temperature 0, output cap 120, minimal/excluded judge reasoning.
Verify the current template against all 6,300 reference essay requests first.
Two votes for a side produce a panel win; otherwise tie. Parseable capped judge
verdicts are retained, consistent with that reference protocol.

Check all source outputs for missing/empty text and completion failures.
Flag unusual length, repetitive text, and refusal-like language for manual
inspection. Do not turn lexical flags into automatic semantic exclusions.
Preserve all source data and exclusion decisions. This does not claim that
all responses have received a new semantic classifier label.

Primary aggregate: equal-weight average of actor win rates, ties excluded,
with the paper's cell-then-pair bootstrap (5,000 draws, seed 20260615).
Sensitivity: count ties as half-wins with the same aggregation. Model intervals
use the paper's exact Bonferroni method over the 14 planned cells (two essay
contrasts x seven models), not a fictitious four-task family. No selecting
sample size, arms, models, or inference rules after seeing new verdicts.

Runner: `paper_reproduction/essay_headroom_empty.py`. API requests require
`judge --run`; preparation, status, and analysis are offline. Use 24 workers,
bounded transient retries, a $15 total collection budget, and a disk guard.
Failed attempts and provider-reported costs are retained. A returned HTTP 200
alone is not success: require a parsed verdict. Completed requests are skipped
on resume. Immutable compressed inputs and append-only compressed responses
avoid copying the large archive or writing duplicate raw output logs.

Evidence: `data/processed/essay_headroom_empty/`.
Report: `docs/reviewer_response/essay_headroom_empty/README.md`.
No changes to `CURRENT_PAPER` or automatic promotion to a paper result.

## Execution Exception

One Gemini judgment repeatedly returned `MALFORMED_RESPONSE` with null content.
After ten default-route attempts, one explicitly recorded Google AI Studio
priority-tier retry also failed. The model, prompt, and generation parameters
were unchanged. The analysis leaves this GLM role panel unresolved, excluding
the entire pair rather than scoring two judges or inventing a tie. Reproduction
requires `analyze --allow-unresolved`; default analysis still refuses incomplete
judging. The original request remains available for resumption. This exception
was handled before inspecting the new aggregate results.
