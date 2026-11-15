# Complete Utility and Framed-Empty Output Screen

Authorized scope: classify all remaining outputs in the main high-low utility
experiment and its framed-empty control pool. Seven actors, four tasks, four
utility domains: 8,400 high, 8,400 low, and 8,400 unique framed-empty outputs.
No new generations, quality judgments, utility fits, or rubric-feature calls.

Use the frozen paper pair tables, not dates or an arbitrary latest run. Match
every extracted text to its saved catalog hash and original task. Stream the
archive once, retaining only the selected records; do not unpack it. Reuse
reviewed labels only on exact output ID and text-hash matches. The expected
reuse is 2,100 low-policy, 1,036 high-policy, and 2,100 framed-empty labels:
5,236 total, leaving 19,964 new classifier requests.

The classifier is openai/gpt-5-mini, temperature 0, max_tokens 150, minimal
reasoning with reasoning excluded. Use the existing source-aware policy
screen's literal template and four definitions: clean, full_refusal,
partial_refusal, degenerate. Supply the full output and the original task
context, including Chinese source text for translations. Do not truncate
inputs. The classifier is not told the actor, high/low identity, fitted utility,
or panel verdict. This is not a new quality rubric or an attempt to exclude
ordinary weak writing.

Read every proposed exclusion against its task before accepting it, including
previously labelled non-clean outputs. Preserve the classifier's raw label and
response separately from the final manually reviewed label. Missing or suspect
completion metadata is also a review flag, not proof of refusal or truncation.
Do not turn a failed classifier request into a clean label. Review cannot
silently remove records or change their identities. Preserve all original
responses and paper results.

Report counts and percentages for all four categories overall and by actor,
task, and utility domain; the empty arm has no intrinsic incentive domain.
Report paired low-minus-high differences for (a) any refusal and (b) degeneracy,
with the paper's two-stage equal-cell bootstrap. These are descriptive 95%
intervals, not a large set of model-wise significance tests.

Recompute existing high-low, high-utility/framed-empty, and harmful/framed-empty
quality results after excluding any pair with a non-clean output. For harmful,
retain the existing harmful-side classifications; only its empty controls are
newly completed here. Reuse existing panel verdicts unchanged. Keep a before/
after table and explicit removed-pair accounting, overall and by task/domain.
Primary means retain the paper's equal-cell weights: actor/task/domain for
utility and actor/task for harmful, excluding panel ties. Aggregate intervals
use the same 5,000-draw cell-then-pair bootstrap; model-task sensitivity tables
retain the existing 28-cell exact Bonferroni convention.

Execution: the compact resumable request runner, 24 workers, $15 total cap,
bounded retries, disk guard, and progress every 30 seconds. Full requests and
responses are compressed; no credentials or account balances are published.
Evidence and report remain separate from CURRENT_PAPER.

Runner: `paper_reproduction/utility_output_screen.py`.
Evidence: `data/processed/utility_output_screen/`.
Report: `docs/reviewer_response/utility_output_screen/README.md`.
