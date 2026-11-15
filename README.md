# When Preferences Fail to Become Incentives

Code, frozen analysis inputs, and paper artifacts for the Utility-Behavior Gap
experiments. This is the public replication repository, not the private research backup.

## Start Here

- **Figures and tables:** [CURRENT_PAPER](CURRENT_PAPER/README.md).
- **Exact experimental prompts:** [canonical prompt book](CURRENT_PAPER/docs/canonical_prompt_book.md).
- **Exact judging prompts:** [judge prompts](CURRENT_PAPER/docs/judge_prompts.md).
- **Reproduce and verify results:** [paper reproduction guide](paper_reproduction/README.md).
- **Current versus historical:** [artifact map](paper_reproduction/ARTIFACTS.md).
- **Policy follow-up and harmful comparison:** [side-by-side feature table](docs/reviewer_response/policy_framed_empty/features/harmful_and_low_policy_feature_table.md) and [qualitative findings](docs/reviewer_response/policy_framed_empty/features/findings.md).

The current paper uses high-low utility, direct user-prompt exhortation, framed
role cues, and harmful-outcome comparisons with baselines. Amount experiments
remain available as historical analyses; they are **not a primary condition of
the current paper**. A filename containing `canonical` does not establish its status.

## Reproduce Locally

Python 3.13 was used for verification. From this directory:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r paper_reproduction/requirements.txt
python paper_reproduction/reproduce.py audit
python paper_reproduction/reproduce.py run
```

These commands make **no model API calls**. They recompute statistics and figures
from the included pair-level outcomes, feature measurements, rubric ratings,
and saved reasoning labels, then compare them with the accepted results. Outputs
go under `outputs/reproduced/`; `CURRENT_PAPER/` is never overwritten.
See the guide for per-analysis commands and the distinction between reproducing
an analysis and recollecting model responses.

## Reasoning Appendix

The original reasoning-on essay experiment contains 4,200 outputs / 2,100 pairs.
The updated suspicion-exclusion analysis adds 1,620 Qwen3.5 follow-up pairs.
Reproduction uses preserved **local Opus labels**, not the rejected API-Opus
labels. The recovered instructions are in
[the stance specification](analysis_specs/reasoning_trace_stance_classifier.md).
See [the appendix report](CURRENT_PAPER/docs/appendix_reasoning_trace_analysis.md)
for results and scope caveats.

## Repository Boundaries

`CURRENT_PAPER/` includes current and retained historical artifacts: use the
explicit artifact map, not timestamps. `src/` retains collection and analysis
tools; `paper_reproduction/` is the tested offline entry point. Large raw API
caches, credentials, and exploratory follow-ups are not automatically published.
No private checkout is needed for the documented offline analyses.

The policy-versus-framed-empty reviewer follow-up is now included in the offline
runner (`--step policy` for that part alone). It reconstructs the saved panel
judgments and rubric ratings, refits both policy and harmful features, and
checks the parallel table. Its compact evidence package preserves the exact
source texts, requests, responses, exclusions, and collection code; no paid
calls are needed to reproduce the analysis.
