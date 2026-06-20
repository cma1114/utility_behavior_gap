import importlib
from pathlib import Path


ENTRYPOINT_MODULES = [
    "utility_behavior_gap.scripts.aggregate_judgments",
    "utility_behavior_gap.scripts.aggregate_results",
    "utility_behavior_gap.scripts.analyze_amount_pooled",
    "utility_behavior_gap.scripts.analyze_essay_rhetoric_features",
    "utility_behavior_gap.scripts.analyze_highlow_cell_mechanisms",
    "utility_behavior_gap.scripts.analyze_highlow_vs_direct_judging",
    "utility_behavior_gap.scripts.analyze_highlow_intervention_final",
    "utility_behavior_gap.scripts.analyze_utility_gap_dose_response",
    "utility_behavior_gap.scripts.audit_max_effort_runs",
    "utility_behavior_gap.scripts.build_essay_direct_trials_from_utilities",
    "utility_behavior_gap.scripts.check_openrouter_logprobs",
    "utility_behavior_gap.scripts.export_emergent_values_options",
    "utility_behavior_gap.scripts.import_emergent_values_utilities",
    "utility_behavior_gap.scripts.prepare_generation_jobs",
    "utility_behavior_gap.scripts.prepare_highlow_vs_direct_judging",
    "utility_behavior_gap.scripts.plot_highlow_main",
    "utility_behavior_gap.scripts.plot_highlow_within_count",
    "utility_behavior_gap.scripts.plot_incentive_amount_main",
    "utility_behavior_gap.scripts.plot_moral_nolabel_main",
    "utility_behavior_gap.scripts.plot_sys_prompt_main",
    "utility_behavior_gap.scripts.plot_utility_top_bottom",
    "utility_behavior_gap.scripts.reproduce_all",
    "utility_behavior_gap.scripts.run_direct_instruction_batch",
    "utility_behavior_gap.scripts.run_generation",
    "utility_behavior_gap.scripts.run_highlow_intervention_batch",
    "utility_behavior_gap.scripts.run_judging",
    "utility_behavior_gap.scripts.run_max_effort_batch",
    "utility_behavior_gap.scripts.run_mimo_utility_refit",
    "utility_behavior_gap.scripts.select_pairs",
    "utility_behavior_gap.scripts.status",
    "utility_behavior_gap.scripts.summarize_paper_tables",
]


def test_reproduction_entrypoints_live_under_source_package():
    for module_name in ENTRYPOINT_MODULES:
        importlib.import_module(module_name)

    repo_root = Path(__file__).resolve().parents[1]
    assert not (repo_root / "scripts").exists()
