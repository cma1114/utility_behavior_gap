import yaml

from utility_behavior_gap.scripts.run_mimo_utility_refit import (
    MODE_SETTINGS,
    call_estimate,
    should_echo_output_line,
    write_agent_config,
)


def test_smoke_utility_refit_call_estimate():
    estimate = call_estimate(12, MODE_SETTINGS["smoke"])

    assert estimate["sampled_fit_edges"] == 52
    assert estimate["fit_calls"] == 104
    assert estimate["holdout_calls"] == 0
    assert estimate["estimated_calls"] == 104


def test_full_utility_refit_call_estimate_includes_holdout():
    estimate = call_estimate(610, MODE_SETTINGS["full"])

    assert estimate["sampled_fit_edges"] == 11610
    assert estimate["fit_calls"] == 232200
    assert estimate["holdout_calls"] == 20000
    assert estimate["estimated_calls"] == 252200


def test_write_agent_config_uses_conservative_openrouter_settings(tmp_path):
    path = write_agent_config(tmp_path, concurrency_limit=5, base_timeout=30, max_tokens=64)

    config = yaml.safe_load(path.read_text())
    assert config["default"]["max_tokens"] == 64
    assert config["default"]["concurrency_limit"] == 5
    assert config["default"]["base_timeout"] == 30
    assert config["default"]["completion_kwargs"] == {"reasoning_effort": "none"}


def test_quiet_filter_keeps_signal_and_suppresses_training_noise():
    assert should_echo_output_line("Wrote checkpoint: /tmp/checkpoint.json")
    assert should_echo_output_line("Iteration 3/22")
    assert should_echo_output_line("RuntimeError: failed")
    assert not should_echo_output_line("Epoch 0, Loss: 0.6935")
    assert not should_echo_output_line("LLM calls:  12%|###")
