import subprocess

from utility_behavior_gap.scripts import run_max_effort_batch


def test_run_actor_step_returns_true_on_success(monkeypatch):
    calls = []

    def fake_run(command):
        calls.append(command)

    monkeypatch.setattr(run_max_effort_batch, "run", fake_run)

    assert run_max_effort_batch.run_actor_step("actor", "step", ["python", "-m", "ok"])
    assert calls == [["python", "-m", "ok"]]


def test_run_actor_step_returns_false_on_subprocess_failure(monkeypatch, capsys):
    def fake_run(command):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(run_max_effort_batch, "run", fake_run)

    assert not run_max_effort_batch.run_actor_step("mimo-v2-pro-or", "run_generation", ["python", "-m", "bad"])
    captured = capsys.readouterr()
    assert "run_generation failed for mimo-v2-pro-or" in captured.out
    assert "skipping this actor and continuing" in captured.out
