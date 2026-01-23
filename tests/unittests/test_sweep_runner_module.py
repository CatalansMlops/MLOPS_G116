"""Tests for sweep_runner module."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mlops_g116.sweep_runner as sweep_runner


def test_main_builds_sweep_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure main builds the expected subprocess command."""
    config = SimpleNamespace(lr=0.01, batch_size=8, epochs=2, model="resnet18", optimizer="adam")
    monkeypatch.setattr(sweep_runner.wandb, "init", lambda: SimpleNamespace(config=config))

    captured: dict[str, list[str]] = {}

    def _fake_check_call(cmd: list[str]) -> None:
        captured["cmd"] = cmd

    monkeypatch.setattr(sweep_runner.subprocess, "check_call", _fake_check_call)

    sweep_runner.main()

    cmd = captured["cmd"]
    assert cmd[0] == sweep_runner.sys.executable
    assert cmd[1:3] == ["-m", "mlops_g116.main"]
    assert "hyperparameters.lr=0.01" in cmd
    assert "hyperparameters.batch_size=8" in cmd
    assert "hyperparameters.epochs=2" in cmd
    assert "model=resnet18" in cmd
    assert "optimizer=adam" in cmd
