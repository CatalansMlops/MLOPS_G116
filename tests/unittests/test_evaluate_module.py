"""Execution tests for evaluate module."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import mlops_g116.evaluate as evaluate_module
from mlops_g116.model import TumorDetectionModelSimple


class DummyProfiler:
    """No-op profiler context manager."""

    def __enter__(self) -> "DummyProfiler":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def step(self) -> None:
        """Advance profiler step."""


class DummyRun:
    """Minimal W&B run stub."""

    id = "dummy-run"

    def log(self, *_args: object, **_kwargs: object) -> None:
        """No-op log."""

    def log_artifact(self, *_args: object, **_kwargs: object) -> None:
        """No-op log_artifact."""

    def link_artifact(self, *_args: object, **_kwargs: object) -> None:
        """No-op link_artifact."""


class DummyArtifact:
    """Minimal W&B artifact stub."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def add_file(self, *_args: object, **_kwargs: object) -> None:
        """No-op add_file."""


def _patch_evaluate_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Patch evaluate runtime dependencies to use temp outputs."""
    output_dir = tmp_path / "outputs"
    runtime = SimpleNamespace(output_dir=str(output_dir))
    monkeypatch.setattr(evaluate_module.HydraConfig, "get", lambda: SimpleNamespace(runtime=runtime))
    monkeypatch.setattr(evaluate_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        evaluate_module.torch.profiler,
        "profile",
        lambda *args, **kwargs: DummyProfiler(),
    )
    monkeypatch.setattr(
        evaluate_module.torch.profiler,
        "tensorboard_trace_handler",
        lambda *args, **kwargs: lambda *_a, **_k: None,
    )
    monkeypatch.setattr(evaluate_module.wandb, "init", lambda **_kwargs: DummyRun())
    monkeypatch.setattr(evaluate_module.wandb, "log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluate_module.wandb, "finish", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluate_module.wandb, "Image", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(evaluate_module.wandb, "Artifact", DummyArtifact)
    monkeypatch.setenv("RUN_SNAKEVIZ", "0")
    monkeypatch.setenv("RUN_TENSORBOARD", "0")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    return output_dir


def _make_dataset(num_samples: int) -> torch.utils.data.TensorDataset:
    """Create a small tensor dataset for evaluation."""
    images = torch.randn(num_samples, 1, 64, 64)
    labels = torch.tensor(list(range(num_samples)))
    return torch.utils.data.TensorDataset(images, labels)


def test_evaluate_raises_when_checkpoint_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure evaluate raises when the checkpoint does not exist."""
    _patch_evaluate_runtime(monkeypatch, tmp_path)
    config = OmegaConf.create(
        {
            "model": {"_target_": "mlops_g116.model.TumorDetectionModelSimple"},
            "evaluation": {"batch_size": 2, "checkpoint_path": str(tmp_path / "missing.pth")},
        }
    )
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        evaluate_module.evaluate.__wrapped__(config)


def test_evaluate_writes_metrics_and_figures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure evaluate writes metrics and figure outputs."""
    output_dir = _patch_evaluate_runtime(monkeypatch, tmp_path)
    model = TumorDetectionModelSimple()
    checkpoint_path = tmp_path / "model.pth"
    torch.save(model.state_dict(), checkpoint_path)
    train_set = _make_dataset(2)
    test_set = _make_dataset(4)
    monkeypatch.setattr(evaluate_module, "load_data", lambda: (train_set, test_set))
    config = OmegaConf.create(
        {
            "model": {"_target_": "mlops_g116.model.TumorDetectionModelSimple"},
            "evaluation": {"batch_size": 2, "checkpoint_path": str(checkpoint_path)},
        }
    )
    evaluate_module.evaluate.__wrapped__(config)

    metrics_path = output_dir / "reports" / "evaluation" / "metrics.json"
    assert metrics_path.exists(), f"Expected metrics file at {metrics_path}"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected_keys = {"test/accuracy", "test/precision", "test/recall", "test/f1"}
    assert expected_keys.issubset(metrics.keys()), f"Missing metric keys: {expected_keys - set(metrics.keys())}"

    cm_path = output_dir / "reports" / "figures" / "evaluation_confusion_matrix.png"
    assert cm_path.exists(), f"Expected confusion matrix at {cm_path}"
    repo_cm_path = tmp_path / "reports" / "figures" / "evaluation_confusion_matrix.png"
    assert repo_cm_path.exists(), f"Expected repo confusion matrix at {repo_cm_path}"
