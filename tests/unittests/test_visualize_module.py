"""Execution tests for visualize module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import mlops_g116.visualize as visualize_module
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


class DummyTSNE:
    """Fast TSNE stand-in."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Return a simple 2D projection."""
        if data.shape[1] >= 2:
            return data[:, :2]
        return np.concatenate([data, np.zeros((data.shape[0], 2 - data.shape[1]))], axis=1)


def _make_dataset(num_samples: int) -> torch.utils.data.TensorDataset:
    """Create a small tensor dataset for visualization."""
    images = torch.randn(num_samples, 1, 64, 64)
    labels = torch.tensor([i % 4 for i in range(num_samples)])
    return torch.utils.data.TensorDataset(images, labels)


def test_build_model_rejects_unknown_name() -> None:
    """Ensure _build_model rejects unsupported names."""
    with pytest.raises(ValueError, match="Unsupported model name"):
        visualize_module._build_model("unknown")


def test_visualize_writes_embedding_figure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure visualize writes the embedding figure."""
    monkeypatch.chdir(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model = TumorDetectionModelSimple()
    checkpoint_path = models_dir / "model.pth"
    torch.save(model.state_dict(), checkpoint_path)
    train_set = _make_dataset(4)
    test_set = _make_dataset(40)
    monkeypatch.setattr(visualize_module, "load_data", lambda: (train_set, test_set))
    monkeypatch.setattr(visualize_module, "TSNE", DummyTSNE)
    monkeypatch.setattr(
        visualize_module.torch.profiler,
        "profile",
        lambda *args, **kwargs: DummyProfiler(),
    )
    monkeypatch.setattr(
        visualize_module.torch.profiler,
        "tensorboard_trace_handler",
        lambda *args, **kwargs: lambda *_a, **_k: None,
    )
    monkeypatch.setattr(visualize_module.wandb, "init", lambda **_kwargs: DummyRun())
    monkeypatch.setattr(visualize_module.wandb, "log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(visualize_module.wandb, "finish", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(visualize_module.wandb, "Image", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(visualize_module.wandb, "Artifact", DummyArtifact)
    monkeypatch.setenv("RUN_SNAKEVIZ", "0")
    monkeypatch.setenv("RUN_TENSORBOARD", "0")
    monkeypatch.setenv("MPLBACKEND", "Agg")

    visualize_module.visualize(
        model_checkpoint=str(checkpoint_path),
        model_name="simple",
        figure_name="embeddings.png",
        batch_size=16,
    )

    figure_path = tmp_path / "reports" / "figures" / "embeddings.png"
    assert figure_path.exists(), f"Expected visualization at {figure_path}"
