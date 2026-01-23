"""Tests for train_boilerplate module."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import mlops_g116.train_boilerplate as train_boilerplate


class DummyModel(nn.Module):
    """Minimal model used for boilerplate training tests."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(1, num_classes)
        self._num_classes = num_classes

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], self._num_classes)


class DummyTrainer:
    """No-op trainer capturing initialization and fit calls."""

    last_instance: "DummyTrainer | None" = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        DummyTrainer.last_instance = self
        self.args = args
        self.kwargs = kwargs
        self.fit_called = False

    def fit(self, *_args: object, **_kwargs: object) -> None:
        self.fit_called = True


class FailingWandbLogger:
    """W&B logger stub that fails to initialize."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("W&B disabled for tests")


def test_evaluate_model_returns_perfect_metrics() -> None:
    """Ensure _evaluate_model returns perfect metrics for perfect predictions."""
    model = DummyModel(num_classes=2)
    images = torch.zeros(4, 1, 8, 8)
    labels = torch.zeros(4, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(images, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    metrics = train_boilerplate._evaluate_model(model, loader)

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_train_rewrites_target_and_saves_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure train rewrites model target, sets lr, and saves outputs."""
    output_dir = tmp_path / "outputs" / "run1"
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = SimpleNamespace(output_dir=str(output_dir))
    monkeypatch.setattr(train_boilerplate.HydraConfig, "get", lambda: SimpleNamespace(runtime=runtime))
    monkeypatch.setattr(train_boilerplate, "REPO_ROOT", tmp_path)

    images = torch.zeros(4, 1, 8, 8)
    labels = torch.zeros(4, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(images, labels)
    monkeypatch.setattr(train_boilerplate, "load_data", lambda: (dataset, dataset))
    monkeypatch.setattr(train_boilerplate.pl, "Trainer", DummyTrainer)
    monkeypatch.setattr(train_boilerplate, "WandbLogger", FailingWandbLogger)

    captured: dict[str, object] = {}

    def _fake_instantiate(config: dict[str, object]) -> DummyModel:
        """Capture instantiate config and return a dummy model."""
        captured["config"] = config
        return DummyModel(num_classes=2)

    monkeypatch.setattr(train_boilerplate, "instantiate", _fake_instantiate)

    config = OmegaConf.create(
        {
            "hyperparameters": {"lr": 0.01, "batch_size": 2, "epochs": 1, "seed": 123},
            "model": {"_target_": "mlops_g116.model.TumorDetectionModelSimple"},
        }
    )

    train_boilerplate.train.__wrapped__(config)

    model_config = captured["config"]
    assert isinstance(model_config, dict), "Expected instantiate to receive a dict config"
    assert model_config["_target_"] == "mlops_g116.model_boilerplate.TumorDetectionModelSimple"
    assert model_config["lr"] == pytest.approx(0.01)

    assert DummyTrainer.last_instance is not None, "Expected DummyTrainer to be constructed"
    assert DummyTrainer.last_instance.fit_called, "Expected trainer.fit to be called"

    assert (output_dir / "models" / "model.pth").exists(), "Expected model artifact in output directory"
    assert (tmp_path / "models" / "model.pth").exists(), "Expected model artifact in repo models directory"
