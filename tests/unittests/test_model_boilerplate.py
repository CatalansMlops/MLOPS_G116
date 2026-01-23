"""Tests for boilerplate model module."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import mlops_g116.model_boilerplate as model_boilerplate


class DummyResNet(nn.Module):
    """Minimal ResNet-style backbone for tests."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc = nn.Linear(64, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = x.mean(dim=(2, 3))
        return self.fc(x)


class DummyDenseFeatures(nn.Module):
    """Minimal DenseNet feature extractor for tests."""

    def __init__(self) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv0(x)


class DummyDenseNet(nn.Module):
    """Minimal DenseNet-style backbone for tests."""

    def __init__(self) -> None:
        super().__init__()
        self.features = DummyDenseFeatures()
        self.classifier = nn.Linear(64, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean(dim=(2, 3))
        return self.classifier(x)


class DummyLightning(model_boilerplate.BaseLightningClassifier):
    """BaseLightningClassifier subclass with deterministic outputs."""

    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__(num_classes=2, lr=lr)
        self.bias = nn.Parameter(torch.zeros(2))
        self.logged: list[str] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(x.shape[0], 2) + self.bias
        logits[:, 0] = logits[:, 0] + 10.0
        return logits

    def log(self, name: str, *_args: object, **_kwargs: object) -> None:
        self.logged.append(name)


def test_base_classifier_logs_train_and_val() -> None:
    """Ensure shared step logs metrics for train and validation stages."""
    model = DummyLightning()
    batch = (torch.randn(4, 1, 8, 8), torch.zeros(4, dtype=torch.long))

    loss = model.training_step(batch, 0)
    assert loss.ndim == 0, "Expected scalar loss from training_step"
    assert "train_loss" in model.logged, "Expected train_loss log"
    assert "train_acc" in model.logged, "Expected train_acc log"

    model.logged.clear()
    model.validation_step(batch, 0)
    assert "val_loss" in model.logged, "Expected val_loss log"
    assert "val_acc" in model.logged, "Expected val_acc log"


def test_configure_optimizers_uses_lr() -> None:
    """Ensure configure_optimizers uses lr from hyperparameters."""
    model = DummyLightning(lr=0.005)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Expected Adam optimizer"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.005)


def test_resnet18_adapts_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ResNet18 adapts conv1 to 1 channel and updates fc output."""
    monkeypatch.setattr(model_boilerplate.models, "resnet18", lambda **_kwargs: DummyResNet())
    model = model_boilerplate.ResNet18(num_classes=3, lr=1e-3)
    assert model.backbone.conv1.in_channels == 1, "Expected conv1 to accept 1 channel"
    assert model.backbone.fc.out_features == 3, "Expected fc to output 3 classes"
    output = model(torch.randn(2, 1, 32, 32))
    assert output.shape == (2, 3), f"Unexpected output shape: {tuple(output.shape)}"


def test_densenet121_adapts_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure DenseNet121 adapts conv0 to 1 channel and updates classifier output."""
    monkeypatch.setattr(model_boilerplate.models, "densenet121", lambda **_kwargs: DummyDenseNet())
    model = model_boilerplate.DenseNet121(num_classes=5, lr=1e-3)
    assert model.backbone.features.conv0.in_channels == 1, "Expected conv0 to accept 1 channel"
    assert model.backbone.classifier.out_features == 5, "Expected classifier to output 5 classes"
    output = model(torch.randn(3, 1, 32, 32))
    assert output.shape == (3, 5), f"Unexpected output shape: {tuple(output.shape)}"


def test_simple_model_output_shape() -> None:
    """Ensure the simple boilerplate model returns logits for 4 classes."""
    model = model_boilerplate.TumorDetectionModelSimple(num_classes=4, lr=1e-3)
    output = model(torch.randn(2, 1, 64, 64))
    assert output.shape == (2, 4), f"Unexpected output shape: {tuple(output.shape)}"


def test_simple_model_validates_input() -> None:
    """Ensure the simple boilerplate model validates input shape and channels."""
    model = model_boilerplate.TumorDetectionModelSimple()
    with pytest.raises(ValueError, match="Expected input to be a 4D tensor"):
        model(torch.randn(1, 64, 64))
    with pytest.raises(ValueError, match="Expected input to have 1 channel"):
        model(torch.randn(1, 3, 64, 64))
