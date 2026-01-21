"""Model tests for mlops_g116."""

from __future__ import annotations

import torch
import torchvision.models as models

from mlops_g116.model import DenseNet121, ResNet18, TumorDetectionModelSimple

_ORIG_RESNET18 = models.resnet18
_ORIG_DENSENET121 = models.densenet121


def _resnet18_no_weights(*args: object, **kwargs: object) -> torch.nn.Module:
    """Return ResNet18 without downloading pretrained weights."""
    kwargs.pop("weights", None)
    return _ORIG_RESNET18(weights=None, **kwargs)


def _densenet121_no_weights(*args: object, **kwargs: object) -> torch.nn.Module:
    """Return DenseNet121 without downloading pretrained weights."""
    kwargs.pop("weights", None)
    return _ORIG_DENSENET121(weights=None, **kwargs)


def test_simple_model_output_shape() -> None:
    """Ensure the simple model returns logits for 4 classes."""
    model = TumorDetectionModelSimple()
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    assert y.shape == (2, 4), f"Expected output shape (2, 4), got {tuple(y.shape)}"


def test_resnet18_grayscale_output_shape(monkeypatch: object) -> None:
    """Ensure ResNet18 adapts to 1-channel input and 4 output classes."""
    monkeypatch.setattr(models, "resnet18", _resnet18_no_weights)
    model = ResNet18(num_classes=4)
    x = torch.randn(1, 1, 224, 224)
    y = model(x)
    assert y.shape == (1, 4), f"Expected output shape (1, 4), got {tuple(y.shape)}"
    assert model.backbone.conv1.in_channels == 1, f"Expected 1 input channel, got {model.backbone.conv1.in_channels}"
    assert model.backbone.fc.out_features == 4, f"Expected 4 outputs, got {model.backbone.fc.out_features}"


def test_densenet121_grayscale_output_shape(monkeypatch: object) -> None:
    """Ensure DenseNet121 adapts to 1-channel input and 4 output classes."""
    monkeypatch.setattr(models, "densenet121", _densenet121_no_weights)
    model = DenseNet121(num_classes=4)
    x = torch.randn(1, 1, 224, 224)
    y = model(x)
    assert y.shape == (1, 4), f"Expected output shape (1, 4), got {tuple(y.shape)}"
    assert model.backbone.features.conv0.in_channels == 1, (
        f"Expected 1 input channel, got {model.backbone.features.conv0.in_channels}"
    )
    assert model.backbone.classifier.out_features == 4, (
        f"Expected 4 outputs, got {model.backbone.classifier.out_features}"
    )
