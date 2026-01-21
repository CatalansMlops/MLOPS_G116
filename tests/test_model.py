"""Model tests for mlops_g116."""

from __future__ import annotations

import torch
import pytest

from mlops_g116.model import DenseNet121, ResNet18, TumorDetectionModelSimple


def test_simple_model_output_shape() -> None:
    """Ensure the simple model returns logits for 4 classes."""
    model = TumorDetectionModelSimple()
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    assert y.shape == (2, 4), f"Expected output shape (2, 4), got {tuple(y.shape)}"


def test_resnet18_output_shape() -> None:
    """Ensure ResNet18 returns logits for 4 classes with grayscale input."""
    model = ResNet18(num_classes=4)
    x = torch.randn(1, 1, 224, 224)
    y = model(x)
    assert y.shape == (1, 4), f"Expected output shape (1, 4), got {tuple(y.shape)}"


def test_densenet121_output_shape() -> None:
    """Ensure DenseNet121 returns logits for 4 classes with grayscale input."""
    model = DenseNet121(num_classes=4)
    x = torch.randn(1, 1, 224, 224)
    y = model(x)
    assert y.shape == (1, 4), f"Expected output shape (1, 4), got {tuple(y.shape)}"


def test_simple_model_raises_on_wrong_input_shape() -> None:
    """Ensure the simple model rejects invalid input shapes."""
    model = TumorDetectionModelSimple()
    with pytest.raises(ValueError, match="Expected input to be a 4D tensor"):
        model(torch.randn(1, 64, 64))
    with pytest.raises(ValueError, match="Expected input to have 1 channel"):
        model(torch.randn(1, 3, 64, 64))


if __name__ == "__main__":
    test_simple_model_output_shape()
    test_resnet18_output_shape()
    test_densenet121_output_shape()
    test_simple_model_raises_on_wrong_input_shape()
    print("test_model passed all tests")
