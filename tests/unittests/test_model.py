"""Model tests for mlops_g116."""

from __future__ import annotations

import torch
import pytest

from mlops_g116.model import DenseNet121, ResNet18, TumorDetectionModelSimple

_BATCH_SIZES = [1, 2, 8]


@pytest.mark.parametrize("batch_size", _BATCH_SIZES)
def test_simple_model_output_shape(batch_size: int) -> None:
    """Ensure the simple model returns logits for 4 classes across batch sizes."""
    model = TumorDetectionModelSimple()
    x = torch.randn(batch_size, 1, 64, 64)
    y = model(x)
    assert y.shape == (batch_size, 4), f"Expected output shape ({batch_size}, 4), got {tuple(y.shape)}"


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
