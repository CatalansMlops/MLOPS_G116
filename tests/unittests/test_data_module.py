"""Tests for data processing utilities."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
import pytest

from mlops_g116.data import IMG_SIZE, load_data, normalize, process_folder


def _write_grayscale_image(path: Path, value: int) -> None:
    """Create and save a small grayscale PNG image."""
    image = Image.new("L", (8, 8), color=value)
    image.save(path)


def test_normalize_zero_mean_unit_std() -> None:
    """Ensure normalize outputs near-zero mean and unit std."""
    x = torch.randn(4, 1, 8, 8)
    y = normalize(x)
    mean = y.mean().item()
    std = y.std().item()
    assert abs(mean) < 1e-6, f"Expected near-zero mean, got {mean}"
    assert abs(std - 1.0) < 1e-6, f"Expected unit std, got {std}"


def test_process_folder_outputs(tmp_path: Path) -> None:
    """Ensure process_folder returns tensors with expected shapes and labels."""
    class_a = tmp_path / "class_a"
    class_b = tmp_path / "class_b"
    class_a.mkdir()
    class_b.mkdir()
    _write_grayscale_image(class_a / "img0.png", 0)
    _write_grayscale_image(class_b / "img1.png", 255)

    images, labels = process_folder(str(tmp_path))
    assert images.shape == (2, 1, IMG_SIZE, IMG_SIZE), f"Unexpected images shape: {tuple(images.shape)}"
    assert labels.shape == (2,), f"Unexpected labels shape: {tuple(labels.shape)}"
    assert set(labels.tolist()) == {0, 1}, f"Unexpected label values: {labels.tolist()}"
    assert torch.isfinite(images).all(), "Normalized images contain non-finite values"


def test_load_data_returns_datasets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure load_data returns datasets from processed tensors."""
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    train_images = torch.randn(3, 1, IMG_SIZE, IMG_SIZE)
    train_target = torch.tensor([0, 1, 2])
    test_images = torch.randn(2, 1, IMG_SIZE, IMG_SIZE)
    test_target = torch.tensor([1, 3])

    torch.save(train_images, processed_dir / "train_images.pt")
    torch.save(train_target, processed_dir / "train_target.pt")
    torch.save(test_images, processed_dir / "test_images.pt")
    torch.save(test_target, processed_dir / "test_target.pt")

    monkeypatch.chdir(tmp_path)
    train_set, test_set = load_data()

    assert len(train_set) == 3, f"Expected 3 training samples, got {len(train_set)}"
    assert len(test_set) == 2, f"Expected 2 test samples, got {len(test_set)}"
    sample_x, sample_y = train_set[0]
    assert sample_x.shape == (1, IMG_SIZE, IMG_SIZE), f"Unexpected sample shape: {tuple(sample_x.shape)}"
    assert sample_y.dim() == 0, f"Expected scalar label tensor, got shape {tuple(sample_y.shape)}"
