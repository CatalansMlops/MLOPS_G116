"""Tests for data_importfromcloud module."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import mlops_g116.data_importfromcloud as data_importfromcloud


class DummyBlob:
    """Minimal blob stub for bucket tests."""

    def __init__(self, name: str, image_bytes: bytes) -> None:
        self.name = name
        self._image_bytes = image_bytes

    def download_as_bytes(self) -> bytes:
        """Return stored image bytes."""
        return self._image_bytes


class DummyBucket:
    """Bucket stub that returns a fixed list of blobs."""

    def __init__(self, blobs: list[DummyBlob]) -> None:
        self._blobs = blobs

    def list_blobs(self, prefix: str) -> list[DummyBlob]:
        """Return blobs regardless of prefix."""
        del prefix
        return self._blobs


def _make_image_bytes(value: int) -> bytes:
    """Create an in-memory grayscale PNG image."""
    image = Image.new("L", (8, 8), color=value)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_process_folder_from_bucket_outputs_tensors() -> None:
    """Ensure process_folder_from_bucket returns normalized tensors and labels."""
    image_a = _make_image_bytes(0)
    image_b = _make_image_bytes(255)
    blobs = [
        DummyBlob("data/raw/brain_dataset/class_a/img0.png", image_a),
        DummyBlob("data/raw/brain_dataset/class_b/img1.png", image_b),
    ]
    bucket = DummyBucket(blobs)

    images, labels = data_importfromcloud.process_folder_from_bucket(bucket, "data/raw/brain_dataset")

    assert images.shape[0] == 2, f"Unexpected image batch size: {images.shape[0]}"
    assert labels.shape == (2,), f"Unexpected labels shape: {tuple(labels.shape)}"
    assert set(labels.tolist()) == {0, 1}, f"Unexpected labels: {labels.tolist()}"
    assert torch.isfinite(images).all(), "Expected normalized images to be finite"


def test_preprocess_writes_processed_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure preprocess saves tensors to the processed directory."""
    processed_dir = tmp_path / "processed"
    train_images = torch.randn(3, 1, data_importfromcloud.IMG_SIZE, data_importfromcloud.IMG_SIZE)
    train_labels = torch.tensor([0, 1, 2])
    test_images = torch.randn(2, 1, data_importfromcloud.IMG_SIZE, data_importfromcloud.IMG_SIZE)
    test_labels = torch.tensor([1, 1])

    calls: list[str] = []

    def _fake_process_folder(bucket: object, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return deterministic tensors for each split."""
        del bucket
        calls.append(prefix)
        if prefix.endswith("Training"):
            return train_images, train_labels
        return test_images, test_labels

    monkeypatch.setattr(data_importfromcloud, "process_folder_from_bucket", _fake_process_folder)

    class DummyClient:
        def bucket(self, name: str) -> object:
            del name
            return SimpleNamespace()

    monkeypatch.setattr(data_importfromcloud.storage, "Client", lambda: DummyClient())

    data_importfromcloud.preprocess(
        bucket_name="test-bucket",
        raw_prefix="data/raw/brain_dataset",
        processed_dir=str(processed_dir),
    )

    assert (processed_dir / "train_images.pt").exists(), "Expected train_images.pt to be saved"
    assert (processed_dir / "train_target.pt").exists(), "Expected train_target.pt to be saved"
    assert (processed_dir / "test_images.pt").exists(), "Expected test_images.pt to be saved"
    assert (processed_dir / "test_target.pt").exists(), "Expected test_target.pt to be saved"
    assert calls == ["data/raw/brain_dataset/Training", "data/raw/brain_dataset/Testing"]


def test_load_data_returns_datasets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure load_data returns datasets from processed tensors."""
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    train_images = torch.randn(2, 1, data_importfromcloud.IMG_SIZE, data_importfromcloud.IMG_SIZE)
    train_labels = torch.tensor([0, 1])
    test_images = torch.randn(1, 1, data_importfromcloud.IMG_SIZE, data_importfromcloud.IMG_SIZE)
    test_labels = torch.tensor([1])

    torch.save(train_images, processed_dir / "train_images.pt")
    torch.save(train_labels, processed_dir / "train_target.pt")
    torch.save(test_images, processed_dir / "test_images.pt")
    torch.save(test_labels, processed_dir / "test_target.pt")

    monkeypatch.chdir(tmp_path)
    train_set, test_set = data_importfromcloud.load_data()

    assert len(train_set) == 2, f"Unexpected train set length: {len(train_set)}"
    assert len(test_set) == 1, f"Unexpected test set length: {len(test_set)}"
