"""Dataset preprocessing utilities for local image folders."""

import os
from pathlib import Path

import torch
import typer
from PIL import Image
from torchvision import transforms

IMG_SIZE = 224

transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
)


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize image tensors to zero mean and unit variance.

    Args:
        images: Image batch tensor.

    Returns:
        Normalized image tensor.
    """
    return (images - images.mean()) / images.std()


def process_folder(folder_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load images from a folder with class subfolders.

    Args:
        folder_path: Root directory that contains class subfolders.

    Returns:
        Tuple with stacked image tensors and integer labels.
    """
    images = []
    labels = []

    class_folders = sorted(os.listdir(folder_path))

    for label, class_folder in enumerate(class_folders):
        class_path = os.path.join(folder_path, class_folder)
        for file in os.listdir(class_path):
            if file.lower().endswith(("jpg", "jpeg", "png")):
                img_path = os.path.join(class_path, file)
                img = Image.open(img_path).convert("L")  # grayscale
                img = transform(img)
                images.append(img)
                labels.append(label)

    images = torch.stack(images)
    labels = torch.tensor(labels)
    images = normalize(images)

    return images, labels


def preprocess(raw_dir: str = "data/raw/brain_dataset", processed_dir: str = "data/processed") -> None:
    """Preprocess local raw images and save tensors to disk.

    Args:
        raw_dir: Directory with raw data organized by Training/Testing subfolders.
        processed_dir: Output directory for serialized tensors.
    """
    os.makedirs(processed_dir, exist_ok=True)

    train_dir = os.path.join(raw_dir, "Training")
    test_dir = os.path.join(raw_dir, "Testing")

    print("Processing training data...")
    x_train, y_train = process_folder(train_dir)
    print(f"Training data: {x_train.shape[0]} images")

    print("Processing test data...")
    x_test, y_test = process_folder(test_dir)
    print(f"Test data: {x_test.shape[0]} images")

    # Save tensors
    torch.save(x_train, os.path.join(processed_dir, "train_images.pt"))
    torch.save(y_train, os.path.join(processed_dir, "train_target.pt"))
    torch.save(x_test, os.path.join(processed_dir, "test_images.pt"))
    torch.save(y_test, os.path.join(processed_dir, "test_target.pt"))

    print(f"Data processed and saved in {processed_dir}")


def _resolve_processed_dir(processed_dir: str | Path | None = None) -> Path:
    """Resolve the processed data directory.

    Args:
        processed_dir: Optional override path to the processed dataset directory.

    Returns:
        Path to the processed dataset directory.
    """
    if processed_dir is not None:
        return Path(processed_dir)
    return Path(os.getenv("DATA_ROOT", "data/processed"))


def load_data(
    processed_dir: str | Path | None = None,
) -> tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
    """Load processed dataset and return PyTorch TensorDataset.

    Args:
        processed_dir: Optional override path to the processed dataset directory.

    Returns:
        Tuple of (train_set, test_set).
    """
    data_root = _resolve_processed_dir(processed_dir)
    train_images = torch.load(data_root / "train_images.pt")
    train_target = torch.load(data_root / "train_target.pt")
    test_images = torch.load(data_root / "test_images.pt")
    test_target = torch.load(data_root / "test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set


def main() -> None:
    """Run preprocessing from the command line."""
    typer.run(preprocess)


if __name__ == "__main__":
    typer.run(preprocess)
