import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
import typer

IMG_SIZE = 224  # Image size (IMG_SIZE x IMG_SIZE)

# Transformations
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),   # Converts [C,H,W] and scales to [0,1]
])

def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalizes images: mean 0, std 1"""
    return (images - images.mean()) / images.std()

def process_folder(folder_path: str):
    """Load all images in a folder (with subfolders for classes)"""
    images = []
    labels = []

    class_folders = sorted(os.listdir(folder_path))  # tumor_type_1, tumor_type_2, etc.

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
    os.makedirs(processed_dir, exist_ok=True)

    # Paths
    train_dir = os.path.join(raw_dir, "Training")
    test_dir = os.path.join(raw_dir, "Testing")

    print("Processing training data...")
    X_train, y_train = process_folder(train_dir)
    print(f"Training data: {X_train.shape[0]} images")

    print("Processing test data...")
    X_test, y_test = process_folder(test_dir)
    print(f"Test data: {X_test.shape[0]} images")

    # Save tensors
    torch.save(X_train, os.path.join(processed_dir, "train_images.pt"))
    torch.save(y_train, os.path.join(processed_dir, "train_target.pt"))
    torch.save(X_test, os.path.join(processed_dir, "test_images.pt"))
    torch.save(y_test, os.path.join(processed_dir, "test_target.pt"))

    print("Data processed and saved in", processed_dir)

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


def load_data(processed_dir: str | Path | None = None):
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
    # Expose the preprocess function as a CLI using Typer
    typer.run(preprocess)

if __name__ == "__main__":
    typer.run(preprocess)
