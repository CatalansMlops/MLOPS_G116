"""Dataset preprocessing utilities for Google Cloud Storage buckets."""

import os
from io import BytesIO

import torch
import typer
from google.cloud import storage
from google.cloud.storage.bucket import Bucket
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


def process_folder_from_bucket(bucket: Bucket, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load images from a GCS prefix with class subfolders.

    Args:
        bucket: GCS bucket containing the dataset.
        prefix: Prefix that points to the dataset split, such as Training/Testing.

    Returns:
        Tuple with stacked image tensors and integer labels.
    """
    images = []
    labels = []

    blobs = list(bucket.list_blobs(prefix=prefix))

    class_names = sorted(
        {blob.name.split("/")[3] for blob in blobs if blob.name.lower().endswith(("jpg", "jpeg", "png"))}
    )

    class_to_label = {name: i for i, name in enumerate(class_names)}

    for blob in blobs:
        if blob.name.lower().endswith(("jpg", "jpeg", "png")):
            class_name = blob.name.split("/")[3]
            label = class_to_label[class_name]

            data = blob.download_as_bytes()
            img = Image.open(BytesIO(data)).convert("L")
            img = transform(img)

            images.append(img)
            labels.append(label)

    images = torch.stack(images)
    labels = torch.tensor(labels)
    images = normalize(images)

    return images, labels


def preprocess(
    bucket_name: str = "mlops116",
    raw_prefix: str = "data/raw/brain_dataset",
    processed_dir: str = "data/processed",
) -> None:
    """Preprocess data from a GCS bucket and save tensors locally.

    This workflow expects Google Cloud credentials (for example via the
    GOOGLE_APPLICATION_CREDENTIALS environment variable). If you prefer a
    local workflow, download the dataset with tools like gsutil or DVC and
    use the local preprocessing pipeline instead.

    Args:
        bucket_name: GCS bucket name containing the dataset.
        raw_prefix: Prefix inside the bucket with Training/Testing subfolders.
        processed_dir: Local directory to store processed tensors.
    """

    os.makedirs(processed_dir, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    train_prefix = f"{raw_prefix}/Training"
    test_prefix = f"{raw_prefix}/Testing"

    print("Processing training data from bucket...")
    x_train, y_train = process_folder_from_bucket(bucket, train_prefix)
    print(f"Training data: {x_train.shape[0]} images")

    print("Processing test data from bucket...")
    x_test, y_test = process_folder_from_bucket(bucket, test_prefix)
    print(f"Test data: {x_test.shape[0]} images")

    torch.save(x_train, os.path.join(processed_dir, "train_images.pt"))
    torch.save(y_train, os.path.join(processed_dir, "train_target.pt"))
    torch.save(x_test, os.path.join(processed_dir, "test_images.pt"))
    torch.save(y_test, os.path.join(processed_dir, "test_target.pt"))

    print(f"Data processed and saved in {processed_dir}")


def load_data() -> tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
    """Load processed dataset and return PyTorch TensorDatasets.

    Returns:
        Tuple with train and test TensorDataset instances.
    """
    train_images = torch.load("data/processed/train_images.pt")
    train_target = torch.load("data/processed/train_target.pt")
    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess)
