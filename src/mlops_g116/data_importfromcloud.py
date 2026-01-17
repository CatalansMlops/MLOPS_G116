import os
from PIL import Image
import torch
from torchvision import transforms
import typer
from google.cloud import storage
from io import BytesIO

IMG_SIZE = 224  # Image size (IMG_SIZE x IMG_SIZE)

# Transformations
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalizes images: mean 0, std 1"""
    return (images - images.mean()) / images.std()

def process_folder_from_bucket(bucket, prefix: str):
    """Load all images from a GCS folder (with subfolders for classes)"""
    images = []
    labels = []

    blobs = list(bucket.list_blobs(prefix=prefix))

    # Extract class folder names
    class_names = sorted({
        blob.name.split("/")[3]
        for blob in blobs
        if blob.name.lower().endswith(("jpg", "jpeg", "png"))
    })

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
    """Preprocess data from GCS bucket and save tensors locally"""

    os.makedirs(processed_dir, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    train_prefix = f"{raw_prefix}/Training"
    test_prefix = f"{raw_prefix}/Testing"

    print("Processing training data from bucket...")
    X_train, y_train = process_folder_from_bucket(bucket, train_prefix)
    print(f"Training data: {X_train.shape[0]} images")

    print("Processing test data from bucket...")
    X_test, y_test = process_folder_from_bucket(bucket, test_prefix)
    print(f"Test data: {X_test.shape[0]} images")

    torch.save(X_train, os.path.join(processed_dir, "train_images.pt"))
    torch.save(y_train, os.path.join(processed_dir, "train_target.pt"))
    torch.save(X_test, os.path.join(processed_dir, "test_images.pt"))
    torch.save(y_test, os.path.join(processed_dir, "test_target.pt"))

    print("Data processed and saved in", processed_dir)

def load_data():
    """Load processed dataset and return PyTorch TensorDataset"""
    train_images = torch.load("data/processed/train_images.pt")
    train_target = torch.load("data/processed/train_target.pt")
    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set

if __name__ == "__main__":
    typer.run(preprocess)
