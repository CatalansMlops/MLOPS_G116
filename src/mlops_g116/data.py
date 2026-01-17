import os
from PIL import Image
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
import typer

IMG_SIZE = 224  # Image size (IMG_ZISE x IMG_SIZE) 

# Transformations
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),   # Converts [C,H,W] and scales to [0,1]
])

def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalizes images: mean 0, std 1"""
    return (images - images.mean()) / images.std()

def preprocess(raw_dir: str = "data/raw/brain_dataset", processed_dir: str = "data/processed") -> None:
    os.makedirs(processed_dir, exist_ok=True)

    images = []
    labels = []

    class_folders = sorted(os.listdir(raw_dir)) 

    print("Loading images...")

    for label, folder in enumerate(class_folders):
        folder_path = os.path.join(raw_dir, folder)

        for file in os.listdir(folder_path):
            if file.lower().endswith(("jpg", "jpeg", "png")):
                img_path = os.path.join(folder_path, file)
                img = Image.open(img_path).convert("L")  # "L" for grayscale ("RGB" for color)
                img = transform(img)
                images.append(img)
                labels.append(label)

    images = torch.stack(images)  # [N, 1, H, W]
    labels = torch.tensor(labels)

    images = normalize(images)

    # Divide in train/test (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Save tensors
    torch.save(X_train, f"{processed_dir}/train_images.pt")
    torch.save(y_train, f"{processed_dir}/train_target.pt")
    torch.save(X_test, f"{processed_dir}/test_images.pt")
    torch.save(y_test, f"{processed_dir}/test_target.pt")

    print("Data processed and saved in", processed_dir)

def load_data():
    """Loads dataset processed and returns TensorDataset for PyTorch"""
    train_images = torch.load("data/processed/train_images.pt")
    train_target = torch.load("data/processed/train_target.pt")
    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set

if __name__ == "__main__":
    typer.run(preprocess)
