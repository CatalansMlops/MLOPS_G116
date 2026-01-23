"""FastAPI backend for brain tumor classification."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

import anyio
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from mlops_g116.model import ResNet18

MODEL_CHECKPOINT = "models/model.pth"
IMG_SIZE = 224


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and tear down global model resources.

    Args:
        app: FastAPI application instance.
    """
    del app
    global model, transform, labels

    labels = ["glioma", "meningioma", "notumor", "pituitary"]

    # 2. Load Model (CPU only, as requested)
    model = ResNet18()

    # Load weights ensuring they map to CPU
    state_dict = torch.load(MODEL_CHECKPOINT, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    print("Model loaded successfully.")
    yield

    del model, transform, labels


app = FastAPI(lifespan=lifespan)


def predict_image(image_path: str, top_k: int = 4) -> list[dict[str, float | str]]:
    """Predict the top classes for a local image path.

    Args:
        image_path: Path to the image stored on disk.
        top_k: Maximum number of classes to return.

    Returns:
        List of dictionaries with class names and probability scores.
    """
    img = Image.open(image_path).convert("L")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    probabilities = output.softmax(dim=-1)

    k = min(top_k, len(labels))
    values, indices = torch.topk(probabilities, k)

    values = values[0].tolist()
    indices = indices[0].tolist()

    results = []
    for i in range(k):
        results.append({"class": labels[indices[i]], "score": values[i]})

    return results


@app.get("/")
async def root() -> dict[str, str]:
    """Return a health message for the backend."""
    return {"message": "Hello from the backend!"}


@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)) -> dict[str, object]:
    """Classify an uploaded image file.

    The file is written to disk before inference. Make sure the model
    checkpoint is present in the container or runtime environment.

    Args:
        file: Uploaded image file.

    Returns:
        Response payload with filename and predictions.
    """
    try:
        contents = await file.read()
        async with await anyio.open_file(file.filename, "wb") as f:
            await f.write(contents)

        top_predictions = predict_image(file.filename)

        return {"filename": file.filename, "predictions": top_predictions}
    except Exception as exc:
        print(f"Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
