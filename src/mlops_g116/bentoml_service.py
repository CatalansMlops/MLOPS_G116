from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import bentoml
import numpy as np
from onnxruntime import InferenceSession
from PIL import Image


# --- Helper Function for Softmax ---
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# --- Service 1: The Preprocessor ---
@bentoml.service
class ImagePreprocessorService:
    """Handles raw image loading and normalization."""

    @bentoml.api
    def preprocess(self, image_file: Path) -> np.ndarray:
        # 1. Open image
        image = Image.open(image_file).convert("RGB")

        # 2. Resize to 224x224
        image = image.resize((224, 224))

        # 3. Convert to numpy and normalize to [0, 1]
        image_arr = np.array(image).astype(np.float32) / 255.0

        # 4. Standard ImageNet Normalization (Mean/Std)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_arr = (image_arr - mean) / std

        # 5. Transpose to (Channels, Height, Width)
        image_arr = np.transpose(image_arr, (2, 0, 1))

        # 6. Add batch dimension -> (1, 3, 224, 224)
        return np.expand_dims(image_arr, axis=0)


# --- Service 2: The Classifier (The Main Entry Point) ---
@bentoml.service
class ImageClassifierService:
    """Orchestrates preprocessing and runs inference."""

    # Inject dependency on the preprocessor
    preprocessor = bentoml.depends(ImagePreprocessorService)

    def __init__(self) -> None:
        # Load Model
        self.model = InferenceSession("resnet18.onnx")

        # Load Labels
        print("Loading ImageNet labels...")
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        with urllib.request.urlopen(url) as f:
            self.labels = json.load(f)

    @bentoml.api
    async def predict(self, image_file: Path) -> list:
        # 1. Call the preprocessor service (asynchronously)
        # Note: We pass the Path object directly.
        input_tensor = await self.preprocessor.to_async.preprocess(image_file)

        # 2. Run Inference
        output = self.model.run(None, {"input": input_tensor.astype(np.float32)})
        scores = output[0][0]

        # 3. Post-processing (Top 5)
        probabilities = softmax(scores)
        top_5_indices = np.argsort(probabilities)[-5:][::-1]

        results = []
        for idx in top_5_indices:
            results.append({"class": self.labels[idx], "probability": float(probabilities[idx])})

        return results
