from fastapi import FastAPI
import json
from contextlib import asynccontextmanager

import anyio
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import models, transforms


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager to start and stop the lifespan events of the FastAPI application."""
    global model, transform, imagenet_classes
    # Load model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )

    async with await anyio.open_file("imagenet-simple-labels.json") as f:
        content = await f.read()  # Read the file asynchronously
        imagenet_classes = json.loads(content) # Parse the JSON content

    yield

    # Clean up
    del model
    del transform
    del imagenet_classes


app = FastAPI(lifespan=lifespan)

def predict_image(image_path: str, top_k: int = 5):
    """Predict and return only the top K results."""
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img)
    
    probabilities = output.softmax(dim=-1)
    
    # Get the top K probabilities and their indices
    # values: the actual scores (0.95, 0.03...)
    # indices: the class IDs (234, 5, 89...)
    values, indices = torch.topk(probabilities, top_k)
    
    # Convert tensors to Python lists
    values = values[0].tolist()
    indices = indices[0].tolist()
    
    # Create a clean list of dictionaries
    results = []
    for i in range(top_k):
        results.append({
            "class": imagenet_classes[indices[i]],
            "score": values[i]
        })
        
    return results



@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from the backend!"}


@app.get("/items/{item_id}")
def read_item(item_id: int):
    """Get an item by id."""
    return {"item_id": item_id}


# FastAPI endpoint for image classification@app.post("/classify/")
@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        async with await anyio.open_file(file.filename, "wb") as f:
            await f.write(contents)
            
        # This now returns a nice list: [{'class': 'cat', 'score': 0.95}, ...]
        top_predictions = predict_image(file.filename)
        
        return {
            "filename": file.filename, 
            "predictions": top_predictions
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    