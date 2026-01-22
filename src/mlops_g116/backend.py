from fastapi import FastAPI
from contextlib import asynccontextmanager

import anyio
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms
# Local
from src.mlops_g116.model import TumorDetectionModelSimple
# Docker
#from model import TumorDetectionModelSimple

# Constants
MODEL_CHECKPOINT = "models/model.pth"
IMG_SIZE = 224 # Make sure this matches your training size

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager to start and stop the lifespan events."""
    global model, transform, labels

    # 1. Define Labels
    labels = ["glioma", "meningioma", "notumor", "pituitary"]

    # 2. Load Model (CPU only, as requested)
    model = TumorDetectionModelSimple()
    
    # Load weights ensuring they map to CPU
    state_dict = torch.load(MODEL_CHECKPOINT, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Define Transforms
    # We combine the resizing, tensor conversion, AND normalization here.
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        
        # NORMALIZATION STEP:
        # Since we don't have the exact dataset mean/std from training,
        # we use 0.5/0.5 to map inputs to the [-1, 1] range.
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])

    print("Model loaded successfully!")
    yield

    # Clean up
    del model, transform, labels


app = FastAPI(lifespan=lifespan)

def predict_image(image_path: str, top_k: int = 4): # Changed default to 4 (max classes)
    """Predict and return the top results."""
    # 1. FIX: Convert to Grayscale ("L") to match your model input
    img = Image.open(image_path).convert("L")
    img = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img)
    
    probabilities = output.softmax(dim=-1)
    
    # 2. FIX: Ensure top_k doesn't exceed actual number of classes (4)
    # If the user asks for 5, we clamp it to 4 to prevent a crash.
    k = min(top_k, len(labels))
    values, indices = torch.topk(probabilities, k)
    
    # Convert tensors to Python lists
    values = values[0].tolist()
    indices = indices[0].tolist()
    
    results = []
    for i in range(k):
        results.append({
            "class": labels[indices[i]],
            "score": values[i]
        })
        
    return results



@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from the backend!"}

# FastAPI endpoint for image classification@app.post("/classify/")
@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        async with await anyio.open_file(file.filename, "wb") as f:
            await f.write(contents)
            
        # This now returns your 4 tumor classes
        top_predictions = predict_image(file.filename)
        
        return {
            "filename": file.filename, 
            "predictions": top_predictions
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    