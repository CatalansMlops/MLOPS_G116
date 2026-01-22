import os
import io
from fastapi.testclient import TestClient
from PIL import Image
# Adjust this import based on where your backend.py is located relative to tests/
# If backend.py is in src/mlops_g116/backend.py:
from src.mlops_g116.backend import app 

def test_read_root():
    """Verify the root endpoint returns the expected welcome message."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello from the backend!"}

def test_classify_endpoint():
    """Test the classify endpoint handles valid image uploads and returns predictions."""
    
    # Generate a simple in-memory grayscale image
    image = Image.new('L', (224, 224), color=255) 
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    # Define multipart/form-data payload
    files = {
        "file": ("test_image.jpg", img_byte_arr, "image/jpeg")
    }

    # Use context manager to ensure app startup/shutdown events run
    with TestClient(app) as client:
        response = client.post("/classify/", files=files)

        # Validate response status and structure
        assert response.status_code == 200
        data = response.json()
        
        assert data["filename"] == "test_image.jpg"
        assert len(data["predictions"]) == 4 
        
        # Verify prediction schema
        first_pred = data["predictions"][0]
        assert "class" in first_pred
        assert "score" in first_pred

    # Cleanup temporary file created by the backend
    if os.path.exists("test_image.jpg"):
        os.remove("test_image.jpg")