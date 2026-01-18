import bentoml
from pathlib import Path

if __name__ == "__main__":
    image_path = Path("/root/dtu/mlops/MLOPS_G116/person.jpg")

    with bentoml.SyncHTTPClient("http://localhost:3000") as client:
        # We pass the file path directly to the 'image_file' argument
        response = client.predict(image_file=image_path)
        
        print(f"Predictions for {image_path}:")
        for i, pred in enumerate(response):
            print(f"{i+1}. {pred['class']}: {pred['probability']:.2%}")