import io
import random

from locust import HttpUser, between, task
from PIL import Image


class BackendUser(HttpUser):
    """Simulates a user interacting with the backend API."""

    # 1. Add the host
    host = "https://my-backend-image-277552599633.europe-west1.run.app"

    # Wait time between tasks (simulating user thinking time)
    wait_time = between(1, 3)

    def on_start(self):
        """Optional: Code to run when a user starts (e.g., login)."""
        pass

    @task(1)
    def get_root(self) -> None:
        """Visit the root endpoint."""
        self.client.get("/")

    @task(3)
    def classify_image(self) -> None:
        """Upload a dummy image to the classification endpoint."""

        # 1. Generate a small in-memory dummy image (100x100 grayscale)
        # This prevents needing a real file on disk
        img = Image.new("L", (224, 224), color=random.randint(0, 255))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)

        # 2. Define the payload matching your FastAPI endpoint (file=...)
        files = {"file": ("perf_test.jpg", img_byte_arr, "image/jpeg")}

        # 3. Send the POST request
        # We catch the response to mark it as failure if valid JSON isn't returned
        with self.client.post("/classify/", files=files, catch_response=True) as response:
            if response.status_code == 200:
                if "predictions" not in response.json():
                    response.failure("Response missing 'predictions' key")
            else:
                response.failure(f"Failed with status code: {response.status_code}")
