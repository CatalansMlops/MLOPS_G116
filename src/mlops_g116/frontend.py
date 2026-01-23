import os

import pandas as pd
import requests
import streamlit as st
from google.cloud import run_v2


@st.cache_resource
def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/dtumlops-484509/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "my-backend-image":
            print(service.uri)
            return service.uri

    # Fallback to local environment variable if not found
    return os.environ.get("BACKEND", "http://127.0.0.1:8000")


def classify_image(image, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/classify/"
    response = requests.post(predict_url, files={"file": image}, timeout=10)
    if response.status_code == 200:
        return response.json()
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Image Classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        result = classify_image(image, backend=backend)

        if result is not None:
            # The backend now gives us exactly what we need
            predictions = result["predictions"]

            # Display Image
            st.image(image, caption="Uploaded Image")

            # Display Text Prediction (The first one is the best one)
            best_class = predictions[0]["class"]
            best_score = predictions[0]["score"]
            st.write(f"**Prediction:** {best_class} ({best_score:.2%})")

            # Create DataFrame directly from the clean list
            df = pd.DataFrame(predictions)

            # Rename columns to match Streamlit expects
            df.columns = ["Class", "Probability"]
            df.set_index("Class", inplace=True)

            # Plot
            st.bar_chart(df, y="Probability")
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()
