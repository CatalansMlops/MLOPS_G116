import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from mlops_g116 import frontend

# --- Test get_backend_url ---


@patch("mlops_g116.frontend.run_v2.ServicesClient")
def test_get_backend_url_found_in_cloud(mock_client_cls):
    """Test that we find the Cloud Run service URL if it exists."""
    # Setup the mock GCP client
    mock_client = mock_client_cls.return_value

    # Create a fake service object
    mock_service = Mock()
    mock_service.name = "projects/dtumlops-484509/locations/europe-west1/services/backend"
    mock_service.uri = "https://my-cloud-run-backend.com"

    # Tell list_services to return our fake service
    mock_client.list_services.return_value = [mock_service]

    url = frontend.get_backend_url()
    assert url == "https://my-cloud-run-backend.com"


@patch("mlops_g116.frontend.run_v2.ServicesClient")
def test_get_backend_url_fallback_local(mock_client_cls):
    """Test fallback to local/env var if Cloud Run service is not found."""

    # Clear any cached value
    frontend.get_backend_url.clear()
    # Setup mock to return EMPTY list of services
    mock_client = mock_client_cls.return_value
    mock_client.list_services.return_value = []

    # Case 1: Default fallback
    # We clear environ to ensure no BACKEND var exists
    with patch.dict(os.environ, {}, clear=True):
        url = frontend.get_backend_url()
        assert url == "http://127.0.0.1:8000"

    # Clear cache again for next case
    frontend.get_backend_url.clear()

    # Case 2: Environment variable set
    with patch.dict(os.environ, {"BACKEND": "http://custom-url:9999"}, clear=True):
        url = frontend.get_backend_url()
        assert url == "http://custom-url:9999"


# --- Test classify_image ---


@patch("mlops_g116.frontend.requests.post")
def test_classify_image_success(mock_post):
    """Test that classify_image returns JSON when status is 200."""
    # Mock the response object
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"predictions": [{"class": "A", "score": 0.9}]}
    mock_post.return_value = mock_response

    result = frontend.classify_image(b"fake_image_bytes", "http://backend")

    assert result == {"predictions": [{"class": "A", "score": 0.9}]}
    mock_post.assert_called_once()


@patch("mlops_g116.frontend.requests.post")
def test_classify_image_failure(mock_post):
    """Test that classify_image returns None on API error."""
    mock_response = Mock()
    mock_response.status_code = 500  # Internal Server Error
    mock_post.return_value = mock_response

    result = frontend.classify_image(b"fake_image_bytes", "http://backend")

    assert result is None


# --- Test Main (The Streamlit UI) ---


@patch("mlops_g116.frontend.st")
@patch("mlops_g116.frontend.classify_image")
@patch("mlops_g116.frontend.get_backend_url")
def test_main_happy_path(mock_get_url, mock_classify, mock_st):
    """Test the full flow: Upload Image -> Classify -> Show Results."""
    # 1. Setup Backend URL
    mock_get_url.return_value = "http://backend"

    # 2. Simulate User Uploading a File
    mock_file = Mock()
    mock_file.read.return_value = b"fake_image_data"
    mock_st.file_uploader.return_value = mock_file

    # 3. Simulate Backend Returning Predictions
    mock_classify.return_value = {
        "predictions": [
            {"class": "glioma", "score": 0.85},
            {"class": "meningioma", "score": 0.15},
        ]
    }

    # Run the main function
    frontend.main()

    # Assertions: Check if Streamlit functions were called
    mock_st.title.assert_called_once()
    mock_st.image.assert_called_once_with(b"fake_image_data", caption="Uploaded Image")

    # Check if we displayed the text result
    # We check if st.write was called with a string containing the prediction
    args, _ = mock_st.write.call_args
    assert "glioma" in args[0]
    assert "85.00%" in args[0]

    # Check if chart was drawn
    mock_st.bar_chart.assert_called_once()


@patch("mlops_g116.frontend.st")
@patch("mlops_g116.frontend.get_backend_url")
def test_main_no_backend_error(mock_get_url, mock_st):
    """Test that main raises an error if backend URL is somehow None."""
    mock_get_url.return_value = None

    with pytest.raises(ValueError, match="Backend service not found"):
        frontend.main()


@patch("mlops_g116.frontend.st")
@patch("mlops_g116.frontend.classify_image")
@patch("mlops_g116.frontend.get_backend_url")
def test_main_api_failure(mock_get_url, mock_classify, mock_st):
    """Test flow when backend API fails (returns None)."""
    mock_get_url.return_value = "http://backend"

    # Simulate file upload
    mock_st.file_uploader.return_value = Mock()

    # Simulate API Failure
    mock_classify.return_value = None

    frontend.main()

    # Should display failure message
    mock_st.write.assert_called_with("Failed to get prediction")
