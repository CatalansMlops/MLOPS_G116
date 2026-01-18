from fastapi.testclient import TestClient
from src.mlops_g116.api import app
from http import HTTPStatus
client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == { "message": HTTPStatus.OK.phrase,
                                "status-code": HTTPStatus.OK}