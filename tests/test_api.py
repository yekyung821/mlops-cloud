import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../opt/mlops/src"))

from fastapi.testclient import TestClient
from webapp import app

client = TestClient(app)

def test_homepage():
    response = client.get("/")
    assert response.status_code == 200

def test_recommend_endpoint():
    response = client.post(
        "/recommend",
        data={"user_id": 10, "top_k": 5}
    )
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
