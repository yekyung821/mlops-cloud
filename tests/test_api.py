import sys
import os
from unittest.mock import patch
import pandas as pd
import pytest

# repo 구조 기준 src 디렉토리까지 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "../opt/mlops/src"))

# webapp import 전에 patch 처리
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "../opt/mlops/dataset/games_log.csv")

with patch("pandas.read_csv", lambda _: pd.read_csv(csv_path)):
    from webapp import app  # import 이후 pd.read_csv는 patch 적용됨

from fastapi.testclient import TestClient

client = TestClient(app)

def test_homepage():
    response = client.get("/")
    assert response.status_code == 200

def test_recommend_endpoint():
    response = client.post(
        "/recommend",
        data={"user_id": 1, "top_k": 5}
    )
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

