import sys
import os

# repo 구조 기준 src 디렉토리까지 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "../opt/mlops/src"))

from fastapi.testclient import TestClient
from webapp import app
import pandas as pd

# webapp.py에서 game_df 읽는 부분을 상대 경로로 변경
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "../opt/mlops/dataset/games_log.csv")

# 테스트 환경에서만 game_df를 읽도록 patch
from unittest.mock import patch

import pytest

@pytest.fixture(autouse=True)
def mock_game_df():
    # pandas.read_csv를 patch해서 상대 경로 사용
    with patch("pandas.read_csv", lambda _: pd.read_csv(csv_path)):
        yield

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

