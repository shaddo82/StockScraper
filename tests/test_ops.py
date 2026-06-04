from __future__ import annotations

from fastapi.testclient import TestClient
import pandas as pd

import main


client = TestClient(main.app)


def test_feedback_api_saves_user_feedback(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "save_feedback", lambda **kwargs: calls.append(kwargs))

    response = client.post(
        "/api/predictions/feedback",
        json={
            "prediction_id": "pred-1",
            "symbol": "AAPL",
            "prediction": 1,
            "correct_label": 0,
            "confidence": 0.56,
            "deployment": "challenger",
            "model_uri": "models:/stock-direction-model@challenger",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "feedback saved"
    assert calls[0]["prediction_id"] == "pred-1"
    assert calls[0]["correct_label"] == 0


def test_ops_status_summarizes_prediction_and_feedback_logs(monkeypatch):
    monkeypatch.setattr(
        main,
        "read_prediction_logs",
        lambda: [
            {
                "prediction": "1",
                "confidence": "0.56",
                "deployment": "challenger",
            },
            {
                "prediction": "1",
                "confidence": "0.90",
                "deployment": "champion",
            },
        ],
    )
    monkeypatch.setattr(
        main,
        "read_feedback_logs",
        lambda: [
            {
                "prediction": "1",
                "correct_label": "0",
                "deployment": "challenger",
            },
            {
                "prediction": "1",
                "correct_label": "1",
                "deployment": "champion",
            },
        ],
    )
    monkeypatch.setattr(
        main,
        "read_verification_logs",
        lambda: [
            {
                "correct": "1",
                "deployment": "champion",
            },
            {
                "correct": "0",
                "deployment": "challenger",
            },
        ],
    )

    response = client.get("/api/ops/status")
    data = response.json()

    assert response.status_code == 200
    assert data["prediction_count"] == 2
    assert data["feedback_count"] == 2
    assert data["verification_count"] == 2
    assert data["low_confidence_count"] == 1
    assert data["deployment_counts"]["challenger"] == 1
    assert data["wrong_feedback_count"] == 1
    assert data["wrong_feedback_rate"] == 0.5
    assert data["deployment_metrics"]["challenger"]["prediction_count"] == 1
    assert data["deployment_metrics"]["challenger"]["wrong_feedback_rate"] == 1.0
    assert data["deployment_metrics"]["champion"]["verified_accuracy"] == 1.0


def test_verify_prediction_api_saves_actual_outcome(monkeypatch):
    calls = []

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period):
            return pd.DataFrame(
                {"Close": [100.0, 105.0]},
                index=pd.to_datetime(["2026-06-01", "2026-06-02"]),
            )

    monkeypatch.setattr(
        main,
        "read_prediction_logs",
        lambda: [
            {
                "prediction_id": "pred-1",
                "symbol": "AAPL",
                "reference_time": "2026-06-01",
                "reference_close": "100.0",
                "prediction": "1",
                "deployment": "champion",
                "model_uri": "models:/stock-direction-model@champion",
            }
        ],
    )
    monkeypatch.setattr(main.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(main, "save_verification_log", lambda **kwargs: calls.append(kwargs))

    response = client.post("/api/predictions/verify/pred-1")
    data = response.json()

    assert response.status_code == 200
    assert data["actual_label"] == 1
    assert data["correct"] is True
    assert calls[0]["actual_close"] == 105.0
    assert calls[0]["deployment"] == "champion"
