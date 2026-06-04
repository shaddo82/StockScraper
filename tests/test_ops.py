from __future__ import annotations

from fastapi.testclient import TestClient

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
                "confidence": "0.56",
                "deployment": "challenger",
            },
            {
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
            },
            {
                "prediction": "1",
                "correct_label": "1",
            },
        ],
    )

    response = client.get("/api/ops/status")
    data = response.json()

    assert response.status_code == 200
    assert data["prediction_count"] == 2
    assert data["feedback_count"] == 2
    assert data["low_confidence_count"] == 1
    assert data["deployment_counts"]["challenger"] == 1
    assert data["wrong_feedback_count"] == 1
    assert data["wrong_feedback_rate"] == 0.5
