from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

import main


client = TestClient(main.app)


class DummyModel:
    def __init__(self, prediction: int):
        self.prediction = prediction

    def predict(self, feature_frame):
        return [self.prediction]


def _make_history(rows: int = 20) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            "Close": [100 + i for i in range(rows)],
            "Volume": [1_000_000 + i * 10_000 for i in range(rows)],
        },
        index=index,
    )


def test_predict_routes_to_challenger_when_canary_is_enabled(monkeypatch):
    monkeypatch.setattr(main.config, "CANARY_ENABLED", True)
    monkeypatch.setattr(main.config, "CANARY_TRAFFIC_RATIO", 1.0)
    monkeypatch.setattr(
        main.config,
        "CHALLENGER_MODEL_URI",
        "models:/stock-direction-model@challenger",
    )
    monkeypatch.setattr(
        main.config,
        "MODEL_URI",
        "models:/stock-direction-model@champion",
    )

    calls = []

    def fake_load_model(*args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("model_uri") == main.config.CHALLENGER_MODEL_URI:
            return DummyModel(1)
        return DummyModel(0)

    monkeypatch.setattr(main, "load_model", fake_load_model)
    monkeypatch.setattr(
        main,
        "get_model_info",
        lambda model_uri=None: {"model_uri": model_uri or main.config.MODEL_URI},
    )

    result = main._predict_stock_direction_from_history("AAPL", _make_history())

    assert result["deployment"] == "challenger"
    assert result["model_uri"] == main.config.CHALLENGER_MODEL_URI
    assert calls[0]["model_uri"] == main.config.CHALLENGER_MODEL_URI


def test_predict_routes_to_champion_when_canary_is_disabled(monkeypatch):
    monkeypatch.setattr(main.config, "CANARY_ENABLED", False)
    monkeypatch.setattr(main.config, "CANARY_TRAFFIC_RATIO", 1.0)

    calls = []

    def fake_load_model(*args, **kwargs):
        calls.append(kwargs)
        return DummyModel(0)

    monkeypatch.setattr(main, "load_model", fake_load_model)
    monkeypatch.setattr(
        main,
        "get_model_info",
        lambda model_uri=None: {"model_uri": model_uri or main.config.MODEL_URI},
    )

    result = main._predict_stock_direction_from_history("AAPL", _make_history())

    assert result["deployment"] == "champion"
    assert result["model_uri"] == main.config.MODEL_URI
    assert calls[0] == {}
