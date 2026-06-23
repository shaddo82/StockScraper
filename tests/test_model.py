from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest

from app import config
from ml.features import build_latest_feature_frame, build_training_frame
from ml.train import _make_estimator, train_from_histories


def _make_history(rows: int = 60) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = pd.Series([100 + i * 0.8 + (i % 5) * 0.2 for i in range(rows)], index=index)
    volume = pd.Series([1_000_000 + i * 5_000 for i in range(rows)], index=index)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=index)


def test_build_training_frame_creates_targeted_features():
    history = _make_history()
    frame = build_training_frame(history)

    assert not frame.empty
    assert set(config.FEATURE_COLUMNS).issubset(frame.columns)
    assert config.TARGET_COLUMN in frame.columns


def test_make_estimator_supports_two_candidates():
    assert _make_estimator("logistic") is not None
    assert _make_estimator("random_forest") is not None


def test_train_and_load_model_roundtrip(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    model_path = artifact_dir / "stock_direction_model.joblib"
    monkeypatch.setattr(config, "ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(config, "MODEL_PATH", model_path)
    monkeypatch.setenv("DISABLE_MLFLOW", "1")

    import app.model_loader as model_loader

    model_loader._model = None
    model_loader._model_info = None

    result = train_from_histories([_make_history()], model_name="logistic")

    assert Path(result.artifact_path).exists()
    assert result.accuracy >= 0.0
    assert result.f1 >= 0.0

    model_loader = importlib.reload(model_loader)
    monkeypatch.setattr(model_loader.config, "ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(model_loader.config, "MODEL_PATH", model_path)
    monkeypatch.setattr(model_loader.config, "MODEL_URI", "")
    monkeypatch.setattr(model_loader.config, "MODEL_FALLBACK_TO_LOCAL", True)
    model_loader._model = None
    model_loader._model_info = None

    model = model_loader.load_model(force_reload=True)
    assert model_loader.reload_model()["source"] == "local"

    latest_features = build_latest_feature_frame(_make_history())
    prediction = model.predict(latest_features)

    assert len(prediction) == 1
    assert prediction[0] in (0, 1)


def test_model_loader_can_disable_local_fallback(tmp_path, monkeypatch):
    import app.model_loader as model_loader

    model_loader = importlib.reload(model_loader)
    monkeypatch.setattr(model_loader.config, "MODEL_URI", "models:/missing@champion")
    monkeypatch.setattr(model_loader.config, "MODEL_PATH", tmp_path / "old_model.joblib")
    monkeypatch.setattr(model_loader.config, "MODEL_FALLBACK_TO_LOCAL", False)
    monkeypatch.setattr(
        model_loader,
        "_load_from_mlflow",
        lambda: (_ for _ in ()).throw(RuntimeError("registry unavailable")),
    )
    model_loader._model = None
    model_loader._model_info = None

    with pytest.raises(FileNotFoundError):
        model_loader.load_model(force_reload=True)

    info = model_loader.get_model_info()

    assert info["source"] == "unavailable"
    assert info["fallback_to_local"] is False
    assert "Local model fallback is disabled" in info["errors"]


def test_model_loader_falls_back_to_local_when_registry_unavailable(monkeypatch):
    import app.model_loader as model_loader

    model_loader = importlib.reload(model_loader)
    local_model = object()
    monkeypatch.setattr(model_loader.config, "MODEL_URI", "models:/missing@champion")
    monkeypatch.setattr(model_loader.config, "MODEL_FALLBACK_TO_LOCAL", True)
    monkeypatch.setattr(
        model_loader,
        "_load_from_mlflow",
        lambda model_uri: (_ for _ in ()).throw(RuntimeError("registry unavailable")),
    )
    monkeypatch.setattr(model_loader, "_load_from_local_path", lambda: local_model)
    model_loader._model = None
    model_loader._model_info = None
    model_loader._model_cache = {}
    model_loader._model_info_cache = {}

    model = model_loader.load_model(force_reload=True)
    info = model_loader.get_model_info()

    assert model is local_model
    assert info["source"] == "local"
    assert info["fallback_to_local"] is True
    assert "registry unavailable" in info["fallback_errors"]
