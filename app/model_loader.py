"""Model loading helpers for serving the latest trained model."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import joblib

from app import config

try:
    import mlflow
    import mlflow.sklearn
except Exception as exc:  # pragma: no cover - optional dependency
    mlflow = None
    _mlflow_import_error = exc
else:
    _mlflow_import_error = None


_model: Any = None
_model_info: Optional[dict[str, Any]] = None


def _loaded_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_from_mlflow() -> Any:
    if mlflow is None:
        raise RuntimeError(f"mlflow import failed: {_mlflow_import_error}")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(config.MODEL_URI)


def _load_from_local_path() -> Any:
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {config.MODEL_PATH}")
    return joblib.load(config.MODEL_PATH)


def load_model(force_reload: bool = False) -> Any:
    """Load the serving model, preferring MLflow registry and falling back to local artifact."""
    global _model, _model_info

    if _model is not None and not force_reload:
        return _model

    errors: list[str] = []

    if config.MODEL_URI:
        try:
            _model = _load_from_mlflow()
            _model_info = {
                "source": "mlflow",
                "model_uri": config.MODEL_URI,
                "tracking_uri": config.MLFLOW_TRACKING_URI,
                "artifact_path": str(config.MODEL_PATH),
                "fallback_to_local": config.MODEL_FALLBACK_TO_LOCAL,
                "loaded_at": _loaded_at(),
            }
            return _model
        except Exception as exc:  # pragma: no cover - registry may be unavailable locally
            errors.append(str(exc))

    if not config.MODEL_FALLBACK_TO_LOCAL:
        _model = None
        _model_info = {
            "source": "unavailable",
            "model_uri": config.MODEL_URI,
            "tracking_uri": config.MLFLOW_TRACKING_URI,
            "artifact_path": str(config.MODEL_PATH),
            "fallback_to_local": False,
            "errors": errors + ["Local model fallback is disabled"],
        }
        raise FileNotFoundError("No trained model is available")

    try:
        _model = _load_from_local_path()
        _model_info = {
            "source": "local",
            "model_uri": None,
            "tracking_uri": config.MLFLOW_TRACKING_URI,
            "artifact_path": str(config.MODEL_PATH),
            "fallback_to_local": config.MODEL_FALLBACK_TO_LOCAL,
            "fallback_errors": errors,
            "loaded_at": _loaded_at(),
        }
        return _model
    except Exception as exc:
        errors.append(str(exc))
        _model = None
        _model_info = {
            "source": "unavailable",
            "model_uri": config.MODEL_URI,
            "tracking_uri": config.MLFLOW_TRACKING_URI,
            "artifact_path": str(config.MODEL_PATH),
            "fallback_to_local": config.MODEL_FALLBACK_TO_LOCAL,
            "errors": errors,
        }
        raise FileNotFoundError("No trained model is available") from exc


def reload_model() -> dict[str, Any]:
    """Force the serving process to discard the cached model and load the current target."""
    load_model(force_reload=True)
    return get_model_info()


def get_model_info() -> dict[str, Any]:
    """Return the current cached model metadata."""
    global _model_info

    if _model_info is None or _model_info.get("source") == "unavailable":
        try:
            load_model()
        except Exception:
            pass

    return _model_info or {
        "source": "unavailable",
        "model_uri": config.MODEL_URI,
        "tracking_uri": config.MLFLOW_TRACKING_URI,
        "artifact_path": str(config.MODEL_PATH),
        "fallback_to_local": config.MODEL_FALLBACK_TO_LOCAL,
    }
