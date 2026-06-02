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
_model_cache: dict[str, Any] = {}
_model_info_cache: dict[str, dict[str, Any]] = {}


def _loaded_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cache_key(model_uri: Optional[str]) -> str:
    return model_uri or config.MODEL_URI


def _load_from_mlflow(model_uri: str) -> Any:
    if mlflow is None:
        raise RuntimeError(f"mlflow import failed: {_mlflow_import_error}")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(model_uri)


def _load_from_local_path() -> Any:
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {config.MODEL_PATH}")
    return joblib.load(config.MODEL_PATH)


def _store_cache(cache_key: str, model: Any, model_info: dict[str, Any]) -> None:
    global _model, _model_info

    _model_cache[cache_key] = model
    _model_info_cache[cache_key] = model_info
    if cache_key == config.MODEL_URI:
        _model = model
        _model_info = model_info


def load_model(
    model_uri: Optional[str] = None,
    force_reload: bool = False,
    allow_local_fallback: Optional[bool] = None,
) -> Any:
    """Load the serving model, preferring MLflow registry and falling back to local artifact."""
    global _model, _model_info

    selected_model_uri = model_uri or config.MODEL_URI
    cache_key = _cache_key(selected_model_uri)

    if cache_key in _model_cache and not force_reload:
        return _model_cache[cache_key]

    if allow_local_fallback is None:
        allow_local_fallback = (
            selected_model_uri == config.MODEL_URI
            and config.MODEL_FALLBACK_TO_LOCAL
        )

    errors: list[str] = []

    if selected_model_uri:
        try:
            model = _load_from_mlflow(selected_model_uri)
            model_info = {
                "source": "mlflow",
                "model_uri": selected_model_uri,
                "tracking_uri": config.MLFLOW_TRACKING_URI,
                "artifact_path": str(config.MODEL_PATH),
                "fallback_to_local": allow_local_fallback,
                "loaded_at": _loaded_at(),
            }
            _store_cache(cache_key, model, model_info)
            return model
        except Exception as exc:  # pragma: no cover - registry may be unavailable locally
            errors.append(str(exc))

    if not allow_local_fallback:
        model_info = {
            "source": "unavailable",
            "model_uri": selected_model_uri,
            "tracking_uri": config.MLFLOW_TRACKING_URI,
            "artifact_path": str(config.MODEL_PATH),
            "fallback_to_local": allow_local_fallback,
            "errors": errors + ["Local model fallback is disabled"],
        }
        _model_cache.pop(cache_key, None)
        _model_info_cache[cache_key] = model_info
        if cache_key == config.MODEL_URI:
            _model = None
            _model_info = model_info
        raise FileNotFoundError("No trained model is available")

    try:
        model = _load_from_local_path()
        model_info = {
            "source": "local",
            "model_uri": None,
            "tracking_uri": config.MLFLOW_TRACKING_URI,
            "artifact_path": str(config.MODEL_PATH),
            "fallback_to_local": allow_local_fallback,
            "fallback_errors": errors,
            "loaded_at": _loaded_at(),
        }
        _store_cache(cache_key, model, model_info)
        return model
    except Exception as exc:
        errors.append(str(exc))
        model_info = {
            "source": "unavailable",
            "model_uri": selected_model_uri,
            "tracking_uri": config.MLFLOW_TRACKING_URI,
            "artifact_path": str(config.MODEL_PATH),
            "fallback_to_local": allow_local_fallback,
            "errors": errors,
        }
        _model_cache.pop(cache_key, None)
        _model_info_cache[cache_key] = model_info
        if cache_key == config.MODEL_URI:
            _model = None
            _model_info = model_info
        raise FileNotFoundError("No trained model is available") from exc


def reload_model(model_uri: Optional[str] = None) -> dict[str, Any]:
    """Force the serving process to discard the cached model and load the current target."""
    selected_model_uri = model_uri or config.MODEL_URI
    load_model(model_uri=selected_model_uri, force_reload=True)
    return get_model_info(selected_model_uri)


def get_model_info(model_uri: Optional[str] = None) -> dict[str, Any]:
    """Return the current cached model metadata."""
    selected_model_uri = model_uri or config.MODEL_URI
    cache_key = _cache_key(selected_model_uri)
    global _model_info

    if cache_key in _model_info_cache:
        return _model_info_cache[cache_key]

    if cache_key == config.MODEL_URI and (_model_info is None or _model_info.get("source") == "unavailable"):
        try:
            load_model(selected_model_uri)
        except Exception:
            pass

    return _model_info_cache.get(cache_key) or _model_info or {
        "source": "unavailable",
        "model_uri": selected_model_uri,
        "tracking_uri": config.MLFLOW_TRACKING_URI,
        "artifact_path": str(config.MODEL_PATH),
        "fallback_to_local": (
            config.MODEL_FALLBACK_TO_LOCAL
            if selected_model_uri == config.MODEL_URI
            else False
        ),
    }
