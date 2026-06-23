"""ML configuration for training and serving."""
from pathlib import Path
import os


def get_env(name: str, default: str) -> str:
    return os.getenv(name) or default


def get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BASE_DIR / "ml"
ARTIFACT_DIR = ML_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "stock_direction_model.joblib"

MLFLOW_TRACKING_URI = get_env("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = get_env(
    "MLFLOW_EXPERIMENT_NAME",
    "stock-direction-classifier",
)
MODEL_REGISTRY_NAME = get_env(
    "MODEL_REGISTRY_NAME",
    "stock-direction-model",
)
MODEL_URI = get_env("MODEL_URI", f"models:/{MODEL_REGISTRY_NAME}@champion")
CHALLENGER_MODEL_URI = get_env(
    "CHALLENGER_MODEL_URI",
    f"models:/{MODEL_REGISTRY_NAME}@challenger",
)
MODEL_FALLBACK_TO_LOCAL = get_env_bool(
    "MODEL_FALLBACK_TO_LOCAL",
    True,
)
CANARY_ENABLED = get_env_bool("CANARY_ENABLED", False)
CANARY_TRAFFIC_RATIO = get_env_float("CANARY_TRAFFIC_RATIO", 0.1)

DEFAULT_TICKERS = ["AAPL", "GOOGL", "MSFT", "NVDA", "AMZN"]
TRAINING_TICKERS = [
    "AAPL",
    "GOOGL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "TSLA",
    "JPM",
    "JNJ",
    "XOM",
]
TRAINING_PERIOD = get_env("TRAINING_PERIOD", "5y")
FEATURE_COLUMNS = [
    "return_1d",
    "return_3d",
    "ma_5_gap",
    "ma_10_gap",
    "volatility_5",
    "volume_change_1d",
]
TARGET_COLUMN = "target"
MIN_HISTORY_ROWS = 15
