"""ML configuration for training and serving."""
from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BASE_DIR / "ml"
ARTIFACT_DIR = ML_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "stock_direction_model.joblib"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "stock-direction-classifier",
)
MODEL_REGISTRY_NAME = os.getenv(
    "MODEL_REGISTRY_NAME",
    "stock-direction-model",
)
MODEL_URI = os.getenv("MODEL_URI", f"models:/{MODEL_REGISTRY_NAME}@champion")

DEFAULT_TICKERS = ["AAPL", "GOOGL", "MSFT", "NVDA", "AMZN"]
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
