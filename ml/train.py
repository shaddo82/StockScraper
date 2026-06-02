"""Train a stock direction classifier and register it with MLflow when available."""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app import config
from ml.features import FEATURE_COLUMNS, TARGET_COLUMN, build_training_frame

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency
    yf = None

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
except Exception:  # pragma: no cover - optional dependency
    mlflow = None


@dataclass
class TrainingResult:
    model_name: str
    accuracy: float
    f1: float
    precision: float
    recall: float
    train_rows: int
    test_rows: int
    artifact_path: str


def fetch_history(symbol: str, period: str = "2y") -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed")

    history = yf.Ticker(symbol).history(period=period)
    if history is None or history.empty:
        raise ValueError(f"no history available for {symbol}")
    history = history.copy()
    history["symbol"] = symbol
    return history


def _make_estimator(model_name: str) -> Pipeline | RandomForestClassifier:
    if model_name == "logistic":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=500, random_state=42),
                ),
            ]
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"unsupported model_name: {model_name}")


def _log_with_mlflow(
    *,
    model_name: str,
    estimator,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    result: TrainingResult,
) -> None:
    if mlflow is None or os.getenv("DISABLE_MLFLOW") == "1":
        return

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    input_example = test_frame[FEATURE_COLUMNS].head(1)
    signature = infer_signature(input_example, estimator.predict(input_example))

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))
        mlflow.log_param("train_rows", result.train_rows)
        mlflow.log_param("test_rows", result.test_rows)
        mlflow.log_metric("accuracy", result.accuracy)
        mlflow.log_metric("f1", result.f1)
        mlflow.log_metric("precision", result.precision)
        mlflow.log_metric("recall", result.recall)
        try:
            mlflow.sklearn.log_model(
                estimator,
                artifact_path="model",
                registered_model_name=config.MODEL_REGISTRY_NAME,
                input_example=input_example,
                signature=signature,
            )
        except Exception as exc:
            mlflow.set_tag("model_artifact_log_status", "failed")
            mlflow.set_tag("model_artifact_log_error", str(exc))
            print(f"MLflow model artifact logging failed: {exc}")


def train_from_histories(
    histories: Sequence[pd.DataFrame],
    model_name: str = "logistic",
) -> TrainingResult:
    frames = [build_training_frame(history) for history in histories]
    training_frame = pd.concat(frames, axis=0).sort_index()
    if len(training_frame) < 20:
        raise ValueError("training data is too small")

    split_index = max(int(len(training_frame) * 0.8), 1)
    if split_index >= len(training_frame):
        split_index = len(training_frame) - 1

    train_frame = training_frame.iloc[:split_index]
    test_frame = training_frame.iloc[split_index:]
    if train_frame.empty or test_frame.empty:
        raise ValueError("train/test split produced an empty partition")

    estimator = _make_estimator(model_name)
    estimator.fit(train_frame[FEATURE_COLUMNS], train_frame[TARGET_COLUMN])
    predictions = estimator.predict(test_frame[FEATURE_COLUMNS])

    result = TrainingResult(
        model_name=model_name,
        accuracy=accuracy_score(test_frame[TARGET_COLUMN], predictions),
        f1=f1_score(test_frame[TARGET_COLUMN], predictions, zero_division=0),
        precision=precision_score(
            test_frame[TARGET_COLUMN], predictions, zero_division=0
        ),
        recall=recall_score(test_frame[TARGET_COLUMN], predictions, zero_division=0),
        train_rows=len(train_frame),
        test_rows=len(test_frame),
        artifact_path=str(config.MODEL_PATH),
    )

    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, config.MODEL_PATH)
    _log_with_mlflow(
        model_name=model_name,
        estimator=estimator,
        train_frame=train_frame,
        test_frame=test_frame,
        result=result,
    )
    return result


def train_from_symbols(
    symbols: Iterable[str],
    period: str = "2y",
    model_name: str = "logistic",
) -> TrainingResult:
    histories = [fetch_history(symbol, period=period) for symbol in symbols]
    return train_from_histories(histories, model_name=model_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the stock direction model")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=config.DEFAULT_TICKERS,
        help="Ticker symbols to include in training",
    )
    parser.add_argument(
        "--period",
        default="2y",
        help="yfinance history period, e.g. 1y or 2y",
    )
    parser.add_argument(
        "--model",
        choices=("logistic", "random_forest"),
        default="logistic",
        help="Model family to train",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_from_symbols(
        args.symbols,
        period=args.period,
        model_name=args.model,
    )
    print(asdict(result))


if __name__ == "__main__":
    main()
