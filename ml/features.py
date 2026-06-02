"""Shared feature engineering for training and inference."""
from __future__ import annotations

import pandas as pd

from app.config import FEATURE_COLUMNS, MIN_HISTORY_ROWS, TARGET_COLUMN


def _prepare_history_frame(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        raise ValueError("price history is empty")

    required_columns = {"Close", "Volume"}
    missing_columns = required_columns.difference(history.columns)
    if missing_columns:
        raise ValueError(f"missing required columns: {sorted(missing_columns)}")

    frame = history.copy().sort_index()
    frame["return_1d"] = frame["Close"].pct_change(1)
    frame["return_3d"] = frame["Close"].pct_change(3)
    ma_5 = frame["Close"].rolling(5).mean()
    ma_10 = frame["Close"].rolling(10).mean()
    frame["ma_5_gap"] = frame["Close"] / ma_5 - 1.0
    frame["ma_10_gap"] = frame["Close"] / ma_10 - 1.0
    frame["volatility_5"] = frame["Close"].pct_change().rolling(5).std()
    frame["volume_change_1d"] = frame["Volume"].pct_change(1)
    frame["target"] = (frame["Close"].shift(-1) > frame["Close"]).astype("int")
    frame = frame.replace([pd.NA, float("inf"), float("-inf")], pd.NA)
    frame = frame.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    return frame


def build_training_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Return rows with engineered features and next-day direction labels."""
    frame = _prepare_history_frame(history)
    if len(frame) < MIN_HISTORY_ROWS:
        raise ValueError(
            f"not enough rows for training: {len(frame)} < {MIN_HISTORY_ROWS}"
        )
    return frame


def build_latest_feature_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Return the latest row of features for prediction."""
    frame = _prepare_history_frame(history)
    if frame.empty:
        raise ValueError("not enough rows to build a prediction row")
    return frame[FEATURE_COLUMNS].tail(1).reset_index(drop=True)
