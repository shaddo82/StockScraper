"""Prediction logging helpers for operation monitoring."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from app.google_sheet_logger import PREDICTION_WORKSHEET, append_row


PREDICTION_LOG_PATH = Path("logs/predictions.csv")
PREDICTION_COLUMNS = [
    "time",
    "prediction_id",
    "symbol",
    "reference_time",
    "reference_close",
    "prediction",
    "direction",
    "confidence",
    "deployment",
    "model_uri",
]


def _format_confidence(confidence: float | None) -> str:
    if confidence is None:
        return ""
    return f"{float(confidence):.4f}"


def _append_csv(path: Path, columns: list[str], row: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    if not is_new:
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            existing_columns = reader.fieldnames or []
            existing_rows = list(reader)
        if existing_columns != columns:
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow({column: existing_row.get(column, "") for column in columns})

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if is_new:
            writer.writerow(columns)
        writer.writerow(row)


def save_prediction_log(
    *,
    prediction_id: str,
    symbol: str,
    reference_time: str,
    reference_close: float,
    prediction: int,
    direction: str,
    confidence: float | None,
    deployment: str,
    model_uri: str,
) -> None:
    row = [
        datetime.now().isoformat(timespec="seconds"),
        prediction_id,
        symbol,
        reference_time,
        f"{float(reference_close):.4f}",
        prediction,
        direction,
        _format_confidence(confidence),
        deployment,
        model_uri,
    ]
    _append_csv(PREDICTION_LOG_PATH, PREDICTION_COLUMNS, row)
    try:
        append_row(PREDICTION_WORKSHEET, row)
    except Exception as exc:
        print(f"Prediction Google Sheet logging failed: {exc}")


def read_prediction_logs() -> list[dict[str, str]]:
    if not PREDICTION_LOG_PATH.exists():
        return []
    with PREDICTION_LOG_PATH.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))
