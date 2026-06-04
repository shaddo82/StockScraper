"""User feedback logging helpers for model operation monitoring."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from app.google_sheet_logger import FEEDBACK_WORKSHEET, append_row


FEEDBACK_LOG_PATH = Path("logs/feedback.csv")
FEEDBACK_COLUMNS = [
    "time",
    "prediction_id",
    "symbol",
    "prediction",
    "correct_label",
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
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if is_new:
            writer.writerow(columns)
        writer.writerow(row)


def save_feedback(
    *,
    prediction_id: str,
    symbol: str,
    prediction: int,
    correct_label: int,
    confidence: float | None,
    deployment: str,
    model_uri: str,
) -> None:
    row = [
        datetime.now().isoformat(timespec="seconds"),
        prediction_id,
        symbol,
        prediction,
        correct_label,
        _format_confidence(confidence),
        deployment,
        model_uri,
    ]
    _append_csv(FEEDBACK_LOG_PATH, FEEDBACK_COLUMNS, row)
    try:
        append_row(FEEDBACK_WORKSHEET, row)
    except Exception as exc:
        print(f"Feedback Google Sheet logging failed: {exc}")


def read_feedback_logs() -> list[dict[str, str]]:
    if not FEEDBACK_LOG_PATH.exists():
        return []
    with FEEDBACK_LOG_PATH.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))
