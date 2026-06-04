"""Actual outcome verification helpers for prediction monitoring."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from app.google_sheet_logger import VERIFICATION_WORKSHEET, append_row


VERIFICATION_LOG_PATH = Path("logs/verifications.csv")
VERIFICATION_COLUMNS = [
    "time",
    "prediction_id",
    "symbol",
    "prediction",
    "actual_label",
    "correct",
    "reference_time",
    "reference_close",
    "actual_time",
    "actual_close",
    "deployment",
    "model_uri",
]


def _append_csv(path: Path, columns: list[str], row: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if is_new:
            writer.writerow(columns)
        writer.writerow(row)


def save_verification_log(
    *,
    prediction_id: str,
    symbol: str,
    prediction: int,
    actual_label: int,
    correct: bool,
    reference_time: str,
    reference_close: float,
    actual_time: str,
    actual_close: float,
    deployment: str,
    model_uri: str,
) -> None:
    row = [
        datetime.now().isoformat(timespec="seconds"),
        prediction_id,
        symbol,
        prediction,
        actual_label,
        int(correct),
        reference_time,
        f"{float(reference_close):.4f}",
        actual_time,
        f"{float(actual_close):.4f}",
        deployment,
        model_uri,
    ]
    _append_csv(VERIFICATION_LOG_PATH, VERIFICATION_COLUMNS, row)
    try:
        append_row(VERIFICATION_WORKSHEET, row)
    except Exception as exc:
        print(f"Verification Google Sheet logging failed: {exc}")


def read_verification_logs() -> list[dict[str, str]]:
    if not VERIFICATION_LOG_PATH.exists():
        return []
    with VERIFICATION_LOG_PATH.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))
