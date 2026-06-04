"""Optional Google Sheets logging for prediction and feedback records."""
from __future__ import annotations

import json
import os
from typing import Any


PREDICTION_WORKSHEET = "prediction_logs"
FEEDBACK_WORKSHEET = "feedback_logs"
VERIFICATION_WORKSHEET = "verification_logs"

_spreadsheet = None


def is_google_sheet_enabled() -> bool:
    return bool(os.getenv("GOOGLE_SHEET_NAME") and os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))


def get_spreadsheet():
    global _spreadsheet

    if _spreadsheet is not None:
        return _spreadsheet

    sheet_name = os.getenv("GOOGLE_SHEET_NAME")
    service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sheet_name or not service_account_json:
        raise RuntimeError("Google Sheets logging is not configured")

    import gspread
    from google.oauth2.service_account import Credentials

    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    info = json.loads(service_account_json)
    credentials = Credentials.from_service_account_info(info, scopes=scope)
    _spreadsheet = gspread.authorize(credentials).open(sheet_name)
    return _spreadsheet


def append_row(worksheet_name: str, row: list[Any]) -> None:
    if not is_google_sheet_enabled():
        return
    spreadsheet = get_spreadsheet()
    spreadsheet.worksheet(worksheet_name).append_row(row)
