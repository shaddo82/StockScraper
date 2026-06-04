"""Streamlit dashboard for prediction and feedback monitoring."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv


PREDICTION_LOG_PATH = Path("logs/predictions.csv")
FEEDBACK_LOG_PATH = Path("logs/feedback.csv")
VERIFICATION_LOG_PATH = Path("logs/verifications.csv")

load_dotenv()


def _load_local_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_sheet(worksheet_name: str) -> pd.DataFrame:
    sheet_name = os.getenv("GOOGLE_SHEET_NAME")
    service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sheet_name or not service_account_json:
        return pd.DataFrame()

    import gspread
    from google.oauth2.service_account import Credentials

    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    info = json.loads(service_account_json)
    credentials = Credentials.from_service_account_info(info, scopes=scope)
    worksheet = gspread.authorize(credentials).open(sheet_name).worksheet(worksheet_name)
    return pd.DataFrame(worksheet.get_all_records())


def _load_sheet_or_empty(worksheet_name: str) -> pd.DataFrame:
    try:
        return _load_sheet(worksheet_name)
    except Exception as exc:
        st.warning(f"{worksheet_name} 시트를 읽지 못했습니다: {exc}")
        return pd.DataFrame()


def load_logs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if os.getenv("GOOGLE_SHEET_NAME") and os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"):
        prediction_df = _load_sheet_or_empty("prediction_logs")
        feedback_df = _load_sheet_or_empty("feedback_logs")
        verification_df = _load_sheet_or_empty("verification_logs")
        if not prediction_df.empty or not feedback_df.empty or not verification_df.empty:
            return prediction_df, feedback_df, verification_df, "Google Sheets"

    return (
        _load_local_csv(PREDICTION_LOG_PATH),
        _load_local_csv(FEEDBACK_LOG_PATH),
        _load_local_csv(VERIFICATION_LOG_PATH),
        "local CSV",
    )


def render_prediction_metrics(prediction_df: pd.DataFrame) -> None:
    st.subheader("Prediction Monitoring")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Requests", len(prediction_df))

    if prediction_df.empty:
        col2.metric("Average Confidence", "-")
        col3.metric("Low Confidence", 0)
        col4.metric("Canary Requests", 0)
        st.info("아직 예측 로그가 없습니다.")
        return

    prediction_df["confidence"] = pd.to_numeric(
        prediction_df.get("confidence"),
        errors="coerce",
    )
    col2.metric("Average Confidence", f"{prediction_df['confidence'].mean():.4f}")
    col3.metric("Low Confidence", int((prediction_df["confidence"] < 0.65).sum()))
    col4.metric(
        "Canary Requests",
        int((prediction_df.get("deployment") == "challenger").sum()),
    )

    st.subheader("Confidence Trend")
    st.line_chart(prediction_df["confidence"].reset_index(drop=True))

    st.subheader("Serving Model Count")
    if "deployment" in prediction_df.columns:
        st.bar_chart(prediction_df["deployment"].value_counts())
        deployment_summary = (
            prediction_df.groupby("deployment")["confidence"]
            .agg(prediction_count="count", avg_confidence="mean")
            .reset_index()
        )
        st.dataframe(deployment_summary, use_container_width=True)

    st.subheader("Recent Predictions")
    st.dataframe(prediction_df.tail(20), use_container_width=True)


def render_feedback_metrics(feedback_df: pd.DataFrame) -> None:
    st.subheader("User Feedback")
    col1, col2, col3 = st.columns(3)
    col1.metric("Feedback Count", len(feedback_df))

    if feedback_df.empty:
        col2.metric("Wrong Prediction Feedback", 0)
        col3.metric("Wrong Feedback Rate", "0.00%")
        st.info("아직 사용자 피드백이 없습니다.")
        return

    wrong_df = feedback_df[
        feedback_df["prediction"].astype(str) != feedback_df["correct_label"].astype(str)
    ]
    wrong_rate = len(wrong_df) / len(feedback_df)
    col2.metric("Wrong Prediction Feedback", len(wrong_df))
    col3.metric("Wrong Feedback Rate", f"{wrong_rate:.2%}")

    st.subheader("Feedback Label Distribution")
    st.bar_chart(feedback_df["correct_label"].astype(str).value_counts())

    st.subheader("Recent Feedback")
    st.dataframe(feedback_df.tail(20), use_container_width=True)

    if not wrong_df.empty:
        st.subheader("Wrong Prediction Cases")
        st.dataframe(wrong_df.tail(20), use_container_width=True)


def render_verification_metrics(verification_df: pd.DataFrame) -> None:
    st.subheader("Actual Outcome Verification")
    col1, col2 = st.columns(2)
    col1.metric("Verification Count", len(verification_df))

    if verification_df.empty:
        col2.metric("Verified Accuracy", "-")
        st.info("아직 자동 검증 로그가 없습니다.")
        return

    verification_df["correct"] = pd.to_numeric(
        verification_df.get("correct"),
        errors="coerce",
    )
    col2.metric("Verified Accuracy", f"{verification_df['correct'].mean():.2%}")

    if "deployment" in verification_df.columns:
        st.subheader("Verified Accuracy by Deployment")
        deployment_summary = (
            verification_df.groupby("deployment")["correct"]
            .agg(verification_count="count", verified_accuracy="mean")
            .reset_index()
        )
        st.dataframe(deployment_summary, use_container_width=True)

    st.subheader("Recent Verifications")
    st.dataframe(verification_df.tail(20), use_container_width=True)


st.set_page_config(page_title="Stock MLOps Monitoring", layout="wide")
st.title("Stock MLOps Monitoring")

prediction_logs, feedback_logs, verification_logs, source = load_logs()
st.caption(f"Log source: {source}")

render_prediction_metrics(prediction_logs)
render_feedback_metrics(feedback_logs)
render_verification_metrics(verification_logs)
