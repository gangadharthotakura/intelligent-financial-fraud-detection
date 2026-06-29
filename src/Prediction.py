"""Streamlit prediction dashboard for fraud detection."""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd
import plotly.express as px
import streamlit as st

from predictor import FraudPredictor


@dataclass(frozen=True)
class PredictionAppConfig:
    model_filename: str = "xgboost.joblib"
    accepted_file_types: tuple[str, ...] = ("csv",)
    max_preview_rows: int = 10
    download_filename: str = "fraud_predictions.csv"
    max_display_rows: int = 20


class PredictionAppError(Exception):
    """Application-level exception for prediction dashboard failures."""


LOGGER = logging.getLogger("PredictionApp")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


@st.cache_resource
def load_predictor(model_path: Path) -> FraudPredictor:
    if not model_path.exists():
        raise PredictionAppError(f"Model artifact not found at {model_path}")
    return FraudPredictor(model_path=model_path)


def load_uploaded_csv(file: BinaryIO) -> pd.DataFrame:
    try:
        df = pd.read_csv(file)
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Unable to parse uploaded CSV file: %s", exc)
        raise PredictionAppError("Uploaded file is not a valid CSV document.") from exc

    if df.empty:
        raise PredictionAppError("Uploaded file is empty.")

    return df


def validate_input_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "Class" in dataframe.columns:
        dataframe = dataframe.drop(columns=["Class"])

    if dataframe.isna().any().any():
        raise PredictionAppError(
            "Input contains missing values. Remove or impute missing values before prediction."
        )

    return dataframe


def format_prediction_summary(predictions: pd.DataFrame) -> str:
    total = len(predictions)
    fraud_count = int(predictions["prediction"].sum())
    fraud_rate = fraud_count / total * 100 if total else 0.0
    average_probability = predictions["fraud_probability"].mean() * 100
    return (
        f"**Predicted fraud cases:** {fraud_count}/{total} "
        f"({fraud_rate:.2f}% of uploaded records)  •  "
        f"**Avg. fraud probability:** {average_probability:.2f}%"
    )


def build_risk_pie_chart(predictions: pd.DataFrame) -> px.Figure:
    risk_counts = (
        predictions["risk_level"]
        .value_counts()
        .rename_axis("Risk Level")
        .reset_index(name="Count")
    )
    return px.pie(
        risk_counts,
        names="Risk Level",
        values="Count",
        title="Predicted Risk Level Distribution",
        color_discrete_sequence=["#2E86AB", "#F4D35E", "#E74C3C"],
    )


def build_probability_bar_chart(predictions: pd.DataFrame) -> px.Figure:
    bucketed = (
        predictions["fraud_probability"]
        .mul(100)
        .round(0)
        .astype(int)
        .value_counts()
        .sort_index()
        .reset_index(name="Count")
    )
    bucketed["Probability Range"] = bucketed["index"].astype(str) + "%"
    return px.bar(
        bucketed,
        x="Probability Range",
        y="Count",
        title="Fraud Probability Distribution",
        labels={"Count": "Number of Records", "Probability Range": "Probability"},
        template="plotly_white",
    )


def render_dashboard() -> None:
    config = PredictionAppConfig()
    st.set_page_config(
        page_title="Fraud Prediction Portal",
        page_icon="🛡️",
        layout="wide",
    )

    st.markdown(
        "<div style='background:#0D1B2A;padding:24px;border-radius:16px;'>"
        "<h1 style='color:#F8F9FA;margin:0;'>Fraud Prediction Workspace</h1>"
        "<p style='color:#A9BCD0;margin:8px 0 0 0;'>"
        "Upload an anonymized transaction CSV to score fraud risk, probability, and confidence levels."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    base_dir = Path(__file__).resolve().parents[1]
    model_path = base_dir / "models" / config.model_filename

    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader(
            "Upload transaction CSV file",
            type=list(config.accepted_file_types),
            help="Upload a transaction file with the same feature schema used for model training.",
        )
        st.divider()
        st.markdown("**Model artifact**")
        st.write(config.model_filename)
        st.button("Run Predictions", key="run_predictions")

    if uploaded_file is None:
        st.info("Upload a valid CSV file in the sidebar to begin fraud predictions.")
        return

    try:
        dataset = load_uploaded_csv(uploaded_file)
    except PredictionAppError as exc:
        st.error(str(exc))
        return

    st.markdown("### Dataset Preview")
    st.markdown(
        f"Records: **{dataset.shape[0]}** · Features: **{dataset.shape[1]}**"
    )
    st.dataframe(dataset.head(config.max_preview_rows), use_container_width=True)

    try:
        predictor = load_predictor(model_path)
    except PredictionAppError as exc:
        st.error(str(exc))
        LOGGER.exception("Predictor load failed: %s", exc)
        return

    try:
        feature_data = validate_input_dataframe(dataset)
        predictions = predictor.predict(feature_data)
    except PredictionAppError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # pragma: no cover
        st.error("Prediction failed. Check the uploaded dataset schema and retry.")
        LOGGER.exception("Prediction execution failed: %s", exc)
        return

    predictions = predictions.reset_index(drop=True)
    annotated = pd.concat([dataset.reset_index(drop=True), predictions], axis=1)

    st.markdown("### Prediction Summary")
    st.success(format_prediction_summary(predictions))

    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.metric("Detected Fraud Records", int(predictions["prediction"].sum()))
        st.metric("High Risk Records", int((predictions["risk_level"] == "High").sum()))
    with summary_col2:
        st.metric(
            "Average Fraud Probability",
            f"{predictions['fraud_probability'].mean() * 100:.2f}%",
        )
        st.metric(
            "Confidence Score",
            f"{predictions['confidence_score'].mean():.2f}%",
        )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(build_risk_pie_chart(predictions), use_container_width=True)
    with chart_col2:
        st.plotly_chart(build_probability_bar_chart(predictions), use_container_width=True)

    st.markdown("### Prediction Table")
    st.dataframe(annotated.head(config.max_display_rows), use_container_width=True)

    csv_bytes = annotated.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Annotated Predictions",
        data=csv_bytes,
        file_name=config.download_filename,
        mime="text/csv",
    )

    st.markdown(
        "<div style='background:#112D4E;padding:16px;border-radius:12px;margin-top:16px;'>"
        "<p style='color:#F8F9FA;margin:0;font-size:14px;'>"
        "Prediction probabilities show the model's confidence in fraud detection. "
        "Review high-risk cases first and update data quality prior to batch ingestion."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    render_dashboard()
