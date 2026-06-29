"""Enterprise-grade fraud detection dashboard for digital banking."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.Analytics import AnalyticsEngine, AnalyticsError
from src.Explainability import ExplainabilityEngine, ExplainabilityError
from src.eda import EDAProcessor, EDAError
from src.predictor import FraudPredictor


LOGGER = logging.getLogger("FraudApp")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


DATA_PATH = Path(__file__).resolve().parents[0] / "data" / "creditcard.csv"
MODEL_PATH = Path(__file__).resolve().parents[0] / "models" / "xgboost.joblib"
SCALER_PATH = Path(__file__).resolve().parents[0] / "data" / "processed" / "scaler.joblib"


def set_theme() -> None:
    st.set_page_config(
        page_title="Intelligent Financial Fraud Detection",
        page_icon="💳",
        layout="wide",
    )
    st.markdown(
        """
        <style>
            body { background-color: #0B132B; color: #E8F1F2; }
            [data-testid='stSidebar'] { background-color: #112D4E; }
            [data-testid='stHeader'] { background-color: transparent; }
            [data-testid='stToolbar'] { background-color: transparent; }
            .stButton>button { background-color: #0B132B; color: #E8F1F2; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_dataset() -> pd.DataFrame:
    processor = EDAProcessor(data_path=DATA_PATH, output_dir=Path("assets") / "eda_reports")
    try:
        return processor.load_dataset()
    except Exception as exc:
        LOGGER.exception("Failed to load dataset: %s", exc)
        raise EDAError("Unable to load dataset.") from exc


def render_header() -> None:
    st.markdown(
        "<div style='background:#09141d;padding:24px;border-radius:16px;margin-bottom:20px;'>"
        "<h1 style='color:#F8F9FA;margin:0;'>Intelligent Financial Fraud Detection Platform</h1>"
        "<p style='color:#A9BCD0;margin:8px 0 0 0;'>"
        "AI-powered risk analytics for digital banking, with explainable fraud detection and reporting."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_dashboard_page(dataset: pd.DataFrame) -> None:
    st.subheader("Executive Dashboard")
    total = len(dataset)
    fraud_count = int(dataset["Class"].sum())
    fraud_rate = fraud_count / total * 100 if total else 0.0
    average_amount = dataset["Amount"].mean() if "Amount" in dataset.columns else 0.0
    risk_score = min(100.0, fraud_rate * 1.4)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total:,}")
    col2.metric("Fraud Transactions", f"{fraud_count:,}")
    col3.metric("Fraud Percentage", f"{fraud_rate:.2f}%")
    col4.metric("Estimated Risk Score", f"{risk_score:.2f}%")

    dataset = dataset.copy()
    if "Time" in dataset.columns:
        dataset["TimeMinutes"] = dataset["Time"] / 60
    dataset["ClassLabel"] = dataset["Class"].map({0: "Legitimate", 1: "Fraud"})

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        pie_data = (
            dataset["Class"]
            .value_counts()
            .rename_axis("ClassLabel")
            .reset_index(name="count")
        )
        pie_data["ClassLabel"] = pie_data["ClassLabel"].map({0: "Legitimate", 1: "Fraud"})
        fig_pie = px.pie(
            pie_data,
            names="ClassLabel",
            values="count",
            title="Transaction Class Mix",
            color_discrete_sequence=["#2E86AB", "#E74C3C"],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        if "Amount" in dataset.columns:
            fig_hist = px.histogram(
                dataset,
                x="Amount",
                nbins=80,
                title="Transaction Amount Distribution",
                labels={"Amount": "Transaction Amount (USD)"},
                template="plotly_dark",
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Amount distribution unavailable: missing Amount column.")

    if "TimeMinutes" in dataset.columns:
        fig_time = px.line(
            dataset.sort_values("TimeMinutes"),
            x="TimeMinutes",
            y="Amount",
            color="ClassLabel",
            title="Transaction Amount Trend Over Time",
            labels={"TimeMinutes": "Time (minutes)", "Amount": "Amount"},
            template="plotly_dark",
        )
        st.plotly_chart(fig_time, use_container_width=True)

    numeric_dataset = dataset.select_dtypes(include=[np.number]).copy()
    if numeric_dataset.shape[1] > 1:
        corr = numeric_dataset.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            title="Correlation Heatmap",
            template="plotly_dark",
        )
        st.plotly_chart(fig_corr, width="stretch")
    else:
        st.info("Correlation heatmap is unavailable because the dataset has insufficient numeric fields.")

    st.markdown("### Summary Insights")
    st.write(
        "- Fraud represents a very low share of transactions but a high risk segment for digital banking operations."
        "<br>- Monitoring hourly and amount spikes is essential for early fraud intervention.",
        unsafe_allow_html=True,
    )


def render_prediction_page() -> None:
    st.subheader("Predictive Fraud Scoring")
    st.markdown(
        "Upload transaction features in CSV format, preview the dataset, and generate fraud probability scores."
    )

    uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV file to launch predictions.")
        return

    try:
        data = pd.read_csv(uploaded_file)
    except Exception as exc:
        LOGGER.exception("Prediction upload failed: %s", exc)
        st.error("Unable to read the uploaded CSV file.")
        return

    st.markdown(f"**Dataset preview:** {data.shape[0]} rows, {data.shape[1]} columns")
    st.dataframe(data.head(10), use_container_width=True)

    if st.button("Run Fraud Predictions"):
        predictor = FraudPredictor(model_path=MODEL_PATH)
        try:
            features = data.drop(columns=["Class"], errors="ignore")
            prediction_df = predictor.predict(features)
        except Exception as exc:
            LOGGER.exception("Prediction execution failed: %s", exc)
            st.error("Prediction failed. Verify the uploaded data schema and try again.")
            return

        result = pd.concat([data.reset_index(drop=True), prediction_df], axis=1)
        st.success("Fraud scoring complete.")
        st.dataframe(result.head(20), use_container_width=True)

        csv_bytes = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download annotated results",
            data=csv_bytes,
            file_name="fraud_predictions.csv",
            mime="text/csv",
        )


def render_analytics_page(dataset: pd.DataFrame) -> None:
    st.subheader("Risk Analytics")
    st.markdown("Interactive trend analysis and suspicious transaction insights.")

    try:
        engine = AnalyticsEngine(data_path=DATA_PATH)
        plots = engine.execute()
    except AnalyticsError as exc:
        LOGGER.exception("Analytics engine failed: %s", exc)
        st.error("Analytics could not be generated. Verify the dataset.")
        return

    columns = st.columns(2)
    columns[0].plotly_chart(plots["monthly_trends"], use_container_width=True)
    columns[1].plotly_chart(plots["hourly_trends"], use_container_width=True)
    st.plotly_chart(plots["amount_distribution"], use_container_width=True)
    st.plotly_chart(plots["correlation_heatmap"], use_container_width=True)

    if "merchant_analysis" in plots:
        st.plotly_chart(plots["merchant_analysis"], use_container_width=True)

    suspicious = plots["top_suspicious_transactions"]
    st.markdown("### Top Suspicious Transactions")
    st.dataframe(suspicious.head(10), use_container_width=True)


def render_explainability_page() -> None:
    st.subheader("Explainable AI")
    st.markdown(
        "Generate SHAP-based explainability reports and understand why each transaction was flagged."
    )

    sample_size = st.slider("Number of rows for SHAP explainability", 10, 200, 50, 10)
    run_analysis = st.button("Run Explainability")

    if not run_analysis:
        st.info("Click the button to compute explainability reports for a dataset sample.")
        return

    try:
        full_data = load_dataset()
        features = full_data.drop(columns=["Class"], errors="ignore").head(sample_size)
        engine = ExplainabilityEngine(model_path=MODEL_PATH, scaler_path=SCALER_PATH)
        engine.explain_dataset(features)
    except (ExplainabilityError, EDAError, FileNotFoundError) as exc:
        LOGGER.exception("Explainability pipeline failed: %s", exc)
        st.error("Unable to run explainability analysis. Verify model and data.")
        return

    transaction_index = st.number_input(
        "Select transaction index", min_value=0, max_value=len(features) - 1, value=0
    )

    try:
        explanation = engine.explain_transaction(transaction_index, features)
    except ExplainabilityError as exc:
        LOGGER.exception("Explain transaction failed: %s", exc)
        st.error(str(exc))
        return

    st.markdown("### Transaction-level Explanation")
    st.write(f"**Fraud probability:** {explanation.fraud_probability:.2%}")
    st.write(f"**Predicted class:** {'Fraud' if explanation.predicted_class == 1 else 'Legitimate'}")
    st.dataframe(explanation.explanation_frame, use_container_width=True)
    st.markdown("#### SHAP Reports")
    st.markdown(
        f"- [Waterfall plot]({explanation.waterfall_path})\n"
        f"- [Force plot]({explanation.force_path})\n"
        f"- [Decision plot]({explanation.decision_path})"
    )


def render_about_page() -> None:
    st.subheader("About This Application")
    st.markdown(
        "This banking-grade fraud detection platform combines XGBoost modeling, "
        "SHAP explainability, and interactive risk analytics for enterprise digital banking."
    )
    st.markdown(
        "**Dataset:** Kaggle credit card fraud detection dataset (creditcard.csv). "
        "**Architecture:** Modular Python backend with Streamlit dashboard and Plotly visuals."
    )


def main() -> None:
    set_theme()
    render_header()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["Dashboard", "Predict Fraud", "Analytics", "Explainability", "About"],
    )

    dataset = None
    if page in {"Dashboard", "Analytics"}:
        try:
            dataset = load_dataset()
        except EDAError as exc:
            LOGGER.exception("Dataset load failure: %s", exc)
            st.error("Unable to load dataset for this page. Ensure data is available.")

    if page == "Dashboard" and dataset is not None:
        render_dashboard_page(dataset)
    elif page == "Predict Fraud":
        render_prediction_page()
    elif page == "Analytics" and dataset is not None:
        render_analytics_page(dataset)
    elif page == "Explainability":
        render_explainability_page()
    elif page == "About":
        render_about_page()
    else:
        st.warning("No dataset available for the selected page.")


if __name__ == "__main__":
    main()
