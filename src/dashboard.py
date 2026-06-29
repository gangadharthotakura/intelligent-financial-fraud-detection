"""Streamlit dashboard for the fraud detection and risk analytics platform."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


@dataclass
class DashboardMetrics:
    total_transactions: int
    fraud_transactions: int
    fraud_percentage: float
    alerts_count: int
    average_risk_score: float
    risk_level: str


class FraudDashboard:
    """Interactive Streamlit dashboard for fraud analytics."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.data = self._load_data()
        self.metrics = self._compute_metrics()

    def _load_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.data_path)
        return data

    def _compute_metrics(self) -> DashboardMetrics:
        total_transactions = len(self.data)
        fraud_transactions = int(self.data["Class"].sum())
        fraud_percentage = round(fraud_transactions / total_transactions * 100, 4)
        alerts_count = int(min(fraud_transactions, 15))
        average_risk_score = round(fraud_percentage * 0.95, 2)
        risk_level = self._render_risk_level(average_risk_score)

        return DashboardMetrics(
            total_transactions=total_transactions,
            fraud_transactions=fraud_transactions,
            fraud_percentage=fraud_percentage,
            alerts_count=alerts_count,
            average_risk_score=average_risk_score,
            risk_level=risk_level,
        )

    def _render_risk_level(self, score: float) -> str:
        if score >= 75:
            return "High"
        if score >= 45:
            return "Medium"
        return "Low"

    def _build_pie_chart(self) -> go.Figure:
        pie_data = (
            self.data["Class"].value_counts()
            .rename_axis("ClassLabel")
            .reset_index(name="count")
        )
        pie_data["ClassLabel"] = pie_data["ClassLabel"].map({0: "Legitimate", 1: "Fraud"})
        fig = px.pie(
            pie_data,
            names="ClassLabel",
            values="count",
            title="Transaction Breakdown: Legitimate vs Fraud",
            color_discrete_sequence=["#2E86AB", "#E63946"],
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), template="plotly_dark")
        return fig

    def _build_time_series(self) -> go.Figure:
        sample = self.data.sample(n=min(5000, len(self.data)), random_state=42).copy()
        sample["TimeMinutes"] = sample["Time"] / 60
        sample["ClassLabel"] = sample["Class"].map({0: "Legitimate", 1: "Fraud"})
        fig = px.line(
            sample.sort_values("TimeMinutes"),
            x="TimeMinutes",
            y="Amount",
            color="ClassLabel",
            title="Transaction Amount Trend over Time",
            labels={"TimeMinutes": "Time (minutes)", "Amount": "Transaction Amount (USD)"},
            color_discrete_map={"Legitimate": "#2E86AB", "Fraud": "#E63946"},
            template="plotly_dark",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        return fig

    def _build_bar_chart(self) -> go.Figure:
        top_features = self.data.drop(columns=["Class"]).abs().mean().sort_values(ascending=False).head(8)
        fig = px.bar(
            top_features,
            x=top_features.index,
            y=top_features.values,
            title="Top Feature Magnitudes Across Transactions",
            labels={"x": "Feature", "y": "Average Magnitude"},
            color=top_features.values,
            color_continuous_scale="blues",
            template="plotly_dark",
        )
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
        return fig

    def _build_heatmap(self) -> go.Figure:
        correlation = self.data.corr()
        fig = px.imshow(
            correlation,
            title="Feature Correlation Heatmap",
            color_continuous_scale="Viridis",
            labels={"x": "Feature", "y": "Feature", "color": "Correlation"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), template="plotly_dark")
        return fig

    def run(self) -> None:
        st.set_page_config(
            page_title="Intelligent Financial Fraud Detection",
            layout="wide",
            page_icon="💳",
        )
        self._render_header()
        self._render_summary_cards()
        self._render_charts()
        self._render_alert_panel()
        self._render_data_overview()

    def _render_header(self) -> None:
        st.markdown(
            "<div style='background:#0D1B2A; padding: 20px; border-radius: 10px;'>"
            "<h1 style='color:#F8F9FA; margin-bottom: 5px;'>Intelligent Financial Fraud Detection</h1>"
            "<p style='color:#A9BCD0; font-size:18px;'>Real-time risk analytics dashboard for digital banking.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    def _render_summary_cards(self) -> None:
        st.markdown("### Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{self.metrics.total_transactions:,}")
        col2.metric("Fraud Transactions", f"{self.metrics.fraud_transactions:,}")
        col3.metric("Fraud Percentage", f"{self.metrics.fraud_percentage:.2f}%")
        col4.metric("Average Risk Score", f"{self.metrics.average_risk_score:.2f}%")

        st.markdown(
            f"<div style='background:#112D4E; border-radius: 10px; padding: 16px; margin-top: 12px;'>"
            f"<strong style='color:#F8F9FA;'>Risk Level:</strong> "
            f"<span style='color:{'#E63946' if self.metrics.risk_level=='High' else '#F4D35E' if self.metrics.risk_level=='Medium' else '#2EC4B6'};'>{self.metrics.risk_level}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    def _render_charts(self) -> None:
        st.markdown("### Interactive Transaction Analytics")
        pie_col, line_col = st.columns((1, 2))
        with pie_col:
            st.plotly_chart(self._build_pie_chart(), use_container_width=True)
        with line_col:
            st.plotly_chart(self._build_time_series(), use_container_width=True)

        bar_col, heatmap_col = st.columns(2)
        with bar_col:
            st.plotly_chart(self._build_bar_chart(), use_container_width=True)
        with heatmap_col:
            st.plotly_chart(self._build_heatmap(), use_container_width=True)

    def _render_alert_panel(self) -> None:
        st.markdown("### Today's Alerts")
        fraud_samples = self.data[self.data["Class"] == 1].head(self.metrics.alerts_count)
        if fraud_samples.empty:
            st.info("No high-priority alerts detected for today.")
            return

        st.dataframe(
            fraud_samples.assign(**{"Risk Level": "High"}).iloc[:, :6],
            use_container_width=True,
        )

    def _render_data_overview(self) -> None:
        with st.expander("View raw transaction snapshot"):
            st.dataframe(self.data.sample(n=min(10, len(self.data))), use_container_width=True)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "creditcard.csv"
    dashboard = FraudDashboard(data_path=data_path)
    dashboard.run()


if __name__ == "__main__":
    main()
