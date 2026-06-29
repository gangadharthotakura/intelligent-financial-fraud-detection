"""Analytics utilities for fraud detection trends and merchant insights."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@dataclass(frozen=True)
class AnalyticsConfig:
    time_origin: str = "2020-01-01"
    top_merchants: int = 10
    top_transactions: int = 10
    plot_margin: dict[str, int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "plot_margin", {"l": 40, "r": 40, "t": 60, "b": 40})


class AnalyticsError(Exception):
    """Custom exception for analytics pipeline failures."""


class AnalyticsEngine:
    """Engine for fraud analytics and interactive visualizations."""

    def __init__(self, data_path: Path, output_dir: Optional[Path] = None) -> None:
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.config = AnalyticsConfig()
        self.logger = self._configure_logger()
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = self._load_dataset()
        self.merchant_column = self._detect_merchant_column()

    def _configure_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False
            logger.setLevel(logging.INFO)
        return logger

    def _load_dataset(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise AnalyticsError(f"Data source not found: {self.data_path}")

        self.logger.info("Loading dataset from %s", self.data_path)
        dataset = pd.read_csv(self.data_path, low_memory=False)
        if dataset.empty:
            raise AnalyticsError("Loaded dataset is empty.")

        self.logger.info("Dataset loaded with %d rows and %d columns", *dataset.shape)
        self._prepare_time_features(dataset)
        return dataset

    def _prepare_time_features(self, dataset: pd.DataFrame) -> None:
        if "Time" not in dataset.columns:
            self.logger.warning(
                "Time column missing; hourly and monthly trend analytics are unavailable."
            )
            return

        try:
            dataset["transaction_datetime"] = pd.to_datetime(
                dataset["Time"], unit="s", origin=self.config.time_origin
            )
            dataset["transaction_month"] = (
                dataset["transaction_datetime"].dt.to_period("M").dt.to_timestamp()
            )
            dataset["transaction_hour"] = dataset["transaction_datetime"].dt.hour
            self.logger.info("Derived time-based features for analytics.")
        except Exception as exc:  # pragma: no cover
            self.logger.exception("Failed to derive time features.")
            raise AnalyticsError("Failed to derive time-based features.") from exc

    def _detect_merchant_column(self) -> Optional[str]:
        merchant_candidates = [
            "Merchant",
            "MerchantID",
            "Merchant_ID",
            "merchant",
            "merchant_id",
            "vendor",
            "vendor_id",
        ]
        for candidate in merchant_candidates:
            if candidate in self.dataset.columns:
                self.logger.info("Detected merchant column: %s", candidate)
                return candidate

        self.logger.warning("No merchant identifier column detected.")
        return None

    @property
    def numeric_columns(self) -> list[str]:
        return self.dataset.select_dtypes(include="number").columns.tolist()

    def _validate_columns(self, required_columns: list[str]) -> None:
        missing = [column for column in required_columns if column not in self.dataset.columns]
        if missing:
            raise AnalyticsError(f"Missing columns required for analysis: {missing}")

    def monthly_fraud_trends(self) -> go.Figure:
        self._validate_columns(["transaction_month", "Class"])

        monthly = (
            self.dataset.groupby("transaction_month")["Class"]
            .mean()
            .mul(100)
            .rename("fraud_rate")
            .reset_index()
        )

        fig = px.line(
            monthly,
            x="transaction_month",
            y="fraud_rate",
            markers=True,
            title="Monthly Fraud Rate Trend",
            labels={"transaction_month": "Month", "fraud_rate": "Fraud Rate (%)"},
            template="plotly_white",
        )
        fig.update_layout(margin=self.config.plot_margin)
        return fig

    def hourly_fraud_trends(self) -> go.Figure:
        self._validate_columns(["transaction_hour", "Class"])

        hourly = (
            self.dataset.groupby("transaction_hour")["Class"]
            .mean()
            .mul(100)
            .rename("fraud_rate")
            .reset_index()
        )

        fig = px.bar(
            hourly,
            x="transaction_hour",
            y="fraud_rate",
            title="Hourly Fraud Rate",
            labels={"transaction_hour": "Hour of Day", "fraud_rate": "Fraud Rate (%)"},
            color="fraud_rate",
            color_continuous_scale="OrRd",
            template="plotly_white",
        )
        fig.update_layout(margin=self.config.plot_margin)
        return fig

    def amount_distribution(self) -> go.Figure:
        self._validate_columns(["Amount"])

        fig = px.histogram(
            self.dataset,
            x="Amount",
            nbins=80,
            title="Transaction Amount Distribution",
            labels={"Amount": "Transaction Amount (USD)"},
            marginal="box",
            template="plotly_white",
        )
        fig.update_layout(margin=self.config.plot_margin)
        return fig

    def top_suspicious_transactions(self, top_n: int = 10) -> pd.DataFrame:
        self._validate_columns(["Class", "Amount"])

        suspicious = self.dataset.loc[self.dataset["Class"] == 1, :].copy()
        suspicious["risk_score"] = suspicious["Amount"]
        top_suspicious = suspicious.nlargest(top_n, "risk_score")
        self.logger.info("Selected top %d suspicious transactions.", top_n)
        return top_suspicious

    def merchant_analysis(self) -> go.Figure:
        if self.merchant_column is None:
            raise AnalyticsError("Merchant analysis requires a merchant identifier column.")

        merchant_summary = (
            self.dataset.groupby(self.merchant_column)["Class"]
            .mean()
            .mul(100)
            .rename("fraud_rate")
            .reset_index()
            .nlargest(self.config.top_merchants, "fraud_rate")
        )

        fig = px.bar(
            merchant_summary,
            x=self.merchant_column,
            y="fraud_rate",
            title="Top Merchant Fraud Rates",
            labels={self.merchant_column: "Merchant", "fraud_rate": "Fraud Rate (%)"},
            color="fraud_rate",
            color_continuous_scale="Reds",
            template="plotly_white",
        )
        fig.update_layout(margin=self.config.plot_margin, xaxis_tickangle=-45)
        return fig

    def correlation_heatmap(self) -> go.Figure:
        if len(self.numeric_columns) < 2:
            raise AnalyticsError("Correlation heatmap requires at least two numeric columns.")

        correlation = self.dataset[self.numeric_columns].corr()
        fig = px.imshow(
            correlation,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Viridis",
            title="Numeric Feature Correlation Heatmap",
            labels={"x": "Feature", "y": "Feature", "color": "Correlation"},
            template="plotly_white",
        )
        fig.update_layout(margin=self.config.plot_margin)
        return fig

    def execute(self) -> dict[str, object]:
        analyses = {
            "monthly_trends": self.monthly_fraud_trends(),
            "hourly_trends": self.hourly_fraud_trends(),
            "amount_distribution": self.amount_distribution(),
            "correlation_heatmap": self.correlation_heatmap(),
        }
        try:
            analyses["merchant_analysis"] = self.merchant_analysis()
        except AnalyticsError:
            self.logger.warning("Skipping merchant analysis; no merchant field was detected.")

        analyses["top_suspicious_transactions"] = self.top_suspicious_transactions(
            self.config.top_transactions
        )
        return analyses


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "creditcard.csv"
    engine = AnalyticsEngine(data_path=data_path)
    results = engine.execute()

    for name, result in results.items():
        if isinstance(result, pd.DataFrame):
            print(f"{name}: {result.shape[0]} rows")
        else:
            print(f"Generated plot: {name}")


if __name__ == "__main__":
    main()
