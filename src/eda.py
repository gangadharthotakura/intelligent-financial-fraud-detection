"""Exploratory Data Analysis module for credit card fraud detection dataset.

This module provides comprehensive EDA capabilities including dataset validation,
statistical summaries, and interactive visualizations for fraud analysis.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Constants
_SAMPLE_SEED: int = 42
_MAX_SCATTER_POINTS: int = 20_000
_DEFAULT_FEATURES: list[str] = ["Time", "Amount", "V1", "V2", "V3"]
_CLASS_LABELS: dict[int, str] = {0: "Legitimate", 1: "Fraud"}
_CLASS_COLORS: dict[int, str] = {0: "#2E86AB", 1: "#E74C3C"}
_PLOTLY_CONFIG: dict = {"mode": "lines+markers", "templates": "plotly_white"}


class EDAError(Exception):
    """Custom exception for EDA processing failures."""

    pass


class EDAProcessor:
    """Exploratory Data Analysis processor for fraud detection datasets.

    Provides dataset validation, statistical reporting, and interactive
    visualizations for fraud detection data exploration.

    Attributes:
        data_path: Path to the input CSV dataset.
        output_dir: Directory for persisting generated visualizations.
        logger: Logger instance for operation tracking.
        dataset: Cached dataset for efficient multi-operation workflows.
    """

    def __init__(self, data_path: Path | str, output_dir: Path | str) -> None:
        """Initialize the EDA processor with data and output paths.

        Args:
            data_path: Path to the source CSV file.
            output_dir: Directory for saving generated reports.

        Raises:
            EDAError: If output directory creation fails.
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise EDAError(f"Failed to create output directory {self.output_dir}: {exc}") from exc

        self.logger = self._configure_logger()
        self.dataset: Optional[pd.DataFrame] = None

    @staticmethod
    def _configure_logger() -> logging.Logger:
        """Configure and return a class-specific logger.

        Returns:
            Configured logger instance.
        """
        logger = logging.getLogger(EDAProcessor.__name__)
        if logger.handlers:
            return logger

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        logger.setLevel(logging.INFO)
        return logger

    def _ensure_dataset_loaded(self) -> pd.DataFrame:
        """Validate that dataset is loaded in memory.

        Returns:
            The cached dataset.

        Raises:
            EDAError: If dataset has not been loaded yet.
        """
        if self.dataset is None:
            raise EDAError(
                "Dataset not loaded. Call load_dataset() before running this method."
            )
        return self.dataset

    def _validate_required_columns(self, columns: list[str]) -> None:
        """Verify required columns exist in the dataset.

        Args:
            columns: List of required column names.

        Raises:
            EDAError: If any required columns are missing.
        """
        dataset = self._ensure_dataset_loaded()
        missing_columns = [col for col in columns if col not in dataset.columns]
        if missing_columns:
            raise EDAError(f"Missing required dataset columns: {missing_columns}")

    def load_dataset(self) -> pd.DataFrame:
        """Load and validate the dataset from CSV source.

        Returns:
            Loaded dataset as DataFrame.

        Raises:
            EDAError: If file not found or read operation fails.
        """
        if not self.data_path.exists():
            raise EDAError(f"Dataset not found at {self.data_path}")

        try:
            self.logger.info("Loading dataset from %s", self.data_path)
            self.dataset = pd.read_csv(self.data_path, low_memory=False)
            self.logger.info(
                "Dataset loaded: %d rows × %d columns", self.dataset.shape[0], self.dataset.shape[1]
            )
        except (OSError, pd.errors.ParserError) as exc:
            raise EDAError(f"Failed to load dataset from {self.data_path}: {exc}") from exc

        return self.dataset

    def display_dataset_info(self) -> None:
        """Log dataset schema, dtypes, and memory usage statistics.

        Raises:
            EDAError: If dataset is not loaded.
        """
        dataset = self._ensure_dataset_loaded()

        buffer = io.StringIO()
        dataset.info(buf=buffer)
        buffer.seek(0)
        self.logger.info("Dataset schema and memory usage:\n%s", buffer.read())

    def report_missing_values(self) -> pd.DataFrame:
        """Generate comprehensive missing value analysis report.

        Returns:
            DataFrame with missing count and percentage for each column.

        Raises:
            EDAError: If dataset is not loaded.
        """
        dataset = self._ensure_dataset_loaded()

        missing_count = dataset.isna().sum()
        missing_report = pd.DataFrame(
            {
                "missing_count": missing_count,
                "missing_ratio_percent": (missing_count / len(dataset) * 100).round(2),
            }
        )

        if missing_count.any():
            self.logger.warning(
                "Missing values detected:\n%s", missing_report[missing_report["missing_count"] > 0]
            )
        else:
            self.logger.info("No missing values detected in dataset")

        return missing_report

    def report_duplicate_records(self) -> pd.DataFrame:
        """Identify and report duplicate rows in the dataset.

        Returns:
            DataFrame containing all duplicate rows (excluding first occurrence).

        Raises:
            EDAError: If dataset is not loaded.
        """
        dataset = self._ensure_dataset_loaded()

        duplicates = dataset[dataset.duplicated(keep="first")]
        duplicate_count = len(duplicates)

        if duplicate_count > 0:
            self.logger.warning("Detected %d duplicate records (%0.2f%%)",
                              duplicate_count, duplicate_count / len(dataset) * 100)
        else:
            self.logger.info("No duplicate records found")

        return duplicates

    def report_fraud_percentage(self) -> pd.DataFrame:
        """Analyze class distribution and fraud prevalence statistics.

        Returns:
            DataFrame with fraud class counts and percentages.

        Raises:
            EDAError: If dataset not loaded or required columns missing.
        """
        dataset = self._ensure_dataset_loaded()
        self._validate_required_columns(["Class"])

        class_counts = dataset["Class"].value_counts(dropna=False).sort_index()
        class_percentages = (class_counts / len(dataset) * 100).round(2)

        report = pd.DataFrame(
            {
                "label": [_CLASS_LABELS.get(idx, f"Unknown_{idx}") for idx in class_counts.index],
                "count": class_counts.values,
                "percentage": class_percentages.values,
            },
            index=class_counts.index,
        )

        self.logger.info("Fraud class distribution:\n%s", report.to_string())
        return report

    def save_plot(self, fig: go.Figure, filename: str) -> Path:
        """Persist a Plotly figure to disk as interactive HTML.

        Args:
            fig: Plotly Figure object to save.
            filename: Output filename for the report.

        Returns:
            Path to the saved file.

        Raises:
            EDAError: If file write operation fails.
        """
        destination = self.output_dir / filename

        try:
            fig.write_html(str(destination), include_plotlyjs="cdn")
            self.logger.info("Saved plot to %s", destination)
        except OSError as exc:
            self.logger.exception("Failed to save plot %s: %s", filename, exc)
            raise EDAError(f"Unable to save plot '{filename}': {exc}") from exc

        return destination

    def plot_correlation_matrix(self) -> Path:
        """Generate and save feature correlation heatmap visualization.

        Returns:
            Path to the saved correlation matrix plot.

        Raises:
            EDAError: If dataset not loaded or visualization fails.
        """
        dataset = self._ensure_dataset_loaded()

        try:
            correlation = dataset.corr(numeric_only=True)
        except Exception as exc:
            raise EDAError(f"Failed to compute correlation matrix: {exc}") from exc

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Matrix",
            labels={"x": "Features", "y": "Features", "color": "Pearson r"},
            zmin=-1,
            zmax=1,
        )
        fig.update_layout(margin=dict(l=80, r=40, t=80, b=80), font=dict(size=10))
        return self.save_plot(fig, "correlation_matrix.html")

    def plot_transaction_amount_distribution(self) -> Path:
        """Generate transaction amount histogram with statistical overlay.

        Returns:
            Path to the saved distribution plot.

        Raises:
            EDAError: If dataset not loaded or required columns missing.
        """
        dataset = self._ensure_dataset_loaded()
        self._validate_required_columns(["Amount"])

        fig = px.histogram(
            dataset,
            x="Amount",
            nbins=100,
            title="Transaction Amount Distribution",
            labels={"Amount": "Transaction Amount (USD)"},
            marginal="box",
            template="plotly_white",
            opacity=0.85,
        )
        fig.update_layout(margin=dict(l=60, r=40, t=80, b=60), font=dict(size=11))
        fig.update_xaxes(title_text="Transaction Amount (USD)")
        fig.update_yaxes(title_text="Frequency")

        return self.save_plot(fig, "transaction_amount_distribution.html")

    def plot_class_imbalance(self) -> Path:
        """Visualize fraud class imbalance as bar chart with counts.

        Returns:
            Path to the saved class imbalance plot.

        Raises:
            EDAError: If dataset not loaded or required columns missing.
        """
        dataset = self._ensure_dataset_loaded()
        self._validate_required_columns(["Class"])

        class_counts = dataset["Class"].value_counts().sort_index()
        labels = [_CLASS_LABELS.get(idx, f"Unknown_{idx}") for idx in class_counts.index]
        colors = [_CLASS_COLORS.get(idx, "#999999") for idx in class_counts.index]

        fig = px.bar(
            x=labels,
            y=class_counts.values,
            labels={"x": "Transaction Class", "y": "Count"},
            title="Fraud Class Imbalance",
            text=class_counts.values,
            color=labels,
            color_discrete_sequence=colors,
            template="plotly_white",
        )
        fig.update_traces(textposition="outside", marker=dict(line=dict(width=1, color="white")))
        fig.update_layout(margin=dict(l=60, r=40, t=80, b=60), font=dict(size=11),
                         showlegend=False)

        return self.save_plot(fig, "class_imbalance_distribution.html")

    def plot_feature_distributions(
        self, feature_list: Optional[list[str]] = None
    ) -> list[Path]:
        """Generate feature distribution plots grouped by fraud class.

        Args:
            feature_list: List of features to plot. Defaults to V1, V2, V3, Time, Amount.

        Returns:
            List of paths to saved distribution plots.

        Raises:
            EDAError: If dataset not loaded.
        """
        dataset = self._ensure_dataset_loaded()
        feature_list = feature_list or _DEFAULT_FEATURES
        plot_paths: list[Path] = []

        for feature in feature_list:
            if feature not in dataset.columns:
                self.logger.warning("Skipping feature %s: not found in dataset", feature)
                continue

            try:
                fig = px.histogram(
                    dataset,
                    x=feature,
                    color="Class",
                    marginal="violin",
                    nbins=80,
                    title=f"Distribution of {feature} by Transaction Class",
                    labels={feature: feature, "Class": "Fraud Status"},
                    template="plotly_white",
                    opacity=0.8,
                    color_discrete_map=_CLASS_COLORS,
                )
                fig.update_layout(margin=dict(l=60, r=40, t=80, b=60), font=dict(size=10))
                filename = f"feature_distribution_{feature.lower()}.html"
                plot_paths.append(self.save_plot(fig, filename))
            except Exception as exc:
                self.logger.error("Failed to plot feature %s: %s", feature, exc)
                continue

        self.logger.info("Generated %d feature distribution plots", len(plot_paths))
        return plot_paths

    def plot_time_vs_amount(self) -> Path:
        """Create time-vs-amount scatter plot with class-based coloring.

        Uses sampling for large datasets to maintain interactivity.

        Returns:
            Path to the saved scatter plot.

        Raises:
            EDAError: If dataset not loaded or required columns missing.
        """
        dataset = self._ensure_dataset_loaded()
        self._validate_required_columns(["Time", "Amount", "Class"])

        # Sample for performance on large datasets
        sample_size = min(_MAX_SCATTER_POINTS, len(dataset))
        sample = dataset.sample(n=sample_size, random_state=_SAMPLE_SEED)

        fig = px.scatter(
            sample,
            x="Time",
            y="Amount",
            color="Class",
            title="Transaction Amount Over Time (Sampled)",
            labels={"Time": "Transaction Time (seconds)", "Amount": "Amount (USD)"},
            opacity=0.6,
            template="plotly_white",
            color_discrete_map=_CLASS_COLORS,
            hover_data=["Time", "Amount"],
        )
        fig.update_layout(margin=dict(l=60, r=40, t=80, b=60), font=dict(size=11),
                         height=500)
        fig.update_traces(marker=dict(size=5, line=dict(width=0.5)))

        return self.save_plot(fig, "time_vs_amount_scatter.html")

    def execute(self) -> None:
        """Execute complete EDA pipeline and persist all outputs.

        Raises:
            EDAError: If any analysis step fails.
        """
        try:
            self.logger.info("Starting EDA pipeline execution")
            self.load_dataset()
            self.display_dataset_info()

            missing_report = self.report_missing_values()
            duplicates = self.report_duplicate_records()

            if not duplicates.empty:
                self.logger.info("Sample duplicate rows:\n%s", duplicates.head().to_string())

            self.report_fraud_percentage()
            self.plot_correlation_matrix()
            self.plot_transaction_amount_distribution()
            self.plot_class_imbalance()
            self.plot_feature_distributions()
            self.plot_time_vs_amount()

            self.logger.info("EDA execution completed successfully. Reports saved to %s",
                           self.output_dir)
        except EDAError as exc:
            self.logger.exception("EDA pipeline failed: %s", exc)
            raise


def main() -> None:
    """Entry point for running the EDA pipeline from command line.

    Raises:
        EDAError: If EDA execution fails.
    """
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "creditcard.csv"
    output_dir = base_dir / "assets" / "eda_reports"

    processor = EDAProcessor(data_path=data_path, output_dir=output_dir)

    try:
        processor.execute()
    except EDAError as exc:
        processor.logger.exception("EDA execution failed")
        raise SystemExit(1) from exc
    except Exception as exc:
        processor.logger.exception("Unexpected error during EDA execution: %s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
