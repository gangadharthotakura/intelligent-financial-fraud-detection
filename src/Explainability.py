"""Explainability module for fraud detection using SHAP visualizations."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import shap


@dataclass(frozen=True)
class ExplainabilityConfig:
    background_sample_size: int = 100
    top_features: int = 10
    output_dir: Path = Path("assets") / "explainability"


@dataclass(frozen=True)
class TransactionExplanation:
    transaction_index: int
    predicted_class: int
    fraud_probability: float
    explanation_frame: pd.DataFrame
    waterfall_path: Path
    force_path: Path
    decision_path: Path


class ExplainabilityError(Exception):
    """Custom exception for explainability failures."""


class ExplainabilityEngine:
    """Engine for SHAP-based fraud explainability."""

    def __init__(
        self,
        model_path: Path,
        scaler_path: Optional[Path] = None,
        config: Optional[ExplainabilityConfig] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path) if scaler_path is not None else None
        self.config = config or ExplainabilityConfig()
        self.logger = self._configure_logger()
        self.output_dir = self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = self._load_model()
        self.scaler = self._load_scaler() if self.scaler_path else None
        self.explainer: Optional[shap.Explainer] = None
        self.background_data: Optional[pd.DataFrame] = None
        self.shap_values: Optional[Any] = None

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

    def _load_model(self) -> Any:
        if not self.model_path.exists():
            raise ExplainabilityError(f"Model artifact not found at {self.model_path}")

        try:
            model = joblib.load(self.model_path)
            self.logger.info("Loaded model from %s", self.model_path)
            return model
        except Exception as exc:  # pragma: no cover
            self.logger.exception("Model loading failed: %s", exc)
            raise ExplainabilityError("Failed to load model artifact.") from exc

    def _load_scaler(self) -> Any:
        if not self.scaler_path or not self.scaler_path.exists():
            raise ExplainabilityError(f"Scaler artifact not found at {self.scaler_path}")

        try:
            scaler = joblib.load(self.scaler_path)
            self.logger.info("Loaded scaler from %s", self.scaler_path)
            return scaler
        except Exception as exc:  # pragma: no cover
            self.logger.exception("Scaler loading failed: %s", exc)
            raise ExplainabilityError("Failed to load scaler artifact.") from exc

    def _validate_input_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            raise ExplainabilityError("Input dataset is empty.")
        if data.isna().any().any():
            raise ExplainabilityError("Input contains missing values. Impute before explainability.")
        return data.copy()

    def _scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None:
            return data
        try:
            scaled_array = self.scaler.transform(data)
            return pd.DataFrame(scaled_array, columns=data.columns, index=data.index)
        except Exception as exc:  # pragma: no cover
            self.logger.exception("Scaler transform failed: %s", exc)
            raise ExplainabilityError("Failed to scale features.") from exc

    def _build_explainer(self, data: pd.DataFrame) -> shap.Explainer:
        if self.background_data is None:
            self.background_data = data.sample(
                n=min(self.config.background_sample_size, len(data)),
                random_state=42,
            )
            self.logger.info("Background dataset (%d rows) created.", len(self.background_data))

        try:
            explainer = shap.Explainer(self.model, self.background_data)
            self.logger.info("SHAP explainer initialized.")
            return explainer
        except Exception as exc:  # pragma: no cover
            self.logger.exception("SHAP explainer initialization failed: %s", exc)
            raise ExplainabilityError("Unable to initialize SHAP explainer.") from exc

    def explain_dataset(self, data: pd.DataFrame) -> None:
        data = self._validate_input_dataframe(data)
        features = self._scale_features(data)
        self.explainer = self._build_explainer(features)

        try:
            self.shap_values = self.explainer(features)
            self.logger.info("Generated SHAP values for %d records.", len(features))
        except Exception as exc:  # pragma: no cover
            self.logger.exception("SHAP computation failed: %s", exc)
            raise ExplainabilityError("Unable to compute SHAP values.") from exc

    def _ensure_explainer_initialized(self) -> None:
        if self.explainer is None or self.shap_values is None:
            raise ExplainabilityError("SHAP explainer not initialized. Call explain_dataset() first.")

    def _validate_transaction_index(self, transaction_index: int) -> None:
        self._ensure_explainer_initialized()
        if transaction_index < 0 or transaction_index >= len(self.shap_values):
            raise ExplainabilityError(
                f"Transaction index must be between 0 and {len(self.shap_values) - 1}."
            )

    def _save_html_plot(self, widget: Any, save_path: Path) -> Path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        shap.save_html(str(save_path), widget)
        self.logger.info("Saved SHAP HTML report to %s", save_path)
        return save_path

    def summary_plot(self, save_path: Path) -> Path:
        self._ensure_explainer_initialized()
        self.logger.info("Rendering SHAP summary plot.")
        widget = shap.plots.beeswarm(self.shap_values, show=False)
        return self._save_html_plot(widget, save_path)

    def feature_importance(self, save_path: Path) -> Path:
        self._ensure_explainer_initialized()
        self.logger.info("Rendering SHAP feature importance plot.")
        widget = shap.plots.bar(self.shap_values, max_display=self.config.top_features, show=False)
        return self._save_html_plot(widget, save_path)

    def waterfall_plot(self, transaction_index: int, save_path: Path) -> Path:
        self._validate_transaction_index(transaction_index)
        self.logger.info("Rendering SHAP waterfall for index %d.", transaction_index)
        widget = shap.plots.waterfall(self.shap_values[transaction_index], show=False)
        return self._save_html_plot(widget, save_path)

    def force_plot(self, transaction_index: int, save_path: Path) -> Path:
        self._validate_transaction_index(transaction_index)
        self.logger.info("Rendering SHAP force plot for index %d.", transaction_index)
        widget = shap.plots.force(self.shap_values[transaction_index], show=False)
        return self._save_html_plot(widget, save_path)

    def decision_plot(self, transaction_index: int, save_path: Path) -> Path:
        self._validate_transaction_index(transaction_index)
        self.logger.info("Rendering SHAP decision plot for index %d.", transaction_index)
        widget = shap.plots.decision(self.shap_values[transaction_index], show=False)
        return self._save_html_plot(widget, save_path)

    def _predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(data)[:, 1]
        if hasattr(self.model, "decision_function"):
            decision_scores = self.model.decision_function(data)
            return 1 / (1 + np.exp(-decision_scores))
        raise ExplainabilityError("Model does not support probability estimation.")

    def explain_transaction(
        self, transaction_index: int, data: pd.DataFrame
    ) -> TransactionExplanation:
        self._validate_transaction_index(transaction_index)
        scaled_data = self._scale_features(self._validate_input_dataframe(data))

        probability = float(self._predict_proba(scaled_data)[transaction_index])
        predicted_class = int(self.model.predict(scaled_data)[transaction_index])
        shap_values_row = self.shap_values[transaction_index].values

        contributions = pd.DataFrame(
            {
                "feature": scaled_data.columns,
                "shap_value": shap_values_row,
                "abs_value": np.abs(shap_values_row),
            }
        )
        contributions = contributions.nlargest(self.config.top_features, "abs_value")
        contributions["direction"] = np.where(
            contributions["shap_value"] > 0, "increases fraud", "decreases fraud"
        )

        base_output = self.output_dir
        waterfall_path = base_output / f"waterfall_{transaction_index}.html"
        force_path = base_output / f"force_{transaction_index}.html"
        decision_path = base_output / f"decision_{transaction_index}.html"

        self.waterfall_plot(transaction_index, waterfall_path)
        self.force_plot(transaction_index, force_path)
        self.decision_plot(transaction_index, decision_path)

        return TransactionExplanation(
            transaction_index=transaction_index,
            predicted_class=predicted_class,
            fraud_probability=probability,
            explanation_frame=contributions,
            waterfall_path=waterfall_path,
            force_path=force_path,
            decision_path=decision_path,
        )


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    model_path = base_dir / "models" / "xgboost.joblib"
    scaler_path = base_dir / "data" / "processed" / "scaler.joblib"
    dataset_path = base_dir / "data" / "creditcard.csv"

    engine = ExplainabilityEngine(model_path=model_path, scaler_path=scaler_path)
    dataset = pd.read_csv(dataset_path)
    features = dataset.drop(columns=["Class"], errors="ignore")
    engine.explain_dataset(features)

    explanation = engine.explain_transaction(transaction_index=0, data=features)
    print(explanation)


if __name__ == "__main__":
    main()
