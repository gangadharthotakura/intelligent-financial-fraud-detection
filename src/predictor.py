"""Prediction service for fraud detection using persisted models."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd


class FraudPredictor:
    """Predictor that loads a trained fraud detection model and returns risk scores."""

    def __init__(self, model_path: Path, threshold: float = 0.5) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.logger = self._configure_logger()
        self.model = self._load_model()

    def _configure_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _load_model(self) -> Any:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Saved model file not found: {self.model_path}")

        try:
            model = joblib.load(self.model_path)
            self.logger.info("Loaded model from %s", self.model_path)
            return model
        except Exception as exc:  # pragma: no cover
            self.logger.exception("Failed to load model: %s", exc)
            raise RuntimeError("Could not load model artifact") from exc

    def _validate_input(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("Input DataFrame is empty.")
        if data.isna().any().any():
            raise ValueError("Input data contains missing values. Please clean or impute before prediction.")
        return data

    def _predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("The loaded model does not support probability predictions.")
        proba = self.model.predict_proba(data)
        if proba.shape[1] < 2:
            raise ValueError("Probability output does not contain a fraud class probability.")
        return proba[:, 1]

    def _map_risk(self, probability: float) -> str:
        if probability >= 0.85:
            return "High"
        if probability >= 0.60:
            return "Medium"
        return "Low"

    def _confidence_score(self, probability: float) -> float:
        return float(round(probability * 100, 2))

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict fraud for an input DataFrame and return detailed risk results."""
        data = self._validate_input(data)

        try:
            fraud_proba = self._predict_proba(data)
        except Exception as exc:  # pragma: no cover
            self.logger.exception("Prediction failed: %s", exc)
            raise

        predicted_labels = (fraud_proba >= self.threshold).astype(int)
        results = pd.DataFrame(
            {
                "prediction": predicted_labels,
                "fraud_probability": fraud_proba,
                "confidence_score": [self._confidence_score(p) for p in fraud_proba],
                "risk_level": [self._map_risk(p) for p in fraud_proba],
            },
            index=data.index,
        )

        self.logger.info("Predicted %d records with fraud threshold %.2f", len(results), self.threshold)
        return results

    def predict_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Predict a single record represented as a dictionary."""
        if not isinstance(record, dict):
            raise TypeError("Single record input must be a dictionary.")
        data = pd.DataFrame([record])
        result = self.predict(data)
        return result.iloc[0].to_dict()


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    model_path = base_dir / "models" / "xgboost.joblib"
    predictor = FraudPredictor(model_path=model_path)
    predictor.logger.info("Predictor initialized successfully.")


if __name__ == "__main__":
    main()
