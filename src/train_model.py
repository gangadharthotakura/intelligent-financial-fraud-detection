"""Model training and evaluation pipeline for fraud detection."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


@dataclass
class ModelPerformance:
    """Evaluation results for a trained model."""

    name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: np.ndarray

    def summary(self) -> Dict[str, object]:
        """Return the performance metrics as a dictionary."""
        return {
            "name": self.name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }


class ModelTrainer:
    """Trainer for multiple classification models with evaluation and persistence."""

    def __init__(self, processed_data_dir: Path, output_dir: Path) -> None:
        self.processed_data_dir = processed_data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._configure_logger()
        self.models = self._initialize_models()
        self.results: Dict[str, ModelPerformance] = {}

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

    def _initialize_models(self) -> Dict[str, object]:
        self.logger.info("Initializing candidate models.")
        return {
            "logistic_regression": LogisticRegression(
                solver="liblinear", max_iter=1000, class_weight="balanced", random_state=42
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                class_weight="balanced",
                random_state=42,
            ),
            "xgboost": XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=1,
                random_state=42,
                verbosity=0,
            ),
        }

    def load_processed_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load preprocessed NumPy data arrays from disk."""
        base = self.processed_data_dir
        self.logger.info("Loading processed data from %s", base)

        X_train = np.load(base / "X_train.npy")
        X_test = np.load(base / "X_test.npy")
        y_train = np.load(base / "y_train.npy")
        y_test = np.load(base / "y_test.npy")

        self.logger.info(
            "Processed data loaded: X_train=%s, X_test=%s, y_train=%s, y_test=%s",
            X_train.shape,
            X_test.shape,
            y_train.shape,
            y_test.shape,
        )
        return X_train, X_test, y_train, y_test

    def train_model(self, name: str, model: object, X_train: np.ndarray, y_train: np.ndarray) -> object:
        """Fit a single model using the training dataset."""
        self.logger.info("Training model: %s", name)
        model.fit(X_train, y_train)
        self.logger.info("Model %s training completed.", name)
        return model

    def evaluate_model(self, name: str, model: object, X_test: np.ndarray, y_test: np.ndarray) -> ModelPerformance:
        """Evaluate a trained model and compute performance metrics."""
        self.logger.info("Evaluating model: %s", name)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

        performance = ModelPerformance(
            name=name,
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1_score=f1_score(y_test, y_pred, zero_division=0),
            roc_auc=roc_auc_score(y_test, y_proba),
            confusion_matrix=confusion_matrix(y_test, y_pred),
        )
        self.logger.info("Model %s results: %s", name, performance.summary())
        return performance

    def select_best_model(self) -> Optional[ModelPerformance]:
        """Select the top-performing model using ROC-AUC and F1 score."""
        if not self.results:
            self.logger.warning("No model results are available for selection.")
            return None

        best = max(
            self.results.values(),
            key=lambda metric: (metric.roc_auc, metric.f1_score, metric.accuracy),
        )
        self.logger.info("Best model selected: %s", best.name)
        return best

    def save_model(self, model: object, model_name: str) -> Path:
        """Persist the selected model to disk with joblib."""
        destination = self.output_dir / f"{model_name}.joblib"
        joblib.dump(model, destination)
        self.logger.info("Saved model %s to %s", model_name, destination)
        return destination

    def save_metrics(self, performance: ModelPerformance) -> Path:
        """Save the best model metrics summary to disk."""
        metrics_path = self.output_dir / f"{performance.name}_metrics.npy"
        np.save(metrics_path, performance.summary(), allow_pickle=True)
        self.logger.info("Saved metrics summary for %s to %s", performance.name, metrics_path)
        return metrics_path

    def execute(self) -> ModelPerformance:
        """Train candidate models, evaluate them, and persist the best model."""
        X_train, X_test, y_train, y_test = self.load_processed_data()

        for name, model in self.models.items():
            trained_model = self.train_model(name, model, X_train, y_train)
            performance = self.evaluate_model(name, trained_model, X_test, y_test)
            self.results[name] = performance

        best = self.select_best_model()
        if best is None:
            raise RuntimeError("Failed to select the best model from training results.")

        best_model = self.models[best.name]
        self.save_model(best_model, best.name)
        self.save_metrics(best)
        self.logger.info("Training pipeline completed. Best model: %s", best.name)
        return best


def main() -> None:
    """Command-line entrypoint for model training."""
    base_dir = Path(__file__).resolve().parents[1]
    processed_data_dir = base_dir / "data" / "processed"
    model_output_dir = base_dir / "models"

    trainer = ModelTrainer(processed_data_dir=processed_data_dir, output_dir=model_output_dir)
    try:
        best_performance = trainer.execute()
        trainer.logger.info("Best model performance summary: %s", best_performance.summary())
    except Exception as exc:  # pragma: no cover
        trainer.logger.exception("Model training pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
