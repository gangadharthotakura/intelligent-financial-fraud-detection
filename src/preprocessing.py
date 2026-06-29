"""Preprocessing utilities for fraud detection model training."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessingResult:
    """Structured container for preprocessed datasets."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    processed_data_path: Path


class FraudPreprocessor:
    """Preprocessor for credit card fraud detection data."""

    def __init__(self, data_path: Path, output_dir: Path) -> None:
        self.data_path = data_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._configure_logger()

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

    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset from a CSV file path."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Could not find dataset at {self.data_path}")

        self.logger.info("Loading dataset from %s", self.data_path)
        df = pd.read_csv(self.data_path)
        self.logger.info("Loaded dataset with %d rows and %d columns", *df.shape)
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset with safe imputation."""
        if df.empty:
            raise ValueError("Dataframe is empty. Cannot preprocess missing values.")

        missing_counts = df.isna().sum()
        if missing_counts.any():
            self.logger.info("Missing values found in columns: %s", missing_counts[missing_counts > 0].to_dict())
            df = df.copy()
            for column in df.columns:
                if df[column].isna().any():
                    if df[column].dtype.kind in "if":
                        df[column].fillna(df[column].median(), inplace=True)
                    else:
                        df[column].fillna(df[column].mode().iloc[0], inplace=True)
            self.logger.info("Missing values imputed successfully.")
        else:
            self.logger.info("No missing values detected.")
        return df

    def split_features_target(
        self, df: pd.DataFrame, target_column: str = "Class"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features from the target label."""
        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' not found in dataset.")

        X = df.drop(columns=[target_column])
        y = df[target_column]
        self.logger.info("Separated features and target column '%s'.", target_column)
        return X, y

    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        scaler: Optional[StandardScaler] = None,
    ) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """Scale numeric features using StandardScaler."""
        if scaler is None:
            scaler = StandardScaler()
            self.logger.info("Created new StandardScaler instance.")

        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.logger.info("Feature scaling completed for training and test sets.")
        return X_train_scaled, X_test_scaled, scaler

    def split_train_test(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Perform a stratified train/test split."""
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        self.logger.info(
            "Split data: %d train rows, %d test rows", len(X_train), len(X_test)
        )
        return X_train, X_test, y_train, y_test

    def balance_training_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sampling_strategy: str = "auto",
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Balance the training set using SMOTE."""
        self.logger.info("Applying SMOTE to balance the training data.")
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        self.logger.info(
            "Balanced training set from %d to %d samples.", len(y_train), len(y_resampled)
        )
        return X_resampled, y_resampled

    def save_processed_data(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        scaler: StandardScaler,
    ) -> PreprocessingResult:
        """Persist processed datasets and scaler to disk."""
        output_base = self.output_dir
        output_base.mkdir(parents=True, exist_ok=True)

        train_X_path = output_base / "X_train.npy"
        test_X_path = output_base / "X_test.npy"
        train_y_path = output_base / "y_train.npy"
        test_y_path = output_base / "y_test.npy"
        scaler_path = output_base / "scaler.joblib"

        np.save(train_X_path, X_train)
        np.save(test_X_path, X_test)
        np.save(train_y_path, y_train)
        np.save(test_y_path, y_test)
        joblib.dump(scaler, scaler_path)

        self.logger.info("Saved processed feature arrays and scaler to %s", output_base)
        return PreprocessingResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler,
            processed_data_path=output_base,
        )

    def run_preprocessing(self) -> PreprocessingResult:
        """Run the full preprocessing pipeline end to end."""
        df = self.load_dataset()
        df = self.handle_missing_values(df)
        X, y = self.split_features_target(df)
        X_train, X_test, y_train, y_test = self.split_train_test(X, y)
        X_train_scaled, X_test_scaled, scaler = self.scale_features(X_train, X_test)
        X_train_balanced, y_train_balanced = self.balance_training_data(
            X_train_scaled, y_train.to_numpy()
        )
        return self.save_processed_data(
            X_train_balanced,
            X_test_scaled,
            y_train_balanced,
            y_test.to_numpy(),
            scaler,
        )


def main() -> None:
    """Command-line entry point for preprocessing."""
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "creditcard.csv"
    output_dir = base_dir / "data" / "processed"

    preprocessor = FraudPreprocessor(data_path=data_path, output_dir=output_dir)
    try:
        result = preprocessor.run_preprocessing()
        preprocessor.logger.info("Preprocessing finished. Processed files at %s", result.processed_data_path)
    except Exception as exc:  # pragma: no cover
        preprocessor.logger.exception("Preprocessing failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
