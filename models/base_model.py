"""
Base model interface for sales forecasting
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import joblib
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)


class BaseForecaster(ABC):
    """Abstract base class for forecasting models"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.feature_columns = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseForecaster":
        """
        Fit the model to training data

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        pass

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance if available

        Returns:
            Dictionary of feature names and their importance scores
        """
        return None

    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        model_data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "model_name": self.model_name,
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"Saved {self.model_name} model to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model"""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)
        self.model = model_data["model"]
        self.feature_columns = model_data["feature_columns"]
        self.model_name = model_data.get("model_name", self.model_name)
        self.is_fitted = True

        logger.info(f"Loaded {self.model_name} model from {filepath}")

    def validate_input(self, X: pd.DataFrame):
        """Validate input data"""
        if self.feature_columns is not None:
            missing_cols = set(self.feature_columns) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Reorder columns to match training order
            X = X[self.feature_columns]

        return X
