"""
Base model class for price prediction.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
import json
from loguru import logger


class BaseModel(ABC):
    """Abstract base class for prediction models."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model.

        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.feature_names = None
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        pass

    def save(self, path: str):
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'config': self.config,
            'is_fitted': self.is_fitted
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """
        Load model from disk.

        Args:
            path: Path to load model from
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.config = model_data.get('config', {})
        self.is_fitted = model_data.get('is_fitted', True)

        logger.info(f"Model loaded from {path}")

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance if supported.

        Returns:
            DataFrame with feature names and importance scores
        """
        return None

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[list] = None
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Prepare data for training/prediction.

        Args:
            df: Input DataFrame
            target_col: Target column name
            feature_cols: List of feature column names (if None, use all except target)

        Returns:
            Tuple of (X, y, feature_names)
        """
        if feature_cols is None:
            # Exclude target columns and metadata columns
            exclude_cols = [target_col, 'timestamp', 'symbol']
            exclude_cols += [col for col in df.columns if col.startswith('target_')]
            feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Remove NaN rows
        df = df.dropna(subset=feature_cols + [target_col])

        X = df[feature_cols].values
        y = df[target_col].values

        self.feature_names = feature_cols

        logger.info(f"Prepared data: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Features: {len(feature_cols)}")

        return X, y, feature_cols


class EnsembleModel(BaseModel):
    """Ensemble of multiple models."""

    def __init__(self, config: Dict[str, Any], models: list):
        """
        Initialize ensemble model.

        Args:
            config: Configuration dictionary
            models: List of BaseModel instances
        """
        super().__init__(config)
        self.models = models
        self.weights = config.get('weights', None)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """Train all models in ensemble."""
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}")
            model.fit(X_train, y_train, X_val, y_val)

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions by averaging all models."""
        predictions = []

        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        if self.weights is not None:
            # Weighted average
            weights = np.array(self.weights)
            return np.average(predictions, axis=0, weights=weights)
        else:
            # Simple average
            return np.mean(predictions, axis=0)