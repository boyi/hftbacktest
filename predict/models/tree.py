"""
Tree-based models (XGBoost, LightGBM).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from loguru import logger

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not available")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logger.warning("LightGBM not available")

from .base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost regression model."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize XGBoost model."""
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is not installed")

        super().__init__(config)
        self.model_params = config.get('xgboost', {})

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """Train XGBoost model."""
        logger.info("Training XGBoost model")

        params = {
            'objective': 'reg:squarederror',
            'n_estimators': self.model_params.get('n_estimators', 1000),
            'max_depth': self.model_params.get('max_depth', 6),
            'learning_rate': self.model_params.get('learning_rate', 0.05),
            'subsample': self.model_params.get('subsample', 0.8),
            'colsample_bytree': self.model_params.get('colsample_bytree', 0.8),
            'random_state': self.config.get('training', {}).get('random_seed', 42)
        }

        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model = xgb.XGBRegressor(**params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=self.model_params.get('early_stopping_rounds', 50),
            verbose=False
        )

        self.is_fitted = True
        logger.info("XGBoost training completed")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")

        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            return None

        importance = self.model.feature_importances_

        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df


class LightGBMModel(BaseModel):
    """LightGBM regression model."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize LightGBM model."""
        if not LGB_AVAILABLE:
            raise ImportError("LightGBM is not installed")

        super().__init__(config)
        self.model_params = config.get('lightgbm', {})

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """Train LightGBM model."""
        logger.info("Training LightGBM model")

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'n_estimators': self.model_params.get('n_estimators', 1000),
            'max_depth': self.model_params.get('max_depth', 6),
            'learning_rate': self.model_params.get('learning_rate', 0.05),
            'num_leaves': self.model_params.get('num_leaves', 31),
            'subsample': self.model_params.get('subsample', 0.8),
            'colsample_bytree': self.model_params.get('colsample_bytree', 0.8),
            'random_state': self.config.get('training', {}).get('random_seed', 42),
            'verbose': self.model_params.get('verbose', -1)
        }

        callbacks = []
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks.append(
                lgb.early_stopping(
                    self.model_params.get('early_stopping_rounds', 50),
                    verbose=False
                )
            )

        self.model = lgb.LGBMRegressor(**params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )

        self.is_fitted = True
        logger.info("LightGBM training completed")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")

        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            return None

        importance = self.model.feature_importances_

        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df