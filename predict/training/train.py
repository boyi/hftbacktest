"""
Training script for price prediction models.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from loguru import logger

from data.processor import DataProcessor
from models.tree import LightGBMModel, XGBoostModel
from utils.metrics import evaluate_predictions, print_metrics


def train_model(config_path: str, data_path: str, model_type: str, target_horizon: int):
    """
    Train a prediction model.

    Args:
        config_path: Path to configuration file
        data_path: Path to processed dataset
        model_type: Type of model ('lightgbm', 'xgboost')
        target_horizon: Prediction horizon in seconds
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Training {model_type} model for {target_horizon}s horizon")

    # Load data
    processor = DataProcessor(config)
    df = processor.load_processed_data(data_path)

    logger.info(f"Loaded data shape: {df.shape}")

    # Prepare features and target
    target_col = f'target_{target_horizon}s'

    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in data")

    # Remove rows with NaN
    df = df.dropna()

    logger.info(f"Data shape after removing NaN: {df.shape}")

    # Prepare data
    exclude_cols = ['timestamp', 'symbol'] + [col for col in df.columns if col.startswith('target_')]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].values
    y = df[target_col].values

    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Samples: {len(X)}")

    # Split data
    test_size = config['training'].get('test_size', 0.2)

    # Use time-based split (important for time series!)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Further split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=config['training']['random_seed']
    )

    logger.info(f"Train size: {len(X_train)}")
    logger.info(f"Validation size: {len(X_val)}")
    logger.info(f"Test size: {len(X_test)}")

    # Initialize model
    if model_type == 'lightgbm':
        model = LightGBMModel(config)
    elif model_type == 'xgboost':
        model = XGBoostModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.feature_names = feature_cols

    # Train model
    model.fit(X_train, y_train, X_val, y_val)

    # Evaluate on validation set
    logger.info("\n=== Validation Set Performance ===")
    y_val_pred = model.predict(X_val)
    val_metrics = evaluate_predictions(y_val, y_val_pred, prefix='val_')
    print_metrics(val_metrics)

    # Evaluate on test set
    logger.info("\n=== Test Set Performance ===")
    y_test_pred = model.predict(X_test)
    test_metrics = evaluate_predictions(y_test, y_test_pred, prefix='test_')
    print_metrics(test_metrics)

    # Feature importance
    feature_importance = model.get_feature_importance()
    if feature_importance is not None:
        logger.info("\n=== Top 20 Feature Importance ===")
        print(feature_importance.head(20))

        # Save feature importance
        output_dir = Path(config['output']['results_path'])
        output_dir.mkdir(parents=True, exist_ok=True)

        fi_path = output_dir / f"feature_importance_{model_type}_{target_horizon}s.csv"
        feature_importance.to_csv(fi_path, index=False)
        logger.info(f"Feature importance saved to {fi_path}")

    # Save model
    model_dir = Path(config['output']['model_path'])
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_type}_{target_horizon}s.pkl"
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # Save predictions
    if config['output'].get('save_predictions', True):
        pred_df = pd.DataFrame({
            'y_true': y_test,
            'y_pred': y_test_pred,
            'error': y_test - y_test_pred,
            'abs_error': np.abs(y_test - y_test_pred)
        })

        pred_path = output_dir / f"predictions_{model_type}_{target_horizon}s.csv"
        pred_df.to_csv(pred_path, index=False)
        logger.info(f"Predictions saved to {pred_path}")

    # Save metrics
    all_metrics = {**val_metrics, **test_metrics}
    metrics_df = pd.DataFrame([all_metrics])

    metrics_path = output_dir / f"metrics_{model_type}_{target_horizon}s.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    return model, test_metrics


def main():
    parser = argparse.ArgumentParser(description='Train price prediction model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to processed dataset'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='lightgbm',
        choices=['lightgbm', 'xgboost'],
        help='Model type'
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=30,
        help='Prediction horizon in seconds'
    )

    args = parser.parse_args()

    train_model(args.config, args.data, args.model, args.horizon)


if __name__ == "__main__":
    main()