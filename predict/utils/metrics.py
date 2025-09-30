"""
Evaluation metrics for price prediction models.
"""

import numpy as np
import pandas as pd
from typing import Dict
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Information Coefficient (Pearson correlation).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        IC value
    """
    return np.corrcoef(y_true, y_pred)[0, 1]


def calculate_rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Rank Information Coefficient (Spearman correlation).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Rank IC value
    """
    return stats.spearmanr(y_true, y_pred)[0]


def calculate_direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate direction accuracy (prediction of up/down movement).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Direction accuracy (0-1)
    """
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)

    return np.mean(true_direction == pred_direction)


def calculate_sharpe_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.0
) -> float:
    """
    Calculate Sharpe ratio of a simple trading strategy.

    Strategy: Long when prediction > threshold, short when prediction < -threshold.

    Args:
        y_true: True returns
        y_pred: Predicted returns
        threshold: Threshold for taking position

    Returns:
        Sharpe ratio
    """
    # Generate signals
    signals = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))

    # Calculate strategy returns
    strategy_returns = signals * y_true

    # Remove no-position periods
    active_returns = strategy_returns[signals != 0]

    if len(active_returns) == 0:
        return 0.0

    # Sharpe ratio
    return np.mean(active_returns) / (np.std(active_returns) + 1e-8)


def calculate_quantile_returns(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_quantiles: int = 5
) -> pd.DataFrame:
    """
    Calculate average returns for prediction quantiles.

    Args:
        y_true: True returns
        y_pred: Predicted returns
        n_quantiles: Number of quantiles

    Returns:
        DataFrame with quantile statistics
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })

    # Assign quantiles
    df['quantile'] = pd.qcut(df['y_pred'], n_quantiles, labels=False, duplicates='drop')

    # Calculate statistics per quantile
    stats = df.groupby('quantile').agg({
        'y_true': ['mean', 'std', 'count'],
        'y_pred': 'mean'
    })

    return stats


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        prefix: Prefix for metric names

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Regression metrics
    metrics[f'{prefix}rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics[f'{prefix}mae'] = mean_absolute_error(y_true, y_pred)
    metrics[f'{prefix}r2'] = r2_score(y_true, y_pred)

    # Trading metrics
    metrics[f'{prefix}ic'] = calculate_ic(y_true, y_pred)
    metrics[f'{prefix}rank_ic'] = calculate_rank_ic(y_true, y_pred)
    metrics[f'{prefix}direction_accuracy'] = calculate_direction_accuracy(y_true, y_pred)
    metrics[f'{prefix}sharpe_ratio'] = calculate_sharpe_ratio(y_true, y_pred)

    return metrics


def print_metrics(metrics: Dict[str, float]):
    """Print metrics in a formatted way."""
    print("\n" + "="*50)
    print("Evaluation Metrics")
    print("="*50)

    for key, value in metrics.items():
        print(f"{key:30s}: {value:10.6f}")

    print("="*50 + "\n")