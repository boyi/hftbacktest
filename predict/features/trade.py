"""
Trade-based features for price prediction.
"""

import numpy as np
import pandas as pd
from typing import List
from loguru import logger


class TradeFeatures:
    """Extract features from trade data."""

    def compute_all(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """
        Compute all trade features.

        Args:
            df: DataFrame with trade information
            windows: List of rolling windows (in number of samples)

        Returns:
            DataFrame with added features
        """
        logger.info("Computing trade features")

        df = self.compute_volume_features(df, windows)
        df = self.compute_trade_direction_features(df, windows)
        df = self.compute_trade_intensity_features(df, windows)

        return df

    def compute_volume_features(
        self,
        df: pd.DataFrame,
        windows: List[int]
    ) -> pd.DataFrame:
        """Compute volume-based features."""
        if 'total_volume' not in df.columns:
            return df

        for window in windows:
            # Rolling volume statistics
            df[f'volume_sum_{window}'] = df['total_volume'].rolling(window).sum()
            df[f'volume_mean_{window}'] = df['total_volume'].rolling(window).mean()
            df[f'volume_std_{window}'] = df['total_volume'].rolling(window).std()
            df[f'volume_max_{window}'] = df['total_volume'].rolling(window).max()

            # Volume momentum
            df[f'volume_change_{window}'] = df['total_volume'].diff(window)
            df[f'volume_change_pct_{window}'] = df['total_volume'].pct_change(window)

        return df

    def compute_trade_direction_features(
        self,
        df: pd.DataFrame,
        windows: List[int]
    ) -> pd.DataFrame:
        """Compute trade direction features."""
        if 'buy_volume' not in df.columns or 'sell_volume' not in df.columns:
            return df

        for window in windows:
            # Buy/sell volume ratio
            buy_vol = df['buy_volume'].rolling(window).sum()
            sell_vol = df['sell_volume'].rolling(window).sum()
            total_vol = buy_vol + sell_vol

            df[f'buy_ratio_{window}'] = buy_vol / total_vol
            df[f'sell_ratio_{window}'] = sell_vol / total_vol
            df[f'buy_sell_imbalance_{window}'] = (buy_vol - sell_vol) / total_vol

            # Delta volume
            df[f'delta_volume_{window}'] = buy_vol - sell_vol

        return df

    def compute_trade_intensity_features(
        self,
        df: pd.DataFrame,
        windows: List[int]
    ) -> pd.DataFrame:
        """Compute trade intensity features."""
        if 'trade_count' not in df.columns:
            return df

        for window in windows:
            # Trade frequency
            df[f'trade_count_sum_{window}'] = df['trade_count'].rolling(window).sum()
            df[f'trade_count_mean_{window}'] = df['trade_count'].rolling(window).mean()

            # Average trade size
            total_vol = df['total_volume'].rolling(window).sum()
            total_trades = df['trade_count'].rolling(window).sum()
            df[f'avg_trade_size_{window}'] = total_vol / total_trades

        return df