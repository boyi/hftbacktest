"""
Technical indicator features.
"""

import numpy as np
import pandas as pd
from typing import List
from loguru import logger


class TechnicalFeatures:
    """Compute technical indicators."""

    def compute_all(
        self,
        df: pd.DataFrame,
        windows: List[int]
    ) -> pd.DataFrame:
        """
        Compute all technical features.

        Args:
            df: DataFrame with price data
            windows: List of lookback windows

        Returns:
            DataFrame with added features
        """
        logger.info("Computing technical features")

        df = self.compute_returns(df, windows)
        df = self.compute_volatility(df, windows)
        df = self.compute_momentum(df, windows)
        df = self.compute_rsi(df, windows)

        return df

    def compute_returns(
        self,
        df: pd.DataFrame,
        windows: List[int]
    ) -> pd.DataFrame:
        """Compute returns."""
        if 'mid_price' not in df.columns:
            return df

        # Simple returns
        for window in windows:
            df[f'return_{window}'] = df['mid_price'].pct_change(window) * 100

        # Log returns
        for window in windows:
            df[f'log_return_{window}'] = (
                np.log(df['mid_price'] / df['mid_price'].shift(window)) * 100
            )

        return df

    def compute_volatility(
        self,
        df: pd.DataFrame,
        windows: List[int]
    ) -> pd.DataFrame:
        """Compute volatility measures."""
        if 'mid_price' not in df.columns:
            return df

        # Returns for volatility calculation
        returns = df['mid_price'].pct_change()

        for window in windows:
            # Historical volatility (std of returns)
            df[f'volatility_{window}'] = returns.rolling(window).std() * np.sqrt(window)

            # Realized volatility (sum of squared returns)
            df[f'realized_vol_{window}'] = np.sqrt(
                (returns ** 2).rolling(window).sum()
            )

        # High-low range
        if 'best_bid' in df.columns and 'best_ask' in df.columns:
            for window in windows:
                high = df['best_ask'].rolling(window).max()
                low = df['best_bid'].rolling(window).min()
                df[f'hl_range_{window}'] = (high - low) / df['mid_price']

        return df

    def compute_momentum(
        self,
        df: pd.DataFrame,
        windows: List[int]
    ) -> pd.DataFrame:
        """Compute momentum indicators."""
        if 'mid_price' not in df.columns:
            return df

        for window in windows:
            # Price momentum
            df[f'momentum_{window}'] = df['mid_price'] - df['mid_price'].shift(window)

            # Rate of change
            df[f'roc_{window}'] = (
                (df['mid_price'] - df['mid_price'].shift(window)) /
                df['mid_price'].shift(window) * 100
            )

        return df

    def compute_rsi(
        self,
        df: pd.DataFrame,
        windows: List[int]
    ) -> pd.DataFrame:
        """Compute RSI (Relative Strength Index)."""
        if 'mid_price' not in df.columns:
            return df

        price_change = df['mid_price'].diff()

        for window in windows:
            # Separate gains and losses
            gains = price_change.where(price_change > 0, 0)
            losses = -price_change.where(price_change < 0, 0)

            # Average gains and losses
            avg_gains = gains.rolling(window).mean()
            avg_losses = losses.rolling(window).mean()

            # RS and RSI
            rs = avg_gains / avg_losses
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))

        return df