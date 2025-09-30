"""
Market microstructure features.
"""

import numpy as np
import pandas as pd
from typing import List
from loguru import logger


class MicrostructureFeatures:
    """Compute market microstructure features."""

    def compute_all(
        self,
        df: pd.DataFrame,
        windows: List[int]
    ) -> pd.DataFrame:
        """
        Compute all microstructure features.

        Args:
            df: DataFrame with market data
            windows: List of lookback windows

        Returns:
            DataFrame with added features
        """
        logger.info("Computing microstructure features")

        df = self.compute_order_flow_imbalance(df, windows)
        df = self.compute_price_impact(df, windows)
        df = self.compute_effective_spread(df)
        df = self.compute_realized_spread(df, windows)

        return df

    def compute_order_flow_imbalance(
        self,
        df: pd.DataFrame,
        windows: List[int]
    ) -> pd.DataFrame:
        """Compute order flow imbalance."""
        if 'buy_volume' not in df.columns or 'sell_volume' not in df.columns:
            return df

        for window in windows:
            buy_vol = df['buy_volume'].rolling(window).sum()
            sell_vol = df['sell_volume'].rolling(window).sum()

            # Order flow imbalance
            df[f'ofi_{window}'] = (buy_vol - sell_vol) / (buy_vol + sell_vol)

        return df

    def compute_price_impact(
        self,
        df: pd.DataFrame,
        windows: List[int]
    ) -> pd.DataFrame:
        """Compute price impact measures."""
        if 'mid_price' not in df.columns or 'total_volume' not in df.columns:
            return df

        price_change = df['mid_price'].diff()

        for window in windows:
            volume = df['total_volume'].rolling(window).sum()

            # Price impact per unit volume
            df[f'price_impact_{window}'] = price_change / (volume + 1e-8)

        return df

    def compute_effective_spread(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute effective spread."""
        if not all(col in df.columns for col in ['mid_price', 'vwap']):
            return df

        # Effective spread (2 * |transaction price - mid price|)
        df['effective_spread'] = 2 * np.abs(df['vwap'] - df['mid_price'])

        # Relative effective spread
        df['relative_effective_spread'] = df['effective_spread'] / df['mid_price']

        return df

    def compute_realized_spread(
        self,
        df: pd.DataFrame,
        windows: List[int]
    ) -> pd.DataFrame:
        """Compute realized spread."""
        if 'mid_price' not in df.columns or 'vwap' not in df.columns:
            return df

        for window in windows:
            # Future mid price
            future_mid = df['mid_price'].shift(-window)

            # Realized spread for buys (assuming positive when profitable)
            df[f'realized_spread_buy_{window}'] = 2 * (future_mid - df['vwap'])

            # Realized spread for sells
            df[f'realized_spread_sell_{window}'] = 2 * (df['vwap'] - future_mid)

        return df