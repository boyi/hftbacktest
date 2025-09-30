"""
Order book features for price prediction.
"""

import numpy as np
import pandas as pd
from typing import List
from loguru import logger


class OrderBookFeatures:
    """Extract features from order book data."""

    def __init__(self, depth_levels: int = 10):
        """
        Initialize OrderBookFeatures.

        Args:
            depth_levels: Number of depth levels to use
        """
        self.depth_levels = depth_levels

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all order book features.

        Args:
            df: DataFrame with order book snapshots

        Returns:
            DataFrame with added features
        """
        logger.info("Computing order book features")

        df = self.compute_spread_features(df)
        df = self.compute_imbalance_features(df)
        df = self.compute_depth_features(df)
        df = self.compute_price_level_features(df)
        df = self.compute_weighted_features(df)

        return df

    def compute_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute spread-related features."""
        # Already have spread and spread_bps from processor

        # Relative spread
        if 'mid_price' in df.columns:
            df['relative_spread'] = df['spread'] / df['mid_price']

        # Spread momentum
        if 'spread' in df.columns:
            df['spread_change'] = df['spread'].diff()
            df['spread_change_pct'] = df['spread'].pct_change()

        return df

    def compute_imbalance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute order book imbalance features."""
        # Volume imbalance at best bid/ask
        if 'bid_qty_0' in df.columns and 'ask_qty_0' in df.columns:
            total_qty = df['bid_qty_0'] + df['ask_qty_0']
            df['bid_ask_imbalance'] = (
                (df['bid_qty_0'] - df['ask_qty_0']) / total_qty
            )

        # Depth imbalance (multiple levels)
        for n in [5, 10]:
            if n > self.depth_levels:
                continue

            bid_cols = [f'bid_qty_{i}' for i in range(n) if f'bid_qty_{i}' in df.columns]
            ask_cols = [f'ask_qty_{i}' for i in range(n) if f'ask_qty_{i}' in df.columns]

            if bid_cols and ask_cols:
                bid_depth = df[bid_cols].sum(axis=1)
                ask_depth = df[ask_cols].sum(axis=1)
                total_depth = bid_depth + ask_depth

                df[f'depth_imbalance_{n}'] = (bid_depth - ask_depth) / total_depth

        # Weighted imbalance
        bid_qty_cols = [f'bid_qty_{i}' for i in range(self.depth_levels)]
        ask_qty_cols = [f'ask_qty_{i}' for i in range(self.depth_levels)]

        if all(col in df.columns for col in bid_qty_cols + ask_qty_cols):
            # Weight by distance from mid price
            weights = np.array([1.0 / (i + 1) for i in range(self.depth_levels)])

            bid_weighted = sum(
                df[f'bid_qty_{i}'] * weights[i]
                for i in range(self.depth_levels)
            )
            ask_weighted = sum(
                df[f'ask_qty_{i}'] * weights[i]
                for i in range(self.depth_levels)
            )

            df['weighted_imbalance'] = (
                (bid_weighted - ask_weighted) / (bid_weighted + ask_weighted)
            )

        return df

    def compute_depth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute order book depth features."""
        # Total depth at different levels
        for n in [3, 5, 10]:
            if n > self.depth_levels:
                continue

            bid_cols = [f'bid_qty_{i}' for i in range(n) if f'bid_qty_{i}' in df.columns]
            ask_cols = [f'ask_qty_{i}' for i in range(n) if f'ask_qty_{i}' in df.columns]

            if bid_cols:
                df[f'total_bid_depth_{n}'] = df[bid_cols].sum(axis=1)
            if ask_cols:
                df[f'total_ask_depth_{n}'] = df[ask_cols].sum(axis=1)
            if bid_cols and ask_cols:
                df[f'total_depth_{n}'] = df[bid_cols + ask_cols].sum(axis=1)

        # Average depth
        bid_cols = [f'bid_qty_{i}' for i in range(self.depth_levels) if f'bid_qty_{i}' in df.columns]
        ask_cols = [f'ask_qty_{i}' for i in range(self.depth_levels) if f'ask_qty_{i}' in df.columns]

        if bid_cols:
            df['avg_bid_depth'] = df[bid_cols].mean(axis=1)
        if ask_cols:
            df['avg_ask_depth'] = df[ask_cols].mean(axis=1)

        return df

    def compute_price_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute price level features."""
        # Price range
        bid_price_cols = [f'bid_price_{i}' for i in range(self.depth_levels) if f'bid_price_{i}' in df.columns]
        ask_price_cols = [f'ask_price_{i}' for i in range(self.depth_levels) if f'ask_price_{i}' in df.columns]

        if bid_price_cols:
            df['bid_price_range'] = df[bid_price_cols].max(axis=1) - df[bid_price_cols].min(axis=1)
        if ask_price_cols:
            df['ask_price_range'] = df[ask_price_cols].max(axis=1) - df[ask_price_cols].min(axis=1)

        # Order book slope (price change per level)
        if len(bid_price_cols) >= 2:
            df['bid_slope'] = (df['bid_price_0'] - df[f'bid_price_{len(bid_price_cols)-1}']) / len(bid_price_cols)

        if len(ask_price_cols) >= 2:
            df['ask_slope'] = (df[f'ask_price_{len(ask_price_cols)-1}'] - df['ask_price_0']) / len(ask_price_cols)

        return df

    def compute_weighted_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume-weighted features."""
        # Weighted mid price
        if 'bid_price_0' in df.columns and 'ask_price_0' in df.columns:
            if 'bid_qty_0' in df.columns and 'ask_qty_0' in df.columns:
                total_qty = df['bid_qty_0'] + df['ask_qty_0']
                df['weighted_mid_price'] = (
                    (df['bid_price_0'] * df['ask_qty_0'] +
                     df['ask_price_0'] * df['bid_qty_0']) / total_qty
                )

        # VWAP for order book
        bid_price_cols = [f'bid_price_{i}' for i in range(self.depth_levels) if f'bid_price_{i}' in df.columns]
        bid_qty_cols = [f'bid_qty_{i}' for i in range(self.depth_levels) if f'bid_qty_{i}' in df.columns]
        ask_price_cols = [f'ask_price_{i}' for i in range(self.depth_levels) if f'ask_price_{i}' in df.columns]
        ask_qty_cols = [f'ask_qty_{i}' for i in range(self.depth_levels) if f'ask_qty_{i}' in df.columns]

        if bid_price_cols and bid_qty_cols:
            bid_value = sum(df[price] * df[qty] for price, qty in zip(bid_price_cols, bid_qty_cols))
            bid_total_qty = df[bid_qty_cols].sum(axis=1)
            df['ob_vwap_bid'] = bid_value / bid_total_qty

        if ask_price_cols and ask_qty_cols:
            ask_value = sum(df[price] * df[qty] for price, qty in zip(ask_price_cols, ask_qty_cols))
            ask_total_qty = df[ask_qty_cols].sum(axis=1)
            df['ob_vwap_ask'] = ask_value / ask_total_qty

        return df

    def compute_rolling_features(
        self,
        df: pd.DataFrame,
        windows: List[int]
    ) -> pd.DataFrame:
        """
        Compute rolling statistics of order book features.

        Args:
            df: DataFrame with order book features
            windows: List of window sizes (in number of samples)

        Returns:
            DataFrame with rolling features
        """
        logger.info(f"Computing rolling features for windows: {windows}")

        feature_cols = [
            'spread', 'spread_bps', 'bid_ask_imbalance',
            'depth_imbalance_5', 'weighted_imbalance',
            'total_depth_5', 'total_depth_10'
        ]

        for window in windows:
            for col in feature_cols:
                if col not in df.columns:
                    continue

                # Mean
                df[f'{col}_mean_{window}'] = df[col].rolling(window).mean()

                # Std
                df[f'{col}_std_{window}'] = df[col].rolling(window).std()

                # Min/Max
                df[f'{col}_min_{window}'] = df[col].rolling(window).min()
                df[f'{col}_max_{window}'] = df[col].rolling(window).max()

                # Change
                df[f'{col}_change_{window}'] = df[col].diff(window)

        return df