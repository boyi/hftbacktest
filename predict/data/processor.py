"""
Data processor for feature engineering and dataset preparation.
Converts raw market data into machine learning ready datasets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger
from pathlib import Path
import pickle


class OrderBookProcessor:
    """Process order book depth updates to maintain current state."""

    def __init__(self, depth_levels: int = 10):
        """
        Initialize OrderBookProcessor.

        Args:
            depth_levels: Number of price levels to maintain
        """
        self.depth_levels = depth_levels
        self.bids = {}  # price -> quantity
        self.asks = {}  # price -> quantity

    def update(self, bids: List[List[str]], asks: List[List[str]]):
        """
        Update order book with new bids and asks.

        Args:
            bids: List of [price, quantity] pairs
            asks: List of [price, quantity] pairs
        """
        # Update bids
        for price_str, qty_str in bids:
            price = float(price_str)
            qty = float(qty_str)

            if qty == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty

        # Update asks
        for price_str, qty_str in asks:
            price = float(price_str)
            qty = float(qty_str)

            if qty == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

    def get_snapshot(self) -> Dict:
        """
        Get current order book snapshot.

        Returns:
            Dictionary with bids and asks
        """
        # Sort and limit to depth_levels
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])

        snapshot = {
            'bids': sorted_bids[:self.depth_levels],
            'asks': sorted_asks[:self.depth_levels]
        }

        return snapshot

    def get_best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        """Get best bid and ask prices."""
        best_bid = max(self.bids.keys()) if self.bids else None
        best_ask = min(self.asks.keys()) if self.asks else None
        return best_bid, best_ask

    def clear(self):
        """Clear order book."""
        self.bids.clear()
        self.asks.clear()


class DataProcessor:
    """Main data processor for creating ML datasets."""

    def __init__(self, config: Dict):
        """
        Initialize DataProcessor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.sample_interval_ms = config['data'].get('sample_interval_ms', 100)
        self.depth_levels = config['features']['orderbook'].get('depth_levels', 10)

    def align_data(
        self,
        dfs: Dict[str, pd.DataFrame],
        sample_interval_ms: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Align different data streams to a common timeline.

        Args:
            dfs: Dictionary of DataFrames (trade, depthUpdate, bookTicker)
            sample_interval_ms: Sampling interval in milliseconds

        Returns:
            Aligned DataFrame with snapshots at regular intervals
        """
        if sample_interval_ms is None:
            sample_interval_ms = self.sample_interval_ms

        logger.info(
            f"Aligning data with {sample_interval_ms}ms interval"
        )

        # Get time range
        min_time = min(df['recv_time'].min() for df in dfs.values())
        max_time = max(df['recv_time'].max() for df in dfs.values())

        # Create regular time grid
        time_range = pd.date_range(
            start=min_time,
            end=max_time,
            freq=f'{sample_interval_ms}ms'
        )

        # Initialize orderbook processor
        orderbook = OrderBookProcessor(self.depth_levels)

        # Process data in chronological order
        snapshots = []

        depth_df = dfs.get('depthUpdate', pd.DataFrame())
        trade_df = dfs.get('trade', pd.DataFrame())
        ticker_df = dfs.get('bookTicker', pd.DataFrame())

        # Combine and sort all events
        all_events = []

        if not depth_df.empty:
            for idx, row in depth_df.iterrows():
                all_events.append({
                    'time': row['recv_time'],
                    'type': 'depth',
                    'data': row
                })

        if not trade_df.empty:
            for idx, row in trade_df.iterrows():
                all_events.append({
                    'time': row['recv_time'],
                    'type': 'trade',
                    'data': row
                })

        if not ticker_df.empty:
            for idx, row in ticker_df.iterrows():
                all_events.append({
                    'time': row['recv_time'],
                    'type': 'ticker',
                    'data': row
                })

        all_events.sort(key=lambda x: x['time'])

        # Process events and sample at regular intervals
        current_sample_idx = 0
        trades_in_interval = []
        last_ticker = None

        for event in all_events:
            event_time = event['time']
            event_type = event['type']
            event_data = event['data']

            # Check if we need to take a snapshot
            while (current_sample_idx < len(time_range) and
                   event_time >= time_range[current_sample_idx]):

                snapshot_time = time_range[current_sample_idx]

                # Create snapshot
                snapshot = self._create_snapshot(
                    snapshot_time,
                    orderbook,
                    trades_in_interval,
                    last_ticker
                )

                if snapshot is not None:
                    snapshots.append(snapshot)

                # Reset interval data
                trades_in_interval = []
                current_sample_idx += 1

            # Process event
            if event_type == 'depth':
                orderbook.update(
                    event_data.get('bids', []),
                    event_data.get('asks', [])
                )
            elif event_type == 'trade':
                trades_in_interval.append(event_data)
            elif event_type == 'ticker':
                last_ticker = event_data

        # Convert to DataFrame
        if not snapshots:
            logger.warning("No snapshots created")
            return pd.DataFrame()

        df = pd.DataFrame(snapshots)
        logger.info(f"Created {len(df)} snapshots")

        return df

    def _create_snapshot(
        self,
        timestamp: pd.Timestamp,
        orderbook: OrderBookProcessor,
        trades: List,
        last_ticker: Optional[pd.Series]
    ) -> Optional[Dict]:
        """Create a snapshot at a given timestamp."""
        best_bid, best_ask = orderbook.get_best_bid_ask()

        if best_bid is None or best_ask is None:
            return None

        snapshot = {
            'timestamp': timestamp,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'mid_price': (best_bid + best_ask) / 2,
            'spread': best_ask - best_bid,
            'spread_bps': (best_ask - best_bid) / ((best_bid + best_ask) / 2) * 10000
        }

        # Add order book depth
        ob_snapshot = orderbook.get_snapshot()

        for i, (price, qty) in enumerate(ob_snapshot['bids']):
            snapshot[f'bid_price_{i}'] = price
            snapshot[f'bid_qty_{i}'] = qty

        for i, (price, qty) in enumerate(ob_snapshot['asks']):
            snapshot[f'ask_price_{i}'] = price
            snapshot[f'ask_qty_{i}'] = qty

        # Add trade statistics
        if trades:
            total_volume = sum(t['quantity'] for t in trades)
            buy_volume = sum(
                t['quantity'] for t in trades if not t['is_buyer_maker']
            )
            sell_volume = sum(
                t['quantity'] for t in trades if t['is_buyer_maker']
            )

            snapshot['trade_count'] = len(trades)
            snapshot['total_volume'] = total_volume
            snapshot['buy_volume'] = buy_volume
            snapshot['sell_volume'] = sell_volume
            snapshot['buy_sell_ratio'] = (
                buy_volume / sell_volume if sell_volume > 0 else 0
            )

            # VWAP
            vwap_num = sum(t['price'] * t['quantity'] for t in trades)
            snapshot['vwap'] = vwap_num / total_volume if total_volume > 0 else 0

        else:
            snapshot['trade_count'] = 0
            snapshot['total_volume'] = 0
            snapshot['buy_volume'] = 0
            snapshot['sell_volume'] = 0
            snapshot['buy_sell_ratio'] = 0
            snapshot['vwap'] = 0

        # Add last ticker data if available
        if last_ticker is not None:
            snapshot['ticker_best_bid'] = last_ticker.get('best_bid')
            snapshot['ticker_best_ask'] = last_ticker.get('best_ask')
            snapshot['ticker_best_bid_qty'] = last_ticker.get('best_bid_qty')
            snapshot['ticker_best_ask_qty'] = last_ticker.get('best_ask_qty')

        return snapshot

    def add_target(
        self,
        df: pd.DataFrame,
        horizons: List[int]
    ) -> pd.DataFrame:
        """
        Add target variables (future returns).

        Args:
            df: Input DataFrame with 'mid_price' column
            horizons: List of prediction horizons in seconds

        Returns:
            DataFrame with added target columns
        """
        logger.info(f"Adding targets for horizons: {horizons}")

        for horizon_sec in horizons:
            # Calculate number of periods
            periods = int(horizon_sec * 1000 / self.sample_interval_ms)

            # Calculate future return
            future_price = df['mid_price'].shift(-periods)
            current_price = df['mid_price']

            df[f'target_{horizon_sec}s'] = (
                (future_price - current_price) / current_price * 100
            )

            # Also add direction
            df[f'target_{horizon_sec}s_direction'] = np.sign(
                df[f'target_{horizon_sec}s']
            )

        # Drop rows with NaN targets (at the end)
        df = df.dropna(subset=[f'target_{h}s' for h in horizons])

        logger.info(f"Dataset shape after adding targets: {df.shape}")

        return df

    def save_processed_data(
        self,
        df: pd.DataFrame,
        output_path: str,
        format: str = 'parquet'
    ):
        """
        Save processed data to disk.

        Args:
            df: Processed DataFrame
            output_path: Output file path
            format: Output format ('parquet', 'csv', 'pickle')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'parquet':
            df.to_parquet(output_path, index=False)
        elif format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(df, f)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Saved processed data to {output_path}")

    def load_processed_data(
        self,
        input_path: str,
        format: str = 'parquet'
    ) -> pd.DataFrame:
        """Load processed data from disk."""
        if format == 'parquet':
            return pd.read_parquet(input_path)
        elif format == 'csv':
            return pd.read_csv(input_path, parse_dates=['timestamp'])
        elif format == 'pickle':
            with open(input_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unknown format: {format}")