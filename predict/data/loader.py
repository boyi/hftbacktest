"""
Data loader for HFT collected data.
Reads and parses .gz files from the collector project.
"""

import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional, Generator, Tuple
from datetime import datetime, date
import pandas as pd
from loguru import logger


class DataLoader:
    """Load and parse collected market data from .gz files."""

    def __init__(self, data_path: str):
        """
        Initialize DataLoader.

        Args:
            data_path: Path to directory containing .gz files
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} does not exist")

    def list_available_files(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[Path]:
        """
        List available data files for a symbol within date range.

        Args:
            symbol: Trading symbol (e.g., 'btcusdt')
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of file paths
        """
        pattern = f"{symbol.lower()}_*.gz"
        files = sorted(self.data_path.glob(pattern))

        if start_date or end_date:
            filtered_files = []
            for file in files:
                # Extract date from filename: symbol_YYYYMMDD.gz
                try:
                    date_str = file.stem.split('_')[-1]
                    file_date = datetime.strptime(date_str, '%Y%m%d').date()

                    if start_date and file_date < start_date:
                        continue
                    if end_date and file_date > end_date:
                        continue

                    filtered_files.append(file)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Could not parse date from filename {file}: {e}")
                    continue

            return filtered_files

        return files

    def read_file_raw(self, file_path: Path) -> Generator[Tuple[int, Dict], None, None]:
        """
        Read raw data from a .gz file.

        Args:
            file_path: Path to .gz file

        Yields:
            Tuple of (timestamp_ns, json_data)
        """
        logger.info(f"Reading file: {file_path}")

        try:
            with gzip.open(file_path, 'rt') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # Parse line: "timestamp json_data"
                        space_idx = line.find(' ')
                        if space_idx == -1:
                            logger.warning(f"Invalid line format at {line_num}")
                            continue

                        timestamp_str = line[:space_idx]
                        json_str = line[space_idx + 1:]

                        timestamp_ns = int(timestamp_str)
                        data = json.loads(json_str)

                        yield timestamp_ns, data

                    except (ValueError, json.JSONDecodeError) as e:
                        logger.warning(
                            f"Error parsing line {line_num} in {file_path.name}: {e}"
                        )
                        continue

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise

    def parse_depth_update(self, data: Dict) -> Optional[Dict]:
        """
        Parse depth update event.

        Args:
            data: Raw JSON data

        Returns:
            Parsed depth update or None if not a depth update
        """
        stream_data = data.get('data') if 'stream' in data else data

        if stream_data.get('e') != 'depthUpdate':
            return None

        return {
            'event': 'depthUpdate',
            'symbol': stream_data.get('s'),
            'exchange_time': stream_data.get('E'),
            'transaction_time': stream_data.get('T'),
            'first_update_id': stream_data.get('U'),
            'final_update_id': stream_data.get('u'),
            'prev_final_update_id': stream_data.get('pu'),
            'bids': stream_data.get('b', []),  # [[price, qty], ...]
            'asks': stream_data.get('a', [])
        }

    def parse_trade(self, data: Dict) -> Optional[Dict]:
        """
        Parse trade event.

        Args:
            data: Raw JSON data

        Returns:
            Parsed trade or None if not a trade
        """
        stream_data = data.get('data') if 'stream' in data else data

        if stream_data.get('e') != 'trade':
            return None

        return {
            'event': 'trade',
            'symbol': stream_data.get('s'),
            'exchange_time': stream_data.get('E'),
            'trade_time': stream_data.get('T'),
            'trade_id': stream_data.get('t'),
            'price': float(stream_data.get('p')),
            'quantity': float(stream_data.get('q')),
            'order_type': stream_data.get('X'),
            'is_buyer_maker': stream_data.get('m')  # True = sell, False = buy
        }

    def parse_book_ticker(self, data: Dict) -> Optional[Dict]:
        """
        Parse bookTicker event.

        Args:
            data: Raw JSON data

        Returns:
            Parsed bookTicker or None
        """
        stream_data = data.get('data') if 'stream' in data else data

        if stream_data.get('e') != 'bookTicker':
            return None

        return {
            'event': 'bookTicker',
            'symbol': stream_data.get('s'),
            'update_id': stream_data.get('u'),
            'best_bid': float(stream_data.get('b')),
            'best_bid_qty': float(stream_data.get('B')),
            'best_ask': float(stream_data.get('a')),
            'best_ask_qty': float(stream_data.get('A')),
            'transaction_time': stream_data.get('T'),
            'exchange_time': stream_data.get('E')
        }

    def load_to_dataframe(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        event_types: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data into pandas DataFrames organized by event type.

        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            event_types: List of event types to load (e.g., ['trade', 'depthUpdate'])
                        If None, loads all types

        Returns:
            Dictionary mapping event type to DataFrame
        """
        files = self.list_available_files(symbol, start_date, end_date)

        if not files:
            logger.warning(
                f"No files found for {symbol} between {start_date} and {end_date}"
            )
            return {}

        # Collect events by type
        events = {
            'trade': [],
            'depthUpdate': [],
            'bookTicker': []
        }

        for file in files:
            for recv_timestamp, data in self.read_file_raw(file):
                # Add receive timestamp to all events
                parsed = None

                trade = self.parse_trade(data)
                if trade:
                    trade['recv_timestamp'] = recv_timestamp
                    events['trade'].append(trade)
                    continue

                depth = self.parse_depth_update(data)
                if depth:
                    depth['recv_timestamp'] = recv_timestamp
                    events['depthUpdate'].append(depth)
                    continue

                ticker = self.parse_book_ticker(data)
                if ticker:
                    ticker['recv_timestamp'] = recv_timestamp
                    events['bookTicker'].append(ticker)
                    continue

        # Convert to DataFrames
        dataframes = {}
        for event_type, event_list in events.items():
            if event_types and event_type not in event_types:
                continue

            if not event_list:
                logger.warning(f"No {event_type} events found")
                continue

            df = pd.DataFrame(event_list)

            # Convert timestamps to datetime
            df['recv_time'] = pd.to_datetime(df['recv_timestamp'], unit='ns')

            if 'exchange_time' in df.columns:
                df['exchange_time'] = pd.to_datetime(df['exchange_time'], unit='ms')

            if 'transaction_time' in df.columns:
                df['transaction_time'] = pd.to_datetime(
                    df['transaction_time'], unit='ms'
                )
            if 'trade_time' in df.columns:
                df['trade_time'] = pd.to_datetime(df['trade_time'], unit='ms')

            df = df.sort_values('recv_timestamp').reset_index(drop=True)

            dataframes[event_type] = df
            logger.info(f"Loaded {len(df)} {event_type} events")

        return dataframes


if __name__ == "__main__":
    # Example usage
    loader = DataLoader("../data/collected")

    # List available files
    files = loader.list_available_files("btcusdt")
    print(f"Found {len(files)} files")

    # Load data
    dfs = loader.load_to_dataframe(
        "btcusdt",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2)
    )

    for event_type, df in dfs.items():
        print(f"\n{event_type}:")
        print(df.head())
        print(f"Shape: {df.shape}")