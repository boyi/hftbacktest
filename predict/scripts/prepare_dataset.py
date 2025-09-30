"""
Prepare dataset from raw collected data.

This script:
1. Loads raw .gz files from collector
2. Aligns data streams
3. Computes features
4. Adds target variables
5. Saves processed dataset
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import argparse
from datetime import datetime, date
from loguru import logger

from data.loader import DataLoader
from data.processor import DataProcessor
from features.orderbook import OrderBookFeatures
from features.trade import TradeFeatures
from features.technical import TechnicalFeatures
from features.microstructure import MicrostructureFeatures


def parse_date(date_str: str) -> date:
    """Parse date string to date object."""
    return datetime.strptime(date_str, '%Y-%m-%d').date()


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset from raw data')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Trading symbol (e.g., btcusdt)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD), overrides config'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD), overrides config'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path, overrides config'
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Config loaded from {args.config}")

    # Parse dates
    if args.start_date:
        start_date = parse_date(args.start_date)
    else:
        start_date = parse_date(config['data']['train_start_date'])

    if args.end_date:
        end_date = parse_date(args.end_date)
    else:
        end_date = parse_date(config['data']['train_end_date'])

    logger.info(f"Processing data for {args.symbol} from {start_date} to {end_date}")

    # Load raw data
    loader = DataLoader(config['data']['raw_data_path'])
    dfs = loader.load_to_dataframe(args.symbol, start_date, end_date)

    if not dfs:
        logger.error("No data loaded")
        return

    # Align data
    processor = DataProcessor(config)
    df_aligned = processor.align_data(dfs)

    if df_aligned.empty:
        logger.error("No aligned data created")
        return

    logger.info(f"Aligned data shape: {df_aligned.shape}")

    # Compute features
    logger.info("Computing features...")

    # Orderbook features
    if config['features']['orderbook']['enabled']:
        ob_features = OrderBookFeatures(config['features']['orderbook']['depth_levels'])
        df_aligned = ob_features.compute_all(df_aligned)

        # Rolling features
        sample_interval_ms = config['data']['sample_interval_ms']
        windows = [
            int(w * 1000 / sample_interval_ms)
            for w in config['features']['orderbook'].get('windows', [5, 10, 30])
        ]
        df_aligned = ob_features.compute_rolling_features(df_aligned, windows)

    # Trade features
    if config['features']['trade']['enabled']:
        trade_features = TradeFeatures()
        windows = [
            int(w * 1000 / sample_interval_ms)
            for w in config['features']['trade']['windows']
        ]
        df_aligned = trade_features.compute_all(df_aligned, windows)

    # Technical features
    if config['features']['technical']['enabled']:
        tech_features = TechnicalFeatures()
        windows = [
            int(w * 1000 / sample_interval_ms)
            for w in [5, 10, 30]  # Default windows
        ]
        df_aligned = tech_features.compute_all(df_aligned, windows)

    # Microstructure features
    if config['features']['microstructure']['enabled']:
        micro_features = MicrostructureFeatures()
        windows = [
            int(w * 1000 / sample_interval_ms)
            for w in [5, 10, 30]
        ]
        df_aligned = micro_features.compute_all(df_aligned, windows)

    logger.info(f"Features computed. Shape: {df_aligned.shape}")

    # Add targets
    horizons = config['target']['horizons']
    df_aligned = processor.add_target(df_aligned, horizons)

    logger.info(f"Final dataset shape: {df_aligned.shape}")

    # Save processed data
    if args.output:
        output_path = args.output
    else:
        output_dir = Path(config['data']['processed_data_path'])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.symbol}_{start_date}_{end_date}.parquet"

    processor.save_processed_data(df_aligned, output_path, format='parquet')

    logger.info(f"Dataset saved to {output_path}")
    logger.info(f"Total rows: {len(df_aligned)}")
    logger.info(f"Total features: {len(df_aligned.columns)}")

    # Print sample statistics
    logger.info("\nTarget statistics:")
    for horizon in horizons:
        col = f'target_{horizon}s'
        if col in df_aligned.columns:
            logger.info(
                f"  {col}: mean={df_aligned[col].mean():.6f}, "
                f"std={df_aligned[col].std():.6f}"
            )


if __name__ == "__main__":
    main()