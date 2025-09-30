"""
Run complete experiment: data preparation + training + evaluation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import argparse
import subprocess
from loguru import logger


def run_command(cmd: list) -> int:
    """Run a command and return exit code."""
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run complete experiment')
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
        help='Trading symbol'
    )
    parser.add_argument(
        '--skip-prepare',
        action='store_true',
        help='Skip data preparation step'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to prepared data (if skipping preparation)'
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger.info("="*60)
    logger.info(f"Starting experiment for {args.symbol}")
    logger.info("="*60)

    # Step 1: Prepare dataset
    if not args.skip_prepare:
        logger.info("\n[Step 1/2] Preparing dataset...")

        data_path = Path(config['data']['processed_data_path']) / f"{args.symbol}_processed.parquet"

        cmd = [
            'python', 'scripts/prepare_dataset.py',
            '--config', args.config,
            '--symbol', args.symbol,
            '--output', str(data_path)
        ]

        if run_command(cmd) != 0:
            logger.error("Data preparation failed")
            return

        logger.info(f"Dataset prepared: {data_path}")
    else:
        if not args.data_path:
            logger.error("--data-path required when skipping preparation")
            return
        data_path = args.data_path
        logger.info(f"Using existing dataset: {data_path}")

    # Step 2: Train models for all horizons
    logger.info("\n[Step 2/2] Training models...")

    horizons = config['target']['horizons']
    model_type = config['model']['type']

    for horizon in horizons:
        logger.info(f"\nTraining model for {horizon}s horizon...")

        cmd = [
            'python', 'training/train.py',
            '--config', args.config,
            '--data', str(data_path),
            '--model', model_type,
            '--horizon', str(horizon)
        ]

        if run_command(cmd) != 0:
            logger.error(f"Training failed for {horizon}s horizon")
            continue

        logger.info(f"Completed training for {horizon}s horizon")

    logger.info("\n" + "="*60)
    logger.info("Experiment completed!")
    logger.info("="*60)
    logger.info(f"Results saved to: {config['output']['results_path']}")
    logger.info(f"Models saved to: {config['output']['model_path']}")


if __name__ == "__main__":
    main()