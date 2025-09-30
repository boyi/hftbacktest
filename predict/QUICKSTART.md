# Quick Start Guide

## Prerequisites

1. You have already collected data using the `collector` project
2. Data is stored in `.gz` format (e.g., `btcusdt_20240101.gz`)

## Installation

```bash
cd predict
pip install -r requirements.txt
```

## Configuration

Edit `config/config.yaml` and update:

```yaml
data:
  raw_data_path: "../data/collected"  # Path to your .gz files
  symbols: ["btcusdt"]  # Your trading symbols
```

## Run Your First Experiment

### Option 1: Complete Pipeline (Recommended)

```bash
python scripts/run_experiment.py --symbol btcusdt
```

This runs everything automatically:
1. ✅ Loads raw data from collector
2. ✅ Aligns orderbook + trades
3. ✅ Computes 100+ features
4. ✅ Trains models for all horizons (5s, 10s, 30s, 60s)
5. ✅ Saves models and results

### Option 2: Step by Step

#### Step 1: Prepare Dataset
```bash
python scripts/prepare_dataset.py \
    --symbol btcusdt \
    --start-date 2024-01-01 \
    --end-date 2024-01-31
```

Output: `data/processed/btcusdt_20240101_20240131.parquet`

#### Step 2: Train Model
```bash
python training/train.py \
    --data data/processed/btcusdt_20240101_20240131.parquet \
    --model lightgbm \
    --horizon 30
```

Output:
- `models/saved/lightgbm_30s.pkl` - Trained model
- `results/predictions_lightgbm_30s.csv` - Predictions
- `results/metrics_lightgbm_30s.csv` - Performance metrics
- `results/feature_importance_lightgbm_30s.csv` - Feature importance

## Check Results

```bash
# View metrics
cat results/metrics_lightgbm_30s.csv

# View feature importance
head -20 results/feature_importance_lightgbm_30s.csv
```

## Expected Performance

For 30-second prediction horizon:
- **IC**: 0.05 - 0.15 (typical for HFT)
- **Direction Accuracy**: 52% - 58%
- **RMSE**: Depends on volatility

## Next Steps

1. **Tune hyperparameters** in `config/config.yaml`
2. **Add more features** in `features/` modules
3. **Try different horizons**: 5s (harder) vs 60s (easier)
4. **Backtest** predictions using HftBacktest
5. **Live trading** integration with connector

## Troubleshooting

### "No files found for symbol"
- Check `raw_data_path` in config
- Verify .gz files exist and match naming pattern: `symbol_YYYYMMDD.gz`

### Out of Memory
- Reduce date range in `prepare_dataset.py`
- Increase `sample_interval_ms` in config (sample less frequently)

### Poor predictions
- Need more training data (at least 1 week)
- Check data quality (no large gaps)
- Try longer prediction horizons first (60s is easier than 5s)

## Example Output

```
[INFO] Training LightGBM model for 30s horizon
[INFO] Train size: 500000
[INFO] Validation size: 125000
[INFO] Test size: 156250

=== Test Set Performance ===
test_rmse         :   0.082341
test_mae          :   0.045123
test_r2           :   0.123456
test_ic           :   0.098765
test_rank_ic      :   0.087654
test_direction_accuracy: 0.567890
test_sharpe_ratio :   1.234567

[INFO] Model saved to models/saved/lightgbm_30s.pkl
```