# Price Prediction for HFT

Machine learning models for predicting short-term price movements in cryptocurrency futures markets using high-frequency data collected from the `collector` project.

## Overview

This project trains supervised learning models to predict future price changes (N seconds ahead) based on:
- Order book features (spread, depth, imbalance, etc.)
- Trade features (volume, buy/sell pressure, VWAP)
- Technical indicators (returns, volatility, momentum, RSI)
- Market microstructure features (order flow, price impact, realized spread)

## Project Structure

```
predict/
├── config/
│   └── config.yaml           # Configuration file
├── data/
│   ├── loader.py            # Load .gz files from collector
│   └── processor.py         # Data alignment and preprocessing
├── features/
│   ├── orderbook.py         # Order book features
│   ├── trade.py             # Trade-based features
│   ├── technical.py         # Technical indicators
│   └── microstructure.py    # Microstructure features
├── models/
│   ├── base.py              # Base model class
│   └── tree.py              # XGBoost/LightGBM models
├── training/
│   └── train.py             # Training script
├── utils/
│   └── metrics.py           # Evaluation metrics
├── scripts/
│   ├── prepare_dataset.py   # Prepare dataset from raw data
│   └── run_experiment.py    # Run complete experiment
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
cd predict
pip install -r requirements.txt
```

### 2. Configure

Edit `config/config.yaml` to set:
- Data paths (raw data from collector, output paths)
- Symbols to trade
- Date ranges for training/testing
- Feature engineering settings
- Model hyperparameters
- Prediction horizons (e.g., 5s, 10s, 30s, 60s)

## Usage

### Quick Start: Run Complete Experiment

```bash
# Run complete pipeline for a symbol
python scripts/run_experiment.py --symbol btcusdt --config config/config.yaml
```

This will:
1. Load raw data from collector
2. Align data streams
3. Compute all features
4. Add target variables
5. Train models for all configured horizons
6. Save models and results

### Step-by-Step Workflow

#### Step 1: Prepare Dataset

Convert raw collector data into ML-ready features:

```bash
python scripts/prepare_dataset.py \
    --config config/config.yaml \
    --symbol btcusdt \
    --start-date 2024-01-01 \
    --end-date 2024-06-30 \
    --output data/processed/btcusdt_train.parquet
```

This creates a dataset with:
- Aligned orderbook, trade, and ticker data
- Computed features from all feature modules
- Target variables (future returns at different horizons)

#### Step 2: Train Model

Train a prediction model:

```bash
python training/train.py \
    --config config/config.yaml \
    --data data/processed/btcusdt_train.parquet \
    --model lightgbm \
    --horizon 30
```

This will:
- Split data into train/validation/test sets (time-based split)
- Train the model with early stopping
- Evaluate on validation and test sets
- Save model, predictions, and metrics
- Display feature importance

### Using Prepared Data

If you already have a prepared dataset:

```bash
python scripts/run_experiment.py \
    --symbol btcusdt \
    --skip-prepare \
    --data-path data/processed/btcusdt_train.parquet
```

## Features

### Order Book Features
- Spread: bid-ask spread, relative spread, spread momentum
- Imbalance: bid/ask volume imbalance at multiple levels
- Depth: total depth at different levels, weighted depth
- Price levels: price range, order book slope
- Weighted features: volume-weighted mid price, order book VWAP

### Trade Features
- Volume: rolling volume statistics, volume momentum
- Direction: buy/sell volume ratio, order flow imbalance
- Trade intensity: trade frequency, average trade size

### Technical Indicators
- Returns: simple and log returns at multiple horizons
- Volatility: historical volatility, realized volatility
- Momentum: price momentum, rate of change
- RSI: Relative Strength Index

### Microstructure Features
- Order flow imbalance
- Price impact per unit volume
- Effective spread
- Realized spread

## Models

### Supported Models
- **LightGBM** (recommended): Fast, accurate tree-based model
- **XGBoost**: Alternative tree-based model

### Model Selection
LightGBM is recommended for:
- Fast training on large datasets
- Good handling of categorical features
- Built-in handling of missing values
- Excellent feature importance analysis

## Evaluation Metrics

### Regression Metrics
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination

### Trading Metrics
- **IC**: Information Coefficient (Pearson correlation)
- **Rank IC**: Rank Information Coefficient (Spearman correlation)
- **Direction Accuracy**: Accuracy of predicting up/down movement
- **Sharpe Ratio**: Sharpe ratio of a simple trading strategy

## Configuration

Key configuration sections in `config/config.yaml`:

### Data
```yaml
data:
  raw_data_path: "../data/collected"  # Path to collector .gz files
  symbols: ["btcusdt", "ethusdt"]
  sample_interval_ms: 100  # Sampling frequency
```

### Features
```yaml
features:
  orderbook:
    enabled: true
    depth_levels: 10
  trade:
    enabled: true
    windows: [5, 10, 30]  # seconds
```

### Targets
```yaml
target:
  horizons: [5, 10, 30, 60]  # Prediction horizons in seconds
  type: "return"  # Predict percentage return
```

### Model
```yaml
model:
  type: "lightgbm"
  lightgbm:
    n_estimators: 1000
    max_depth: 6
    learning_rate: 0.05
    early_stopping_rounds: 50
```

## Output

### Results Directory Structure
```
results/
├── predictions_lightgbm_30s.csv      # Predictions vs actual
├── metrics_lightgbm_30s.csv          # Evaluation metrics
└── feature_importance_lightgbm_30s.csv  # Feature importance

models/saved/
└── lightgbm_30s.pkl                  # Trained model
```

## Tips

### Data Quality
- Ensure collector data is clean and continuous
- Check for gaps in data collection
- Validate orderbook update sequences

### Feature Engineering
- Start with basic features, add complexity gradually
- Monitor feature importance to focus on useful features
- Consider market regime changes

### Model Training
- Use time-based splits (no shuffling!)
- Monitor validation metrics for overfitting
- Experiment with different horizons

### Hyperparameter Tuning
- Start with default config
- Tune learning_rate and n_estimators first
- Adjust max_depth and regularization parameters

## Troubleshooting

### Out of Memory
- Reduce `sample_interval_ms` (sample less frequently)
- Process data in smaller date ranges
- Reduce `depth_levels` in order book features

### Poor Performance
- Check data quality and alignment
- Verify feature distributions (no extreme values)
- Try different prediction horizons
- Increase training data size

### Slow Training
- Use LightGBM instead of XGBoost
- Reduce `n_estimators` or increase `learning_rate`
- Reduce number of features

## Next Steps

1. **Backtest**: Integrate predictions with HftBacktest simulator
2. **Live Trading**: Deploy model with connector for real-time prediction
3. **Ensemble**: Combine multiple models for better predictions
4. **Deep Learning**: Experiment with LSTM/Transformer models
5. **Multi-Asset**: Train models across multiple trading pairs

## References

- HftBacktest Documentation: https://hftbacktest.readthedocs.io/
- LightGBM: https://lightgbm.readthedocs.io/
- Market Microstructure: "Trading and Exchanges" by Larry Harris