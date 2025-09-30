"""
Test script to verify installation and imports.
"""

import sys
from pathlib import Path

print("Testing predict package installation...\n")

# Test imports
errors = []

try:
    print("✓ Importing data modules...")
    from data.loader import DataLoader
    from data.processor import DataProcessor
    print("  - DataLoader: OK")
    print("  - DataProcessor: OK")
except Exception as e:
    errors.append(f"Data modules: {e}")
    print(f"  ✗ Error: {e}")

try:
    print("\n✓ Importing feature modules...")
    from features.orderbook import OrderBookFeatures
    from features.trade import TradeFeatures
    from features.technical import TechnicalFeatures
    from features.microstructure import MicrostructureFeatures
    print("  - OrderBookFeatures: OK")
    print("  - TradeFeatures: OK")
    print("  - TechnicalFeatures: OK")
    print("  - MicrostructureFeatures: OK")
except Exception as e:
    errors.append(f"Feature modules: {e}")
    print(f"  ✗ Error: {e}")

try:
    print("\n✓ Importing model modules...")
    from models.base import BaseModel
    from models.tree import LightGBMModel, XGBoostModel
    print("  - BaseModel: OK")
    print("  - LightGBMModel: OK")
    print("  - XGBoostModel: OK")
except Exception as e:
    errors.append(f"Model modules: {e}")
    print(f"  ✗ Error: {e}")

try:
    print("\n✓ Importing utility modules...")
    from utils.metrics import evaluate_predictions, calculate_ic
    print("  - Metrics: OK")
except Exception as e:
    errors.append(f"Utility modules: {e}")
    print(f"  ✗ Error: {e}")

try:
    print("\n✓ Testing dependencies...")
    import numpy as np
    import pandas as pd
    import yaml
    from loguru import logger
    print("  - numpy: OK")
    print("  - pandas: OK")
    print("  - yaml: OK")
    print("  - loguru: OK")

    try:
        import lightgbm as lgb
        print("  - lightgbm: OK")
    except ImportError:
        print("  - lightgbm: NOT INSTALLED (optional)")

    try:
        import xgboost as xgb
        print("  - xgboost: OK")
    except ImportError:
        print("  - xgboost: NOT INSTALLED (optional)")

    try:
        import torch
        print("  - torch: OK")
    except ImportError:
        print("  - torch: NOT INSTALLED (optional for deep learning)")

except Exception as e:
    errors.append(f"Dependencies: {e}")
    print(f"  ✗ Error: {e}")

# Summary
print("\n" + "="*60)
if errors:
    print("❌ Installation test FAILED")
    print("\nErrors:")
    for error in errors:
        print(f"  - {error}")
    print("\nPlease run: pip install -r requirements.txt")
    sys.exit(1)
else:
    print("✅ Installation test PASSED")
    print("\nAll modules imported successfully!")
    print("You can now run experiments with:")
    print("  python scripts/run_experiment.py --symbol btcusdt")
print("="*60)