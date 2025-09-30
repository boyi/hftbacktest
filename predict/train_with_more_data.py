"""
使用更多数据训练 - 10天数据
"""
import gzip
import json
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from loguru import logger
from scipy.stats import pearsonr, spearmanr
import joblib

def load_bybit_data(file_path):
    orderbook_data = []
    trade_data = []
    
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) != 2:
                continue
            
            ts_ns = int(parts[0])
            json_data = json.loads(parts[1])
            topic = json_data.get('topic', '')
            
            if topic == 'orderbook.1.SOLUSDT':
                d = json_data.get('data', {})
                b = d.get('b', [[]])
                a = d.get('a', [[]])
                if b and a and b[0] and a[0]:
                    orderbook_data.append({
                        'timestamp': ts_ns / 1e9,
                        'bid': float(b[0][0]),
                        'bid_qty': float(b[0][1]),
                        'ask': float(a[0][0]),
                        'ask_qty': float(a[0][1])
                    })
            elif topic == 'publicTrade.SOLUSDT':
                trades = json_data.get('data', [])
                for trade in trades:
                    trade_data.append({
                        'timestamp': ts_ns / 1e9,
                        'price': float(trade['p']),
                        'volume': float(trade['v']),
                        'is_buy': trade['S'] == 'Buy'
                    })
    
    return pd.DataFrame(orderbook_data), pd.DataFrame(trade_data)

logger.info("="*70)
logger.info("使用更多数据训练 - 10天")
logger.info("="*70)

logger.info("\n[1/4] 加载数据...")
all_ob, all_trade = [], []

# 使用 9月20-29日的数据
dates = [
    '20250920', '20250921', '20250922', '20250923', '20250924',
    '20250925', '20250926', '20250927', '20250928', '20250929'
]

for date in dates:
    file_path = f'../data/solusdt_{date}.gz'
    logger.info(f"  Loading {date}...")
    try:
        ob, trade = load_bybit_data(file_path)
        logger.info(f"    Orderbook: {len(ob):,}, Trades: {len(trade):,}")
        all_ob.append(ob)
        all_trade.append(trade)
    except FileNotFoundError:
        logger.warning(f"    File not found: {file_path}")
    except Exception as e:
        logger.error(f"    Error loading {file_path}: {e}")

logger.info(f"\n  Concatenating data...")
ob_df = pd.concat(all_ob, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
trade_df = pd.concat(all_trade, ignore_index=True).sort_values('timestamp').reset_index(drop=True)

logger.info(f"  Total orderbook: {len(ob_df):,}")
logger.info(f"  Total trades: {len(trade_df):,}")

# 聚合到1秒
ob_df['timestamp_sec'] = ob_df['timestamp'].astype(int)
ob_df = ob_df.groupby('timestamp_sec').last().reset_index()
logger.info(f"  After 1s aggregation: {len(ob_df):,} samples")

logger.info("\n[2/4] 特征工程...")

# 基础特征
ob_df['mid_price'] = (ob_df['bid'] + ob_df['ask']) / 2
ob_df['spread'] = ob_df['ask'] - ob_df['bid']
ob_df['spread_bps'] = (ob_df['spread'] / ob_df['mid_price']) * 10000
ob_df['bid_ask_imbalance'] = (ob_df['bid_qty'] - ob_df['ask_qty']) / (ob_df['bid_qty'] + ob_df['ask_qty'])

# 多时间窗口
for window in [5, 10, 30, 60]:
    ob_df[f'mid_price_ma_{window}'] = ob_df['mid_price'].rolling(window).mean()
    ob_df[f'spread_ma_{window}'] = ob_df['spread'].rolling(window).mean()
    ob_df[f'imbalance_ma_{window}'] = ob_df['bid_ask_imbalance'].rolling(window).mean()
    ob_df[f'return_{window}s'] = ob_df['mid_price'].pct_change(window, fill_method=None)

# 波动率
for window in [10, 30, 60]:
    ob_df[f'volatility_{window}s'] = ob_df['mid_price'].pct_change(fill_method=None).rolling(window).std()

# Trade 特征
logger.info("  Processing trade features...")
trade_df['timestamp_sec'] = trade_df['timestamp'].astype(int)
trade_agg = trade_df.groupby('timestamp_sec').agg({'volume': 'sum', 'price': 'mean'}).reset_index()
trade_agg.columns = ['timestamp_sec', 'trade_volume', 'trade_price']

trade_df['buy_volume'] = trade_df['volume'] * trade_df['is_buy']
trade_df['sell_volume'] = trade_df['volume'] * (~trade_df['is_buy'])
trade_flow = trade_df.groupby('timestamp_sec').agg({
    'buy_volume': 'sum', 'sell_volume': 'sum', 'is_buy': 'count'
}).reset_index()
trade_flow.columns = ['timestamp_sec', 'buy_volume', 'sell_volume', 'trade_count']
trade_flow['order_flow_imbalance'] = (trade_flow['buy_volume'] - trade_flow['sell_volume']) / \
                                       (trade_flow['buy_volume'] + trade_flow['sell_volume'] + 1e-8)

trade_features = pd.merge(trade_agg, trade_flow, on='timestamp_sec', how='outer')
ob_df = pd.merge(ob_df, trade_features, on='timestamp_sec', how='left')

# 填充缺失值
ob_df['trade_volume'] = ob_df['trade_volume'].fillna(0)
ob_df['trade_count'] = ob_df['trade_count'].fillna(0)
ob_df['order_flow_imbalance'] = ob_df['order_flow_imbalance'].fillna(0)
ob_df['trade_price'] = ob_df['trade_price'].ffill()

# Trade 滚动特征
for window in [5, 10, 30]:
    ob_df[f'trade_volume_ma_{window}'] = ob_df['trade_volume'].rolling(window).mean()
    ob_df[f'order_flow_ma_{window}'] = ob_df['order_flow_imbalance'].rolling(window).mean()

# 目标变量
ob_df['target_5s'] = ob_df['mid_price'].shift(-5) / ob_df['mid_price'] - 1

logger.info(f"  Before cleaning: {len(ob_df):,} samples")

# 只保留有效数据
ob_df = ob_df[60:-5]
logger.info(f"  After cleaning: {len(ob_df):,} samples")

# 填充剩余 NaN
ob_df = ob_df.ffill().bfill().fillna(0)

# 特征列
feature_cols = [col for col in ob_df.columns if col not in 
                ['timestamp', 'timestamp_sec', 'target_5s', 'bid', 'ask', 
                 'bid_qty', 'ask_qty', 'trade_price', 'buy_volume', 'sell_volume']]

logger.info(f"  Total features: {len(feature_cols)}")

X = ob_df[feature_cols].values
y = ob_df['target_5s'].values * 10000  # bps

# 划分数据集
# 前8天训练，第9天验证，第10天测试
n = len(X)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

logger.info(f"\n[3/4] 训练模型...")
logger.info(f"  Train: {len(X_train):,} samples")
logger.info(f"  Val:   {len(X_val):,} samples")
logger.info(f"  Test:  {len(X_test):,} samples")

# 使用更强的正则化
model = LGBMRegressor(
    objective='regression',
    n_estimators=500,
    max_depth=4,
    learning_rate=0.02,
    num_leaves=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=0.5,
    min_child_samples=200,
    verbose=-1,
    random_state=42
)

logger.info("  Training...")
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])

logger.info("\n[4/4] 评估...")

y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

train_ic = pearsonr(y_train, y_pred_train)[0]
val_ic = pearsonr(y_val, y_pred_val)[0]
test_ic = pearsonr(y_test, y_pred_test)[0]

train_rank_ic = spearmanr(y_train, y_pred_train)[0]
val_rank_ic = spearmanr(y_val, y_pred_val)[0]
test_rank_ic = spearmanr(y_test, y_pred_test)[0]

logger.info("\n" + "="*70)
logger.info(f"最终结果 - {len(ob_df):,} 样本 (10天数据)")
logger.info("="*70)
logger.info(f"{'Set':<10} {'IC':<12} {'Rank IC':<12}")
logger.info("-"*70)
logger.info(f"{'Train':<10} {train_ic:<12.4f} {train_rank_ic:<12.4f}")
logger.info(f"{'Val':<10} {val_ic:<12.4f} {val_rank_ic:<12.4f}")
logger.info(f"{'Test':<10} {test_ic:<12.4f} {test_rank_ic:<12.4f}")
logger.info("="*70)

# Feature importance
fi = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
fi = fi.sort_values('importance', ascending=False)

logger.info("\nTop 20 Features:")
for idx, row in fi.head(20).iterrows():
    logger.info(f"  {row['feature']:<35s}: {row['importance']:.1f}")

# Save
joblib.dump(model, 'models/saved/model_10days.pkl')
fi.to_csv('results/feature_importance_10days.csv', index=False)

logger.info("\n✅ 训练完成！")
logger.info(f"  模型已保存: models/saved/model_10days.pkl")
logger.info(f"  特征重要性: results/feature_importance_10days.csv")
