"""
高级微观结构特征 + 10天数据
结合之前的两个最佳实践：
1. 高级特征（Microprice, OFI, LOB slope, Queue fragility, Impact response, Regime）
2. 10天数据（20250920-20250929）
"""
import gzip
import json
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from loguru import logger
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
import joblib

def load_bybit_data_full(file_path):
    """加载完整的 orderbook 和 trade 数据"""
    orderbook_l1 = []  # Level 1
    orderbook_l50 = []  # Level 50 (for depth analysis)
    trade_data = []

    with gzip.open(file_path, 'rt') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) != 2:
                continue

            ts_ns = int(parts[0])
            json_data = json.loads(parts[1])
            topic = json_data.get('topic', '')

            # Level 1 (best bid/ask)
            if topic == 'orderbook.1.SOLUSDT':
                d = json_data.get('data', {})
                b = d.get('b', [[]])
                a = d.get('a', [[]])
                if b and a and b[0] and a[0]:
                    orderbook_l1.append({
                        'timestamp': ts_ns / 1e9,
                        'bid': float(b[0][0]),
                        'bid_qty': float(b[0][1]),
                        'ask': float(a[0][0]),
                        'ask_qty': float(a[0][1])
                    })

            # Level 50 (for depth features)
            elif topic == 'orderbook.50.SOLUSDT':
                d = json_data.get('data', {})
                bids = d.get('b', [])
                asks = d.get('a', [])
                if bids and asks:
                    # 提取前5档
                    bid_prices = [float(b[0]) for b in bids[:5]]
                    bid_volumes = [float(b[1]) for b in bids[:5]]
                    ask_prices = [float(a[0]) for a in asks[:5]]
                    ask_volumes = [float(a[1]) for a in asks[:5]]

                    orderbook_l50.append({
                        'timestamp': ts_ns / 1e9,
                        'bid_prices': bid_prices,
                        'bid_volumes': bid_volumes,
                        'ask_prices': ask_prices,
                        'ask_volumes': ask_volumes
                    })

            # Trades
            elif topic == 'publicTrade.SOLUSDT':
                trades = json_data.get('data', [])
                for trade in trades:
                    trade_data.append({
                        'timestamp': ts_ns / 1e9,
                        'price': float(trade['p']),
                        'volume': float(trade['v']),
                        'is_buy': trade['S'] == 'Buy'
                    })

    return pd.DataFrame(orderbook_l1), pd.DataFrame(orderbook_l50), pd.DataFrame(trade_data)

logger.info("="*70)
logger.info("高级微观结构特征 + 10天数据训练")
logger.info("="*70)

logger.info("\n[1/5] 加载数据（10天）...")
all_ob_l1, all_ob_l50, all_trade = [], [], []

dates = ['20250920', '20250921', '20250922', '20250923', '20250924',
         '20250925', '20250926', '20250927', '20250928', '20250929']

for date in dates:
    file_path = f'../data/solusdt_{date}.gz'
    logger.info(f"  Loading {date}...")
    ob_l1, ob_l50, trade = load_bybit_data_full(file_path)
    logger.info(f"    L1: {len(ob_l1):,}, L50: {len(ob_l50):,}, Trades: {len(trade):,}")
    all_ob_l1.append(ob_l1)
    all_ob_l50.append(ob_l50)
    all_trade.append(trade)

ob_l1 = pd.concat(all_ob_l1, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
ob_l50 = pd.concat(all_ob_l50, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
trade_df = pd.concat(all_trade, ignore_index=True).sort_values('timestamp').reset_index(drop=True)

logger.info(f"  Total: L1={len(ob_l1):,}, L50={len(ob_l50):,}, Trades={len(trade_df):,}")

# 聚合到1秒
ob_l1['timestamp_sec'] = ob_l1['timestamp'].astype(int)
ob_l1 = ob_l1.groupby('timestamp_sec').last().reset_index()

ob_l50['timestamp_sec'] = ob_l50['timestamp'].astype(int)
ob_l50 = ob_l50.groupby('timestamp_sec').last().reset_index()

logger.info(f"  After 1s aggregation: {len(ob_l1):,} samples")

logger.info("\n[2/5] 基础特征...")
df = ob_l1.copy()

# 基础
df['mid_price'] = (df['bid'] + df['ask']) / 2
df['spread'] = df['ask'] - df['bid']
df['spread_bps'] = (df['spread'] / df['mid_price']) * 10000
df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'])

logger.info("\n[3/5] 高级微观结构特征...")

# ========================================
# 1. Microprice / Weighted Mid
# ========================================
logger.info("  [1] Microprice (深度加权中间价)")
df['microprice'] = (df['bid'] * df['ask_qty'] + df['ask'] * df['bid_qty']) / \
                   (df['bid_qty'] + df['ask_qty'])
df['microprice_mid_diff'] = df['microprice'] - df['mid_price']
df['microprice_mid_diff_bps'] = (df['microprice_mid_diff'] / df['mid_price']) * 10000

# ========================================
# 2. OFI (Order Flow Imbalance)
# ========================================
logger.info("  [2] OFI (Order Flow Imbalance)")
df['bid_qty_delta'] = df['bid_qty'].diff()
df['ask_qty_delta'] = df['ask_qty'].diff()
df['ofi'] = df['bid_qty_delta'] - df['ask_qty_delta']

# 累积 OFI
for window in [5, 10, 30]:
    df[f'ofi_sum_{window}'] = df['ofi'].rolling(window).sum()
    df[f'ofi_mean_{window}'] = df['ofi'].rolling(window).mean()

# ========================================
# 3. LOB 斜率/梯度（使用 L50 数据）
# ========================================
logger.info("  [3] LOB 斜率/梯度")
ob_l50_processed = []
for idx, row in ob_l50.iterrows():
    if len(row['bid_prices']) >= 3 and len(row['ask_prices']) >= 3:
        # Bid 斜率：(累积量) / (价格差)
        bid_cum_vol = sum(row['bid_volumes'][:3])
        bid_price_range = row['bid_prices'][0] - row['bid_prices'][2]
        bid_slope = bid_cum_vol / (bid_price_range + 1e-8)

        # Ask 斜率
        ask_cum_vol = sum(row['ask_volumes'][:3])
        ask_price_range = row['ask_prices'][2] - row['ask_prices'][0]
        ask_slope = ask_cum_vol / (ask_price_range + 1e-8)

        ob_l50_processed.append({
            'timestamp_sec': row['timestamp_sec'],
            'lob_bid_slope': bid_slope,
            'lob_ask_slope': ask_slope,
            'lob_slope_imbalance': (bid_slope - ask_slope) / (bid_slope + ask_slope + 1e-8)
        })

lob_features = pd.DataFrame(ob_l50_processed)
df = pd.merge(df, lob_features, on='timestamp_sec', how='left')

# ========================================
# 4. 取消率/刷新率（队列脆弱度）
# ========================================
logger.info("  [4] 取消率/刷新率")
df['bid_qty_change_rate'] = df['bid_qty'].pct_change(fill_method=None)
df['ask_qty_change_rate'] = df['ask_qty'].pct_change(fill_method=None)

df['queue_fragility_bid'] = df['bid_qty_change_rate'].abs()
df['queue_fragility_ask'] = df['ask_qty_change_rate'].abs()
df['queue_fragility'] = (df['queue_fragility_bid'] + df['queue_fragility_ask']) / 2

for window in [5, 10, 30]:
    df[f'queue_fragility_ma_{window}'] = df['queue_fragility'].rolling(window).mean()

# ========================================
# 5. 冲击响应（主动单净额与未来价格的交互）
# ========================================
logger.info("  [5] 冲击响应")
trade_df['timestamp_sec'] = trade_df['timestamp'].astype(int)
trade_df['signed_volume'] = trade_df['volume'] * (2 * trade_df['is_buy'] - 1)

trade_impact = trade_df.groupby('timestamp_sec').agg({
    'signed_volume': 'sum',
    'volume': 'sum'
}).reset_index()
trade_impact.columns = ['timestamp_sec', 'signed_volume', 'total_volume']

df = pd.merge(df, trade_impact, on='timestamp_sec', how='left')
df['signed_volume'] = df['signed_volume'].fillna(0)
df['total_volume'] = df['total_volume'].fillna(0)

for window in [3, 5, 10]:
    df[f'impact_signed_vol_{window}'] = df['signed_volume'].rolling(window).sum()
    df[f'impact_ratio_{window}'] = df[f'impact_signed_vol_{window}'] / \
                                     (df['total_volume'].rolling(window).sum() + 1e-8)

df['price_momentum_5'] = df['mid_price'].pct_change(5, fill_method=None)
df['price_momentum_10'] = df['mid_price'].pct_change(10, fill_method=None)

# ========================================
# 6. Regime 特征（波动状态机）
# ========================================
logger.info("  [6] Regime 特征（波动状态）")
df['volatility_30s'] = df['mid_price'].pct_change(fill_method=None).rolling(30).std()

df['volatility_percentile'] = df['volatility_30s'].rolling(300).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
)

df['regime'] = pd.cut(df['volatility_percentile'],
                       bins=[0, 0.33, 0.67, 1.0],
                       labels=[0, 1, 2])
df['regime'] = df['regime'].astype(float)

df['regime_duration'] = (df['regime'] == df['regime'].shift()).astype(int).groupby(
    (df['regime'] != df['regime'].shift()).cumsum()
).cumsum()

logger.info("\n[4/5] 传统特征...")
# 多时间窗口
for window in [5, 10, 30, 60]:
    df[f'return_{window}s'] = df['mid_price'].pct_change(window, fill_method=None)

# 目标
df['target_5s'] = df['mid_price'].shift(-5) / df['mid_price'] - 1

logger.info(f"  Before cleaning: {len(df):,}")
df = df[60:-5]
logger.info(f"  After cleaning: {len(df):,}")

# 填充 NaN
df = df.ffill().bfill().fillna(0)

# 特征列
feature_cols = [col for col in df.columns if col not in
                ['timestamp', 'timestamp_sec', 'target_5s', 'bid', 'ask',
                 'bid_qty', 'ask_qty', 'bid_prices', 'bid_volumes',
                 'ask_prices', 'ask_volumes', 'signed_volume', 'total_volume']]

logger.info(f"  Total features: {len(feature_cols)}")
logger.info(f"  Advanced features: {len([c for c in feature_cols if any(x in c for x in ['microprice', 'ofi', 'lob', 'fragility', 'impact', 'regime'])])}")

X = df[feature_cols].values
y = df['target_5s'].values * 10000

# 数据集划分
n = len(X)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

logger.info(f"\n[5/5] 训练...")
logger.info(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

model = LGBMRegressor(
    n_estimators=400,
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

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])

# 评估
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

train_ic = pearsonr(y_train, y_pred_train)[0]
train_rank_ic = spearmanr(y_train, y_pred_train)[0]
val_ic = pearsonr(y_val, y_pred_val)[0]
val_rank_ic = spearmanr(y_val, y_pred_val)[0]
test_ic = pearsonr(y_test, y_pred_test)[0]
test_rank_ic = spearmanr(y_test, y_pred_test)[0]

logger.info("\n" + "="*70)
logger.info(f"高级特征模型结果 - {len(df):,} 样本 (10天数据)")
logger.info("="*70)
logger.info("Set        IC           Rank IC     ")
logger.info("-"*70)
logger.info(f"Train      {train_ic:.4f}       {train_rank_ic:.4f}      ")
logger.info(f"Val        {val_ic:.4f}       {val_rank_ic:.4f}      ")
logger.info(f"Test       {test_ic:.4f}       {test_rank_ic:.4f}      ")
logger.info("="*70)

# Top features
fi = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
fi = fi.sort_values('importance', ascending=False)

logger.info("\nTop 30 Features:")
for idx, row in fi.head(30).iterrows():
    is_advanced = any(x in row['feature'] for x in
                     ['microprice', 'ofi', 'lob', 'fragility', 'impact', 'regime'])
    marker = "🆕" if is_advanced else "   "
    logger.info(f"{marker} {row['feature']:<40s}: {row['importance']:.1f}")

# 统计高级特征的贡献
advanced_features = [f for f in feature_cols if any(x in f for x in
                    ['microprice', 'ofi', 'lob', 'fragility', 'impact', 'regime'])]
advanced_importance = fi[fi['feature'].isin(advanced_features)]['importance'].sum()
total_importance = fi['importance'].sum()

logger.info(f"\n高级特征统计:")
logger.info(f"  数量: {len(advanced_features)} / {len(feature_cols)}")
logger.info(f"  重要性占比: {advanced_importance / total_importance * 100:.1f}%")

# 与基线对比
logger.info("\n" + "="*70)
logger.info("对比基线模型（10天数据，传统特征）:")
logger.info("-"*70)
logger.info("基线  Test IC: 0.1591, Rank IC: 0.1936")
logger.info(f"高级  Test IC: {test_ic:.4f}, Rank IC: {test_rank_ic:.4f}")
logger.info(f"提升  Test IC: {(test_ic - 0.1591):.4f}, Rank IC: {(test_rank_ic - 0.1936):.4f}")
logger.info("="*70)

# 保存
joblib.dump(model, 'models/saved/model_advanced_10days.pkl')
fi.to_csv('results/feature_importance_advanced_10days.csv', index=False)

logger.info("\n✅ 训练完成！")
logger.info("  模型: models/saved/model_advanced_10days.pkl")
logger.info("  特征重要性: results/feature_importance_advanced_10days.csv")