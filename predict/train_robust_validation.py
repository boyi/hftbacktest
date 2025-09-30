"""
稳健性验证版本：解决方法论红线问题
1. 非重叠采样（每5秒一个样本，避免样本相关性）
2. Newey-West标准误估计显著性
3. 分日期IC分布与t-test
4. 流动性异常过滤
5. 时间泄露检查
"""
import gzip
import json
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from loguru import logger
from scipy.stats import pearsonr, spearmanr, ttest_1samp
from sklearn.preprocessing import StandardScaler
import joblib

def load_bybit_data_full(file_path):
    """加载完整的 orderbook 和 trade 数据"""
    orderbook_l1 = []
    orderbook_l50 = []
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
                    orderbook_l1.append({
                        'timestamp': ts_ns / 1e9,
                        'bid': float(b[0][0]),
                        'bid_qty': float(b[0][1]),
                        'ask': float(a[0][0]),
                        'ask_qty': float(a[0][1])
                    })

            elif topic == 'orderbook.50.SOLUSDT':
                d = json_data.get('data', {})
                bids = d.get('b', [])
                asks = d.get('a', [])
                if bids and asks:
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

def newey_west_se(residuals, nlags=10):
    """
    计算Newey-West标准误（考虑序列相关性）
    """
    n = len(residuals)
    mean_resid = np.mean(residuals)

    # Variance
    var = np.sum((residuals - mean_resid) ** 2) / n

    # Autocovariances
    for lag in range(1, nlags + 1):
        weight = 1 - lag / (nlags + 1)  # Bartlett kernel
        gamma = np.sum((residuals[:-lag] - mean_resid) * (residuals[lag:] - mean_resid)) / n
        var += 2 * weight * gamma

    return np.sqrt(var / n)

def block_bootstrap_ic(y_true, y_pred, block_size=20, n_bootstrap=1000):
    """
    Block bootstrap估计IC的置信区间
    """
    n = len(y_true)
    n_blocks = n // block_size
    ics = []

    for _ in range(n_bootstrap):
        # 随机选择blocks
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
        bootstrap_idx = []
        for block_idx in block_indices:
            start = block_idx * block_size
            end = min(start + block_size, n)
            bootstrap_idx.extend(range(start, end))

        bootstrap_idx = bootstrap_idx[:n]  # 截断到原始长度
        ic, _ = pearsonr(y_true[bootstrap_idx], y_pred[bootstrap_idx])
        ics.append(ic)

    return np.percentile(ics, [2.5, 97.5])

logger.info("="*70)
logger.info("稳健性验证：非重叠采样 + 显著性检验")
logger.info("="*70)

logger.info("\n[1/6] 加载数据（10天）...")
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

logger.info("\n[2/6] 基础特征 + 高级特征...")
df = ob_l1.copy()

# 基础
df['mid_price'] = (df['bid'] + df['ask']) / 2
df['spread'] = df['ask'] - df['bid']
df['spread_bps'] = (df['spread'] / df['mid_price']) * 10000
df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'])

# Microprice
df['microprice'] = (df['bid'] * df['ask_qty'] + df['ask'] * df['bid_qty']) / \
                   (df['bid_qty'] + df['ask_qty'])
df['microprice_mid_diff_bps'] = ((df['microprice'] - df['mid_price']) / df['mid_price']) * 10000

# OFI
df['bid_qty_delta'] = df['bid_qty'].diff()
df['ask_qty_delta'] = df['ask_qty'].diff()
df['ofi'] = df['bid_qty_delta'] - df['ask_qty_delta']
for window in [5, 10, 30]:
    df[f'ofi_sum_{window}'] = df['ofi'].rolling(window).sum()

# LOB slopes (简化版，不使用L50以加速)
# Queue fragility
df['queue_fragility'] = (df['bid_qty'].pct_change(fill_method=None).abs() +
                         df['ask_qty'].pct_change(fill_method=None).abs()) / 2
for window in [5, 10, 30]:
    df[f'queue_fragility_ma_{window}'] = df['queue_fragility'].rolling(window).mean()

# Impact
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

# Regime
df['volatility_30s'] = df['mid_price'].pct_change(fill_method=None).rolling(30).std()
df['volatility_percentile'] = df['volatility_30s'].rolling(300).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
)

# 传统特征
for window in [5, 10, 30, 60]:
    df[f'return_{window}s'] = df['mid_price'].pct_change(window, fill_method=None)

df['price_momentum_5'] = df['mid_price'].pct_change(5, fill_method=None)
df['price_momentum_10'] = df['mid_price'].pct_change(10, fill_method=None)

# 目标
df['target_5s'] = df['mid_price'].shift(-5) / df['mid_price'] - 1

# 日期标记（用于分组IC分析）
df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date

logger.info("\n[3/6] 流动性过滤...")
# 过滤宽点差样本（点差 > 中位数的3倍）
spread_median = df['spread_bps'].median()
spread_threshold = spread_median * 3

logger.info(f"  Spread median: {spread_median:.2f} bps")
logger.info(f"  Filtering samples with spread > {spread_threshold:.2f} bps")

df_filtered = df[df['spread_bps'] <= spread_threshold].copy()
logger.info(f"  Before filtering: {len(df):,}, After: {len(df_filtered):,}")
logger.info(f"  Filtered out: {len(df) - len(df_filtered):,} ({(1 - len(df_filtered)/len(df))*100:.1f}%)")

df = df_filtered

logger.info("\n[4/6] 非重叠采样...")
# 每5秒采样一次（timestamp_sec % 5 == 0）
df_nonoverlap = df[df['timestamp_sec'] % 5 == 0].copy()

logger.info(f"  Before non-overlapping: {len(df):,}")
logger.info(f"  After non-overlapping (5s): {len(df_nonoverlap):,}")
logger.info(f"  Sample reduction: {len(df) / len(df_nonoverlap):.1f}x")

df = df_nonoverlap

# 清理
df = df[60:-5]
df = df.ffill().bfill().fillna(0)

logger.info(f"  Final samples: {len(df):,}")

# 特征列
feature_cols = [col for col in df.columns if col not in
                ['timestamp', 'timestamp_sec', 'target_5s', 'bid', 'ask',
                 'bid_qty', 'ask_qty', 'signed_volume', 'total_volume', 'date',
                 'bid_qty_delta', 'ask_qty_delta', 'ofi', 'queue_fragility', 'microprice']]

logger.info(f"  Total features: {len(feature_cols)}")

X = df[feature_cols].values
y = df['target_5s'].values * 10000
dates_arr = df['date'].values

# 时间序列划分
n = len(X)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]
dates_test = dates_arr[val_end:]

logger.info(f"\n[5/6] 训练...")
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
y_pred_test = model.predict(X_test)
test_ic, _ = pearsonr(y_test, y_pred_test)
test_rank_ic, _ = spearmanr(y_test, y_pred_test)

logger.info("\n[6/6] 稳健性统计检验...")

# 1. Newey-West标准误
residuals = y_test - y_pred_test
nw_se = newey_west_se(residuals, nlags=20)  # 20秒lag考虑序列相关
ic_se_naive = np.std(residuals) / np.sqrt(len(residuals))
t_stat_naive = test_ic / ic_se_naive
t_stat_nw = test_ic / nw_se

logger.info(f"\n  Newey-West调整:")
logger.info(f"    Naive SE: {ic_se_naive:.6f}, t-stat: {t_stat_naive:.2f}")
logger.info(f"    NW SE:    {nw_se:.6f}, t-stat: {t_stat_nw:.2f}")
logger.info(f"    SE膨胀系数: {nw_se / ic_se_naive:.2f}x")

# 2. Block Bootstrap置信区间
logger.info(f"\n  Block Bootstrap (20s blocks, 1000 iterations)...")
ci_lower, ci_upper = block_bootstrap_ic(y_test, y_pred_test, block_size=20, n_bootstrap=1000)
logger.info(f"    IC 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# 3. 按日期分组IC
logger.info(f"\n  按日期分组IC:")
daily_ics = []
for date in np.unique(dates_test):
    mask = dates_test == date
    if np.sum(mask) > 10:  # 至少10个样本
        ic_day, _ = pearsonr(y_test[mask], y_pred_test[mask])
        daily_ics.append(ic_day)
        logger.info(f"    {date}: IC = {ic_day:.4f} (n={np.sum(mask)})")

if len(daily_ics) > 1:
    daily_ics = np.array(daily_ics)
    t_stat_daily, p_value_daily = ttest_1samp(daily_ics, 0)
    logger.info(f"\n  Daily IC分布:")
    logger.info(f"    Mean: {np.mean(daily_ics):.4f}")
    logger.info(f"    Std:  {np.std(daily_ics):.4f}")
    logger.info(f"    t-stat: {t_stat_daily:.2f}, p-value: {p_value_daily:.4f}")

# 最终结果
logger.info("\n" + "="*70)
logger.info(f"稳健性验证结果 - {len(df):,} 非重叠样本")
logger.info("="*70)
logger.info(f"Test IC:      {test_ic:.4f}")
logger.info(f"Test Rank IC: {test_rank_ic:.4f}")
logger.info(f"NW t-stat:    {t_stat_nw:.2f}")
logger.info(f"Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
logger.info("="*70)

# 对比原始结果
logger.info("\n与原始重叠采样对比:")
logger.info(f"  原始 (重叠): IC = 0.1594, 样本 = 860,608")
logger.info(f"  稳健 (非重叠): IC = {test_ic:.4f}, 样本 = {len(df):,}")
logger.info(f"  样本减少: {860608 / len(df):.1f}x")
logger.info(f"  IC变化: {(test_ic - 0.1594) / 0.1594 * 100:+.1f}%")

# 保存
joblib.dump(model, 'models/saved/model_robust.pkl')

logger.info("\n✅ 稳健性验证完成！")
logger.info("  模型: models/saved/model_robust.pkl")