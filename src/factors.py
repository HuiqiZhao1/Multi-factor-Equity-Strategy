"""
factors.py
对每只股票每月计算因子得分（动量、价值、规模），并排名。
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_data():
    """读取价格和收益率数据。"""
    prices = pd.read_csv(
        PROJECT_ROOT / "data" / "raw" / "monthly_prices.csv",
        index_col=0, parse_dates=True,
    )
    returns = pd.read_csv(
        PROCESSED_DIR / "monthly_returns.csv",
        index_col=0, parse_dates=True,
    )
    return prices, returns


def compute_momentum(returns, lookback=11, skip=1):
    """
    动量因子：过去12个月累计收益，跳过最近1个月。

    对于每个月 t：
    - 看 t-12 到 t-2 这11个月的累计收益（跳过 t-1）
    - 累计收益 = (1+r1)*(1+r2)*...*(1+r11) - 1

    参数:
        lookback: 回看月数（默认11）
        skip: 跳过最近几个月（默认1）
    """
    momentum = pd.DataFrame(index=returns.index, columns=returns.columns)

    for i in range(lookback + skip, len(returns)):
        # 取 [i-lookback-skip, i-skip) 这段窗口
        window = returns.iloc[i - lookback - skip : i - skip]
        # 累计收益
        cumulative = (1 + window).prod() - 1
        momentum.iloc[i] = cumulative

    momentum = momentum.dropna(how="all").astype(float)
    print(f"动量因子: {momentum.shape[0]} 个月 × {momentum.shape[1]} 只股票")
    return momentum


def compute_size(prices):
    """
    规模因子：用价格作为市值的代理。

    理想情况下应该用 市值 = 股价 × 总股数，
    但 yfinance 的免费数据不容易拿到总股数，
    所以我们用股价作为近似（大盘股通常股价高）。
    这是一个简化，在文档中会注明。

    注意：规模因子是"小的好"，所以排名时要反转。
    """
    size = prices.copy()
    # 去掉没有动量数据的月份（保持对齐）
    print(f"规模因子: {size.shape[0]} 个月 × {size.shape[1]} 只股票")
    return size


def compute_value(prices):
    """
    价值因子：用 1/价格 作为账面市值比的代理。

    理想情况下应该用 账面价值/市值（B/M ratio），
    但财务报表数据不在 yfinance 免费接口中。
    用 1/价格 的逻辑：便宜的股票 = 价格低 = 1/价格高。
    这是一个简化，在文档中会注明。
    """
    value = 1.0 / prices
    print(f"价值因子: {value.shape[0]} 个月 × {value.shape[1]} 只股票")
    return value


def rank_stocks(factor_df, ascending=True):
    """
    每个月对所有股票按因子值排名。

    排名用百分位数（0到1之间），这样不同因子之间可以直接比较。
    ascending=True 表示因子值越大排名越靠前（动量、价值）。
    ascending=False 表示因子值越小排名越靠前（规模：小的好）。

    返回: 每只股票每月的排名分数（0到1之间，1=最好）
    """
    if ascending:
        ranked = factor_df.rank(axis=1, pct=True)
    else:
        ranked = factor_df.rank(axis=1, pct=True, ascending=False)
    return ranked


def compute_combined_score(returns, prices):
    """
    计算综合因子得分 = 动量排名 + 价值排名 + 规模排名 的等权平均。
    """
    # 计算三个因子的原始值
    momentum = compute_momentum(returns)
    size = compute_size(prices)
    value = compute_value(prices)

    # 对齐日期：用动量的日期范围（最短的）
    common_dates = momentum.index
    size = size.loc[size.index.isin(common_dates)]
    value = value.loc[value.index.isin(common_dates)]

    # 确保日期完全对齐
    common_dates = momentum.index.intersection(size.index).intersection(value.index)
    momentum = momentum.loc[common_dates]
    size = size.loc[common_dates]
    value = value.loc[common_dates]

    # 排名（百分位数）
    mom_rank = rank_stocks(momentum, ascending=True)    # 动量越高越好
    size_rank = rank_stocks(size, ascending=False)       # 市值越小越好
    value_rank = rank_stocks(value, ascending=True)      # 1/价格越高越好（越便宜越好）

    # 等权综合得分
    combined = (mom_rank + size_rank + value_rank) / 3

    print(f"\n综合得分矩阵: {combined.shape[0]} 个月 × {combined.shape[1]} 只股票")
    print(f"日期范围: {combined.index[0].strftime('%Y-%m')} → {combined.index[-1].strftime('%Y-%m')}")

    return combined, mom_rank, size_rank, value_rank


def save_scores(combined, mom_rank, size_rank, value_rank):
    """保存所有因子得分。"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    combined.to_csv(PROCESSED_DIR / "combined_scores.csv")
    mom_rank.to_csv(PROCESSED_DIR / "momentum_ranks.csv")
    size_rank.to_csv(PROCESSED_DIR / "size_ranks.csv")
    value_rank.to_csv(PROCESSED_DIR / "value_ranks.csv")

    print(f"\n所有因子得分已保存到 {PROCESSED_DIR}")


if __name__ == "__main__":
    try:
        prices, returns = load_data()
        combined, mom_rank, size_rank, value_rank = compute_combined_score(returns, prices)
        save_scores(combined, mom_rank, size_rank, value_rank)

        # 展示一个月的排名示例
        sample_date = combined.index[0]
        print(f"\n示例: {sample_date.strftime('%Y-%m')} 的综合得分排名")
        sample = combined.loc[sample_date].sort_values(ascending=False)
        print("Top 5 (做多候选):")
        print(sample.head().round(4))
        print("\nBottom 5 (做空候选):")
        print(sample.tail().round(4))

        print("\n完成！")
    except Exception as e:
        print(f"出错了: {e}")
        import traceback
        traceback.print_exc()