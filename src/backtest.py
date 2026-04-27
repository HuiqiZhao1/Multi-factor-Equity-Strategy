"""
backtest.py
多空组合回测引擎：根据因子得分做多最好的、做空最差的，计算组合收益。
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_data():
    """读取因子得分和收益率数据。"""
    combined = pd.read_csv(
        PROCESSED_DIR / "combined_scores.csv",
        index_col=0, parse_dates=True,
    )
    returns = pd.read_csv(
        PROCESSED_DIR / "monthly_returns.csv",
        index_col=0, parse_dates=True,
    )
    factors = pd.read_csv(
        PROCESSED_DIR / "ff_factors.csv",
        index_col=0, parse_dates=True,
    )
    return combined, returns, factors


def backtest(combined, returns, n_long=10, n_short=10):
    """
    每月根据综合得分：
    - 做多得分最高的 n_long 只股票（等权）
    - 做空得分最低的 n_short 只股票（等权）
    - 组合收益 = 做多端平均收益 - 做空端平均收益

    参数:
        combined: 综合因子得分 DataFrame
        returns: 月度收益率 DataFrame
        n_long: 做多股票数量（默认10只，即top quintile of 50）
        n_short: 做空股票数量（默认10只）

    返回:
        results: DataFrame，包含每月的做多、做空、组合收益
    """
    # 对齐日期
    common_dates = combined.index.intersection(returns.index)
    combined = combined.loc[common_dates]
    returns = returns.loc[common_dates]

    records = []

    for i in range(len(combined) - 1):
        # 本月的因子得分用来决定持仓
        score_date = combined.index[i]
        # 下个月的收益用来计算盈亏
        return_date = combined.index[i + 1]

        scores = combined.iloc[i].dropna()
        month_returns = returns.loc[return_date]

        # 只保留两边都有数据的股票
        valid_stocks = scores.index.intersection(month_returns.dropna().index)
        scores = scores[valid_stocks]

        # 排序选股
        sorted_stocks = scores.sort_values(ascending=False)
        long_stocks = sorted_stocks.head(n_long).index    # 得分最高的
        short_stocks = sorted_stocks.tail(n_short).index   # 得分最低的

        # 等权平均收益
        long_return = month_returns[long_stocks].mean()
        short_return = month_returns[short_stocks].mean()

        # 多空组合收益 = 做多 - 做空
        portfolio_return = long_return - short_return

        records.append({
            "date": return_date,
            "long_return": long_return,
            "short_return": short_return,
            "portfolio_return": portfolio_return,
            "long_stocks": ", ".join(long_stocks[:5]),   # 记录前5只
            "short_stocks": ", ".join(short_stocks[:5]),
        })

    results = pd.DataFrame(records).set_index("date")
    return results


def compute_metrics(results, factors):
    """
    计算策略的绩效指标。
    """
    port = results["portfolio_return"]

    # 对齐因子数据，获取无风险利率
    common = port.index.intersection(factors.index)
    rf = factors.loc[common, "RF"]
    port_aligned = port.loc[common]

    # 基本统计
    total_months = len(port)
    annual_return = port.mean() * 12
    annual_vol = port.std() * np.sqrt(12)
    sharpe = (port.mean() - rf.mean()) / port.std() * np.sqrt(12)

    # 累计收益
    cumulative = (1 + port).cumprod()
    total_return = cumulative.iloc[-1] - 1

    # 最大回撤
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # 胜率
    win_rate = (port > 0).mean()

    # 做多端 vs 做空端
    long_annual = results["long_return"].mean() * 12
    short_annual = results["short_return"].mean() * 12

    metrics = {
        "总月数": total_months,
        "年化收益": f"{annual_return:.2%}",
        "年化波动率": f"{annual_vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "累计收益": f"{total_return:.2%}",
        "最大回撤": f"{max_drawdown:.2%}",
        "月胜率": f"{win_rate:.2%}",
        "做多端年化": f"{long_annual:.2%}",
        "做空端年化": f"{short_annual:.2%}",
    }

    return metrics, cumulative, drawdown


def save_results(results, cumulative):
    """保存回测结果。"""
    results.to_csv(PROCESSED_DIR / "backtest_results.csv")
    cumulative.to_csv(PROCESSED_DIR / "cumulative_returns.csv")
    print(f"回测结果已保存到 {PROCESSED_DIR}")


if __name__ == "__main__":
    try:
        # 加载数据
        combined, returns, factors = load_data()
        print(f"因子得分: {combined.shape}")
        print(f"收益率: {returns.shape}")

        # 运行回测
        print("\n正在运行回测...")
        results = backtest(combined, returns, n_long=10, n_short=10)

        # 计算指标
        metrics, cumulative, drawdown = compute_metrics(results, factors)

        # 打印结果
        print("\n" + "=" * 40)
        print("  回测绩效报告")
        print("=" * 40)
        for key, val in metrics.items():
            print(f"  {key}: {val}")
        print("=" * 40)

        # 展示前几个月的持仓
        print("\n前3个月的持仓:")
        for i in range(3):
            row = results.iloc[i]
            print(f"\n{results.index[i].strftime('%Y-%m')}:")
            print(f"  做多: {row['long_stocks']}")
            print(f"  做空: {row['short_stocks']}")
            print(f"  组合收益: {row['portfolio_return']:.2%}")

        # 保存
        save_results(results, cumulative)
        print("\n完成！")

    except Exception as e:
        print(f"出错了: {e}")
        import traceback
        traceback.print_exc()