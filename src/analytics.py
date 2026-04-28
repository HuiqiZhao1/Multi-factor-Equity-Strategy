"""
analytics.py
统计验证：Fama-MacBeth 回归 + Newey-West 标准误。
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_data():
    """读取所有需要的数据。"""
    returns = pd.read_csv(
        PROCESSED_DIR / "monthly_returns.csv",
        index_col=0, parse_dates=True,
    )
    factors = pd.read_csv(
        PROCESSED_DIR / "ff_factors.csv",
        index_col=0, parse_dates=True,
    )
    backtest = pd.read_csv(
        PROCESSED_DIR / "backtest_results.csv",
        index_col=0, parse_dates=True,
    )
    mom_rank = pd.read_csv(
        PROCESSED_DIR / "momentum_ranks.csv",
        index_col=0, parse_dates=True,
    )
    size_rank = pd.read_csv(
        PROCESSED_DIR / "size_ranks.csv",
        index_col=0, parse_dates=True,
    )
    value_rank = pd.read_csv(
        PROCESSED_DIR / "value_ranks.csv",
        index_col=0, parse_dates=True,
    )
    return returns, factors, backtest, mom_rank, size_rank, value_rank


# ── 1. 四因子回归：检验策略的 alpha ──────────────────────

def four_factor_regression(backtest, factors):
    """
    把组合收益对四因子做时序回归，看 alpha 是否显著。

    模型: R_portfolio = alpha + b1*MktRF + b2*SMB + b3*HML + b4*WML + e

    如果 alpha 显著为正，说明策略有超出四因子的额外收益。
    用 Newey-West 标准误修正自相关和异方差。
    """
    print("=" * 50)
    print("  1. 四因子时序回归（Newey-West 修正）")
    print("=" * 50)

    # 对齐日期
    common = backtest.index.intersection(factors.index)
    port_ret = backtest.loc[common, "portfolio_return"]
    ff = factors.loc[common, ["Mkt-RF", "SMB", "HML", "WML"]]

    # 加常数项（截距 = alpha）
    X = sm.add_constant(ff)
    y = port_ret

    # OLS 回归，用 Newey-West 标准误（lag = 6个月）
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

    print(f"\n  观测数: {model.nobs:.0f}")
    print(f"  R²: {model.rsquared:.4f}")
    print(f"\n  {'变量':<10} {'系数':>10} {'t值':>10} {'p值':>10}")
    print("  " + "-" * 42)
    for name, coef, tval, pval in zip(
        model.params.index, model.params, model.tvalues, model.pvalues
    ):
        label = "alpha" if name == "const" else name
        sig = ""
        if pval < 0.01:
            sig = "***"
        elif pval < 0.05:
            sig = "**"
        elif pval < 0.10:
            sig = "*"
        print(f"  {label:<10} {coef:>10.4f} {tval:>10.2f} {pval:>10.4f} {sig}")

    print(f"\n  Alpha（月度）: {model.params['const']:.4f}")
    print(f"  Alpha（年化）: {model.params['const'] * 12:.4f}")

    if model.pvalues["const"] < 0.05:
        print("  → Alpha 在 5% 水平下显著！策略有超出因子的额外收益。")
    else:
        print("  → Alpha 不显著。策略收益可以被四因子解释。")

    return model


# ── 2. Fama-MacBeth 回归：检验因子是否定价 ─────────────────

def fama_macbeth(returns, factors, mom_rank, size_rank, value_rank):
    """
    Fama-MacBeth 两步回归：

    第一步（截面回归）：每个月 t，用股票的因子暴露来解释截面收益
        R_i,t = gamma0_t + gamma1_t * Mom_i + gamma2_t * Size_i + gamma3_t * Value_i + e_i,t

    第二步（时序平均）：对每个 gamma 取时间序列平均，用 t 检验看是否显著不为零

    如果 gamma 显著，说明该因子对截面收益有解释力。
    """
    print("\n" + "=" * 50)
    print("  2. Fama-MacBeth 截面回归")
    print("=" * 50)

    # 对齐所有数据的日期
    common_dates = (
        returns.index
        .intersection(mom_rank.index)
        .intersection(size_rank.index)
        .intersection(value_rank.index)
    )

    # 每个月跑一次截面回归
    gammas = []

    for date in common_dates:
        # 这个月每只股票的收益率
        y = returns.loc[date].dropna()

        # 这个月每只股票的因子排名
        mom = mom_rank.loc[date].reindex(y.index).dropna()
        size = size_rank.loc[date].reindex(y.index).dropna()
        val = value_rank.loc[date].reindex(y.index).dropna()

        # 取交集
        valid = y.index.intersection(mom.index).intersection(size.index).intersection(val.index)
        if len(valid) < 10:
            continue

        y_valid = y[valid]
        X_valid = pd.DataFrame({
            "Mom": mom[valid],
            "Size": size[valid],
            "Value": val[valid],
        })
        X_valid = sm.add_constant(X_valid)

        # 截面回归
        try:
            result = sm.OLS(y_valid, X_valid).fit()
            gammas.append(result.params)
        except Exception:
            continue

    gammas_df = pd.DataFrame(gammas)

    # 时序平均 + Newey-West t检验
    print(f"\n  截面回归月数: {len(gammas_df)}")
    print(f"\n  {'因子':<10} {'平均gamma':>12} {'t值':>10} {'p值':>10}")
    print("  " + "-" * 44)

    results_list = []

    for col in gammas_df.columns:
        series = gammas_df[col].dropna().values
        # Newey-West t检验
        nw_model = sm.OLS(series, np.ones(len(series))).fit(
            cov_type="HAC", cov_kwds={"maxlags": 6}
        )
        mean_gamma = float(nw_model.params[0])
        t_stat = float(nw_model.tvalues[0])
        p_val = float(nw_model.pvalues[0])

        label = "截距" if col == "const" else col
        sig = ""
        if p_val < 0.01:
            sig = "***"
        elif p_val < 0.05:
            sig = "**"
        elif p_val < 0.10:
            sig = "*"
        print(f"  {label:<10} {mean_gamma:>12.6f} {t_stat:>10.2f} {p_val:>10.4f} {sig}")

        results_list.append({
            "factor": label,
            "mean_gamma": mean_gamma,
            "t_stat": t_stat,
            "p_val": p_val,
        })

    print("\n  解读:")
    print("  - t值绝对值 > 1.96 → 在5%水平下显著")
    print("  - t值绝对值 > 2.58 → 在1%水平下显著")
    print("  - 显著的gamma说明该因子对股票截面收益有解释力")

    return pd.DataFrame(results_list)


# ── 3. 策略收益的 t 检验 ────────────────────────────────

def return_ttest(backtest, factors):
    """
    检验策略月度收益是否显著不为零。
    """
    print("\n" + "=" * 50)
    print("  3. 策略收益 t 检验（Newey-West）")
    print("=" * 50)

    common = backtest.index.intersection(factors.index)
    port_ret = backtest.loc[common, "portfolio_return"]
    rf = factors.loc[common, "RF"]
    excess = port_ret - rf

    # 检验超额收益是否显著不为零
    nw = sm.OLS(excess, np.ones(len(excess))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6}
    )

    print(f"\n  月均超额收益: {nw.params.iloc[0]:.4f} ({nw.params.iloc[0]*12:.4f} 年化)")
    print(f"  t值: {nw.tvalues.iloc[0]:.2f}")
    print(f"  p值: {nw.pvalues.iloc[0]:.4f}")

    if nw.pvalues.iloc[0] < 0.05:
        print("  → 策略超额收益在5%水平下显著不为零！")
    else:
        print("  → 策略超额收益不显著。")

    return nw


if __name__ == "__main__":
    try:
        returns, factors, backtest, mom_rank, size_rank, value_rank = load_data()

        # 1. 四因子回归
        model = four_factor_regression(backtest, factors)

        # 2. Fama-MacBeth
        fm_results = fama_macbeth(returns, factors, mom_rank, size_rank, value_rank)

        # 3. t检验
        ttest = return_ttest(backtest, factors)

        # 保存Fama-MacBeth结果
        fm_results.to_csv(PROCESSED_DIR / "fama_macbeth_results.csv", index=False)
        print(f"\nFama-MacBeth结果已保存到 {PROCESSED_DIR / 'fama_macbeth_results.csv'}")

        print("\n全部统计检验完成！")

    except Exception as e:
        print(f"出错了: {e}")
        import traceback
        traceback.print_exc()