"""
data_loader.py
从 yfinance 拉取 S&P 500 成分股的历史价格数据，计算月度收益率。
"""

import yfinance as yf
import pandas as pd
import os
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "META",
    "JPM", "BAC", "GS", "MS", "BRK-B",
    "JNJ", "UNH", "PFE", "ABBV", "MRK",
    "AMZN", "TSLA", "HD", "MCD", "NKE",
    "CAT", "BA", "GE", "HON", "UPS",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "DIS", "NFLX", "CMCSA", "T", "VZ",
    "PG", "KO", "PEP", "WMT", "COST",
    "NEE", "DUK", "SO", "D", "AEP",
    "LIN", "APD", "FCX", "AMT", "PLD",
]

START_DATE = "2015-01-01"
END_DATE = "2025-01-01"

# 用 Path 定位项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def download_prices():
    print(f"正在下载 {len(TICKERS)} 只股票的数据 ({START_DATE} ~ {END_DATE})...")

    data = yf.download(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
        interval="1mo",
        auto_adjust=True,
        progress=True,
    )

    prices = data["Close"]

    # 去掉多层列索引，只保留股票代码
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.droplevel(0)

    prices = prices.dropna(axis=1, how="all")
    print(f"成功下载 {prices.shape[1]} 只股票，共 {prices.shape[0]} 个月")
    return prices


def compute_returns(prices):
    returns = prices.pct_change().dropna()
    print(f"收益率矩阵: {returns.shape[0]} 个月 × {returns.shape[1]} 只股票")
    return returns


def save_data(prices, returns):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    prices.to_csv(RAW_DIR / "monthly_prices.csv")
    returns.to_csv(PROCESSED_DIR / "monthly_returns.csv")

    print(f"价格数据已保存: {RAW_DIR / 'monthly_prices.csv'}")
    print(f"收益率数据已保存: {PROCESSED_DIR / 'monthly_returns.csv'}")


if __name__ == "__main__":
    try:
        prices = download_prices()
        returns = compute_returns(prices)
        save_data(prices, returns)
        print("\n完成！数据已准备好。")
    except Exception as e:
        print(f"出错了: {e}")
        import traceback
        traceback.print_exc()