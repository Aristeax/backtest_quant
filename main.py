#!/usr/bin/env python3
"""
Main execution script for vectorized backtest and Backtesting.py sanity check.
"""

import pandas as pd
import matplotlib.pyplot as plt

from src.indicators import load_data, add_indicators
from src.signals import make_signals
from src.backtest_vector import vector_backtest
from src.strategies import SMASanityCheck
from src.utils import compute_metrics
from backtesting import Backtest


def main():
    # 1) Load & prepare data
    df = load_data()
    df_ind = add_indicators(df)
    df_sig = make_signals(df_ind)

    # 2) Run vectorized backtest
    df_v, trades_v = vector_backtest(df_sig)
    metrics_v = compute_metrics(df_v["Equity"])
    print("\n=== Vectorized Backtest Metrics ===")
    print(metrics_v)
    print("\n=== Vectorized Trades ===")
    print(trades_v)

    # 3) Run Backtesting.py engine
    price_df = df_sig[["Open", "High", "Low", "Close", "Volume"]].copy()
    bt = Backtest(
        price_df,
        SMASanityCheck,
        cash=500_000,
        commission=0.002,
        trade_on_close=False,  # next-bar open execution
    )
    stats = bt.run()
    metrics_bt = compute_metrics(stats["_equity_curve"]["Equity"])
    print("\n=== Backtesting.py Metrics ===")
    print(metrics_bt)

    # 4) Plot equity comparison
    plt.figure(figsize=(10, 5))
    plt.plot(df_v.index, df_v["Equity"], label="Vectorized")
    eq_bt = stats["_equity_curve"]["Equity"]
    plt.plot(eq_bt.index, eq_bt.values, label="Backtesting.py")
    plt.title("Equity Curve Comparison")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 5) Ensure output directories exist
    import os

    os.makedirs("outputs/equity", exist_ok=True)
    os.makedirs("outputs/trades", exist_ok=True)
    os.makedirs("outputs/summary", exist_ok=True)

    # 6) Save CSV outputs
    df_v[["Equity"]].to_csv("outputs/equity/vector_equity.csv")
    trades_v.to_csv("outputs/trades/vector_trades.csv", index=False)
    # Summary stats without nested objects
    pd.Series(stats.drop(labels=["_equity_curve", "_trades"])).to_csv(
        "outputs/summary/bt_summary_stats.csv"
    )
    stats["_equity_curve"].to_csv("outputs/equity/bt_equity.csv")
    stats["_trades"].to_csv("outputs/trades/bt_trades.csv", index=False)

    print("✅ All outputs saved under 'outputs/' directory.")


if __name__ == "__main__":
    main()
