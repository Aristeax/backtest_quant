import pandas as pd


def vector_backtest(
    df: pd.DataFrame, initial_cash: float = 500_000, commission: float = 0.002
) -> (pd.DataFrame, pd.DataFrame):
    """
    Simulate bar-by-bar PnL using precomputed 'Signal'.
    Returns:
      - df (with 'ExecOpen', 'Equity')
      - trades_v (entry/exit log DataFrame)
    """
    df = df.copy()
    cash = initial_cash
    position = 0.0
    equity = []
    trades = []

    df["ExecOpen"] = df["Open"].shift(-1)
    df = df.dropna(subset=["ExecOpen"])

    for idx, row in df.iterrows():
        sig = row["Signal"]
        price = row["ExecOpen"]

        # ENTRY
        if sig == 1 and position == 0:
            position = cash / price
            cost = position * price
            fee = cost * commission
            cash -= cost + fee
            trades.append({"EntryTime": idx, "EntryPrice": price, "FeeEntry": fee})

        # EXIT
        elif sig == -1 and position > 0:
            proceeds = position * price
            fee = proceeds * commission
            cash += proceeds - fee
            trades[-1].update({"ExitTime": idx, "ExitPrice": price, "FeeExit": fee})
            position = 0.0

        equity.append(cash + position * row["Close"])

    df["Equity"] = pd.Series(equity, index=df.index)
    trades_v = pd.DataFrame(trades)
    trades_v["PnL"] = trades_v["ExitPrice"] - trades_v["EntryPrice"]

    return df, trades_v
