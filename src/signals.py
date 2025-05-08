import pandas as pd

from src.indicators import add_indicators


def make_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate buy/sell signals based on SMA crossovers.
    Returns DataFrame with 'Signal' column: 1=buy, -1=sell, 0=flat.
    """
    df = add_indicators(df)
    buy = (
        (df["SMA7"] > df["SMA25"])
        & (df["SMA7"].shift() <= df["SMA25"].shift())
        & (df["SMA7"] > df["SMA99"])
    )
    sell = (
        (df["SMA7"] < df["SMA25"]) & (df["SMA7"].shift() >= df["SMA25"].shift())
    ) | (df["SMA7"] < df["SMA99"])
    df["Signal"] = 0
    df.loc[buy, "Signal"] = 1
    df.loc[sell, "Signal"] = -1
    return df
