import pandas as pd
import yfinance as yf
import functools


@functools.lru_cache()
def load_data(
    ticker: str = "BTC-USD", start: str = "2022-01-01", end: str = "2025-04-01"
) -> pd.DataFrame:
    """
    Download OHLCV data via yfinance and ensure no MultiIndex.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel("Ticker")
    df.index.name = "Date"
    return df.dropna()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute moving averages: 7, 25, 99 periods.
    """
    df = df.copy()
    df["SMA7"] = df["Close"].rolling(7).mean()
    df["SMA25"] = df["Close"].rolling(25).mean()
    df["SMA99"] = df["Close"].rolling(99).mean()
    return df.dropna()
