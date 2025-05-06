import pandas as pd
import numpy as np


def compute_metrics(equity: pd.Series) -> pd.Series:
    """
    Calculate CAGR, annual vol, Sharpe, and max drawdown.
    """
    returns = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    dd = (equity / equity.cummax() - 1).min()

    return pd.Series({"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": dd})
