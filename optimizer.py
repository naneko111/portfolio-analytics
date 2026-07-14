"""Mean-variance / max-Sharpe optimizer (extends the original Solver)."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config


def neg_sharpe(weights, returns, rf):
    port = returns @ weights
    excess = port - rf / config.TRADING_DAYS
    return -excess.mean() / excess.std(ddof=1) * np.sqrt(config.TRADING_DAYS)


def max_sharpe_portfolio(returns: pd.DataFrame,
                         rf: float = config.RISK_FREE_RATE) -> dict:
    n = returns.shape[1]
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, 1)] * n                       # long-only
    w0 = np.ones(n) / n

    res = minimize(neg_sharpe, w0, args=(returns, rf),
                   method="SLSQP", bounds=bounds, constraints=cons)
    return {
        "weights": pd.Series(res.x, index=returns.columns).round(4),
        "sharpe":  -res.fun,
        "success": res.success,
    }


def efficient_frontier(returns: pd.DataFrame,
                       n_points: int = 50,
                       rf: float = config.RISK_FREE_RATE) -> pd.DataFrame:
    """Trace the long-only efficient frontier."""
    n = returns.shape[1]
    mu  = returns.mean() * config.TRADING_DAYS
    cov = returns.cov()  * config.TRADING_DAYS

    def port_vol(w): return np.sqrt(w @ cov @ w)

    target_returns = np.linspace(mu.min(), mu.max(), n_points)
    vols, rets, ws = [], [], []
    for tr in target_returns:
        cons = [
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "eq", "fun": lambda w: w @ mu - tr},
        ]
        res = minimize(port_vol, np.ones(n) / n, method="SLSQP",
                       bounds=[(0, 1)] * n, constraints=cons)
        if res.success:
            vols.append(res.fun); rets.append(tr); ws.append(res.x)
    return pd.DataFrame({"return": rets, "vol": vols,
                         "weights": ws}).sort_values("vol")
