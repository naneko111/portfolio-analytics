"""Risk & performance metrics, all in one place."""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

from . import config


# ---------- helpers ----------
def annualize_return(daily_returns: pd.Series) -> float:
    return (1 + daily_returns).prod() ** (config.TRADING_DAYS / len(daily_returns)) - 1


def annualize_vol(daily_returns: pd.Series) -> float:
    return daily_returns.std(ddof=1) * np.sqrt(config.TRADING_DAYS)


# ---------- return / risk ----------
def total_return(prices: pd.Series) -> float:
    return prices.iloc[-1] / prices.iloc[0] - 1


def cagr(prices: pd.Series) -> float:
    days = (prices.index[-1] - prices.index[0]).days
    return (prices.iloc[-1] / prices.iloc[0]) ** (365.0 / days) - 1


def max_drawdown(prices: pd.Series) -> float:
    peak = prices.cummax()
    dd = prices / peak - 1
    return dd.min()


# ---------- risk-adjusted ----------
def sharpe_ratio(daily_returns: pd.Series,
                 rf: float = config.RISK_FREE_RATE) -> float:
    excess = daily_returns - rf / config.TRADING_DAYS
    return excess.mean() / excess.std(ddof=1) * np.sqrt(config.TRADING_DAYS)


def sortino_ratio(daily_returns: pd.Series,
                  rf: float = config.RISK_FREE_RATE) -> float:
    excess = daily_returns - rf / config.TRADING_DAYS
    downside = excess[excess < 0]
    dd = np.sqrt((downside ** 2).mean()) * np.sqrt(config.TRADING_DAYS)
    return excess.mean() * config.TRADING_DAYS / dd


def calmar_ratio(prices: pd.Series) -> float:
    return cagr(prices) / abs(max_drawdown(prices)) if max_drawdown(prices) != 0 else np.nan


# ---------- VaR ----------
def historical_var(returns: pd.Series, alpha: float = config.VAR_CONFIDENCE) -> float:
    """Historical (non-parametric) VaR as a positive loss number."""
    return -np.quantile(returns, 1 - alpha)


def parametric_var(returns: pd.Series, alpha: float = config.VAR_CONFIDENCE) -> float:
    mu, sigma = returns.mean(), returns.std(ddof=1)
    return -(mu + stats.norm.ppf(1 - alpha) * sigma)


def cvar(returns: pd.Series, alpha: float = config.VAR_CONFIDENCE) -> float:
    """Expected Shortfall / CVaR — average loss beyond VaR."""
    var_threshold = -historical_var(returns, alpha)
    return -returns[returns <= var_threshold].mean()


# ---------- Monte-Carlo VaR (parametric) ----------
def monte_carlo_var(returns: pd.Series,
                    horizon: int = config.VAR_HORIZON_DAYS,
                    n_sims:  int   = 10_000,
                    alpha:   float = config.VAR_CONFIDENCE) -> float:
    mu, sigma = returns.mean(), returns.std(ddof=1)
    sims = np.random.normal(mu * horizon, sigma * np.sqrt(horizon), n_sims)
    return -np.quantile(sims, 1 - alpha)


# ---------- correlation ----------
def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


# ---------- all-in-one summary ----------
def summary(returns: pd.Series, prices: pd.Series | None = None) -> dict:
    out = {
        "Annual Return":    annualize_return(returns),
        "Annual Vol":       annualize_vol(returns),
        "Sharpe":           sharpe_ratio(returns),
        "Sortino":          sortino_ratio(returns),
        "VaR (95%, 1d)":    historical_var(returns),
        "CVaR (95%, 1d)":   cvar(returns),
    }
    if prices is not None:
        out["CAGR"]       = cagr(prices)
        out["Max Drawdown"] = max_drawdown(prices)
        out["Calmar"]     = calmar_ratio(prices)
    return out
