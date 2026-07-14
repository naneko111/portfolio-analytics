"""Publication-quality charts."""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from . import config

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (11, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11


# 1. Normalized price evolution
def plot_prices(prices: pd.DataFrame, out: str | None = None):
    norm = prices / prices.iloc[0] * 100
    ax = norm.plot(linewidth=1.5)
    ax.set_title("Normalized Prices (Start = 100)")
    ax.set_ylabel("Index level")
    ax.legend(loc="best", fontsize=8, ncol=2)
    if out: plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()


# 2. Correlation heatmap
def plot_correlation(corr: pd.DataFrame, out: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Correlation Matrix — Daily Returns")
    if out: plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()


# 3. Risk / Return scatter
def plot_risk_return(summary_df: pd.DataFrame, out: str | None = None):
    ax = summary_df.plot.scatter(x="Annual Vol", y="Annual Return",
                                 s=120, c="Sharpe", cmap="viridis",
                                 colorbar=True, figsize=(10, 6))
    for t, r in summary_df.iterrows():
        ax.annotate(t, (r["Annual Vol"], r["Annual Return"]),
                    xytext=(7, 4), textcoords="offset points", fontsize=9)
    ax.set_title("Risk vs Return (color = Sharpe ratio)")
    if out: plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()


# 4. Drawdown
def plot_drawdown(prices: pd.DataFrame, out: str | None = None):
    dd = (prices / prices.cummax() - 1) * 100
    dd.plot(linewidth=1.2)
    plt.title("Drawdown (%)")
    plt.ylabel("Drawdown")
    if out: plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()


# 5. VaR histogram with loss cones
def plot_var(returns: pd.Series, var_value: float,
             cvar_value: float, out: str | None = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(returns, bins=80, density=True, alpha=0.6, color="steelblue")
    # Fit normal
    mu, sigma = returns.mean(), returns.std(ddof=1)
    x = np.linspace(returns.min(), returns.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "k--", lw=1.5, label="Normal fit")
    ax.axvline(-var_value,  color="red",  lw=2,
               label=f"VaR 95% = {var_value:.2%}")
    ax.axvline(-cvar_value, color="darkred", lw=2, ls="--",
               label=f"CVaR 95% = {cvar_value:.2%}")
    ax.set_title("Return Distribution & Value-at-Risk")
    ax.set_xlabel("Daily return")
    ax.legend()
    if out: plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()


# 6. Efficient frontier
def plot_frontier(frontier: pd.DataFrame, max_sharpe_point: tuple[float, float],
                  out: str | None = None):
    plt.plot(frontier["vol"], frontier["return"], "b-", label="Efficient frontier")
    plt.scatter(*max_sharpe_point, s=200, c="red", marker="*",
                label="Max Sharpe", zorder=5)
    plt.xlabel("Annual Volatility")
    plt.ylabel("Annual Return")
    plt.title("Efficient Frontier (long-only)")
    plt.legend()
    if out: plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()


# 7. Portfolio weights pie
def plot_weights(weights: pd.Series, out: str | None = None):
    weights = weights[weights > 0]
    weights.plot.pie(autopct="%1.1f%%", figsize=(7, 7), startangle=90)
    plt.title("Portfolio Weights")
    plt.ylabel("")
    if out: plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
