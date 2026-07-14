"""Portfolio container: weights + assets."""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from . import metrics, config


@dataclass
class Portfolio:
    prices:  pd.DataFrame            # adjusted close
    returns: pd.DataFrame            # daily simple returns
    weights: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.weights is None:
            self.weights = np.ones(len(self.returns.columns)) / len(self.returns.columns)
        else:
            assert abs(self.weights.sum() - 1) < 1e-6, "Weights must sum to 1"
            self.weights = np.asarray(self.weights, dtype=float)

    @property
    def tickers(self) -> list[str]:
        return list(self.returns.columns)

    @property
    def value(self) -> pd.Series:
        """Cumulative portfolio value starting at 1.0."""
        port_ret = self.returns @ self.weights
        return (1 + port_ret).cumprod()

    @property
    def daily_returns(self) -> pd.Series:
        return self.returns @ self.weights

    def summary(self) -> pd.DataFrame:
        """Per-asset + portfolio summary, all in a tidy DataFrame."""
        rows = []
        for t, r in self.returns.items():
            rows.append({"Asset": t, **metrics.summary(r, self.prices[t])})
        rows.append({"Asset": "PORTFOLIO", **metrics.summary(self.daily_returns, self.value)})
        return pd.DataFrame(rows).set_index("Asset").round(4)

    def correlation(self) -> pd.DataFrame:
        return metrics.correlation_matrix(self.returns)

    def rebalance(self, weights: np.ndarray) -> "Portfolio":
        return Portfolio(self.prices, self.returns, weights)
