"""Fetch historical OHLCV data with a clean, unified interface."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:                                   # pragma: no cover
    yf = None

try:
    from alpha_vantage.timeseries import TimeSeries
except ImportError:                                   # pragma: no cover
    TimeSeries = None

from . import config


@dataclass
class PriceData:
    """Container for adjusted close prices."""
    prices: pd.DataFrame           # columns = tickers, index = dates
    returns: pd.DataFrame          # simple daily returns


def _clean(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Keep only the requested tickers, drop all-NaN columns, ffill gaps."""
    df = df.reindex(columns=tickers)
    df = df.ffill().dropna(how="all")
    return df


def load_yahoo(tickers: List[str],
               start: pd.Timestamp,
               end: pd.Timestamp) -> PriceData:
    if yf is None:
        raise ImportError("Install yfinance: pip install yfinance")
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,           # already split/dividend adjusted
        progress=False,
        threads=True,
    )["Close"]
    prices = _clean(data, tickers)
    rets = prices.pct_change().dropna()
    return PriceData(prices=prices, returns=rets)


def load_alphavantage(tickers: List[str], key: str) -> PriceData:
    if TimeSeries is None:
        raise ImportError("Install alpha-vantage: pip install alpha-vantage")
    ts = TimeSeries(key=key, output_format="pandas")
    frames = {}
    for t in tickers:
        df, _ = ts.get_daily_adjusted(symbol=t, outputsize="full")
        df = df.sort_index()["5. adjusted close"].rename(t).to_frame()
        frames[t] = df
        time.sleep(12)              # respect free-tier 5 req/min
    prices = pd.concat(frames.values(), axis=1).ffill().dropna()
    rets = prices.pct_change().dropna()
    return PriceData(prices=prices, returns=rets)


def load(tickers: List[str],
         start: pd.Timestamp = config.START_DATE,
         end:   pd.Timestamp = config.END_DATE) -> PriceData:
    if config.DATA_SOURCE == "alphavantage":
        return load_alphavantage(tickers, config.ALPHA_VANTAGE_KEY)
    return load_yahoo(tickers, start, end)
