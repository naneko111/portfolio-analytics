"""Central configuration for the analytics tool."""

from datetime import datetime, timedelta

# ---- Data source ----
DATA_SOURCE = "yahoo"          # "yahoo" or "alphavantage"
ALPHA_VANTAGE_KEY = "YOUR_KEY" # only needed if DATA_SOURCE = "alphavantage"

# ---- Universe ----
TICKERS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet
    "AMZN",   # Amazon
    "JPM",    # JP Morgan
    "GLD",    # Gold ETF
    "TLT",    # Long-duration Treasuries
    "SPY",    # S&P 500 (benchmark)
]

BENCHMARK = "SPY"

# ---- Time window ----
END_DATE   = datetime.today()
START_DATE = END_DATE - timedelta(days=5 * 365)  # 5 years

# ---- Risk parameters ----
RISK_FREE_RATE   = 0.045      # annualized, e.g. 3-month T-bill
TRADING_DAYS     = 252
VAR_CONFIDENCE   = 0.95
VAR_HORIZON_DAYS = 1          # 1-day VaR

# ---- Output ----
OUTPUT_DIR = "output"
