# Portfolio Risk & Analytics Tool

A Python toolkit for portfolio risk analysis: pulls historical price data,
computes returns, volatility, Sharpe/Sortino ratios, VaR, CVaR, correlation
matrices, and visualizes them with publication-quality charts.

## Features

- 📈 **Data**: Yahoo Finance (default) or Alpha Vantage
- 📊 **Metrics**: Sharpe, Sortino, Calmar, Max Drawdown, CAGR
- 🛡️ **Risk**: Historical VaR, Parametric VaR, Monte-Carlo VaR, CVaR
- 🔗 **Diversification**: Correlation matrix + heatmap
- 🎯 **Optimization**: Max-Sharpe & Efficient Frontier (long-only)
- 📑 **Reports**: Self-contained HTML report

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/portfolio-analytics.git
cd portfolio-analytics
pip install -r requirements.txt
