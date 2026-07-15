"""End-to-end driver: fetch → analyze → visualize → report."""

from pathlib import Path

from . import config, data_loader, portfolio, metrics, optimizer, visualizer, reports


def main():
    out = Path(config.OUTPUT_DIR); out.mkdir(exist_ok=True)

    # 1) Load data
    print("Loading price data...")
    tickers = [t for t in config.TICKERS if t != config.BENCHMARK]
    data = data_loader.load(tickers)
    print(f"   {len(data.prices)} days, {len(tickers)} assets")

    # 2) Build a portfolio (equal-weight)
    port = portfolio.Portfolio(data.prices, data.returns)
    charts = {}

    # 3) Per-asset & portfolio summary
    summary_df = port.summary()
    reports.print_table(summary_df, "Risk / Return Summary")

    # 4) Visualizations
    visualizer.plot_prices(data.prices, out / "prices.png");     charts["prices"]   = out / "prices.png"
    visualizer.plot_correlation(port.correlation(), out / "corr.png"); charts["correlation"] = out / "corr.png"
    visualizer.plot_risk_return(summary_df, out / "risk_return.png"); charts["risk_return"] = out / "risk_return.png"
    visualizer.plot_drawdown(data.prices, out / "drawdown.png"); charts["drawdown"] = out / "drawdown.png"

    # 5) VaR analysis on the portfolio
    pr = port.daily_returns
    var  = metrics.historical_var(pr)
    cvar = metrics.cvar(pr)
    mcvar = metrics.monte_carlo_var(pr)
    print(f"\n=== VaR (95%, 1-day) ===")
    print(f"   Historical: {var:.2%}   CVaR: {cvar:.2%}   Monte-Carlo: {mcvar:.2%}")
    visualizer.plot_var(pr, var, cvar, out / "var.png"); charts["var"] = out / "var.png"

    # 6) Optimization (extends original Solver)
    opt = optimizer.max_sharpe_portfolio(data.returns)
    print("\n=== Max-Sharpe Portfolio ===")
    print(opt["weights"].to_string())
    print(f"Sharpe = {opt['sharpe']:.3f}")

    # 7) Efficient frontier
    fr = optimizer.efficient_frontier(data.returns)
    ms_vol = (opt["weights"].values @ data.returns.cov().values * 252
              @ opt["weights"].values) ** 0.5
    ms_ret = (opt["weights"].values @ data.returns.mean().values) * 252
    visualizer.plot_frontier(fr, (ms_vol, ms_ret), out / "frontier.png")
    charts["frontier"] = out / "frontier.png"
    visualizer.plot_weights(opt["weights"], out / "weights.png")
    charts["weights"] = out / "weights.png"

    # 8) Apply optimal weights & show optimized portfolio
    opt_port = port.rebalance(opt["weights"].values)
    opt_summary = opt_port.summary().iloc[[-1]]
    print("\n=== Optimized Portfolio Metrics ===")
    print(opt_summary.to_string())

    # 9) HTML report
    full_summary = pd.concat([summary_df, opt_summary])
    reports.build_html(full_summary, charts, out / "report.html")
    print(f"\nHTML report written to {out/'report.html'}")


if __name__ == "__main__":
    main()
