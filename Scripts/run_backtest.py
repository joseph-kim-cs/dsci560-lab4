import pandas as pd

from backtest import BacktestConfig, moving_average_crossover_signals, run_portfolio_backtest, save_outputs


def load_prices(path: str = "data/stockprice.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").set_index("Date")


def build_signals(prices: pd.DataFrame, tickers):
    signals = {}
    for t in tickers:
        df_t = prices[[t]].rename(columns={t: "Close"}).dropna()
        signals[t] = moving_average_crossover_signals(df_t, short_window=20, long_window=50)
    return signals


if __name__ == "__main__":
    cfg = BacktestConfig(
        initial_cash_total=100_000.0,
        tickers=("NVDA", "AAPL", "MSFT"),
        fee_rate=0.0,
        slippage_rate=0.0
    )

    prices = load_prices("data/stockprice.csv")
    signals = build_signals(prices, cfg.tickers)

    results = run_portfolio_backtest(signals, cfg)
    save_outputs(results, out_dir="output")

    print("Portfolio Metrics:")
    print(results["portfolio_metrics"])
    print("\nPer-ticker Metrics:")
    print(results["per_ticker_metrics"])
    print("\nSaved outputs to ./output/")
