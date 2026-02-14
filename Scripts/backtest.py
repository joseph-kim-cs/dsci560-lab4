import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

def moving_average_crossover_signals(df: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> pd.DataFrame:
    """
    Creates SMA crossover signals on a DataFrame with column 'Close'.
    Signal =  1 when SMA_short > SMA_long
    Signal = -1 when SMA_short < SMA_long
    Position = Signal.diff() -> buy/sell events
    """
    data = df.copy()
    data["SMA_short"] = data["Close"].rolling(window=short_window, min_periods=1).mean()
    data["SMA_long"] = data["Close"].rolling(window=long_window, min_periods=1).mean()

    data["Signal"] = 0.0
    data.loc[data["SMA_short"] > data["SMA_long"], "Signal"] = 1.0
    data.loc[data["SMA_short"] < data["SMA_long"], "Signal"] = -1.0

    data["Position"] = data["Signal"].diff()
    return data

@dataclass
class BacktestConfig:
    initial_cash_total: float = 100_000.0
    tickers: Tuple[str, ...] = ("NVDA", "AMD", "AAPL")
    fee_rate: float = 0.0
    slippage_rate: float = 0.0
    periods_per_year: int = 252


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def compute_returns(values: pd.Series) -> pd.Series:
    return values.pct_change().dropna()


def annualized_return(daily_rets: pd.Series, periods_per_year: int = 252) -> float:
    if len(daily_rets) == 0:
        return 0.0
    compounded = (1 + daily_rets).prod()
    return compounded ** (periods_per_year / len(daily_rets)) - 1


def sharpe_ratio(daily_rets: pd.Series, periods_per_year: int = 252, risk_free_annual: float = 0.0) -> float:
    if len(daily_rets) < 2:
        return 0.0
    rf_daily = (1 + risk_free_annual) ** (1 / periods_per_year) - 1
    excess = daily_rets - rf_daily
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return (excess.mean() / std) * np.sqrt(periods_per_year)


def backtest_long_only(
    signals_df: pd.DataFrame,
    initial_cash: float,
    price_col: str = "Close",
    position_col: str = "Position",
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Long-only backtest using crossover events:
      Position ==  2  -> BUY
      Position == -2  -> SELL

    Tracks daily cash, shares, and portfolio value.
    """
    df = signals_df.copy().dropna(subset=[price_col]).sort_index()

    cash = float(initial_cash)
    shares = 0.0
    equity_rows = []
    trades = []

    for dt, row in df.iterrows():
        price = float(row[price_col])
        pos = row.get(position_col, np.nan)

        buy_price = price * (1 + slippage_rate)
        sell_price = price * (1 - slippage_rate)

        # BUY (all-in for this asset allocation)
        if pos == 2 and shares == 0 and cash > 0:
            fee = cash * fee_rate
            spendable = cash - fee
            shares = spendable / buy_price
            cash = 0.0
            trades.append({"Date": dt, "Action": "BUY", "ExecPrice": buy_price, "Shares": shares, "Fee": fee})

        # SELL (liquidate)
        elif pos == -2 and shares > 0:
            gross = shares * sell_price
            fee = gross * fee_rate
            cash = gross - fee
            trades.append({"Date": dt, "Action": "SELL", "ExecPrice": sell_price, "Shares": shares, "Fee": fee})
            shares = 0.0

        portfolio_value = cash + shares * price
        equity_rows.append({"Date": dt, "Cash": cash, "Shares": shares, "Close": price, "PortfolioValue": portfolio_value})

    equity_df = pd.DataFrame(equity_rows).set_index("Date")
    trades_df = pd.DataFrame(trades)
    return equity_df, trades_df


def run_portfolio_backtest(
    signals_by_ticker: Dict[str, pd.DataFrame],
    cfg: BacktestConfig,
) -> Dict[str, object]:
    """
    Portfolio = equal allocation across tickers.
    Each ticker is traded independently using the same trading rule.
    """
    n = len(cfg.tickers)
    cash_per = cfg.initial_cash_total / n

    equity_curves = {}
    trade_logs = {}
    per_ticker_metrics = {}

    for t in cfg.tickers:
        eq, tr = backtest_long_only(
            signals_by_ticker[t],
            initial_cash=cash_per,
            fee_rate=cfg.fee_rate,
            slippage_rate=cfg.slippage_rate,
        )
        equity_curves[t] = eq
        trade_logs[t] = tr

        rets = compute_returns(eq["PortfolioValue"])
        per_ticker_metrics[t] = {
            "final_value": float(eq["PortfolioValue"].iloc[-1]),
            "total_return": float(eq["PortfolioValue"].iloc[-1] / eq["PortfolioValue"].iloc[0] - 1),
            "annualized_return": float(annualized_return(rets, cfg.periods_per_year)),
            "sharpe_ratio": float(sharpe_ratio(rets, cfg.periods_per_year)),
            "num_trades": int(len(tr)),
        }

    # Aggregate portfolio curve
    pv = pd.concat([equity_curves[t]["PortfolioValue"].rename(t) for t in cfg.tickers], axis=1).dropna()
    pv["TotalPortfolioValue"] = pv.sum(axis=1)

    port_rets = compute_returns(pv["TotalPortfolioValue"])
    portfolio_metrics = {
        "final_value": float(pv["TotalPortfolioValue"].iloc[-1]),
        "total_return": float(pv["TotalPortfolioValue"].iloc[-1] / pv["TotalPortfolioValue"].iloc[0] - 1),
        "annualized_return": float(annualized_return(port_rets, cfg.periods_per_year)),
        "sharpe_ratio": float(sharpe_ratio(port_rets, cfg.periods_per_year)),
    }

    return {
        "equity_curves": equity_curves,
        "trade_logs": trade_logs,
        "portfolio_curve": pv,
        "per_ticker_metrics": per_ticker_metrics,
        "portfolio_metrics": portfolio_metrics,
    }


def save_outputs(results: Dict[str, object], out_dir: str = "output") -> None:
    out = _ensure_dir(out_dir)

    results["portfolio_curve"].to_csv(out / "portfolio_equity_curve.csv")

    for t, eq in results["equity_curves"].items():
        eq.to_csv(out / f"{t}_equity_curve.csv")

    all_trades = []
    for t, tr in results["trade_logs"].items():
        if len(tr) > 0:
            tmp = tr.copy()
            tmp["Ticker"] = t
            all_trades.append(tmp)
    trades_df = pd.concat(all_trades, axis=0) if all_trades else pd.DataFrame(
        columns=["Date", "Action", "ExecPrice", "Shares", "Fee", "Ticker"]
    )
    trades_df.to_csv(out / "trades.csv", index=False)

    metrics = {"per_ticker": results["per_ticker_metrics"], "portfolio": results["portfolio_metrics"]}
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
