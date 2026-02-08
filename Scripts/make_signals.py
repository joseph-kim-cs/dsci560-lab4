import pandas as pd

from stock_algos import moving_average_crossover, lstm_price_prediction
from signals import sma_position_for_backtest, lstm_position_for_backtest

TICKERS = ["NVDA", "AMD", "AAPL"]

# SMA parameters
SHORT_W = 20
LONG_W = 50

# LSTM parameters
WINDOW_SIZE = 60
EPOCHS = 10
BATCH_SIZE = 32
THRESHOLD = 0.005

def load_one_ticker(df_prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df_prices[[ticker]].rename(columns={ticker: "Close"}).dropna()
    return df

def main():
    df_prices = pd.read_csv("../data/stockprice.csv", parse_dates=["Date"])
    df_prices = df_prices.sort_values("Date").set_index("Date")

    for ticker in TICKERS:
        df = load_one_ticker(df_prices, ticker)

        sma_df = moving_average_crossover(
            df,
            short_window=SHORT_W,
            long_window=LONG_W,
            column="Close"
        )

        sma_df["Position"] = sma_position_for_backtest(sma_df)

        sma_out = sma_df[["Close", "Position"]].copy()
        sma_out.to_csv(f"../data/sma_{ticker}.csv")

        preds, rmse = lstm_price_prediction(
            df,
            column="Close",
            window_size=WINDOW_SIZE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )

        lstm_df = df.iloc[WINDOW_SIZE:].copy()
        lstm_df["LSTM_Prediction"] = preds.flatten()
        lstm_df["Position"] = lstm_position_for_backtest(lstm_df, threshold=THRESHOLD)

        lstm_out = lstm_df[["Close", "LSTM_Prediction", "Position"]].copy()
        lstm_out.to_csv(f"../data/lstm_{ticker}.csv")

        print(f"{ticker}: done (LSTM RMSE={rmse:.4f})")

    print("Saved per-ticker signal files:")
    print(" - ../data/sma_<TICKER>.csv")
    print(" - ../data/lstm_<TICKER>.csv")

if __name__ == "__main__":
    main()
