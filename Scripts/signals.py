import pandas as pd

def sma_position_for_backtest(signals_df: pd.DataFrame) -> pd.Series:
    """
      Position =  2 -> BUY
      Position = -2 -> SELL
      else          -> HOLD

    signals_df is output of moving_average_crossover() and must contain 'Position'.
    We convert whatever diff values exist into the required 2/-2/0 format.
    """
    pos = signals_df["Position"].fillna(0.0)

    out = pd.Series(0, index=signals_df.index, dtype=int)
    out[pos > 0] = 2
    out[pos < 0] = -2
    return out


def lstm_position_for_backtest(lstm_df: pd.DataFrame, threshold: float = 0.005) -> pd.Series:
    """
    Convert LSTM predictions into 2/-2/0 Position signal.
      pred > close*(1+threshold) ->  2
      pred < close*(1-threshold) -> -2
      else                       ->  0

    lstm_df must have columns: Close, LSTM_Prediction
    """
    close = lstm_df["Close"]
    pred = lstm_df["LSTM_Prediction"]

    out = pd.Series(0, index=lstm_df.index, dtype=int)
    out[pred > close * (1.0 + threshold)] = 2
    out[pred < close * (1.0 - threshold)] = -2
    return out
