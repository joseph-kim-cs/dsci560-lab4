# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# %%
def moving_average_crossover(data, short_window=20, long_window=50, column='Close'):
    """
    Parameters:
    data (pd.DataFrame): DataFrame containing stock prices with a 'Close' column.
    short_window (int): The period for the short moving average.
    long_window (int): The period for the long moving average.
    
    Returns:
    pd.DataFrame: DataFrame with signals for buy and sell.
    """
    
    data = data.copy() # create a copy to prevent modifying the original DataFrame

    data["SMA_short"] = (data[column].rolling(window=short_window, min_periods=1).mean())
    data["SMA_long"] = (data[column].rolling(window=long_window, min_periods=1).mean())

    data["Signal"] = 0.0

    data.loc[data["SMA_short"] > data["SMA_long"], "Signal"] = 1.0
    data.loc[data["SMA_short"] < data["SMA_long"], "Signal"] = -1.0

    # this line identifies the points where the signal changes from 0 to 1 (buy) or from 0 to -1 (sell)
    data["Position"] = data["Signal"].diff()

    return data


# %%
# helper function: creates time-series sequences for LSTM input
def create_sequences(data, window_size):
    '''
    Parameters:
    data (pd DataFrame column, or array): Array of stock closing stock prices.
    '''
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])

    return np.array(X), np.array(y)



# %%
def lstm_price_prediction(data, column = "Close", window_size=60, epochs=10, batch_size=32):
    """
    Trains an LSTM model to predict stock closing prices.

    Parameters:
    df: Stock DataFrame with a 'Close' column.
    price_col: Column name for closing price.
    window_size: Number of past days used for prediction.
    epochs: Training epochs.
    batch_size: Batch size for training.
    """

    # Scale prices to [0,1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(data[[column]])

    # Create sequences
    X, y = create_sequences(scaled_prices, window_size)

    # Reshape for LSTM: (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train model
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    # Generate predictions
    predictions = model.predict(X)

    # Inverse scaling
    predictions = scaler.inverse_transform(predictions)
    y_true = scaler.inverse_transform(y.reshape(-1, 1))

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(y_true, predictions))

    return predictions, rmse



# %%
df_prices = pd.read_csv("../data/stockprice.csv", parse_dates=["Date"])
df_prices = df_prices.sort_values("Date").set_index("Date")

ticker = "AAPL" # choose the stock ticker you want to analyze from data/stockprice.csv (e.g., "AAPL" for Apple Inc.)

df = df_prices[[ticker]].rename(columns={ticker: "Close"}).dropna()

# %%
signals_df = moving_average_crossover(
    df,
    short_window=20,
    long_window=50
)

print("Moving Average Crossover (last 5 rows):")
print(signals_df[["Close", "SMA_short", "SMA_long", "Signal", "Position"]].tail())


# %%
preds, rmse = lstm_price_prediction(
    df,
    column="Close",
    window_size=60,
    epochs=10,
    batch_size=32
)

print(f"\nLSTM RMSE: {rmse:.4f}")

# align predictions back to dates (preds start after `window_size`)
lstm_df = df.iloc[60:].copy()
lstm_df["LSTM_Prediction"] = preds

print("\nLSTM predictions (last 5 rows):")
print(lstm_df.tail())

