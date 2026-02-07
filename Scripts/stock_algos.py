import pandas as pd
import numpy as np

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

