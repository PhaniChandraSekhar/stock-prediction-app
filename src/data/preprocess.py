"""
Data preprocessing and feature engineering for stock prediction.
This module provides functions to prepare stock data for machine learning models.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Stock price dataframe with columns 'Open', 'High', 'Low', 'Close', 'Volume'
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with added technical indicators
    """
    logger.info("Adding technical indicators...")
    
    # Make a copy to avoid modifying the original df
    df = df.copy()
    
    # Moving Averages
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA14'] = df['Close'].rolling(window=14).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD (Moving Average Convergence Divergence)
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['UpperBand'] = df['MA20'] + (df['Close'].rolling(window=20).std() * 2)
    df['LowerBand'] = df['MA20'] - (df['Close'].rolling(window=20).std() * 2)
    
    # Price Rate of Change
    df['ROC'] = df['Close'].pct_change(periods=5) * 100
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Daily Returns
    df['DailyReturn'] = df['Close'].pct_change()
    
    # Log Returns
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Volatility (standard deviation of returns)
    df['Volatility7d'] = df['LogReturn'].rolling(window=7).std() * np.sqrt(7)
    df['Volatility30d'] = df['LogReturn'].rolling(window=30).std() * np.sqrt(30)
    
    # Fill NaN values
    df = df.fillna(method='bfill')
    
    logger.info(f"Added {df.shape[1] - 5} technical indicators")
    return df

def prepare_target_variables(df, forecast_periods=[90, 180]):
    """
    Prepare target variables for different forecast horizons (3 months, 6 months).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Stock price dataframe
    forecast_periods : list, default [90, 180]
        Number of days to forecast (90 days = 3 months, 180 days = 6 months)
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with added target variables
    """
    logger.info(f"Preparing target variables for {forecast_periods} days ahead...")
    
    df = df.copy()
    
    for days in forecast_periods:
        # Future price
        df[f'Close_future_{days}d'] = df['Close'].shift(-days)
        
        # Price change (absolute)
        df[f'PriceChange_{days}d'] = df[f'Close_future_{days}d'] - df['Close']
        
        # Price change (percentage)
        df[f'PriceChangePct_{days}d'] = df[f'PriceChange_{days}d'] / df['Close'] * 100
        
        # Binary target: 1 if price goes up, 0 if down
        df[f'PriceUp_{days}d'] = (df[f'PriceChange_{days}d'] > 0).astype(int)
    
    # Drop future rows that don't have targets
    max_days = max(forecast_periods)
    df = df.iloc[:-max_days]
    
    logger.info(f"Added {len(forecast_periods)*4} target variables")
    return df

def create_sequences(df, target_col, sequence_length=60, forecast_horizon=1):
    """
    Create sequences for time series prediction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed stock data
    target_col : str
        Name of the target column
    sequence_length : int, default 60
        Number of time steps in each sequence (e.g., 60 days of data)
    forecast_horizon : int, default 1
        Number of days ahead to predict
        
    Returns:
    --------
    tuple
        (X, y) where X is the input sequences and y is the target values
    """
    logger.info(f"Creating sequences with length {sequence_length} for {target_col}")
    
    data = df.values
    X, y = [], []
    
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length + forecast_horizon - 1, df.columns.get_loc(target_col)])
    
    return np.array(X), np.array(y)

def scale_data(df, exclude_cols=None):
    """
    Scale the data using MinMaxScaler.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Stock price dataframe
    exclude_cols : list, default None
        List of columns to exclude from scaling
        
    Returns:
    --------
    tuple
        (scaled_df, scaler) where scaled_df is the scaled dataframe and scaler is the fitted scaler
    """
    logger.info("Scaling data...")
    
    if exclude_cols is None:
        exclude_cols = []
    
    # Identify columns to scale
    scale_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Create a copy of the dataframe
    df_scaled = df.copy()
    
    # Initialize scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit and transform the data
    df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])
    
    logger.info(f"Scaled {len(scale_cols)} columns")
    return df_scaled, scaler

def create_train_val_test_split(X, y, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split the data into training, validation, and test sets.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Features
    y : numpy.ndarray
        Targets
    val_size : float, default 0.15
        Proportion of the data to include in the validation set
    test_size : float, default 0.15
        Proportion of the data to include in the test set
    random_state : int, default 42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("Splitting data into train, validation, and test sets...")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    # Second split: separate validation set from the remaining data
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, shuffle=False
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_data_for_lstm(df, target_col, sequence_length=60, val_size=0.15, test_size=0.15):
    """
    Prepare data for LSTM model including scaling and sequence creation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed stock data
    target_col : str
        Name of the target column
    sequence_length : int, default 60
        Number of time steps in each sequence
    val_size : float, default 0.15
        Proportion of the data to include in the validation set
    test_size : float, default 0.15
        Proportion of the data to include in the test set
        
    Returns:
    --------
    dict
        Dictionary containing all the prepared data and transformers
    """
    logger.info(f"Preparing data for LSTM with target: {target_col}")
    
    # Make sure the target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    # Scale the data
    df_scaled, scaler = scale_data(df)
    
    # Create sequences
    X, y = create_sequences(df_scaled, target_col, sequence_length)
    
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(
        X, y, val_size, test_size
    )
    
    # Get the target column index
    target_idx = df_scaled.columns.get_loc(target_col)
    
    # Create a dictionary with all the prepared data
    prepared_data = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
        'target_idx': target_idx,
        'target_col': target_col,
        'feature_names': df.columns.tolist(),
        'sequence_length': sequence_length
    }
    
    logger.info("Data preparation complete")
    return prepared_data

if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from data.fetch import fetch_stock_data
    
    # Fetch some sample data
    df = fetch_stock_data('AAPL', period='2y')
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Prepare target variables
    df = prepare_target_variables(df)
    
    # Prepare data for LSTM
    prepared_data = prepare_data_for_lstm(df, 'Close_future_90d')
    
    print(f"Input shape: {prepared_data['X_train'].shape}")
    print(f"Target shape: {prepared_data['y_train'].shape}")