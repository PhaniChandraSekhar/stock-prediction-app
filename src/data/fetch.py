"""
Yahoo Finance data fetching utilities.
This module provides functions to download and manage stock data from Yahoo Finance.
"""

import os
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_stock_data(ticker_symbol, period='5y', interval='1d'):
    """
    Fetch historical stock data from Yahoo Finance.
    
    Parameters:
    -----------
    ticker_symbol : str
        The stock ticker symbol (e.g., 'AAPL' for Apple)
    period : str, default '5y'
        The time period to fetch data for. 
        Valid values: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
    interval : str, default '1d'
        The data interval.
        Valid values: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
    
    Returns:
    --------
    pandas.DataFrame
        Historical stock data with columns: Open, High, Low, Close, Volume, Dividends, Stock Splits
    """
    logger.info(f"Fetching data for {ticker_symbol} with period={period} and interval={interval}")
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            logger.warning(f"No data found for {ticker_symbol}")
            return None
            
        logger.info(f"Successfully fetched {len(df)} records for {ticker_symbol}")
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {ticker_symbol}: {str(e)}")
        raise

def fetch_multiple_stocks(ticker_list, period='5y', interval='1d'):
    """
    Fetch historical data for multiple stocks from Yahoo Finance.
    
    Parameters:
    -----------
    ticker_list : list of str
        List of stock ticker symbols
    period : str, default '5y'
        The time period to fetch data for
    interval : str, default '1d'
        The data interval
    
    Returns:
    --------
    dict
        Dictionary with ticker symbols as keys and pandas.DataFrame as values
    """
    logger.info(f"Fetching data for {len(ticker_list)} stocks")
    results = {}
    for ticker in ticker_list:
        try:
            df = fetch_stock_data(ticker, period, interval)
            if df is not None:
                results[ticker] = df
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {str(e)}")
    
    logger.info(f"Successfully fetched data for {len(results)}/{len(ticker_list)} stocks")
    return results

def save_stock_data(df, ticker_symbol, output_dir='data'):
    """
    Save stock data to CSV file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Stock data to save
    ticker_symbol : str
        Stock ticker symbol
    output_dir : str, default 'data'
        Directory to save the data
    
    Returns:
    --------
    str
        Path to the saved file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    timestamp = datetime.now().strftime('%Y%m%d')
    filename = f"{ticker_symbol}_{timestamp}.csv"
    file_path = os.path.join(output_dir, filename)
    
    df.to_csv(file_path)
    logger.info(f"Saved {ticker_symbol} data to {file_path}")
    return file_path

def get_stock_info(ticker_symbol):
    """
    Get basic company information for a ticker symbol.
    
    Parameters:
    -----------
    ticker_symbol : str
        Stock ticker symbol
    
    Returns:
    --------
    dict
        Company information including name, sector, industry, etc.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        return info
    except Exception as e:
        logger.error(f"Error fetching info for {ticker_symbol}: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"
    df = fetch_stock_data(ticker, period='2y')
    if df is not None:
        print(f"Fetched {len(df)} records for {ticker}")
        print(df.head())
        save_stock_data(df, ticker)
    
    # Multiple stocks example
    # tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    # data_dict = fetch_multiple_stocks(tickers, period='1y')
    # for tick, data in data_dict.items():
    #     save_stock_data(data, tick)