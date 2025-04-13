"""
Script for making predictions with trained stock models.
This module provides functions to load models and make predictions.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Add the project root directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, project_root)

from src.data.fetch import fetch_stock_data
from src.data.preprocess import add_technical_indicators, scale_data, create_sequences
from src.model.lstm import StockLSTM
from src.utils.visualization import plot_forecast

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, "logs", "prediction.log"), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        os.path.join(project_root, "logs"),
        os.path.join(project_root, "outputs", "predictions")
    ]
    
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Make predictions with trained stock models")
    
    parser.add_argument("--ticker", type=str, required=True, 
                        help="Stock ticker symbol")
    parser.add_argument("--model_path", type=str, 
                        help="Path to the trained model file")
    parser.add_argument("--metadata_path", type=str, 
                        help="Path to the model metadata file")
    parser.add_argument("--scaler_path", type=str, 
                        help="Path to the scaler file")
    parser.add_argument("--forecast_horizon", type=int, default=90, 
                        help="Forecast horizon in days (default: 90)")
    parser.add_argument("--period", type=str, default="1y", 
                        help="Historical data period (default: 1y)")
    parser.add_argument("--interval", type=str, default="1d", 
                        help="Data interval (default: 1d)")
    parser.add_argument("--output_format", type=str, choices=["csv", "json", "html"], default="html",
                        help="Output format for predictions (default: html)")
    parser.add_argument("--sequence_length", type=int, default=60, 
                        help="Sequence length for LSTM model (default: 60)")
    
    return parser.parse_args()

def load_model_and_metadata(args):
    """
    Load the model and associated metadata.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    tuple
        (model, metadata)
    """
    if args.model_path:
        model_path = args.model_path
    else:
        # Try to find a model file for the specified ticker and forecast horizon
        model_dir = os.path.join(project_root, "models")
        model_files = [
            f for f in os.listdir(model_dir) 
            if f.startswith(f"{args.ticker}_") and f.endswith(".keras")
        ]
        
        if not model_files:
            raise FileNotFoundError(
                f"No model file found for {args.ticker}. "
                f"Please specify a model file using --model_path."
            )
        
        # Use the most recent model file
        model_files.sort(reverse=True)
        model_path = os.path.join(model_dir, model_files[0])
    
    # Determine metadata path
    if args.metadata_path:
        metadata_path = args.metadata_path
    else:
        # Try to find a metadata file
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        metadata_path = os.path.join(
            os.path.dirname(model_path),
            f"{model_name}_metadata.joblib"
        )
        
        if not os.path.exists(metadata_path):
            metadata_path = None
    
    # Determine scaler path
    if args.scaler_path:
        scaler_path = args.scaler_path
    else:
        # Try to find a scaler file
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        scaler_path = os.path.join(
            os.path.dirname(model_path),
            f"{model_name}_scaler.joblib"
        )
        
        if not os.path.exists(scaler_path):
            scaler_path = None
    
    # Load the model
    logger.info(f"Loading model from {model_path}")
    model = StockLSTM.load(model_path, metadata_path, scaler_path)
    
    return model

def prepare_data_for_prediction(ticker, period, interval, sequence_length):
    """
    Prepare data for making predictions.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    period : str
        Historical data period
    interval : str
        Data interval
    sequence_length : int
        Sequence length for LSTM model
        
    Returns:
    --------
    tuple
        (df, last_sequence, scaler)
    """
    # Fetch the data
    logger.info(f"Fetching data for {ticker}...")
    df = fetch_stock_data(ticker, period=period, interval=interval)
    
    if df is None or df.empty:
        raise ValueError(f"Failed to fetch data for {ticker}")
    
    logger.info(f"Fetched {len(df)} records for {ticker}")
    
    # Preprocess the data
    logger.info("Adding technical indicators...")
    df = add_technical_indicators(df)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Scale the data
    df_scaled, scaler = scale_data(df)
    
    # Create the last sequence for prediction
    X = df_scaled.values
    last_sequence = X[-sequence_length:].reshape(1, sequence_length, X.shape[1])
    
    return df, last_sequence, scaler

def make_prediction(model, last_sequence, forecast_horizon, scaler=None):
    """
    Make a forecast using the trained model.
    
    Parameters:
    -----------
    model : StockLSTM
        Trained model
    last_sequence : numpy.ndarray
        Last observed sequence
    forecast_horizon : int
        Forecast horizon in days
    scaler : sklearn.preprocessing.MinMaxScaler, optional
        Scaler used to transform the data
        
    Returns:
    --------
    numpy.ndarray
        Forecasted values
    """
    logger.info(f"Generating {forecast_horizon}-day forecast")
    
    # Generate forecast
    forecasts = model.forecast(last_sequence, steps=forecast_horizon, scaler=scaler)
    
    return forecasts

def save_predictions(df, forecasts, ticker, forecast_horizon, output_format, output_dir=None):
    """
    Save the predictions to file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Historical data
    forecasts : numpy.ndarray
        Forecasted values
    ticker : str
        Stock ticker symbol
    forecast_horizon : int
        Forecast horizon in days
    output_format : str
        Output format (csv, json, or html)
    output_dir : str, optional
        Output directory
        
    Returns:
    --------
    str
        Path to the saved file
    """
    if output_dir is None:
        output_dir = os.path.join(project_root, "outputs", "predictions")
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate dates for the forecast
    last_date = df.index[-1]
    forecast_dates = pd.date_range(
        start=last_date, 
        periods=forecast_horizon+1
    )[1:]  # Exclude the first date (which is the last historical date)
    
    # Create a dataframe with the forecasts
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Predicted_Close': forecasts
    })
    forecast_df.set_index('Date', inplace=True)
    
    # Save the forecasts
    if output_format == 'csv':
        output_path = os.path.join(
            output_dir, 
            f"{ticker}_{forecast_horizon}d_forecast_{timestamp}.csv"
        )
        forecast_df.to_csv(output_path)
    
    elif output_format == 'json':
        output_path = os.path.join(
            output_dir, 
            f"{ticker}_{forecast_horizon}d_forecast_{timestamp}.json"
        )
        forecast_df.to_json(output_path, orient='index')
    
    elif output_format == 'html':
        # Create a plotly figure
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Plot the forecast
        fig = plot_forecast(
            df['Close'][-100:],  # Last 100 days of historical data
            forecasts,
            forecast_dates,
            ticker_symbol=ticker
        )
        
        # Save the figure
        output_path = os.path.join(
            output_dir, 
            f"{ticker}_{forecast_horizon}d_forecast_{timestamp}.html"
        )
        fig.write_html(output_path)
    
    logger.info(f"Saved forecast to {output_path}")
    
    return output_path

def main():
    """Main function to run the prediction process."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create necessary directories
    create_directories()
    
    # Log the prediction run
    logger.info(f"Starting prediction run for {args.ticker}")
    logger.info(f"Command line arguments: {args}")
    
    try:
        # Load the model and metadata
        model = load_model_and_metadata(args)
        
        # Prepare data for prediction
        df, last_sequence, scaler = prepare_data_for_prediction(
            args.ticker, 
            args.period, 
            args.interval, 
            args.sequence_length
        )
        
        # Make the prediction
        forecasts = make_prediction(
            model, 
            last_sequence, 
            args.forecast_horizon, 
            scaler
        )
        
        # Log the forecast
        logger.info(f"Forecast for next {args.forecast_horizon} days: {forecasts[0]:.2f} to {forecasts[-1]:.2f}")
        
        # Save the predictions
        output_path = save_predictions(
            df, 
            forecasts, 
            args.ticker, 
            args.forecast_horizon, 
            args.output_format
        )
        
        # Print the path to the saved file
        print(f"Forecast saved to: {output_path}")
        
    except Exception as e:
        logger.exception(f"Error during prediction: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()