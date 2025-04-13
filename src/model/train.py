"""
Script for training stock prediction models.
This module provides a command-line interface for training models on stock data.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Add the project root directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, project_root)

from src.data.fetch import fetch_stock_data, save_stock_data
from src.data.preprocess import (
    add_technical_indicators, 
    prepare_target_variables, 
    prepare_data_for_lstm
)
from src.model.baseline import (
    LastValuePredictor, 
    MeanValuePredictor, 
    LinearRegressionModel, 
    RandomForestModel,
    evaluate_model,
    compare_models
)
from src.model.lstm import create_and_train_lstm_model
from src.utils.visualization import (
    plot_stock_price_history,
    plot_technical_indicators,
    plot_prediction_vs_actual,
    plot_forecast
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, "logs", "training.log"), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        os.path.join(project_root, "data"),
        os.path.join(project_root, "models"),
        os.path.join(project_root, "logs")
    ]
    
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train stock price prediction models")
    
    parser.add_argument("--ticker", type=str, default="AAPL", 
                        help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--period", type=str, default="5y", 
                        help="Historical data period (default: 5y)")
    parser.add_argument("--interval", type=str, default="1d", 
                        help="Data interval (default: 1d)")
    parser.add_argument("--forecast_horizons", type=str, default="90,180", 
                        help="Comma-separated list of forecast horizons in days (default: 90,180)")
    parser.add_argument("--sequence_length", type=int, default=60, 
                        help="Sequence length for LSTM model (default: 60)")
    parser.add_argument("--use_baseline", action="store_true", 
                        help="Train baseline models for comparison")
    parser.add_argument("--lstm_units", type=int, default=50, 
                        help="Number of LSTM units in each layer (default: 50)")
    parser.add_argument("--dropout_rate", type=float, default=0.2, 
                        help="Dropout rate for LSTM layers (default: 0.2)")
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                        help="Learning rate for the optimizer (default: 0.001)")
    parser.add_argument("--epochs", type=int, default=100, 
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training (default: 32)")
    parser.add_argument("--patience", type=int, default=20, 
                        help="Patience for early stopping (default: 20)")
    parser.add_argument("--save_data", action="store_true", 
                        help="Save the fetched and processed data")
    parser.add_argument("--no_plots", action="store_true", 
                        help="Disable plot generation")
    
    return parser.parse_args()

def train_and_evaluate_lstm(df, forecast_horizon, sequence_length, args):
    """
    Train and evaluate LSTM model for a specific forecast horizon.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed stock data
    forecast_horizon : int
        Forecast horizon in days
    sequence_length : int
        Sequence length for LSTM model
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    tuple
        (model, evaluation, forecasts)
    """
    logger.info(f"Training LSTM model for {forecast_horizon}-day forecast")
    
    # Target column name
    target_col = f'Close_future_{forecast_horizon}d'
    
    # Prepare data for LSTM
    prepared_data = prepare_data_for_lstm(
        df, target_col, sequence_length=sequence_length
    )
    
    # Train the model
    model, evaluation = create_and_train_lstm_model(
        prepared_data,
        units=args.lstm_units,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        model_dir=os.path.join(project_root, "models")
    )
    
    # Print evaluation results
    logger.info(f"{forecast_horizon}-day Forecast Model Evaluation:")
    logger.info(f"RMSE: {evaluation['rmse']:.4f}")
    logger.info(f"MAE: {evaluation['mae']:.4f}")
    logger.info(f"MAPE: {evaluation['mape']:.2f}%")
    
    # Generate a forecast
    last_sequence = prepared_data['X_test'][-1:].copy()
    forecasts = model.forecast(
        last_sequence, 
        steps=forecast_horizon, 
        scaler=prepared_data['scaler']
    )
    
    logger.info(f"Forecast for next {forecast_horizon} days: {forecasts[0]:.2f} to {forecasts[-1]:.2f}")
    
    # Plot actual vs predicted if not disabled
    if not args.no_plots:
        import matplotlib.pyplot as plt
        
        # Get the actual test values
        y_test = prepared_data['y_test']
        y_pred = evaluation['y_pred']
        
        # Plot actual vs predicted
        test_plot = plot_prediction_vs_actual(
            y_test, y_pred, ticker_symbol=args.ticker,
            figsize=(12, 6)
        )
        test_plot_path = os.path.join(
            project_root, "outputs", 
            f"{args.ticker}_{forecast_horizon}d_test_predictions.png"
        )
        os.makedirs(os.path.dirname(test_plot_path), exist_ok=True)
        test_plot.savefig(test_plot_path)
        plt.close()
        
        # Convert forecasts to pandas Series with dates
        last_date = df.index[-1]
        forecast_dates = pd.date_range(
            start=last_date, 
            periods=forecast_horizon+1
        )[1:]  # Exclude the first date (which is the last historical date)
        
        # Plot forecast
        forecast_plot = plot_forecast(
            df['Close'][-100:],  # Last 100 days of historical data
            forecasts,
            forecast_dates,
            ticker_symbol=args.ticker
        )
        forecast_html_path = os.path.join(
            project_root, "outputs", 
            f"{args.ticker}_{forecast_horizon}d_forecast.html"
        )
        forecast_plot.write_html(forecast_html_path)
    
    return model, evaluation, forecasts

def main():
    """Main function to run the training process."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create necessary directories
    create_directories()
    
    # Log the training run
    logger.info(f"Starting training run for {args.ticker} with period={args.period}")
    logger.info(f"Command line arguments: {args}")
    
    # Fetch the data
    logger.info(f"Fetching data for {args.ticker}...")
    df = fetch_stock_data(args.ticker, period=args.period, interval=args.interval)
    
    if df is None or df.empty:
        logger.error(f"Failed to fetch data for {args.ticker}")
        return
    
    logger.info(f"Fetched {len(df)} records for {args.ticker}")
    
    # Save raw data if requested
    if args.save_data:
        raw_data_path = save_stock_data(df, args.ticker, os.path.join(project_root, "data", "raw"))
        logger.info(f"Saved raw data to {raw_data_path}")
    
    # Preprocess the data
    logger.info("Adding technical indicators...")
    df = add_technical_indicators(df)
    
    # Parse forecast horizons
    forecast_horizons = [int(h) for h in args.forecast_horizons.split(',')]
    logger.info(f"Forecast horizons: {forecast_horizons} days")
    
    # Prepare target variables
    logger.info("Preparing target variables...")
    df = prepare_target_variables(df, forecast_horizons)
    
    # Save processed data if requested
    if args.save_data:
        processed_data_path = os.path.join(
            project_root, "data", "processed", 
            f"{args.ticker}_processed_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        df.to_csv(processed_data_path)
        logger.info(f"Saved processed data to {processed_data_path}")
    
    # Generate basic plots if not disabled
    if not args.no_plots:
        import matplotlib.pyplot as plt
        
        # Create the outputs directory if it doesn't exist
        outputs_dir = os.path.join(project_root, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        
        # Plot stock price history
        price_history_plot = plot_stock_price_history(df, args.ticker)
        price_history_plot.savefig(
            os.path.join(outputs_dir, f"{args.ticker}_price_history.png")
        )
        plt.close()
        
        # Plot technical indicators
        tech_indicators_plot = plot_technical_indicators(df, args.ticker)
        tech_indicators_plot.savefig(
            os.path.join(outputs_dir, f"{args.ticker}_technical_indicators.png")
        )
        plt.close()
    
    # Train baseline models if requested
    if args.use_baseline:
        logger.info("Training baseline models...")
        
        for horizon in forecast_horizons:
            # Target column
            target_col = f'Close_future_{horizon}d'
            
            # Drop rows with NaN values
            df_clean = df.dropna()
            
            # Prepare features and target
            feature_cols = [col for col in df_clean.columns if col not in [
                f'Close_future_{h}d' for h in forecast_horizons
            ] + [
                f'PriceChange_{h}d' for h in forecast_horizons
            ] + [
                f'PriceChangePct_{h}d' for h in forecast_horizons
            ] + [
                f'PriceUp_{h}d' for h in forecast_horizons
            ]]
            
            X = df_clean[feature_cols].values
            y = df_clean[target_col].values
            
            # Train-test split (use the last 20% for testing)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Define models to compare
            models = {
                'LastValue': LastValuePredictor(),
                'MeanValue': MeanValuePredictor(),
                'LinearRegression': LinearRegressionModel(),
                'RandomForest': RandomForestModel(n_estimators=100)
            }
            
            # Compare models
            results = compare_models(models, X_train, y_train, X_test, y_test)
            
            # Save the results
            baseline_results_path = os.path.join(
                project_root, "models", 
                f"{args.ticker}_baseline_results_{horizon}d.joblib"
            )
            joblib.dump(results, baseline_results_path)
            logger.info(f"Saved baseline results to {baseline_results_path}")
            
            # Save the best model
            best_model_name = min(results.items(), key=lambda x: x[1]['rmse'])[0]
            best_model = models[best_model_name]
            best_model_path = os.path.join(
                project_root, "models", 
                f"{args.ticker}_best_baseline_{horizon}d.joblib"
            )
            joblib.dump(best_model, best_model_path)
            logger.info(f"Best baseline model for {horizon}-day forecast: {best_model_name}")
            logger.info(f"Saved best baseline model to {best_model_path}")
    
    # Train LSTM models for each forecast horizon
    lstm_results = {}
    for horizon in forecast_horizons:
        model, evaluation, forecasts = train_and_evaluate_lstm(
            df, horizon, args.sequence_length, args
        )
        
        # Store the results
        lstm_results[horizon] = {
            'model': model,
            'evaluation': evaluation,
            'forecasts': forecasts
        }
    
    # Save the final results
    results_summary = {
        'ticker': args.ticker,
        'period': args.period,
        'interval': args.interval,
        'forecast_horizons': forecast_horizons,
        'sequence_length': args.sequence_length,
        'lstm_params': {
            'units': args.lstm_units,
            'dropout_rate': args.dropout_rate,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'patience': args.patience
        },
        'results': {
            horizon: {
                'rmse': lstm_results[horizon]['evaluation']['rmse'],
                'mae': lstm_results[horizon]['evaluation']['mae'],
                'mape': lstm_results[horizon]['evaluation']['mape'],
                'forecast_start': forecasts[0],
                'forecast_end': forecasts[-1]
            }
            for horizon, forecasts in [(h, lstm_results[h]['forecasts']) for h in forecast_horizons]
        }
    }
    
    # Save the summary
    summary_path = os.path.join(
        project_root, "models", 
        f"{args.ticker}_training_summary.joblib"
    )
    joblib.dump(results_summary, summary_path)
    logger.info(f"Saved training summary to {summary_path}")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error during training: {str(e)}")
        sys.exit(1)