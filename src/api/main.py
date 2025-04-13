"""
FastAPI application for stock price prediction API.
This module provides REST API endpoints for making stock price predictions.
"""

import os
import sys
import logging
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib

# Add the project root directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, project_root)

from src.data.fetch import fetch_stock_data
from src.data.preprocess import add_technical_indicators, scale_data
from src.model.lstm import StockLSTM

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, "logs", "api.log"), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)

# Create the FastAPI app
app = FastAPI(
    title="Stock Price Prediction API",
    description="API for predicting stock prices using machine learning models",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Define model cache
model_cache = {}

# Define Pydantic models for request and response
class StockPredictionRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    forecast_horizon: int = Field(90, description="Forecast horizon in days (e.g., 90 for 3 months, 180 for 6 months)")
    model_type: Optional[str] = Field("lstm", description="Model type to use (lstm or baseline)")

class StockPredictionResponse(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    forecast_horizon: int = Field(..., description="Forecast horizon in days")
    forecast_start_date: str = Field(..., description="Start date of the forecast period")
    forecast_end_date: str = Field(..., description="End date of the forecast period")
    predictions: List[Dict[str, Any]] = Field(..., description="Predicted values for each date")
    metadata: Dict[str, Any] = Field(..., description="Metadata about the prediction")

# Helper functions
def load_model(ticker: str, forecast_horizon: int = 90, model_type: str = "lstm"):
    """
    Load a model from the model directory or cache.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    forecast_horizon : int, default 90
        Forecast horizon in days
    model_type : str, default "lstm"
        Type of model to load
        
    Returns:
    --------
    object
        Loaded model
    """
    # Check if the model is in the cache
    cache_key = f"{ticker}_{forecast_horizon}_{model_type}"
    if cache_key in model_cache:
        logger.info(f"Using cached model for {cache_key}")
        return model_cache[cache_key]
    
    # Model is not in cache, try to load it
    model_dir = os.path.join(project_root, "models")
    
    if model_type == "lstm":
        # Look for LSTM model files
        model_files = [
            f for f in os.listdir(model_dir) 
            if f.startswith(f"{ticker}_") and f.endswith(".keras")
        ]
        
        if not model_files:
            raise HTTPException(
                status_code=404,
                detail=f"No LSTM model found for {ticker}"
            )
        
        # Use the most recent model file
        model_files.sort(reverse=True)
        model_path = os.path.join(model_dir, model_files[0])
        
        # Try to find associated metadata and scaler files
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        metadata_path = os.path.join(model_dir, f"{model_name}_metadata.joblib")
        scaler_path = os.path.join(model_dir, f"{model_name}_scaler.joblib")
        
        # Check if the files exist
        if not os.path.exists(metadata_path):
            metadata_path = None
        if not os.path.exists(scaler_path):
            scaler_path = None
        
        # Load the model
        logger.info(f"Loading LSTM model from {model_path}")
        model = StockLSTM.load(model_path, metadata_path, scaler_path)
    
    else:  # baseline model
        # Look for baseline model files
        model_files = [
            f for f in os.listdir(model_dir) 
            if f.startswith(f"{ticker}_best_baseline_{forecast_horizon}d") and f.endswith(".joblib")
        ]
        
        if not model_files:
            raise HTTPException(
                status_code=404,
                detail=f"No baseline model found for {ticker} with forecast horizon {forecast_horizon}"
            )
        
        # Use the first model file found
        model_path = os.path.join(model_dir, model_files[0])
        
        # Load the model
        logger.info(f"Loading baseline model from {model_path}")
        model = joblib.load(model_path)
    
    # Cache the model
    model_cache[cache_key] = model
    
    return model

def prepare_data_for_prediction(ticker: str, sequence_length: int = 60):
    """
    Prepare data for making predictions.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    sequence_length : int, default 60
        Sequence length for LSTM model
        
    Returns:
    --------
    tuple
        (df, last_sequence, scaler)
    """
    # Fetch the data
    logger.info(f"Fetching data for {ticker}...")
    df = fetch_stock_data(ticker, period="1y", interval="1d")
    
    if df is None or df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Failed to fetch data for {ticker}"
        )
    
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

# API routes
@app.get("/")
async def root():
    """Root endpoint that returns API information."""
    return {
        "name": "Stock Price Prediction API",
        "version": "1.0.0",
        "description": "API for predicting stock prices using machine learning models",
        "endpoints": [
            {"path": "/predict", "method": "POST", "description": "Make stock price predictions"},
            {"path": "/predict/{ticker}", "method": "GET", "description": "Make stock price predictions for a specific ticker"},
            {"path": "/models", "method": "GET", "description": "List available models"},
            {"path": "/health", "method": "GET", "description": "Check API health"}
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/models")
async def list_models():
    """List available models."""
    model_dir = os.path.join(project_root, "models")
    
    # Check if the model directory exists
    if not os.path.exists(model_dir):
        return {"models": []}
    
    # List model files
    lstm_models = [
        f for f in os.listdir(model_dir) 
        if f.endswith(".keras")
    ]
    
    baseline_models = [
        f for f in os.listdir(model_dir) 
        if f.startswith("") and f.endswith(".joblib") and "best_baseline" in f
    ]
    
    # Extract ticker symbols from model files
    lstm_tickers = set()
    for model_file in lstm_models:
        ticker = model_file.split("_")[0]
        lstm_tickers.add(ticker)
    
    baseline_tickers = set()
    for model_file in baseline_models:
        ticker = model_file.split("_")[0]
        baseline_tickers.add(ticker)
    
    return {
        "lstm_models": list(lstm_tickers),
        "baseline_models": list(baseline_tickers),
        "total_models": len(lstm_models) + len(baseline_models)
    }

@app.post("/predict", response_model=StockPredictionResponse)
async def predict_stock(request: StockPredictionRequest):
    """
    Make stock price predictions based on the request parameters.
    
    Parameters:
    -----------
    request : StockPredictionRequest
        Request containing ticker, forecast horizon, and model type
        
    Returns:
    --------
    StockPredictionResponse
        Response containing predictions and metadata
    """
    try:
        # Load the model
        model = load_model(request.ticker, request.forecast_horizon, request.model_type)
        
        # Prepare data for prediction
        df, last_sequence, scaler = prepare_data_for_prediction(request.ticker)
        
        # Make the prediction
        if request.model_type == "lstm":
            forecasts = model.forecast(
                last_sequence, 
                steps=request.forecast_horizon, 
                scaler=scaler
            )
        else:  # baseline model
            # For baseline models, we need to reshape the input
            X_test = last_sequence.reshape(1, -1)
            forecasts = np.array([model.predict(X_test)[0]] * request.forecast_horizon)
        
        # Generate dates for the forecast
        last_date = df.index[-1]
        forecast_dates = pd.date_range(
            start=last_date, 
            periods=request.forecast_horizon+1
        )[1:]  # Exclude the first date (which is the last historical date)
        
        # Create a list of predictions with dates
        predictions = []
        for i, date in enumerate(forecast_dates):
            predictions.append({
                "date": date.strftime("%Y-%m-%d"),
                "price": float(forecasts[i])
            })
        
        # Create response
        response = {
            "ticker": request.ticker,
            "forecast_horizon": request.forecast_horizon,
            "forecast_start_date": forecast_dates[0].strftime("%Y-%m-%d"),
            "forecast_end_date": forecast_dates[-1].strftime("%Y-%m-%d"),
            "predictions": predictions,
            "metadata": {
                "model_type": request.model_type,
                "last_price": float(df["Close"].iloc[-1]),
                "prediction_timestamp": datetime.now().isoformat(),
                "historical_start_date": df.index[0].strftime("%Y-%m-%d"),
                "historical_end_date": df.index[-1].strftime("%Y-%m-%d")
            }
        }
        
        return response
    
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    
    except Exception as e:
        # Log the error and return a 500 response
        logger.exception(f"Error making prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

@app.get("/predict/{ticker}", response_model=StockPredictionResponse)
async def predict_stock_get(
    ticker: str,
    forecast_horizon: int = Query(90, description="Forecast horizon in days"),
    model_type: str = Query("lstm", description="Model type (lstm or baseline)")
):
    """
    Make stock price predictions for a specific ticker.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    forecast_horizon : int, default 90
        Forecast horizon in days
    model_type : str, default "lstm"
        Type of model to use
        
    Returns:
    --------
    StockPredictionResponse
        Response containing predictions and metadata
    """
    # Create a request object and call the POST endpoint
    request = StockPredictionRequest(
        ticker=ticker,
        forecast_horizon=forecast_horizon,
        model_type=model_type
    )
    
    return await predict_stock(request)

if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)