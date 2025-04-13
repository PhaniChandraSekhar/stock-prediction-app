"""
Pydantic models for the FastAPI application.
This module defines the data models for API requests and responses.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime, date

class StockPredictionRequest(BaseModel):
    """Request model for stock prediction API."""
    
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    forecast_horizon: int = Field(90, description="Forecast horizon in days (e.g., 90 for 3 months, 180 for 6 months)")
    model_type: Optional[str] = Field("lstm", description="Model type to use (lstm or baseline)")
    
    @validator('forecast_horizon')
    def validate_forecast_horizon(cls, v):
        """Validate forecast horizon is reasonable."""
        if v <= 0:
            raise ValueError("Forecast horizon must be positive")
        if v > 365:
            raise ValueError("Forecast horizon must be at most 365 days")
        return v
    
    @validator('model_type')
    def validate_model_type(cls, v):
        """Validate model type is supported."""
        if v not in ["lstm", "baseline"]:
            raise ValueError("Model type must be either 'lstm' or 'baseline'")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "ticker": "AAPL",
                "forecast_horizon": 90,
                "model_type": "lstm"
            }
        }

class StockPrediction(BaseModel):
    """Model for a single stock prediction."""
    
    date: str = Field(..., description="Date of the prediction")
    price: float = Field(..., description="Predicted stock price")
    
    class Config:
        schema_extra = {
            "example": {
                "date": "2025-07-15",
                "price": 185.42
            }
        }

class StockPredictionMetadata(BaseModel):
    """Model for stock prediction metadata."""
    
    model_type: str = Field(..., description="Type of model used for prediction")
    last_price: float = Field(..., description="Last observed stock price")
    prediction_timestamp: str = Field(..., description="Timestamp when the prediction was made")
    historical_start_date: str = Field(..., description="Start date of historical data used")
    historical_end_date: str = Field(..., description="End date of historical data used")
    
    class Config:
        schema_extra = {
            "example": {
                "model_type": "lstm",
                "last_price": 175.29,
                "prediction_timestamp": "2025-04-13T12:30:45",
                "historical_start_date": "2024-04-13",
                "historical_end_date": "2025-04-13"
            }
        }

class StockPredictionResponse(BaseModel):
    """Response model for stock prediction API."""
    
    ticker: str = Field(..., description="Stock ticker symbol")
    forecast_horizon: int = Field(..., description="Forecast horizon in days")
    forecast_start_date: str = Field(..., description="Start date of the forecast period")
    forecast_end_date: str = Field(..., description="End date of the forecast period")
    predictions: List[StockPrediction] = Field(..., description="Predicted values for each date")
    metadata: StockPredictionMetadata = Field(..., description="Metadata about the prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "ticker": "AAPL",
                "forecast_horizon": 90,
                "forecast_start_date": "2025-04-14",
                "forecast_end_date": "2025-07-13",
                "predictions": [
                    {"date": "2025-04-14", "price": 176.21},
                    {"date": "2025-04-15", "price": 177.05},
                    # ... additional predictions omitted for brevity
                    {"date": "2025-07-13", "price": 185.42}
                ],
                "metadata": {
                    "model_type": "lstm",
                    "last_price": 175.29,
                    "prediction_timestamp": "2025-04-13T12:30:45",
                    "historical_start_date": "2024-04-13",
                    "historical_end_date": "2025-04-13"
                }
            }
        }

class ModelInfo(BaseModel):
    """Model for information about available models."""
    
    lstm_models: List[str] = Field(..., description="List of tickers with LSTM models")
    baseline_models: List[str] = Field(..., description="List of tickers with baseline models")
    total_models: int = Field(..., description="Total number of models available")
    
    class Config:
        schema_extra = {
            "example": {
                "lstm_models": ["AAPL", "MSFT", "GOOGL"],
                "baseline_models": ["AAPL", "MSFT", "GOOGL", "AMZN"],
                "total_models": 7
            }
        }

class HealthCheck(BaseModel):
    """Model for API health check response."""
    
    status: str = Field(..., description="API status")
    timestamp: str = Field(..., description="Timestamp of health check")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "ok",
                "timestamp": "2025-04-13T12:30:45"
            }
        }

class ApiInfo(BaseModel):
    """Model for API information response."""
    
    name: str = Field(..., description="API name")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="API description")
    endpoints: List[Dict[str, str]] = Field(..., description="List of available endpoints")
    
    class Config:
        schema_extra = {
            "example": {
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
        }

class ErrorResponse(BaseModel):
    """Model for API error response."""
    
    detail: str = Field(..., description="Error detail message")
    
    class Config:
        schema_extra = {
            "example": {
                "detail": "Error making prediction: Failed to fetch data for INVALID"
            }
        }