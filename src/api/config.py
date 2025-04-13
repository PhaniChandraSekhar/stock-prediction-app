"""
Configuration settings for the FastAPI application.
This module provides configuration parameters and environment variable handling.
"""

import os
from pydantic import BaseSettings
from typing import List, Optional

# Determine the project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, "../.."))

class Settings(BaseSettings):
    """Settings for the API application."""
    
    # API settings
    API_TITLE: str = "Stock Price Prediction API"
    API_DESCRIPTION: str = "API for predicting stock prices using machine learning models"
    API_VERSION: str = "1.0.0"
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False
    DEBUG: bool = False
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # Model settings
    MODEL_DIR: str = os.path.join(PROJECT_ROOT, "models")
    MODELS_CACHE_SIZE: int = 10  # Number of models to cache in memory
    
    # Data settings
    DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
    DEFAULT_HISTORICAL_PERIOD: str = "1y"
    DEFAULT_DATA_INTERVAL: str = "1d"
    MAX_FORECAST_DAYS: int = 365
    DEFAULT_FORECAST_DAYS: int = 90
    
    # Logging settings
    LOG_DIR: str = os.path.join(PROJECT_ROOT, "logs")
    LOG_LEVEL: str = "INFO"
    
    class Config:
        """Pydantic config for the Settings class."""
        
        env_file = os.path.join(PROJECT_ROOT, ".env")
        env_file_encoding = 'utf-8'
        case_sensitive = True

# Create a settings instance
settings = Settings()

# Create necessary directories
os.makedirs(settings.MODEL_DIR, exist_ok=True)
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.LOG_DIR, exist_ok=True)