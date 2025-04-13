"""
Baseline models for stock price prediction.
This module provides simple baseline models for comparison with more complex models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LastValuePredictor:
    """
    Baseline predictor that uses the last observed value as the prediction.
    """
    
    def __init__(self):
        """Initialize the LastValuePredictor."""
        self.last_value = None
    
    def fit(self, X, y):
        """
        Fit the model by storing the last value of y.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data (not used in this model)
        y : array-like of shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : object
        """
        self.last_value = y[-1]
        return self
    
    def predict(self, X):
        """
        Predict using the last observed value.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples (not used in this model)
        
        Returns:
        --------
        array-like of shape (n_samples,)
            Returns an array of the last observed value
        """
        return np.full(shape=X.shape[0], fill_value=self.last_value)

class MeanValuePredictor:
    """
    Baseline predictor that uses the mean of observed values as the prediction.
    """
    
    def __init__(self):
        """Initialize the MeanValuePredictor."""
        self.mean_value = None
    
    def fit(self, X, y):
        """
        Fit the model by calculating the mean of y.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data (not used in this model)
        y : array-like of shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : object
        """
        self.mean_value = np.mean(y)
        return self
    
    def predict(self, X):
        """
        Predict using the mean observed value.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples (not used in this model)
        
        Returns:
        --------
        array-like of shape (n_samples,)
            Returns an array of the mean observed value
        """
        return np.full(shape=X.shape[0], fill_value=self.mean_value)

class LinearRegressionModel:
    """
    Linear regression model for stock price prediction.
    """
    
    def __init__(self):
        """Initialize the LinearRegressionModel."""
        self.model = LinearRegression()
    
    def fit(self, X, y):
        """
        Fit the linear regression model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : object
        """
        logger.info(f"Fitting linear regression model with {X.shape[1]} features")
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Predict using the linear regression model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples
        
        Returns:
        --------
        array-like of shape (n_samples,)
            Returns predicted values
        """
        return self.model.predict(X)
    
    def get_feature_importance(self, feature_names=None):
        """
        Get the feature importance from the model.
        
        Parameters:
        -----------
        feature_names : list, default None
            List of feature names
        
        Returns:
        --------
        dict
            Dictionary mapping feature names to importance values
        """
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.model.coef_))]
        
        importance = np.abs(self.model.coef_)
        return dict(zip(feature_names, importance))

class RandomForestModel:
    """
    Random Forest model for stock price prediction.
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the RandomForestModel.
        
        Parameters:
        -----------
        n_estimators : int, default 100
            Number of trees in the forest
        random_state : int, default 42
            Random seed for reproducibility
        """
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    
    def fit(self, X, y):
        """
        Fit the Random Forest model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : object
        """
        logger.info(f"Fitting Random Forest model with {X.shape[1]} features")
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Predict using the Random Forest model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples
        
        Returns:
        --------
        array-like of shape (n_samples,)
            Returns predicted values
        """
        return self.model.predict(X)
    
    def get_feature_importance(self, feature_names=None):
        """
        Get the feature importance from the model.
        
        Parameters:
        -----------
        feature_names : list, default None
            List of feature names
        
        Returns:
        --------
        dict
            Dictionary mapping feature names to importance values
        """
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.model.feature_importances_))]
        
        importance = self.model.feature_importances_
        return dict(zip(feature_names, importance))

def evaluate_model(model, X, y, model_name=None):
    """
    Evaluate a model's performance using various metrics.
    
    Parameters:
    -----------
    model : object
        The model to evaluate (must have a predict method)
    X : array-like of shape (n_samples, n_features)
        Test samples
    y : array-like of shape (n_samples,)
        True values
    model_name : str, default None
        Name of the model for logging
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    if model_name is None:
        model_name = model.__class__.__name__
    
    logger.info(f"Evaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    
    # Log results
    logger.info(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
    
    # Return metrics as dictionary
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'y_pred': y_pred
    }

def compare_models(models, X_train, y_train, X_test, y_test):
    """
    Compare multiple models on the same dataset.
    
    Parameters:
    -----------
    models : dict
        Dictionary mapping model names to model objects
    X_train : array-like of shape (n_samples, n_features)
        Training samples
    y_train : array-like of shape (n_samples,)
        Training target values
    X_test : array-like of shape (n_samples, n_features)
        Test samples
    y_test : array-like of shape (n_samples,)
        Test target values
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics for each model
    """
    logger.info(f"Comparing {len(models)} models...")
    
    results = {}
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate on test data
        metrics = evaluate_model(model, X_test, y_test, name)
        results[name] = metrics
    
    # Determine the best model by RMSE
    best_model = min(results.items(), key=lambda x: x[1]['rmse'])
    logger.info(f"Best model: {best_model[0]} with RMSE: {best_model[1]['rmse']:.4f}")
    
    return results

if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from data.fetch import fetch_stock_data
    from data.preprocess import add_technical_indicators, prepare_target_variables, scale_data
    
    # Fetch and prepare data
    df = fetch_stock_data('AAPL', period='2y')
    df = add_technical_indicators(df)
    df = prepare_target_variables(df, [90])  # 3-month forecast
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['Close_future_90d', 'PriceChange_90d', 'PriceChangePct_90d', 'PriceUp_90d']]
    X = df[feature_cols].values
    y = df['Close_future_90d'].values
    
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