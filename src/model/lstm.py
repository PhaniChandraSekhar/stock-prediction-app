"""
LSTM Model for stock price prediction.
This module provides an LSTM-based model for time series prediction of stock prices.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import logging
import joblib
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockLSTM:
    """
    Long Short-Term Memory (LSTM) model for stock price prediction.
    """
    
    def __init__(self, sequence_length=60, n_features=1, target_column='Close_future_90d'):
        """
        Initialize the StockLSTM model.
        
        Parameters:
        -----------
        sequence_length : int, default 60
            Number of time steps in each sequence
        n_features : int, default 1
            Number of features in the input data
        target_column : str, default 'Close_future_90d'
            Name of the target column to predict
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.target_column = target_column
        self.model = None
        self.history = None
        self.scaler = None
        self.feature_indices = None
        self.feature_names = None
    
    def build_model(self, units=50, dropout_rate=0.2, learning_rate=0.001):
        """
        Build the LSTM model architecture.
        
        Parameters:
        -----------
        units : int, default 50
            Number of LSTM units in each layer
        dropout_rate : float, default 0.2
            Dropout rate for regularization
        learning_rate : float, default 0.001
            Learning rate for the Adam optimizer
            
        Returns:
        --------
        tensorflow.keras.models.Sequential
            The built LSTM model
        """
        logger.info(f"Building LSTM model with {units} units and {dropout_rate} dropout rate")
        
        model = Sequential([
            # First LSTM layer with return sequences
            LSTM(units=units, return_sequences=True, 
                 input_shape=(self.sequence_length, self.n_features)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # Second LSTM layer
            LSTM(units=units, return_sequences=True),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # Third LSTM layer
            LSTM(units=units, return_sequences=False),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # Output layer
            Dense(units=1)
        ])
        
        # Compile the model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        logger.info("Model summary:")
        model.summary(print_fn=logger.info)
        
        self.model = model
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32,
            patience=20, model_dir='models', tensorboard_dir=None):
        """
        Train the LSTM model.
        
        Parameters:
        -----------
        X_train : numpy.ndarray of shape (n_samples, sequence_length, n_features)
            Training sequences
        y_train : numpy.ndarray of shape (n_samples,)
            Training target values
        X_val : numpy.ndarray of shape (n_samples, sequence_length, n_features), optional
            Validation sequences
        y_val : numpy.ndarray of shape (n_samples,), optional
            Validation target values
        epochs : int, default 100
            Number of training epochs
        batch_size : int, default 32
            Batch size for training
        patience : int, default 20
            Number of epochs with no improvement after which training will be stopped
        model_dir : str, default 'models'
            Directory to save the model
        tensorboard_dir : str, optional
            Directory for TensorBoard logs
            
        Returns:
        --------
        tensorflow.keras.callbacks.History
            Training history
        """
        logger.info(f"Training LSTM model with {len(X_train)} samples for {epochs} epochs")
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Generate a timestamp for the model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"lstm_stock_{timestamp}.keras"
        model_path = os.path.join(model_dir, model_name)
        
        # Prepare callbacks
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint to save the best model
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate when a metric has stopped improving
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Add TensorBoard callback if directory is provided
        if tensorboard_dir:
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(tensorboard_dir, timestamp),
                histogram_freq=1
            )
            callbacks.append(tensorboard_callback)
        
        # Train the model
        if X_val is not None and y_val is not None:
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        self.history = history.history
        logger.info(f"Model training completed and saved to {model_path}")
        
        return history
    
    def predict(self, X):
        """
        Make predictions using the trained LSTM model.
        
        Parameters:
        -----------
        X : numpy.ndarray of shape (n_samples, sequence_length, n_features)
            Input sequences
            
        Returns:
        --------
        numpy.ndarray of shape (n_samples,)
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model is not trained. Call fit() first or load a trained model.")
        
        logger.info(f"Making predictions for {len(X)} samples")
        predictions = self.model.predict(X)
        return predictions.squeeze()
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : numpy.ndarray of shape (n_samples, sequence_length, n_features)
            Test sequences
        y_test : numpy.ndarray of shape (n_samples,)
            Test target values
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model is not trained. Call fit() first or load a trained model.")
        
        logger.info(f"Evaluating model on {len(X_test)} test samples")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = np.mean(np.square(y_test - y_pred))
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Log results
        logger.info(f"Test MSE: {mse:.4f}")
        logger.info(f"Test RMSE: {rmse:.4f}")
        logger.info(f"Test MAE: {mae:.4f}")
        logger.info(f"Test MAPE: {mape:.2f}%")
        
        # Return metrics as dictionary
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'y_pred': y_pred
        }
    
    def save(self, model_dir='models', model_name=None):
        """
        Save the trained model and associated metadata.
        
        Parameters:
        -----------
        model_dir : str, default 'models'
            Directory to save the model
        model_name : str, optional
            Base name for the model files
            
        Returns:
        --------
        dict
            Dictionary containing paths to saved files
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Generate a timestamp for the model name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"lstm_stock_{timestamp}"
        
        # Save the Keras model
        model_path = os.path.join(model_dir, f"{model_name}.keras")
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'target_column': self.target_column,
            'feature_indices': self.feature_indices,
            'feature_names': self.feature_names,
            'history': self.history
        }
        metadata_path = os.path.join(model_dir, f"{model_name}_metadata.joblib")
        joblib.dump(metadata, metadata_path)
        logger.info(f"Model metadata saved to {metadata_path}")
        
        # Save scaler if available
        if self.scaler is not None:
            scaler_path = os.path.join(model_dir, f"{model_name}_scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
        else:
            scaler_path = None
        
        return {
            'model_path': model_path,
            'metadata_path': metadata_path,
            'scaler_path': scaler_path
        }
    
    @classmethod
    def load(cls, model_path, metadata_path=None, scaler_path=None):
        """
        Load a trained model and associated metadata.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model file
        metadata_path : str, optional
            Path to the saved metadata file
        scaler_path : str, optional
            Path to the saved scaler file
            
        Returns:
        --------
        StockLSTM
            Loaded model instance
        """
        logger.info(f"Loading model from {model_path}")
        
        # Initialize a new instance
        instance = cls()
        
        # Load the Keras model
        instance.model = load_model(model_path)
        
        # Load metadata if provided
        if metadata_path is not None and os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            instance.sequence_length = metadata.get('sequence_length', 60)
            instance.n_features = metadata.get('n_features', 1)
            instance.target_column = metadata.get('target_column', 'Close_future_90d')
            instance.feature_indices = metadata.get('feature_indices', None)
            instance.feature_names = metadata.get('feature_names', None)
            instance.history = metadata.get('history', None)
            logger.info(f"Loaded metadata from {metadata_path}")
        
        # Load scaler if provided
        if scaler_path is not None and os.path.exists(scaler_path):
            instance.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        
        return instance
    
    def forecast(self, last_sequence, steps=90, scaler=None):
        """
        Generate multi-step forecasts.
        
        Parameters:
        -----------
        last_sequence : numpy.ndarray of shape (1, sequence_length, n_features)
            The last observed sequence
        steps : int, default 90
            Number of steps to forecast (90 days = 3 months)
        scaler : sklearn.preprocessing.MinMaxScaler, optional
            Scaler used to transform the data
            
        Returns:
        --------
        numpy.ndarray
            Forecasted values
        """
        if self.model is None:
            raise ValueError("Model is not trained. Call fit() first or load a trained model.")
        
        logger.info(f"Generating {steps}-step forecast")
        
        # Make a copy of the last sequence
        curr_sequence = last_sequence.copy()
        forecasts = []
        
        for _ in range(steps):
            # Predict the next value
            pred = self.model.predict(curr_sequence)
            forecasts.append(pred[0, 0])
            
            # Update the sequence by shifting and adding the new prediction
            curr_sequence = np.roll(curr_sequence, -1, axis=1)
            curr_sequence[0, -1, 0] = pred[0, 0]  # Assuming we're forecasting the closing price
        
        # Convert to numpy array
        forecasts = np.array(forecasts)
        
        # Inverse transform if scaler is provided
        if scaler is not None:
            # Reshape for inverse transform
            forecasts_reshaped = forecasts.reshape(-1, 1)
            forecasts = scaler.inverse_transform(forecasts_reshaped).flatten()
        
        return forecasts

def create_and_train_lstm_model(prepared_data, units=50, dropout_rate=0.2, learning_rate=0.001,
                              epochs=100, batch_size=32, patience=20, model_dir='models'):
    """
    Create and train an LSTM model using prepared data.
    
    Parameters:
    -----------
    prepared_data : dict
        Dictionary containing prepared data from prepare_data_for_lstm function
    units : int, default 50
        Number of LSTM units in each layer
    dropout_rate : float, default 0.2
        Dropout rate for regularization
    learning_rate : float, default 0.001
        Learning rate for the Adam optimizer
    epochs : int, default 100
        Number of training epochs
    batch_size : int, default 32
        Batch size for training
    patience : int, default 20
        Number of epochs with no improvement after which training will be stopped
    model_dir : str, default 'models'
        Directory to save the model
        
    Returns:
    --------
    tuple
        (trained_model, evaluation_results)
    """
    logger.info("Creating and training LSTM model...")
    
    # Extract data from the prepared_data dictionary
    X_train = prepared_data['X_train']
    y_train = prepared_data['y_train']
    X_val = prepared_data['X_val']
    y_val = prepared_data['y_val']
    X_test = prepared_data['X_test']
    y_test = prepared_data['y_test']
    
    sequence_length = prepared_data['sequence_length']
    n_features = X_train.shape[2]
    target_column = prepared_data['target_col']
    
    # Create and build the model
    model = StockLSTM(sequence_length=sequence_length, n_features=n_features, target_column=target_column)
    model.build_model(units=units, dropout_rate=dropout_rate, learning_rate=learning_rate)
    
    # Store feature information and scaler
    model.feature_names = prepared_data.get('feature_names', None)
    model.feature_indices = prepared_data.get('target_idx', None)
    model.scaler = prepared_data.get('scaler', None)
    
    # Train the model
    model.fit(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size,
            patience=patience, model_dir=model_dir)
    
    # Evaluate the model
    evaluation = model.evaluate(X_test, y_test)
    
    # Save the model
    model.save(model_dir=model_dir)
    
    return model, evaluation

if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from data.fetch import fetch_stock_data
    from data.preprocess import add_technical_indicators, prepare_target_variables, prepare_data_for_lstm
    
    # Fetch and prepare data
    df = fetch_stock_data('AAPL', period='3y')
    df = add_technical_indicators(df)
    df = prepare_target_variables(df, [90, 180])  # 3-month and 6-month forecasts
    
    # Prepare data for LSTM - 3 month forecast
    prepared_data_3m = prepare_data_for_lstm(df, 'Close_future_90d', sequence_length=60)
    
    # Train the model
    model_3m, eval_3m = create_and_train_lstm_model(prepared_data_3m, epochs=50)
    
    # Print evaluation results
    print("\n3-Month Forecast Model Evaluation:")
    print(f"RMSE: {eval_3m['rmse']:.4f}")
    print(f"MAE: {eval_3m['mae']:.4f}")
    print(f"MAPE: {eval_3m['mape']:.2f}%")
    
    # Generate a forecast
    last_sequence = prepared_data_3m['X_test'][-1:].copy()
    forecast_3m = model_3m.forecast(last_sequence, steps=90, scaler=prepared_data_3m['scaler'])
    print(f"\nForecast for next 90 days (3 months): {forecast_3m[0]:.2f} to {forecast_3m[-1]:.2f}")
    
    # Prepare data for LSTM - 6 month forecast
    prepared_data_6m = prepare_data_for_lstm(df, 'Close_future_180d', sequence_length=60)
    
    # Train another model for 6-month forecasts
    model_6m, eval_6m = create_and_train_lstm_model(prepared_data_6m, epochs=50)
    
    # Print evaluation results
    print("\n6-Month Forecast Model Evaluation:")
    print(f"RMSE: {eval_6m['rmse']:.4f}")
    print(f"MAE: {eval_6m['mae']:.4f}")
    print(f"MAPE: {eval_6m['mape']:.2f}%")
    
    # Generate a forecast
    last_sequence = prepared_data_6m['X_test'][-1:].copy()
    forecast_6m = model_6m.forecast(last_sequence, steps=180, scaler=prepared_data_6m['scaler'])
    print(f"\nForecast for next 180 days (6 months): {forecast_6m[0]:.2f} to {forecast_6m[-1]:.2f}")