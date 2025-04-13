# Stock Market Prediction Model

This repository contains a machine learning model for predicting stock prices over different time horizons (3 months, 6 months) based on historical data from Yahoo Finance.

## Model Description

### Model Architecture

The model uses a deep learning approach based on Long Short-Term Memory (LSTM) networks, which are well-suited for time series prediction tasks. The LSTM model is capable of learning patterns in the historical stock price data and using them to make predictions about future prices.

The model architecture consists of:
- Three stacked LSTM layers with batch normalization and dropout
- 50 LSTM units per layer
- 0.2 dropout rate for regularization
- Adam optimizer with learning rate of 0.001
- Mean squared error loss function

### Performance

The model is evaluated using several metrics:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

Performance varies by stock and forecast horizon, but typical results include:
- 3-month forecast: RMSE between 5-15%, MAPE between 5-12%
- 6-month forecast: RMSE between 8-20%, MAPE between 8-18%

### Limitations

- The model's accuracy decreases as the forecast horizon increases
- The model does not explicitly account for unexpected events (e.g., company announcements, global events)
- Performance varies significantly across different stocks
- Past performance is not indicative of future results

## Usage

### API Endpoints

The model is deployed as a REST API with the following endpoints:

#### Predict Stock Prices

```
POST /predict
```

**Request Body:**
```json
{
  "ticker": "AAPL",
  "forecast_horizon": 90,
  "model_type": "lstm"
}
```

**Parameters:**
- `ticker` (required): Stock ticker symbol (e.g., "AAPL" for Apple)
- `forecast_horizon` (optional, default: 90): Number of days to forecast
- `model_type` (optional, default: "lstm"): Type of model to use ("lstm" or "baseline")

**Response:**
```json
{
  "ticker": "AAPL",
  "forecast_horizon": 90,
  "forecast_start_date": "2025-04-14",
  "forecast_end_date": "2025-07-13",
  "predictions": [
    {"date": "2025-04-14", "price": 176.21},
    {"date": "2025-04-15", "price": 177.05},
    ...
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
```

#### GET Prediction by Ticker

```
GET /predict/{ticker}?forecast_horizon=90&model_type=lstm
```

**Parameters:**
- `ticker` (path parameter, required): Stock ticker symbol
- `forecast_horizon` (query parameter, optional, default: 90): Number of days to forecast
- `model_type` (query parameter, optional, default: "lstm"): Type of model to use

#### List Available Models

```
GET /models
```

**Response:**
```json
{
  "lstm_models": ["AAPL", "MSFT", "GOOGL"],
  "baseline_models": ["AAPL", "MSFT", "GOOGL", "AMZN"],
  "total_models": 7
}
```

### Example Usage

```python
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Make a prediction request
response = requests.post(
    "https://api-endpoint.huggingface.co/predict",
    json={
        "ticker": "AAPL",
        "forecast_horizon": 90,
        "model_type": "lstm"
    }
)
data = response.json()

# Convert predictions to DataFrame
predictions = pd.DataFrame(data["predictions"])
predictions["date"] = pd.to_datetime(predictions["date"])
predictions.set_index("date", inplace=True)

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.plot(predictions.index, predictions["price"])
plt.title(f"{data['ticker']} Price Forecast for {data['forecast_horizon']} Days")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.grid(True)
plt.show()
```

## Training Data

The model is trained on historical stock price data from Yahoo Finance, including:
- Daily open, high, low, close prices
- Trading volume
- Technical indicators (moving averages, RSI, MACD, etc.)

The data is preprocessed to create features and handle missing values before training.

## Ethics & Limitations

- This model is for educational and research purposes only
- Financial investments involve risk, and this model should not be the sole basis for making investment decisions
- The model does not consider all factors that might affect stock prices, such as company news, market sentiment, or global events
- Users should consult with financial advisors before making investment decisions

## Citation

If you use this model in your research, please cite:

```
@software{stock_prediction_app,
  author = {PhaniChandraSekhar},
  title = {Stock Market Prediction Model},
  url = {https://github.com/PhaniChandraSekhar/stock-prediction-app},
  year = {2023},
}
```