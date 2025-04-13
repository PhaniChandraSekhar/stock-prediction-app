"""
Visualization utilities for stock data analysis.
This module provides functions to visualize stock data and model predictions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set default styles
plt.style.use('ggplot')
sns.set(style="whitegrid")

def plot_stock_price_history(df, ticker_symbol=None, figsize=(15, 8)):
    """
    Plot the stock price history with volume.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Stock price dataframe with columns 'Open', 'High', 'Low', 'Close', 'Volume'
    ticker_symbol : str, optional
        Stock ticker symbol for the plot title
    figsize : tuple, default (15, 8)
        Figure size for the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    logger.info("Plotting stock price history...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price
    ax1.plot(df.index, df['Close'], label='Close Price')
    ax1.set_title(f"{ticker_symbol if ticker_symbol else 'Stock'} Price History")
    ax1.set_ylabel('Price ($)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot volume
    ax2.bar(df.index, df['Volume'], color='blue', alpha=0.5)
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Date')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def plot_technical_indicators(df, ticker_symbol=None, figsize=(15, 15)):
    """
    Plot technical indicators for stock analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Stock price dataframe with technical indicators
    ticker_symbol : str, optional
        Stock ticker symbol for the plot title
    figsize : tuple, default (15, 15)
        Figure size for the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    logger.info("Plotting technical indicators...")
    
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    
    # Plot 1: Price and Moving Averages
    axes[0].plot(df.index, df['Close'], label='Close Price')
    if 'MA50' in df.columns:
        axes[0].plot(df.index, df['MA50'], label='50-day MA', alpha=0.7)
    if 'MA200' in df.columns:
        axes[0].plot(df.index, df['MA200'], label='200-day MA', alpha=0.7)
    axes[0].set_title(f"{ticker_symbol if ticker_symbol else 'Stock'} Price and Moving Averages")
    axes[0].set_ylabel('Price ($)')
    axes[0].grid(True)
    axes[0].legend()
    
    # Plot 2: MACD
    if 'MACD' in df.columns and 'MACD_signal' in df.columns:
        axes[1].plot(df.index, df['MACD'], label='MACD')
        axes[1].plot(df.index, df['MACD_signal'], label='Signal Line')
        axes[1].bar(df.index, df['MACD'] - df['MACD_signal'], alpha=0.3, label='MACD Histogram')
        axes[1].set_title('MACD')
        axes[1].grid(True)
        axes[1].legend()
    
    # Plot 3: RSI
    if 'RSI' in df.columns:
        axes[2].plot(df.index, df['RSI'], label='RSI', color='purple')
        axes[2].axhline(70, color='red', linestyle='--', alpha=0.5)
        axes[2].axhline(30, color='green', linestyle='--', alpha=0.5)
        axes[2].set_title('Relative Strength Index (RSI)')
        axes[2].set_ylabel('RSI')
        axes[2].grid(True)
    
    # Plot 4: Bollinger Bands
    if 'UpperBand' in df.columns and 'LowerBand' in df.columns and 'MA20' in df.columns:
        axes[3].plot(df.index, df['Close'], label='Close Price')
        axes[3].plot(df.index, df['UpperBand'], label='Upper Band', color='red', alpha=0.7)
        axes[3].plot(df.index, df['MA20'], label='20-day MA', color='orange', alpha=0.7)
        axes[3].plot(df.index, df['LowerBand'], label='Lower Band', color='green', alpha=0.7)
        axes[3].set_title('Bollinger Bands')
        axes[3].set_ylabel('Price ($)')
        axes[3].set_xlabel('Date')
        axes[3].grid(True)
        axes[3].legend()
    
    plt.tight_layout()
    return fig

def create_candlestick_chart(df, ticker_symbol=None, include_volume=True):
    """
    Create an interactive candlestick chart using Plotly.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Stock price dataframe with columns 'Open', 'High', 'Low', 'Close', 'Volume'
    ticker_symbol : str, optional
        Stock ticker symbol for the plot title
    include_volume : bool, default True
        Whether to include volume in the chart
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The created figure
    """
    logger.info("Creating candlestick chart...")
    
    if include_volume:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                          row_heights=[0.8, 0.2])
    else:
        fig = go.Figure()
    
    # Add candlestick chart
    candlestick = go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=ticker_symbol if ticker_symbol else 'Stock'
    )
    
    if include_volume:
        fig.add_trace(candlestick, row=1, col=1)
    else:
        fig.add_trace(candlestick)
    
    # Add volume bar chart
    if include_volume:
        colors = ['red' if row['Open'] > row['Close'] else 'green' for _, row in df.iterrows()]
        volume_bar = go.Bar(
            x=df.index,
            y=df['Volume'],
            marker_color=colors,
            opacity=0.5,
            name='Volume'
        )
        fig.add_trace(volume_bar, row=2, col=1)
    
    # Add moving averages if available
    if 'MA50' in df.columns:
        ma50 = go.Scatter(
            x=df.index,
            y=df['MA50'],
            line=dict(color='blue', width=1.5),
            name='50-day MA'
        )
        if include_volume:
            fig.add_trace(ma50, row=1, col=1)
        else:
            fig.add_trace(ma50)
    
    if 'MA200' in df.columns:
        ma200 = go.Scatter(
            x=df.index,
            y=df['MA200'],
            line=dict(color='orange', width=1.5),
            name='200-day MA'
        )
        if include_volume:
            fig.add_trace(ma200, row=1, col=1)
        else:
            fig.add_trace(ma200)
    
    # Customize the layout
    title = f"{ticker_symbol if ticker_symbol else 'Stock'} Candlestick Chart"
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=800,
        yaxis=dict(
            title="Price ($)",
            showgrid=True,
            zeroline=False
        ),
        yaxis2=dict(
            title="Volume",
            showgrid=True,
            zeroline=False
        ) if include_volume else None,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    return fig

def plot_correlation_matrix(df, figsize=(12, 10)):
    """
    Plot correlation matrix of stock features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Stock price dataframe with features
    figsize : tuple, default (12, 10)
        Figure size for the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    logger.info("Plotting correlation matrix...")
    
    # Calculate the correlation matrix
    corr = df.corr()
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5, ax=ax)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    return fig

def plot_prediction_vs_actual(actual, predicted, dates=None, ticker_symbol=None, figsize=(15, 8)):
    """
    Plot predicted vs actual stock prices.
    
    Parameters:
    -----------
    actual : array-like
        Actual stock prices
    predicted : array-like
        Predicted stock prices
    dates : array-like, optional
        Date values for the x-axis
    ticker_symbol : str, optional
        Stock ticker symbol for the plot title
    figsize : tuple, default (15, 8)
        Figure size for the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    logger.info("Plotting prediction vs actual prices...")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = dates if dates is not None else np.arange(len(actual))
    
    ax.plot(x, actual, label='Actual', color='blue', marker='o', alpha=0.7, markersize=4)
    ax.plot(x, predicted, label='Predicted', color='red', marker='o', alpha=0.7, markersize=4)
    
    title = f"{ticker_symbol if ticker_symbol else 'Stock'} Price: Actual vs Predicted"
    ax.set_title(title)
    ax.set_ylabel('Price ($)')
    
    if dates is not None:
        ax.set_xlabel('Date')
        plt.xticks(rotation=45)
    else:
        ax.set_xlabel('Time')
    
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_forecast(historical, forecast, forecast_dates, ticker_symbol=None, figsize=(15, 8)):
    """
    Plot historical prices with forecasted future prices.
    
    Parameters:
    -----------
    historical : pandas.DataFrame or array-like
        Historical stock prices
    forecast : array-like
        Forecasted stock prices
    forecast_dates : array-like
        Dates for the forecasted prices
    ticker_symbol : str, optional
        Stock ticker symbol for the plot title
    figsize : tuple, default (15, 8)
        Figure size for the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The created figure
    """
    logger.info("Plotting forecast with historical data...")
    
    fig = go.Figure()
    
    # Plot historical data
    if isinstance(historical, pd.DataFrame):
        historical_x = historical.index
        historical_y = historical['Close']
    else:
        historical_x = np.arange(len(historical))
        historical_y = historical
    
    fig.add_trace(go.Scatter(
        x=historical_x,
        y=historical_y,
        mode='lines',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Plot forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', dash='dash'),
        marker=dict(size=6)
    ))
    
    # Add confidence interval (if provided)
    if len(forecast.shape) > 1 and forecast.shape[1] >= 3:
        lower_bound = forecast[:, 1]
        upper_bound = forecast[:, 2]
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            name='95% Confidence Interval'
        ))
    
    # Customize the layout
    title = f"{ticker_symbol if ticker_symbol else 'Stock'} Price Forecast"
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        shapes=[
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=historical_x[-1],
                y0=0,
                x1=historical_x[-1],
                y1=1,
                line=dict(
                    color="green",
                    width=2,
                    dash="dash",
                )
            )
        ],
        annotations=[
            dict(
                x=historical_x[-1],
                y=1.05,
                xref="x",
                yref="paper",
                text="Forecast Start",
                showarrow=False,
                font=dict(
                    color="green"
                )
            )
        ]
    )
    
    return fig

def plot_feature_importance(feature_names, importances, figsize=(12, 8)):
    """
    Plot feature importance from a machine learning model.
    
    Parameters:
    -----------
    feature_names : list
        Names of the features
    importances : array-like
        Importance scores for each feature
    figsize : tuple, default (12, 8)
        Figure size for the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    logger.info("Plotting feature importance...")
    
    # Sort features by importance
    indices = np.argsort(importances)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from data.fetch import fetch_stock_data
    from data.preprocess import add_technical_indicators
    
    # Fetch some sample data
    df = fetch_stock_data('AAPL', period='2y')
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Create and display plots
    fig1 = plot_stock_price_history(df, 'AAPL')
    plt.savefig('stock_history.png')
    
    fig2 = plot_technical_indicators(df, 'AAPL')
    plt.savefig('technical_indicators.png')
    
    fig3 = create_candlestick_chart(df, 'AAPL')
    fig3.write_html('candlestick.html')