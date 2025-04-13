# Stock Market Prediction App

A machine learning-based application for predicting stock prices over 3-month and 6-month horizons using historical data from Yahoo Finance.

## Features

- **Data Collection**: Automatically fetch historical stock data from Yahoo Finance
- **Machine Learning Model**: LSTM-based time series prediction model
- **FastAPI Backend**: Robust API for model inference
- **Streamlit Frontend**: Interactive UI for stock selection and prediction visualization
- **Hugging Face Deployment**: Easily deploy the model and application to Hugging Face Spaces

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone the repository:
```bash
git clone https://github.com/PhaniChandraSekhar/stock-prediction-app.git
cd stock-prediction-app
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- On Windows:
```bash
venv\Scripts\activate
```
- On macOS/Linux:
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training a model
```bash
python src/model/train.py --ticker AAPL --epochs 100
```

### Running the API locally
```bash
uvicorn src.api.main:app --reload
```

### Running the Streamlit app locally
```bash
streamlit run src/app/streamlit_app.py
```

## Project Structure
```
stock-prediction-app/
├── data/                    # Data storage directory
├── models/                  # Saved model files
├── notebooks/               # Jupyter notebooks for exploration
├── src/
│   ├── api/                 # FastAPI application
│   │   ├── __init__.py
│   │   └── main.py
│   ├── app/                 # Streamlit application
│   │   ├── __init__.py
│   │   └── streamlit_app.py
│   ├── data/                # Data processing utilities
│   │   ├── __init__.py
│   │   ├── fetch.py
│   │   └── preprocess.py
│   ├── model/               # ML model implementation
│   │   ├── __init__.py
│   │   ├── predict.py
│   │   └── train.py
│   └── utils/               # Utility functions
│       ├── __init__.py
│       └── visualization.py
├── tests/                   # Unit tests
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

## Deployment

The application is deployed on Hugging Face Spaces:
- **Model**: [Hugging Face Model Link]
- **App**: [Hugging Face Space Link]

## License

MIT