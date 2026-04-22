# Apple Stock Price Forecasting using ARIMA and SARIMA Models

## Project Description

This project performs comprehensive time series forecasting on Apple (AAPL) stock prices utilizing ARIMA and SARIMA models. It features automated model order selection, seasonality incorporation, detailed exploratory analysis, and evaluation with multiple performance metrics. Visualizations include trend decomposition, autocorrelation analysis, and multi-step price forecasts with confidence intervals for actionable insights.

---

## 1. Project Objective

Develop a reliable and interpretable forecasting system for AAPL stock prices that can:

- Analyze historical stock price trends and seasonality
- Automatically select optimal ARIMA or SARIMA parameters for prediction
- Visualize forecasts with confidence intervals for uncertainty quantification
- Evaluate model accuracy using RMSE, MAE, MAPE, and R² metrics
- Generate buy/hold recommendations based on smoothed forecasted price changes
- Save forecasting results for further analysis or reporting

---

## 2. Dataset Information

- **Source**: Historical AAPL stock data (CSV file, daily prices)
- **Records**: Over 10,800 records, spanning from December 1980 to June 2022
- **Fields**: Date, Open, High, Low, Close, Volume (Date set as index)
- **Frequency**: Business days with forward fill for missing days to maintain consistency

---

## 3. Methodology

### Data Preparation and EDA

- Loads and preprocesses Apple stock price data, setting business day frequency
- Plots stock price history to visualize market behavior over time
- Performs seasonal-trend decomposition to separate observed, trend, seasonal, and residual components

### Stationarity & Correlation Analysis

- Conducts Augmented Dickey-Fuller (ADF) tests to assess stationarity; applies differencing if needed
- Generates ACF and PACF plots to identify AR and MA orders and detect seasonality patterns

### Model Building & Selection

- Uses `pmdarima`’s `auto_arima` for automated ARIMA order selection based on AIC
- Implements Seasonal ARIMA (SARIMA) model to incorporate monthly seasonality (~22 business days)
- Compares ARIMA and SARIMA models using RMSE, MAE, MAPE, and R² on a hold-out test set to select the best model

### Forecast Generation & Visualization

- Plots actual vs predicted prices with 95% confidence intervals for test period
- Produces 30-day ahead price forecast with uncertainty band to anticipate short-term trends
- Summarizes model performance metrics

### Buy Recommendation

- Uses smoothed average forecast price over 5 days to provide buy/hold investment recommendations

---

## 4. Key Features Implemented

### Core Functionality

- Automated hyperparameter tuning for ARIMA and SARIMA models
- Stationarity checks and seasonal-trend decomposition
- Detailed visualizations: EDA, decomposition, ACF/PACF, forecast overlays
- Model performance comparison and results export
- Buy/hold signal generation based on price forecast

### Technical Features

- Business-day frequency time index with missing data imputation
- Confidence intervals derived from statistical forecasting models
- Comprehensive metrics for evaluation: RMSE, MAE, MAPE, R²
- Modular Python implementation using `pandas`, `matplotlib`, `statsmodels`, and `pmdarima`

---

## 5. Project Setup and Requirements

### Requirements

- Python 3.x
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- pmdarima
- statsmodels

### Installation

Install dependencies by running:

```bash
 pip install pandas numpy matpnumpy pandas matplotlib seaborn scikit-learn pmdarima statsmodels
```


---

## 6. Running the Project

1. Place your AAPL stock price CSV file (`AAPL.csv`) in the project directory.
2. Run the main script:

```bash
python NSE.py
```

### The system will:

- Load and preprocess stock price data
- Perform exploratory and stationarity analyses
- Automatically fit ARIMA and SARIMA models
- Generate visual performance comparisons and forecasts
- Save forecast outputs and generate buy/hold recommendations

---

## 7. Output Files

- `forecast_results/AAPL_arima_forecast.csv` — Forecasted vs actual prices with error
- `forecast_results/arima_model_comparison.csv` — Performance metrics comparison of models

---

## 8. Visualization Overview
A comprehensive set of visualizations supporting this project is provided separately in the [Visualization Document](Visualization.md). This document includes detailed descriptions and analyses of all key plots

### Accessing Visualizations

The actual plot images referenced in the visualization document are stored in the `/images` directory within the project repository.

We recommend reviewing the visualization document alongside the main README for a thorough understanding of the model's performance and insightful data interpretations.

---

## 9. Future Enhancements

### Technical Improvements

- Implement cross-validation with rolling window to better validate model stability
- Explore advanced models such as SARIMAX with exogenous variables or deep learning-based forecasting (LSTM)
- Automate seasonal period detection for SARIMA without manual tuning

### Business Applications

- Integration with real-time data feeds for live forecasting
- Develop a dashboard interface to visualize forecasts and signals interactively
- Extend to multi-stock portfolio forecasting and risk analysis

### User Experience

- Create an interactive web app or notebook for user-friendly exploration
- Add alert notifications for buy signals based on forecast thresholds

---

## 10. Contact

For questions or collaboration:

- **Name**: Ghanashyam T V
- **Email**: [ghanashyamtv16@gmail.com](mailto:ghanashyamtv16@gmail.com)
- **LinkedIn**: [linkedin.com/in/ghanashyam-tv](https://www.linkedin.com/in/ghanashyam-tv)

---

Thank you for exploring the Apple Stock Price Forecasting project using ARIMA and SARIMA models! This project offers a practical guide to applied time series forecasting with real historical market data.
