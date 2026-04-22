import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)


def main():
    # ====== 1. DATA LOADING & PREPROCESSING ======
    file_path = "AAPL.csv"  # Use actual filename
    df = pd.read_csv(file_path)
    df = df.rename(columns={
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    df = df.asfreq('B')  # Business day frequency
    df['close'].fillna(method='ffill', inplace=True)  # Fill missing days
    print(f"Data loaded: {df.shape[0]} records, {df.index.min()} to {df.index.max()}")

    all_stock_data = {'AAPL': df}

    # ====== 2. EXPLORATORY DATA ANALYSIS ======
    exploratory_data_analysis(all_stock_data)

    # ====== 3. TREND DECOMPOSITION ======
    trend_decomposition_analysis(all_stock_data)

    # ====== 4. STATIONARITY ANALYSIS ======
    stationarity_results = stationarity_analysis(all_stock_data)

    # ====== 5. ACF & PACF ======
    acf_pacf_analysis(all_stock_data)

    # ====== 6. MODEL SELECTION & FORECASTING ======
    forecast_results = {}
    for stock, data in all_stock_data.items():
        print(f"\n{'='*50}\nFORECASTING with Auto-ARIMA and SARIMA: {stock}\n{'='*50}")
        series = data['close'].dropna()

        # Auto-ARIMA forecasting
        auto_model, auto_train, auto_test, auto_forecast = auto_arima_forecasting(series)
        auto_metrics = calculate_all_metrics(auto_test, auto_forecast)
        print(f"Auto-ARIMA Metrics: RMSE={auto_metrics['RMSE']:.2f}, MAE={auto_metrics['MAE']:.2f}, MAPE={auto_metrics['MAPE']:.2f}%")

        # SARIMA forecasting
        sarima_model, sarima_train, sarima_test, sarima_forecast = sarima_forecasting(series, seasonal_period=22)
        sarima_metrics = calculate_all_metrics(sarima_test, sarima_forecast)
        print(f"SARIMA Metrics: RMSE={sarima_metrics['RMSE']:.2f}, MAE={sarima_metrics['MAE']:.2f}, MAPE={sarima_metrics['MAPE']:.2f}%")

        # Choose best model
        if auto_metrics['RMSE'] <= sarima_metrics['RMSE']:
            best_model, train, test, forecast, best_metrics = auto_model, auto_train, auto_test, auto_forecast, auto_metrics
            best_order_desc = f"Auto-ARIMA {auto_model.order}"
        else:
            best_model, train, test, forecast, best_metrics = sarima_model, sarima_train, sarima_test, sarima_forecast, sarima_metrics
            best_order_desc = f"SARIMA {sarima_model.model_orders}"

        forecast_results[stock] = {
            'order': best_order_desc,
            'rmse': best_metrics['RMSE'],
            'model': best_model,
            'forecast': forecast,
            'actual': test,
            'train': train,
            'metrics': best_metrics
        }

        print(f"\n🎯 BEST MODEL for {stock}: {best_order_desc} (RMSE: {best_metrics['RMSE']:.2f})")
        plot_forecast_with_ci(stock, series, test, forecast, best_model)
        plot_future_forecast(stock, series, test, best_model, best_order_desc)

    # ====== 7. COMPARATIVE ANALYSIS & SAVE ======
    comparative_analysis(forecast_results)
    save_results(forecast_results, all_stock_data)
    display_final_summary(forecast_results)
    generate_smooth_buy_recommendation(forecast_results, all_stock_data, threshold=0.01, days=5)


def exploratory_data_analysis(all_stock_data):
    plt.figure(figsize=(15, 6))
    for stock, data in all_stock_data.items():
        plt.plot(data.index, data['close'], label=stock, linewidth=2)
    plt.title('AAPL Stock Price History', fontsize=16, fontweight='bold')
    plt.ylabel('Price (USD)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def trend_decomposition_analysis(all_stock_data):
    from statsmodels.tsa.seasonal import seasonal_decompose
    for stock, data in all_stock_data.items():
        try:
            recent_data = data['close'].last('2Y').dropna()
            if len(recent_data) < 60:
                recent_data = data['close'].dropna()
            result = seasonal_decompose(recent_data, model='multiplicative', period=30)
            plt.figure(figsize=(16, 12))
            plt.suptitle(f'{stock} - Time Series Decomposition', fontsize=16, fontweight='bold', y=0.99)
            plt.subplot(4, 1, 1); plt.plot(result.observed); plt.title('Observed')
            plt.subplot(4, 1, 2); plt.plot(result.trend, color='orange'); plt.title('Trend')
            plt.subplot(4, 1, 3); plt.plot(result.seasonal, color='green'); plt.title('Seasonal')
            plt.subplot(4, 1, 4); plt.plot(result.resid, color='red'); plt.title('Residual')
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()
        except Exception as e:
            print(f"❌ Decomposition failed for {stock}: {e}")


def adf_test(timeseries, name=""):
    from statsmodels.tsa.stattools import adfuller
    print(f'\nResults for {name}:')
    dftest = adfuller(timeseries.dropna(), autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Obs Used'])
    for key, value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    print(dfoutput)
    if dftest[1] <= 0.05:
        print("✅ Stationary (reject null hypothesis)")
        return True
    else:
        print("❌ Non-stationary (cannot reject null hypothesis)")
        return False


def stationarity_analysis(all_stock_data):
    stationarity_results = {}
    for stock, data in all_stock_data.items():
        print(f"\n{'='*40}\nSTATIONARITY CHECK: {stock}\n{'='*40}")
        print("Original Series:")
        stat_orig = adf_test(data['close'], "Original Price")
        print("\nFirst Difference:")
        price_diff = data['close'].diff().dropna()
        stat_diff = adf_test(price_diff, "First Difference")
        stationarity_results[stock] = {'original_stationary': stat_orig, 'diff_stationary': stat_diff}
    return stationarity_results


def acf_pacf_analysis(all_stock_data):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    for stock, data in all_stock_data.items():
        price_diff = data['close'].diff().dropna()
        plt.figure(figsize=(16, 6)); plt.suptitle(f'{stock} - Autocorrelation Analysis', fontsize=16, fontweight='bold')
        plt.subplot(1, 2, 1); plot_acf(price_diff, lags=40, ax=plt.gca()); plt.title('ACF')
        plt.subplot(1, 2, 2); plot_pacf(price_diff, lags=40, ax=plt.gca()); plt.title('PACF')
        plt.tight_layout(); plt.show()


def calculate_all_metrics(actual, predicted):
    from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error
    return {
        'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
        'MAE': mean_absolute_error(actual, predicted),
        'MAPE': mean_absolute_percentage_error(actual, predicted) * 100,
        'R2': r2_score(actual, predicted),
        'MSE': mean_squared_error(actual, predicted)
    }


def auto_arima_forecasting(series, train_size=0.8):
    train_len = int(len(series) * train_size)
    train, test = series[:train_len], series[train_len:]
    print(f"Auto-ARIMA training from {train.index.min()} to {train.index.max()}")
    model = pm.auto_arima(train, seasonal=False, stepwise=True, trace=True,
                          error_action='ignore', suppress_warnings=True)
    forecast = model.predict(n_periods=len(test))
    forecast = pd.Series(forecast, index=test.index)
    return model, train, test, forecast


def sarima_forecasting(series, seasonal_period=22, train_size=0.8):
    train_len = int(len(series) * train_size)
    train, test = series[:train_len], series[train_len:]
    print(f"SARIMA training from {train.index.min()} to {train.index.max()}, seasonal_period={seasonal_period}")
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_period)).fit(disp=False)
    forecast = model.get_forecast(steps=len(test)).predicted_mean
    forecast = pd.Series(forecast, index=test.index)
    return model, train, test, forecast


def plot_forecast_with_ci(stock, series, test, forecast, model):
    plt.figure(figsize=(15, 8))
    plt.plot(series.index, series, label='Historical', color='blue', alpha=0.7)
    plt.plot(test.index, test, label='Actual', color='green', linewidth=2)
    plt.plot(forecast.index, forecast, label='Forecast', color='red', linewidth=2)
    try:
        if hasattr(model, 'get_forecast'):
            conf_int = model.get_forecast(steps=len(test)).conf_int()
            plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3, label='95% CI')
        elif hasattr(model, 'conf_int'):
            conf_int = model.conf_int()
            plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3, label='95% CI')
    except Exception:
        try:
            pred_summary = model.predict(n_periods=len(test), return_conf_int=True)
            conf_int = pred_summary[1]
            plt.fill_between(test.index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3, label='95% CI')
        except Exception:
            pass
    plt.title(f'{stock} Forecast vs Actual with 95% Confidence Interval', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_future_forecast(stock, series, test, model, order_desc):
    future_forecast = model.get_forecast(steps=30)
    future_values = future_forecast.predicted_mean
    conf_int = future_forecast.conf_int()
    from pandas.tseries.offsets import BDay
    future_dates = pd.date_range(start=test.index[-1] + BDay(1), periods=30, freq='B')
    plt.figure(figsize=(15, 8))
    recent_data = series[series.index > (series.index[-1] - pd.Timedelta(days=365))]
    plt.plot(recent_data.index, recent_data, label='Recent Data', color='blue')
    plt.plot(future_dates, future_values, label='30-Day Forecast', color='orange', linewidth=3)
    plt.fill_between(future_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.2, label='95% CI')
    plt.title(f' AAPL - 30-Day Price Forecast ', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def comparative_analysis(forecast_results):
    if not forecast_results:
        print("❌ No forecast results available for comparison")
        return
    comparison_data = []
    for stock, results in forecast_results.items():
        comparison_data.append({
            'Stock': stock,
            'ARIMA Order': results['order'],
            'RMSE (USD)': results['rmse'],
            'MAE (USD)': results['metrics']['MAE'],
            'MAPE (%)': results['metrics']['MAPE'],
            'R²': results['metrics']['R2'],
            'AIC': results['model'].aic
        })
    comparison_df = pd.DataFrame(comparison_data).sort_values('RMSE (USD)')
    print("📊 FORECASTING PERFORMANCE COMPARISON:")
    print(comparison_df.to_string(index=False))
    plt.figure(figsize=(7, 4))
    plt.bar(comparison_df['Stock'], comparison_df['RMSE (USD)'], color='#2E86AB')
    plt.title('Forecasting Accuracy (Lower RMSE is Better)', fontsize=14, fontweight='bold')
    plt.ylabel('RMSE (USD)'); plt.xlabel('Stock'); plt.xticks(rotation=45); plt.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(comparison_df['RMSE (USD)']):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout(); plt.show()
    print("\n💡 INSIGHTS:")
    print(f"- Best forecasting performance: {comparison_df.iloc[0]['Stock']} (RMSE: ${comparison_df.iloc[0]['RMSE (USD)']:.2f})")


def save_results(forecast_results, all_stock_data):
    if not forecast_results:
        print("❌ No forecast results to save")
        return
    os.makedirs('forecast_results', exist_ok=True)
    for stock, results in forecast_results.items():
        actual_test = results['actual']
        forecast = results['forecast']
        forecast_df = pd.DataFrame({
            'date': actual_test.index,
            'actual_price': actual_test.values,
            'forecast_price': forecast.values,
            'error': actual_test.values - forecast.values
        })
        filename = f"forecast_results/{stock}_arima_forecast.csv"
        forecast_df.to_csv(filename, index=False)
        print(f"✅ Saved {stock} forecast results to {filename}")
    comparison_data = []
    for stock, results in forecast_results.items():
        comparison_data.append({
            'Stock': stock,
            'ARIMA_Order': str(results['order']),
            'RMSE': results['rmse'],
            'MAE': results['metrics']['MAE'],
            'MAPE': results['metrics']['MAPE'],
            'R2': results['metrics']['R2'],
            'AIC': results['model'].aic
        })
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('forecast_results/arima_model_comparison.csv', index=False)
    print("✅ Saved model comparison results to 'forecast_results/arima_model_comparison.csv'")


def display_final_summary(forecast_results):
    print("\n" + "="*60 + "\nFINAL PERFORMANCE SUMMARY\n" + "="*60)
    if forecast_results:
        for stock, results in forecast_results.items():
            print(f"{stock}: {results['order']}, RMSE=${results['rmse']:.2f}, MAE=${results['metrics']['MAE']:.2f}, MAPE={results['metrics']['MAPE']:.2f}%, R2={results['metrics']['R2']:.4f}")
    print("\n🎯 PROJECT COMPLETED SUCCESSFULLY!")
    print("📊 Analyzed stock with comprehensive ARIMA and SARIMA modeling")
    print("📈 Generated forecasts with performance metrics")
    print("="*60)


def generate_smooth_buy_recommendation(forecast_results, all_stock_data, threshold=0.01, days=5):
    from pandas.tseries.offsets import BDay
    print("\n" + "="*60)
    print("SMOOTHED STOCK BUY RECOMMENDATIONS (5-DAY AVG)")
    print("="*60)
    for stock, results in forecast_results.items():
        latest_price = all_stock_data[stock]['close'].iloc[-1]
        forecast = results['model'].get_forecast(steps=days)
        forecasted_prices = forecast.predicted_mean.values
        avg_forecast_price = forecasted_prices.mean()
        change_pct = (avg_forecast_price - latest_price) / latest_price
        if change_pct > threshold:
            decision = "BUY"
        else:
            decision = "HOLD / NO BUY"
        print(f"{stock}: Latest Price = ${latest_price:.2f}, Average Forecast (Next {days} days) = ${avg_forecast_price:.2f}")
        print(f"Expected Change: {change_pct*100:.2f}% => Recommendation: {decision}\n")


if __name__ == "__main__":
    main()
