import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AAPL Stock Forecaster",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #2E86AB; text-align: center; margin-bottom: 0.5rem; font-weight: bold; }
    .sub-header { text-align: center; color: #6c757d; margin-bottom: 2rem; font-size: 1rem; }
    .metric-card { background-color: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #2E86AB; }
    .buy-signal { background-color: #d4edda; padding: 1.5rem; border-radius: 8px; border-left: 6px solid #28a745; font-size: 1.2rem; }
    .hold-signal { background-color: #fff3cd; padding: 1.5rem; border-radius: 8px; border-left: 6px solid #ffc107; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📈 Apple Stock Price Forecaster</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Time Series Forecasting with ARIMA/SARIMA — BUY / HOLD Signal</div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_dir, "AAPL.csv"))
    df = df.rename(columns={
        'Date': 'date', 'Open': 'open', 'High': 'high',
        'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    df = df.asfreq('B')
    df['close'] = df['close'].ffill()
    return df

@st.cache_resource
def run_forecast(series):
    import pmdarima as pm
    train_len = int(len(series) * 0.8)
    train, test = series[:train_len], series[train_len:]
    model = pm.auto_arima(
        train, seasonal=False, stepwise=True,
        error_action='ignore', suppress_warnings=True
    )
    forecast = model.predict(n_periods=len(test))
    forecast = pd.Series(forecast, index=test.index)
    return model, train, test, forecast

@st.cache_resource
def run_future_forecast(_model, test_index):
    from pandas.tseries.offsets import BDay
    future_forecast = _model.get_forecast(steps=30)
    future_values = future_forecast.predicted_mean
    conf_int = future_forecast.conf_int()
    future_dates = pd.date_range(start=test_index[-1] + BDay(1), periods=30, freq='B')
    return future_dates, future_values, conf_int

def calculate_metrics(actual, predicted):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
    return {
        'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
        'MAE': mean_absolute_error(actual, predicted),
        'MAPE': mean_absolute_percentage_error(actual, predicted) * 100,
        'R2': r2_score(actual, predicted)
    }

# Sidebar
st.sidebar.title("⚙️ Settings")
forecast_days = st.sidebar.slider("Forecast horizon (days)", 5, 30, 30)
threshold = st.sidebar.slider("BUY signal threshold (%)", 0.5, 5.0, 1.0) / 100
show_volume = st.sidebar.checkbox("Show Volume Chart", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.markdown("Uses Auto-ARIMA to forecast AAPL stock prices and generate BUY/HOLD signals.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Built by Ghanashyam T V**")
st.sidebar.markdown("[GitHub](https://github.com/shyam16843) | [LinkedIn](https://linkedin.com/in/ghanashyam-tv)")

# Load data
with st.spinner("Loading AAPL stock data..."):
    try:
        df = load_data()
        st.success(f"✅ Data loaded: {len(df)} trading days ({df.index.min().date()} to {df.index.max().date()})")
    except Exception as e:
        st.error(f"❌ Could not load AAPL.csv: {e}")
        st.stop()

# Stock overview
st.subheader("📊 Stock Price Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Latest Price", f"${df['close'].iloc[-1]:.2f}")
with col2:
    st.metric("52W High", f"${df['close'].tail(252).max():.2f}")
with col3:
    st.metric("52W Low", f"${df['close'].tail(252).min():.2f}")
with col4:
    change = df['close'].iloc[-1] - df['close'].iloc[-2]
    st.metric("1D Change", f"${change:.2f}", f"{(change/df['close'].iloc[-2])*100:.2f}%")

# Price chart
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df.index[-504:], df['close'].tail(504), color='#2E86AB', linewidth=1.5, label='AAPL Close Price')
ax.fill_between(df.index[-504:], df['close'].tail(504), alpha=0.1, color='#2E86AB')
ax.set_title('AAPL Stock Price — Last 2 Years', fontsize=14, fontweight='bold')
ax.set_ylabel('Price (USD)')
ax.grid(True, alpha=0.3)
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Volume chart
if show_volume and 'volume' in df.columns:
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.bar(df.index[-504:], df['volume'].tail(504), color='#6c757d', alpha=0.6)
    ax.set_title('Trading Volume — Last 2 Years', fontsize=12)
    ax.set_ylabel('Volume')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# Forecast
st.subheader("🔮 ARIMA Forecast")
if st.button("▶️ Run Forecast", type="primary", use_container_width=True):
    with st.spinner("Training Auto-ARIMA model... this may take 1-2 minutes ⏳"):
        try:
            series = df['close'].dropna()
            model, train, test, forecast = run_forecast(series)
            metrics = calculate_metrics(test, forecast)
            future_dates, future_values, conf_int = run_future_forecast(model, test.index)

            # Metrics
            st.subheader("📏 Model Performance")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("RMSE", f"${metrics['RMSE']:.2f}")
            with c2:
                st.metric("MAE", f"${metrics['MAE']:.2f}")
            with c3:
                st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            with c4:
                st.metric("R² Score", f"{metrics['R2']:.4f}")

            # Forecast vs Actual chart
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(train.index[-120:], train.tail(120), color='#2E86AB', label='Training Data', linewidth=1.5)
            ax.plot(test.index, test, color='green', label='Actual Price', linewidth=2)
            ax.plot(forecast.index, forecast, color='red', label='ARIMA Forecast', linewidth=2, linestyle='--')
            ax.set_title('ARIMA Forecast vs Actual Price', fontsize=14, fontweight='bold')
            ax.set_ylabel('Price (USD)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # 30-day future forecast
            st.subheader(f"📅 {forecast_days}-Day Future Forecast")
            fig, ax = plt.subplots(figsize=(14, 6))
            recent = series.tail(120)
            ax.plot(recent.index, recent, color='#2E86AB', label='Recent Price', linewidth=1.5)
            ax.plot(future_dates[:forecast_days], future_values[:forecast_days],
                    color='orange', label=f'{forecast_days}-Day Forecast', linewidth=2.5)
            ax.fill_between(
                future_dates[:forecast_days],
                conf_int.iloc[:forecast_days, 0],
                conf_int.iloc[:forecast_days, 1],
                color='orange', alpha=0.2, label='95% Confidence Interval'
            )
            ax.set_title(f'AAPL — {forecast_days}-Day Price Forecast', fontsize=14, fontweight='bold')
            ax.set_ylabel('Price (USD)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # BUY / HOLD signal
            st.subheader("💡 Trading Signal")
            latest_price = series.iloc[-1]
            avg_forecast = future_values[:5].mean()
            change_pct = (avg_forecast - latest_price) / latest_price

            if change_pct > threshold:
                st.markdown(f"""
                <div class="buy-signal">
                    <strong>🟢 BUY SIGNAL</strong><br>
                    Current Price: <strong>${latest_price:.2f}</strong><br>
                    Avg Forecast (Next 5 days): <strong>${avg_forecast:.2f}</strong><br>
                    Expected Change: <strong>+{change_pct*100:.2f}%</strong><br>
                    The model predicts upward movement above the {threshold*100:.1f}% threshold.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="hold-signal">
                    <strong>🟡 HOLD / NO BUY SIGNAL</strong><br>
                    Current Price: <strong>${latest_price:.2f}</strong><br>
                    Avg Forecast (Next 5 days): <strong>${avg_forecast:.2f}</strong><br>
                    Expected Change: <strong>{change_pct*100:.2f}%</strong><br>
                    The model does not predict sufficient upward movement.
                </div>
                """, unsafe_allow_html=True)

            st.warning("⚠️ This is for educational purposes only. Not financial advice.")

            # Download forecast
            forecast_df = pd.DataFrame({
                'Date': future_dates[:forecast_days],
                'Forecasted_Price': future_values[:forecast_days].values,
                'Lower_CI': conf_int.iloc[:forecast_days, 0].values,
                'Upper_CI': conf_int.iloc[:forecast_days, 1].values
            })
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Forecast CSV", csv, "aapl_forecast.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Forecast failed: {e}")

st.markdown("---")
st.markdown("**Built by Ghanashyam T V** | [GitHub](https://github.com/shyam16843) | [LinkedIn](https://linkedin.com/in/ghanashyam-tv)")
