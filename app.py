# Rewriting the full Streamlit app using SARIMAX (Seasonal ARIMA)
# With: order = (3, 0, 0), seasonal_order = (1, 0, 2, 52)

# Full SARIMA Streamlit app with performance improvements via caching

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats

st.title("Chocolate Sales Forecast (SARIMA Model)")

st.write("⏳ Loading data...")
df = pd.read_csv("chocolate_sales.csv", parse_dates=["date"])
df.set_index("date", inplace=True)
st.write(f"✅ Data loaded! Shape: {df.shape}")

st.write("⏳ Splitting data into train/test...")
train = df.iloc[:-52]
test = df.iloc[-52:]
st.write(f"✅ Training set shape: {train.shape}")
st.write(f"✅ Test set shape: {test.shape}")

order = (3, 0, 0)
seasonal_order = (1, 0, 2, 52)

@st.cache_resource
def train_sarima_model(series, order, seasonal_order):
    st.write("⏳ Training SARIMA model...")
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    st.write("✅ Model training complete.")
    return model_fit

model_fit = train_sarima_model(train["sales"], order, seasonal_order)

@st.cache_data
def get_forecast(model_fit, steps, start_date):
    st.write(f"⏳ Generating forecast for {steps} weeks starting {start_date}...")
    forecast_result = model_fit.get_forecast(steps=steps)
    forecast = forecast_result.predicted_mean.round(2)
    conf_int = forecast_result.conf_int().round(2)
    forecast.index = pd.date_range(start=start_date, periods=steps, freq='W-SUN')
    conf_int.index = forecast.index
    st.write("✅ Forecast generated.")
    return forecast, conf_int

forecast_2025, conf_int_2025 = get_forecast(model_fit, 52, start_date="2025-01-05")
forecast_2024, _ = get_forecast(model_fit, 52, start_date=test.index[0])

st.write("✅ Setup complete — ready to display tabs and charts.")

tabs = st.tabs([
    "2025 Forecast & Summary",
    "2024 Model Evaluation",
    "Residual Diagnostics",
    "Historical Sales Lookup"
])

with tabs[0]:
    st.subheader("Forecasted Chocolate Sales for 2025")

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=forecast_2025.index, y=forecast_2025, mode="lines", name="Forecast", line=dict(color="blue")))
    fig_forecast.add_trace(go.Scatter(
        x=list(forecast_2025.index) + list(forecast_2025.index[::-1]),
        y=list(conf_int_2025.iloc[:, 0]) + list(conf_int_2025.iloc[:, 1][::-1]),
        fill="toself", fillcolor="rgba(0,0,255,0.2)",
        line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
        name="90% Confidence Interval"
    ))
    fig_forecast.update_layout(
        title="Projected Chocolate Sales (2025)",
        xaxis_title="Week",
        yaxis_title="Sales ($)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.subheader("Select a Week in 2025")
    selected_date = st.date_input(
        "Choose a forecast week:",
        min_value=forecast_2025.index.min().date(),
        max_value=forecast_2025.index.max().date(),
        value=forecast_2025.index.min().date(),
        key="forecast_date"
    )
    selected_date = pd.to_datetime(selected_date)

    if selected_date not in forecast_2025.index:
        st.warning("Please select a valid forecast week in 2025.")
    else:
        selected_forecast = forecast_2025[selected_date]
        selected_ci = conf_int_2025.loc[selected_date]
        st.metric("Forecasted Sales", f"{selected_forecast:.2f}")
        st.write(f"90% Confidence Interval: **[{selected_ci[0]:.2f}, {selected_ci[1]:.2f}]**")

    st.subheader("2025 Forecast Summary")
    total_sales = forecast_2025.sum()
    avg_sales = forecast_2025.mean()
    min_sales = forecast_2025.min()
    max_sales = forecast_2025.max()
    min_week = forecast_2025.idxmin().date()
    max_week = forecast_2025.idxmax().date()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Forecast Sales", f"{total_sales:,.2f}")
    col2.metric("Average Weekly Sales", f"{avg_sales:.2f}")
    col3.metric("Min Weekly Sales", f"{min_sales:.2f}", f"Week of {min_week}")
    col4.metric("Max Weekly Sales", f"{max_sales:.2f}", f"Week of {max_week}")

    download_df = pd.DataFrame({
        "date": forecast_2025.index,
        "forecasted_sales": forecast_2025.values,
        "ci_lower_90": conf_int_2025.iloc[:, 0].values,
        "ci_upper_90": conf_int_2025.iloc[:, 1].values
    }).set_index("date")

    csv = download_df.to_csv().encode('utf-8')
    st.download_button("Download 2025 Forecast as CSV", csv, "chocolate_sales_forecast_2025.csv", "text/csv")

