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

# ------------------------- Load Data -------------------------
df = pd.read_csv("chocolate_sales.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

# ------------------------- Logo and Title -------------------------
logo_url = "https://i.imgur.com/oDM4ECC.jpeg"
col1, col2 = st.columns([1, 8])
with col1:
    st.image(logo_url, use_container_width=True)
with col2:
    st.title("Chocolate Sales Forecast (SARIMA Model)")

# ------------------------- Split Train/Test -------------------------
train = df.iloc[:-52]
test = df.iloc[-52:]

# ------------------------- Model Parameters -------------------------
order = (3, 0, 0)
seasonal_order = (1, 0, 2, 52)

# ------------------------- Cache SARIMA Model -------------------------
@st.cache_resource
def train_sarima_model(series, order, seasonal_order):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    return model_fit

model_fit = train_sarima_model(train["sales"], order, seasonal_order)

# ------------------------- Cache Forecast Function -------------------------
@st.cache_data
def get_forecast(model_fit, steps, start_date):
    forecast_result = model_fit.get_forecast(steps=steps)
    forecast = forecast_result.predicted_mean.round(2)
    conf_int = forecast_result.conf_int().round(2)
    forecast.index = pd.date_range(start=start_date, periods=steps, freq='W-SUN')
    conf_int.index = forecast.index
    return forecast, conf_int

forecast_2025, conf_int_2025 = get_forecast(model_fit, 52, start_date="2025-01-05")
forecast_2024, _ = get_forecast(model_fit, 52, start_date=test.index[0])

# ------------------------- Tabs -------------------------
tabs = st.tabs([
    "2025 Forecast & Summary",
    "2024 Model Evaluation",
    "Residual Diagnostics",
    "Historical Sales Lookup"
])

# ------------------------- Tab 1: Forecast & Summary -------------------------
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
        st.write(f"90% Confidence Interval: **[{selected_ci[0]:.2f}_]()**
# ------------------------- Tab 2: 2024 Evaluation -------------------------
with tabs[1]:
    st.subheader("Model Performance on 2024 Actual Data")

    test_forecast = forecast_2024.copy()
    test_forecast.index = test.index
    test_forecast_rounded = test_forecast.round(2)

    r2 = r2_score(test["sales"], test_forecast_rounded)
    mse = mean_squared_error(test["sales"], test_forecast_rounded)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test["sales"], test_forecast_rounded)
    mape = np.mean(np.abs((test["sales"] - test_forecast_rounded) / test["sales"])) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{r2:.3f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("MAE", f"{mae:.2f}")
    col4.metric("MAPE", f"{mape:.2f}%")

    fig_eval = go.Figure()
    fig_eval.add_trace(go.Scatter(x=test.index, y=test["sales"], mode="lines", name="Actual", line=dict(color="black")))
    fig_eval.add_trace(go.Scatter(x=test_forecast.index, y=test_forecast_rounded, mode="lines", name="Forecast", line=dict(color="blue")))
    fig_eval.update_layout(
        title="Forecast vs Actual Sales (2024)",
        xaxis_title="Week",
        yaxis_title="Sales ($)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_eval, use_container_width=True)

# ------------------------- Tab 3: Residual Diagnostics -------------------------
with tabs[2]:
    st.subheader("Residual Diagnostics")
    residuals = model_fit.resid

    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(x=train.index, y=residuals, mode="lines", name="Residuals"))
    fig_resid.update_layout(title="Residuals Over Time", xaxis_title="Date", yaxis_title="Residual")
    st.plotly_chart(fig_resid, use_container_width=True)

    st.subheader("Residual Distribution")
    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
    ax_hist.hist(residuals, bins=20, edgecolor="k", alpha=0.7)
    ax_hist.set_title("Histogram of Residuals")
    ax_hist.set_xlabel("Residual")
    ax_hist.set_ylabel("Frequency")
    st.pyplot(fig_hist)

    fig_qq, ax_qq = plt.subplots(figsize=(6, 6))
    stats.probplot(residuals, dist="norm", plot=ax_qq)
    ax_qq.set_title("Q-Q Plot of Residuals")
    st.pyplot(fig_qq)

    fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
    plot_acf(residuals, ax=ax_acf, lags=40)
    ax_acf.set_title("Autocorrelation (ACF) of Residuals")
    st.pyplot(fig_acf)

# ------------------------- Tab 4: Historical Sales -------------------------
with tabs[3]:
    st.subheader("Historical Weekly Sales")

    fig_hist_sales = go.Figure()
    fig_hist_sales.add_trace(go.Scatter(
        x=df.index,
        y=df["sales"],
        mode="lines",
        name="Historical Sales",
        line=dict(color="black")
    ))
    fig_hist_sales.update_layout(
        title="All-Time Weekly Chocolate Sales",
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_hist_sales, use_container_width=True)

    st.subheader("Look Up Actual Sales for a Past Week")
    historical_date = st.date_input(
        "Choose a date to view historical sales:",
        min_value=df.index.min().date(),
        max_value=df.index.max().date(),
        value=df.index[-1].date(),
        key="history_date"
    )
    historical_date = pd.to_datetime(historical_date)

    if historical_date not in df.index:
        st.warning("Selected date is not in the dataset.")
    else:
        actual_sales = df.loc[historical_date, "sales"]
        st.metric("Actual Sales", f"{actual_sales:.2f}")

# ------------------------- Footer -------------------------
st.markdown("""
<style>
.footer {
    position: relative;
    bottom: 0;
    width: 100%;
    background-color: #f0f2f6;
    padding: 10px 20px;
    font-size: 0.9em;
    color: #555555;
    text-align: center;
    border-top: 1px solid #d3d3d3;
    margin-top: 20px;
}
.footer a {
    color: #1a73e8;
    text-decoration: none;
    margin: 0 8px;
}
.footer a:hover {
    text-decoration: underline;
}
</style>

<div class="footer">
    &copy; 2024 The Forecast Company. All Rights Reserved.  
    <br>
    Contact Us: 
    <a href="tel:+18563040922">856-304-0922</a> | 
    <a href="mailto:theforecastcompany@gmail.com">theforecastcompany@gmail.com</a>
</div>
""", unsafe_allow_html=True)

