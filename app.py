import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.graph_objects as go

# -------------------------
# Load model from Google Drive
# -------------------------
@st.cache_resource
def load_model_from_url(url, local_filename="sarima_model.pkl"):
    if not os.path.exists(local_filename):
        st.write("Downloading pre-trained model...")
        response = requests.get(url)
        with open(local_filename, "wb") as f:
            f.write(response.content)
        st.write("Model downloaded.")
    else:
        st.write("Using cached model.")
    return joblib.load(local_filename)

# Replace with your actual file ID
google_drive_url = "https://drive.google.com/uc?export=download&id=11fmreFztoPmZCubd2SUIBFlqp6GXozIY"
model_fit = load_model_from_url(google_drive_url)

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("chocolate_sales.csv", parse_dates=["date"])
df.set_index("date", inplace=True)
train = df.iloc[:-52]
test = df.iloc[-52:]

# -------------------------
# Forecasts
# -------------------------
def generate_forecast(model, steps, start_date):
    forecast_result = model.get_forecast(steps=steps)
    forecast = forecast_result.predicted_mean.round(2)
    conf_int = forecast_result.conf_int().round(2)
    forecast.index = pd.date_range(start=start_date, periods=steps, freq="W-SUN")
    conf_int.index = forecast.index
    return forecast, conf_int

forecast_2025, conf_int_2025 = generate_forecast(model_fit, 52, start_date="2025-01-05")
forecast_2024, _ = generate_forecast(model_fit, 52, start_date=test.index[0])

# -------------------------
# App Layout
# -------------------------
st.set_page_config(layout="wide")
st.title("Chocolate Sales Forecast")

tabs = st.tabs([
    "2025 Forecast",
    "2024 Model Evaluation",
    "Historical Lookup",
])

# -------------------------
# TAB 1: Forecast
# -------------------------
with tabs[0]:
    st.subheader("Projected Weekly Sales for 2025")

    # Forecast Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_2025.index, y=forecast_2025, mode="lines", name="Forecast", line=dict(color="blue")))
    fig.add_trace(go.Scatter(
        x=list(forecast_2025.index) + list(forecast_2025.index[::-1]),
        y=list(conf_int_2025.iloc[:, 0]) + list(conf_int_2025.iloc[:, 1][::-1]),
        fill="toself", fillcolor="rgba(0,0,255,0.2)",
        line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", name="90% Confidence Interval"
    ))
    fig.update_layout(
        title="2025 Weekly Forecast",
        xaxis_title="Week",
        yaxis_title="Sales ($)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Week Lookup
    st.subheader("Look Up a Specific Week")
    selected_date = st.date_input("Choose a week in 2025:", min_value=forecast_2025.index.min().date(), max_value=forecast_2025.index.max().date(), value=forecast_2025.index.min().date(), key="calendar_2025")
    selected_date = pd.to_datetime(selected_date)

    if selected_date in forecast_2025.index:
        selected_value = forecast_2025[selected_date]
        selected_ci = conf_int_2025.loc[selected_date]
        st.metric("Forecasted Sales", f"${selected_value:.2f}")
        st.write(f"90% Confidence Interval: **${selected_ci[0]:.2f} – ${selected_ci[1]:.2f}**")
    else:
        st.warning("Invalid date selection. Please pick a Sunday in 2025.")

    # Summary Stats
    st.subheader("2025 Forecast Summary")
    total_sales = forecast_2025.sum()
    avg_sales = forecast_2025.mean()
    min_sales = forecast_2025.min()
    max_sales = forecast_2025.max()
    min_week = forecast_2025.idxmin().date()
    max_week = forecast_2025.idxmax().date()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Forecast Sales", f"${total_sales:,.2f}")
    col2.metric("Average Weekly Sales", f"${avg_sales:.2f}")
    col3.metric("Lowest Week", f"${min_sales:.2f}", f"{min_week}")
    col4.metric("Highest Week", f"${max_sales:.2f}", f"{max_week}")

    # Download
    st.subheader("Download Forecast")
    download_df = pd.DataFrame({
        "date": forecast_2025.index,
        "forecasted_sales": forecast_2025.values,
        "ci_lower": conf_int_2025.iloc[:, 0].values,
        "ci_upper": conf_int_2025.iloc[:, 1].values
    }).set_index("date")
    csv = download_df.to_csv().encode("utf-8")
    st.download_button("Download CSV", csv, "chocolate_forecast_2025.csv", "text/csv")

# -------------------------
# TAB 2: Evaluation on 2024
# -------------------------
with tabs[1]:
    st.subheader("Model Evaluation on 2024 Test Set")
    r2 = r2_score(test["sales"], forecast_2024)
    mse = mean_squared_error(test["sales"], forecast_2024)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test["sales"], forecast_2024)
    mape = np.mean(np.abs((test["sales"] - forecast_2024) / test["sales"])) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{r2:.3f}")
    col2.metric("RMSE", f"${rmse:.2f}")
    col3.metric("MAE", f"${mae:.2f}")
    col4.metric("MAPE", f"{mape:.2f}%")

    fig_eval = go.Figure()
    fig_eval.add_trace(go.Scatter(x=test.index, y=test["sales"], name="Actual", line=dict(color="black")))
    fig_eval.add_trace(go.Scatter(x=test.index, y=forecast_2024, name="Forecast", line=dict(color="blue")))
    fig_eval.update_layout(title="Forecast vs Actual (2024)", xaxis_title="Week", yaxis_title="Sales ($)", hovermode="x unified")
    st.plotly_chart(fig_eval, use_container_width=True)

# -------------------------
# TAB 3: Historical Lookup
# -------------------------
with tabs[2]:
    st.subheader("Look Up Historical Weekly Sales")
    hist_date = st.date_input("Select a historical week:", min_value=df.index.min().date(), max_value=df.index.max().date(), value=df.index.max().date(), key="hist_date")
    hist_date = pd.to_datetime(hist_date)

    if hist_date in df.index:
        hist_value = df.loc[hist_date, "sales"]
        st.metric("Recorded Sales", f"${hist_value:.2f}")
    else:
        st.warning("Date not found in dataset. Please select a valid Sunday.")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 0.85em;">
© 2024 The Forecast Company. All Rights Reserved.  
<br>
📧 <a href="mailto:theforecastcompany@gmail.com">theforecastcompany@gmail.com</a> &nbsp;|&nbsp;
📞 <a href="tel:8563040922">(856) 304-0922</a>
</div>
""", unsafe_allow_html=True)


