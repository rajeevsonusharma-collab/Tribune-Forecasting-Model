"""
The Tribune Trust — Forecasting Dashboard
------------------------------------------
Interactive Streamlit app for forecasting the 5 synthetic metrics
(print circulation, digital visitors, ad revenue, subscription revenue,
total revenue) using SARIMA, Holt-Winters, or a lag-feature Random Forest.

Run with:
    streamlit run streamlit_app.py

Expects these files in the same folder (all produced by
tribune_forecasting_model.ipynb):
    tribune_trust_synthetic_data.csv
    models/sarima_<metric>.joblib   (optional — app fits live if missing)
    model_comparison_metrics.csv    (optional — shown in the sidebar)
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

DATA_PATH = "tribune_trust_synthetic_data.csv"
MODELS_DIR = "models"

METRICS = {
    "Total_Revenue_INR_Lakh": "Total Revenue (INR Lakh)",
    "Ad_Revenue_INR_Lakh": "Ad Revenue (INR Lakh)",
    "Subscription_Revenue_INR_Lakh": "Subscription Revenue (INR Lakh)",
    "Print_Circulation_Copies_Per_Day": "Print Circulation (copies/day)",
    "Digital_Unique_Visitors": "Digital Unique Visitors",
}

st.set_page_config(page_title="The Tribune Trust — Forecast", layout="wide")


# ----------------------------------------------------------------------
# Data & model helpers (cached)
# ----------------------------------------------------------------------
@st.cache_data
def load_data(path: str, uploaded_bytes: bytes | None) -> pd.DataFrame:
    if uploaded_bytes is not None:
        df = pd.read_csv(pd.io.common.BytesIO(uploaded_bytes), parse_dates=["Month"])
    else:
        df = pd.read_csv(path, parse_dates=["Month"])
    df = df.set_index("Month").asfreq("MS")
    return df


@st.cache_resource(show_spinner=False)
def get_sarima_fit(series: pd.Series, metric: str):
    """Load a pre-trained SARIMA model if available, else fit on the fly."""
    saved_path = os.path.join(MODELS_DIR, f"sarima_{metric}.joblib")
    if os.path.exists(saved_path):
        try:
            return joblib.load(saved_path)
        except Exception:
            pass
    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def forecast_sarima(series: pd.Series, metric: str, horizon: int):
    fit = get_sarima_fit(series, metric)
    fc = fit.get_forecast(steps=horizon)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.2)
    future_index = pd.date_range(series.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    mean.index = future_index
    ci.index = future_index
    return mean, ci.iloc[:, 0], ci.iloc[:, 1]


def forecast_holt_winters(series: pd.Series, horizon: int):
    fit = ExponentialSmoothing(
        series, trend="add", seasonal="add", seasonal_periods=12
    ).fit()
    mean = fit.forecast(horizon)
    future_index = pd.date_range(series.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    mean.index = future_index
    resid_std = np.std(fit.resid)
    lower = mean - 1.28 * resid_std
    upper = mean + 1.28 * resid_std
    return mean, lower, upper


def make_features(series: pd.Series, lags=(1, 2, 3, 12), roll_window=3) -> pd.DataFrame:
    feat = pd.DataFrame({"y": series})
    for lag in lags:
        feat[f"lag_{lag}"] = series.shift(lag)
    feat[f"roll_mean_{roll_window}"] = series.shift(1).rolling(roll_window).mean()
    feat["month"] = feat.index.month
    feat["quarter"] = feat.index.quarter
    return feat.dropna()


def forecast_random_forest(series: pd.Series, horizon: int):
    """Recursive multi-step forecast: predict one step, append it, repeat."""
    lags = (1, 2, 3, 12)
    roll_window = 3
    feat_df = make_features(series, lags, roll_window)
    feature_cols = [c for c in feat_df.columns if c != "y"]

    rf = RandomForestRegressor(n_estimators=400, max_depth=6, random_state=42)
    rf.fit(feat_df[feature_cols], feat_df["y"])

    history = series.copy()
    preds = []
    future_index = pd.date_range(series.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    for month in future_index:
        row = {}
        for lag in lags:
            row[f"lag_{lag}"] = history.iloc[-lag]
        row[f"roll_mean_{roll_window}"] = history.iloc[-roll_window:].mean()
        row["month"] = month.month
        row["quarter"] = month.quarter
        x = pd.DataFrame([row])[feature_cols]
        y_hat = rf.predict(x)[0]
        preds.append(y_hat)
        history.loc[month] = y_hat

    mean = pd.Series(preds, index=future_index)
    # RF has no native interval — approximate with tree-spread
    tree_preds = np.stack([
        [est.predict(pd.DataFrame([{
            **{f"lag_{lag}": (series if i == 0 else pd.concat([series, mean.iloc[:i]]))
                .iloc[-lag] for lag in lags},
            f"roll_mean_{roll_window}": (series if i == 0 else pd.concat([series, mean.iloc[:i]]))
                .iloc[-roll_window:].mean(),
            "month": future_index[i].month,
            "quarter": future_index[i].quarter,
        }])[feature_cols])[0] for est in rf.estimators_]
        for i in range(horizon)
    ])
    lower = pd.Series(np.percentile(tree_preds, 10, axis=1), index=future_index)
    upper = pd.Series(np.percentile(tree_preds, 90, axis=1), index=future_index)
    return mean, lower, upper


# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.title("Forecast settings")

uploaded = st.sidebar.file_uploader(
    "Optional: upload your own CSV (same columns)", type=["csv"]
)
uploaded_bytes = uploaded.read() if uploaded is not None else None

df = load_data(DATA_PATH, uploaded_bytes)

metric = st.sidebar.selectbox(
    "Metric to forecast", options=list(METRICS.keys()), format_func=lambda k: METRICS[k]
)
model_choice = st.sidebar.radio(
    "Model", ["SARIMA", "Holt-Winters", "Random Forest (lag features)"], index=0
)
horizon = st.sidebar.slider("Forecast horizon (months)", min_value=3, max_value=24, value=12)

st.sidebar.markdown("---")
if os.path.exists("model_comparison_metrics.csv"):
    st.sidebar.caption("Test-set accuracy (Total Revenue, from notebook)")
    st.sidebar.dataframe(pd.read_csv("model_comparison_metrics.csv"), hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------
# Main panel
# ----------------------------------------------------------------------
st.title("The Tribune Trust — Forecasting Dashboard")
st.caption("Synthetic data · SARIMA / Holt-Winters / Random Forest forecasts · not real financials")

series = df[metric].dropna()

with st.spinner(f"Fitting {model_choice} on {METRICS[metric]}..."):
    if model_choice == "SARIMA":
        mean, lower, upper = forecast_sarima(series, metric, horizon)
    elif model_choice == "Holt-Winters":
        mean, lower, upper = forecast_holt_winters(series, horizon)
    else:
        mean, lower, upper = forecast_random_forest(series, horizon)

col1, col2, col3 = st.columns(3)
col1.metric("Last actual value", f"{series.iloc[-1]:,.1f}", help=str(series.index[-1].date()))
col2.metric(f"Forecast — month {horizon}", f"{mean.iloc[-1]:,.1f}")
pct_change = (mean.iloc[-1] / series.iloc[-1] - 1) * 100
col3.metric("Change vs last actual", f"{pct_change:+.1f}%")

fig = go.Figure()
fig.add_trace(go.Scatter(x=series.index, y=series.values, name="History", line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(x=mean.index, y=mean.values, name=f"{model_choice} forecast", line=dict(color="#ff7f0e")))
fig.add_trace(go.Scatter(
    x=list(mean.index) + list(mean.index[::-1]),
    y=list(upper.values) + list(lower.values[::-1]),
    fill="toself",
    fillcolor="rgba(255,127,14,0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    name="Interval",
    showlegend=True,
))
fig.update_layout(
    title=f"{METRICS[metric]} — history & {horizon}-month forecast",
    xaxis_title="Month",
    yaxis_title=METRICS[metric],
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=60),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Forecast table")
forecast_table = pd.DataFrame({
    "Month": mean.index.strftime("%Y-%m"),
    "Forecast": mean.values.round(1),
    "Lower": lower.values.round(1),
    "Upper": upper.values.round(1),
})
st.dataframe(forecast_table, hide_index=True, use_container_width=True)

csv_bytes = forecast_table.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download forecast as CSV",
    data=csv_bytes,
    file_name=f"{metric}_forecast_{model_choice.split()[0].lower()}.csv",
    mime="text/csv",
)

with st.expander("Show raw historical data"):
    st.dataframe(df.reset_index(), hide_index=True, use_container_width=True)
