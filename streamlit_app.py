"""
The Tribune Trust — Forecasting Dashboard
------------------------------------------
Interactive Streamlit app for forecasting the 5 synthetic metrics
(print circulation, digital visitors, ad revenue, subscription revenue,
total revenue) using SARIMA, Holt-Winters, or a lag-feature Random Forest.

Run with:
    streamlit run streamlit_app.py

Set APP_PASSWORD in the environment or Streamlit secrets before starting.

Expects these files in the same folder (all produced by
tribune_forecasting_model.ipynb):
    tribune_trust_synthetic_data.csv
    models/sarima_<metric>.joblib   (optional — app fits live if missing)
    model_comparison_metrics.csv    (optional — shown in the sidebar)
"""

import hmac
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "tribune_trust_synthetic_data.csv"
MODELS_DIR = APP_DIR / "models"

METRICS = {
    "Total_Revenue_INR_Lakh": "Total Revenue (INR Lakh)",
    "Ad_Revenue_INR_Lakh": "Ad Revenue (INR Lakh)",
    "Subscription_Revenue_INR_Lakh": "Subscription Revenue (INR Lakh)",
    "Print_Circulation_Copies_Per_Day": "Print Circulation (copies/day)",
    "Digital_Unique_Visitors": "Digital Unique Visitors",
}

st.set_page_config(page_title="The Tribune Trust — Forecast", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --ink: #15343b;
        --muted: #60757a;
        --paper: #fffaf1;
        --panel: #ffffff;
        --teal: #087f8c;
        --coral: #f05a3c;
        --sun: #f6c453;
        --line: #e7ddd0;
    }

    .stApp {
        background:
            radial-gradient(circle at 92% 2%, rgba(246, 196, 83, 0.28), transparent 24rem),
            linear-gradient(135deg, #fffaf1 0%, #f4fbf8 48%, #fff7ee 100%);
        color: var(--ink);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #123f46 0%, #0b6872 65%, #0b7f84 100%);
        border-right: 0;
    }

    [data-testid="stSidebar"] * { color: #f7fffa; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="popover"] {
        background: rgba(255, 255, 255, 0.12);
        border-color: rgba(255, 255, 255, 0.34);
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        border: 1px dashed rgba(255, 255, 255, 0.55);
        background: rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebar"] [data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    .block-container { padding-top: 3rem; padding-bottom: 4rem; max-width: 1500px; }
    h1, h2, h3 { color: var(--ink); font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    h1 { font-size: clamp(2rem, 4vw, 4.25rem); line-height: 0.98; margin-bottom: 0.6rem; }
    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-top: 5px solid var(--teal);
        border-radius: 14px;
        box-shadow: 0 12px 30px rgba(21, 52, 59, 0.08);
        padding: 1.1rem 1.25rem;
    }
    [data-testid="stMetric"]:nth-child(2) { border-top-color: var(--coral); }
    [data-testid="stMetric"]:nth-child(3) { border-top-color: var(--sun); }
    [data-testid="stMetricLabel"] { color: var(--muted); font-family: 'DM Mono', monospace; text-transform: uppercase; font-size: 0.7rem; }
    [data-testid="stMetricValue"] { color: var(--ink); font-family: 'Space Grotesk', sans-serif; }
    [data-testid="stDownloadButton"] button {
        background: var(--coral);
        border: 0;
        border-radius: 999px;
        color: #fff;
        font-weight: 700;
        padding: 0.7rem 1.2rem;
    }
    [data-testid="stDownloadButton"] button:hover { background: #d9472c; color: #fff; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: rgba(255, 255, 255, 0.58); }
    .eyebrow {
        color: var(--coral);
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.85rem;
    }
    .masthead-note { color: var(--muted); font-size: 1rem; margin-bottom: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_configured_password() -> str | None:
    password = os.getenv("APP_PASSWORD")
    if password:
        return password
    try:
        return st.secrets["APP_PASSWORD"]
    except Exception:
        return None


def require_login() -> None:
    if st.session_state.get("authenticated", False):
        if st.sidebar.button("Log out"):
            st.session_state.authenticated = False
            st.rerun()
        st.sidebar.caption("Signed in")
        return

    st.markdown('<div class="eyebrow">THE TRIBUNE TRUST / PRIVATE FORECAST DESK</div>', unsafe_allow_html=True)
    st.title("Sign in to the forecast desk")
    st.markdown(
        '<div class="masthead-note">Enter the authorised access password to view forecasting data and model outputs.</div>',
        unsafe_allow_html=True,
    )
    configured_password = get_configured_password()
    with st.form("login_form"):
        password = st.text_input("Access password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
    if submitted:
        if configured_password and hmac.compare_digest(password, configured_password):
            st.session_state.authenticated = True
            st.rerun()
        elif configured_password is None:
            st.error("Login is unavailable until APP_PASSWORD is configured.")
        else:
            st.error("Incorrect password.")
    st.stop()


require_login()


# ----------------------------------------------------------------------
# Data & model helpers (cached)
# ----------------------------------------------------------------------
@st.cache_data
def load_data(path: str, uploaded_bytes: bytes | None) -> pd.DataFrame:
    if uploaded_bytes is not None:
        try:
            df = pd.read_csv(pd.io.common.BytesIO(uploaded_bytes), parse_dates=["Month"])
        except ValueError as exc:
            raise ValueError("The uploaded CSV must contain a valid 'Month' column.") from exc
    else:
        df = pd.read_csv(path, parse_dates=["Month"])
    required_columns = ["Month", *METRICS]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing_columns)}")
    if df["Month"].isna().any():
        raise ValueError("The 'Month' column contains invalid dates.")
    if df["Month"].duplicated().any():
        raise ValueError("The CSV contains duplicate months. Each month must appear once.")
    df = df.sort_values("Month").set_index("Month").asfreq("MS")
    for column in METRICS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[list(METRICS)].isna().all().any():
        invalid_columns = df[list(METRICS)].columns[df[list(METRICS)].isna().all()]
        raise ValueError(f"CSV has no usable numeric values for: {', '.join(invalid_columns)}")
    return df


@st.cache_resource(show_spinner=False)
def get_sarima_fit(series: pd.Series, metric: str):
    """Load a pre-trained SARIMA model if available, else fit on the fly."""
    saved_path = MODELS_DIR / f"sarima_{metric}.joblib"
    if saved_path.exists():
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
    if len(series) < 24:
        raise ValueError("SARIMA requires at least 24 complete monthly observations.")
    fit = get_sarima_fit(series, metric)
    fc = fit.get_forecast(steps=horizon)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.2)
    future_index = pd.date_range(series.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    mean.index = future_index
    ci.index = future_index
    return mean, ci.iloc[:, 0], ci.iloc[:, 1]


def forecast_holt_winters(series: pd.Series, horizon: int):
    if len(series) < 24:
        raise ValueError("Holt-Winters requires at least 24 complete monthly observations.")
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
    if len(series) < 13:
        raise ValueError("Random Forest requires at least 13 complete monthly observations.")
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
metrics_path = APP_DIR / "model_comparison_metrics.csv"
if metrics_path.exists():
    st.sidebar.caption("Test-set accuracy (Total Revenue, from notebook)")
    st.sidebar.dataframe(pd.read_csv(metrics_path), hide_index=True, width="stretch")

# ----------------------------------------------------------------------
# Main panel
# ----------------------------------------------------------------------
st.markdown('<div class="eyebrow">THE TRIBUNE TRUST / FORECAST DESK</div>', unsafe_allow_html=True)
st.title("The Tribune Trust — Forecasting Dashboard")
st.markdown(
    '<div class="masthead-note">A clear view of what the next 3–24 months may hold across circulation, audience, and revenue.</div>',
    unsafe_allow_html=True,
)

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
fig.add_trace(go.Scatter(x=series.index, y=series.values, name="History", line=dict(color="#087f8c", width=3)))
fig.add_trace(go.Scatter(x=mean.index, y=mean.values, name=f"{model_choice} forecast", line=dict(color="#f05a3c", width=3)))
fig.add_trace(go.Scatter(
    x=list(mean.index) + list(mean.index[::-1]),
    y=list(upper.values) + list(lower.values[::-1]),
    fill="toself",
    fillcolor="rgba(240,90,60,0.16)",
    line=dict(color="rgba(255,255,255,0)"),
    name="Interval",
    showlegend=True,
))
fig.update_layout(
    title=f"{METRICS[metric]} — history & {horizon}-month forecast",
    xaxis_title="Month",
    yaxis_title=METRICS[metric],
    hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#fffdf8",
    font=dict(family="Space Grotesk, sans-serif", color="#15343b"),
    xaxis=dict(gridcolor="#e7ddd0", zeroline=False),
    yaxis=dict(gridcolor="#e7ddd0", zeroline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=60),
)
st.plotly_chart(fig, width="stretch")

st.subheader("Forecast table")
forecast_table = pd.DataFrame({
    "Month": mean.index.strftime("%Y-%m"),
    "Forecast": mean.values.round(1),
    "Lower": lower.values.round(1),
    "Upper": upper.values.round(1),
})
st.dataframe(forecast_table, hide_index=True, width="stretch")

csv_bytes = forecast_table.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download forecast as CSV",
    data=csv_bytes,
    file_name=f"{metric}_forecast_{model_choice.split()[0].lower()}.csv",
    mime="text/csv",
)

with st.expander("Show raw historical data"):
    st.dataframe(df.reset_index(), hide_index=True, width="stretch")
