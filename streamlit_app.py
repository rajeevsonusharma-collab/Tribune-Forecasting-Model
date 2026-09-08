"""
The Tribune Trust — Forecasting Console
-----------------------------------------
Interactive Streamlit dashboard forecasting five newsroom/business metrics
(print circulation, digital visitors, ad revenue, subscription revenue,
total revenue) with SARIMA, Holt-Winters, or a lag-feature Random Forest —
plus live backtesting against a holdout period, naive-baseline comparison,
and a trend/seasonality decomposition view.

Run with:
    streamlit run streamlit_app.py

Expects these files in the same folder (all produced by
tribune_forecasting_model.ipynb):
    tribune_trust_synthetic_data.csv
    models/sarima_<metric>.joblib   (optional — app fits live if missing)
    model_comparison_metrics.csv    (optional — shown in the sidebar)
Expects, in the same folder:
    tribune_trust_synthetic_data.csv   (required — or upload one in-app)
    models/sarima_<metric>.joblib      (optional — app fits live if missing)
"""

import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import streamlit as st
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "tribune_trust_synthetic_data.csv"
MODELS_DIR = APP_DIR / "models"
REQUIRED_COLUMNS = [
    "Month",
    "Print_Circulation_Copies_Per_Day",
    "Digital_Unique_Visitors",
    "Ad_Revenue_INR_Lakh",
    "Subscription_Revenue_INR_Lakh",
    "Total_Revenue_INR_Lakh",
]
METRICS = {
    "Total_Revenue_INR_Lakh": ("Total Revenue", "INR Lakh", "🧾"),
    "Ad_Revenue_INR_Lakh": ("Ad Revenue", "INR Lakh", "📢"),
    "Subscription_Revenue_INR_Lakh": ("Subscription Revenue", "INR Lakh", "🗞️"),
    "Print_Circulation_Copies_Per_Day": ("Print Circulation", "copies/day", "📰"),
    "Digital_Unique_Visitors": ("Digital Unique Visitors", "visitors/mo", "💻"),
}
MODEL_NAMES = ["SARIMA", "Holt-Winters", "Random Forest"]
COLORS = {
    "ink": "#14162A",
    "paper": "#FBF8F3",
    "card": "#FFFFFF",
    "red": "#E63946",
    "gold": "#F2A900",
    "teal": "#1FB6A6",
    "green": "#2BB673",
    "soft": "#6B7099",
    "hairline": "#E7E1D3",
}

st.set_page_config(page_title="The Tribune Trust — Forecast Console", page_icon="📰", layout="wide")


def kpi_card(label: str, value: str, sub: str, accent: str):
    st.markdown(
        f"""
        <div class="kpi-card" style="--accent:{accent}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, title: str, y_title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(family="Inter", size=16, color=COLORS["ink"])),
        font=dict(family="Inter", size=12, color=COLORS["ink"]),
        plot_bgcolor=COLORS["card"],
        paper_bgcolor=COLORS["card"],
        xaxis=dict(title="Month", gridcolor=COLORS["hairline"], showline=True, linecolor=COLORS["hairline"]),
        yaxis=dict(title=y_title, gridcolor=COLORS["hairline"], showline=True, linecolor=COLORS["hairline"]),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
        margin=dict(t=55, l=10, r=10, b=10),
    )
    return fig

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
    [data-testid="stTextInput"] input {
        background: #fff7ed;
        border: 1px solid #f3d9a8;
        border-radius: 8px;
    }
    [data-testid="stFormSubmitButton"] button {
        background: #e63946;
        border: 0;
        color: #fff;
        font-weight: 700;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        background: #bd2633;
        color: #fff;
    }
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
    .login-page {
        max-width: 1120px;
        margin: 3vh auto 0;
    }
    .login-visual {
        min-height: 455px;
        padding: 2.5rem;
        border-radius: 18px;
        color: #fff;
        overflow: hidden;
        position: relative;
        background: linear-gradient(145deg, #14162a 0%, #242849 70%, #263e59 100%);
        box-shadow: 0 22px 55px rgba(20, 22, 42, 0.2);
    }
    .login-visual::after {
        content: "";
        position: absolute;
        width: 230px;
        height: 230px;
        right: -74px;
        top: -76px;
        border: 1px solid rgba(242, 169, 0, 0.45);
        border-radius: 50%;
        box-shadow: 0 0 0 22px rgba(242, 169, 0, 0.05), 0 0 0 44px rgba(242, 169, 0, 0.04);
    }
    .login-kicker {
        color: #f2a900;
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }
    .news-masthead {
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 1rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.24);
    }
    .news-brand {
        color: #fff;
        font-family: 'Playfair Display', serif;
        font-size: clamp(1.55rem, 3vw, 2.2rem);
        font-weight: 800;
        letter-spacing: 0.02em;
        line-height: 1;
    }
    .news-edition {
        color: #c7cae4;
        font-family: 'DM Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.08em;
        line-height: 1.5;
        text-align: right;
        text-transform: uppercase;
    }
    .news-rule {
        display: flex;
        justify-content: space-between;
        color: #9ea3c4;
        font-family: 'DM Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.08em;
        margin-top: 0.7rem;
        text-transform: uppercase;
    }
    .login-visual h2 {
        color: #fff;
        font-family: 'Playfair Display', serif;
        font-size: clamp(2rem, 4vw, 3.6rem);
        line-height: 1.02;
        margin: 1.1rem 0 0.8rem;
        max-width: 440px;
    }
    .login-visual p { color: #c7cae4; max-width: 390px; line-height: 1.6; }
    .signal-chart {
        height: 130px;
        display: flex;
        align-items: end;
        gap: 9px;
        margin-top: 2.4rem;
        padding: 1rem 0 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    }
    .signal-chart span {
        flex: 1;
        min-width: 12px;
        border-radius: 5px 5px 0 0;
        background: linear-gradient(180deg, #f2a900, #e63946);
        opacity: 0.9;
    }
    .signal-chart span:nth-child(2n) { background: linear-gradient(180deg, #1fb6a6, #167b8a); }
    .signal-chart span:nth-child(1) { height: 32%; }
    .signal-chart span:nth-child(2) { height: 44%; }
    .signal-chart span:nth-child(3) { height: 39%; }
    .signal-chart span:nth-child(4) { height: 60%; }
    .signal-chart span:nth-child(5) { height: 55%; }
    .signal-chart span:nth-child(6) { height: 76%; }
    .signal-chart span:nth-child(7) { height: 68%; }
    .signal-chart span:nth-child(8) { height: 94%; }
    .signal-caption {
        display: flex;
        justify-content: space-between;
        margin-top: 0.55rem;
        color: #9ea3c4;
        font-family: 'DM Mono', monospace;
        font-size: 0.66rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .signal-summary {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.6rem;
        margin-top: 1.25rem;
    }
    .signal-summary div {
        padding: 0.7rem 0.75rem;
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 9px;
        background: rgba(255, 255, 255, 0.06);
    }
    .signal-summary strong {
        display: block;
        color: #fff;
        font-size: 1.1rem;
    }
    .signal-summary span {
        color: #9ea3c4;
        font-family: 'DM Mono', monospace;
        font-size: 0.62rem;
        text-transform: uppercase;
    }
    .field-label {
        color: #6b7099;
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .login-form-panel {
        min-height: 455px;
        padding: 2.5rem 2.2rem;
        border: 1px solid #e7e1d3;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.82);
        box-shadow: 0 22px 55px rgba(20, 22, 42, 0.08);
    }
    .login-form-panel h1 { font-size: clamp(2rem, 4vw, 3.25rem); }
    .welcome-message {
        margin: -0.15rem 0 0.8rem;
        color: #15343b;
        font-size: 0.98rem;
        line-height: 1.5;
    }
    .welcome-message strong { color: #e63946; }
    @media (max-width: 760px) {
        .login-page { margin-top: 1rem; }
        .login-visual, .login-form-panel { min-height: auto; padding: 1.7rem; }
        .signal-chart { margin-top: 1.6rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Data loading & validation
# ----------------------------------------------------------------------
def validate_dataframe(raw: pd.DataFrame) -> tuple[bool, list[str]]:
    problems = []
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_cols:
        problems.append(f"Missing required column(s): {', '.join(missing_cols)}")
        return False, problems
    try:
        pd.to_datetime(raw["Month"])
    except Exception:
        problems.append("The 'Month' column could not be parsed as dates (expected e.g. 2024-01 or 2024-01-01).")
        return False, problems
    numeric_cols = [c for c in REQUIRED_COLUMNS if c != "Month"]
    for c in numeric_cols:
        if not pd.api.types.is_numeric_dtype(pd.to_numeric(raw[c], errors="coerce")):
            problems.append(f"Column '{c}' contains non-numeric values.")
    if len(raw) < 24:
        problems.append(
            f"Only {len(raw)} rows found — at least 24 months (2 full seasonal cycles) is "
            "recommended for reliable seasonal forecasting."
        )
    negatives = {c: int((pd.to_numeric(raw[c], errors="coerce") < 0).sum()) for c in numeric_cols}
    neg_flag = {c: n for c, n in negatives.items() if n > 0}
    if neg_flag:
        problems.append(f"Negative values found in: {', '.join(neg_flag.keys())} — please double-check the source data.")
    # only the first two problem types are fatal (return False); row-count / negatives are warnings
    fatal = any(p.startswith(("Missing required", "The 'Month'", "Column '")) for p in problems)
    return (not fatal), problems


@st.cache_data(show_spinner=False)
def load_data(path: str, uploaded_bytes: bytes | None):
    if uploaded_bytes is not None:
        raw = pd.read_csv(pd.io.common.BytesIO(uploaded_bytes))
    else:
        raw = pd.read_csv(path)
    ok, problems = validate_dataframe(raw)
    if not ok:
        return None, problems
    raw["Month"] = pd.to_datetime(raw["Month"])
    df = raw.set_index("Month").sort_index().asfreq("MS")
    return df, problems


# ----------------------------------------------------------------------
# Forecasting models
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_sarima_fit(series: pd.Series, metric: str, cache_key: str):
    saved_path = os.path.join(MODELS_DIR, f"sarima_{metric}.joblib")
    if os.path.exists(saved_path):
        try:
            return joblib.load(saved_path)
        except Exception:
            pass
    model = SARIMAX(
        series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    return model.fit(disp=False)


def forecast_sarima(series: pd.Series, metric: str, horizon: int, confidence: int, use_cache=True):
    alpha = 1 - confidence / 100
    if use_cache:
        fit = get_sarima_fit(series, metric, cache_key=f"{len(series)}-{series.index.max()}")
    else:
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                         enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
    fc = fit.get_forecast(steps=horizon)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=alpha)
    idx = pd.date_range(series.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    mean.index = idx
    ci.index = idx
    return mean, ci.iloc[:, 0], ci.iloc[:, 1]


def forecast_holt_winters(series: pd.Series, horizon: int, confidence: int):
    alpha = 1 - confidence / 100
    z = norm.ppf(1 - alpha / 2)
    fit = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=12).fit()
    mean = fit.forecast(horizon)
    idx = pd.date_range(series.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    mean.index = idx
    resid_std = np.std(fit.resid)
    return mean, mean - z * resid_std, mean + z * resid_std


def make_features(series: pd.Series, lags=(1, 2, 3, 12), roll_window=3) -> pd.DataFrame:
    feat = pd.DataFrame({"y": series})
    for lag in lags:
        feat[f"lag_{lag}"] = series.shift(lag)
    feat[f"roll_mean_{roll_window}"] = series.shift(1).rolling(roll_window).mean()
    feat["month"] = feat.index.month
    feat["quarter"] = feat.index.quarter
    return feat.dropna()


def forecast_random_forest(series: pd.Series, horizon: int, confidence: int):
    """Recursive multi-step forecast; per-tree spread gives the interval."""
    lags, roll_window = (1, 2, 3, 12), 3
    feat_df = make_features(series, lags, roll_window)
    feature_cols = [c for c in feat_df.columns if c != "y"]

    rf = RandomForestRegressor(n_estimators=400, max_depth=6, random_state=42)
    rf.fit(feat_df[feature_cols], feat_df["y"])

    history = series.copy()
    idx = pd.date_range(series.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    means, lowers, uppers = [], [], []
    alpha = 1 - confidence / 100
    lo_pct, hi_pct = (alpha / 2) * 100, 100 - (alpha / 2) * 100

    for month in idx:
        row = {f"lag_{lag}": history.iloc[-lag] for lag in lags}
        row[f"roll_mean_{roll_window}"] = history.iloc[-roll_window:].mean()
        row["month"], row["quarter"] = month.month, month.quarter
        x = pd.DataFrame([row])[feature_cols]
        tree_preds = np.array([est.predict(x)[0] for est in rf.estimators_])
        y_hat = float(tree_preds.mean())
        means.append(y_hat)
        lowers.append(float(np.percentile(tree_preds, lo_pct)))
        uppers.append(float(np.percentile(tree_preds, hi_pct)))
        history.loc[month] = y_hat

    return pd.Series(means, index=idx), pd.Series(lowers, index=idx), pd.Series(uppers, index=idx)


def run_model(name: str, series: pd.Series, metric: str, horizon: int, confidence: int, use_cache=True):
    if name == "SARIMA":
        return forecast_sarima(series, metric, horizon, confidence, use_cache=use_cache)
    if name == "Holt-Winters":
        return forecast_holt_winters(series, horizon, confidence)
    return forecast_random_forest(series, horizon, confidence)


# ----------------------------------------------------------------------
# Naive baselines
# ----------------------------------------------------------------------
def naive_forecast(train: pd.Series, horizon: int) -> pd.Series:
    idx = pd.date_range(train.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    return pd.Series([train.iloc[-1]] * horizon, index=idx)


def seasonal_naive_forecast(train: pd.Series, horizon: int) -> pd.Series:
    idx = pd.date_range(train.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    vals = [train.iloc[-12 + (i % 12)] if len(train) >= 12 else train.iloc[-1] for i in range(horizon)]
    return pd.Series(vals, index=idx)


def error_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    a, p = actual.values, predicted.values
    mape = float(np.mean(np.abs((a - p) / a)) * 100)
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    mae = float(np.mean(np.abs(a - p)))
    return {"MAPE %": round(mape, 2), "RMSE": round(rmse, 1), "MAE": round(mae, 1)}


@st.cache_data(show_spinner=False)
def run_backtest(series: pd.Series, holdout: int):
    """Fit every model on data up to (len-holdout) and score against the holdout."""
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]
    rows, curves, errs = {}, {}, {}

    for name, fn in [("Naive", naive_forecast), ("Seasonal Naive", seasonal_naive_forecast)]:
        pred = fn(train, holdout)
        rows[name] = error_metrics(test, pred)
        curves[name] = pred

    for name in MODEL_NAMES:
        try:
            mean, _, _ = run_model(name, train, "backtest", holdout, confidence=80, use_cache=False)
            rows[name] = error_metrics(test, mean)
            curves[name] = mean
        except Exception as e:
            errs[name] = str(e)

    return train, test, rows, curves, errs


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sb-chip">
            <h3>The Tribune Trust</h3>
            <span>Forecasting Console</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload your own CSV (same columns)", type=["csv"])
    uploaded_bytes = uploaded.read() if uploaded is not None else None

    df, data_problems = load_data(DATA_PATH, uploaded_bytes)

    if df is None:
        st.error("This file can't be used yet:\n\n" + "\n".join(f"• {p}" for p in data_problems))
        st.stop()

    metric = st.selectbox(
        "Metric", options=list(METRICS.keys()),
        format_func=lambda k: f"{METRICS[k][2]} {METRICS[k][0]}",
    )
    model_choice = st.radio("Model", MODEL_NAMES, index=0)
    horizon = st.slider("Forecast horizon (months)", 3, 24, 12)
    confidence = st.select_slider("Confidence level", options=[70, 80, 90, 95], value=80)
    compare_models = st.checkbox("Overlay all 3 models on the chart", value=False)

    st.markdown("---")
    st.caption(f"📅 Data: {df.index.min():%b %Y} – {df.index.max():%b %Y}  ·  {len(df)} months")
    if len(data_problems) > 0:
        with st.expander("⚠️ Data quality notes"):
            for p in data_problems:
                st.write(f"• {p}")

series = df[metric].dropna()
label, unit, icon = METRICS[metric]

# ----------------------------------------------------------------------
# Masthead
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <div class="masthead">
        <h1>📰 The Tribune Trust — Forecasting Console</h1>
        <p>Business &amp; Analytics Desk · {label} forecast, {horizon} months ahead ·
        synthetic data, illustrative only</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_forecast, tab_backtest, tab_trend, tab_data = st.tabs(
    ["📈 Forecast", "🎯 Accuracy & Backtest", "🔍 Trend & Seasonality", "🗂️ Data"]
)

# ----------------------------------------------------------------------
# TAB 1 — Forecast
# ----------------------------------------------------------------------
with tab_forecast:
    try:
        with st.spinner(f"Fitting {model_choice} on {label}..."):
            mean, lower, upper = run_model(model_choice, series, metric, horizon, confidence)
    except Exception as e:
        st.error(
            f"**{model_choice} couldn't be fit on this series.** "
            f"This usually means there isn't enough history yet, or the data has a gap.\n\n"
            f"Details: {e}"
        )
        st.stop()

    # quick, cheap backtest (last 6 months) just to power the trust indicator KPI
    quick_holdout = min(6, max(3, len(series) // 6))
    try:
        _, quick_test, quick_rows, _, _ = run_backtest(series, quick_holdout)
        quick_mape = quick_rows.get(model_choice, {}).get("MAPE %")
    except Exception:
        quick_mape = None

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card(f"{icon} Latest actual", f"{series.iloc[-1]:,.1f}", f"{series.index[-1]:%b %Y}  ·  {unit}", COLORS["teal"])
    with k2:
        kpi_card(
            f"Forecast — {mean.index[-1]:%b %Y}", f"{mean.iloc[-1]:,.1f}",
            f"{confidence}% interval: {lower.iloc[-1]:,.0f} – {upper.iloc[-1]:,.0f}", COLORS["red"],
        )
    with k3:
        pct_change = (mean.iloc[-1] / series.iloc[-1] - 1) * 100
        kpi_card(
            "Projected change", f"{pct_change:+.1f}%",
            f"vs. latest actual, over {horizon} months", COLORS["gold"] if pct_change >= 0 else COLORS["red"],
        )
    with k4:
        if quick_mape is not None:
            trust_color = COLORS["green"] if quick_mape < 5 else (COLORS["gold"] if quick_mape < 10 else COLORS["red"])
            kpi_card("Recent model accuracy", f"{quick_mape:.1f}% MAPE", f"{model_choice}, last {quick_holdout} months held out", trust_color)
        else:
            kpi_card("Recent model accuracy", "n/a", "not enough history to backtest", COLORS["soft"])

    st.write("")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name="History", line=dict(color=COLORS["ink"], width=2)))
    fig.add_trace(go.Scatter(x=mean.index, y=mean.values, name=f"{model_choice} forecast", line=dict(color=COLORS["red"], width=2.5)))
    fig.add_trace(go.Scatter(
        x=list(mean.index) + list(mean.index[::-1]),
        y=list(upper.values) + list(lower.values[::-1]),
        fill="toself", fillcolor="rgba(230,57,70,0.12)",
        line=dict(color="rgba(255,255,255,0)"), name=f"{confidence}% interval", showlegend=True,
    ))

    if compare_models:
        overlay_colors = {"SARIMA": COLORS["red"], "Holt-Winters": COLORS["gold"], "Random Forest": COLORS["teal"]}
        for name in MODEL_NAMES:
            if name == model_choice:
                continue
            try:
                m2, _, _ = run_model(name, series, metric, horizon, confidence)
                fig.add_trace(go.Scatter(x=m2.index, y=m2.values, name=f"{name} (compare)",
                                          line=dict(color=overlay_colors[name], width=1.5, dash="dot")))
            except Exception:
                pass

    fig = style_fig(fig, f"{label} — history & {horizon}-month forecast", f"{label} ({unit})")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Forecast table")
    forecast_table = pd.DataFrame({
        "Month": mean.index.strftime("%Y-%m"),
        "Forecast": mean.values.round(1),
        f"Lower ({confidence}%)": lower.values.round(1),
        f"Upper ({confidence}%)": upper.values.round(1),
    })
    st.dataframe(forecast_table, hide_index=True, width="stretch")

    csv_bytes = forecast_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download forecast as CSV", data=csv_bytes,
        file_name=f"{metric}_forecast_{model_choice.replace(' ', '_').lower()}.csv", mime="text/csv",
    )

# ----------------------------------------------------------------------
# TAB 2 — Accuracy & Backtest
# ----------------------------------------------------------------------
with tab_backtest:
    st.markdown(
        """
        <div class="note-box">
        We hold out the most recent months, fit every model on the data <em>before</em> that window,
        and check how close each one gets to what actually happened — the same discipline a
        credit-rating or equity analyst would apply before trusting a forecast.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    max_holdout = max(3, min(12, len(series) // 4))
    holdout = st.slider("Holdout window (months)", 3, max_holdout, min(6, max_holdout), key="holdout")

    if len(series) - holdout < 24:
        st.warning("Not enough training history left after this holdout for a reliable seasonal fit — try a shorter holdout.")
    else:
        with st.spinner("Backtesting SARIMA, Holt-Winters, Random Forest, and naive baselines..."):
            train, test, rows, curves, errs = run_backtest(series, holdout)

        metrics_df = pd.DataFrame(rows).T.reset_index().rename(columns={"index": "Model"})
        best_model = metrics_df.loc[metrics_df["MAPE %"].idxmin(), "Model"]
        st.subheader(f"Accuracy on the last {holdout} months — {label}")
        st.dataframe(
            metrics_df.style.apply(
                lambda r: ["background-color:#E8F7EF" if r["Model"] == best_model else "" for _ in r], axis=1
            ),
            hide_index=True, width="stretch",
        )
        st.caption(f"✅ Lowest error on this window: **{best_model}**. MAPE = mean absolute % error, lower is better.")

        if errs:
            for name, msg in errs.items():
                st.caption(f"⚠️ {name} failed during backtest: {msg}")

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=train.index[-12:], y=train.values[-12:], name="History (train)", line=dict(color=COLORS["ink"], width=2)))
        fig2.add_trace(go.Scatter(x=test.index, y=test.values, name="Actual (held out)", line=dict(color=COLORS["ink"], width=3, dash="solid")))
        palette = {"Naive": COLORS["soft"], "Seasonal Naive": "#B7BAD6", "SARIMA": COLORS["red"], "Holt-Winters": COLORS["gold"], "Random Forest": COLORS["teal"]}
        for name, curve in curves.items():
            fig2.add_trace(go.Scatter(x=curve.index, y=curve.values, name=name, line=dict(color=palette.get(name, "#999"), width=2, dash="dot")))
        fig2 = style_fig(fig2, f"Holdout check — predicted vs. actual ({label})", f"{label} ({unit})")
        st.plotly_chart(fig2, width="stretch")

# ----------------------------------------------------------------------
# TAB 3 — Trend & Seasonality
# ----------------------------------------------------------------------
with tab_trend:
    st.markdown(
        """
        <div class="note-box">
        Before trusting a seasonal model, it helps to see the seasonality — this splits the series
        into its underlying trend, repeating yearly pattern, and what's left over (noise).
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    if len(series) < 24:
        st.warning("At least 24 months of data are needed for a stable trend/seasonality split.")
    else:
        stl = STL(series, period=12, robust=True).fit()
        decomp_fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                       subplot_titles=("Trend", "Seasonal pattern", "Residual (noise)"))
        decomp_fig.add_trace(go.Scatter(x=series.index, y=stl.trend, line=dict(color=COLORS["red"], width=2)), row=1, col=1)
        decomp_fig.add_trace(go.Scatter(x=series.index, y=stl.seasonal, line=dict(color=COLORS["teal"], width=2)), row=2, col=1)
        decomp_fig.add_trace(go.Scatter(x=series.index, y=stl.resid, mode="markers", marker=dict(color=COLORS["gold"], size=4)), row=3, col=1)
        decomp_fig.update_layout(
            height=620, showlegend=False, plot_bgcolor=COLORS["card"], paper_bgcolor=COLORS["card"],
            font=dict(family="Inter", size=12, color=COLORS["ink"]), margin=dict(t=40, l=10, r=10, b=10),
            title=dict(text=f"{label} — trend & seasonality decomposition", font=dict(family="Inter", size=16, color=COLORS["ink"])),
        )
        decomp_fig.update_xaxes(gridcolor=COLORS["hairline"])
        decomp_fig.update_yaxes(gridcolor=COLORS["hairline"])
        st.plotly_chart(decomp_fig, width="stretch")

        seasonal_swing = stl.seasonal.max() - stl.seasonal.min()
        st.caption(
            f"Seasonal swing: about **{seasonal_swing:,.0f} {unit}** between the strongest and weakest "
            "month of a typical year, on top of the trend."
        )

# ----------------------------------------------------------------------
# TAB 4 — Data
# ----------------------------------------------------------------------
with tab_data:
    st.subheader("Raw historical data")
    st.dataframe(df.reset_index(), hide_index=True, width="stretch")

    st.subheader("Data quality summary")
    dq = pd.DataFrame({
        "Metric": list(METRICS.keys()),
        "Min": [df[c].min() for c in METRICS],
        "Max": [df[c].max() for c in METRICS],
        "Mean": [round(df[c].mean(), 1) for c in METRICS],
        "Missing months": [int(df[c].isna().sum()) for c in METRICS],
    })
    st.dataframe(dq, hide_index=True, width="stretch")

    full_range = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    gap_count = len(full_range) - len(df.index)
    if gap_count > 0:
        st.warning(f"{gap_count} month(s) appear to be missing from a continuous monthly sequence.")
    else:
        st.success("No missing months detected — the series is a continuous monthly sequence.")
