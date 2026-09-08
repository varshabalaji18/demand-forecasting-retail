"""Interactive demand forecasting application.

Run with: streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_pipeline import load_sales_csv, prepare_daily_series
from src.model import train_and_forecast


PROJECT_ROOT = Path(__file__).parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "retail_sales.csv"

st.set_page_config(page_title="Demand Forecasting in Retail", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 48%, #ecfdf5 100%); }
    [data-testid="stSidebar"] { background: #111827; }
    [data-testid="stSidebar"] * { color: #f8fafc !important; }
    [data-testid="stMetric"] { background: rgba(255,255,255,.88); border: 1px solid #e2e8f0; padding: 16px; border-radius: 14px; box-shadow: 0 4px 16px rgba(15,23,42,.06); }
    h1 { color: #111827; letter-spacing: -0.03em; }
    h2, h3 { color: #1e293b; }
    .hero { background: linear-gradient(110deg, #312e81, #047857); color: white; padding: 24px 28px; border-radius: 18px; margin-bottom: 18px; box-shadow: 0 12px 28px rgba(30,41,59,.18); }
    .hero h1 { color: white; margin: 0; }
    .hero p { margin: 8px 0 0; color: #dbeafe; font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return load_sales_csv(path)


def make_chart(history: pd.DataFrame, forecast: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["ds"], y=history["y"], name="Historical demand", line=dict(color="#4f46e5", width=2)))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], name="Upper interval", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], name="90% planning interval", fill="tonexty", fillcolor="rgba(16,185,129,0.18)", line=dict(width=0)))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast", line=dict(color="#059669", width=3, dash="dash")))
    fig.update_layout(title=title, height=520, hovermode="x unified", margin=dict(l=20, r=20, t=60, b=20), legend=dict(orientation="h", y=1.08))
    fig.update_yaxes(title="Units sold", rangemode="tozero")
    fig.update_xaxes(title=None)
    return fig


st.markdown(
    '<div class="hero"><h1>Demand Forecasting in Retail</h1><p>Turn daily sales history into an inventory planning range — in one click.</p></div>',
    unsafe_allow_html=True,
)
st.caption("Portfolio demo • Holt-Winters forecast • seasonal-naive benchmark • approximate 90% planning interval")

with st.sidebar:
    st.header("🎛️ Forecast controls")
    if "data_source" not in st.session_state:
        st.session_state.data_source = "Synthetic demo"
    if st.button("▶ Run synthetic demo", type="primary", use_container_width=True):
        st.session_state.data_source = "Synthetic demo"
        st.rerun()
    data_source = st.radio("Data source", ["Synthetic demo", "Upload CSV"], key="data_source")
    uploaded = st.file_uploader("Upload retail CSV", type=["csv"], help="Required columns: date and sales. Optional columns: store and item.") if data_source == "Upload CSV" else None
    horizon = st.select_slider("Forecast horizon", options=[30, 60, 90], value=30, format_func=lambda x: f"{x} days")
    history_days = st.slider("Historical context", min_value=60, max_value=730, value=365, step=30)
    zero_fill = st.checkbox('Treat missing calendar days as zero sales', value=False)

try:
    raw = pd.read_csv(uploaded) if uploaded is not None else load_data(str(DEFAULT_DATA))
    if "store" in raw.columns:
        stores = sorted(raw["store"].dropna().unique().tolist())
        selected_store = st.sidebar.selectbox("Store", ["All stores"] + stores)
        if selected_store != "All stores":
            raw = raw[raw["store"] == selected_store]
    if "item" in raw.columns:
        items = sorted(raw["item"].dropna().unique().tolist())
        selected_item = st.sidebar.selectbox("Item", ["All items"] + items)
        if selected_item != "All items":
            raw = raw[raw["item"] == selected_item]

    if data_source == "Synthetic demo":
        st.info("🎬 Demo mode: realistic synthetic retail demand is loaded from the repository.")
    else:
        st.info("📄 Upload mode: provide a CSV with date and sales columns.")
    with st.spinner("Loading data and fitting the forecasting models…"):
        series = prepare_daily_series(raw, fill_missing_dates=zero_fill)
        result = train_and_forecast(series, horizon=horizon)
        visible_history = series.tail(history_days)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast horizon", f"{horizon} days")
    c2.metric("Expected demand", f"{result.forecast['yhat'].sum():,.0f} units")
    c3.metric("Backtest WAPE", f"{result.metrics['wape']:.1%}")
    c4.metric("Average daily forecast", f"{result.forecast['yhat'].mean():,.0f}")

    with st.expander("💡 What am I looking at?", expanded=True):
        st.markdown(
            "**1. Historical demand** is the purple line. **2. Forecast** is the green dashed line. "
            "**3. Planning interval** is the shaded band: a plausible range, not a guarantee. "
            "Use the midpoint for the base plan and the upper edge for a more conservative replenishment decision."
        )
        best = result.comparison.sort_values("wape").iloc[0]
        st.success(f"For this {horizon}-day horizon, **{best['model']}** is the strongest backtest model with {best['wape']:.1%} WAPE. Lower WAPE is better.")

    st.plotly_chart(make_chart(visible_history, result.forecast, "Historical demand and forecast"), width="stretch")
    st.info('Approximate 90% residual-bootstrap intervals do not propagate trend or parameter uncertainty. Check observed holdout coverage below; 90% coverage is not guaranteed.')
    st.subheader('Benchmark comparison')
    st.caption(f'Three expanding-window folds, each forecasting {horizon} days. Pooled metrics; WAPE and coverage are fractions. No interval is estimated for seasonal naive.')
    st.dataframe(result.comparison, hide_index=True, width="stretch")
    with st.expander('Inspect evaluation folds'):
        st.dataframe(result.folds, hide_index=True, width="stretch")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Forecast output")
        st.dataframe(result.forecast.style.format({"yhat": "{:,.1f}", "yhat_lower": "{:,.1f}", "yhat_upper": "{:,.1f}"}), width="stretch", hide_index=True)
    with right:
        st.subheader("Backtest performance")
        st.dataframe(pd.DataFrame([result.metrics]).T.rename(columns={0: "value"}).style.format("{:.3f}"), width="stretch")
        export = result.forecast.to_csv(index=False).encode("utf-8")
        st.download_button("Download forecast CSV", export, "demand_forecast.csv", "text/csv", width="stretch")
        metrics_export = result.comparison.to_csv(index=False).encode("utf-8")
        st.download_button("Download metrics CSV", metrics_export, "forecast_metrics.csv", "text/csv", width="stretch")
except Exception as exc:
    st.error(f"Unable to produce a forecast: {exc}")
    st.stop()

