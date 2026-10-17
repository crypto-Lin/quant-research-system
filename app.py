from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from core.backtest.daily_momentum import run_backtest
from core.market_data.daily import generate_demo_data, load_ohlcv_csv


@st.cache_data
def _demo_data() -> pd.DataFrame:
    return generate_demo_data()


st.set_page_config(page_title="Quant Research Workbench", layout="wide")
st.title("Quant Research Workbench")
st.caption("US equities + crypto | Daily bars | Local-only research and simulation")
st.warning(
    "This workbench does not fetch live prices or connect to a broker. "
    "Backtests and paper-mode fills are simulations, not investment advice."
)

source_choice = st.sidebar.radio(
    "Data source",
    ("Synthetic demo (not market data)", "Upload historical daily CSV"),
)

if source_choice == "Synthetic demo (not market data)":
    market_data = _demo_data()
    source_label = "Deterministic synthetic demo — not real historical or live market data"
else:
    uploaded = st.sidebar.file_uploader("Daily OHLCV CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to continue. See the CSV schema in the README.")
        st.stop()
    try:
        market_data = load_ohlcv_csv(uploaded)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    source_label = "User-uploaded CSV — provider and data accuracy are not independently verified"

st.write(f"**Data boundary:** {source_label}")
available_symbols = sorted(market_data["symbol"].unique().tolist())
with st.sidebar.form("backtest-settings"):
    selected_symbols = st.multiselect(
        "Assets",
        options=available_symbols,
        default=available_symbols,
    )
    initial_capital = st.number_input(
        "Initial capital (USD)",
        min_value=100.0,
        value=100_000.0,
        step=10_000.0,
    )
    commission_bps = st.number_input(
        "Commission (basis points per fill)",
        min_value=0.0,
        value=1.0,
        step=0.5,
    )
    slippage_bps = st.number_input(
        "Slippage (basis points per fill)",
        min_value=0.0,
        value=5.0,
        step=1.0,
    )
    fast_window = st.number_input(
        "Momentum lookback (bars)",
        min_value=1,
        max_value=100,
        value=20,
        step=1,
    )
    slow_window = st.number_input(
        "Trend SMA (bars)",
        min_value=2,
        max_value=300,
        value=50,
        step=1,
    )
    submitted = st.form_submit_button("Run simulated backtest", type="primary")

current_data = market_data[market_data["symbol"].isin(selected_symbols)].copy()
settings = (
    source_choice,
    hashlib.sha256(pd.util.hash_pandas_object(market_data, index=True).values.tobytes()).hexdigest(),
    tuple(selected_symbols),
    float(initial_capital),
    float(commission_bps),
    float(slippage_bps),
    int(fast_window),
    int(slow_window),
)

if submitted:
    if not selected_symbols:
        st.error("Select at least one asset.")
    elif fast_window >= slow_window:
        st.error("The momentum lookback must be shorter than the trend SMA.")
    else:
        result = run_backtest(
            current_data,
            initial_capital=float(initial_capital),
            commission_bps=float(commission_bps),
            slippage_bps=float(slippage_bps),
            fast_window=int(fast_window),
            slow_window=int(slow_window),
        )
        st.session_state["backtest_result"] = result
        st.session_state["backtest_settings"] = settings

if st.session_state.get("backtest_settings") == settings:
    result = st.session_state["backtest_result"]
    metrics = result.metrics
    st.subheader("Simulated portfolio")
    st.caption(
        f"{source_label} · {metrics['start_date']} to {metrics['end_date']} · "
        f"signals are filled at the next available asset bar"
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("Total return", f"{metrics['total_return']:.2%}")
    metric_columns[1].metric("Annualized return*", f"{metrics['annualized_return']:.2%}")
    metric_columns[2].metric("Maximum drawdown", f"{metrics['max_drawdown']:.2%}")
    metric_columns[3].metric("Sharpe ratio*", f"{metrics['sharpe_ratio']:.2f}")
    risk_columns = st.columns(4)
    risk_columns[0].metric("Annualized volatility*", f"{metrics['annualized_volatility']:.2%}")
    risk_columns[1].metric("Closed trades", str(metrics["closed_trades"]))
    risk_columns[2].metric("Executed fills", str(metrics["executed_orders"]))
    risk_columns[3].metric("Fees paid (USD)", f"${metrics['fees_paid']:,.2f}")
    st.caption(
        "*Annualization assumes 365 calendar days when crypto is included and 252 trading "
        "days for equity-only data. Mixed-asset figures are simplified diagnostics."
    )

    chart_data = result.daily_equity.set_index("date")[["equity"]]
    st.line_chart(chart_data)
    left, right = st.columns(2)
    with left:
        st.subheader("Orders")
        st.dataframe(result.orders, hide_index=True)
        st.download_button(
            "Download orders CSV",
            result.orders.to_csv(index=False).encode("utf-8"),
            file_name="simulated_orders.csv",
            mime="text/csv",
        )
    with right:
        st.subheader("Trades, including open positions")
        st.dataframe(result.trades, hide_index=True)
    with st.expander("Strategy and execution assumptions"):
        st.write(
            "Long-only baseline: close above the slow simple moving average and positive "
            "lookback momentum. Entry/exit decisions made at a daily close execute at the "
            "next available bar's open with configurable slippage and commission. US equities "
            "use whole shares; crypto uses up to 8 decimal places. No leverage, dividends, "
            "taxes, financing, exchange-specific fee schedules, limit orders, or broker rules "
            "are modeled."
        )
else:
    st.info("Choose assets and assumptions in the sidebar, then run the simulated backtest.")
