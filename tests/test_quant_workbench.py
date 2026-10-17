from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

from core.backtest.daily_momentum import calculate_signals, run_backtest
from core.market_data.daily import generate_demo_data, load_ohlcv_csv, validate_ohlcv


def _market_data(symbol: str, asset_class: str, start_price: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    closes = [
        start_price,
        start_price,
        start_price,
        start_price * 1.2,
        start_price * 1.3,
        start_price * 1.2,
        start_price * 1.1,
    ]
    opens = [closes[0], *closes[:-1]]
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "asset_class": asset_class,
            "open": opens,
            "high": [max(open_price, close) * 1.01 for open_price, close in zip(opens, closes)],
            "low": [min(open_price, close) * 0.99 for open_price, close in zip(opens, closes)],
            "close": closes,
            "volume": 1000,
        }
    )


def test_csv_load_infers_default_symbols_and_sorts_rows() -> None:
    csv = BytesIO(
        b"date,symbol,open,high,low,close,volume\n"
        b"2024-01-02,SPY,101,103,100,102,1000\n"
        b"2024-01-01,SPY,100,102,99,101,900\n"
    )

    loaded = load_ohlcv_csv(csv)

    assert loaded["date"].is_monotonic_increasing
    assert loaded["asset_class"].unique().tolist() == ["us_equity"]


def test_unknown_symbol_requires_explicit_asset_class() -> None:
    frame = _market_data("XYZ", "us_equity").drop(columns="asset_class")

    with pytest.raises(ValueError, match="asset_class is required"):
        validate_ohlcv(frame)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"high": 1.0}, "OHLC values are inconsistent"),
        ({"volume": -1}, "volume cannot be negative"),
    ],
)
def test_invalid_market_rows_are_rejected(change: dict[str, float], message: str) -> None:
    frame = _market_data("SPY", "us_equity")
    for column, value in change.items():
        frame.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        validate_ohlcv(frame)


def test_demo_is_reproducible_and_uses_different_market_calendars() -> None:
    first = generate_demo_data(periods=8, seed=5)
    second = generate_demo_data(periods=8, seed=5)

    pd.testing.assert_frame_equal(first, second)
    crypto_dates = first.loc[first["symbol"] == "BTC", "date"]
    equity_dates = first.loc[first["symbol"] == "SPY", "date"]
    assert len(crypto_dates) == 8
    assert len(equity_dates) == 8
    assert (crypto_dates.dt.dayofweek >= 5).any()
    assert not (equity_dates.dt.dayofweek >= 5).any()


def test_signal_is_lagged_to_next_available_bar_and_costs_reduce_equity() -> None:
    data = _market_data("SPY", "us_equity")
    signals = calculate_signals(data, fast_window=1, slow_window=2)
    assert bool(signals.loc[3, "long_signal"])

    no_costs = run_backtest(
        data,
        initial_capital=10_000,
        commission_bps=0,
        slippage_bps=0,
        fast_window=1,
        slow_window=2,
    )
    with_costs = run_backtest(
        data,
        initial_capital=10_000,
        commission_bps=10,
        slippage_bps=20,
        fast_window=1,
        slow_window=2,
    )

    first_buy = no_costs.orders.query("side == 'buy' and status == 'executed'").iloc[0]
    assert first_buy["signal_date"] == data.loc[3, "date"]
    assert first_buy["execute_date"] == data.loc[4, "date"]
    assert with_costs.metrics["fees_paid"] > 0
    assert with_costs.daily_equity["equity"].iloc[-1] < no_costs.daily_equity["equity"].iloc[-1]


def test_sizing_uses_whole_equity_shares_and_fractional_crypto() -> None:
    data = pd.concat(
        [
            _market_data("SPY", "us_equity", start_price=250),
            _market_data("BTC", "crypto", start_price=500),
        ],
        ignore_index=True,
    )
    result = run_backtest(
        data,
        initial_capital=1_000,
        commission_bps=0,
        slippage_bps=0,
        max_investment_ratio=0.8,
        fast_window=1,
        slow_window=2,
    )
    buys = result.orders.query("side == 'buy' and status == 'executed'").set_index("symbol")

    assert buys.loc["SPY", "quantity"] == 1
    assert 0 < buys.loc["BTC", "quantity"] < 1
