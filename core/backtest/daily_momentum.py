from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.market_data.daily import validate_ohlcv


@dataclass(frozen=True)
class BacktestResult:
    daily_equity: pd.DataFrame
    orders: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float | int | str]


def calculate_signals(
    market_data: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 50,
) -> pd.DataFrame:
    """Long when close is above the slow SMA and fast-window momentum is positive."""
    if fast_window < 1 or slow_window <= fast_window:
        raise ValueError("windows must satisfy 1 <= fast_window < slow_window")
    frame = validate_ohlcv(market_data)
    frame["slow_sma"] = frame.groupby("symbol", sort=False)["close"].transform(
        lambda close: close.rolling(slow_window, min_periods=slow_window).mean()
    )
    frame["momentum"] = frame.groupby("symbol", sort=False)["close"].pct_change(
        periods=fast_window
    )
    frame["long_signal"] = (
        (frame["close"] > frame["slow_sma"]) & (frame["momentum"] > 0)
    ).fillna(False)
    return frame


def _floor_quantity(quantity: float, asset_class: str) -> float:
    if asset_class == "us_equity":
        return float(math.floor(quantity))
    step = 1e-8
    return float(math.floor(quantity / step + 1e-9) * step)


def _build_metrics(
    daily_equity: pd.DataFrame,
    orders: pd.DataFrame,
    trades: pd.DataFrame,
    asset_classes: set[str],
    initial_capital: float,
) -> dict[str, float | int | str]:
    equity = daily_equity["equity"]
    returns = daily_equity["daily_return"]
    total_return = float(equity.iloc[-1] / initial_capital - 1.0)
    annualizer = 365.0 if "crypto" in asset_classes else 252.0
    intervals = max(len(daily_equity) - 1, 1)
    annual_return = float((equity.iloc[-1] / initial_capital) ** (annualizer / intervals) - 1.0)
    volatility_daily = float(returns.std(ddof=0)) if not returns.empty else 0.0
    sharpe = (
        float(returns.mean() / volatility_daily * math.sqrt(annualizer))
        if volatility_daily > 0
        else 0.0
    )
    drawdown = equity / equity.cummax() - 1.0
    closed_trades = trades[trades["exit_date"].notna()] if not trades.empty else trades
    win_rate = (
        float((closed_trades["pnl"] > 0).mean()) if not closed_trades.empty else 0.0
    )
    executed = orders[orders["status"] == "executed"] if not orders.empty else orders
    return {
        "start_date": str(daily_equity["date"].iloc[0].date()),
        "end_date": str(daily_equity["date"].iloc[-1].date()),
        "total_return": total_return,
        "annualized_return": annual_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(drawdown.min()),
        "annualized_volatility": volatility_daily * math.sqrt(annualizer),
        "win_rate": win_rate,
        "executed_orders": int(len(executed)),
        "closed_trades": int(len(closed_trades)),
        "fees_paid": float(orders["fee"].sum()) if not orders.empty else 0.0,
        "annualization_days": int(annualizer),
    }


def run_backtest(
    market_data: pd.DataFrame,
    initial_capital: float = 100_000.0,
    commission_bps: float = 1.0,
    slippage_bps: float = 5.0,
    max_investment_ratio: float = 0.90,
    fast_window: int = 20,
    slow_window: int = 50,
) -> BacktestResult:
    """Run long-only daily paper simulation with decisions filled on the next asset bar."""
    if initial_capital <= 0 or not math.isfinite(initial_capital):
        raise ValueError("initial_capital must be a finite positive number")
    if not 0 <= commission_bps < 10_000 or not 0 <= slippage_bps < 10_000:
        raise ValueError("commission_bps and slippage_bps must be in [0, 10000)")
    if not 0 < max_investment_ratio <= 1:
        raise ValueError("max_investment_ratio must be in (0, 1]")

    frame = calculate_signals(market_data, fast_window, slow_window)
    symbols = frame["symbol"].unique().tolist()
    budget_per_symbol = initial_capital * max_investment_ratio / len(symbols)
    fee_rate = commission_bps / 10_000.0
    slippage_rate = slippage_bps / 10_000.0
    next_bar: dict[tuple[pd.Timestamp, str], pd.Timestamp] = {}
    for symbol, group in frame.groupby("symbol", sort=False):
        dates = group["date"].tolist()
        for current, following in zip(dates, dates[1:]):
            next_bar[(current, symbol)] = following

    cash = float(initial_capital)
    positions: dict[str, dict[str, object]] = {}
    last_prices: dict[str, float] = {}
    pending: dict[tuple[pd.Timestamp, str], dict[str, object]] = {}
    orders_log: list[dict[str, object]] = []
    trades_log: list[dict[str, object]] = []
    equity_log: list[dict[str, object]] = []

    for date, bars in frame.groupby("date", sort=True):
        current_date = pd.Timestamp(date)
        for _, bar in bars.iterrows():
            symbol = str(bar["symbol"])
            asset_class = str(bar["asset_class"])
            order = pending.pop((current_date, symbol), None)
            if order is not None:
                side = str(order["side"])
                open_price = float(bar["open"])
                fill_price = open_price * (
                    1.0 + slippage_rate if side == "buy" else 1.0 - slippage_rate
                )
                position = positions.get(symbol)
                quantity = 0.0
                fee = 0.0
                status = "executed"
                if side == "buy":
                    available_budget = min(budget_per_symbol, cash) / (1.0 + fee_rate)
                    quantity = _floor_quantity(available_budget / fill_price, asset_class)
                    if quantity <= 0:
                        status = "skipped_insufficient_cash"
                    else:
                        notional = quantity * fill_price
                        fee = notional * fee_rate
                        cash -= notional + fee
                        positions[symbol] = {
                            "quantity": quantity,
                            "entry_date": current_date,
                            "entry_price": fill_price,
                            "cost_basis": notional + fee,
                            "entry_fee": fee,
                            "asset_class": asset_class,
                        }
                elif position is None:
                    status = "skipped_no_position"
                else:
                    quantity = float(position["quantity"])
                    notional = quantity * fill_price
                    fee = notional * fee_rate
                    cash += notional - fee
                    trades_log.append(
                        {
                            "symbol": symbol,
                            "asset_class": asset_class,
                            "entry_date": position["entry_date"],
                            "exit_date": current_date,
                            "entry_price": float(position["entry_price"]),
                            "exit_price": fill_price,
                            "quantity": quantity,
                            "pnl": notional - fee - float(position["cost_basis"]),
                            "fees": float(position["entry_fee"]) + fee,
                            "holding_days": int(
                                (current_date - pd.Timestamp(position["entry_date"])).days
                            ),
                        }
                    )
                    del positions[symbol]

                orders_log.append(
                    {
                        "signal_date": order["signal_date"],
                        "execute_date": current_date,
                        "symbol": symbol,
                        "asset_class": asset_class,
                        "side": side,
                        "status": status,
                        "price": fill_price if status == "executed" else np.nan,
                        "quantity": quantity if status == "executed" else 0.0,
                        "fee": fee,
                    }
                )

            last_prices[symbol] = float(bar["close"])
            has_position = symbol in positions
            should_hold = bool(bar["long_signal"])
            next_date = next_bar.get((current_date, symbol))
            if next_date is not None and should_hold != has_position:
                pending[(next_date, symbol)] = {
                    "signal_date": current_date,
                    "side": "buy" if should_hold else "sell",
                }

        market_value = sum(
            float(position["quantity"]) * last_prices.get(symbol, 0.0)
            for symbol, position in positions.items()
        )
        equity = cash + market_value
        previous_equity = (
            float(equity_log[-1]["equity"]) if equity_log else float(initial_capital)
        )
        equity_log.append(
            {
                "date": current_date,
                "equity": equity,
                "cash": cash,
                "position_value": market_value,
                "daily_return": equity / previous_equity - 1.0 if previous_equity else 0.0,
            }
        )

    final_date = pd.Timestamp(frame["date"].max())
    for symbol, position in positions.items():
        trades_log.append(
            {
                "symbol": symbol,
                "asset_class": str(position["asset_class"]),
                "entry_date": position["entry_date"],
                "exit_date": pd.NaT,
                "entry_price": float(position["entry_price"]),
                "exit_price": np.nan,
                "quantity": float(position["quantity"]),
                "pnl": (
                    float(position["quantity"]) * last_prices[symbol]
                    - float(position["cost_basis"])
                ),
                "fees": float(position["entry_fee"]),
                "holding_days": int((final_date - pd.Timestamp(position["entry_date"])).days),
            }
        )

    daily_equity = pd.DataFrame(equity_log)
    orders = pd.DataFrame(
        orders_log,
        columns=[
            "signal_date",
            "execute_date",
            "symbol",
            "asset_class",
            "side",
            "status",
            "price",
            "quantity",
            "fee",
        ],
    )
    trades = pd.DataFrame(
        trades_log,
        columns=[
            "symbol",
            "asset_class",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "quantity",
            "pnl",
            "fees",
            "holding_days",
        ],
    )
    metrics = _build_metrics(
        daily_equity,
        orders,
        trades,
        set(frame["asset_class"].astype(str)),
        initial_capital,
    )
    return BacktestResult(daily_equity, orders, trades, metrics)
