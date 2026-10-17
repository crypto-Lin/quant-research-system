from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = ("SPY", "QQQ", "BTC", "ETH")
REQUIRED_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume")
KNOWN_ASSET_CLASSES = {
    "SPY": "us_equity",
    "QQQ": "us_equity",
    "BTC": "crypto",
    "BTC-USD": "crypto",
    "ETH": "crypto",
    "ETH-USD": "crypto",
}


def validate_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Return normalized daily OHLCV data or raise ValueError with the violated rule."""
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"OHLCV data is missing required columns: {missing}")
    if data.empty:
        raise ValueError("OHLCV data must contain at least one row")

    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("date contains missing or invalid values")
    frame["date"] = frame["date"].dt.normalize()

    frame["symbol"] = frame["symbol"].astype("string").str.strip().str.upper()
    if frame["symbol"].isna().any() or frame["symbol"].eq("").any():
        raise ValueError("symbol contains missing or empty values")

    if "asset_class" not in frame.columns:
        frame["asset_class"] = frame["symbol"].map(KNOWN_ASSET_CLASSES)
        if frame["asset_class"].isna().any():
            unknown = sorted(frame.loc[frame["asset_class"].isna(), "symbol"].unique())
            raise ValueError(
                "asset_class is required for unrecognized symbols: "
                + ", ".join(unknown)
            )
    else:
        class_aliases = {
            "equity": "us_equity",
            "stock": "us_equity",
            "us_equity": "us_equity",
            "crypto": "crypto",
        }
        raw_classes = frame["asset_class"].astype("string").str.strip().str.lower()
        frame["asset_class"] = raw_classes.map(class_aliases)
        if frame["asset_class"].isna().any():
            raise ValueError("asset_class must be 'us_equity' or 'crypto'")
    if frame.groupby("symbol", sort=False)["asset_class"].nunique().gt(1).any():
        raise ValueError("each symbol must use a single asset_class")

    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        values = frame[column].to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains missing, non-numeric, or non-finite values")

    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (frame["volume"] < 0).any():
        raise ValueError("volume cannot be negative")
    invalid_ohlc = (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
    )
    if invalid_ohlc.any():
        raise ValueError("OHLC values are inconsistent: high/low do not contain open and close")
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("duplicate (date, symbol) rows are not allowed")

    return frame[
        ["date", "symbol", "asset_class", "open", "high", "low", "close", "volume"]
    ].sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)


def load_ohlcv_csv(source: str | Path | IO[str] | IO[bytes] | BytesIO) -> pd.DataFrame:
    """Load and validate a user-provided daily OHLCV CSV file."""
    try:
        data = pd.read_csv(source)
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc
    data.columns = [str(column).strip().lower() for column in data.columns]
    return validate_ohlcv(data)


def generate_demo_data(
    start: str | pd.Timestamp = "2020-01-01",
    periods: int = 520,
    seed: int = 17,
) -> pd.DataFrame:
    """Create deterministic synthetic OHLCV bars; these are not market observations."""
    if periods < 2:
        raise ValueError("periods must be at least 2")

    start_date = pd.Timestamp(start).normalize()
    assets = (
        ("SPY", "us_equity", 400.0, 0.00025, 0.010),
        ("QQQ", "us_equity", 320.0, 0.00030, 0.013),
        ("BTC", "crypto", 40000.0, 0.00035, 0.035),
        ("ETH", "crypto", 2200.0, 0.00032, 0.040),
    )
    frames: list[pd.DataFrame] = []
    for index, (symbol, asset_class, base_price, drift, volatility) in enumerate(assets):
        dates = (
            pd.date_range(start_date, periods=periods, freq="D")
            if asset_class == "crypto"
            else pd.bdate_range(start_date, periods=periods)
        )
        rng = np.random.default_rng(seed + index)
        returns = rng.normal(drift, volatility, size=periods)
        close = base_price * np.exp(np.cumsum(returns))
        previous_close = np.concatenate(([base_price], close[:-1]))
        open_price = previous_close * (1.0 + rng.normal(0.0, volatility * 0.25, size=periods))
        spread = np.abs(rng.normal(volatility * 0.35, volatility * 0.15, size=periods))
        high = np.maximum(open_price, close) * (1.0 + spread)
        low = np.minimum(open_price, close) * np.maximum(1.0 - spread, 0.01)
        volume = rng.integers(100_000, 5_000_000, size=periods)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "asset_class": asset_class,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )
    return validate_ohlcv(pd.concat(frames, ignore_index=True))
