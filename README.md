# quant-research-system

quant-research-system is a research-oriented quantitative project. The existing scripts contain an A-share long-only research pipeline; the new local workbench adds an isolated first version for US equities and crypto without replacing that pipeline.

The repository is designed to demonstrate end-to-end quant engineering capabilities, including:

- data preparation and universe construction
- feature engineering workflows
- alpha model training and evaluation
- portfolio backtesting and risk analysis
- strategy parameter sweeps and reporting

## Project Goal

The goal is to provide modular workflows for turning market data into simulated portfolio decisions, then validating them through reproducible backtests and diagnostics.

## US Equities + Crypto Workbench (MVP)

The MVP is a local Streamlit app for daily OHLCV research. It includes an explainable long-only baseline: an asset is held when its close is above a slow simple moving average and its selected lookback return is positive. A close-based signal is filled at the next available bar's open. The simulator models configurable commission and slippage, whole-share US-equity sizing, and fractional crypto sizing.

This is a backtest/paper-mode simulation only. It does not connect to a broker, place orders, retrieve live prices, or verify the accuracy or licensing of uploaded data. Results are historical simulations, not predictions or investment advice. Dividends, taxes, funding, exchange-specific fee schedules, and other production execution rules are not modeled. The demo series are deterministic synthetic data for exercising the application; they are not real or reconstructed market history and must not be used to evaluate strategy performance.

### Run locally

Use Python 3.10 or newer. Install only the app dependencies rather than the legacy A-share research environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-app.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

Choose **Synthetic demo** to explore the interface or upload a historical CSV. To run the new focused tests, install `requirements-test.txt` and run `python -m pytest tests/test_quant_workbench.py`.

### CSV schema and provenance

CSV files must contain one daily bar per asset with columns `date,symbol,open,high,low,close,volume`. Prices and initial capital are assumed to be denominated in USD. An optional `asset_class` column must be `us_equity` or `crypto` (the aliases `equity` and `stock` are also accepted for US equities). If `asset_class` is omitted, only the built-in symbols `SPY`, `QQQ`, `BTC`, `BTC-USD`, `ETH`, and `ETH-USD` are inferred; any other symbol must include the column. Dates should identify the daily bar's calendar date. Duplicate `(date, symbol)` rows, non-positive OHLC prices, negative volume, and inconsistent OHLC ranges are rejected.

The app reads only the uploaded file or generates its clearly labeled synthetic demo locally. It does not attribute an uploaded file to a provider. Users are responsible for recording their source, date range, corporate-action adjustments, and data-use rights. Minute bars and online data providers are outside this MVP.

### Existing A-share pipeline

The `core/data_build`, `core/universe`, `core/features`, and `core/models` scripts retain their existing A-share-oriented paths and assumptions. Their historical data files are not included in this checkout. The standalone app dependencies are listed in `requirements-app.txt`; the repository's pre-existing Poetry metadata and `requirements.txt` describe a separate legacy research environment.

## What This Repository Highlights

- **System design**: clear separation between data, features, modeling, and backtesting modules  
- **Research workflow**: iterative experiments for strategy tuning and risk/return trade-offs  
- **Engineering practices**: script-based pipelines, structured outputs, and test coverage  

## Repository Structure

- `core/data_build`: data ingestion and canonicalization
- `core/universe`: tradable universe construction rules
- `core/features`: feature pipeline components
- `core/models/Xgboost`: alpha modeling, prediction, and evaluation
- `core/backtest`: strategy simulation, parameter sweeps, and reports
- `dq`: data quality checks
- `tests`: validation and guardrail tests

## Notes on Public Version

This public repository is intentionally sanitized for portfolio and interview purposes.
Certain strategy-sensitive details (for example, proprietary alpha definitions and some production configurations) are omitted or abstracted.

The focus of this version is to showcase the architecture, workflow, and implementation quality of a practical quant research system.
