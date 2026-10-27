# GoldWatch

GoldWatch collects historical gold futures data, calculates a small set of price and volatility features, and presents the resulting series in an interactive Streamlit dashboard. Dashboard anomaly signals are calculated from prior trading sessions and are intended for exploration, not trading advice.

## Features

- Downloads Yahoo Finance `GC=F` OHLCV history at a selected date range and interval.
- Normalizes dates and column names, sorts and deduplicates observations, and checks price and volume data for invalid values and OHLC bounds.
- Preserves valid price rows when volume is missing; it does not impute missing prices. Adds daily return, absolute return, price change, and intraday range features.
- Saves raw and processed CSV files and prints a run summary.
- Provides a dashboard with date filtering, candlestick/line/area price views, volume and return charts, data exploration and CSV export.
- Calculates causal rolling robust Z-score signals for returns and observed volume. Its alert threshold and lookback are adjustable. Optional compatible analysis or backtest CSVs can be loaded; backtest results are shown only when such data is available.

## Stack

Python, pandas, NumPy, yfinance, Streamlit, Plotly, and pytest. See `requirements.txt` for package requirements.

## Installation

Use Python 3.10 or newer. From the project root, create and activate a virtual environment if desired, then install the requirements:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the data pipeline

From the project root:

```bash
python scripts/run_data_pipeline.py
```

By default, this downloads the maximum history available from Yahoo Finance at daily frequency and writes:

- `data/raw/gold_futures_raw.csv`
- `data/processed/gold_futures_processed.csv`

Specify a range or interval with `--start`, `--end`, and `--interval`. For example:

```bash
python scripts/run_data_pipeline.py --start 2015-01-01 --end 2025-01-01 --interval 1d
python scripts/run_data_pipeline.py --help
```

The end date follows yfinance's date-range behavior and may be exclusive. Use `--output-dir` to change the base data directory, `--no-raw` or `--no-processed` to skip an export, and `--verbose` for debug logging.

## Run the dashboard

The dashboard expects `data/processed/gold_futures_processed.csv`; generate it with the pipeline first if it is missing. Start Streamlit from the project root:

```bash
streamlit run app.py
```

Use the sidebar to select the date range, chart style, rolling lookback, alert threshold, and volume score weight. The dashboard also offers a raw-data toggle when `data/raw/gold_futures_raw.csv` exists. It exports the selected table and date range as CSV.

Anomaly scores and event labels are descriptive statistical signals based on available history. They do not establish that an event is economically significant or predict future prices. The backtest tab requires compatible forward-outcome data and does not manufacture performance results when it is absent.

## Tests

Run the offline pytest suite from the project root:

```bash
python -m pytest -v
```

The data collection tests mock Yahoo Finance downloads; preprocessing tests use synthetic data. The suite does not require a live market-data request.

## Data source and limitations

The pipeline requests the Yahoo Finance continuous gold futures symbol `GC=F` through yfinance, with `auto_adjust=False`. Yahoo Finance is an external source; availability, historical coverage, field definitions, and returned values may change. This is not a direct exchange feed. The continuous ticker is not an individual futures contract, and its reported volume should not be assumed to represent total COMEX market activity or open interest. Zero source-volume values are excluded from dashboard volume scoring and treated as unavailable observed volume. Returns are measured between consecutive available observations, so gaps may include weekends, exchange holidays, or missing source data.

## Outputs

- `data/raw/gold_futures_raw.csv`: normalized downloaded OHLCV observations.
- `data/processed/gold_futures_processed.csv`: cleaned observations with return and price-range features.
- Optional dashboard integrations: `data/processed/gold_futures_analysis.csv`, `data/processed/gold_futures_backtest.csv`, or `app_data/analysis.csv`. The dashboard accepts analysis Z-score fields and forward-return/backtest fields as implemented in `app.py`; without compatible backtest data, no backtest metrics are presented.
