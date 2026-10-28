# GoldWatch

GoldWatch collects historical gold futures data, calculates a small set of price and volatility features, and presents the resulting series in an interactive Streamlit dashboard. Dashboard anomaly signals are calculated from prior trading sessions and are intended for exploration, not trading advice.

## Features

- Downloads Yahoo Finance `GC=F` OHLCV history at a selected date range and interval.
- Normalizes dates and column names, sorts and deduplicates observations, and checks price and volume data for invalid values and OHLC bounds.
- Preserves valid price rows when volume is missing; it does not impute missing prices. Adds daily return, absolute return, price change, and intraday range features.
- Saves raw and processed CSV files and prints a run summary.
- Flags unusual days with a rolling Z-score on returns and volume, and backtests whether upward spikes are followed by declines (`src/anomaly_detection.py`, `src/backtest.py`).
- Provides an interactive dashboard: price chart with flagged days, an explanation for every flagged day, backtest results, a data browser with CSV export, and a plain-language guide to the method.

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

The dashboard runs the anomaly detection and backtest directly, so its figures match `scripts/run_anomaly_detection.py`. It has five pages, listed in the top bar:

| Page | What it shows |
|---|---|
| Overview | Latest price, range selector, price chart with flagged days, key statistics, the latest flagged day, and a card for each section |
| Anomalies | Counts by type, return Z-score chart, and a table of flagged days. Select a day to see in plain words why it was flagged |
| Backtest | How often gold fell after an upward spike vs. after any day, for 1–20 day holding periods, with a significance test |
| Data | Browse prices, anomaly scores or backtest events, filter by date, and download what you see |
| How it works | The four-step method, a worked example, a glossary and the limitations |

The detection window and threshold are under **Settings** (top right) and apply to every page; **Help** explains each page and term. Motion is kept light: panels rise in when a page opens, key numbers count up, and all of it is turned off when the operating system's "reduce motion" setting is on.

Code layout: `app.py` (entry point and navigation), `dashboard/core.py` (data access, shared components, charts), `dashboard/views/` (one file per page), `dashboard/style.css` (styling), `static/` (logo and the bundled IBM Plex Sans font, SIL Open Font License), and `.streamlit/config.toml` (theme).

Anomaly scores and event labels are descriptive statistical signals based on available history. They do not establish that an event is economically significant or predict future prices.

## Tests

Run the offline pytest suite from the project root:

```bash
python -m pytest -v
```

The data collection tests mock Yahoo Finance downloads; preprocessing and anomaly detection tests use synthetic data; dashboard tests render every page with Streamlit's headless test runner. The suite does not require a live market-data request.

## Data source and limitations

The pipeline requests the Yahoo Finance continuous gold futures symbol `GC=F` through yfinance, with `auto_adjust=False`. Yahoo Finance is an external source; availability, historical coverage, field definitions, and returned values may change. This is not a direct exchange feed. The continuous ticker is not an individual futures contract, and its reported volume should not be assumed to represent total COMEX market activity or open interest. Volume before 2020 is too small to be real exchange volume, so volume signals only use data from 2020, and zero-volume days are treated as unavailable. Returns are measured between consecutive available observations, so gaps may include weekends, exchange holidays, or missing source data.

## Outputs

- `data/raw/gold_futures_raw.csv`: normalized downloaded OHLCV observations.
- `data/processed/gold_futures_processed.csv`: cleaned observations with return and price-range features.
- `data/processed/gold_futures_analysis.csv`, `gold_futures_backtest.csv`, `gold_futures_backtest_summary.csv`: anomaly scores and backtest results written by `scripts/run_anomaly_detection.py`. The dashboard computes the same results live, so these files are for reports and further analysis.
