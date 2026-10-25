# GoldWatch — Gold Futures Market Anomaly Detection

**GoldWatch** is a market anomaly detection platform designed to identify unusual price shocks, volatility spikes, and volume dislocations in gold futures markets.

This module implements **Person A: Data Collection & Preprocessing**, delivering a modular, robust, and reproducible data pipeline that retrieves historical gold futures data from Yahoo Finance, validates and cleans the series according to rigorous financial time-series standards, computes daily returns and volatility metrics, and exports standardized datasets for downstream anomaly detection, backtesting, and dashboard visualization.

---

## 1. Instrument & Data Source Overview

- **Ticker:** `GC=F` (Gold Futures Continuous Contract, COMEX)
- **Data Provider:** Yahoo Finance via `yfinance`
- **Asset Class:** Commodity Futures
- **Historical Depth:** August 30, 2000 to Present (~26+ years of daily historical data)
- **Default Resolution:** Daily (`interval="1d"`)
- **Adjustment Setting:** `auto_adjust=False` (retaining explicit Close and Adjusted Close)

---

## 2. Architecture & Directory Structure

```text
DavProject/
├── data/
│   ├── raw/
│   │   └── gold_futures_raw.csv         <- Downloaded raw OHLCV market data
│   └── processed/
│       └── gold_futures_processed.csv   <- Cleaned, validated dataset with returns
├── src/
│   ├── __init__.py                     <- Package initialization and public API
│   ├── data_collection.py              <- Ticker downloader, MultiIndex flattener, date handler
│   ├── preprocessing.py                <- Financial validation, missing data rules, return engine
│   └── pipeline.py                     <- End-to-end orchestration and CSV export
├── scripts/
│   └── run_data_pipeline.py            <- Command-line interface (CLI) entry point
├── tests/
│   ├── test_data_collection.py         <- Mocked network and download tests
│   └── test_preprocessing.py           <- 12 financial cleaning & return calculation tests
├── pytest.ini                          <- Pytest configuration
├── requirements.txt                    <- Dependency specifications
├── progress.md                         <- Person A implementation status tracker
└── README.md                           <- Documentation and integration guide
```

---

## 3. Installation & Prerequisites

Python 3.10+ is recommended (Python 3.13 supported).

```bash
# 1. Activate your virtual environment (if using .venv on Windows)
.venv\Scripts\activate

# 2. Install required dependencies
pip install -r requirements.txt
```

### Dependencies
- `yfinance>=0.2.38` — Market data retrieval
- `pandas>=2.0.0` — Tabular data manipulation and series calculations
- `numpy>=1.24.0` — Numerical operations
- `pytest>=7.0.0` — Automated unit testing framework

---

## 4. Execution Guide

### Option A: Command-Line Interface (CLI)

Run from the project root:

```bash
# Download maximum historical dataset (from 2000 to present)
python scripts/run_data_pipeline.py

# Download a specific historical window
python scripts/run_data_pipeline.py --start 2015-01-01 --end 2025-12-31

# Download with custom interval and custom output directory
python scripts/run_data_pipeline.py --start 2020-01-01 --interval 1d --output-dir data/custom

# Enable detailed verbose logging
python scripts/run_data_pipeline.py --verbose
```

#### CLI Options
| Argument | Description | Default |
|---|---|---|
| `--start` | Historical start date (`YYYY-MM-DD`) | Earliest available (`2000-08-30`) |
| `--end` | Historical end date (`YYYY-MM-DD`) | Current trading date |
| `--interval` | Sampling frequency (e.g. `1d`, `1wk`, `1mo`) | `1d` |
| `--output-dir` | Custom base directory for outputs | `data/` |
| `--no-raw` | Skip exporting raw CSV | `False` |
| `--no-processed`| Skip exporting processed CSV | `False` |
| `--verbose`, `-v`| Enable debug-level logging | `False` |

---

### Option B: Programmatic Python API

Downstream modules can import and invoke the pipeline directly in Python:

```python
from src.pipeline import run_data_pipeline

# Fetch and process the maximum historical gold dataset
df = run_data_pipeline()

# Or fetch a custom window without saving to disk
df_custom = run_data_pipeline(
    start_date="2022-01-01",
    end_date="2024-12-31",
    save_raw=False,
    save_processed=False
)

print(df.head())
```

---

## 5. Dataset Schema & Column Specifications

The processed dataset is exported to `data/processed/gold_futures_processed.csv`.

| Column | Data Type | Description |
|---|---|---|
| `Date` | `string` (`YYYY-MM-DD`) | Daily trading date, sorted chronologically ascending |
| `Open` | `float64` | Opening price in USD per troy ounce |
| `High` | `float64` | Session high price in USD per troy ounce |
| `Low` | `float64` | Session low price in USD per troy ounce |
| `Close` | `float64` | Session closing / settlement price in USD |
| `Adj Close` | `float64` | Adjusted closing price |
| `Volume` | `float64` / `int64` | Session trading volume reported by Yahoo Finance |
| `Daily_Return` | `float64` | Daily percentage return: $(Close_t / Close_{t-1}) - 1$ |
| `Daily_Price_Change` | `float64` | Absolute dollar change: $Close_t - Close_{t-1}$ |
| `Absolute_Return` | `float64` | Absolute return magnitude: $\|Daily\_Return\|$ |
| `Price_Range` | `float64` | Intraday spread: $High_t - Low_t$ |
| `Price_Range_Percentage` | `float64` | Relative intraday volatility: $Price\_Range / Close_t$ |

---

## 6. Financial Methodology & Missing Data Policy

### Return Calculation
$$Daily\_Return_t = \frac{Close_t}{Close_{t-1}} - 1$$

- **Chronological Sequencing:** Observations are strictly ordered by date prior to return calculations.
- **First Observation:** Because no prior observation $Close_{t-1}$ exists for the first record, its `Daily_Return`, `Daily_Price_Change`, and `Absolute_Return` remain strictly `NaN`. They are **never** filled with zero.
- **Infinite Value Protection:** Core prices are verified positive ($Close > 0$), preventing division by zero. Any resulting infinite values are coerced to `NaN`.

### Missing Data & Cleaning Policy
1. **No Price Forward-Filling:** Missing or non-positive Close prices are dropped, never imputed via forward-fill or linear interpolation. In financial time-series, forward-filling artificially dampens volatility and skews anomaly detection.
2. **No Volume Fabrication:** Missing volume observations are retained as `NaN`. Missing volume in futures data indicates unrecorded volume for that session rather than zero contract turnover.
3. **Physical Market Bounds:** Rows with impossible relationships ($High < Low$, $High < Open$, $High < Close$, $Low > Open$, $Low > Close$, or $Volume < 0$) are rejected as corrupt feed records and logged.

---

## 7. Known Data Limitations

1. **Volume Interpretation in Futures (`GC=F`):** Unlike equities listed on centralized exchanges (NYSE/NASDAQ), gold futures trade across open-outcry pits, CME Globex electronic systems, and OTC transfers. The `Volume` field in Yahoo Finance represents tick/contract turnover reported to Yahoo's feed, not total aggregate COMEX open interest or multi-contract turnover. Downstream anomaly algorithms should take this into account.
2. **Weekend & Holiday Gaps:** Missing calendar dates represent weekends and COMEX exchange holidays (e.g., Thanksgiving, Christmas, New Year's Day). Trading returns are calculated between consecutive active trading sessions.
3. **MultiIndex Columns in yfinance:** Modern versions of `yfinance` return multi-tier column headers (`('Close', 'GC=F')`). The `data_collection` module automatically normalizes these into clean flat columns.

---

## 8. Integration Guide

### For Person B: Anomaly Detection & Backtesting
Person B can load the processed CSV directly or call `run_data_pipeline`:

```python
import pandas as pd

# Option 1: Load pre-generated processed dataset
df = pd.read_csv("data/processed/gold_futures_processed.csv", parse_dates=["Date"])
df = df.set_index("Date")

# Compute rolling Z-score on Daily_Return
rolling_mean = df["Daily_Return"].rolling(window=30).mean()
rolling_std = df["Daily_Return"].rolling(window=30).std()
df["Z_Score"] = (df["Daily_Return"] - rolling_mean) / rolling_std

# Option 2: Programmatic integration
from src.pipeline import run_data_pipeline
df = run_data_pipeline(start_date="2018-01-01", save_raw=False, save_processed=False)
```

### For Person C: Streamlit Dashboard
Person C can load `gold_futures_processed.csv` directly into Streamlit components:

```python
import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/gold_futures_processed.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

st.title("GoldWatch: Gold Futures Anomaly Dashboard")
st.line_chart(df.set_index("Date")["Close"])
st.bar_chart(df.set_index("Date")["Volume"])
st.line_chart(df.set_index("Date")["Daily_Return"])
```

---

## 9. Running Tests

Execute the automated pytest suite (all 18 unit tests run offline using synthetic data and mocked downloads):

```bash
pytest -v
```
