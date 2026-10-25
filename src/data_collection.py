"""
Data Collection Module for GoldWatch.

Fetches historical gold futures data (GC=F) from Yahoo Finance using yfinance.
Handles MultiIndex column structures, timezone normalization, duplicate dates,
and network exceptions gracefully.
"""

import logging
from typing import Optional, Union
from datetime import datetime, date
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Canonical OHLCV column mapping
CANONICAL_COLUMNS = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "Adj Close",
    "volume": "Volume",
    "date": "Date",
}


class DataCollectionError(Exception):
    """Raised when data collection fails due to network, validation, or empty response."""
    pass


def normalize_columns(df: pd.DataFrame, ticker: str = "GC=F") -> pd.DataFrame:
    """
    Standardize DataFrame column names from yfinance, flattening MultiIndex columns
    and converting names to standard Title Case (Open, High, Low, Close, Adj Close, Volume).

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame returned by yfinance download.
    ticker : str
        Ticker symbol downloaded, used for matching MultiIndex levels.

    Returns
    -------
    pd.DataFrame
        DataFrame with flattened, standardized column names.
    """
    if df.empty:
        return df.copy()

    df = df.copy()

    # Handle MultiIndex column structures
    if isinstance(df.columns, pd.MultiIndex):
        logger.debug("MultiIndex detected in columns: %s", df.columns)
        # Check if level 0 contains price names
        level_0_lower = [str(col).lower() for col in df.columns.get_level_values(0)]
        level_1_lower = [str(col).lower() for col in df.columns.get_level_values(1)]

        price_keywords = {"open", "high", "low", "close", "adj close", "volume"}

        if any(col in price_keywords for col in level_0_lower):
            df.columns = df.columns.get_level_values(0)
        elif any(col in price_keywords for col in level_1_lower):
            df.columns = df.columns.get_level_values(1)
        else:
            # Fallback: join levels if not distinguishable
            df.columns = [
                "_".join([str(c) for c in col if str(c) and str(c) != ticker]).strip("_")
                for col in df.columns
            ]

    # Normalize column names to Title Case canonical mapping
    renamed_cols = {}
    for col in df.columns:
        col_clean = str(col).strip().lower()
        if col_clean in CANONICAL_COLUMNS:
            renamed_cols[col] = CANONICAL_COLUMNS[col_clean]
        else:
            renamed_cols[col] = str(col).strip()

    df = df.rename(columns=renamed_cols)
    return df


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has a consistent 'Date' column in YYYY-MM-DD format,
    sorted chronologically, with duplicate dates resolved.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Date either in columns or in the index.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'Date' as the first column and integer index.
    """
    if df.empty:
        return df.copy()

    df = df.copy()

    # If Date is in index, reset it to a column
    if "Date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex) or str(df.index.name).lower() == "date":
            df["Date"] = df.index
            df = df.reset_index(drop=True)
        else:
            raise DataCollectionError("Downloaded data lacks a recognizable Date index or column.")

    # Convert Date to datetime and strip timezone if present
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)

    # Remove any rows with unparseable dates
    invalid_dates = df["Date"].isna().sum()
    if invalid_dates > 0:
        logger.warning("Dropping %d rows with unparseable dates.", invalid_dates)
        df = df.dropna(subset=["Date"])

    # Format Date as clean daily string YYYY-MM-DD
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    # Sort chronologically
    df = df.sort_values(by="Date", ascending=True)

    # Check and handle duplicate dates
    duplicate_count = df.duplicated(subset=["Date"]).sum()
    if duplicate_count > 0:
        logger.warning(
            "Found %d duplicate date entries in raw download. Retaining latest record per date.",
            duplicate_count,
        )
        df = df.drop_duplicates(subset=["Date"], keep="last")

    # Reorder columns so Date is first
    cols = ["Date"] + [col for col in df.columns if col != "Date"]
    df = df[cols].reset_index(drop=True)

    return df


def fetch_gold_data(
    start_date: Optional[Union[str, datetime, date]] = None,
    end_date: Optional[Union[str, datetime, date]] = None,
    interval: str = "1d",
    ticker: str = "GC=F",
) -> pd.DataFrame:
    """
    Fetch historical gold futures data from Yahoo Finance.

    Parameters
    ----------
    start_date : str, datetime, date, optional
        Start date formatted as 'YYYY-MM-DD'. If None, downloads maximum practical history.
    end_date : str, datetime, date, optional
        End date formatted as 'YYYY-MM-DD'. If None, fetches up to latest available date.
    interval : str, default '1d'
        Data download interval (e.g. '1d' for daily).
    ticker : str, default 'GC=F'
        Yahoo Finance ticker symbol for gold futures.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        sorted chronologically.

    Raises
    ------
    DataCollectionError
        If download fails, returns empty data, or lacks essential price fields.
    """
    # Normalize date parameters to string YYYY-MM-DD if provided
    start_str = start_date.strftime("%Y-%m-%d") if isinstance(start_date, (datetime, date)) else start_date
    end_str = end_date.strftime("%Y-%m-%d") if isinstance(end_date, (datetime, date)) else end_date

    logger.info(
        "Fetching data for %s (start: %s, end: %s, interval: %s, auto_adjust=False)",
        ticker,
        start_str or "MAX_HISTORY",
        end_str or "LATEST",
        interval,
    )

    download_kwargs = {
        "tickers": ticker,
        "interval": interval,
        "auto_adjust": False,
        "progress": False,
    }

    if start_str is not None or end_str is not None:
        if start_str is not None:
            download_kwargs["start"] = start_str
        if end_str is not None:
            download_kwargs["end"] = end_str
    else:
        download_kwargs["period"] = "max"

    try:
        raw_df = yf.download(**download_kwargs)
    except Exception as exc:
        err_msg = f"Network or API failure while downloading {ticker} from Yahoo Finance: {exc}"
        logger.error(err_msg)
        raise DataCollectionError(err_msg) from exc

    if raw_df is None or raw_df.empty:
        err_msg = (
            f"No data returned by Yahoo Finance for ticker '{ticker}' "
            f"(start={start_str}, end={end_str}, interval={interval}). "
            "Please check network connectivity, ticker symbol, or specified date range."
        )
        logger.error(err_msg)
        raise DataCollectionError(err_msg)

    # Normalize columns and dates
    df = normalize_columns(raw_df, ticker=ticker)
    df = normalize_dates(df)

    # Validate that at least Close exists
    if "Close" not in df.columns:
        err_msg = f"Downloaded dataset for {ticker} does not contain essential 'Close' price column."
        logger.error(err_msg)
        raise DataCollectionError(err_msg)

    logger.info(
        "Successfully fetched %d records for %s from %s to %s.",
        len(df),
        ticker,
        df["Date"].min(),
        df["Date"].max(),
    )

    return df
