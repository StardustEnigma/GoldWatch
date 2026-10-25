"""
Data Preprocessing and Feature Engineering Module for GoldWatch.

Standardizes OHLCV data, performs financial market sanity checks, handles missing
values responsibly (no forward-filling of prices, no fabricated volumes), computes
daily returns, and generates lightweight analysis features.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CleaningConfig:
    """Configuration options for data cleaning and validation."""
    drop_invalid_ohlc: bool = True       # Drop rows where High < Low or OHLC <= 0
    enforce_strict_bounds: bool = True   # Drop rows where High < Open/Close or Low > Open/Close
    allow_missing_volume: bool = True    # Keep rows with missing/NaN volume (do not invent volume)
    flag_anomalous_spreads: bool = True  # Flag rows with unusual price relationships without dropping


class PreprocessingError(Exception):
    """Raised when preprocessing encounters unrecoverable structural or data errors."""
    pass


def clean_market_data(
    df: pd.DataFrame,
    config: Optional[CleaningConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean and validate historical market data according to sound financial principles.

    Cleaning Rules & Methodology:
    1. Chronological Sorting & Deduplication: Data is strictly sorted ascending by Date.
       Duplicate dates are identified; the latest record is preserved.
    2. Numerical Validity: Non-numeric strings and inf values are coerced to NaN.
    3. Essential Price Integrity: Rows with missing or non-positive (<= 0) Close prices
       are dropped. Close prices are NEVER forward-filled or interpolated.
    4. OHLC Sanity Checks: Rows where High < Low, High < Open, High < Close, Low > Open,
       or Low > Close are rejected as corrupted feed records.
    5. Volume Treatment: Volume is verified to be non-negative (Volume >= 0). Missing Volume
       is preserved as NaN rather than imputed with 0, as missing volume in GC=F signifies
       unreported session volume rather than zero market activity. Rows with valid prices
       are retained even if Volume is missing.

    Parameters
    ----------
    df : pd.DataFrame
        Raw market DataFrame containing at minimum 'Date' and 'Close'.
    config : CleaningConfig, optional
        Cleaning policy configuration.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Cleaned DataFrame and execution report dictionary.
    """
    if config is None:
        config = CleaningConfig()

    report: Dict[str, Any] = {
        "initial_rows": len(df) if df is not None else 0,
        "duplicate_dates_removed": 0,
        "missing_close_removed": 0,
        "non_positive_prices_removed": 0,
        "invalid_ohlc_bounds_removed": 0,
        "negative_volume_removed": 0,
        "missing_volume_preserved": 0,
        "infinite_values_coerced": 0,
        "final_cleaned_rows": 0,
    }

    if df is None or df.empty:
        logger.warning("Empty or None DataFrame passed to clean_market_data.")
        report["final_cleaned_rows"] = 0
        return pd.DataFrame(), report

    clean_df = df.copy()

    # Ensure Date column is present
    if "Date" not in clean_df.columns:
        if isinstance(clean_df.index, pd.DatetimeIndex) or str(clean_df.index.name).lower() == "date":
            clean_df["Date"] = pd.to_datetime(clean_df.index).strftime("%Y-%m-%d")
            clean_df = clean_df.reset_index(drop=True)
        else:
            raise PreprocessingError("Input DataFrame missing required 'Date' column or DatetimeIndex.")

    # Standardize Date representation
    clean_df["Date"] = pd.to_datetime(clean_df["Date"], errors="coerce")
    clean_df = clean_df.dropna(subset=["Date"])
    clean_df["Date"] = clean_df["Date"].dt.strftime("%Y-%m-%d")

    # Chronological sort
    clean_df = clean_df.sort_values(by="Date", ascending=True)

    # Duplicate dates check
    initial_dups = clean_df.duplicated(subset=["Date"]).sum()
    if initial_dups > 0:
        logger.warning("Detected %d duplicate dates. Keeping latest occurrence.", initial_dups)
        clean_df = clean_df.drop_duplicates(subset=["Date"], keep="last")
        report["duplicate_dates_removed"] = int(initial_dups)

    # Numeric conversion for price and volume columns
    numeric_cols = [col for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if col in clean_df.columns]
    for col in numeric_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # Check for infinite values across numeric columns
    inf_count = np.isinf(clean_df[numeric_cols].to_numpy()).sum()
    if inf_count > 0:
        logger.warning("Coercing %d infinite values to NaN across numeric fields.", inf_count)
        clean_df[numeric_cols] = clean_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        report["infinite_values_coerced"] = int(inf_count)

    # 1. Close price validation: Drop rows with missing Close
    if "Close" not in clean_df.columns:
        raise PreprocessingError("Dataset lacks mandatory 'Close' price column.")

    missing_close = clean_df["Close"].isna()
    if missing_close.sum() > 0:
        num_missing_close = int(missing_close.sum())
        logger.warning(
            "Dropping %d rows with missing Close prices (strict no-imputation policy).",
            num_missing_close,
        )
        report["missing_close_removed"] = num_missing_close
        clean_df = clean_df[~missing_close]

    # 2. Price completeness & positivity validation: Open, High, Low, Close must be present and strictly positive (> 0)
    core_price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in clean_df.columns]
    unusable_price_mask = pd.Series(False, index=clean_df.index)
    for col in core_price_cols:
        # Unusable if NaN/coerced-inf or non-positive (<= 0)
        unusable_price_mask |= clean_df[col].isna() | (clean_df[col] <= 0)

    if unusable_price_mask.sum() > 0:
        num_unusable = int(unusable_price_mask.sum())
        logger.warning("Dropping %d rows with unusable (missing, infinite, or non-positive) core prices.", num_unusable)
        report["non_positive_prices_removed"] = num_unusable
        clean_df = clean_df[~unusable_price_mask]

    # 3. OHLC bounds sanity checks
    if config.drop_invalid_ohlc and {"High", "Low"}.issubset(clean_df.columns):
        # High cannot be lower than Low
        high_low_invalid = clean_df["High"] < clean_df["Low"]
        if high_low_invalid.sum() > 0:
            num_hl_invalid = int(high_low_invalid.sum())
            logger.warning("Dropping %d rows where High < Low.", num_hl_invalid)
            report["invalid_ohlc_bounds_removed"] += num_hl_invalid
            clean_df = clean_df[~high_low_invalid]

    if config.enforce_strict_bounds and {"Open", "High", "Low", "Close"}.issubset(clean_df.columns):
        # High must be >= Open and >= Close
        # Low must be <= Open and <= Close
        strict_invalid = (
            (clean_df["High"] < clean_df["Open"])
            | (clean_df["High"] < clean_df["Close"])
            | (clean_df["Low"] > clean_df["Open"])
            | (clean_df["Low"] > clean_df["Close"])
        )
        if strict_invalid.sum() > 0:
            num_strict_invalid = int(strict_invalid.sum())
            logger.warning("Dropping %d rows violating High/Low OHLC bounding constraints.", num_strict_invalid)
            report["invalid_ohlc_bounds_removed"] += num_strict_invalid
            clean_df = clean_df[~strict_invalid]

    # 4. Volume validation
    if "Volume" in clean_df.columns:
        # Negative volume is physically impossible
        neg_volume = clean_df["Volume"] < 0
        if neg_volume.sum() > 0:
            num_neg_vol = int(neg_volume.sum())
            logger.warning("Dropping %d rows with negative Volume.", num_neg_vol)
            report["negative_volume_removed"] = num_neg_vol
            clean_df = clean_df[~neg_volume]

        # Missing volume count
        missing_vol_count = clean_df["Volume"].isna().sum()
        report["missing_volume_preserved"] = int(missing_vol_count)
        if missing_vol_count > 0:
            logger.info(
                "Preserved %d rows with NaN Volume (Volume not imputed with zero; valid prices retained).",
                missing_vol_count,
            )

    clean_df = clean_df.reset_index(drop=True)
    report["final_cleaned_rows"] = len(clean_df)

    logger.info(
        "Market data cleaning complete: %d rows in, %d rows out.",
        report["initial_rows"],
        report["final_cleaned_rows"],
    )

    return clean_df, report


def calculate_returns_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns and supporting market metrics.

    Features Added:
    - Daily_Return: Percentage change in Close: (Close_t / Close_{t-1}) - 1.
      The first observation remains NaN.
    - Daily_Price_Change: Close_t - Close_{t-1}. The first observation remains NaN.
    - Absolute_Return: Absolute magnitude |Daily_Return|. First observation is NaN.
    - Price_Range: High - Low. Reflects intraday price dispersion.
    - Price_Range_Percentage: Price_Range / Close. Relative intraday volatility.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned market DataFrame sorted chronologically.

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with return and volatility features.
    """
    if df.empty:
        return df.copy()

    df = df.copy()

    # Verify chronological sorting
    if "Date" in df.columns:
        df = df.sort_values(by="Date", ascending=True).reset_index(drop=True)

    if "Close" not in df.columns:
        raise PreprocessingError("Cannot calculate returns: 'Close' column missing.")

    # Calculate Daily_Return: (Close_t / Close_{t-1}) - 1
    # Close price series is guaranteed positive and non-null from cleaning step
    close_series = df["Close"]
    prev_close = close_series.shift(1)

    # Explicit daily return calculation
    daily_return = (close_series / prev_close) - 1.0

    # Ensure the first observation is explicitly NaN
    daily_return.iloc[0] = np.nan

    # Replace any potential infinite values with NaN (defensive check)
    daily_return = daily_return.replace([np.inf, -np.inf], np.nan)

    df["Daily_Return"] = daily_return

    # Calculate Daily_Price_Change: Close_t - Close_{t-1}
    price_change = close_series - prev_close
    price_change.iloc[0] = np.nan
    df["Daily_Price_Change"] = price_change

    # Calculate Absolute_Return: |Daily_Return|
    df["Absolute_Return"] = df["Daily_Return"].abs()

    # Calculate intraday volatility metrics if High and Low are present
    if {"High", "Low"}.issubset(df.columns):
        price_range = df["High"] - df["Low"]
        df["Price_Range"] = price_range
        # Price_Range_Percentage = Price_Range / Close
        df["Price_Range_Percentage"] = price_range / df["Close"]
        # Replace any infs with NaN defensively
        df["Price_Range_Percentage"] = df["Price_Range_Percentage"].replace([np.inf, -np.inf], np.nan)

    return df


def preprocess_gold_data(
    df: pd.DataFrame,
    config: Optional[CleaningConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute full preprocessing workflow: clean raw market data, validate integrity,
    and compute return and volatility features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw market data from data_collection module.
    config : CleaningConfig, optional
        Cleaning policy configuration.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (processed_dataframe, data_cleaning_report)
    """
    cleaned_df, report = clean_market_data(df, config=config)
    if cleaned_df.empty:
        return cleaned_df, report

    processed_df = calculate_returns_and_features(cleaned_df)
    return processed_df, report
