"""
Unit tests for data validation, cleaning, and feature engineering in GoldWatch.
All tests use synthetic data and require NO live network connectivity.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    clean_market_data,
    calculate_returns_and_features,
    preprocess_gold_data,
    CleaningConfig,
)
from src.data_collection import normalize_columns


@pytest.fixture
def sample_valid_market_data() -> pd.DataFrame:
    """Fixture providing clean synthetic OHLCV market data."""
    return pd.DataFrame({
        "Date": ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"],
        "Open": [1800.0, 1810.0, 1805.0, 1820.0],
        "High": [1825.0, 1830.0, 1820.0, 1840.0],
        "Low": [1795.0, 1800.0, 1790.0, 1815.0],
        "Close": [1820.0, 1805.0, 1815.0, 1835.0],
        "Volume": [150000.0, 140000.0, 160000.0, 175000.0],
    })


# 1. Correct calculation of daily returns
def test_correct_calculation_of_daily_returns():
    df = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "Close": [100.0, 110.0, 99.0],
    })
    res = calculate_returns_and_features(df)
    assert "Daily_Return" in res.columns
    # Row 0 must be NaN
    assert pd.isna(res.loc[0, "Daily_Return"])
    # Row 1: (110 - 100) / 100 = 0.10 (+10%)
    assert pytest.approx(res.loc[1, "Daily_Return"], 1e-6) == 0.10
    # Row 2: (99 - 110) / 110 = -0.10 (-10%)
    assert pytest.approx(res.loc[2, "Daily_Return"], 1e-6) == -0.10


# 2. Correct chronological sorting
def test_correct_chronological_sorting():
    unsorted_df = pd.DataFrame({
        "Date": ["2023-01-05", "2023-01-01", "2023-01-03"],
        "Open": [100.0, 90.0, 95.0],
        "High": [105.0, 95.0, 100.0],
        "Low": [98.0, 88.0, 92.0],
        "Close": [102.0, 92.0, 98.0],
        "Volume": [1000, 1200, 1100],
    })
    clean_df, _ = clean_market_data(unsorted_df)
    assert clean_df["Date"].tolist() == ["2023-01-01", "2023-01-03", "2023-01-05"]

    # Returns must be computed on chronologically sorted order
    processed = calculate_returns_and_features(clean_df)
    # Expected return between 2023-01-01 (Close=92.0) and 2023-01-03 (Close=98.0)
    expected_ret = (98.0 / 92.0) - 1.0
    assert pytest.approx(processed.loc[1, "Daily_Return"], 1e-6) == expected_ret


# 3. Duplicate-date handling
def test_duplicate_date_handling():
    df_with_dups = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03"],
        "Open": [100.0, 101.0, 102.0, 105.0],
        "High": [105.0, 106.0, 107.0, 110.0],
        "Low": [95.0, 96.0, 97.0, 100.0],
        "Close": [102.0, 103.0, 104.0, 108.0],
        "Volume": [1000, 1500, 1600, 2000],
    })
    clean_df, report = clean_market_data(df_with_dups)
    assert len(clean_df) == 3
    assert clean_df["Date"].tolist() == ["2023-01-01", "2023-01-02", "2023-01-03"]
    assert report["duplicate_dates_removed"] == 1
    # Check that latest duplicate record was kept (Close=104.0)
    assert clean_df.loc[clean_df["Date"] == "2023-01-02", "Close"].values[0] == 104.0


# 4. Missing Close prices
def test_missing_close_prices_dropped():
    df_missing_close = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "Open": [100.0, 105.0, 110.0],
        "High": [105.0, 110.0, 115.0],
        "Low": [95.0, 100.0, 105.0],
        "Close": [102.0, np.nan, 112.0],
        "Volume": [1000, 2000, 3000],
    })
    clean_df, report = clean_market_data(df_missing_close)
    # The middle row with NaN Close must be dropped, NOT forward-filled
    assert len(clean_df) == 2
    assert "2023-01-02" not in clean_df["Date"].values
    assert report["missing_close_removed"] == 1


# 5. Missing Volume values preserved (not imputed with zero, row retained)
def test_missing_volume_values_preserved():
    df_missing_vol = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02"],
        "Open": [100.0, 105.0],
        "High": [105.0, 110.0],
        "Low": [95.0, 100.0],
        "Close": [102.0, 108.0],
        "Volume": [1000.0, np.nan],
    })
    clean_df, report = clean_market_data(df_missing_vol)
    # Row with NaN volume must be preserved
    assert len(clean_df) == 2
    assert pd.isna(clean_df.loc[1, "Volume"])
    assert report["missing_volume_preserved"] == 1


# 6. Infinite or invalid numerical values
def test_infinite_and_invalid_numerical_values():
    df_inf = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "Open": [100.0, np.inf, 110.0],
        "High": [105.0, 115.0, 115.0],
        "Low": [95.0, 95.0, 105.0],
        "Close": [102.0, 105.0, -np.inf],
        "Volume": [1000, 2000, 3000],
    })
    clean_df, report = clean_market_data(df_inf)
    # Row 2 has Close = -inf (coerced to NaN, so dropped)
    # Row 1 has Open = inf (coerced to NaN, violating positive price check)
    assert report["infinite_values_coerced"] >= 2
    assert len(clean_df) == 1
    assert clean_df.iloc[0]["Date"] == "2023-01-01"


# 7. Invalid OHLC relationships
def test_invalid_ohlc_relationships():
    df_invalid_ohlc = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
        # Valid row
        "Open": [100.0, 100.0, 100.0, 100.0, -10.0],
        # Row 1: High < Low (invalid)
        # Row 2: High < Close (invalid)
        # Row 3: Low > Open (invalid)
        # Row 4: Negative price (invalid)
        "High": [105.0, 90.0, 102.0, 110.0, 10.0],
        "Low": [95.0, 95.0, 95.0, 102.0, -15.0],
        "Close": [102.0, 92.0, 108.0, 105.0, 5.0],
        "Volume": [1000, 1000, 1000, 1000, 1000],
    })
    clean_df, report = clean_market_data(df_invalid_ohlc)
    assert len(clean_df) == 1
    assert clean_df.iloc[0]["Date"] == "2023-01-01"
    assert report["invalid_ohlc_bounds_removed"] >= 3 or report["non_positive_prices_removed"] >= 1


# 8. Negative Volume
def test_negative_volume_rejected():
    df_neg_vol = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02"],
        "Open": [100.0, 105.0],
        "High": [105.0, 110.0],
        "Low": [95.0, 100.0],
        "Close": [102.0, 108.0],
        "Volume": [1000.0, -500.0],
    })
    clean_df, report = clean_market_data(df_neg_vol)
    assert len(clean_df) == 1
    assert clean_df.iloc[0]["Date"] == "2023-01-01"
    assert report["negative_volume_removed"] == 1


# 9. Empty DataFrame input
def test_empty_dataframe_input():
    empty_df = pd.DataFrame()
    clean_df, report = clean_market_data(empty_df)
    assert clean_df.empty
    assert report["final_cleaned_rows"] == 0

    processed_df = calculate_returns_and_features(clean_df)
    assert processed_df.empty


# 10. First daily return remains undefined
def test_first_daily_return_remains_undefined(sample_valid_market_data):
    processed = calculate_returns_and_features(sample_valid_market_data)
    first_return = processed.loc[0, "Daily_Return"]
    assert pd.isna(first_return)
    assert not first_return == 0.0


# 11. No infinite values remain in calculated returns
def test_no_infinite_values_in_calculated_returns():
    # Test with very large and small Close prices
    df = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "Close": [1e-4, 1e8, 1e-4],
    })
    processed = calculate_returns_and_features(df)
    assert not np.isinf(processed["Daily_Return"]).any()
    assert not np.isinf(processed["Absolute_Return"]).any()


# 12. Correct handling of yfinance MultiIndex columns
def test_yfinance_multiindex_handling():
    # MultiIndex level 0 = Price, level 1 = Ticker
    cols = pd.MultiIndex.from_tuples([
        ("Close", "GC=F"),
        ("Open", "GC=F"),
        ("High", "GC=F"),
        ("Low", "GC=F"),
        ("Volume", "GC=F"),
        ("Adj Close", "GC=F"),
    ], names=["Price", "Ticker"])

    df = pd.DataFrame(
        [
            [1800.0, 1790.0, 1810.0, 1785.0, 100000, 1800.0],
            [1815.0, 1800.0, 1820.0, 1795.0, 120000, 1815.0],
        ],
        columns=cols,
        index=pd.date_range("2023-01-01", periods=2, freq="D", name="Date"),
    )

    normalized = normalize_columns(df, ticker="GC=F")
    assert not isinstance(normalized.columns, pd.MultiIndex)
    assert "Close" in normalized.columns
    assert "Open" in normalized.columns
    assert "High" in normalized.columns
    assert "Low" in normalized.columns
    assert "Volume" in normalized.columns


# Optional features test: Daily_Price_Change, Absolute_Return, Price_Range, Price_Range_Percentage
def test_optional_features_calculation(sample_valid_market_data):
    processed = calculate_returns_and_features(sample_valid_market_data)
    assert "Daily_Price_Change" in processed.columns
    assert "Absolute_Return" in processed.columns
    assert "Price_Range" in processed.columns
    assert "Price_Range_Percentage" in processed.columns

    # First row differences must be NaN
    assert pd.isna(processed.loc[0, "Daily_Price_Change"])
    assert pd.isna(processed.loc[0, "Absolute_Return"])

    # High=1825.0, Low=1795.0 -> Price_Range = 30.0
    assert pytest.approx(processed.loc[0, "Price_Range"], 1e-6) == 30.0
    # Close=1820.0 -> Price_Range_Percentage = 30.0 / 1820.0
    assert pytest.approx(processed.loc[0, "Price_Range_Percentage"], 1e-6) == (30.0 / 1820.0)
