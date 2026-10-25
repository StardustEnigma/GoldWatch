"""
Unit tests for data collection module in GoldWatch.
Uses mocking to test download behavior, network failures, empty responses,
and parameter configurations without making live HTTP requests.
"""

from unittest.mock import patch
import pandas as pd
import pytest

from src.data_collection import (
    fetch_gold_data,
    DataCollectionError,
    normalize_columns,
    normalize_dates,
)


@pytest.fixture
def mock_yf_response() -> pd.DataFrame:
    """Fixture providing a mock yfinance DataFrame with MultiIndex columns."""
    dates = pd.date_range("2023-01-01", periods=3, freq="D", name="Date")
    cols = pd.MultiIndex.from_tuples([
        ("Close", "GC=F"),
        ("Open", "GC=F"),
        ("High", "GC=F"),
        ("Low", "GC=F"),
        ("Volume", "GC=F"),
        ("Adj Close", "GC=F"),
    ], names=["Price", "Ticker"])
    data = [
        [1820.0, 1800.0, 1825.0, 1795.0, 100000, 1820.0],
        [1830.0, 1815.0, 1835.0, 1810.0, 120000, 1830.0],
        [1825.0, 1828.0, 1840.0, 1820.0, 110000, 1825.0],
    ]
    return pd.DataFrame(data, index=dates, columns=cols)


def test_fetch_gold_data_success(mock_yf_response):
    with patch("yfinance.download", return_value=mock_yf_response) as mock_dl:
        df = fetch_gold_data(start_date="2023-01-01", end_date="2023-01-03", interval="1d")

        mock_dl.assert_called_once_with(
            tickers="GC=F",
            interval="1d",
            auto_adjust=False,
            progress=False,
            start="2023-01-01",
            end="2023-01-03",
        )
        assert len(df) == 3
        assert "Date" in df.columns
        assert "Close" in df.columns
        assert df["Date"].tolist() == ["2023-01-01", "2023-01-02", "2023-01-03"]


def test_fetch_gold_data_max_period(mock_yf_response):
    with patch("yfinance.download", return_value=mock_yf_response) as mock_dl:
        df = fetch_gold_data(start_date=None, end_date=None, interval="1d")
        mock_dl.assert_called_once_with(
            tickers="GC=F",
            interval="1d",
            auto_adjust=False,
            progress=False,
            period="max",
        )
        assert len(df) == 3


def test_fetch_gold_data_network_failure():
    with patch("yfinance.download", side_effect=Exception("Connection timed out")):
        with pytest.raises(DataCollectionError) as exc_info:
            fetch_gold_data(start_date="2023-01-01", end_date="2023-01-03")
        assert "Network or API failure" in str(exc_info.value)


def test_fetch_gold_data_empty_response():
    with patch("yfinance.download", return_value=pd.DataFrame()):
        with pytest.raises(DataCollectionError) as exc_info:
            fetch_gold_data(start_date="2099-01-01", end_date="2099-01-02")
        assert "No data returned by Yahoo Finance" in str(exc_info.value)


def test_fetch_gold_data_missing_close():
    # DataFrame without Close price
    bad_df = pd.DataFrame({
        "Open": [1800.0],
        "High": [1810.0],
        "Low": [1790.0],
        "Volume": [1000],
    }, index=pd.date_range("2023-01-01", periods=1, name="Date"))

    with patch("yfinance.download", return_value=bad_df):
        with pytest.raises(DataCollectionError) as exc_info:
            fetch_gold_data()
        assert "essential 'Close' price column" in str(exc_info.value)
