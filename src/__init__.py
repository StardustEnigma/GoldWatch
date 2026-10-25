"""
GoldWatch - Market Anomaly Detection
Package for data collection, preprocessing, and market data analysis.
"""

from src.data_collection import fetch_gold_data
from src.preprocessing import clean_market_data, calculate_returns_and_features
from src.pipeline import run_data_pipeline

__all__ = [
    "fetch_gold_data",
    "clean_market_data",
    "calculate_returns_and_features",
    "run_data_pipeline",
]
