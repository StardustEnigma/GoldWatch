"""
End-to-End Orchestration Pipeline for GoldWatch Data Collection & Preprocessing.

Coordinates fetching, raw storage, cleaning, validation, feature engineering,
and processed storage for downstream consumers (Person B and Person C).
"""

import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
import pandas as pd

from src.data_collection import fetch_gold_data, DataCollectionError
from src.preprocessing import (
    clean_market_data,
    calculate_returns_and_features,
    CleaningConfig,
    PreprocessingError,
)

logger = logging.getLogger(__name__)

# Default storage locations relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_RAW_PATH = DEFAULT_RAW_DIR / "gold_futures_raw.csv"
DEFAULT_PROCESSED_PATH = DEFAULT_PROCESSED_DIR / "gold_futures_processed.csv"


def save_dataset(df: pd.DataFrame, filepath: Union[str, Path]) -> Path:
    """
    Save DataFrame to CSV, ensuring parent directories exist and formatting Date explicitly.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to persist.
    filepath : Union[str, Path]
        Target file path.

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Successfully persisted dataset to: %s (%d records)", path, len(df))
    return path


def run_data_pipeline(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
    save_raw: bool = True,
    save_processed: bool = True,
    raw_path: Optional[Union[str, Path]] = None,
    processed_path: Optional[Union[str, Path]] = None,
    config: Optional[CleaningConfig] = None,
    return_summary: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Execute the complete GoldWatch data collection and preprocessing pipeline.

    Steps:
    1. Fetch historical gold futures data (GC=F) from Yahoo Finance.
    2. Normalize schema, MultiIndex columns, and date ordering.
    3. Save raw unadulterated dataset to CSV (if save_raw=True).
    4. Clean and validate market data (positive prices, strict OHLC bounds, no Close imputation).
    5. Calculate Daily_Return and supporting feature set.
    6. Perform quality checks and summarize results.
    7. Save processed dataset to CSV (if save_processed=True).
    8. Return processed DataFrame (and summary dictionary if requested).

    Parameters
    ----------
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format.
    end_date : str, optional
        End date in 'YYYY-MM-DD' format.
    interval : str, default '1d'
        Data frequency.
    save_raw : bool, default True
        Whether to export the raw downloaded data to CSV.
    save_processed : bool, default True
        Whether to export the cleaned processed data to CSV.
    raw_path : Union[str, Path], optional
        Custom path for raw CSV.
    processed_path : Union[str, Path], optional
        Custom path for processed CSV.
    config : CleaningConfig, optional
        Cleaning policy options.
    return_summary : bool, default False
        If True, returns a tuple (processed_df, summary_dict).

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]
        Processed DataFrame, or tuple of (processed_df, summary_dict).
    """
    target_raw_path = Path(raw_path) if raw_path else DEFAULT_RAW_PATH
    target_processed_path = Path(processed_path) if processed_path else DEFAULT_PROCESSED_PATH

    logger.info("=" * 60)
    logger.info("STARTING GOLDWATCH DATA PIPELINE (Person A)")
    logger.info("=" * 60)

    # 1. Fetch raw data
    raw_df = fetch_gold_data(
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        ticker="GC=F",
    )

    # 2. Persist raw dataset
    if save_raw:
        save_dataset(raw_df, target_raw_path)

    # 3. Clean and validate market data
    cleaned_df, cleaning_report = clean_market_data(raw_df, config=config)

    # 4. Feature engineering (Returns and Volatility Metrics)
    processed_df = calculate_returns_and_features(cleaned_df)

    # 5. Persist processed dataset
    if save_processed:
        save_dataset(processed_df, target_processed_path)

    # 6. Quality summary construction
    missing_counts = processed_df.isna().sum().to_dict()
    earliest_date = str(processed_df["Date"].min()) if not processed_df.empty else "N/A"
    latest_date = str(processed_df["Date"].max()) if not processed_df.empty else "N/A"

    summary: Dict[str, Any] = {
        "ticker": "GC=F",
        "interval": interval,
        "date_range_requested": {"start": start_date, "end": end_date},
        "raw_records_count": len(raw_df),
        "processed_records_count": len(processed_df),
        "rows_removed_during_cleaning": len(raw_df) - len(processed_df),
        "earliest_date": earliest_date,
        "latest_date": latest_date,
        "missing_values": missing_counts,
        "raw_file_location": str(target_raw_path) if save_raw else None,
        "processed_file_location": str(target_processed_path) if save_processed else None,
        "cleaning_report": cleaning_report,
    }

    # Attach summary to DataFrame attributes for convenient programmatic access
    processed_df.attrs["pipeline_summary"] = summary

    # Log concise summary
    logger.info("-" * 60)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("Downloaded Records   : %d", summary["raw_records_count"])
    logger.info("Cleaned Records      : %d", summary["processed_records_count"])
    logger.info("Rows Removed         : %d", summary["rows_removed_during_cleaning"])
    logger.info("Date Range           : %s to %s", earliest_date, latest_date)
    logger.info("Processed Columns    : %s", list(processed_df.columns))
    logger.info("=" * 60)

    if return_summary:
        return processed_df, summary
    return processed_df
