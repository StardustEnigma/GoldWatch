#!/usr/bin/env python
"""
Command-Line Entry Point for GoldWatch Data Collection & Preprocessing.

Usage examples:
    python scripts/run_data_pipeline.py
    python scripts/run_data_pipeline.py --start 2015-01-01 --end 2025-12-31
    python scripts/run_data_pipeline.py --start 2020-01-01 --interval 1d --output-dir data/custom
"""

import sys
import argparse
import logging
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_data_pipeline


def configure_logging(verbose: bool = False) -> None:
    """Configure stdout logging format and verbosity level."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GoldWatch (Person A): Historical Gold Futures (GC=F) Data Collection & Preprocessing Pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date in 'YYYY-MM-DD' format. If omitted, downloads maximum historical range.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date in 'YYYY-MM-DD' format. If omitted, downloads up to current available date.",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Data collection frequency (e.g. '1d' for daily historical data).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom root data directory. By default saves to project data/ directory.",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Skip saving the unadulterated raw dataset.",
    )
    parser.add_argument(
        "--no-processed",
        action="store_true",
        help="Skip saving the processed dataset.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed debug logging.",
    )
    return parser.parse_args()


def main() -> int:
    """Execute pipeline with parsed arguments."""
    args = parse_args()
    configure_logging(verbose=args.verbose)

    raw_path = None
    processed_path = None
    if args.output_dir:
        base_dir = Path(args.output_dir)
        raw_path = base_dir / "raw" / "gold_futures_raw.csv"
        processed_path = base_dir / "processed" / "gold_futures_processed.csv"

    try:
        df, summary = run_data_pipeline(
            start_date=args.start,
            end_date=args.end,
            interval=args.interval,
            save_raw=not args.no_raw,
            save_processed=not args.no_processed,
            raw_path=raw_path,
            processed_path=processed_path,
            return_summary=True,
        )

        print("\n" + "=" * 65)
        print("          GOLDWATCH DATA PIPELINE EXECUTION REPORT          ")
        print("=" * 65)
        print(f"Ticker                  : {summary['ticker']}")
        print(f"Interval                : {summary['interval']}")
        print(f"Date Range Downloaded   : {summary['earliest_date']} to {summary['latest_date']}")
        print(f"Raw Records Fetched     : {summary['raw_records_count']:,}")
        print(f"Cleaned Records Output  : {summary['processed_records_count']:,}")
        print(f"Rows Removed (Cleaning) : {summary['rows_removed_during_cleaning']:,}")
        if summary["raw_file_location"]:
            print(f"Raw CSV Saved To        : {summary['raw_file_location']}")
        if summary["processed_file_location"]:
            print(f"Processed CSV Saved To  : {summary['processed_file_location']}")

        print("\nMissing Value Summary in Processed Dataset:")
        for col, count in summary["missing_values"].items():
            print(f"  - {col:<22}: {count} missing")

        print("\nFirst 3 rows of processed dataset:")
        print(df.head(3).to_string(index=False))
        print("\nLast 3 rows of processed dataset:")
        print(df.tail(3).to_string(index=False))
        print("=" * 65)
        print("Pipeline execution completed successfully.\n")
        return 0

    except Exception as exc:
        logging.error("Pipeline execution failed: %s", exc, exc_info=args.verbose)
        print(f"\n[ERROR] Pipeline run failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
