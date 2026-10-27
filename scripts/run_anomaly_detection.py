#!/usr/bin/env python
"""
Command-Line Entry Point for GoldWatch Anomaly Detection & Backtesting (Person B).

Usage examples:
    python scripts/run_anomaly_detection.py
    python scripts/run_anomaly_detection.py --window 60 --threshold 3
    python scripts/run_anomaly_detection.py --start 2015-01-01 --horizon 10 --top 15
"""

import sys
import argparse
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import DEFAULT_PROCESSED_PATH
from src.anomaly_detection import (
    DEFAULT_THRESHOLD,
    DEFAULT_VOLUME_WEIGHT,
    DEFAULT_WINDOW,
    detect_anomalies,
    save_analysis,
)
from src.backtest import (
    DEFAULT_HORIZON,
    DEFAULT_HORIZONS,
    build_event_table,
    run_backtest,
    save_backtest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GoldWatch (Person B): rolling Z-score anomaly detection and spike-reversal backtest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, default=str(DEFAULT_PROCESSED_PATH),
                        help="Processed CSV produced by Person A's pipeline.")
    parser.add_argument("--start", type=str, default=None, help="Only analyse data from this date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="Only analyse data up to this date (YYYY-MM-DD).")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Rolling window in trading sessions.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="|Z| needed to flag an anomaly.")
    parser.add_argument("--volume-weight", type=float, default=DEFAULT_VOLUME_WEIGHT,
                        help="Weight of the volume Z-score in the anomaly score (0 = ignore volume).")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                        help="Days after a spike used for the per-event backtest file shown in the dashboard.")
    parser.add_argument("--top", type=int, default=10, help="Number of top anomalies to print.")
    parser.add_argument("--no-save", action="store_true", help="Print results without writing CSV files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    try:
        # STEP 1: load Person A's processed data
        df = pd.read_csv(args.input, parse_dates=["Date"])
        if args.start:
            df = df[df["Date"] >= args.start]
        if args.end:
            df = df[df["Date"] <= args.end]

        # STEP 2: Algorithm 1 (rolling Z-score) + Algorithm 2 (scoring)
        analysis = detect_anomalies(df, window=args.window, threshold=args.threshold,
                                    volume_weight=args.volume_weight)

        # STEP 3: Algorithm 3 (event study) + Algorithm 4 (significance test)
        horizons = sorted(set(DEFAULT_HORIZONS) | {args.horizon})
        summary = run_backtest(analysis, horizons=horizons)
        events = build_event_table(analysis, horizon=args.horizon)

        # STEP 4: save outputs for the dashboard
        if not args.no_save:
            save_analysis(analysis)
            save_backtest(events, summary)

        # STEP 5: report
        flagged = analysis[analysis["Is_Anomaly"]]
        print("\n" + "=" * 78)
        print("            GOLDWATCH ANOMALY DETECTION & BACKTEST REPORT")
        print("=" * 78)
        print(f"Period analysed     : {analysis['Date'].min():%Y-%m-%d} to {analysis['Date'].max():%Y-%m-%d}")
        print(f"Sessions analysed   : {len(analysis):,}")
        print(f"Settings            : window={args.window}, threshold=|Z|>={args.threshold}, "
              f"volume weight={args.volume_weight}")
        print(f"Anomalous sessions  : {len(flagged):,} ({len(flagged) / len(analysis):.1%})")
        for event_type, count in flagged["Event_Type"].value_counts().items():
            print(f"  - {event_type:<20}: {count:,}")

        print(f"\nTop {args.top} anomalies by score:")
        top = (
            flagged.assign(_abs_z=flagged["Return_ZScore"].abs())
            .sort_values(["Anomaly_Score", "_abs_z"], ascending=False)
            .head(args.top)
            [["Date", "Close", "Daily_Return", "Return_ZScore", "Volume_ZScore", "Anomaly_Score", "Event_Type"]]
        )
        top["Date"] = top["Date"].dt.strftime("%Y-%m-%d")
        top["Daily_Return"] = top["Daily_Return"].map(lambda v: f"{v:+.2%}")
        print(top.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

        print("\nBacktest: are upward price spikes followed by a decline?")
        table = summary.copy()
        for col in ("Decline_Rate", "Baseline_Decline_Rate", "Lift"):
            table[col] = table[col].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
        for col in ("Avg_Forward_Return", "Baseline_Avg_Forward_Return"):
            table[col] = table[col].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "—")
        table = table.rename(columns={"Baseline_Decline_Rate": "Baseline_Rate",
                                      "Avg_Forward_Return": "Avg_Fwd_Ret",
                                      "Baseline_Avg_Forward_Return": "Base_Fwd_Ret"})
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\nRead it as: Lift > 0 and P_Value < 0.05 => spikes are followed by drops more often than chance.")
        print("=" * 78 + "\n")
        return 0

    except Exception as exc:
        logging.error("Anomaly detection failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
