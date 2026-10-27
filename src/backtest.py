"""
Backtesting Module for GoldWatch (Person B).

Research question
-----------------
    "When gold has an unusual UPWARD price spike, does the price usually
     FALL over the following days?"  (a gold version of pump-and-dump)

Workflow
--------
    anomaly output (Algorithms 1 & 2)
            │
            ▼
    ALGORITHM 3: Event-study backtest   -> forward return after every spike
            │
            ▼
    ALGORITHM 4: Proportion Z-test      -> is the decline rate better than chance?
            │
            ▼
    data/processed/gold_futures_backtest.csv          (one row per spike, for the dashboard)
    data/processed/gold_futures_backtest_summary.csv  (one row per horizon)
"""

import logging
import math
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd

from src.anomaly_detection import EVENT_UP

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BACKTEST_PATH = PROJECT_ROOT / "data" / "processed" / "gold_futures_backtest.csv"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "gold_futures_backtest_summary.csv"

DEFAULT_HORIZON = 5                  # main horizon: one trading week
DEFAULT_HORIZONS = (1, 5, 10, 20)    # day, week, two weeks, month
SIGNIFICANCE_LEVEL = 0.05


def forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    """
    Return from the close of day t to the close of day t+horizon:
        Forward_Return_t = Close[t + horizon] / Close[t] - 1
    The last `horizon` days have no future price yet, so they stay NaN.
    """
    return close.shift(-horizon) / close - 1


# =============================================================================
# ALGORITHM 3: EVENT-STUDY BACKTEST
# -----------------------------------------------------------------------------
#   Input: days labelled "Upward spike" by Algorithm 2.
#
#   For each spike day t:
#     1. Entry price  = Close[t]               (price at the end of the spike day)
#     2. Exit price   = Close[t + h]           (h trading days later)
#     3. Forward return = Exit / Entry - 1
#     4. Declined     = Forward return < 0     (price went down = "dump")
#
#   Then:
#     Decline rate (spikes)   = (# spikes that declined) / (# spikes)
#     Baseline decline rate   = same calculation over ALL days
#                               (how often gold falls over h days anyway)
#     Lift                    = Decline rate (spikes) - Baseline
#
#   Lift > 0 means spikes are followed by drops MORE often than a random day.
#   Comparing against a baseline matters: if gold falls 47% of the time anyway,
#   a 50% decline rate after spikes is not impressive.
# =============================================================================
def build_event_table(analysis: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    """One row per upward spike with its realised forward return (steps 1-4)."""
    df = analysis.sort_values("Date").reset_index(drop=True)
    close = pd.to_numeric(df["Close"], errors="coerce")
    exit_close = close.shift(-horizon)

    is_spike = df["Event_Type"] == EVENT_UP
    events = pd.DataFrame({
        "Date": pd.to_datetime(df["Date"]),
        "Signal_Date": pd.to_datetime(df["Date"]),
        "Entry_Close": close,                                   # step 1
        "Exit_Close": exit_close,                               # step 2
        "Spike_Return": df["Daily_Return"],
        "Spike_ZScore": df["Return_ZScore"],
        "Spike_Volume_ZScore": df["Volume_ZScore"],
        "Forward_Return": exit_close / close - 1,               # step 3
        "Evaluation_Horizon": horizon,
    })[is_spike].reset_index(drop=True)

    # step 4 (left empty when the future price is not known yet)
    events["Declined"] = (events["Forward_Return"] < 0).where(events["Forward_Return"].notna())
    return events


# =============================================================================
# ALGORITHM 4: ONE-PROPORTION Z-TEST
# -----------------------------------------------------------------------------
#   Null hypothesis H0: spikes are followed by a decline no more often than
#                       a normal day          (p_spike = p_baseline)
#   Alternative H1:     spikes are followed by a decline MORE often
#                       (p_spike > p_baseline)
#
#   z = (p_spike - p_baseline) / sqrt( p_baseline * (1 - p_baseline) / n )
#   p-value = P(Z >= z) = 1 - Phi(z)     where Phi = standard normal CDF
#
#   If p-value < 0.05 we reject H0: the "spike -> drop" pattern is
#   statistically significant. Otherwise the pattern could be due to luck.
# =============================================================================
def normal_cdf(z: float) -> float:
    """Standard normal cumulative distribution function, Phi(z)."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def proportion_z_test(successes: int, n: int, p0: float) -> tuple:
    """One-sided test of H1: true proportion > p0. Returns (z, p_value)."""
    if n == 0 or p0 <= 0 or p0 >= 1:
        return np.nan, np.nan
    p_hat = successes / n
    z = (p_hat - p0) / math.sqrt(p0 * (1 - p0) / n)
    return z, 1 - normal_cdf(z)


def summarize_events(
    events: pd.DataFrame,
    baseline_forward: pd.Series,
    horizon: int,
) -> dict:
    """Decline rate vs. baseline (Algorithm 3) plus significance (Algorithm 4)."""
    completed = events.dropna(subset=["Forward_Return"])
    baseline_forward = baseline_forward.dropna()

    n = len(completed)
    declines = int((completed["Forward_Return"] < 0).sum())
    decline_rate = declines / n if n else np.nan
    baseline_rate = float((baseline_forward < 0).mean()) if len(baseline_forward) else np.nan
    z, p_value = proportion_z_test(declines, n, baseline_rate)

    return {
        "Horizon_Days": horizon,
        "Events": n,
        "Declines": declines,
        "Decline_Rate": decline_rate,
        "Baseline_Decline_Rate": baseline_rate,
        "Lift": decline_rate - baseline_rate,
        "Avg_Forward_Return": completed["Forward_Return"].mean() if n else np.nan,
        "Baseline_Avg_Forward_Return": baseline_forward.mean(),
        "Z_Stat": z,
        "P_Value": p_value,
        "Significant": bool(p_value < SIGNIFICANCE_LEVEL) if not np.isnan(p_value) else False,
    }


def run_backtest(
    analysis: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Run Algorithms 3 and 4 for every horizon; one summary row per horizon."""
    df = analysis.sort_values("Date").reset_index(drop=True)
    close = pd.to_numeric(df["Close"], errors="coerce")
    # Baseline only uses days where the Z-score exists, so both sides cover the same period.
    has_signal = df["Return_ZScore"].notna()

    rows = []
    for h in horizons:
        events = build_event_table(df, horizon=h)
        baseline = forward_returns(close, h)[has_signal]
        rows.append(summarize_events(events, baseline, h))
    return pd.DataFrame(rows)


def save_backtest(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    events_path: Union[str, Path] = DEFAULT_BACKTEST_PATH,
    summary_path: Union[str, Path] = DEFAULT_SUMMARY_PATH,
) -> None:
    """Persist the per-event table (for the dashboard) and the summary table."""
    events_path, summary_path = Path(events_path), Path(summary_path)
    events_path.parent.mkdir(parents=True, exist_ok=True)

    out = events.copy()
    for col in ("Date", "Signal_Date"):
        out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")
    out.to_csv(events_path, index=False)
    summary.to_csv(summary_path, index=False)
    logger.info("Saved %d backtest events to %s and summary to %s.", len(out), events_path, summary_path)
