"""
Anomaly Detection Module for GoldWatch (Person B).

Workflow
--------
    processed CSV (from Person A)
            │
            ▼
    ALGORITHM 1: Rolling Z-score       -> Return_ZScore, Volume_ZScore
            │
            ▼
    ALGORITHM 2: Anomaly scoring       -> Anomaly_Score (0-100), Is_Anomaly, Event_Type
            │
            ▼
    data/processed/gold_futures_analysis.csv  (read by the dashboard and the backtest)
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANALYSIS_PATH = PROJECT_ROOT / "data" / "processed" / "gold_futures_analysis.csv"

DEFAULT_WINDOW = 30         # look-back window in trading sessions (~1.5 months)
DEFAULT_THRESHOLD = 2.5     # |Z| at or above this value counts as unusual
DEFAULT_VOLUME_WEIGHT = 0.5  # volume matters half as much as price in the score

# Yahoo's GC=F volume before 2020 is tiny (median ~100 contracts/day, many zeros);
# from 2020 it jumps to ~160,000/day, i.e. real exchange volume. Earlier volume is ignored.
VOLUME_RELIABLE_FROM = "2020-01-01"

EVENT_UP = "Upward spike"
EVENT_DOWN = "Downward shock"
EVENT_VOLUME = "Volume spike"
EVENT_NONE = "—"


# =============================================================================
# ALGORITHM 1: ROLLING Z-SCORE
# -----------------------------------------------------------------------------
# Question: "How unusual is today's value compared with the recent past?"
#
#   For each day t:
#     1. Take the previous `window` values  x[t-window] ... x[t-1]
#        (today is NOT included, so a spike cannot hide itself by inflating
#        its own mean/std, and we never use future data = no look-ahead bias)
#     2. mean_t = average of those values
#     3. std_t  = standard deviation of those values
#     4. Z_t    = (x_t - mean_t) / std_t
#
#   Z = 0  -> a perfectly normal day
#   Z = +3 -> today is 3 standard deviations ABOVE the recent average
#   Z = -3 -> today is 3 standard deviations BELOW the recent average
#
# Complexity: O(n) with pandas rolling windows (n = number of days).
# =============================================================================
def rolling_zscore(series: pd.Series, window: int = DEFAULT_WINDOW) -> pd.Series:
    """Causal rolling Z-score: compare each value to the previous `window` values."""
    past = series.shift(1)                                   # step 1: exclude today
    mean = past.rolling(window, min_periods=window).mean()   # step 2
    std = past.rolling(window, min_periods=window).std()     # step 3
    z = (series - mean) / std.replace(0, np.nan)             # step 4 (avoid divide by 0)
    return z.replace([np.inf, -np.inf], np.nan)


def compute_zscores(
    df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    volume_reliable_from: Optional[str] = VOLUME_RELIABLE_FROM,
) -> pd.DataFrame:
    """
    Apply Algorithm 1 to gold returns and trading volume.

    - Returns: used directly (Daily_Return from Person A).
    - Volume:  log-transformed first, because raw volume is heavily right-skewed
      (a few huge days) and Z-scores assume a roughly symmetric distribution.
      Zero volume and volume before `volume_reliable_from` are treated as
      "not reported" (NaN). Pass None to use all volume data.
    """
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").reset_index(drop=True)

    returns = pd.to_numeric(out["Daily_Return"], errors="coerce")
    out["Return_Rolling_Mean"] = returns.shift(1).rolling(window, min_periods=window).mean()
    out["Return_Rolling_Std"] = returns.shift(1).rolling(window, min_periods=window).std()
    out["Return_ZScore"] = rolling_zscore(returns, window)

    volume = pd.to_numeric(out.get("Volume", pd.Series(np.nan, index=out.index)), errors="coerce")
    usable = volume > 0
    if volume_reliable_from is not None:
        usable &= out["Date"] >= pd.Timestamp(volume_reliable_from)
    log_volume = np.log(volume.where(usable))
    out["Volume_ZScore"] = rolling_zscore(log_volume, window)

    return out


# =============================================================================
# ALGORITHM 2: ANOMALY SCORING & CLASSIFICATION
# -----------------------------------------------------------------------------
# Turns the two Z-scores into one easy-to-read score and a label.
#
#   Price can be unusual in BOTH directions (|Return_Z|), but for volume only
#   unusually HIGH trading matters (Volume_Z > 0). Unusually LOW volume in
#   GC=F is almost always a contract-roll day (end of month), not a market event.
#
#   Step 1 - Cap each Z-score so one extreme day cannot dominate:
#              return_comp = min(|Return_Z|, 2*threshold) / (2*threshold)       -> 0..1
#              volume_comp = min(max(Volume_Z, 0), 2*threshold) / (2*threshold) -> 0..1
#   Step 2 - Weighted average of the two components:
#              score = 100 * (return_comp + w * volume_comp) / (1 + w)
#            (if volume is missing that day, only the return component is used)
#   Step 3 - Flag the day if EITHER signal crosses the threshold:
#              Is_Anomaly = |Return_Z| >= threshold  OR  Volume_Z >= threshold
#   Step 4 - Label the event:
#              Return_Z >= +threshold            -> "Upward spike"
#              Return_Z <= -threshold            -> "Downward shock"
#              only volume crossed the threshold -> "Volume spike"
# =============================================================================
def score_anomalies(
    df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    volume_weight: float = DEFAULT_VOLUME_WEIGHT,
) -> pd.DataFrame:
    """Apply Algorithm 2 to a frame that already has Return_ZScore and Volume_ZScore."""
    out = df.copy()
    cap = 2 * threshold

    # Step 1: capped components in the range 0..1
    return_comp = out["Return_ZScore"].abs().clip(upper=cap) / cap
    volume_comp = out["Volume_ZScore"].clip(lower=0, upper=cap) / cap

    # Step 2: weighted average (volume weight drops out when volume is missing)
    has_volume = volume_comp.notna()
    numerator = return_comp + volume_weight * volume_comp.fillna(0)
    denominator = 1 + volume_weight * has_volume
    out["Anomaly_Score"] = 100 * numerator / denominator

    # Step 3: flag
    return_alert = out["Return_ZScore"].abs() >= threshold
    volume_alert = (out["Volume_ZScore"] >= threshold) & (volume_weight > 0)
    out["Is_Anomaly"] = return_alert | volume_alert

    # Step 4: label
    out["Event_Type"] = np.select(
        [
            out["Return_ZScore"] >= threshold,
            out["Return_ZScore"] <= -threshold,
            volume_alert & ~return_alert,
        ],
        [EVENT_UP, EVENT_DOWN, EVENT_VOLUME],
        default=EVENT_NONE,
    )
    return out


def detect_anomalies(
    df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
    volume_weight: float = DEFAULT_VOLUME_WEIGHT,
    volume_reliable_from: Optional[str] = VOLUME_RELIABLE_FROM,
) -> pd.DataFrame:
    """Full detection workflow: Algorithm 1 followed by Algorithm 2."""
    if df is None or df.empty:
        raise ValueError("Cannot detect anomalies on an empty dataset.")
    for col in ("Date", "Close", "Daily_Return"):
        if col not in df.columns:
            raise ValueError(f"Input dataset is missing required column '{col}'.")

    with_z = compute_zscores(df, window=window, volume_reliable_from=volume_reliable_from)
    scored = score_anomalies(with_z, threshold=threshold, volume_weight=volume_weight)

    logger.info(
        "Detected %d anomalous sessions out of %d (window=%d, threshold=%.2f).",
        int(scored["Is_Anomaly"].sum()), len(scored), window, threshold,
    )
    return scored


ANALYSIS_COLUMNS = [
    "Date",
    "Close",
    "Volume",
    "Daily_Return",
    "Return_Rolling_Mean",
    "Return_Rolling_Std",
    "Return_ZScore",
    "Volume_ZScore",
    "Anomaly_Score",
    "Is_Anomaly",
    "Event_Type",
]


def save_analysis(df: pd.DataFrame, path: Union[str, Path] = DEFAULT_ANALYSIS_PATH) -> Path:
    """Save the detection output in the format the dashboard expects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[[c for c in ANALYSIS_COLUMNS if c in df.columns]].copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
    logger.info("Saved anomaly analysis to %s (%d rows).", path, len(out))
    return path
