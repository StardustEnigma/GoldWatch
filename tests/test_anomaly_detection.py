"""
Unit tests for Person B: anomaly detection (Algorithms 1 & 2) and backtesting (Algorithms 3 & 4).
All tests use small synthetic datasets so the expected answers can be checked by hand.
"""

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection import (
    EVENT_DOWN,
    EVENT_NONE,
    EVENT_UP,
    EVENT_VOLUME,
    detect_anomalies,
    rolling_zscore,
    score_anomalies,
)
from src.backtest import (
    build_event_table,
    forward_returns,
    normal_cdf,
    proportion_z_test,
    run_backtest,
)


def make_market(returns, volume=None, start="2021-01-01"):
    """Build a minimal processed dataset from a list of daily returns."""
    returns = np.asarray(returns, dtype=float)
    close = 100 * np.cumprod(1 + np.nan_to_num(returns))
    return pd.DataFrame({
        "Date": pd.bdate_range(start, periods=len(returns)),
        "Close": close,
        "Daily_Return": returns,
        "Volume": volume if volume is not None else np.full(len(returns), 1000.0),
    })


def calm_returns(n, seed=0):
    return np.random.default_rng(seed).normal(0, 0.005, n)


# ---------------------------------------------------------------- Algorithm 1
def test_rolling_zscore_hand_calculation():
    s = pd.Series([1.0, 2.0, 3.0, 10.0])
    z = rolling_zscore(s, window=3)
    # previous 3 values = [1, 2, 3] -> mean 2, std 1 -> Z = (10 - 2) / 1 = 8
    assert z.iloc[3] == pytest.approx(8.0)
    assert z.iloc[:3].isna().all()


def test_rolling_zscore_excludes_today():
    s = pd.Series([0.0, 1.0] * 10 + [50.0])
    z = rolling_zscore(s, window=20)
    past = s.iloc[:-1]
    expected = (50.0 - past.mean()) / past.std()
    assert z.iloc[-1] == pytest.approx(expected)


def test_rolling_zscore_constant_window_is_nan_not_inf():
    z = rolling_zscore(pd.Series([5.0] * 10 + [6.0]), window=5)
    assert not np.isinf(z).any()
    assert np.isnan(z.iloc[-1])


def test_zero_and_old_volume_ignored():
    n = 40
    volume = np.full(n, 1000.0)
    volume[35] = 0.0
    df = make_market(calm_returns(n), volume, start="2019-12-01")
    out = detect_anomalies(df, window=5)
    assert np.isnan(out.loc[35, "Volume_ZScore"])
    assert out.loc[out["Date"] < "2020-01-01", "Volume_ZScore"].isna().all()


# ---------------------------------------------------------------- Algorithm 2
def test_detects_upward_and_downward_spikes():
    r = calm_returns(80)
    r[50], r[70] = 0.06, -0.06
    out = detect_anomalies(make_market(r), window=30, threshold=2.5)
    assert out.loc[50, "Event_Type"] == EVENT_UP
    assert out.loc[70, "Event_Type"] == EVENT_DOWN
    assert out.loc[50, "Is_Anomaly"] and out.loc[70, "Is_Anomaly"]


def test_only_high_volume_counts():
    df = pd.DataFrame({"Return_ZScore": [0.0, 0.0, 0.0], "Volume_ZScore": [4.0, -6.0, np.nan]})
    out = score_anomalies(df, threshold=2.5, volume_weight=0.5)
    assert out["Event_Type"].tolist() == [EVENT_VOLUME, EVENT_NONE, EVENT_NONE]
    assert out["Is_Anomaly"].tolist() == [True, False, False]


def test_anomaly_score_formula():
    df = pd.DataFrame({"Return_ZScore": [2.5, 10.0, 2.5], "Volume_ZScore": [0.0, 10.0, np.nan]})
    out = score_anomalies(df, threshold=2.5, volume_weight=0.5)
    # row 0: return_comp = 2.5/5 = 0.5, volume_comp = 0 -> 100 * 0.5 / 1.5
    assert out.loc[0, "Anomaly_Score"] == pytest.approx(100 * 0.5 / 1.5)
    # row 1: both capped at 1 -> 100
    assert out.loc[1, "Anomaly_Score"] == pytest.approx(100.0)
    # row 2: no volume -> score uses return only -> 50
    assert out.loc[2, "Anomaly_Score"] == pytest.approx(50.0)


def test_missing_required_column_raises():
    with pytest.raises(ValueError):
        detect_anomalies(pd.DataFrame({"Date": ["2021-01-01"], "Close": [1.0]}))


# ---------------------------------------------------------------- Algorithm 3
def test_forward_returns():
    fr = forward_returns(pd.Series([100.0, 110.0, 99.0]), horizon=1)
    assert fr.iloc[0] == pytest.approx(0.10)
    assert fr.iloc[1] == pytest.approx(-0.10)
    assert np.isnan(fr.iloc[2])


def test_event_table_measures_drop_after_spike():
    r = calm_returns(60)
    r[40] = 0.06        # spike
    r[41:46] = -0.01    # followed by a week of declines
    analysis = detect_anomalies(make_market(r), window=30)
    events = build_event_table(analysis, horizon=5)

    assert len(events) == 1
    ev = events.iloc[0]
    assert ev["Signal_Date"] == analysis.loc[40, "Date"]
    assert ev["Forward_Return"] == pytest.approx(0.99 ** 5 - 1)
    assert ev["Declined"] == True  # noqa: E712 (stored as object to allow NaN)


def test_event_at_end_has_no_forward_return():
    r = calm_returns(40)
    r[-1] = 0.08
    events = build_event_table(detect_anomalies(make_market(r), window=30), horizon=5)
    assert len(events) == 1 and np.isnan(events.loc[0, "Forward_Return"])


def test_backtest_output_matches_dashboard_contract():
    r = calm_returns(100)
    r[50] = 0.06
    analysis = detect_anomalies(make_market(r), window=30)
    events = build_event_table(analysis, horizon=5)
    for col in ("Date", "Signal_Date", "Forward_Return", "Evaluation_Horizon"):
        assert col in events.columns
    # Must not contain analysis columns, or the dashboard would mistake it for the analysis file
    assert not {"Return_ZScore", "Volume_ZScore", "Anomaly_Score", "Is_Anomaly"} & set(events.columns)

    summary = run_backtest(analysis, horizons=[1, 5])
    assert summary["Horizon_Days"].tolist() == [1, 5]
    assert (summary["Events"] == 1).all()


# ---------------------------------------------------------------- Algorithm 4
def test_normal_cdf():
    assert normal_cdf(0) == pytest.approx(0.5)
    assert normal_cdf(1.96) == pytest.approx(0.975, abs=1e-3)


def test_proportion_z_test():
    # 70 declines out of 100 vs. baseline 50% -> z = 0.2 / sqrt(0.25/100) = 4
    z, p = proportion_z_test(70, 100, 0.5)
    assert z == pytest.approx(4.0)
    assert p < 0.001
    # exactly at baseline -> z = 0, p = 0.5
    z, p = proportion_z_test(50, 100, 0.5)
    assert z == pytest.approx(0.0) and p == pytest.approx(0.5)
    # no events -> undefined
    assert all(np.isnan(v) for v in proportion_z_test(0, 0, 0.5))
