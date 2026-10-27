# Person B — Anomaly Detection & Backtesting: Algorithms and Workflow

**Question:** When gold has an unusual upward price spike, does the price usually fall afterwards (a "pump-and-dump" pattern)?

**Code:**

| File | Contents |
|---|---|
| `src/anomaly_detection.py` | Algorithm 1 (rolling Z-score) and Algorithm 2 (anomaly scoring) |
| `src/backtest.py` | Algorithm 3 (event-study backtest) and Algorithm 4 (proportion Z-test) |
| `scripts/run_anomaly_detection.py` | Command-line script that runs all four algorithms in order |
| `tests/test_anomaly_detection.py` | Unit tests with hand-checkable examples |

## Workflow

```text
data/processed/gold_futures_processed.csv        (Person A: Date, Close, Volume, Daily_Return)
        │
        ▼
ALGORITHM 1  Rolling Z-score      → Return_ZScore, Volume_ZScore
        │
        ▼
ALGORITHM 2  Anomaly scoring      → Anomaly_Score (0–100), Is_Anomaly, Event_Type
        │                                  └─► gold_futures_analysis.csv  ─► Dashboard (Person C)
        ▼
ALGORITHM 3  Event-study backtest → forward return after every upward spike
        │                                  └─► gold_futures_backtest.csv  ─► Dashboard (Person C)
        ▼
ALGORITHM 4  Proportion Z-test    → is the decline rate better than chance?
                                           └─► gold_futures_backtest_summary.csv
```

Run it:

```bash
python scripts/run_anomaly_detection.py                       # defaults: window=30, threshold=2.5, horizon=5
python scripts/run_anomaly_detection.py --window 60 --threshold 3
python scripts/run_anomaly_detection.py --start 2020-01-01 --top 15
```

---

## Algorithm 1 — Rolling Z-score

Measures how unusual today is compared with the **previous** `window` days.

```text
for each day t:
    past   = x[t-window], ..., x[t-1]       # today is excluded
    mean_t = mean(past)
    std_t  = std(past)
    Z_t    = (x_t - mean_t) / std_t
```

- **Worked example:** if the previous returns are `[1%, 2%, 3%]` and today's return is `10%`, then mean = 2%, std = 1%, so Z = (10 − 2) / 1 = **8**. Today is 8 standard deviations above normal.
- **Why exclude today:** if today's spike were inside its own window, it would raise the mean and std and make itself look less extreme. Excluding it also means we only use information available on day t, so there is no look-ahead bias.
- **Why "rolling":** gold's volatility changes a lot over time (2008 vs. 2017). A fixed historical mean and std would flag every day of a volatile year. A rolling window adapts to recent conditions.
- **Applied to:**
  - `Daily_Return`, used as-is.
  - `log(Volume)`. Volume is right-skewed, and the log makes it closer to symmetric, which Z-scores assume. Zero volume and all volume before 2020 are ignored: before 2020, Yahoo reports about 100 contracts/day for `GC=F`; from 2020 it reports about 160,000/day, which is real exchange volume.
- **Complexity:** O(n) using pandas rolling windows.

## Algorithm 2 — Anomaly scoring and classification

Combines the two Z-scores into one 0–100 score and a label.

```text
cap         = 2 × threshold
return_comp = min(|Return_Z|, cap) / cap                 # 0..1, both directions count
volume_comp = min(max(Volume_Z, 0), cap) / cap           # 0..1, only HIGH volume counts
score       = 100 × (return_comp + w × volume_comp) / (1 + w)    # w = 0.5 by default
              (if volume is missing, score = 100 × return_comp)

Is_Anomaly  = |Return_Z| ≥ threshold  OR  Volume_Z ≥ threshold

Event_Type  = "Upward spike"    if Return_Z ≥ +threshold
              "Downward shock"  if Return_Z ≤ −threshold
              "Volume spike"    if only volume crossed the threshold
```

- **Capping:** stops one extreme day (e.g. Z = 12) from dominating the scale.
- **Only high volume counts:** in `GC=F`, unusually *low* volume almost always falls on a contract-roll day at month-end, when Yahoo switches to the next futures contract. That is a data artifact, not a market event.

## Algorithm 3 — Event-study backtest

Tests what happens after each upward spike.

```text
for each day t labelled "Upward spike":
    entry          = Close[t]
    exit           = Close[t + h]               # h = 1, 5, 10, 20 trading days
    forward_return = exit / entry − 1
    declined       = forward_return < 0

decline_rate  = (# spikes that declined) / (# spikes)
baseline_rate = same calculation over ALL days      # how often gold falls anyway
lift          = decline_rate − baseline_rate
```

- **Why a baseline:** if gold falls over 5 days 44% of the time anyway, a 45% decline rate after spikes means nothing. Only the **lift** over the baseline shows a real effect.
- Spikes in the last `h` days have no future price yet, so they are left out of the statistics.

## Algorithm 4 — One-proportion Z-test

Checks whether the lift is statistically significant or could be luck.

```text
H0: p_spike = p_baseline    (spikes are followed by drops no more often than a normal day)
H1: p_spike > p_baseline    (spikes are followed by drops more often)

z       = (p_spike − p_baseline) / sqrt(p_baseline × (1 − p_baseline) / n)
p-value = 1 − Φ(z)           # Φ = standard normal CDF
reject H0 if p-value < 0.05
```

- **Worked example:** 70 declines out of 100 spikes, baseline 50% → z = 0.20 / √(0.25/100) = **4.0**, p < 0.001, so the result is significant.

---

## Results (GC=F, 2000-08-30 → 2026-10-23, window = 30, threshold = 2.5)

- **Anomalous sessions:** 239 of 6,122 (3.9%)
  - 101 upward spikes
  - 118 downward shocks
  - 20 volume spikes
- **Real events among the top anomalies:**
  - 2001-09-14: after 9/11
  - 2013-04-15: the April 2013 gold crash
  - 2016-06-24: the Brexit vote
  - 2019-06-20: the Fed's dovish turn

| Horizon | Spikes | Decline rate | Baseline | Lift | p-value | Significant |
|---|---|---|---|---|---|---|
| 1 day | 101 | 43.6% | 46.5% | −3.0% | 0.725 | No |
| 5 days | 101 | 44.6% | 44.3% | +0.3% | 0.478 | No |
| 10 days | 101 | 47.5% | 44.2% | +3.4% | 0.249 | No |
| 20 days | 101 | 54.5% | 43.0% | +11.4% | 0.010 | Yes |

**Interpretation:** in the short term (1–10 days), gold does **not** dump after a spike; it behaves like any other day. Over about a month, spikes are followed by declines noticeably more often than usual (54.5% vs. 43.0%), which points to a slow partial reversal rather than a crypto-style pump-and-dump. That fits gold being a large, liquid market that is hard to manipulate.

## Limitations

1. **Overlapping events:** spikes often cluster (e.g. autumn 2008). Their 20-day windows overlap, so the events are not fully independent, which makes the p-value somewhat optimistic.
2. **Multiple testing:** four horizons are tested. With a Bonferroni correction (0.05 / 4 = 0.0125), the 20-day result (p = 0.010) is still just significant.
3. **Data gaps:** Person A's cleaning drops some invalid rows, so a few `Daily_Return` values span more than one day (e.g. 2008-09-18 covers 12–18 Sept 2008).
4. **Volume:** only usable from 2020, so volume has no influence on the score for earlier years.
5. **Not a trading strategy:** no transaction costs, slippage, or position sizing are modelled.
