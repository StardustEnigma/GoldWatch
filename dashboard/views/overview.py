import numpy as np
import streamlit as st

import core

df = core.analysis()
core.page_header(
    "Overview",
    "<b>Spot unusual moves in gold prices.</b> GoldWatch checks every trading day since 2000 and flags the "
    "days that moved far more than usual.",
)

last, prev = df.iloc[-1], df.iloc[-2]
change = last["Close"] - prev["Close"]
arrow = "▲" if change >= 0 else "▼"
flagged_all = df[df["Is_Anomaly"]]
latest = flagged_all.iloc[-1]

main, side = st.columns([2, 1])
with main:
    with core.panel("quote"):
        st.html(
            f'<div class="gw-quote__name">Gold futures <span>COMEX GC=F · USD per troy ounce</span></div>'
            f'<div class="gw-quote__line">'
            f'<span class="gw-quote__price">{core.money(last["Close"])}</span>'
            f'<span class="gw-chg {core.tone(change)}">{arrow} {abs(change):,.2f} '
            f'({abs(change / prev["Close"]) * 100:.2f}%)</span>'
            f'<span class="gw-quote__asof">Closing price on {core.day(last["Date"])}</span></div>'
        )
        controls, toggle = st.columns([3, 2], vertical_alignment="center")
        with controls:
            choice = st.segmented_control("Time range", core.RANGES, default="1Y", required=True,
                                          key="overview_range", label_visibility="collapsed",
                                          persist_state="session")
        with toggle:
            with st.container(horizontal=True, horizontal_alignment="right"):
                show_anomalies = st.toggle("Show flagged days", value=True, key="overview_anomalies",
                                           persist_state="session")
        period = df[df["Date"] >= core.range_start(df["Date"], choice)]
        core.show(core.price_chart(period, show_anomalies, height=400),
                  alt=f"Gold futures closing price over the last {choice}, with flagged days marked")
        core.how_to_read("the line is the closing price, green if it rose over the chosen range and red if it "
                         "fell. Triangles and diamonds mark flagged days. Grey bars show trading volume.")

year = df[df["Date"] > df["Date"].max() - np.timedelta64(365, "D")]
range_return = period["Close"].iloc[-1] / period["Close"].iloc[0] - 1
volatility = df["Daily_Return"].tail(30).std() * np.sqrt(252)

with side:
    with core.panel("stats"):
        core.section("Key statistics", f"Range figures use the {choice} view.")
        core.stat_grid([
            ("Previous close", core.money(prev["Close"])),
            ("Open", core.money(last["Open"])),
            ("Day range", f'{core.money(last["Low"])} – {core.money(last["High"])}'),
            ("52-week range", f'{core.money(year["Low"].min())} – {core.money(year["High"].max())}'),
            ("Volume (contracts)", f'{last["Volume"]:,.0f}'),
            ("30-day volatility, yearly", core.pct(volatility, sign=False)),
            (f"Change over {choice}", core.pct(range_return), core.tone(range_return)),
            (f"Flagged days in {choice}", f"{period['Is_Anomaly'].sum():,}"),
        ], columns=1)
    with core.panel("latest"):
        st.html(f'<p class="gw-sub">Latest flagged day</p>'
                f'<h2 class="gw-h2">{core.day(latest["Date"])}</h2>'
                f'<p class="gw-event" style="color:{core.EVENT_COLOR[latest["Event_Type"]]}">'
                f'{core.EVENT_LABEL[latest["Event_Type"]]}</p>'
                f'<p class="gw-card-text">{core.explain_day(latest, *core.settings())}</p>')
        st.page_link(core.PAGES["anomalies"][0], label="See every flagged day", icon=":material/arrow_forward:",
                     icon_position="right")

st.space("small")
core.section("Explore")
s5 = core.backtest().loc[5]
cards = st.columns(4)
with cards[0]:
    core.explore_card("anomalies", core.count(len(flagged_all)), f"unusual days since {df['Date'].dt.year.min()}",
                      f"{core.EVENT_LABEL[core.EVENT_UP][2:]}s, downward shocks and volume spikes.",
                      f"See all {len(flagged_all):,} anomalies")
with cards[1]:
    core.explore_card("backtest", core.count(s5.Decline_Rate * 100, suffix="%"),
                      "of sharp rises fell within a week",
                      f"Any day: {s5.Baseline_Decline_Rate:.0%}. "
                      + ("Falls were more common than usual." if s5.Significant
                         else "No clear reversal within a week."), "Open the backtest")
with cards[2]:
    core.explore_card("data", core.count(len(df)), "trading days of data",
                      "Prices, scores and test results to filter and download.", "Browse the data")
with cards[3]:
    core.explore_card("method", core.count(4), "steps from prices to results",
                      "With a real worked example and a glossary.", "Learn how it works")
