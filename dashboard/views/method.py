import streamlit as st

import core

core.page_header("How it works", "From raw prices to a tested result in four steps.")

df = core.analysis()
window, threshold = core.settings()
summary = core.backtest()
flagged = int(df["Is_Anomaly"].sum())

steps = [
    ("Collect", f"Download daily gold futures prices from Yahoo Finance and clean them. "
                f"{len(df):,} trading days remain."),
    ("Measure", f"For each day, compare its price change with the previous {window} days using a rolling "
                "Z-score."),
    ("Flag", f"Days more than {threshold:g} typical moves from normal are flagged and labelled. "
             f"{flagged:,} days qualify."),
    ("Test", "For every sharp rise, check whether the price was lower days later, and compare with any day."),
]
st.html('<ol class="gw-steps">' + "".join(
    f'<li class="gw-step"><span class="gw-step__n">{i}</span><h3 class="gw-h3">{title}</h3>'
    f'<p>{text}</p></li>' for i, (title, text) in enumerate(steps, 1)) + "</ol>")

example = "2016-06-24"
row = df[df["Date"] == example].iloc[0]
with core.panel("example"):
    core.section("Worked example: the Brexit vote, 24 June 2016",
                 "Steps 2 and 3 for a single day, using the current settings")
    left, right = st.columns([3, 2], gap="large")
    with left:
        core.show(core.window_chart(df, example),
                  alt=f"The {window} daily returns before 24 June 2016 and the return on that day")
        core.how_to_read(f"grey bars are the {window} days before the vote. The shaded band is the normal "
                         "range. The gold bar is the day being tested.")
    with right:
        mu, sd, ret, z = (row["Return_Rolling_Mean"] * 100, row["Return_Rolling_Std"] * 100,
                          row["Daily_Return"] * 100, row["Return_ZScore"])
        core.stat_grid([
            ("Average daily change, μ", f"{mu:+.2f}%"),
            ("Typical move, σ", f"±{sd:.2f}%"),
            ("Change on 24 June", f"{ret:+.2f}%", "up"),
            ("Z = (change − μ) ÷ σ", f"{z:+.1f}"),
            ("Threshold", f"±{threshold:g}"),
            ("Result", "Flagged as an upward spike" if z >= threshold else "Not flagged"),
        ], columns=1)

with core.panel("test"):
    core.section("How the test works", "Step 4 in plain language")
    s20 = summary.loc[20]
    st.html(
        '<ol class="gw-list">'
        "<li>Take every upward spike and note the closing price that day.</li>"
        "<li>Look at the closing price a set number of trading days later: 1, 5, 10 or 20.</li>"
        "<li>Count how often the price was lower. Do the same starting from every day, to get the baseline.</li>"
        "<li>Use a Z-test to check whether the difference could be chance. Below p = 0.05 counts as real.</li>"
        "</ol>"
        f'<p class="gw-explain">Result so far: after 20 trading days, gold was lower after '
        f"<b>{s20.Decline_Rate:.1%}</b> of spikes against <b>{s20.Baseline_Decline_Rate:.1%}</b> for any day "
        f"(p = {s20.P_Value:.3f}).</p>"
    )
    st.page_link(core.PAGES["backtest"][0], label="Open the backtest", icon=":material/arrow_forward:",
                 icon_position="right")

left, right = st.columns(2)
with left:
    with core.panel("glossary"):
        core.section("What the terms mean")
        core.glossary()
with right:
    with core.panel("limits"):
        core.section("Limits of this analysis")
        st.html(
            '<ul class="gw-list">'
            "<li>Spikes often come in clusters, so their outcomes overlap. The p-value is a little optimistic.</li>"
            "<li>Four holding periods are tested, which raises the chance of a lucky result.</li>"
            "<li>Yahoo Finance volume is only reliable from 2020, so earlier volume is ignored.</li>"
            "<li>A few rows were removed during cleaning, so some returns span more than one day.</li>"
            "<li>Trading costs are not included. This is analysis, not a trading strategy.</li>"
            "</ul>"
        )
    with core.panel("team"):
        core.section("About this project")
        st.html(
            '<p class="gw-explain">Built for the Data Analytics and Visualization course by a team of three: '
            "data collection and cleaning (Member A), anomaly detection and backtesting (Member B), and the "
            "dashboard (Member C).</p>"
        )
