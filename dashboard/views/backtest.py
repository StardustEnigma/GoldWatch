import pandas as pd
import streamlit as st

import core

core.page_header(
    "Backtest",
    "Do sharp rises in gold tend to fall back? For every upward spike we check the price a few days later "
    "and compare it with what happens after an ordinary day.",
)

summary = core.backtest()
horizon = st.segmented_control("Holding period: days after the spike", core.HORIZONS, default=5, required=True,
                               format_func=lambda h: f"{h} day" + ("s" if h > 1 else ""),
                               key="backtest_horizon", persist_state="session")
row = summary.loc[horizon]
period = f"{horizon} trading day" + ("s" if horizon > 1 else "")
significant = bool(row["Significant"])

core.kpi_row([
    ("Spikes tested", core.count(int(row["Events"])), f"Upward spikes with a price {period} later"),
    ("Fell afterwards", core.count(row["Decline_Rate"] * 100, 1, "%"),
     f"{int(row['Declines'])} of {int(row['Events'])} spikes were lower"),
    ("Any day falls", core.count(row["Baseline_Decline_Rate"] * 100, 1, "%"),
     f"How often gold is lower after {period}"),
    ("Result", "Significant" if significant else "Not significant",
     f"p = {row['P_Value']:.3f}. Significant means below 0.05", "up" if significant else ""),
])

body = (f"After {period}, gold was lower after <b>{row['Decline_Rate']:.1%}</b> of upward spikes, compared with "
        f"<b>{row['Baseline_Decline_Rate']:.1%}</b> after any day (p = {row['P_Value']:.3f}).")
if significant:
    core.callout("Spikes were followed by a fall more often than usual",
                 body + " A gap this large is unlikely to be chance.", "positive")
else:
    core.callout(f"No clear reversal after {period}", body + " The gap is small enough to be chance.")

left, right = st.columns(2)
with left:
    with core.panel("declines"):
        core.section("How often the price fell", "Spikes compared with any day, for each holding period")
        core.show(core.decline_chart(summary, horizon),
                  alt="Grouped bar chart of the share of spikes followed by a fall, against the baseline")
        core.how_to_read("if the blue bar is clearly taller than the grey bar, spikes tend to be followed by "
                         "falls. Your selected holding period is the darker blue bar.")
with right:
    with core.panel("path"):
        path = core.event_path()
        core.section("Average price path after a spike",
                     f"Cumulative return over 20 trading days, average of {core.plural(int(path['Events'].iloc[0]), 'spike')}")
        core.show(core.path_chart(path), alt="Line chart of the average return in the 20 days after a spike")
        core.how_to_read("a blue line below the dashed line means gold did worse than usual after a spike. "
                         "The shaded band shows the range the true average is likely to be in.")

ev = core.events(horizon).sort_values("Signal_Date", ascending=False)
table = pd.DataFrame({
    "Spike date": ev["Signal_Date"].dt.date,
    "Rise that day": ev["Spike_Return"] * 100,
    "Return Z": ev["Spike_ZScore"],
    "Price at spike": ev["Entry_Close"],
    f"Price {horizon}d later": ev["Exit_Close"],
    "Change since": ev["Forward_Return"] * 100,
    "Outcome": ev["Forward_Return"].map(lambda v: "Not yet known" if pd.isna(v) else "Fell" if v < 0 else "Rose"),
})

with core.panel("events"):
    core.section("Every spike", f"{core.plural(len(table), 'upward spike')}, newest first, with the outcome "
                                f"after {period}")
    outcome_colors = {"Fell": core.DOWN, "Rose": core.UP, "Not yet known": core.INK_3}
    st.dataframe(
        table.style.apply(lambda c: [f"color: {outcome_colors[v]}; font-weight: 500" for v in c],
                          subset=["Outcome"]),
        hide_index=True,
        height=380,
        placeholder="—",
        column_config={
            "Spike date": st.column_config.DateColumn(format="D MMM YYYY"),
            "Rise that day": st.column_config.NumberColumn(format="%+.2f%%"),
            "Return Z": st.column_config.NumberColumn(format="%+.2f"),
            "Price at spike": st.column_config.NumberColumn(format="dollar"),
            f"Price {horizon}d later": st.column_config.NumberColumn(format="dollar"),
            "Change since": st.column_config.NumberColumn(format="%+.2f%%"),
        },
    )
    with st.container(horizontal=True, vertical_alignment="center"):
        st.download_button(f"Download {core.plural(len(table), 'row')} as CSV", table.to_csv(index=False),
                           f"goldwatch_backtest_{horizon}d.csv", "text/csv", icon=":material/download:")
        st.space("stretch")
        st.page_link(core.PAGES["method"][0], label="How this test works", icon=":material/arrow_forward:",
                     icon_position="right")
