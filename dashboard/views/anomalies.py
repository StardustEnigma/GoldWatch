import streamlit as st

import core

core.page_header(
    "Anomalies",
    "Days when gold moved far more than usual, compared with the trading days just before them.",
)

df = core.analysis()
window, threshold = core.settings()
flagged = df[df["Is_Anomaly"]]
counts = flagged["Event_Type"].value_counts()

core.kpi_row([
    ("Unusual days", core.count(len(flagged)), f"{len(flagged) / len(df):.1%} of {len(df):,} trading days"),
    ("Upward spikes", core.count(counts.get(core.EVENT_UP, 0)), "Unusually large rises", "up"),
    ("Downward shocks", core.count(counts.get(core.EVENT_DOWN, 0)), "Unusually large falls", "down"),
    ("Volume spikes", core.count(counts.get(core.EVENT_VOLUME, 0)), "Heavy trading, normal price move"),
])

with core.panel("zscore"):
    filters, period_col = st.columns([3, 2], vertical_alignment="bottom")
    with filters:
        types = st.pills("Show", list(core.EVENT_LABEL), selection_mode="multi", default=list(core.EVENT_LABEL),
                         format_func=core.EVENT_LABEL.get, key="anomaly_types", persist_state="session", wrap=True)
    with period_col:
        choice = st.segmented_control("Period", core.RANGES, default="Max", required=True, key="anomaly_range",
                                      persist_state="session", wrap=True)
    period = df[df["Date"] >= core.range_start(df["Date"], choice)]
    core.section("How unusual each day was", f"Return Z-score for every trading day, {choice} view")
    core.show(core.zscore_chart(period, threshold),
              alt="Bar chart of the daily return Z-score with dashed threshold lines")
    core.how_to_read(f"each bar is one trading day. Grey bars are normal days. Bars past the dashed lines "
                     f"(±{threshold:g}) were flagged: green for rises, red for falls.")

rows = period[period["Is_Anomaly"] & period["Event_Type"].isin(types or [])].sort_values("Date", ascending=False)

table_col, detail_col = st.columns([3, 2])
with table_col:
    with core.panel("table"):
        core.section("Flagged days", f"{core.plural(len(rows), 'day')}, newest first. "
                                     "Select a row to see why it was flagged.")
        selection = None
        if not types:
            st.info("Choose at least one event type above to see flagged days.", icon=":material/filter_alt:")
        elif rows.empty:
            st.info("No flagged days in this period. Try a longer period or a lower threshold in Settings.",
                    icon=":material/search_off:")
        else:
            selection = core.anomaly_table(rows, height=560, key=f"anomaly_table_{choice}", selectable=True,
                                           compact=True)
            st.download_button(f"Download {core.plural(len(rows), 'row')} as CSV", rows.to_csv(index=False),
                               "goldwatch_anomalies.csv", "text/csv", icon=":material/download:")

with detail_col:
    with core.panel("detail"):
        picked = selection.selection.rows if selection is not None else []
        if picked and picked[0] < len(rows):
            core.day_detail(rows.iloc[picked[0]], df)
        else:
            st.html('<div class="gw-empty"><span class="material-symbols-rounded">touch_app</span>'
                    '<p>Select a day in the table to see why it was flagged.</p></div>')
