import streamlit as st

import core

core.page_header("Data", "Browse, filter and download the numbers behind every chart.")

analysis = core.analysis()
horizon = st.session_state.get("backtest_horizon", 5)

DATASETS = {
    "Daily prices": {
        "about": "Cleaned daily prices for COMEX gold futures, one row per trading day.",
        "columns": [("Open, High, Low, Close", "Prices in US dollars per troy ounce"),
                    ("Volume", "Contracts traded. Not reliable before 2020"),
                    ("Daily return", "Change in the close since the previous trading day"),
                    ("High − low", "How far the price moved within the day")],
    },
    "Anomaly scores": {
        "about": "Every trading day with its Z-scores, anomaly score and label.",
        "columns": [("Return Z", "How many typical moves the day was from normal"),
                    ("Volume Z", "How unusual trading volume was. From 2020 only"),
                    ("Score", "Combined anomaly score, 0 to 100"),
                    ("Flagged", "Ticked if the day counts as an anomaly"),
                    ("Event", "Upward spike, downward shock or volume spike")],
    },
    "Backtest events": {
        "about": f"Every upward spike and what the price did {horizon} trading days later. Change the holding "
                 "period on the Backtest page.",
        "columns": [("Spike return", "How much gold rose on the spike day"),
                    ("Entry close", "Closing price on the spike day"),
                    ("Exit close", f"Closing price {horizon} trading days later"),
                    ("Return after", "Change from entry to exit")],
    },
}

with core.panel("filters"):
    choose, dates_col, extra = st.columns([3, 2, 1.2], vertical_alignment="bottom")
    with choose:
        dataset = st.segmented_control("1. Choose a dataset", list(DATASETS), default="Daily prices",
                                       required=True, key="data_set", persist_state="session")

if dataset == "Daily prices":
    data = core.prices()[["Date", "Open", "High", "Low", "Close", "Volume", "Daily_Return", "Price_Range"]].copy()
    data["Daily_Return"] *= 100
    config = {
        "Open": st.column_config.NumberColumn(format="dollar"),
        "High": st.column_config.NumberColumn(format="dollar"),
        "Low": st.column_config.NumberColumn(format="dollar"),
        "Close": st.column_config.NumberColumn(format="dollar"),
        "Volume": st.column_config.NumberColumn(format="localized"),
        "Daily_Return": st.column_config.NumberColumn("Daily return", format="%+.2f%%"),
        "Price_Range": st.column_config.NumberColumn("High − low", format="dollar"),
    }
elif dataset == "Anomaly scores":
    data = analysis[["Date", "Close", "Daily_Return", "Return_ZScore", "Volume_ZScore", "Anomaly_Score",
                     "Is_Anomaly", "Event_Type"]].copy()
    data["Daily_Return"] *= 100
    data["Event_Type"] = data["Event_Type"].map(core.EVENT_LABEL).fillna("")
    config = {
        "Close": st.column_config.NumberColumn(format="dollar"),
        "Daily_Return": st.column_config.NumberColumn("Daily return", format="%+.2f%%"),
        "Return_ZScore": st.column_config.NumberColumn("Return Z", format="%+.2f"),
        "Volume_ZScore": st.column_config.NumberColumn("Volume Z", format="%+.2f"),
        "Anomaly_Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
        "Is_Anomaly": st.column_config.CheckboxColumn("Flagged"),
        "Event_Type": st.column_config.TextColumn("Event"),
    }
else:
    ev = core.events(horizon)
    data = ev[["Signal_Date", "Spike_Return", "Spike_ZScore", "Entry_Close", "Exit_Close", "Forward_Return"]]
    data = data.rename(columns={"Signal_Date": "Date"})
    data[["Spike_Return", "Forward_Return"]] *= 100
    config = {
        "Spike_Return": st.column_config.NumberColumn("Spike return", format="%+.2f%%"),
        "Spike_ZScore": st.column_config.NumberColumn("Return Z", format="%+.2f"),
        "Entry_Close": st.column_config.NumberColumn("Entry close", format="dollar"),
        "Exit_Close": st.column_config.NumberColumn("Exit close", format="dollar"),
        "Forward_Return": st.column_config.NumberColumn("Return after", format="%+.2f%%"),
    }
config["Date"] = st.column_config.DateColumn(format="D MMM YYYY")

first, last = data["Date"].min().date(), data["Date"].max().date()
with dates_col:
    picked = st.date_input("2. Pick dates", value=(first, last), min_value=first, max_value=last,
                           format="DD/MM/YYYY", key=f"data_dates_{dataset}")
with extra:
    only_flagged = st.toggle("Flagged days only", key="data_flagged", disabled=dataset != "Anomaly scores",
                             help="Available for the anomaly scores dataset")
    only_flagged = only_flagged and dataset == "Anomaly scores"

start, end = (picked[0], picked[1]) if len(picked) == 2 else (picked[0], last)
view = data[data["Date"].dt.date.between(start, end)]
if only_flagged:
    view = view[view["Is_Anomaly"]]
view = view.sort_values("Date", ascending=False)

with core.panel("table"):
    st.html(f'<h2 class="gw-h2">{dataset}</h2><p class="gw-sub">{DATASETS[dataset]["about"]} Showing '
            f'<b>{len(view):,}</b> of {len(data):,} rows, {core.day(start)} to {core.day(end)}, newest first.</p>')
    if view.empty:
        st.info("No rows match these filters. Widen the date range or turn off “Flagged days only”.",
                icon=":material/search_off:")
    else:
        st.dataframe(view.assign(Date=view["Date"].dt.date), hide_index=True, height=520, column_config=config,
                     placeholder="—")
        with st.container(horizontal=True, vertical_alignment="center"):
            export = view.assign(Date=view["Date"].dt.strftime("%Y-%m-%d"))
            st.download_button(f"3. Download {len(view):,} rows as CSV", export.to_csv(index=False),
                               f"goldwatch_{dataset.lower().replace(' ', '_')}.csv", "text/csv", type="primary",
                               icon=":material/download:")
            st.caption("Click a column header to sort. Hover over the table to search or view it full screen.")

with st.expander("What the columns mean", icon=":material/info:"):
    core.stat_grid([(name, meaning) for name, meaning in DATASETS[dataset]["columns"]], columns=1)
