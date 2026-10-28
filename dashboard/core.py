"""Shared data access, page furniture and charts for the GoldWatch dashboard."""

import sys
from datetime import date
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DASHBOARD_DIR = Path(__file__).resolve().parent
ROOT = DASHBOARD_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anomaly_detection import (  # noqa: E402
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW,
    EVENT_DOWN,
    EVENT_UP,
    EVENT_VOLUME,
    detect_anomalies,
)
from src.backtest import build_event_table, forward_returns, run_backtest  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "gold_futures_processed.csv"
SOURCE_URL = "https://finance.yahoo.com/quote/GC%3DF/"
HORIZONS = (1, 5, 10, 20)
RANGES = ["1M", "6M", "YTD", "1Y", "5Y", "Max"]

PAGES = {
    "overview": ("dashboard/views/overview.py", "Overview", ":material/home:"),
    "anomalies": ("dashboard/views/anomalies.py", "Anomalies", ":material/warning:"),
    "backtest": ("dashboard/views/backtest.py", "Backtest", ":material/science:"),
    "data": ("dashboard/views/data.py", "Data", ":material/table_chart:"),
    "method": ("dashboard/views/method.py", "How it works", ":material/menu_book:"),
}

INK, INK_2, INK_3 = "#16191D", "#4B5563", "#6B7280"
LINE, GRID, MUTED_BAR = "#E3E6EA", "#EEF0F3", "#C4C9D0"
UP, DOWN, ACCENT, GOLD = "#137333", "#C5221F", "#1F5FAD", "#B8860B"
FONT = "IBM Plex Sans, Helvetica Neue, Arial, sans-serif"

EVENT_LABEL = {EVENT_UP: "▲ Upward spike", EVENT_DOWN: "▼ Downward shock", EVENT_VOLUME: "◆ Volume spike"}
EVENT_COLOR = {EVENT_UP: UP, EVENT_DOWN: DOWN, EVENT_VOLUME: ACCENT}
CHART_CONFIG = {"displayModeBar": False, "responsive": True}

GLOSSARY = [
    ("Daily return", "How much the closing price changed since the previous trading day, in percent."),
    ("Typical move", "The standard deviation of daily returns over the look-back window: how big an "
                     "ordinary day's move has recently been."),
    ("Z-score", "How many typical moves a day is away from the recent average. 0 is a normal day; "
                "beyond ±2.5 is unusual."),
    ("Anomaly", "A day whose return Z-score is beyond the threshold, or whose volume Z-score is above it."),
    ("Upward spike", "An unusually large rise in price."),
    ("Downward shock", "An unusually large fall in price."),
    ("Volume spike", "Unusually heavy trading without an unusual price move. Volume is only used from 2020."),
    ("Anomaly score", "A 0 to 100 summary of how extreme a day was, combining price and volume."),
    ("Holding period", "How many trading days after a spike we check the price."),
    ("Baseline", "How often gold falls over the same holding period after any day. The fair comparison."),
    ("p-value", "The chance of seeing a difference this large if spikes were no different from ordinary "
                "days. Below 0.05 counts as significant."),
]


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def _load_prices(mtime: float) -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["Date"])


def _mtime() -> float:
    return DATA_PATH.stat().st_mtime


def prices() -> pd.DataFrame:
    return _load_prices(_mtime())


@st.cache_data(show_spinner="Finding unusual days…")
def _analysis(window: int, threshold: float, mtime: float) -> pd.DataFrame:
    return detect_anomalies(_load_prices(mtime), window=window, threshold=threshold)


@st.cache_data(show_spinner="Testing what happened after each spike…")
def _backtest(window: int, threshold: float, mtime: float) -> pd.DataFrame:
    return run_backtest(_analysis(window, threshold, mtime), horizons=HORIZONS).set_index("Horizon_Days")


@st.cache_data(show_spinner=False)
def _events(window: int, threshold: float, mtime: float, horizon: int) -> pd.DataFrame:
    return build_event_table(_analysis(window, threshold, mtime), horizon=horizon)


@st.cache_data(show_spinner=False)
def _event_path(window: int, threshold: float, mtime: float, max_h: int = 20) -> pd.DataFrame:
    a = _analysis(window, threshold, mtime)
    close = a["Close"].reset_index(drop=True)
    has_z = a["Return_ZScore"].notna().values
    spikes = np.where((a["Event_Type"] == EVENT_UP).values)[0]
    spikes = spikes[spikes + max_h < len(close)]
    days = np.arange(max_h + 1)
    base = np.array([forward_returns(close, k)[has_z].mean() if k else 0.0 for k in days])
    if len(spikes) < 2:
        return pd.DataFrame({"Day": days, "Spike": np.nan, "Low": np.nan, "High": np.nan, "Baseline": base,
                             "Events": len(spikes)})
    paths = np.array([[close[t + k] / close[t] - 1 for k in days] for t in spikes])
    mean = paths.mean(axis=0)
    se = paths.std(axis=0, ddof=1) / np.sqrt(len(spikes))
    return pd.DataFrame({"Day": days, "Spike": mean, "Low": mean - 1.96 * se, "High": mean + 1.96 * se,
                         "Baseline": base, "Events": len(spikes)})


def settings() -> tuple:
    return int(st.session_state.get("window", DEFAULT_WINDOW)), float(st.session_state.get("threshold",
                                                                                           DEFAULT_THRESHOLD))


def analysis() -> pd.DataFrame:
    return _analysis(*settings(), _mtime())


def backtest() -> pd.DataFrame:
    return _backtest(*settings(), _mtime())


def events(horizon: int) -> pd.DataFrame:
    return _events(*settings(), _mtime(), horizon)


def event_path() -> pd.DataFrame:
    return _event_path(*settings(), _mtime())


def range_start(dates: pd.Series, choice: str) -> pd.Timestamp:
    last = dates.max()
    offsets = {"1M": pd.DateOffset(months=1), "6M": pd.DateOffset(months=6), "1Y": pd.DateOffset(years=1),
               "5Y": pd.DateOffset(years=5)}
    if choice == "YTD":
        return pd.Timestamp(year=last.year, month=1, day=1)
    if choice in offsets:
        return last - offsets[choice]
    return dates.min()


# ----------------------------------------------------------------------------- formatting
def money(v) -> str:
    return "—" if pd.isna(v) else f"${v:,.2f}"


def pct(v, digits: int = 2, sign: bool = True) -> str:
    return "—" if pd.isna(v) else f"{v * 100:{'+' if sign else ''}.{digits}f}%"


def day(d) -> str:
    d = pd.Timestamp(d)
    return f"{d.day} {d:%b %Y}"


def tone(v) -> str:
    return "up" if v > 0 else "down" if v < 0 else "muted"


def plural(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


# ----------------------------------------------------------------------------- settings & help
def init_state() -> None:
    st.session_state.setdefault("window", DEFAULT_WINDOW)
    st.session_state.setdefault("threshold", DEFAULT_THRESHOLD)


def _settings_changed() -> None:
    window, threshold = settings()
    st.toast(f"Updated: {window}-day window, threshold {threshold:g}", icon=":material/check_circle:")


def _reset_settings() -> None:
    st.session_state["window"] = DEFAULT_WINDOW
    st.session_state["threshold"] = DEFAULT_THRESHOLD
    st.toast("Settings reset to the defaults", icon=":material/restart_alt:")


def _settings_popover() -> None:
    with st.popover("Settings", icon=":material/tune:"):
        st.markdown("**Detection settings**")
        st.slider("Look-back window (trading days)", 10, 120, step=5, key="window", persist_state="session",
                  on_change=_settings_changed, help="How many previous trading days define a normal move.")
        st.slider("Alert threshold (Z-score)", 1.5, 4.0, step=0.25, key="threshold", persist_state="session",
                  on_change=_settings_changed,
                  help="How unusual a day must be to be flagged. 2.5 means 2.5 typical moves from the average.")
        st.button("Reset to defaults", icon=":material/restart_alt:", on_click=_reset_settings,
                  disabled=settings() == (DEFAULT_WINDOW, DEFAULT_THRESHOLD), width="stretch")
        st.caption("Changes apply to every page.")


def glossary() -> None:
    rows = "".join(f"<div><dt>{escape(t)}</dt><dd>{escape(d)}</dd></div>" for t, d in GLOSSARY)
    st.html(f'<dl class="gw-glossary">{rows}</dl>')


@st.dialog("Help", width="large")
def _help_dialog() -> None:
    st.markdown("GoldWatch flags days when the gold price moved far more than usual, then tests whether sharp "
                "rises tend to fall back.")
    st.markdown("- **Overview**: today's price and a summary\n"
                "- **Anomalies**: every flagged day, and why it was flagged\n"
                "- **Backtest**: what happened after sharp rises\n"
                "- **Data**: browse and download the numbers\n"
                "- **How it works**: the method, step by step")
    st.markdown("**What the terms mean**")
    glossary()


def utilities() -> None:
    """Top-right utility bar, the same on every page."""
    with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", gap="small"):
        st.html(f'<span class="gw-asof"><span class="material-symbols-rounded">event</span>'
                f'Data to {day(prices()["Date"].max())}</span>')
        _settings_popover()
        if st.button("Help", icon=":material/help:", key="help_button"):
            _help_dialog()


# ----------------------------------------------------------------------------- page furniture
def inject_css() -> None:
    st.html(f"<style>{(DASHBOARD_DIR / 'style.css').read_text()}</style>")


def page_header(title: str, lede: str) -> None:
    left, right = st.columns([3, 2], vertical_alignment="top")
    with left:
        st.html(f'<h1 class="gw-title">{escape(title)}</h1><p class="gw-lede">{lede}</p>')
    with right:
        utilities()


def panel(key: str):
    return st.container(border=True, key=f"panel-{key}")


def section(title: str, sub: str = "") -> None:
    st.html(f'<h2 class="gw-h2">{escape(title)}</h2>' + (f'<p class="gw-sub">{sub}</p>' if sub else ""))


def how_to_read(text: str) -> None:
    st.html(f'<p class="gw-read"><span>How to read this:</span> {text}</p>')


def stat_grid(items, columns: int = 3) -> None:
    """items: (label, value) or (label, value, css_class)."""
    rows = []
    for item in items:
        label, value, cls = (*item, "")[:3]
        rows.append(f'<div class="gw-stat"><dt>{escape(label)}</dt><dd class="{cls}">{value}</dd></div>')
    st.html(f'<dl class="gw-stats gw-stats--{columns}">{"".join(rows)}</dl>')


def count(value: float, decimals: int = 0, suffix: str = "") -> str:
    """A number that counts up when it first appears. Screen readers get the plain value."""
    plain = f"{value:,.{decimals}f}{suffix}"
    whole_text, _, frac = f"{abs(value):.{decimals}f}".partition(".")
    whole = int(whole_text)
    tail = f".{frac}" if frac else ""
    if whole >= 1000:
        digits = f'<span class="gw-count gw-count--k" style="--k:{whole // 1000};--r:{whole % 1000}"></span>'
    else:
        digits = f'<span class="gw-count" style="--to:{whole}"></span>'
    sign = "−" if value < 0 else ""
    return (f'<span class="gw-sr">{plain}</span>'
            f'<span aria-hidden="true">{sign}{digits}{tail}{escape(suffix)}</span>')


def kpi_row(items) -> None:
    """items: (label, value_html, note) or (label, value_html, note, css_class)."""
    cards = []
    for item in items:
        label, value, note, cls = (*item, "")[:4]
        cards.append(f'<div class="gw-kpi"><div class="gw-kpi__label">{escape(label)}</div>'
                     f'<div class="gw-kpi__value {cls}">{value}</div><div class="gw-kpi__note">{note}</div></div>')
    st.html(f'<div class="gw-kpis">{"".join(cards)}</div>')


def callout(title: str, body: str, kind: str = "info") -> None:
    st.html(f'<div class="gw-callout gw-callout--{kind}"><div class="gw-callout__title">{escape(title)}</div>'
            f'<p>{body}</p></div>')


def explore_card(page: str, stat: str, stat_label: str, text: str, link_label: str) -> None:
    """A clickable summary card: icon, headline number, one line of context and a link."""
    path, title, icon = PAGES[page]
    with st.container(border=True, key=f"card-{page}"):
        st.html(f'<div class="gw-card-head"><span class="gw-card-icon material-symbols-rounded">'
                f'{icon.strip(":").split("/")[1]}</span><h3 class="gw-h3">{escape(title)}</h3></div>'
                f'<div class="gw-card-stat">{stat}</div><div class="gw-card-stat-label">{stat_label}</div>'
                f'<p class="gw-card-text">{text}</p>')
        st.page_link(path, label=link_label, icon=":material/arrow_forward:", icon_position="right")


def footer() -> None:
    st.html(
        f'<footer class="gw-footer">Data: daily settlement prices for COMEX gold futures from '
        f'<a href="{SOURCE_URL}" target="_blank" rel="noopener">Yahoo Finance\'s GC=F page</a>, up to '
        f'{day(prices()["Date"].max())}. Volume before 2020 is not reliable and is not used.<br>'
        f'© {date.today().year} GoldWatch, a Data Analytics and Visualization course project. '
        f'Not investment advice.</footer>'
    )


def event_style(col: pd.Series) -> list:
    colors = {label: EVENT_COLOR[ev] for ev, label in EVENT_LABEL.items()}
    return [f"color: {colors.get(v, INK)}; font-weight: 500" for v in col]


def anomaly_table(rows: pd.DataFrame, height="auto", key: str | None = None, selectable: bool = False,
                  compact: bool = False):
    table = pd.DataFrame({
        "Date": rows["Date"].dt.date,
        "Event": rows["Event_Type"].map(EVENT_LABEL),
        "Close": rows["Close"],
        "Return": rows["Daily_Return"] * 100,
        "Return Z": rows["Return_ZScore"],
        "Volume Z": rows["Volume_ZScore"],
        "Score": rows["Anomaly_Score"],
    })
    if compact:
        table = table.drop(columns=["Close", "Volume Z"])
    return st.dataframe(
        table.style.apply(event_style, subset=["Event"]),
        hide_index=True,
        height=height,
        placeholder="—",
        key=key,
        on_select="rerun" if selectable else "ignore",
        selection_mode="single-row",
        selection_default={"selection": {"rows": [0]}} if selectable else None,
        column_config={
            "Date": st.column_config.DateColumn(format="D MMM YYYY", width="small"),
            "Event": st.column_config.TextColumn(width="medium"),
            "Close": st.column_config.NumberColumn("Close", format="dollar", help="Closing price, USD per ounce"),
            "Return": st.column_config.NumberColumn(format="%+.2f%%", help="Change from the previous close"),
            "Return Z": st.column_config.NumberColumn(format="%+.2f", help="How many typical moves from normal"),
            "Volume Z": st.column_config.NumberColumn(format="%+.2f", help="How unusual volume was (from 2020)"),
            "Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f",
                                                     help="Combined anomaly score, 0 to 100"),
        },
    )


def explain_day(row: pd.Series, window: int, threshold: float) -> str:
    """Plain-language reason a day was flagged."""
    move = "rose" if row["Daily_Return"] >= 0 else "fell"
    text = (f"Gold {move} <b>{abs(row['Daily_Return']) * 100:.2f}%</b> to {money(row['Close'])}. Over the previous "
            f"{window} trading days, a typical daily move was about <b>±{row['Return_Rolling_Std'] * 100:.2f}%</b>.")
    z = row["Return_ZScore"]
    if abs(z) >= threshold:
        text += (f" This move was <b>{abs(z):.1f} times</b> that typical size, beyond the {threshold:g} threshold, "
                 f"so the day was flagged.")
    else:
        text += f" The price move itself was within the normal range (Z = {z:+.1f})."
    vz = row["Volume_ZScore"]
    if pd.notna(vz) and vz >= threshold:
        text += f" Trading volume was also unusually heavy (volume Z = {vz:+.1f})."
    return text


def day_detail(row: pd.Series, df: pd.DataFrame, chart: bool = True) -> None:
    """Explain one flagged day: headline, plain-language reason, price around it, key numbers."""
    window, threshold = settings()
    st.html(f'<p class="gw-sub">Selected day</p><h2 class="gw-h2">{day(row["Date"])}</h2>'
            f'<p class="gw-event" style="color:{EVENT_COLOR[row["Event_Type"]]}">'
            f'{EVENT_LABEL[row["Event_Type"]]}</p>'
            f'<p class="gw-explain">{explain_day(row, window, threshold)}</p>')
    if chart:
        i = df.index[df["Date"] == row["Date"]][0]
        around = df.iloc[max(0, i - 60):i + 41]
        show(price_chart(around, show_anomalies=False, height=230, volume=False, highlight=row["Date"]),
             alt=f"Gold price in the weeks around {day(row['Date'])}, with the day circled")
        st.caption("Price in the months around this day. The day is circled.")
    vz = row["Volume_ZScore"]
    stat_grid([
        ("Close", money(row["Close"])),
        ("Change that day", pct(row["Daily_Return"]), tone(row["Daily_Return"])),
        (f"Typical move ({window} days)", f"±{row['Return_Rolling_Std'] * 100:.2f}%"),
        ("Return Z-score", f"{row['Return_ZScore']:+.2f}"),
        ("Volume Z-score", "—" if pd.isna(vz) else f"{vz:+.2f}"),
        ("Anomaly score", f"{row['Anomaly_Score']:.0f} / 100"),
    ], columns=1)


# ----------------------------------------------------------------------------- charts
def _layout(fig: go.Figure, height: int, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8 if not legend else 36, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=FONT, size=12, color=INK_2),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", bordercolor=LINE, font=dict(family=FONT, color=INK, size=12)),
        showlegend=legend,
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", font=dict(size=12, color=INK_2),
                    bgcolor="rgba(0,0,0,0)", traceorder="normal"),
        bargap=0.15,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=LINE, ticks="outside", tickcolor=LINE,
                     showspikes=True, spikecolor=INK_3, spikethickness=1, spikedash="dot", spikemode="across",
                     automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=LINE, side="right", ticks="", automargin=True)
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha})"


def price_chart(d: pd.DataFrame, show_anomalies: bool, height: int = 430, volume: bool = True,
                highlight=None) -> go.Figure:
    color = UP if d["Close"].iloc[-1] >= d["Close"].iloc[0] else DOWN
    has_volume = volume and d["Volume"].fillna(0).gt(0).any()
    fig = make_subplots(rows=2 if has_volume else 1, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.8, 0.2] if has_volume else [1])
    lo, hi = d["Low"].min(), d["High"].max()
    pad = (hi - lo) * 0.06 or hi * 0.01
    fig.add_trace(go.Scatter(x=d["Date"], y=np.full(len(d), lo - pad), line=dict(width=0), hoverinfo="skip",
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["Close"], name="Close", line=dict(color=color, width=1.8),
                             showlegend=False, fill="tonexty", fillcolor=_rgba(color, 0.07),
                             hovertemplate="Close  <b>$%{y:,.2f}</b><extra></extra>"), row=1, col=1)
    if show_anomalies:
        for ev, symbol in [(EVENT_UP, "triangle-up"), (EVENT_DOWN, "triangle-down"), (EVENT_VOLUME, "diamond")]:
            e = d[d["Event_Type"] == ev]
            if e.empty:
                continue
            fig.add_trace(go.Scatter(
                x=e["Date"], y=e["Close"], mode="markers", name=EVENT_LABEL[ev][2:],
                marker=dict(symbol=symbol, size=9, color=EVENT_COLOR[ev], line=dict(color="white", width=1)),
                customdata=np.c_[e["Return_ZScore"]],
                hovertemplate=f"{EVENT_LABEL[ev][2:]}  Z %{{customdata[0]:+.1f}}<extra></extra>"), row=1, col=1)
    if highlight is not None:
        h = d[d["Date"] == highlight]
        fig.add_trace(go.Scatter(x=h["Date"], y=h["Close"], mode="markers", name="Selected day",
                                 marker=dict(size=13, color="rgba(0,0,0,0)", line=dict(color=INK, width=2)),
                                 hoverinfo="skip"), row=1, col=1)
    if has_volume:
        fig.add_trace(go.Bar(x=d["Date"], y=d["Volume"].where(d["Volume"] > 0), name="Volume",
                             marker_color=MUTED_BAR, showlegend=False,
                             hovertemplate="Volume  %{y:,.0f}<extra></extra>"), row=2, col=1)
        fig.update_yaxes(tickformat="~s", nticks=3, row=2, col=1)
    fig.update_yaxes(range=[lo - pad, hi + pad], tickprefix="$", tickformat=",.0f", row=1, col=1)
    return _layout(fig, height, legend=show_anomalies)


def zscore_chart(d: pd.DataFrame, threshold: float) -> go.Figure:
    z = d["Return_ZScore"]
    colors = np.where(z >= threshold, UP, np.where(z <= -threshold, DOWN, MUTED_BAR))
    fig = go.Figure(go.Bar(x=d["Date"], y=z, marker_color=colors, marker_line_width=0, name="Return Z",
                           hovertemplate="Return Z  <b>%{y:+.2f}</b><extra></extra>"))
    for y in (threshold, -threshold):
        fig.add_hline(y=y, line=dict(color=INK_3, width=1, dash="dash"))
    return _layout(fig, 300)


def window_chart(a: pd.DataFrame, when) -> go.Figure:
    """The previous 30 returns, today's return and the normal range: Algorithm 1 in one picture."""
    i = a.index[a["Date"] == pd.Timestamp(when)][0]
    window, _ = settings()
    past = a.iloc[i - window:i]
    mu, sd = a.loc[i, "Return_Rolling_Mean"] * 100, a.loc[i, "Return_Rolling_Std"] * 100
    threshold = settings()[1]
    fig = go.Figure()
    fig.add_hrect(y0=mu - threshold * sd, y1=mu + threshold * sd, fillcolor=_rgba(ACCENT, 0.08), line_width=0)
    fig.add_trace(go.Bar(x=past["Date"], y=past["Daily_Return"] * 100, marker_color=MUTED_BAR,
                         name=f"Previous {window} days", hovertemplate="%{x|%d %b %Y}  %{y:+.2f}%<extra></extra>"))
    fig.add_trace(go.Bar(x=[a.loc[i, "Date"]], y=[a.loc[i, "Daily_Return"] * 100], marker_color=GOLD,
                         name="The day being tested", hovertemplate="%{x|%d %b %Y}  %{y:+.2f}%<extra></extra>"))
    fig.add_hline(y=mu, line=dict(color=INK_2, width=1, dash="dot"))
    fig.add_annotation(xref="paper", x=0, y=mu + threshold * sd, yanchor="bottom", xanchor="left",
                       showarrow=False, text=f"normal range (±{threshold:g} typical moves)",
                       font=dict(size=11, color=ACCENT))
    fig = _layout(fig, 320, legend=True)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(showspikes=False)
    fig.update_yaxes(ticksuffix="%", side="left")
    return fig


def decline_chart(summary: pd.DataFrame, selected: int) -> go.Figure:
    labels = [f"{h} day" + ("s" if h > 1 else "") for h in summary.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=summary["Decline_Rate"] * 100, name="After an upward spike",
                         marker_color=[ACCENT if h == selected else _rgba(ACCENT, 0.45) for h in summary.index],
                         text=[f"{v:.0%}" for v in summary["Decline_Rate"]], textposition="outside",
                         hovertemplate="After spike  %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Bar(x=labels, y=summary["Baseline_Decline_Rate"] * 100, name="Any day (baseline)",
                         marker_color=MUTED_BAR, text=[f"{v:.0%}" for v in summary["Baseline_Decline_Rate"]],
                         textposition="outside", hovertemplate="Baseline  %{y:.1f}%<extra></extra>"))
    fig.update_traces(textfont=dict(size=12, color=INK_2), cliponaxis=False)
    fig.update_layout(barmode="group", bargroupgap=0.08)
    fig = _layout(fig, 320, legend=True)
    fig.update_layout(hovermode="x")
    fig.update_xaxes(showspikes=False)
    fig.update_yaxes(range=[0, 75], ticksuffix="%", side="left")
    return fig


def path_chart(p: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[p["Day"], p["Day"][::-1]], y=np.r_[p["High"], p["Low"][::-1]] * 100,
                             fill="toself", fillcolor=_rgba(ACCENT, 0.12), line=dict(width=0), hoverinfo="skip",
                             name="95% confidence"))
    fig.add_trace(go.Scatter(x=p["Day"], y=p["Spike"] * 100, name="After an upward spike",
                             line=dict(color=ACCENT, width=2.2), mode="lines+markers", marker=dict(size=4),
                             hovertemplate="After spike  %{y:+.2f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=p["Day"], y=p["Baseline"] * 100, name="Any day (baseline)",
                             line=dict(color=INK_3, width=2, dash="dash"),
                             hovertemplate="Baseline  %{y:+.2f}%<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=LINE, width=1))
    fig = _layout(fig, 320, legend=True)
    fig.update_xaxes(title=dict(text="Trading days after the spike", font=dict(size=12)), showspikes=False)
    fig.update_yaxes(ticksuffix="%", side="left")
    return fig


def show(fig: go.Figure, alt: str) -> None:
    st.plotly_chart(fig, theme=None, config=CHART_CONFIG, alt=alt)
