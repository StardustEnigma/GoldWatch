"""GoldWatch: Institutional Financial Intelligence Dashboard.

Commodities Intelligence & Anomaly Detection for COMEX Gold Futures (GC=F).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Project Paths & Page Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "gold_futures_processed.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "gold_futures_raw.csv"
CSS_PATH = PROJECT_ROOT / "assets" / "style.css"
ANALYSIS_PATHS = (
    PROJECT_ROOT / "data" / "processed" / "gold_futures_analysis.csv",
    PROJECT_ROOT / "data" / "processed" / "gold_futures_backtest.csv",
    PROJECT_ROOT / "app_data" / "analysis.csv",
)

st.set_page_config(
    page_title="GoldWatch | Financial Intelligence",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_custom_css() -> None:
    """Inject centralized CSS design system from assets/style.css."""
    if CSS_PATH.exists():
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data Loading & Caching (Preserved Data Contracts)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str, modified_ns: int) -> pd.DataFrame:
    """Load market dataset and normalize dates and numeric fields for analysis."""
    del modified_ns
    frame = pd.read_csv(path)
    if not {"Date", "Close"}.issubset(frame.columns):
        raise ValueError("The data file must contain Date and Close columns.")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    for col in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Daily_Return",
        "Price_Range",
        "Price_Range_Percentage",
    ]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = (
        frame.dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
    # Historical GC=F zero volumes in early source data do not represent
    # a verified zero-contract session. Mark observed vs reported zero.
    if "Volume" in frame:
        frame["Volume_Observed"] = frame["Volume"].where(frame["Volume"] > 0)
        frame["Volume_Data_Status"] = np.where(
            frame["Volume"] > 0, "Observed", "Unavailable / reported as zero"
        )
    return frame


@st.cache_data(show_spinner=False)
def compute_signals(
    frame: pd.DataFrame, window: int, threshold: float, volume_weight: float = 0.5
) -> pd.DataFrame:
    """Build causal rolling signals using robust median/MAD and balanced components."""
    out = frame.copy().sort_values("Date").reset_index(drop=True)
    returns = pd.to_numeric(
        out.get("Daily_Return", out["Close"].pct_change()), errors="coerce"
    )
    out["Daily_Return"] = returns
    prior_returns = returns.shift(1)
    ret_median = prior_returns.rolling(window, min_periods=window).median()
    ret_mad = prior_returns.rolling(window, min_periods=window).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )
    ret_scale = (1.4826 * ret_mad).replace(0, np.nan)
    out["Return_ZScore"] = ((returns - ret_median) / ret_scale).replace(
        [np.inf, -np.inf], np.nan
    )

    if "Volume_Observed" not in out:
        volume = pd.to_numeric(
            out.get("Volume", pd.Series(np.nan, index=out.index)), errors="coerce"
        )
        out["Volume_Observed"] = volume.where(volume > 0)
    observed_volume = pd.to_numeric(out["Volume_Observed"], errors="coerce")
    log_volume = np.log1p(observed_volume)
    prior_volume = log_volume.shift(1)
    vol_median = prior_volume.rolling(window, min_periods=window).median()
    vol_mad = prior_volume.rolling(window, min_periods=window).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )
    vol_scale = (1.4826 * vol_mad).replace(0, np.nan)
    out["Volume_ZScore"] = ((log_volume - vol_median) / vol_scale).replace(
        [np.inf, -np.inf], np.nan
    )

    out["Return_Contribution"] = (
        out["Return_ZScore"].abs().clip(upper=threshold * 2) / (threshold * 2)
    )
    out["Volume_Contribution"] = (
        out["Volume_ZScore"].abs().clip(upper=threshold * 2) / (threshold * 2)
    )
    components = pd.concat(
        [out["Return_Contribution"], out["Volume_Contribution"]], axis=1
    )
    weights = pd.Series({"Return_Contribution": 1.0, "Volume_Contribution": volume_weight})
    weighted = components.mul(weights, axis=1)
    available_weight = components.notna().mul(weights, axis=1).sum(axis=1)
    out["Anomaly_Score"] = (
        weighted.sum(axis=1, min_count=1).div(available_weight.replace(0, np.nan))
        * 100
    )
    out["Return_Alert"] = out["Return_ZScore"].abs() >= threshold
    out["Volume_Alert"] = (out["Volume_ZScore"].abs() >= threshold) & (volume_weight > 0)
    out["Is_Anomaly"] = out["Return_Alert"] | out["Volume_Alert"]
    out["Event_Type"] = np.select(
        [
            out["Return_ZScore"] >= threshold,
            out["Return_ZScore"] <= -threshold,
            out["Volume_Alert"] & ~out["Return_Alert"],
        ],
        ["Upward spike", "Downward shock", "Volume dislocation"],
        default="—",
    )
    return out


def load_person_b_outputs(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    """Load Person B's analysis/backtest artifacts when available."""
    analysis = None
    backtest = None
    note = "Person B analysis not found · showing transparent dashboard preview signals"
    for path in ANALYSIS_PATHS:
        if not path.exists():
            continue
        try:
            candidate = pd.read_csv(path)
            if "Date" in candidate:
                candidate["Date"] = pd.to_datetime(candidate["Date"], errors="coerce")
                if {
                    "Return_ZScore",
                    "Volume_ZScore",
                    "Anomaly_Score",
                    "Is_Anomaly",
                }.intersection(candidate.columns):
                    analysis = candidate.dropna(subset=["Date"]).drop_duplicates(
                        "Date", keep="last"
                    )
                    note = f"Person B analysis loaded · {path.name}"
                if (
                    "Forward_Return" in candidate
                    or "Backtest_Result" in candidate
                    or "Signal_Date" in candidate
                ):
                    backtest = candidate
        except (OSError, ValueError, pd.errors.ParserError):
            continue
    if analysis is not None:
        out = frame.merge(analysis, on="Date", how="left", suffixes=("", "_person_b"))
        return out, backtest, note
    return None, backtest, note


# ---------------------------------------------------------------------------
# Formatting & Styling Helpers
# ---------------------------------------------------------------------------
def money(value: float) -> str:
    return f"${value:,.2f}" if pd.notna(value) else "—"


def pct(value: float, decimals: int = 2) -> str:
    return f"{value:+.{decimals}f}%" if pd.notna(value) else "—"


def build_event_reasons(row: pd.Series, threshold: float) -> str:
    reasons = []
    rz, vz = row.get("Return_ZScore", np.nan), row.get("Volume_ZScore", np.nan)
    if pd.notna(rz) and abs(rz) >= threshold:
        reasons.append(f"Return {'+' if rz > 0 else ''}{rz:.2f}σ")
    if pd.notna(vz) and abs(vz) >= threshold:
        reasons.append(f"Volume {'+' if vz > 0 else ''}{vz:.2f}σ")
    return "; ".join(reasons) if reasons else "Sub-threshold contribution"


def create_dark_chart(height: int = 340) -> go.Figure:
    """Create a standardized institutional-grade dark Plotly figure."""
    fig = go.Figure()
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, -apple-system, sans-serif",
            color="#94a3b8",
            size=11,
        ),
        margin=dict(l=10, r=16, t=28, b=10),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0b1722",
            bordercolor="#223c52",
            font=dict(family="JetBrains Mono, monospace", size=11, color="#f1f5f9"),
        ),
        legend=dict(
            orientation="h",
            y=1.06,
            x=0,
            bgcolor="rgba(11,23,34,0.75)",
            bordercolor="#162838",
            borderwidth=1,
            font=dict(size=10, color="#94a3b8"),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#132433",
            gridwidth=1,
            linecolor="#1a2e40",
            zeroline=False,
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color="#64748b"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#132433",
            gridwidth=1,
            linecolor="#1a2e40",
            zeroline=False,
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color="#64748b"),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Main Application Flow
# ---------------------------------------------------------------------------
def main() -> None:
    load_custom_css()

    # --- Sidebar: Streamlined Institutional Controls ---
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:0.6rem; padding: 0.2rem 0 0.8rem 0;">
                <div style="width:28px; height:28px; background:linear-gradient(135deg, rgba(245,189,79,0.25) 0%, rgba(20,184,166,0.2) 100%); border:1px solid rgba(245,189,79,0.4); border-radius:4px; display:flex; align-items:center; justify-content:center; color:#f5bd4f; font-weight:700; font-size:13px;">GW</div>
                <div>
                    <div style="font-size:1.05rem; font-weight:700; color:#f1f5f9; letter-spacing:-0.02em;">Gold<span style="color:#f5bd4f;">Watch</span></div>
                    <div style="font-size:0.68rem; font-family:'JetBrains Mono',monospace; color:#64748b; letter-spacing:0.04em;">COMEX · GC=F TERMINAL</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("MARKET SELECTION")
        st.selectbox("Instrument", ["COMEX Gold Futures (GC=F)"], index=0, disabled=True)

        if not DATA_PATH.exists():
            st.error("Processed dataset is missing. Run the data pipeline first.")
            st.stop()

        selected_path = DATA_PATH
        with st.expander("Data pipeline mode", expanded=False):
            raw_mode = st.toggle(
                "Use raw unadjusted data",
                value=False,
                help="Raw mode uses uncleaned Yahoo Finance OHLCV. Processed dataset is recommended for accurate volume filtering.",
            )
            if raw_mode and RAW_DATA_PATH.exists():
                selected_path = RAW_DATA_PATH
            elif raw_mode:
                st.warning("Raw data file unavailable; falling back to processed dataset.")

        try:
            all_data = load_data(str(selected_path), selected_path.stat().st_mtime_ns)
        except Exception as exc:
            st.error(f"Unable to load market data: {exc}")
            st.stop()

        if all_data.empty:
            st.error("The selected dataset contains no valid observations.")
            st.stop()

        st.markdown("---")
        st.caption("DATE RANGE")
        first_day, last_day = all_data.Date.min().date(), all_data.Date.max().date()
        date_value = st.date_input(
            "Observation horizon",
            value=(first_day, last_day),
            min_value=first_day,
            max_value=last_day,
        )
        start_date, end_date = (
            date_value
            if isinstance(date_value, (tuple, list)) and len(date_value) == 2
            else (date_value, date_value)
        )

        st.markdown("---")
        st.caption("PRICE VISUALIZATION")
        chart_type = st.selectbox(
            "Price chart mode",
            ["Area (Gold Glow)", "Candlestick", "Line"],
            index=0,
        )
        show_volume = st.toggle("Include volume panel", value=True)

        st.markdown("---")
        st.caption("DETECTION MODEL (CAUSAL ROBUST Z)")
        with st.expander("Model parameters", expanded=True):
            window = st.slider("Lookback window (sessions)", 10, 120, 30, 5)
            threshold = st.slider("Alert threshold (σ)", 1.5, 5.0, 2.5, 0.25)
            volume_weight = st.slider(
                "Volume score weight",
                0.0,
                1.0,
                0.5,
                0.1,
                help="Weighted contribution of volume Z-score into unified 0–100 anomaly score.",
            )

    # --- Data Calculations ---
    filtered = (
        all_data[all_data.Date.dt.date.between(start_date, end_date)]
        .copy()
        .reset_index(drop=True)
    )
    if filtered.empty:
        st.warning("No observations found for the selected date range.")
        st.stop()

    signals_all = compute_signals(all_data, window, threshold, volume_weight)
    pb_analysis, pb_backtest, analysis_note = load_person_b_outputs(all_data)

    if pb_analysis is not None:
        for col in ["Return_ZScore", "Volume_ZScore"]:
            if col in pb_analysis:
                signals_all[col] = (
                    pd.to_numeric(pb_analysis[col], errors="coerce")
                    .combine_first(signals_all[col])
                )
        signals_all["Return_Alert"] = signals_all.Return_ZScore.abs() >= threshold
        signals_all["Volume_Alert"] = (
            signals_all.Volume_ZScore.abs() >= threshold
        ) & (volume_weight > 0)
        signals_all["Return_Contribution"] = (
            signals_all.Return_ZScore.abs().clip(upper=threshold * 2) / (threshold * 2)
        )
        signals_all["Volume_Contribution"] = (
            signals_all.Volume_ZScore.abs().clip(upper=threshold * 2) / (threshold * 2)
        )
        components = signals_all[["Return_Contribution", "Volume_Contribution"]]
        weights = pd.Series(
            {"Return_Contribution": 1.0, "Volume_Contribution": volume_weight}
        )
        available_weight = components.notna().mul(weights, axis=1).sum(axis=1)
        signals_all["Anomaly_Score"] = (
            components.mul(weights, axis=1)
            .sum(axis=1, min_count=1)
            .div(available_weight.replace(0, np.nan))
            * 100
        )
        signals_all["Is_Anomaly"] = (
            signals_all.Return_Alert | signals_all.Volume_Alert
        )
        signals_all["Event_Type"] = np.select(
            [
                signals_all.Return_ZScore >= threshold,
                signals_all.Return_ZScore <= -threshold,
                signals_all.Volume_Alert & ~signals_all.Return_Alert,
            ],
            ["Upward spike", "Downward shock", "Volume dislocation"],
            default="—",
        )

    signals = signals_all[
        signals_all.Date.dt.date.between(start_date, end_date)
    ].copy()
    filtered = signals.copy()

    # Metrics computation
    last = filtered.iloc[-1]
    previous = all_data[all_data.Date < last.Date].tail(1)
    previous_close = float(previous.Close.iloc[0]) if not previous.empty else np.nan
    session_move = (
        (float(last.Close) / previous_close - 1) * 100
        if pd.notna(previous_close) and previous_close
        else np.nan
    )
    returns = pd.to_numeric(filtered["Daily_Return"], errors="coerce").dropna()
    annualized_vol = (
        returns.std(ddof=1) * np.sqrt(252) * 100 if len(returns) > 1 else np.nan
    )
    valid_sessions = int(filtered.Close.notna().sum())
    flagged = filtered[filtered.Is_Anomaly.fillna(False)]
    move_class = (
        "positive" if session_move >= 0 else "negative" if session_move < 0 else ""
    )
    vol_value = pct(annualized_vol) if len(returns) > 1 else "Unavailable"

    # --- Top Application Header / Market Identity ---
    st.markdown(
        f"""
        <div class="gw-header">
            <div class="gw-brand">
                <div class="gw-logo-icon">GW</div>
                <div>
                    <div class="gw-brand-title">Gold<span class="accent">Watch</span></div>
                    <div class="gw-brand-subtitle">COMEX · GOLD FUTURES · GC=F</div>
                </div>
            </div>
            <div class="gw-market-meta">
                <div class="gw-badge gw-badge-live">
                    <span class="gw-dot"></span> Historical Market Data
                </div>
                <div class="gw-badge">
                    <span>Range:</span> <span class="font-mono">{start_date:%d %b %Y} → {end_date:%d %b %Y}</span>
                </div>
                <div class="gw-badge">
                    <span>Last Observation:</span> <span class="font-mono">{all_data.Date.max():%d %b %Y}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- KPI Financial Strip ---
    st.markdown(
        f"""
        <div class="gw-kpi-strip">
            <div class="gw-kpi-cell">
                <div class="gw-kpi-label">Latest Settlement <span class="font-mono">USD/oz</span></div>
                <div class="gw-kpi-value gold">{money(float(last.Close))}</div>
                <div class="gw-kpi-sub font-mono">COMEX GC=F · {last.Date:%d %b %Y}</div>
            </div>
            <div class="gw-kpi-cell">
                <div class="gw-kpi-label">Session Move <span class="font-mono">1D Δ</span></div>
                <div class="gw-kpi-value {move_class}">{pct(session_move)}</div>
                <div class="gw-kpi-sub font-mono">Prior close {money(previous_close)}</div>
            </div>
            <div class="gw-kpi-cell">
                <div class="gw-kpi-label">Annualized Volatility <span class="font-mono">σ · 252D</span></div>
                <div class="gw-kpi-value">{vol_value}</div>
                <div class="gw-kpi-sub font-mono">Sample: {len(returns):,} valid sessions</div>
            </div>
            <div class="gw-kpi-cell">
                <div class="gw-kpi-label">Unusual Sessions <span class="font-mono">|Z| ≥ {threshold:g}σ</span></div>
                <div class="gw-kpi-value">{len(flagged):,} <span style="font-size:0.95rem; font-weight:400; color:#64748b;">/ {valid_sessions:,}</span></div>
                <div class="gw-kpi-sub font-mono">{len(flagged)/max(1, valid_sessions):.1%} of selected horizon</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Workspace Navigation Tabs ---
    price_tab, signals_tab, backtest_tab, data_tab = st.tabs(
        [
            "Overview & Market Centerpiece",
            "Anomaly Signals & Scoring",
            "Event-Study Backtest",
            "Data Explorer & Export",
        ]
    )

    # =========================================================================
    # TAB 1: OVERVIEW & MARKET CENTERPIECE
    # =========================================================================
    with price_tab:
        has_volume = show_volume and "Volume" in filtered.columns

        # Main Hero Chart Construction
        chart = make_subplots(
            rows=2 if has_volume else 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.035,
            row_heights=[0.75, 0.25] if has_volume else [1],
        )

        hover_price = (
            "<b>%{x|%d %b %Y}</b><br>"
            "Open: $%{open:,.2f}<br>"
            "High: $%{high:,.2f}<br>"
            "Low: $%{low:,.2f}<br>"
            "Close: $%{close:,.2f}<extra></extra>"
        )

        # 1. Primary Gold Price Series (Using Warm Gold)
        if (
            chart_type == "Candlestick"
            and {"Open", "High", "Low"}.issubset(filtered.columns)
        ):
            chart.add_trace(
                go.Candlestick(
                    x=filtered.Date,
                    open=filtered.Open,
                    high=filtered.High,
                    low=filtered.Low,
                    close=filtered.Close,
                    name="COMEX Gold (GC=F)",
                    increasing_line_color="#10b981",
                    decreasing_line_color="#f43f5e",
                    increasing_fillcolor="#10b981",
                    decreasing_fillcolor="#f43f5e",
                    customdata=np.column_stack(
                        [
                            filtered.Daily_Return,
                            filtered.Volume_Observed,
                            filtered.Anomaly_Score,
                        ]
                    ),
                    hovertemplate=(
                        hover_price.replace(
                            "<extra></extra>",
                            "<br>Return: %{customdata[0]:+.2%}<br>Obs. Volume: %{customdata[1]:,.0f}<br>Score: %{customdata[2]:.1f}/100<extra></extra>",
                        )
                    ),
                ),
                row=1,
                col=1,
            )
        else:
            is_area = chart_type.startswith("Area")
            chart.add_trace(
                go.Scatter(
                    x=filtered.Date,
                    y=filtered.Close,
                    mode="lines",
                    name="Gold Close (USD/oz)",
                    line=dict(color="#f5bd4f", width=2.4),
                    fill="tozeroy" if is_area else None,
                    fillcolor="rgba(245, 189, 79, 0.08)" if is_area else None,
                    customdata=np.column_stack(
                        [
                            filtered.Daily_Return,
                            filtered.Volume_Observed,
                            filtered.Anomaly_Score,
                        ]
                    ),
                    hovertemplate=(
                        "<b>%{x|%d %b %Y}</b><br>"
                        "Gold Close: <b>$%{y:,.2f}</b><br>"
                        "Daily Return: %{customdata[0]:+.2%}<br>"
                        "Obs. Volume: %{customdata[1]:,.0f}<br>"
                        "Anomaly Score: %{customdata[2]:.1f}/100<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

        # 2. Anomaly Event Markers on Price Chart
        event_markers = {
            "Upward spike": ("#10b981", "triangle-up", 9),
            "Downward shock": ("#f43f5e", "triangle-down", 9),
            "Volume dislocation": ("#38bdf8", "diamond", 7),
        }
        for event_type, (color, symbol, size) in event_markers.items():
            pts = filtered[filtered.Event_Type == event_type]
            if not pts.empty:
                chart.add_trace(
                    go.Scatter(
                        x=pts.Date,
                        y=pts.Close,
                        mode="markers",
                        name=f"Anomaly: {event_type}",
                        marker=dict(
                            color=color,
                            size=size,
                            symbol=symbol,
                            line=dict(width=1, color="#0b1722"),
                        ),
                        customdata=np.column_stack(
                            [pts.Return_ZScore, pts.Volume_ZScore, pts.Anomaly_Score]
                        ),
                        hovertemplate=(
                            f"<b>{event_type}</b><br>"
                            "%{x|%d %b %Y}<br>"
                            "Price: $%{y:,.2f}<br>"
                            "Return Z: %{customdata[0]:+.2f}σ<br>"
                            "Volume Z: %{customdata[1]:+.2f}σ<br>"
                            "Score: %{customdata[2]:.1f}/100<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )

        # 3. Synchronized Volume Panel
        if has_volume:
            vol_colors = np.where(
                filtered.Close.diff().fillna(0) >= 0,
                "rgba(16, 185, 129, 0.45)",
                "rgba(244, 63, 94, 0.45)",
            )
            chart.add_trace(
                go.Bar(
                    x=filtered.Date,
                    y=filtered.Volume_Observed,
                    marker_color=vol_colors,
                    name="Contract Volume",
                    customdata=filtered.Volume_Data_Status,
                    hovertemplate=(
                        "<b>%{x|%d %b %Y}</b><br>"
                        "Volume: %{y:,.0f} contracts<br>"
                        "Status: %{customdata}<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )

            # Mark reported-zero sessions distinctly
            missing_vol = filtered[filtered.Volume_Observed.isna()]
            if not missing_vol.empty:
                chart.add_trace(
                    go.Scatter(
                        x=missing_vol.Date,
                        y=np.zeros(len(missing_vol)),
                        mode="markers",
                        name="Volume Unavailable (0)",
                        marker=dict(color="#475569", size=4, symbol="x"),
                        hovertemplate=(
                            "<b>%{x|%d %b %Y}</b><br>"
                            "Volume: Source reports 0 (omitted from scoring)<extra></extra>"
                        ),
                    ),
                    row=2,
                    col=1,
                )

        chart.update_layout(
            height=510 if has_volume else 420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
            margin=dict(l=8, r=16, t=20, b=8),
            hovermode="x unified",
            xaxis_rangeslider_visible=False,
            hoverlabel=dict(
                bgcolor="#0b1722",
                bordercolor="#223c52",
                font=dict(family="JetBrains Mono, monospace", size=11, color="#f1f5f9"),
            ),
            legend=dict(
                orientation="h",
                y=1.06,
                x=0,
                bgcolor="rgba(11,23,34,0.75)",
                bordercolor="#162838",
                borderwidth=1,
            ),
        )
        chart.update_xaxes(
            showgrid=True,
            gridcolor="#132433",
            linecolor="#1a2e40",
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color="#64748b"),
        )
        chart.update_yaxes(
            showgrid=True,
            gridcolor="#132433",
            linecolor="#1a2e40",
            title_text="Price (USD / oz)",
            title_font=dict(size=10, color="#64748b"),
            tickprefix="$",
            tickformat=",.0f",
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color="#64748b"),
            row=1,
            col=1,
        )
        if has_volume:
            chart.update_yaxes(
                showgrid=True,
                gridcolor="#132433",
                linecolor="#1a2e40",
                title_text="Volume",
                title_font=dict(size=10, color="#64748b"),
                tickformat="~s",
                tickfont=dict(
                    family="JetBrains Mono, monospace", size=9, color="#64748b"
                ),
                row=2,
                col=1,
            )

        # Render Main Chart Panel Header
        st.markdown(
            """
            <div class="gw-panel-header">
                <div class="gw-panel-title">
                    COMEX Gold Futures · Primary Market Centerpiece
                </div>
                <div class="gw-panel-badge font-mono">
                    HOVER FOR SESSION DETAIL · DRAG TO ZOOM
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            chart,
            width="stretch",
            theme=None,
            config={"displaylogo": False, "scrollZoom": True},
        )

        # Supporting Analytics: Market Pulse & Return Dynamics
        c_left, c_right = st.columns([1.1, 1.9], gap="medium")

        with c_left:
            period_return = (
                (float(filtered.Close.iloc[-1]) / float(filtered.Close.iloc[0]) - 1)
                * 100
                if len(filtered) > 1 and float(filtered.Close.iloc[0]) != 0
                else np.nan
            )
            peak_val = filtered.Close.max()
            trough_val = filtered.Close.min()
            obs_volume_mean = (
                filtered.Volume_Observed.dropna().mean()
                if "Volume_Observed" in filtered
                else np.nan
            )

            st.markdown(
                f"""
                <div class="gw-panel" style="height: 100%;">
                    <div class="gw-panel-header">
                        <div class="gw-panel-title">Horizon Performance Pulse</div>
                        <div class="gw-panel-badge font-mono">STATISTICS</div>
                    </div>
                    <div style="display:flex; flex-direction:column; gap:0.75rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center; padding:0.4rem 0; border-bottom:1px solid #132433;">
                            <span style="font-size:0.8rem; color:#94a3b8;">Period Return</span>
                            <span class="font-mono" style="font-weight:700; color:{'#10b981' if period_return>=0 else '#f43f5e'}; font-size:1rem;">{pct(period_return)}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center; padding:0.4rem 0; border-bottom:1px solid #132433;">
                            <span style="font-size:0.8rem; color:#94a3b8;">Horizon Peak</span>
                            <span class="font-mono" style="font-weight:600; color:#f1f5f9;">{money(peak_val)}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center; padding:0.4rem 0; border-bottom:1px solid #132433;">
                            <span style="font-size:0.8rem; color:#94a3b8;">Horizon Trough</span>
                            <span class="font-mono" style="font-weight:600; color:#f1f5f9;">{money(trough_val)}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center; padding:0.4rem 0; border-bottom:1px solid #132433;">
                            <span style="font-size:0.8rem; color:#94a3b8;">Mean Daily Volume</span>
                            <span class="font-mono" style="font-weight:600; color:#38bdf8;">{f'{obs_volume_mean:,.0f} contracts' if pd.notna(obs_volume_mean) else '—'}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center; padding:0.4rem 0;">
                            <span style="font-size:0.8rem; color:#94a3b8;">Flagged Anomaly Rate</span>
                            <span class="font-mono" style="font-weight:600; color:#f5bd4f;">{len(flagged):,} sessions ({len(flagged)/max(1, valid_sessions):.1%})</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c_right:
            st.markdown(
                """
                <div class="gw-panel" style="height: 100%;">
                    <div class="gw-panel-header">
                        <div class="gw-panel-title">Session Return Distribution</div>
                        <div class="gw-panel-badge font-mono">DENSITY & FREQUENCY</div>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            hist_fig = create_dark_chart(height=240)
            hist_fig.add_trace(
                go.Histogram(
                    x=returns * 100,
                    nbinsx=40,
                    marker=dict(
                        color="#38bdf8",
                        line=dict(color="#10232d", width=0.8),
                    ),
                    opacity=0.85,
                    name="Daily Return %",
                    hovertemplate="Return Bin: %{x:.2f}%<br>Sessions: %{y}<extra></extra>",
                )
            )
            hist_fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                bargap=0.08,
            )
            hist_fig.update_xaxes(title_text="Daily Return (%)", ticksuffix="%")
            hist_fig.update_yaxes(title_text="Trading Sessions")
            st.plotly_chart(hist_fig, width="stretch", theme=None, config={"displaylogo": False})
            st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: ANOMALY SIGNALS & SCORING
    # =========================================================================
    with signals_tab:
        st.markdown(
            f"""
            <div class="gw-alert-box info">
                <b>Signal Model Status:</b> {analysis_note}. Absolute Z-score threshold set to <b>|Z| ≥ {threshold:g}σ</b>. Volume values reported as zero are treated as unobserved and omitted from volume anomaly scoring.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Dual Synchronized Z-Score Charts
        col_rz, col_vz = st.columns(2, gap="medium")
        with col_rz:
            st.markdown(
                """
                <div class="gw-panel-header">
                    <div class="gw-panel-title">Return Z-Score (σ)</div>
                    <div class="gw-panel-badge font-mono">ROBUST MEDIAN/MAD</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            rz_fig = create_dark_chart(height=260)
            rz_fig.add_trace(
                go.Scatter(
                    x=filtered.Date,
                    y=filtered.Return_ZScore,
                    mode="lines",
                    name="Return Z",
                    line=dict(color="#f5bd4f", width=1.6),
                    hovertemplate="%{x|%d %b %Y}<br>Return Z: %{y:+.2f}σ<extra></extra>",
                )
            )
            rz_fig.add_hline(
                y=threshold,
                line_dash="dot",
                line_color="#f43f5e",
                annotation_text=f"+{threshold:g}σ",
                annotation_position="top right",
            )
            rz_fig.add_hline(
                y=-threshold,
                line_dash="dot",
                line_color="#f43f5e",
                annotation_text=f"-{threshold:g}σ",
                annotation_position="bottom right",
            )
            rz_fig.update_yaxes(title_text="Return Z (σ)", tickformat="+.1f")
            st.plotly_chart(rz_fig, width="stretch", theme=None, config={"displaylogo": False})

        with col_vz:
            st.markdown(
                """
                <div class="gw-panel-header">
                    <div class="gw-panel-title">Volume Z-Score (σ)</div>
                    <div class="gw-panel-badge font-mono">LOG-TRANSFORMED MAD</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            vz_fig = create_dark_chart(height=260)
            vz_fig.add_trace(
                go.Scatter(
                    x=filtered.Date,
                    y=filtered.Volume_ZScore,
                    mode="lines",
                    name="Volume Z",
                    line=dict(color="#38bdf8", width=1.6),
                    connectgaps=False,
                    hovertemplate="%{x|%d %b %Y}<br>Volume Z: %{y:+.2f}σ<extra></extra>",
                )
            )
            vz_fig.add_hline(
                y=threshold,
                line_dash="dot",
                line_color="#f43f5e",
                annotation_text=f"+{threshold:g}σ",
                annotation_position="top right",
            )
            vz_fig.add_hline(
                y=-threshold,
                line_dash="dot",
                line_color="#f43f5e",
                annotation_text=f"-{threshold:g}σ",
                annotation_position="bottom right",
            )
            vz_fig.update_yaxes(title_text="Volume Z (σ)", tickformat="+.1f")
            st.plotly_chart(vz_fig, width="stretch", theme=None, config={"displaylogo": False})

        # Ranked Anomaly Events Explorer
        events = filtered[filtered.Is_Anomaly.fillna(False)].copy()
        if events.empty:
            st.info(
                f"No qualifying anomaly sessions found at threshold |Z| ≥ {threshold:g}σ. Reduce the threshold slider in the sidebar to inspect milder events."
            )
        else:
            events["Why flagged"] = [
                build_event_reasons(row, threshold) for _, row in events.iterrows()
            ]
            events["Price"] = events.Close

            st.markdown(
                f"""
                <div class="gw-panel-header" style="margin-top:1.25rem;">
                    <div class="gw-panel-title">Identified Market Anomalies ({len(events):,} Sessions)</div>
                    <div class="gw-panel-badge font-mono">RANKED BY ANOMALY SEVERITY</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Filtering & sorting controls
            fc1, fc2, fc3 = st.columns([2, 1.5, 1.2])
            with fc1:
                available_types = sorted(events.Event_Type.dropna().unique().tolist())
                chosen_types = st.multiselect(
                    "Filter event types", available_types, default=available_types
                )
            with fc2:
                sort_col = st.selectbox(
                    "Sort events by",
                    ["Anomaly_Score", "Date", "Return_ZScore", "Volume_ZScore"],
                    index=0,
                    format_func=lambda x: {
                        "Anomaly_Score": "Anomaly Score (0–100)",
                        "Date": "Date",
                        "Return_ZScore": "Return Z-score",
                        "Volume_ZScore": "Volume Z-score",
                    }[x],
                )
            with fc3:
                per_page = st.selectbox("Page size", [25, 50, 100], index=0)

            filtered_events = events[events.Event_Type.isin(chosen_types)]
            ascending = sort_col == "Date"
            display_events = filtered_events.sort_values(sort_col, ascending=ascending)

            pages = max(1, int(np.ceil(len(display_events) / per_page)))
            page = st.number_input(
                f"Page (1 to {pages})",
                min_value=1,
                max_value=pages,
                value=1,
                step=1,
                key="event_tbl_page",
            )

            order = [
                "Date",
                "Price",
                "Event_Type",
                "Anomaly_Score",
                "Return_ZScore",
                "Volume_ZScore",
                "Why flagged",
            ]
            page_slice = display_events[order].iloc[
                (page - 1) * per_page : page * per_page
            ]

            st.dataframe(
                page_slice,
                width="stretch",
                hide_index=True,
                column_config={
                    "Date": st.column_config.DateColumn("Session Date"),
                    "Price": st.column_config.NumberColumn(
                        "Close (USD/oz)", format="$%.2f"
                    ),
                    "Event_Type": st.column_config.TextColumn("Classification"),
                    "Anomaly_Score": st.column_config.NumberColumn(
                        "Score (0–100)", format="%.1f"
                    ),
                    "Return_ZScore": st.column_config.NumberColumn(
                        "Return Z (σ)", format="%+.2f"
                    ),
                    "Volume_ZScore": st.column_config.NumberColumn(
                        "Volume Z (σ)", format="%+.2f"
                    ),
                    "Why flagged": st.column_config.TextColumn("Trigger Condition"),
                },
            )

            # Score Composition Visualization
            st.markdown(
                """
                <div style="margin-top: 1.25rem;">
                    <div style="font-size:0.85rem; font-weight:600; color:#f1f5f9; margin-bottom:0.25rem;">Score Composition (Capped Return vs Volume Contributions)</div>
                    <div style="font-size:0.75rem; color:#64748b; margin-bottom:0.75rem;">Return weight = 1.0; Volume weight = configured setting. Normalized to 0–100 scale.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            comp = filtered_events[
                ["Date", "Return_Contribution", "Volume_Contribution"]
            ].sort_values("Date")
            cf = create_dark_chart(height=230)
            cf.add_trace(
                go.Bar(
                    x=comp.Date,
                    y=comp.Return_Contribution * 100,
                    name="Return Contribution",
                    marker_color="#f5bd4f",
                )
            )
            cf.add_trace(
                go.Bar(
                    x=comp.Date,
                    y=comp.Volume_Contribution * 100,
                    name="Volume Contribution",
                    marker_color="#38bdf8",
                )
            )
            cf.update_layout(barmode="group", yaxis_title="Normalized Contribution (%)")
            st.plotly_chart(cf, width="stretch", theme=None, config={"displaylogo": False})

    # =========================================================================
    # TAB 3: EVENT-STUDY BACKTEST
    # =========================================================================
    with backtest_tab:
        st.markdown(
            """
            <div class="gw-panel-header" style="margin-bottom: 0.75rem;">
                <div class="gw-panel-title">Spike Reversal Event Study & Post-Alert Outcomes</div>
                <div class="gw-panel-badge font-mono">PERSON B RESEARCH INTEGRATION</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if pb_backtest is None:
            st.markdown(
                f"""
                <div class="gw-alert-box warning">
                    <div style="font-weight:600; color:#f59e0b; margin-bottom:0.35rem;">Backtest Results File Not Found</div>
                    <div>Person B's backtest dataset is not currently detected in <code>data/processed/gold_futures_backtest.csv</code> or <code>app_data/analysis.csv</code>. To maintain mathematical integrity, provisional candidate counts are not presented as completed backtest outcomes.</div>
                </div>
                <div class="gw-panel">
                    <div class="gw-panel-title" style="margin-bottom:0.5rem;">Expected Integration Schema</div>
                    <div style="font-size:0.82rem; color:#94a3b8; line-height:1.5;">
                        Supply a CSV artifact with the following verified columns:<br>
                        • <code>Signal_Date</code> or <code>Date</code>: Trading session of the trigger.<br>
                        • <code>Forward_Return</code> or <code>Forward_Return_Pct</code>: Realized return over the evaluation horizon.<br>
                        • <code>Evaluation_Horizon</code> (optional): Number of holding sessions (e.g. 5, 10, 20).<br>
                        • <code>Baseline_Decline_Rate</code> (optional): Unconditional market decline rate for benchmark.<br>
                    </div>
                    <div style="margin-top:0.9rem; font-size:0.75rem; font-family:'JetBrains Mono',monospace; color:#64748b;">
                        Current workspace status: {len(flagged):,} potential spike signals identified in the selected date range.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            bt = pb_backtest.copy()
            date_col = (
                "Signal_Date"
                if "Signal_Date" in bt
                else "Date"
                if "Date" in bt
                else None
            )
            forward_col = next(
                (
                    c
                    for c in ["Forward_Return", "Forward_Return_Pct", "Outcome_Return"]
                    if c in bt
                ),
                None,
            )
            if date_col is None or forward_col is None:
                st.error(
                    "Backtest file found, but required signal date and forward return columns are missing."
                )
            else:
                bt[date_col] = pd.to_datetime(bt[date_col], errors="coerce")
                bt[forward_col] = pd.to_numeric(bt[forward_col], errors="coerce")
                if forward_col == "Forward_Return_Pct":
                    bt[forward_col] = bt[forward_col] / 100
                bt = bt[bt[date_col].dt.date.between(start_date, end_date)].dropna(
                    subset=[date_col, forward_col]
                )

                if bt.empty:
                    st.info(
                        "No completed forward outcomes remain in the selected date range."
                    )
                else:
                    decline_threshold = st.slider(
                        "Decline hurdle threshold (%)",
                        0.0,
                        5.0,
                        0.0,
                        0.25,
                        format="%.2f%%",
                        help="Hurdle below which a forward return is classified as a realized decline.",
                    )
                    declines = bt[forward_col] < -(decline_threshold / 100)
                    horizon_col = next(
                        (
                            c
                            for c in [
                                "Evaluation_Horizon",
                                "Horizon_Sessions",
                                "Forward_Horizon",
                            ]
                            if c in bt
                        ),
                        None,
                    )
                    horizon_value = (
                        bt[horizon_col].dropna().iloc[0]
                        if horizon_col and bt[horizon_col].notna().any()
                        else None
                    )
                    horizon_note = (
                        f"{horizon_value} sessions"
                        if horizon_value is not None
                        else "configured horizon"
                    )

                    baseline_col = next(
                        (
                            c
                            for c in ["Baseline_Decline_Rate", "BaselineDeclineRate"]
                            if c in bt
                        ),
                        None,
                    )
                    baseline = (
                        pd.to_numeric(bt[baseline_col], errors="coerce").dropna().mean()
                        * 100
                        if baseline_col
                        else np.nan
                    )
                    if pd.isna(baseline) and horizon_value is not None:
                        horizon_sessions = pd.to_numeric(
                            pd.Series([horizon_value]), errors="coerce"
                        ).iloc[0]
                        if pd.notna(horizon_sessions) and horizon_sessions > 0:
                            horizon_sessions = int(horizon_sessions)
                            market_forward = (
                                filtered.Close.shift(-horizon_sessions) / filtered.Close
                                - 1
                            ).dropna()
                            if not market_forward.empty:
                                baseline = (
                                    market_forward < -(decline_threshold / 100)
                                ).mean() * 100

                    # Backtest KPI Strip
                    st.markdown(
                        f"""
                        <div class="gw-kpi-strip" style="grid-template-columns: 1fr 1fr 1fr; margin-bottom:1.25rem;">
                            <div class="gw-kpi-cell">
                                <div class="gw-kpi-label">Qualifying Spikes</div>
                                <div class="gw-kpi-value gold">{len(bt):,}</div>
                                <div class="gw-kpi-sub font-mono">Completed outcomes</div>
                            </div>
                            <div class="gw-kpi-cell">
                                <div class="gw-kpi-label">Post-Alert Decline Rate</div>
                                <div class="gw-kpi-value {'positive' if declines.mean()>=0.5 else 'negative'}">{pct(declines.mean()*100)}</div>
                                <div class="gw-kpi-sub font-mono">Horizon: {horizon_note}</div>
                            </div>
                            <div class="gw-kpi-cell">
                                <div class="gw-kpi-label">Market Baseline Decline Rate</div>
                                <div class="gw-kpi-value">{pct(baseline) if pd.notna(baseline) else '—'}</div>
                                <div class="gw-kpi-sub font-mono">Unconditional benchmark</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Forward Return Histogram
                    st.markdown(
                        """
                        <div class="gw-panel-header">
                            <div class="gw-panel-title">Forward Return Distribution Following Spikes</div>
                            <div class="gw-panel-badge font-mono">OUTCOME DENSITY</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    bf = create_dark_chart(height=260)
                    bf.add_trace(
                        go.Histogram(
                            x=bt[forward_col] * 100,
                            nbinsx=24,
                            marker_color="#14b8a6",
                            name="Forward Return %",
                        )
                    )
                    bf.add_vline(
                        x=-(decline_threshold),
                        line_dash="dot",
                        line_color="#f43f5e",
                        annotation_text=f"Decline hurdle -{decline_threshold:.2f}%",
                    )
                    bf.add_vline(x=0, line_dash="solid", line_color="#475569")
                    bf.update_xaxes(title_text="Forward Return (%)", ticksuffix="%")
                    bf.update_yaxes(title_text="Event Count")
                    st.plotly_chart(bf, width="stretch", theme=None, config={"displaylogo": False})
                    st.dataframe(
                        bt.sort_values(date_col, ascending=False),
                        width="stretch",
                        hide_index=True,
                    )

    # =========================================================================
    # TAB 4: DATA EXPLORER & EXPORT
    # =========================================================================
    with data_tab:
        st.markdown(
            f"""
            <div class="gw-panel-header" style="margin-bottom:0.75rem;">
                <div class="gw-panel-title">Gold Futures Market Series & Engineered Signals</div>
                <div class="gw-panel-badge font-mono">DATASET EXPLORER · {len(filtered):,} SESSIONS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Missing values and completeness quality check
        missing_counts = filtered.isna().sum()
        missing_table = pd.DataFrame(
            {
                "Field": missing_counts.index,
                "Unavailable Sessions": missing_counts.values,
            }
        )
        missing_table = missing_table[missing_table["Unavailable Sessions"] > 0]
        if not missing_table.empty:
            with st.expander("Data completeness & audit details", expanded=False):
                st.dataframe(missing_table, width="stretch", hide_index=True)
                st.caption(
                    "Note: Initial rolling lookback sessions lack prior lookback history; zero-volume records in early historical GC=F data are treated as unobserved."
                )

        # Table configuration & filtering
        tc1, tc2 = st.columns([2.5, 1.5])
        numeric_cols = filtered.select_dtypes(include="number").columns.tolist()
        with tc1:
            chosen_cols = st.multiselect(
                "Select table columns",
                options=["Date", *numeric_cols],
                default=[
                    c
                    for c in [
                        "Date",
                        "Close",
                        "Daily_Return",
                        "Volume_Observed",
                        "Return_ZScore",
                        "Volume_ZScore",
                        "Anomaly_Score",
                    ]
                    if c in ["Date", *numeric_cols]
                ],
            )
        with tc2:
            tbl_sort = st.selectbox(
                "Sort table by",
                ["Newest First", "Oldest First", "Highest Anomaly Score"],
                index=0,
            )

        tbl = filtered[chosen_cols].copy()
        if tbl_sort == "Newest First":
            tbl = tbl.sort_values("Date", ascending=False)
        elif tbl_sort == "Oldest First":
            tbl = tbl.sort_values("Date", ascending=True)
        elif "Anomaly_Score" in tbl:
            tbl = tbl.sort_values("Anomaly_Score", ascending=False)

        col_labels = {
            "Date": "Trading Date",
            "Close": "Close (USD/oz)",
            "Open": "Open (USD/oz)",
            "High": "High (USD/oz)",
            "Low": "Low (USD/oz)",
            "Volume": "Source Volume",
            "Volume_Observed": "Observed Volume",
            "Daily_Return": "Daily Return (%)",
            "Return_ZScore": "Return Z (σ)",
            "Volume_ZScore": "Volume Z (σ)",
            "Anomaly_Score": "Anomaly Score",
            "Event_Type": "Classification",
        }
        col_formats = {
            "Date": st.column_config.DateColumn("Trading Date"),
            "Close": st.column_config.NumberColumn("Close (USD/oz)", format="$%.2f"),
            "Open": st.column_config.NumberColumn("Open (USD/oz)", format="$%.2f"),
            "High": st.column_config.NumberColumn("High (USD/oz)", format="$%.2f"),
            "Low": st.column_config.NumberColumn("Low (USD/oz)", format="$%.2f"),
            "Volume_Observed": st.column_config.NumberColumn(
                "Observed Volume", format="%.0f"
            ),
            "Daily_Return": st.column_config.NumberColumn(
                "Daily Return (%)", format="%+.2f%%"
            ),
            "Return_ZScore": st.column_config.NumberColumn(
                "Return Z (σ)", format="%+.2f"
            ),
            "Volume_ZScore": st.column_config.NumberColumn(
                "Volume Z (σ)", format="%+.2f"
            ),
            "Anomaly_Score": st.column_config.NumberColumn(
                "Anomaly Score", format="%.1f"
            ),
        }

        row_limit = st.selectbox("Rows per page", [25, 50, 100, 250], index=1)
        total_pages = max(1, int(np.ceil(len(tbl) / row_limit)))
        tbl_page = st.number_input(
            f"Page (1 to {total_pages})",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            key="data_tbl_pg",
        )

        # Prepare display copy with Daily Return expressed in %
        display_tbl = tbl.copy()
        if "Daily_Return" in display_tbl:
            display_tbl["Daily_Return"] = display_tbl["Daily_Return"] * 100

        st.dataframe(
            display_tbl.iloc[(tbl_page - 1) * row_limit : tbl_page * row_limit].rename(
                columns=col_labels
            ),
            width="stretch",
            hide_index=True,
            column_config={
                col_labels.get(c, c): v for c, v in col_formats.items()
            },
        )

        # CSV Download Action (Exports exact underlying unscaled decimals)
        csv_bytes = tbl.to_csv(index=False, na_rep="—").encode("utf-8")
        export_mode = (
            "raw" if selected_path == RAW_DATA_PATH else "processed_signals"
        )
        st.download_button(
            label="Download Selected Dataset (CSV)",
            data=csv_bytes,
            file_name=f"goldwatch_{export_mode}_{start_date}_{end_date}.csv",
            mime="text/csv",
        )
        st.caption(
            f"Export contains {len(tbl):,} sessions · {len(chosen_cols)} columns · {export_mode} mode. Missing/unobserved values exported as '—'."
        )


if __name__ == "__main__":
    main()
