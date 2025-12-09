# streamlit run hReview_Summary.py
#git add hReview_Summary.py
#git commit -m "Update investor summary UI"
#git push
# hReview_Summary.py — Investor Dashboard (Live)
# Sections: Header (clock + market chip) → 5 KPIs → 3 Dials → Traffic Lights → Live Positions → (optional) Closed Trades

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta, timezone
from textwrap import dedent
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from zoneinfo import ZoneInfo

from alpaca_trade_api.rest import REST

st.cache_data.clear()

# Prefer local Clock/ folder, then project root, then /mnt/data (for notebook runs)
_CLOCK_JS_CANDIDATES = [
    "Clock/quantml_clock_embed.js",
    "quantml_clock_embed.js",
    "/mnt/data/quantml_clock_embed.js",
]
_SINGLEFILE_CLOCK_HTML = [
    "Clock/quantml_logo_clock_singlefile.html",
    "quantml_logo_clock_singlefile.html",
    "/mnt/data/quantml_logo_clock_singlefile.html",
]

# ── SPY vs QuantML view config ────────────────────────────────────────────────
START_1M_FROM = pd.Timestamp("2025-09-20").date()   # anchor start for 1M view

# =============================================================================
# Page + Branding
# =============================================================================
st.set_page_config(page_title="Investor Summary — Live", layout="wide", page_icon="💼")

PLOTLY_SEQ = px.colors.qualitative.Safe
BRAND = {
    "bg": "#0B1220",
    "panel": "#10172A",
    "text": "#111827",
    "primary": "#4F46E5",
    "accent": "#10B981",
    "success": "#16A34A",
    "warning": "#EAB308",
    "danger":  "#DC2626",
}
# Global Plotly viewer config (used by all charts)
PLOTLY_CONFIG = {
    "displayModeBar": False,   # hide toolbar
    "responsive": True,
    "scrollZoom": False,
    "doubleClick": "reset",    # double-click to reset axes
    # You can add export options if you like:
    # "toImageButtonOptions": {"format": "png", "filename": "quantml_chart", "scale": 2},
}

# --- Traffic Light thresholds (percent points, not decimals) ---
TL_THRESH_PCT = 0.10   # ±0.10% band -> amber
TL_GREEN = BRAND["success"]
TL_AMBER = BRAND["warning"]
TL_RED   = BRAND["danger"]

# hReview_Summary.py — Sentiment & News panel

from alpaca_trade_api.rest import REST, TimeFrame
import config as CFG
from datetime import datetime, timedelta, timezone

def _map_benchmark_symbol_for_alpaca(symbol: str) -> str:
    s = str(symbol).strip()
    m = getattr(CFG, "BENCHMARK_YAHOO_TO_ALPACA", {})
    return m.get(s, s)  # default: unchanged


def _get_benchmark_bars_alpaca(api: REST, symbol: str, timeframe: str, *, days: int):
    """
    Fetch SPY / benchmark bars from Alpaca only.
    timeframe: "1D" or "1Min" (mapped to TimeFrame.Day / TimeFrame.Minute).
    Returns DataFrame with columns: ts (UTC datetime), close (float).
    """
    if api is None:
        return pd.DataFrame(columns=["ts", "close"])

    alpaca_sym = _map_benchmark_symbol_for_alpaca(symbol)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 3)

    # Map your string timeframe to Alpaca TimeFrame
    if timeframe.lower() in ("1d", "day", "1day", "daily"):
        tf = TimeFrame.Day
    elif timeframe.lower() in ("1min", "minute", "1m"):
        tf = TimeFrame.Minute
    else:
        # default to daily
        tf = TimeFrame.Day

    bars = api.get_bars(
        alpaca_sym,
        tf,
        start=start.isoformat(),
        end=end.isoformat(),
        limit=None,
    )

    if bars is None or len(bars) == 0:
        return pd.DataFrame(columns=["ts", "close"])

    # Alpaca v2/pydantic objects → DataFrame
    try:
        df = bars.df  # new SDK / DataFrame property
    except Exception:
        df = pd.DataFrame([b._raw for b in bars])

    # normalise columns: index or 't' / 'timestamp'
    if not df.empty:
        if "timestamp" in df.columns:
            ts_raw = df["timestamp"]
        elif "t" in df.columns:
            ts_raw = df["t"]
        else:
            ts_raw = df.index

        ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
        # close column may be 'c' or 'close'
        if "close" in df.columns:
            close = df["close"]
        elif "c" in df.columns:
            close = df["c"]
        else:
            close = pd.NA

        out = pd.DataFrame({"ts": ts, "close": close})
        out = out.dropna(subset=["ts", "close"])
        out = out.sort_values("ts")
        return out

    return pd.DataFrame(columns=["ts", "close"])


def _load_features_for(symbol: str) -> pd.DataFrame:
    """
    Load the *_test_features.csv for a ticker from 2.Features_test (latest schema).
    """
    base = Path("2.Features_test")
    candidates = [
        base / f"{symbol}_test_features.csv",
        base / f"{symbol}_test_features_with_predictions.csv",
        base / f"{symbol}_with_predictions.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                # normalize date
                dt = "date" if "date" in df.columns else ("Date" if "Date" in df.columns else None)
                if dt:
                    df[dt] = pd.to_datetime(df[dt], errors="coerce")
                return df
            except Exception:
                pass
    return pd.DataFrame()

def render_sentiment_news_panel(positions_df: pd.DataFrame | None = None):
    st.subheader("📰 Sentiment & News")

    # --- Build list of symbols from *open positions* -------------------------
    open_syms: list[str] = []
    if positions_df is not None and not positions_df.empty:
        # Reuse the same helper you already use elsewhere (positions panel)
        sym_col = _first_existing_col(positions_df, "Ticker", "symbol", "Asset", "Symbol", "asset")
        if sym_col is not None:
            open_syms = (
                positions_df[sym_col]
                .dropna()
                .astype(str)
                .str.upper()
                .unique()
                .tolist()
            )
            open_syms = sorted(open_syms)

    # Decide options + build a nice compact dropdown on the left
    if open_syms:
        ticker_options = open_syms
        helper_text = None
    else:
        ticker_options = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
        helper_text = "No open positions detected — showing default watchlist."

    sel_col, _ = st.columns([0.30, 0.70])  # 30% width for the dropdown
    with sel_col:
        st.caption("Ticker")
        if helper_text:
            st.caption(f"_{helper_text}_")
        symbol = st.selectbox(
            "Ticker",                 # internal label (hidden)
            options=ticker_options,
            label_visibility="collapsed",
            key="sentiment_news_ticker",
        )

    # --- Load per-ticker features -------------------------------------------
    df = _load_features_for(symbol)
    if df.empty:
        st.info("No features file found for this ticker in 2.Features_test.")
        return

    # pick last ~90 rows
    dtcol = "date" if "date" in df.columns else ("Date" if "Date" in df.columns else None)
    if not dtcol:
        st.info("No date column in features file.")
        return
    df = df.sort_values(dtcol).tail(90).reset_index(drop=True)

    # compute compact view series
    m3  = pd.to_numeric(df.get("sent_mean_3d"), errors="coerce")
    m7  = pd.to_numeric(df.get("sent_mean_7d"), errors="coerce")
    v7  = pd.to_numeric(df.get("sent_vol_7d"),  errors="coerce")
    tr  = pd.to_numeric(df.get("sent_trend_3v7"), errors="coerce")
    e0  = pd.to_numeric(df.get("ev_earnings_d0"), errors="coerce").fillna(0).astype(int)
    en  = pd.to_numeric(df.get("ev_earnings_in_3d"), errors="coerce").fillna(0).astype(int)

    c1, c2 = st.columns([0.65, 0.35])
    with c1:
        line_df = pd.DataFrame({
            dtcol: df[dtcol],
            "sent_mean_3d": m3,
            "sent_mean_7d": m7,
            "sent_trend_3v7": tr,
        })
        st.line_chart(line_df, x=dtcol, height=260)
        st.caption("Rolling sentiment (3d/7d) and short-vs-long trend.")

    with c2:
        bar_df = pd.DataFrame({
            dtcol: df[dtcol],
            "sent_vol_7d": v7,
        })
        st.bar_chart(bar_df, x=dtcol, height=260)
        st.caption("News flow (7-day headline volume).")

    # event badges
    has_evt = (e0.sum() + en.sum()) > 0
    if has_evt:
        st.markdown(
            "**Upcoming earnings window (within 3 business days)**"
            if en.any() else "**No near-term earnings**"
        )
        st.dataframe(
            df[[dtcol, "ev_earnings_d0", "ev_earnings_in_3d", "ev_dividend_d0", "ev_split_d0"]]
            .tail(15)
            .style.format({}),
            width="stretch",
            hide_index=True,
        )


def apply_branding() -> None:
    px.defaults.template = "plotly_white"
    px.defaults.color_discrete_sequence = PLOTLY_SEQ

    st.markdown("""
    <style>
      .block-container{
        padding-top:1.0rem;     /* was 2.4rem */
        padding-bottom:1.2rem;  /* was 2rem   */
        max-width:1750px;
      }
      @media (min-width: 1900px){ .block-container{ max-width:2000px; } }

      /* Reduce top/bottom spacing between header and the next section */
      .header-wrap { margin-bottom: 4px; }             /* wrapper we use in render_header */
      .ticker-wrap { margin: 6px 0 10px 0; }           /* ribbon */
      .stSubheader, h2, h3 { margin-top: 0.4rem; }     /* Streamlit heading tweaks */

      /* Left-align st.dataframe headers + cells */
      div[data-testid="stDataFrame"] [role="columnheader"],
      div[data-testid="stDataFrame"] [role="gridcell"]{
        justify-content:flex-start !important;
        text-align:left !important;
      }

      /* KPI cards */
      .kpi-card{ padding:8px 10px; border:1px solid #E5E7EB; border-radius:8px; }
      .kpi-title{ font-weight:600; font-size:.9rem; color:#6B7280; margin-bottom:2px; }
      .kpi-value{ font-weight:700; font-size:1.4rem; line-height:1.25; }
      .kpi-caption{ font-size:.85rem; color:#64748B; margin-top:2px; }
      .kpi-value.positive{ color:#16A34A; }
      .kpi-value.negative{ color:#DC2626; }

      /* Small OPEN/CLOSED chip */
      .chip{display:inline-flex;align-items:center;gap:8px;padding:6px 10px;
            border-radius:16px;background:#0E1B3A15;border:1px solid #0E1B3A44;}
      .dot{width:12px;height:12px;border-radius:50%;display:inline-block;}
      .label{font-weight:600;font-size:0.9rem;color:#0E1B3A;}
      /* === Confidence pills (compact SL/TP cards) === */
      .qml-pill{border:1px solid #e5e7eb;border-left:6px solid #16a34a;border-radius:10px;padding:10px 12px;margin:6px 0;background:#fff}
      .qml-pill.warn{border-left-color:#f59e0b}
      .qml-pill.danger{border-left-color:#dc2626}
      .qml-pill .hdr{font-weight:700;margin-bottom:4px}
      .qml-pill .line{font-size:.92rem;color:#374151}
      .qml-pill .mini{height:8px;border-radius:6px;background:#f3f4f6;overflow:hidden;margin-top:6px;position:relative}
      .qml-pill .sl{position:absolute;left:0;top:0;height:100%;background:#ef4444}
      .qml-pill .tp{position:absolute;right:0;top:0;height:100%;background:#16a34a}
      /* micro kpis under confidence pill */
      .qml-mini { display:flex; gap:14px; margin:6px 2px 2px 2px; flex-wrap:wrap; }
      .qml-mini .itm { font-size:.92rem; color:#4b5563; }
      .qml-mini .lbl { font-weight:600; color:#111827; margin-right:6px; }
      .qml-rr.good  { color:#16a34a; font-weight:700; }
      .qml-rr.warn  { color:#f59e0b; font-weight:700; }
      .qml-rr.bad   { color:#dc2626; font-weight:700; }
      /* small note under each confidence pill */
      .qml-note { font-size:.85rem; color:#64748B; margin:4px 2px 0; line-height:1.25; }
      .qml-note .em { color:#111827; font-weight:600; }
      /* mini chart legend chips */
      .qml-chip {display:inline-flex;align-items:center;gap:6px;padding:2px 8px;border:1px solid #e5e7eb;border-radius:999px;font-size:.85rem;color:#374151;background:#fff}
      .qml-chip .sw {width:10px;height:10px;border-radius:2px;display:inline-block}
      .qml-chip.good {border-color:#16a34a22}
      .qml-chip.warn {border-color:#f59e0b22}
      .qml-chip.bad  {border-color:#dc262622}
      .qml-mini-explain{font-size:.9rem;color:#64748B;margin-top:6px}
      /* session chip colours (legend for mini chart) */
      .qml-chip.pre  .sw{ background:#fde68a }  /* soft amber */
      .qml-chip.post .sw{ background:#c7d2fe }  /* soft indigo */
      .qml-chip.reg  .sw{ background:#bbf7d0 }  /* soft green */
      /* arrow tint for EMA trend chip */
      .qml-chip.up    { border-color:#16a34a33 }
      .qml-chip.down  { border-color:#dc262633 }
      .qml-chip.flat  { border-color:#9ca3af33 }
      .qml-chip.up   { color:#15803d; }
      .qml-chip.down { color:#b91c1c; }
      .qml-chip.flat { color:#334155; }


    </style>
    """, unsafe_allow_html=True)


apply_branding()


# =============================================================================
# Config
# =============================================================================
try:
    from config import PORTFOLIO_BUDGET as _PORTFOLIO_BUDGET
except Exception:
    _PORTFOLIO_BUDGET = 5000.0


from plotly.subplots import make_subplots

def mini_trailing_chart_rich(api, symbol: str, tp: float | None, sl: float | None, *,
                             days:int=5, window:int=50, entry: float | None=None, atr: float | None=None):
    """
    Compact, information-dense mini chart:
    • Price (last ~50 samples) + EMA(20), EMA(50)
    • Bollinger bands(20,2) using close
    • Last price dot + 'Updated Xm ago' badge
    • Shaded TP/SL zones with % & ATR distances
    """
    bars = _get_symbol_bars(api, symbol, "5Min", days=days).tail(max(60, window))
    if bars is None or bars.empty:
        st.info("No recent bars for mini chart.")
        return

    z = bars.copy()
    z["close"] = pd.to_numeric(z["close"], errors="coerce")
    z = z.dropna(subset=["close"])
    z["ema20"] = z["close"].ewm(span=20, adjust=False).mean()
    z["ema50"] = z["close"].ewm(span=50, adjust=False).mean()

    # Bollinger (20,2)
    mid = z["close"].rolling(20).mean()
    std = z["close"].rolling(20).std(ddof=0)
    z["bb_mid"], z["bb_up"], z["bb_dn"] = mid, mid + 2*std, mid - 2*std

    x = z["ts"]; y = z["close"]
    # ----- Session shading (pre / regular / post) in US/Eastern -----
    from zoneinfo import ZoneInfo
    et = ZoneInfo("US/Eastern")
    zt = pd.to_datetime(z["ts"], utc=True, errors="coerce").dt.tz_convert(et)
    z = z.assign(ts_et=zt)

    from datetime import time as _t

    def _day_bounds(_d):
        # Pre: 04:00–09:30  · Reg: 09:30–16:00  · Post: 16:00–20:00 (US/Eastern)
        d0_naive = datetime.combine(_d, _t(4, 0))
        p1_naive = datetime.combine(_d, _t(9, 30))
        r1_naive = datetime.combine(_d, _t(16, 0))
        q1_naive = datetime.combine(_d, _t(20, 0))

        d0 = pd.Timestamp(d0_naive).tz_localize(et)
        p1 = pd.Timestamp(p1_naive).tz_localize(et)
        r1 = pd.Timestamp(r1_naive).tz_localize(et)
        q1 = pd.Timestamp(q1_naive).tz_localize(et)
        return d0, p1, r1, q1

    shapes = []
    for d, g in z.groupby(z["ts_et"].dt.date):
        d0, p1, r1, q1 = _day_bounds(d)
        # convert back to UTC for Plotly shapes (our x-axis is UTC)
        def U(t): return pd.Timestamp(t).tz_convert("UTC")
        # Pre-market region
        pre = g[(g["ts_et"] >= d0) & (g["ts_et"] < p1)]
        if not pre.empty:
            shapes.append(dict(type="rect", xref="x", yref="paper",
                            x0=U(pre["ts_et"].iloc[0]), x1=U(pre["ts_et"].iloc[-1]),
                            y0=0, y1=1, fillcolor="rgba(253,230,138,0.15)", line_width=0, layer="below"))
        # Post-market region
        post = g[(g["ts_et"] >= r1) & (g["ts_et"] < q1)]
        if not post.empty:
            shapes.append(dict(type="rect", xref="x", yref="paper",
                            x0=U(post["ts_et"].iloc[0]), x1=U(post["ts_et"].iloc[-1]),
                            y0=0, y1=1, fillcolor="rgba(199,210,254,0.18)", line_width=0, layer="below"))
    # regular session left unshaded for clarity

    last_px = float(y.iloc[-1])
    last_ts = pd.to_datetime(x.iloc[-1], utc=True, errors="coerce")

    # Compute % & ATR distances (safe if None)
    def _pct(a,b): 
        try: return (a-b)/b
        except: return None
    tp_pct = _pct(tp, last_px) if tp is not None else None
    sl_pct = _pct(last_px, sl) if sl is not None else None
    tp_atr = (abs((tp or last_px) - last_px) / atr) if (tp is not None and atr and atr>0) else None
    sl_atr = (abs(last_px - (sl or last_px)) / atr) if (sl is not None and atr and atr>0) else None

    fig = go.Figure()
    # Price + EMAs
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Price", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=x, y=z["ema20"], mode="lines", name="EMA20", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=x, y=z["ema50"], mode="lines", name="EMA50", line=dict(width=1)))

    # Bands (as area)
    fig.add_trace(go.Scatter(x=x, y=z["bb_up"], line=dict(width=0), name="BB(20,2)",
                             hoverinfo="skip", showlegend=False,
                             fill=None))
    fig.add_trace(go.Scatter(x=x, y=z["bb_dn"], line=dict(width=0), name="BB(20,2)",
                             hoverinfo="skip", showlegend=False,
                             fill='tonexty', fillcolor="rgba(99,102,241,0.12)"))

    # TP/SL guides + zones
    ymax = max(float(np.nanmax(y)), float(tp) if tp else float(np.nanmax(y)))
    ymin = min(float(np.nanmin(y)), float(sl) if sl else float(np.nanmin(y)))
    if tp is not None:
        fig.add_hline(y=float(tp), line_dash="dot",
                      annotation_text=f"TP • {tp_pct:+.2%}" + (f" ({tp_atr:.2f} ATR)" if tp_atr else ""),
                      annotation_position="top right")
        fig.add_shape(type="rect", x0=x.iloc[0], x1=x.iloc[-1], y0=float(tp), y1=ymax,
                      fillcolor="rgba(16,185,129,0.08)", line_width=0, layer="below")
    if sl is not None:
        fig.add_hline(y=float(sl), line_dash="dot",
                      annotation_text=f"SL • {sl_pct:+.2%}" + (f" ({sl_atr:.2f} ATR)" if sl_atr else ""),
                      annotation_position="bottom right")
        fig.add_shape(type="rect", x0=x.iloc[0], x1=x.iloc[-1], y0=ymin, y1=float(sl),
                      fillcolor="rgba(239,68,68,0.08)", line_width=0, layer="below")

    # Entry line (optional)
    if entry is not None and np.isfinite(entry):
        fig.add_hline(y=float(entry), line_dash="dash", annotation_text="Entry")

    # Last price dot + updated time
    fig.add_trace(go.Scatter(x=[x.iloc[-1]], y=[last_px], mode="markers",
                             marker=dict(size=7), name="Last", showlegend=False))
    if pd.notna(last_ts):
        mins = int((pd.Timestamp.utcnow() - last_ts).total_seconds() // 60)
        fig.add_annotation(xref="paper", yref="paper", x=1, y=1.15, showarrow=False,
                           text=f"Updated {mins}m ago")

    # Apply session shading rectangles (must be before plotting)
    if shapes:
        fig.update_layout(shapes=shapes)

    fig.update_layout(
        height=200, margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    )
    st.plotly_chart(
        fig,
        config=PLOTLY_CONFIG,
        use_container_width=True
    )

    # ===== compact explanation (trend • bands • position • TP/SL) =====
    # Trend by EMAs
    ema20_now, ema50_now = float(z["ema20"].iloc[-1]), float(z["ema50"].iloc[-1])
    ema_slope = np.sign(z["ema20"].diff().iloc[-1])
    trend = ("up" if (ema20_now >= ema50_now and ema_slope >= 0) else
            "down" if (ema20_now <= ema50_now and ema_slope <= 0) else
            "mixed")

    # Band width percentile (squeeze vs expanding)
    bw = (z["bb_up"] - z["bb_dn"]) / z["bb_mid"]
    bw_last = float(bw.iloc[-1]) if np.isfinite(bw.iloc[-1]) else np.nan
    pctile = (np.nanpercentile(bw, 20), np.nanpercentile(bw, 80))
    band_state = ("squeeze" if bw_last <= pctile[0] else
                "expanding" if bw_last >= pctile[1] else
                "normal")
    # Price position vs bands
    pos = ("below band" if last_px < float(z["bb_dn"].iloc[-1]) else
        "above band" if last_px > float(z["bb_up"].iloc[-1]) else
        "inside band")

    # TP/SL distances summary
    tp_txt = (f"{tp_pct:+.2%}" if tp_pct is not None else "—")
    sl_txt = (f"{sl_pct:+.2%}" if sl_pct is not None else "—")

    # Optional ATR text
    atr_txt = []
    if tp_atr: atr_txt.append(f"TP {tp_atr:.2f} ATR")
    if sl_atr: atr_txt.append(f"SL {sl_atr:.2f} ATR")
    atr_txt = (" • " + " · ".join(atr_txt)) if atr_txt else ""

    # Arrow for EMA trend
    arrow = "▲" if trend=="up" else ("▼" if trend=="down" else "↔")
    trend_cls = "up" if trend=="up" else ("down" if trend=="down" else "flat")

    trend_chip = f"<span class='qml-chip {trend_cls}'><span class='sw' style='background:#4f46e5'></span>EMA trend: {arrow} {trend}</span>"
    band_chip  = f"<span class='qml-chip {'warn' if band_state=='squeeze' else 'good' if band_state=='expanding' else ''}'><span class='sw' style='background:#6366f1'></span>Bands: {band_state}</span>"

    # ✅ ADD THIS LINE (defines pos_chip)
    pos_chip   = f"<span class='qml-chip'><span class='sw' style='background:#94a3b8'></span>Price: {pos}</span>"

    pre_chip  = "<span class='qml-chip pre'><span class='sw'></span>Pre</span>"
    reg_chip  = "<span class='qml-chip reg'><span class='sw'></span>Regular</span>"
    post_chip = "<span class='qml-chip post'><span class='sw'></span>Post</span>"

    st.markdown(
        f"""
        <div class="qml-mini-explain">
        {trend_chip} &nbsp; {band_chip} &nbsp; {pos_chip} &nbsp; {pre_chip} &nbsp; {reg_chip} &nbsp; {post_chip}<br>
        Price is <b>{tp_txt}</b> from TP and <b>{sl_txt}</b> from SL{atr_txt}.
        </div>
        """,
        unsafe_allow_html=True
    )


def _safe_rr(tp_buf: float, sl_buf: float) -> float:
    """Finite RR: ∞ if SL buffer==0 and TP>0; 0 if both zero; otherwise TP/SL."""
    if not np.isfinite(tp_buf): tp_buf = 0.0
    if not np.isfinite(sl_buf): sl_buf = 0.0
    if sl_buf <= 0:
        return float("inf") if tp_buf > 0 else 0.0
    return tp_buf / sl_buf

def _fmt_rr(rr: float) -> str:
    if not np.isfinite(rr):  # inf
        return "∞"
    return f"{rr:.2f}"

def _rr_class(rr: float | None) -> str:
    if rr is None or not np.isfinite(rr): return "warn"
    if rr >= 2.0:  return "good"   # strong
    if rr >= 1.0:  return "warn"   # okay
    return "bad"                   # < 1 = asymmetric risk

def _pill_severity(sl_atr: float | None) -> str:
    """Return '', 'warn', or 'danger' for the left border colour based on SL distance in ATRs."""
    if sl_atr is None or not np.isfinite(sl_atr):
        return "warn"      # unknown → amber
    if sl_atr < 0.4:
        return "danger"    # < 0.4 ATR is risky
    if sl_atr < 0.75:
        return "warn"      # OK-ish
    return ""              # healthy

def confidence_bar(ticker, side, price, tp, sl, atr=None):
    if not price: return

    # Distances (signed by side)
    if str(side).lower() == "long":
        tp_d  = max((tp - price), 0) if tp else 0
        sl_d  = max((price - sl), 0) if sl else 0
    else:
        tp_d  = max((price - tp), 0) if tp else 0
        sl_d  = max((sl - price), 0) if sl else 0

    tot = max(tp_d + sl_d, 1e-9)
    tp_pct = tp_d / price if price else 0
    sl_pct = sl_d / price if price else 0
    tp_atr = (tp_d / atr) if atr else np.nan
    sl_atr = (sl_d / atr) if atr else np.nan
    rr = (tp_d / sl_d) if sl_d > 0 else np.inf

    # Alert tints based on ATR buffers
    sl_color = "#ef4444" if (atr and sl_atr < 0.40) else "#f97316" if (atr and sl_atr < 0.75) else "#ef4444"
    tp_color = "#22c55e" if (atr and tp_atr >= 0.50) else "#d4d4d4"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[sl_d], y=[ticker], orientation="h", name="SL buffer", marker_color=sl_color,
        hovertemplate=f"SL buffer: ${sl_d:.2f} ({sl_pct:.2%})<br>ATR: {sl_atr:.2f}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=[tp_d], y=[ticker], orientation="h", name="TP buffer", marker_color=tp_color,
        hovertemplate=f"TP buffer: ${tp_d:.2f} ({tp_pct:.2%})<br>ATR: {tp_atr:.2f}<extra></extra>"
    ))
    fig.update_layout(
        barmode="stack", height=70, margin=dict(l=8,r=8,t=22,b=8),
        xaxis=dict(visible=False, range=[0, tot]), yaxis=dict(visible=False), showlegend=False,
        annotations=[dict(
            x=tot, y=0, xref="x", yref="y", showarrow=False,
            text=f"RR {rr:.2f}× • TP {tp_pct:.2%} / SL {sl_pct:.2%}" + (f" • ({tp_atr:.2f}/{sl_atr:.2f} ATR)" if atr else ""),
            xanchor="right", yanchor="bottom", font=dict(size=11)
        )]
    )
    st.plotly_chart(fig, config={**PLOTLY_CONFIG, "responsive": True}, use_container_width=True)

import streamlit as st
def kpi_triplet(tp_pct, sl_pct, rr, *, tp_atr=None, sl_atr=None):
    c1, c2, c3 = st.columns([1, 1, 1])
    fmt = lambda v: f"{v:.2%}" if np.isfinite(v) else "—"
    c1.metric("→ TP buffer", fmt(tp_pct), None,
              help=None if tp_atr is None else f"{tp_atr:.2f} ATR")
    c2.metric("← SL buffer", fmt(sl_pct), None,
              help=None if sl_atr is None else f"{sl_atr:.2f} ATR")
    c3.metric("Risk / Reward", f"{rr:.2f}×")

# === Unified Positions Panel (grouping + ATR badge + mini charts + confidence bars + filters) ===
def trailing_chart_with_context(bars, *, price_now, tp, sl, entry, atr,
                                last_tp_ts=None, last_sl_ts=None, updated_ts=None):
    # bars: DataFrame with columns ['ts','close'] (recent -> newest at end)
    x = bars['ts']; y = bars['close']

    # Distances
    def pct(a,b): 
        try: return (a-b)/b
        except: return np.nan
    tp_pct = pct(tp, price_now); sl_pct = pct(price_now, sl)
    tp_atr = (abs(tp - price_now) / atr) if atr else np.nan
    sl_atr = (abs(price_now - sl) / atr) if atr else np.nan

    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Price", hovertemplate="%{x}<br>%{y:.2f}",))

    # Background zones (shaded)
    ymax = max(np.nanmax(y), tp if pd.notna(tp) else np.nanmax(y)) * 1.001
    ymin = min(np.nanmin(y), sl if pd.notna(sl) else np.nanmin(y)) * 0.999
    if pd.notna(tp):
        fig.add_shape(type="rect", x0=x.iloc[0], x1=x.iloc[-1], y0=tp, y1=ymax,
                      fillcolor="rgba(16,185,129,0.08)", line_width=0, layer="below")  # TP zone (green tint)
    if pd.notna(sl):
        fig.add_shape(type="rect", x0=x.iloc[0], x1=x.iloc[-1], y0=ymin, y1=sl,
                      fillcolor="rgba(239,68,68,0.08)", line_width=0, layer="below")  # SL zone (red tint)

    # Guide lines
    if pd.notna(tp):    fig.add_hline(y=tp, line_dash="dot", annotation_text=f"TP  • {tp_pct:+.2%} ({tp_atr:.2f} ATR)")
    if pd.notna(sl):    fig.add_hline(y=sl, line_dash="dot", annotation_text=f"SL  • {sl_pct:+.2%} ({sl_atr:.2f} ATR)",
                                      annotation_position="bottom right")
    if pd.notna(entry): fig.add_hline(y=entry, line_dash="dash", annotation_text="Entry")

    # Mark last ratchets (optional)
    def _mark(ts, label):
        if ts is None: return
        ts = pd.to_datetime(ts)
        row = bars.iloc[(bars['ts']-ts).abs().argsort()[:1]]
        fig.add_trace(go.Scatter(x=row['ts'], y=row['close'],
                                 mode="markers+text", text=[label], textposition="top center",
                                 marker=dict(size=8, symbol="triangle-up")))
    _mark(last_tp_ts, "TP↑")
    _mark(last_sl_ts, "SL↑")

    # Footer badge
    if updated_ts:
        ago_min = int((datetime.now(timezone.utc) - pd.to_datetime(updated_ts, utc=True)).total_seconds()//60)
        fig.add_annotation(xref="paper", yref="paper", x=1, y=1.12, showarrow=False,
                           text=f"Updated {ago_min}m ago")

    fig.update_layout(height=200, margin=dict(l=10,r=10,t=30,b=10),
                      showlegend=False, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"))
    st.plotly_chart(fig, config={**PLOTLY_CONFIG, "responsive": True}, use_container_width=True)

def state_badge(*, last_tp_ts, last_sl_ts, cooldown_min, move_atr, trigger_atr):
    import pandas as pd, numpy as np, datetime as dt
    now = pd.Timestamp.utcnow()
    cool = False
    if last_tp_ts is not None:
        cool = (now - pd.to_datetime(last_tp_ts, utc=True)).total_seconds() < cooldown_min*60
    if cool:
        st.caption("⏳ Cooling down")
    elif np.isfinite(move_atr) and move_atr >= trigger_atr:
        st.caption("✅ Ready to ratchet")
    else:
        st.caption("ℹ️ Watching…")

def render_positions_panel(api, positions: pd.DataFrame, *, atr_mode: dict | None = None) -> None:
    """
    positions: live positions df from pull_live_positions() / compute_derived_metrics()
    atr_mode:  e.g. {"mode":"ATR","step_atr":0.50,"trigger_atr":1.00} or None
    """
    st.subheader("Positions — Overview")

    # --- Fetch Adaptive ATR table so we can attach TP/SL + last update for filters
    atr_df = build_adaptive_atr_df(api, positions)
    # keep only the bits we need for merge
    keep_cols = ["Ticker","Side","Current Price","Current TP","Current SL","Updated (ET)"]
    atr_df = atr_df[[c for c in keep_cols if c in atr_df.columns]].copy()

    # --------------------- Side-normalize for merge (ROBUST) ---------------------
    z = positions.copy()

    # pick a symbol column if one exists; else create "Ticker" from index
    sym_col = _first_existing_col(z, "Ticker", "symbol", "Asset", "Symbol", "asset")
    if sym_col is None:
        # synthesize a Ticker column from the index so downstream never crashes
        z = z.copy()
        z["Ticker"] = z.index.astype(str)
        sym_col = "Ticker"

    # normalize symbols to UPPER
    z[sym_col] = z[sym_col].astype(str).str.upper()

    # ensure ATR table has a "Ticker" column in UPPER for the join
    if "Ticker" not in atr_df.columns:
        cand = _first_existing_col(atr_df, "Ticker", "symbol", "Asset", "Symbol", "asset")
        if cand is not None:
            atr_df = atr_df.copy()
            atr_df["Ticker"] = atr_df[cand].astype(str).str.upper()
        else:
            atr_df = atr_df.copy()
            atr_df["Ticker"] = ""

    atr_df["Ticker"] = atr_df["Ticker"].astype(str).str.upper()

    merged = z.merge(
        atr_df,
        left_on=sym_col,
        right_on="Ticker",
        how="left",
        suffixes=("", "_atr")
    )

    # ===================== Header w/ ATR badge =====================
    eff = atr_mode or {"mode":"ATR","step_atr":"—","trigger_atr":"—"}
    st.markdown(
        f"""
        <div style="display:flex;gap:10px;align-items:center;">
          <span style="font-weight:600;font-size:18px;">Dynamic Stop trailing (Adaptive ATR)</span>
          <span style="background:{BRAND['accent']};color:white;border-radius:999px;padding:3px 10px;font-size:12px;">
            {eff.get('mode','ATR')} • Step={eff.get('step_atr','—')} • Trigger={eff.get('trigger_atr','—')}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ===================== Filters (Long/Short + recent updates) =====================
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        side_filter = st.segmented_control("Side", ["All", "Longs", "Shorts"], default="All", key="pos_side_filter")
    with c2:
        recent_only = st.toggle("Only Updated in last 30m", value=False, key="pos_recent_toggle")

    df = merged.copy()
    # Apply side filter
    if side_filter == "Longs":
        df = df[df.get("Side","Long").astype(str).str.lower().eq("long")]
    elif side_filter == "Shorts":
        df = df[df.get("Side","Long").astype(str).str.lower().eq("short")]

    # Apply recency filter using Updated (ET) if present
    if recent_only and "Updated (ET)" in df.columns:
        ts = pd.to_datetime(df["Updated (ET)"], errors="coerce", utc=True)
        df = df[ts >= (pd.Timestamp.utcnow() - pd.Timedelta(minutes=30))]

    # ===================== Grouped tables (Long / Short) =====================
    def _show_group(title, gdf):
        if gdf is None or gdf.empty:
            st.markdown(f"**{title}** — none")
            return
        view = pd.DataFrame({
            "Ticker":        gdf[sym_col],
            "Side":          gdf.get("Side","Long"),
            "Qty":           pd.to_numeric(gdf.get("qty"), errors="coerce"),
            "Avg Entry":     pd.to_numeric(gdf.get("entry_price"), errors="coerce"),
            "Current Price": pd.to_numeric(gdf.get("Current Price", gdf.get("current_price")), errors="coerce"),
            "Current TP":    pd.to_numeric(gdf.get("Current TP"), errors="coerce"),
            "Current SL":    pd.to_numeric(gdf.get("Current SL"), errors="coerce"),
            "P&L $":         pd.to_numeric(gdf.get("pl_$", gdf.get("pl_usd")), errors="coerce"),
            "P&L %":         pd.to_numeric(gdf.get("pl_%"), errors="coerce"),
            "Updated (ET)":  gdf.get("Updated (ET)",""),
        })
        st.markdown(f"**{title}**")
        st.dataframe(
            view.sort_values("P&L %", ascending=False).style.format({
                "Qty":"{:.0f}",
                "Avg Entry":"{:.2f}",
                "Current Price":"{:.2f}",
                "Current TP":"{:.2f}",
                "Current SL":"{:.2f}",
                "P&L $":"${:,.2f}",
                "P&L %":"{:+.2f}%",
            }, na_rep="—"),
            width='stretch', hide_index=True
        )

    long_df  = df[df.get("Side","Long").astype(str).str.lower().eq("long")]
    short_df = df[df.get("Side","Long").astype(str).str.lower().eq("short")]

    g1, g2 = st.columns(2)
    with g1: _show_group("Long", long_df)
    with g2: _show_group("Short", short_df)

    st.divider()

    # ===================== Mini trailing visualization (last ~50 bars) =====================

    # --- Build symbol choices and pick default ---
    syms = sorted(df[sym_col].dropna().astype(str).unique())
    choices = ["ALL"] + syms
    default_sym = st.session_state.get("pos_focus_symbol2")
    if default_sym not in choices:
        default_sym = (syms[:1] or ["ALL"])[0]

    # --- Header + selector on the same row ---
    left, right = st.columns([0.5, 0.5])
    with left:
        st.markdown("**Mini trailing chart**")
    with right:
        focus_symbol = st.selectbox(
            "Mini chart symbol",
            choices,
            index=choices.index(default_sym),
            key="pos_focus_symbol2",
            label_visibility="collapsed",
        )
    # --- Show single or multi-mini-chart view ---
    if focus_symbol == "ALL":
        # Show grid of mini-charts (top 6 by RR)
        tmp = df[[sym_col, "Side", "Current Price", "Current TP", "Current SL"]].dropna(subset=[sym_col]).copy()

        def _rr_quick(r):
            side = str(r.get("Side", "Long")).lower()
            p  = float(r.get("Current Price") or np.nan)
            tp = float(r.get("Current TP") or np.nan)
            sl = float(r.get("Current SL") or np.nan)
            if not (np.isfinite(p) and p > 0): return np.nan
            if side == "long":
                tp_d = max((tp - p), 0) if np.isfinite(tp) else 0
                sl_d = max((p - sl), 0) if np.isfinite(sl) else 0
            else:
                tp_d = max((p - tp), 0) if np.isfinite(tp) else 0
                sl_d = max((sl - p), 0) if np.isfinite(sl) else 0
            return (tp_d / sl_d) if sl_d > 0 else np.inf

        tmp["__rr"] = tmp.apply(_rr_quick, axis=1)
        top = (
            tmp.sort_values(["__rr", sym_col], ascending=[False, True])
            .head(6)[sym_col].astype(str).tolist()
        )

        grid = st.columns(3)
        for i, sym in enumerate(top):
            with grid[i % 3]:
                r = df[df[sym_col] == sym].head(1)
                t = float(pd.to_numeric(r["Current TP"], errors="coerce").iloc[0]) if "Current TP" in r.columns else np.nan
                s = float(pd.to_numeric(r["Current SL"], errors="coerce").iloc[0]) if "Current SL" in r.columns else np.nan
                st.markdown(f"**{sym}**")
                mini_trailing_chart_rich(api, sym, t if np.isfinite(t) else None, s if np.isfinite(s) else None)
        st.divider()

    else:
        row = df[df[sym_col] == focus_symbol].head(1)
        tp_val = float(pd.to_numeric(row["Current TP"], errors="coerce").iloc[0]) \
                if ("Current TP" in row.columns and len(row)) else None
        sl_val = float(pd.to_numeric(row["Current SL"], errors="coerce").iloc[0]) \
                if ("Current SL" in row.columns and len(row)) else None

        mini_trailing_chart_rich(api, str(focus_symbol), tp_val, sl_val)
        st.divider()

    # ===================== SL/TP Confidence bar (per row) =====================
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown("**SL/TP confidence bars**")
    def _buffers(side: str, price: float, tp: float | None, sl: float | None) -> tuple[float,float]:
        """Returns (tp_buffer_pct, sl_buffer_pct) as proportions of price."""
        if not (np.isfinite(price) and price > 0): return (0.0, 0.0)
        def _safe(v): 
            try: return float(v)
            except: return np.nan
        tp = _safe(tp); sl = _safe(sl)
        if str(side).lower() == "long":
            tp_buf = max(((tp - price) / price) if pd.notna(tp) else 0.0, 0.0)
            sl_buf = max(((price - sl) / price) if pd.notna(sl) else 0.0, 0.0)
        else:
            tp_buf = max(((price - tp) / price) if pd.notna(tp) else 0.0, 0.0)
            sl_buf = max(((sl - price) / price) if pd.notna(sl) else 0.0, 0.0)
        return (tp_buf, sl_buf)

    # --- At-risk filter (SL distance < 0.5 ATR) + sort by risk ---
    atr_col = next((c for c in df.columns if c.lower() in ("atr","atr_14","atr14")), None)

    if atr_col and {"Current Price","Current SL"}.issubset(df.columns):
        # SL distance expressed in ATRs
        df["sl_atr"] = (
            (pd.to_numeric(df["Current Price"], errors="coerce") -
            pd.to_numeric(df["Current SL"],    errors="coerce")).abs()
            / pd.to_numeric(df[atr_col], errors="coerce")
        )

        at_risk = st.toggle("Show only at-risk (SL < 0.5 ATR)", value=False, key="pos_at_risk_toggle")
        if at_risk:
            df = df[pd.to_numeric(df["sl_atr"], errors="coerce") < 0.5]

        # Lowest SL buffer first
        df = df.sort_values("sl_atr", ascending=True, na_position="last")
    else:
        # Fallback if no ATR column: use absolute SL % distance (0.5% of price as rough proxy)
        if "SL %" in df.columns:
            df["sl_abs_pct"] = pd.to_numeric(df["SL %"], errors="coerce").abs() / 100.0
            at_risk = st.toggle("Show only at-risk (SL < 0.5% of price)", value=False, key="pos_at_risk_pct_toggle")
            if at_risk:
                df = df[df["sl_abs_pct"] < 0.005]
            df = df.sort_values("sl_abs_pct", ascending=True, na_position="last")

    # Show a compact stacked bar per ticker
    # === Compact pill rows (sorted by riskiest first) ===
    rows = df[[sym_col,"Side","Current Price","Current TP","Current SL"]].dropna(subset=[sym_col]).copy()

    # compute buffers (% of price) and SL distance in ATRs
    atr_col = next((c for c in df.columns if c.lower() in ("atr","atr_14","atr14")), None)
    def _calc(row):
        side  = str(row.get("Side","Long")).lower()
        p     = float(row.get("Current Price") or np.nan)
        tp    = float(row.get("Current TP")   or np.nan)
        sl    = float(row.get("Current SL")   or np.nan)
        if not (np.isfinite(p) and p > 0):
            return pd.Series({"tp_buf":np.nan,"sl_buf":np.nan,"sl_atr":np.nan})
        if side == "long":
            tp_buf = max(((tp - p)/p) if np.isfinite(tp) else 0.0, 0.0)
            sl_buf = max(((p - sl)/p) if np.isfinite(sl) else 0.0, 0.0)
        else:
            tp_buf = max(((p - tp)/p) if np.isfinite(tp) else 0.0, 0.0)
            sl_buf = max(((sl - p)/p) if np.isfinite(sl) else 0.0, 0.0)
        sl_atr = np.nan
        if atr_col and np.isfinite(p) and np.isfinite(sl) and np.isfinite(float(row.get(atr_col) or np.nan)):
            sl_atr = abs(p - sl) / float(row[atr_col])
        return pd.Series({"tp_buf":tp_buf,"sl_buf":sl_buf,"sl_atr":sl_atr})

    # compute buffers (% of price) and SL distance in ATRs
    # compute buffers (% of price) and SL distance in ATRs
    atr_col = next((c for c in df.columns if c.lower() in ("atr","atr_14","atr14")), None)

    def _calc(row):
        side  = str(row.get("Side","Long")).lower()
        p     = float(row.get("Current Price") or np.nan)
        tp    = float(row.get("Current TP")   or np.nan)
        sl    = float(row.get("Current SL")   or np.nan)
        if not (np.isfinite(p) and p > 0):
            return pd.Series({"tp_buf": np.nan, "sl_buf": np.nan, "sl_atr": np.nan})
        if side == "long":
            tp_buf = max(((tp - p)/p) if np.isfinite(tp) else 0.0, 0.0)
            sl_buf = max(((p - sl)/p) if np.isfinite(sl) else 0.0, 0.0)
        else:
            tp_buf = max(((p - tp)/p) if np.isfinite(tp) else 0.0, 0.0)
            sl_buf = max(((sl - p)/p) if np.isfinite(sl) else 0.0, 0.0)
        sl_atr = np.nan
        if atr_col and np.isfinite(p) and np.isfinite(sl) and np.isfinite(float(row.get(atr_col) or np.nan)):
            sl_atr = abs(p - sl) / float(row[atr_col])
        return pd.Series({"tp_buf": tp_buf, "sl_buf": sl_buf, "sl_atr": sl_atr})

    # Safe expand → ensure expected columns exist even if rows is empty
    calc = rows.apply(_calc, axis=1, result_type="expand") if not rows.empty else pd.DataFrame()
    calc = calc.reindex(columns=["tp_buf", "sl_buf", "sl_atr"], fill_value=np.nan)

    if rows.empty:
        # nothing to render below; bail out gracefully
        st.info("No positions match the current filters.")
        return

    rows[["tp_buf", "sl_buf", "sl_atr"]] = calc[["tp_buf", "sl_buf", "sl_atr"]]

    # sort: riskiest first (smallest sl_atr), then lower RR
    rows["rr"] = np.where(
        pd.to_numeric(rows["sl_buf"], errors="coerce") > 0,
        pd.to_numeric(rows["tp_buf"], errors="coerce") / pd.to_numeric(rows["sl_buf"], errors="coerce"),
        np.inf  # sort ∞ last by second key (we still rank by sl_atr first)
    )
    # --- sort by Risk/Reward (high → low). Use sl_atr as tie-breaker (lower = riskier, shown later)
    rows["__rr_rank"] = rows["rr"].replace([np.inf, -np.inf], 1e12)  # put ∞ at the very top
    rows = rows.sort_values(["__rr_rank", "sl_atr"], ascending=[False, True], na_position="last").drop(columns="__rr_rank").head(12)

    # grid (3 per row feels right)
    cols = st.columns(3)
    for i, (_, r) in enumerate(rows.iterrows()):
        with cols[i % 3]:
            tkr   = str(r[sym_col])
            side  = r.get("Side","Long")
            p     = float(r.get("Current Price") or np.nan)
            tpbuf = float(r.get("tp_buf") or 0.0)
            slbuf = float(r.get("sl_buf") or 0.0)
            p     = float(r.get("Current Price") or 0.0)

            # get SL distance in ATRs from the precomputed column
            slatr = float(r["sl_atr"]) if ("sl_atr" in r and pd.notna(r["sl_atr"])) else None

            # ---- SAFE RR + color class
            rr     = _safe_rr(tpbuf, slbuf)          # ∞ if SL=0 & TP>0; 0 if both zero
            rr_cls = _rr_class(rr)                   # 'good' / 'warn' / 'bad'
            rr_txt = _fmt_rr(rr) + "×"

            # ---- Pill severity: if NOT danger and RR is good, force green border
            sev = _pill_severity(slatr)              # '', 'warn', 'danger'
            if sev != "danger" and rr_cls == "good":
                sev = ""                             # default (green) border

            # mini bar widths
            slw = max(0.0, min(100.0, slbuf*100))
            tpw = max(0.0, min(100.0, tpbuf*100))

            # render pill
            st.markdown(
                f"""
                <div class="qml-pill {sev}">
                <div class="hdr">{tkr} · {side}</div>
                <div class="line">
                    TP {tpbuf:.2%} &nbsp;&middot;&nbsp; SL {slbuf:.2%} &nbsp;&middot;&nbsp; RR {rr_txt}
                    {f"&nbsp;&middot;&nbsp; SL {slatr:.2f} ATR" if slatr is not None and np.isfinite(slatr) else ""}
                </div>
                <div class="mini">
                    <span class="sl" style="width:{slw}%"></span>
                    <span class="tp" style="width:{tpw}%"></span>
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ---- Compact micro-KPIs row (smaller + colour-coded RR) ----
            rr_cls = _rr_class(rr)  # uses thresholds defined above (good/warn/bad)
            st.markdown(
                f"""
                <div class="qml-mini">
                <div class="itm"><span class="lbl">→ TP buffer</span>{tpbuf:.2%}</div>
                <div class="itm"><span class="lbl">← SL buffer</span>{slbuf:.2%}</div>
                <div class="itm"><span class="lbl">Risk / Reward</span><span class="qml-rr {rr_cls}">{rr_txt}</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            # ---- 1–2 line natural language summary under each pill ----
            stance = ("favourable" if rr_cls == "good" else
                    "balanced"   if rr_cls == "warn" else
                    "unfavourable")

            sl_txt = (f"{slatr:.2f} ATR" if (slatr is not None and np.isfinite(slatr)) else "ATR n/a")
            dir_txt = "above" if str(side).lower()=="long" else "below"   # orientation hint

            st.markdown(
                f"""
                <div class="qml-note">
                Price is <span class="em">{tpbuf:.2%}</span> from TP ({dir_txt}) and
                <span class="em">{slbuf:.2%}</span> from SL.
                Risk/Reward is <span class="em">{rr_txt}</span> → <span class="em">{stance}</span>
                (SL distance: {sl_txt}).
                </div>
                """,
                unsafe_allow_html=True
            )


# --- helper: add TP% / SL% columns beside Current TP / SL ---
def _add_tp_sl_percent_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      • TP % = ((Current TP / Current Price) - 1) * 100
      • SL % = ((Current SL / Current Price) - 1) * 100
    Inserts each % column immediately to the right of its base column.
    """
    if df is None or df.empty:
        return df

    d = df.copy()
    # Coerce to numeric where needed
    for c in ("Current Price", "Current TP", "Current SL"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    if {"Current TP", "Current Price"}.issubset(d.columns):
        d["TP %"] = ((d["Current TP"] / d["Current Price"]) - 1.0) * 100.0
    if {"Current SL", "Current Price"}.issubset(d.columns):
        d["SL %"] = ((d["Current SL"] / d["Current Price"]) - 1.0) * 100.0

    def _insert_after(df_, after_col, new_col):
        if new_col in df_.columns and after_col in df_.columns:
            cols = list(df_.columns)
            cols.remove(new_col)
            i = cols.index(after_col)
            cols.insert(i + 1, new_col)
            return df_[cols]
        return df_

    if "TP %" in d.columns:
        d = _insert_after(d, "Current TP", "TP %")
    if "SL %" in d.columns:
        d = _insert_after(d, "Current SL", "SL %")

    return d

def _fmt_pct(v):
    if pd.isna(v): return ""
    return f"{v:+.2f}%"

def _color_pct(val):
    if pd.isna(val): return ""
    if val > 0:  return "color:#0f8a00;font-weight:600;"
    if val < 0:  return "color:#b00020;font-weight:600;"
    return ""

# --- tiny util used by both helpers ---
def _insert_after(df_, after_col, new_col):
    if new_col in df_.columns and after_col in df_.columns:
        cols = list(df_.columns)
        cols.remove(new_col)
        i = cols.index(after_col)
        cols.insert(i + 1, new_col)
        return df_[cols]
    return df_

# --- A) PnL %: side-aware gain/loss vs entry (Cost per Share) ---
def _add_pnl_percent_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'PnL %' = side-aware return from entry:
        Long : (CurrentPrice / CostPerShare - 1) * 100
        Short: (CostPerShare / CurrentPrice - 1) * 100
    Inserts right after 'Current Price'.
    """
    if df is None or df.empty:
        return df

    d = df.copy()
    for c in ("Current Price", "Cost per Share", "Side"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c].astype(str).str.replace("$","", regex=False), errors="coerce") \
                   if c != "Side" else d[c]

    def _pnl(row):
        cp, cost = row.get("Current Price"), row.get("Cost per Share")
        sd = str(row.get("Side", "")).strip().lower()
        if cp and cost and cp > 0 and cost > 0:
            if sd == "short":
                return (cost / cp - 1.0) * 100.0
            else:
                return (cp / cost - 1.0) * 100.0
        return None

    d["PnL %"] = d.apply(_pnl, axis=1)

    if "PnL %" in d.columns:
        d = _insert_after(d, "Current Price", "PnL %")

    return d

# --- B) SL (ATR ×): how many ATRs between Current Price and Current SL ---
def _add_sl_atr_multiple(df: pd.DataFrame, atr_col_candidates=("ATR","atr_14","atr14")) -> pd.DataFrame:
    """
    Adds 'SL (ATR ×)' = |CurrentPrice - CurrentSL| / ATR
    Looks for an ATR column by common names; if not found, leaves blank.
    Inserts after 'SL %' if present else after 'Current SL'.
    """
    if df is None or df.empty:
        return df

    d = df.copy()
    # pick an ATR column if present
    atr_col = next((c for c in atr_col_candidates if c in d.columns), None)
    if atr_col is None:
        # nothing to compute -> just return as-is
        return d

    for c in ("Current Price", "Current SL", atr_col):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c].astype(str).str.replace("$","", regex=False), errors="coerce")

    def _sl_atr(row):
        cp, sl, atr = row.get("Current Price"), row.get("Current SL"), row.get(atr_col)
        if cp and sl and atr and atr > 0:
            return abs(cp - sl) / float(atr)
        return None

    d["SL (ATR ×)"] = d.apply(_sl_atr, axis=1)

    if "SL (ATR ×)" in d.columns:
        anchor = "SL %" if "SL %" in d.columns else "Current SL"
        d = _insert_after(d, anchor, "SL (ATR ×)")

    return d

# --- C) styling helpers for % columns (keep TP%/SL% neutral, PnL% colored) ---
def _fmt_signed_pct(v):
    if pd.isna(v): return ""
    try: return f"{float(v):+.2f}%"
    except Exception: return ""

def _fmt_atr_mult(v):
    if pd.isna(v): return ""
    try: return f"{float(v):.2f}×"
    except Exception: return ""

def _color_pct_signed(val):
    # green for positive, red for negative
    try:
        if pd.isna(val): return ""
        v = float(val)
        if v > 0:  return "color:#0f8a00;font-weight:600;"
        if v < 0:  return "color:#b00020;font-weight:600;"
    except Exception:
        pass
    return ""

def _style_tp_sl_percent(val):
    # neutral for TP% and SL%
    return "color:#555555;" if not pd.isna(val) else ""

def _ensure_ts_and_date(df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Return df with datetime 'ts' and a 'date' (ts.date) column.
    If df is None/empty, return an empty frame with the right columns so .dt never crashes.
    """
    import pandas as pd
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts", "ret", "date"])
    z = df.copy()
    z["ts"] = pd.to_datetime(z["ts"], utc=True, errors="coerce")
    z["date"] = z["ts"].dt.date
    return z


def _traffic_group(value: float) -> int:
    """
    Map to groups using the SAME thresholds as _tl_color_for_pct():
      0 = green (≥ 0%)
      1 = amber ( -0.8% ≤ x < 0% )
      2 = red   ( < -0.8% )
    """
    try:
        v = float(value)
    except Exception:
        v = 0.0

    col = _tl_color_for_pct(v)   # uses the central thresholds
    if col == TL_GREEN:
        return 0
    if col == TL_AMBER:
        return 1
    return 2  # TL_RED or anything else

def _default_timeframe_for_period(period: str) -> str:
    """
    Choose an Alpaca‑valid timeframe for a given UI period.
    1D → 5Min, 1W → 15Min, 1M/3M/6M → 1H, 1Y/all → 1D
    """
    p = (period or "").strip().upper()
    if p == "1D": return "5Min"
    if p == "1W": return "15Min"
    if p in {"1M", "3M", "6M"}: return "1H"
    return "1D"  # 1Y (1A after normalization) and 'all'

def _normalize_period(period: str) -> str:
    """Map user/UI periods to Alpaca's expected units (A=years)."""
    if not period:
        return "1D"
    period = period.strip()
    return {"1Y": "1A"}.get(period, period)  # pass-through for others incl. "all"

def _order_by_green_amber_red(series: pd.Series) -> list[str]:
    """Order symbols: Green → Amber → Red.
       Within each group:
         • Green (≥0%):        high → low
         • Amber (-0.8% to <0): |x| ascending (closest to 0 first)
         • Red   (< -0.8%):    descending by value (biggest loss first)
    """
    v = pd.to_numeric(series, errors="coerce").fillna(0.0)
    grp = v.apply(_traffic_group)  # 0=green, 1=amber, 2=red

    # same ‘within’ ranking as Traffic Lights
    within = np.select(
        [grp.eq(0),        grp.eq(1),  grp.eq(2)],
        [-v,               v.abs(),    -v],
        default=v,
    )

    tmp = pd.DataFrame({"sym": v.index, "grp": grp, "__within": within})
    return tmp.sort_values(["grp", "__within"])["sym"].tolist()

# ========= Position Transaction History (per open-date per ticker) =========
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

@st.cache_data(ttl=300, show_spinner=False)
def _pull_all_fills_full(_api: Optional[REST], days: int | None = 180) -> pd.DataFrame:
    """
    Pull FILL activities for the last `days` (or all if None).
    Returns: ts(UTC), symbol, side('buy'|'sell'), qty, price.
    """
    cols = ["ts","symbol","side","qty","price"]
    if _api is None:
        return pd.DataFrame(columns=cols)

    after = None
    if days is not None:
        after = (datetime.now(timezone.utc) - timedelta(days=int(days)+2)).isoformat()

    acts = []
    try:
        acts = _api.get_activities(activity_types="FILL", after=after) if after else _api.get_activities(activity_types="FILL")
    except TypeError:
        try:
            acts = _api.get_activities("FILL", after=after) if after else _api.get_activities("FILL")
        except Exception:
            try:
                acts = _api.get_account_activities("FILL", after=after) if after else _api.get_account_activities("FILL")
            except Exception:
                acts = []

    rows = []
    for a in acts or []:
        sym  = (getattr(a, "symbol", None) or getattr(a, "asset_symbol", None) or "").upper()
        side = (getattr(a, "side", "") or "").lower()
        try:
            price = float(getattr(a, "price", getattr(a, "fill_price", 0.0)) or 0.0)
            qty   = float(getattr(a, "qty",   getattr(a, "quantity", 0.0)) or 0.0)
        except Exception:
            price, qty = 0.0, 0.0
        ts = getattr(a, "transaction_time", getattr(a, "timestamp", getattr(a, "date", None)))
        ts = pd.to_datetime(str(ts), utc=True, errors="coerce")
        if pd.isna(ts) or not sym or qty <= 0 or price <= 0:
            continue
        rows.append({"ts": ts, "symbol": sym, "side": side, "qty": qty, "price": price})

    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df.sort_values("ts").reset_index(drop=True)


def _live_positions_map(api: Optional[REST]) -> dict:
    """
    Current open positions keyed by symbol:
      {SYM: {"side":"Long|Short","qty":float,"avg_entry_price":float,"current_price":float}}
    """
    out = {}
    if api is None:
        return out
    try:
        pos = api.list_positions()
    except Exception:
        return out
    for p in pos or []:
        try:
            out[str(p.symbol).upper()] = {
                "side": "Long" if str(getattr(p,"side","long")).lower()=="long" else "Short",
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "mv": float(getattr(p,"market_value", float(p.qty)*float(p.current_price)) or 0.0),
                "unrl_pl": float(getattr(p,"unrealized_pl", 0.0) or 0.0),
            }
        except Exception:
            pass
    return out

def _fetch_fills_paged(_api: REST, *, after_iso: str | None = None, until_iso: str | None = None) -> list:
    """
    Fetch ALL FILL activities between [after_iso, until_iso] using Alpaca pagination.
    Returns a flat list of activity objects.
    """
    if _api is None:
        return []
    items, token = [], None
    kw = {"activity_types": "FILL"}
    if after_iso: kw["after"] = after_iso
    if until_iso: kw["until"] = until_iso
    while True:
        try:
            res = _api.get_activities(page_token=token, **kw) if token else _api.get_activities(**kw)
        except TypeError:
            # older SDK signature
            res = _api.get_account_activities("FILL", page_token=token, **{k:v for k,v in kw.items() if v})
        except Exception:
            break
        if not res:
            break
        items.extend(res)
        # page_token lives on the *last* item in many SDKs
        token = getattr(res[-1], "id", None) or getattr(res[-1], "activity_id", None)
        if not token or len(res) < 100:  # heuristic: last page
            break
    return items


def _tp_sl_for_history(api: Optional[REST]) -> pd.DataFrame:
    """
    Pull open exit legs once (uses the helper already in file) and return a tidy df:
      symbol, side('buy'|'sell' close-side), tp(limit), sl(stop/trailing/stop_limit)
    """
    exits = _open_exits_df(api)
    if exits.empty:
        return pd.DataFrame(columns=["symbol","tp","sl","close_side"])
    # latest per symbol (close-side) for TP/SL
    out = []
    for (sym, side), grp in exits.groupby(["symbol","side"]):
        # TP = newest LIMIT
        tp = grp.loc[grp["leg_type"].eq("limit"), "limit_price"].dropna().tail(1)
        # SL = newest of stop/stop_limit/trailing_stop → prefer stop_price then trail
        stp = grp[grp["leg_type"].isin(["stop","stop_limit","trailing_stop"])].sort_values("submitted_at").tail(1)
        sl = np.nan
        if not stp.empty:
            v1 = stp["stop_price"].iloc[0] if "stop_price" in stp.columns else np.nan
            v2 = stp["trail_price"].iloc[0] if "trail_price" in stp.columns else np.nan
            sl = float(v1) if pd.notna(v1) else (float(v2) if pd.notna(v2) else np.nan)
        out.append({"symbol":sym, "close_side":side, "tp": float(tp.iloc[0]) if len(tp) else np.nan, "sl": sl})
    return pd.DataFrame(out)

import pandas as pd
import numpy as np
from datetime import datetime, timezone

def _status_badge(s: str) -> str:
    if s.startswith("Closed – Profit"):   color = "#16a34a"  # green
    elif s.startswith("Closed – Loss"):   color = "#dc2626"  # red
    elif s.startswith("Closed – Breakeven"): color = "#6b7280"  # gray
    elif s == "Open":                     color = "#2563eb"  # blue
    else:                                 color = "#374151"
    return f'<span style="padding:2px 8px;border-radius:9999px;background:{color};color:white;font-weight:600;">{s}</span>'

def _to_dt(x):
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, (pd.Timestamp, datetime)):
        return pd.to_datetime(x).tz_convert("UTC") if getattr(x, "tzinfo", None) else pd.to_datetime(x, utc=True)
    return pd.to_datetime(str(x), utc=True, errors="coerce")

def _side_from_qty(qty):
    # positive qty means LONG, negative means SHORT
    return "Long" if qty > 0 else "Short"

def _signed(qty, side):
    return qty if side == "Long" else -qty

def _vwap(prices: pd.Series, qtys: pd.Series) -> float:
    qtys = qtys.astype(float).abs()
    if qtys.sum() == 0:
        return np.nan
    return float(np.dot(prices.astype(float), qtys) / qtys.sum())
@st.cache_data(ttl=600, show_spinner=False)
def _period_lookback_days(period: str) -> int:
    p = (period or "1M").upper()
    return 2 if p == "1D" else (32 if p == "1M" else (95 if p == "3M" else 365))

def _tf_for_returns(period: str) -> str:
    # Use intraday bars only for the 1‑day view; otherwise daily bars
    return "5Min" if (period or "").upper() == "1D" else "1D"

@st.cache_data(ttl=600, show_spinner=False)
def _fetch_symbol_bars_generic(_api: Optional[REST], symbol: str, timeframe: str, *, days: int) -> pd.DataFrame:
    """
    Best‑effort fetch of bars for `symbol` over the last `days` using Alpaca's REST client,
    across old/new SDK signatures. Returns columns: ts (UTC), close (float).
    """
    if _api is None:
        return pd.DataFrame(columns=["ts", "close"])

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(2, int(days)+2))  # pad a touch for market holidays

    # Try modern alpaca_trade_api.get_bars
    try:
        # Newer SDKs expect TimeFrame or "5Min"/"1D" (both covered here)
        from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
        tf = timeframe
        if timeframe == "1D":
            tf = TimeFrame.Day
        elif timeframe == "5Min":
            # many SDKs support the (n, unit) ctor
            try:
                tf = TimeFrame(5, TimeFrameUnit.Minute)
            except Exception:
                tf = "5Min"

        bars = _api.get_bars(symbol, timeframe, start.isoformat(), end.isoformat(),
                     adjustment="raw", feed="iex")  # <= enforce IEX for paper

        # DF path (most common)
        if hasattr(bars, "df"):
            df = bars.df.reset_index()
            # Newer df is multi-index (symbol, time)
            tcol = "timestamp" if "timestamp" in df.columns else "time"
            close_col = "close" if "close" in df.columns else "c"
            out = df.rename(columns={tcol: "ts", close_col: "close"})
            if "symbol" in out.columns:
                out = out[out["symbol"].astype(str).str.upper().eq(symbol.upper())]
            out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
            return out[["ts", "close"]].dropna()
        # Iterable bars path
        rows = []
        for b in bars:
            ts = getattr(b, "t", getattr(b, "timestamp", getattr(b, "time", None)))
            px = getattr(b, "c", getattr(b, "close", None))
            if ts is None or px is None:
                continue
            rows.append({"ts": pd.to_datetime(str(ts), utc=True, errors="coerce"), "close": float(px)})
        if rows:
            return pd.DataFrame(rows).dropna()
    except Exception:
        pass

    # Very old SDK fallback: get_barset (deprecated)
    try:
        limit = 390 if timeframe != "1D" else min(1000, days + 10)
        barset = _api.get_barset(symbol, "minute" if timeframe != "1D" else "day", limit=limit)
        series = barset[symbol]
        rows = []
        for b in series:
            ts = getattr(b, "t", getattr(b, "time", None))
            px = getattr(b, "c", getattr(b, "close", None))
            if ts is None or px is None:
                continue
            rows.append({"ts": pd.to_datetime(str(ts), utc=True, errors="coerce"), "close": float(px)})
        if rows:
            df = pd.DataFrame(rows).dropna()
            # Clip to range
            return df[(df["ts"] >= start) & (df["ts"] <= end)]
    except Exception:
        pass

    return pd.DataFrame(columns=["ts", "close"])

def _get_symbol_bars(api: Optional[REST], symbol: str, timeframe: str, *, days: int) -> pd.DataFrame:
    """
    Fetch bars for `symbol` over the last `days`, returning columns: ts (UTC), close (float).
    """
    if api is None:
        return pd.DataFrame(columns=["ts", "close"])

    sym = str(symbol).strip()

    # 🔹 1) Benchmarks: go straight to Alpaca ETF and never hit Yahoo
    bench_map = getattr(CFG, "BENCHMARK_YAHOO_TO_ALPACA", {})
    if sym in bench_map:
        return _get_benchmark_bars_alpaca(api, sym, timeframe, days=days)

    # 🔹 2) Normal equities: your existing Alpaca path
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 3)
    feed = _feed_for_account(api)  # your existing helper

    try:
        tf = _map_timeframe(timeframe)  # if you already have a helper; else inline like above
        bars = api.get_bars(sym, tf, start=start.isoformat(), end=end.isoformat(), limit=None)
        ...
        # same as before: build df with ['ts','close'] and return
    except Exception:
        pass

    # 🔹 3) Final fallback ONLY for non-benchmarks: Yahoo
    try:
        y = _get_symbol_bars_yahoo(sym, timeframe, days=days)
        if not y.empty:
            return y
    except Exception:
        pass

    return pd.DataFrame(columns=["ts", "close"])

def _style_tp_sl_percent(val):
    if pd.isna(val): return ""
    return "color:#555555;"  # neutral grey text

def _style_adaptive_atr(row: pd.Series):
    """
    Cleaner color logic for Adaptive ATR table:
    - Only color 'Current Price' by P&L direction (green good, red bad).
    - Color 'Side' as a subtle cue (blue Long, orange Short).
    Uses Index.get_loc() to avoid 'Index is not callable' errors.
    """
    # start with neutral styles
    styles = [""] * len(row)

    # quick column lookups (safe even if columns missing)
    idx = row.index
    has_cp   = "Current Price" in idx
    has_cost = "Cost per Share" in idx
    has_side = "Side" in idx

    # favor numeric values; fall back to coercion
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            try:
                return float(str(x).replace("$", ""))
            except Exception:
                return None

    side = str(row.get("Side", "")).strip().lower()
    cp   = _to_float(row.get("Current Price"))
    cost = _to_float(row.get("Cost per Share"))

    # color Current Price by favorable/unfavorable vs entry, respecting side
    if has_cp and has_cost and cp is not None and cost is not None:
        cp_loc = idx.get_loc("Current Price")
        if side == "long":
            styles[cp_loc] = "color:#0f8a00;font-weight:600;" if cp > cost else "color:#b00020;font-weight:600;"
        elif side == "short":
            styles[cp_loc] = "color:#0f8a00;font-weight:600;" if cp < cost else "color:#b00020;font-weight:600;"

    # subtle 'Side' cue
    if has_side:
        side_loc = idx.get_loc("Side")
        color = "#0077b6" if side == "long" else "#e67e22"
        styles[side_loc] = f"color:{color};font-weight:600;"

    return styles

def build_position_transaction_history(api, days: int = 180) -> pd.DataFrame:
    """
    One row per position *open lot*. Becomes Closed once cumulative qty returns to zero.
    Emits:
      open_date (UTC), open_ts (UTC), close_ts (UTC or None), symbol, side, status,
      entry_vwap, exit_vwap (None if open), qty, days_open, realized_pl$, realized_pl%, fees$, tp, sl, notes
    """
    fills = _load_fills_dataframe(api, days=days)
    if fills.empty:
        return pd.DataFrame(columns=[
            "open_date","open_ts","close_ts","symbol","side","status",
            "entry_vwap","exit_vwap","qty","days_open","realized_pl$","realized_pl%","fees$","tp","sl","notes"
        ])

    out_rows = []
    for sym, g in fills.groupby("symbol", sort=False):
        g = g.sort_values("time").reset_index(drop=True)

        cum = 0.0
        lot_start_idx = None
        lot_side = None

        for i, r in g.iterrows():
            prev_cum = cum
            cum += r["signed_qty"]

            # --- Open a new lot on transition 0 -> non-zero
            if prev_cum == 0 and cum != 0 and lot_start_idx is None:
                lot_start_idx = i
                lot_side = "Long" if cum > 0 else "Short"

            # --- Lot closes exactly when cum returns to zero
            if lot_start_idx is not None and cum == 0:
                lot_df = g.iloc[lot_start_idx:i+1].copy()

                if lot_side == "Long":
                    entries = lot_df[lot_df["signed_qty"] > 0]
                    exits   = lot_df[lot_df["signed_qty"] < 0]
                    qty = float(entries["qty"].sum())
                    entry_vwap = _vwap(entries["price"], entries["qty"])
                    exit_vwap  = _vwap(exits["price"], exits["qty"].abs())
                    realized   = (exit_vwap - entry_vwap) * qty  # long P&L
                else:
                    entries = lot_df[lot_df["signed_qty"] < 0]    # shorts open with sells
                    exits   = lot_df[lot_df["signed_qty"] > 0]    # close with buys
                    qty = float(entries["qty"].sum())
                    entry_vwap = _vwap(entries["price"], entries["qty"])
                    exit_vwap  = _vwap(exits["price"], exits["qty"])
                    realized   = (entry_vwap - exit_vwap) * qty   # short P&L

                fees = float(lot_df.get("fee", pd.Series(0.0)).sum())
                open_time  = lot_df["time"].iloc[0]
                close_time = lot_df["time"].iloc[-1]
                # ensure tz-aware UTC (your _load_fills_dataframe already gives tz-aware)
                if getattr(open_time, "tzinfo", None) is not None:
                    open_utc  = open_time.tz_convert("UTC")
                    close_utc = close_time.tz_convert("UTC")
                else:
                    open_utc  = pd.to_datetime(open_time, utc=True)
                    close_utc = pd.to_datetime(close_time, utc=True)

                days_open  = (close_utc - open_utc).total_seconds() / 86400.0
                realized_pct = (realized / (entry_vwap * qty) * 100.0) if (entry_vwap and qty) else np.nan

                # ✅ CLOSED lot (ONLY append closed here)
                out_rows.append({
                    "open_date": open_utc,
                    "open_ts":   open_utc,
                    "close_ts":  close_utc,
                    "symbol": sym,
                    "side": lot_side,
                    "status": "Closed",
                    "entry_vwap": round(entry_vwap, 4),
                    "exit_vwap": round(exit_vwap, 4),
                    "qty": int(qty),
                    "days_open": round(days_open, 2),
                    "realized_pl$": round(realized - fees, 2),
                    "realized_pl%": round(realized_pct, 2),
                    "fees$": round(fees, 2),
                    "tp": None, "sl": None,
                    "notes": "",
                })

                # reset for next lot
                lot_start_idx = None
                lot_side = None

        # --- If a lot is still open at the end, emit an Open row (genuine open case)
        if lot_start_idx is not None and cum != 0:
            lot_df = g.iloc[lot_start_idx:].copy()
            if lot_side == "Long":
                entries = lot_df[lot_df["signed_qty"] > 0]
                qty = float(entries["qty"].sum()) - float(lot_df[lot_df["signed_qty"] < 0]["qty"].sum())
                entry_vwap = _vwap(entries["price"], entries["qty"])
            else:
                entries = lot_df[lot_df["signed_qty"] < 0]
                qty = float(entries["qty"].sum()) - float(lot_df[lot_df["signed_qty"] > 0]["qty"].sum())
                qty = abs(qty)
                entry_vwap = _vwap(entries["price"], entries["qty"])

            open_time = lot_df["time"].iloc[0]
            if getattr(open_time, "tzinfo", None) is not None:
                open_utc = open_time.tz_convert("UTC")
            else:
                open_utc = pd.to_datetime(open_time, utc=True)

            days_open = (pd.Timestamp.now(tz=timezone.utc) - open_utc).total_seconds() / 86400.0

            out_rows.append({
                "open_date": open_utc,
                "open_ts":   open_utc,      # include for drill-down window
                "close_ts":  None,          # open lot has no close
                "symbol": sym,
                "side": lot_side,
                "status": "Open",
                "entry_vwap": round(entry_vwap, 4),
                "exit_vwap": None,
                "qty": int(qty),
                "days_open": round(days_open, 2),
                "realized_pl$": 0.0,
                "realized_pl%": 0.0,
                "fees$": round(float(lot_df.get("fee", pd.Series(0.0)).sum()), 2),
                "tp": None, "sl": None,
                "notes": "",
            })

    df_out = pd.DataFrame(out_rows).sort_values(["open_date","symbol"]).reset_index(drop=True)
    return df_out

def tidy_transaction_history(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure open_date is datetime
    df["open_date"] = pd.to_datetime(df["open_date"], errors="coerce")

    # Round numeric columns to 2 decimals
    num_cols = df.select_dtypes(include=["float", "int"]).columns
    df[num_cols] = df[num_cols].map(lambda x: round(x, 2) if pd.notna(x) else x)

    # Replace NaN/None with blank
    df = df.replace({None: "", np.nan: ""})

    # Rename visible headers
    rename_map = {
        "open_date":     "Date",
        "symbol":        "Ticker",
        "side":          "Side",
        "status":        "Status",
        "entry_vwap":    "Entry",
        "exit_vwap":     "Exit",
        "qty":           "Qty",
        "days_open":     "Days Open",
        "realized_pl$":  "P&L $",
        "realized_pl%":  "P&L %",
        "fees$":         "Fees $",
        # do not include tp/sl/notes
    }
    df = df.rename(columns=rename_map)

    # Drop TP/SL/Notes if they sneak through
    drop_cols = [c for c in ["tp","sl","notes","TP","SL","Notes"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Enforce column order
    order = ["Date","Ticker","Side","Status","Entry","Exit","Qty","Days Open","P&L $","P&L %","Fees $"]
    df = df[[c for c in order if c in df.columns]]

    # ✅ Sort newest → oldest
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values(["Date", "Ticker"], ascending=[False, True])
        # Back to display-friendly format
        df["Date"] = df["Date"].dt.strftime("%d/%m/%Y")

    return df

def _force_closed_if_flat(df_hist: pd.DataFrame, api: Optional[REST]) -> pd.DataFrame:
    """
    If a row is 'Open' but the symbol is NOT in current live positions, flip the status to
    Closed – Profit/Loss/Breakeven using the lot's realized P&L (when available).

    Expects the *base* history frame from build_position_transaction_history()
    (i.e., columns like: symbol, status, realized_pl$, open_date, open_ts/close_ts, etc.).
    """
    if df_hist is None or df_hist.empty:
        return df_hist

    # Build a set of symbols we currently hold (live positions)
    live_map = _live_positions_map(api) if callable(globals().get("_live_positions_map")) else {}
    live_syms = {s.upper() for s in live_map.keys()}

    out = df_hist.copy()
    # Only rows marked Open AND not in live positions
    if "symbol" not in out.columns or "status" not in out.columns:
        return out

    mask = (out["status"].astype(str).str.lower().eq("open")) & \
           (~out["symbol"].astype(str).str.upper().isin(live_syms))

    if not mask.any():
        return out

    # Use realized_pl$ if present to classify; otherwise just mark Closed
    if "realized_pl$" in out.columns:
        pl = pd.to_numeric(out.loc[mask, "realized_pl$"], errors="coerce").fillna(0.0)
        labels = np.where(pl > 1e-4, "Closed – Profit",
                  np.where(pl < -1e-4, "Closed – Loss", "Closed – Breakeven"))
        out.loc[mask, "status"] = labels
    else:
        out.loc[mask, "status"] = "Closed"

    # Optionally stamp close_ts as now for forced-closed rows that lack it
    if "close_ts" in out.columns:
        now_utc = pd.Timestamp.now(tz=timezone.utc)
        out.loc[mask & out["close_ts"].isna(), "close_ts"] = now_utc

    return out

def render_transaction_history_positions(api: Optional[REST], *, days: int | None = 180) -> None:
    st.subheader("Transaction History (per open date per ticker)")

    # Build lots DF (internal view with timestamps)
    base_df = build_position_transaction_history(api, days=days)

    # Optional: force-close rows when flat now
    base_df = _force_closed_if_flat(base_df, api)   # if you kept the earlier patch

    # -------- High-level (tidy) table --------
    tidy_df = add_closed_outcome(base_df.copy())    # classify win/loss/breakeven
    tidy_df = tidy_transaction_history(tidy_df)

    if tidy_df.empty:
        st.info("No fills from Alpaca to build history.")
        return
    num_cols = ["Entry","Exit","Qty","Days Open","P&L $","P&L %","Fees $"]
    for c in num_cols:
        if c in tidy_df.columns:
            tidy_df[c] = pd.to_numeric(tidy_df[c], errors="coerce")

    st.dataframe(
        tidy_df.style.format({
            "Entry": "{:.2f}", "Exit": "{:.2f}", "Qty": "{:.0f}", "Days Open": "{:.2f}",
            "P&L $": "${:,.2f}", "P&L %": "{:+.2f}%", "Fees $": "${:,.2f}"
        }, na_rep=""),
        width="stretch", hide_index=True
    )

    # -------- Drill-down: per-fill for each lot --------
    st.markdown("**Details by lot (drill-down)**")
    fills_all = _pull_all_fills_df(api)  # already normalized to ts/symbol/side/qty/price
    if fills_all.empty:
        st.caption("No individual fill activity available.")
        return

    # Build a join key between base_df rows and tidy_df rows (date + symbol)
    lot_rows = base_df.copy()
    lot_rows["Date"]   = pd.to_datetime(lot_rows["open_date"]).dt.strftime("%d/%m/%Y")
    lot_rows["Ticker"] = lot_rows["symbol"]

    # For display order, follow the tidy (already latest→oldest)
    display_keys = tidy_df[["Date","Ticker","Status"]].to_dict(orient="records")

    for key in display_keys:
        d = key["Date"]; sym = key["Ticker"]

        # fetch the corresponding internal row (there can be multiple if same date/ticker across lots; take first)
        match = lot_rows[(lot_rows["Date"] == d) & (lot_rows["Ticker"] == sym)]
        if match.empty:
            continue
        r = match.iloc[0]
        open_ts  = r.get("open_ts")
        close_ts = r.get("close_ts")

        with st.expander(f"{d} · {sym} · {key['Status']}"):
            # per-fill slice
            sub = _fills_for_lot_window(fills_all, sym, open_ts, close_ts)

            # quick header stats (vwap/qty/fees)
            entry = r.get("entry_vwap", None); exitp = r.get("exit_vwap", None)
            qty   = r.get("qty", None); fees  = r.get("fees$", 0.0)
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Entry VWAP", f"{entry:.2f}" if entry else "—")
            with col2: st.metric("Exit VWAP",  f"{exitp:.2f}" if pd.notna(exitp) else "—")
            with col3: st.metric("Qty",        f"{qty:.0f}" if qty else "—")
            with col4: st.metric("Fees",       f"${fees:,.2f}")

            if sub.empty:
                st.caption("No fills matched for this lot window.")
            else:
                st.dataframe(
                    sub.style.format({
                        "Qty": "{:.0f}", "Price": "{:.2f}", "Notional": "${:,.2f}"
                    }), width="stretch", hide_index=True
                )

def add_closed_outcome(df: pd.DataFrame, breakeven_tol: float = 1e-4) -> pd.DataFrame:
    """
    Convert Status 'Closed' into 'Closed – Profit/Loss/Breakeven' based on realized_pl$.
    breakeven_tol avoids classifying tiny rounding noise as win/loss.
    """
    if "realized_pl$" in df.columns:          # before tidy()
        pl_col = "realized_pl$"
        status_col = "status"
    else:                                      # after tidy() renames
        pl_col = "P&L $"
        status_col = "Status"

    pl = pd.to_numeric(df[pl_col], errors="coerce").fillna(0.0)

    def classify(row_pl, row_status):
        if row_status != "Closed":
            return row_status
        if row_pl > breakeven_tol:
            return "Closed – Profit"
        if row_pl < -breakeven_tol:
            return "Closed – Loss"
        return "Closed – Breakeven"

    df[status_col] = [classify(p, s) for p, s in zip(pl, df[status_col])]
    return df

# --- SPY vs QuantML: helpers ---
@st.cache_data(ttl=600, show_spinner=False)
def _lookback_for(period: str) -> int:
    p = (period or "1M").upper()
    if p == "1D": return 2
    if p == "2D": return 5      # pad for weekends/holidays
    if p == "1W": return 8
    if p == "1M": return 32
    if p == "3M": return 95
    return 365

def _tf_for(period: str) -> str:
    p = (period or "").upper()
    if p in {"1D", "2D"}: return "5Min"   # intraday for both 1D and 2D
    if p == "1W": return "1D"
    return "1D"

def _parse_init_from_client_id(client_order_id: str) -> tuple[float | float, float | float]:
    """
    Extract (init_tp, init_sl) from client_order_id formatted like:
      QML|tp=123.45|sl=67.89|sym=ABC
    Returns (nan, nan) if not present.
    """
    import math, re
    if not client_order_id:
        return (math.nan, math.nan)
    s = str(client_order_id)
    m_tp = re.search(r"\btp=([0-9]+(?:\.[0-9]+)?)", s)
    m_sl = re.search(r"\bsl=([0-9]+(?:\.[0-9]+)?)", s)
    try:
        tp = float(m_tp.group(1)) if m_tp else math.nan
        sl = float(m_sl.group(1)) if m_sl else math.nan
        return (tp, sl)
    except Exception:
        return (math.nan, math.nan)

def _feed_for_account(api: Optional[REST]) -> str:
    # Paper → IEX; Live → SIP. If anything fails, default to IEX (works on paper).
    try:
        acct = api.get_account()
        is_live = bool(getattr(acct, "trading_blocked", False)) is False and \
                  "paper" not in str(getattr(api, "base_url", "")).lower()
        return "sip" if is_live else "iex"
    except Exception:
        return "iex"

@st.cache_data(ttl=600, show_spinner=False)
def _get_symbol_bars(_api: Optional[REST], symbol: str, timeframe: str, *, days: int) -> pd.DataFrame:
    """
    Fetch bars for `symbol` over the last `days`, returning columns: ts (UTC), close (float).
    Robust to old/new Alpaca SDKs and paper/live feed differences.
    """
    if _api is None:
        return pd.DataFrame(columns=["ts", "close"])

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 3)  # pad for weekends/holidays
    feed = _feed_for_account(_api)  # "iex" on paper, "sip" on live

    # Map string timeframe → proper TimeFrame object when possible
    tf_obj = timeframe
    try:
        # inside _get_symbol_bars(), in the TimeFrame mapping block
        from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
        s = str(timeframe).lower()
        if s in {"1d", "day", "d"}:
            tf_obj = TimeFrame.Day
        elif s in {"5min", "5m"}:
            try:
                tf_obj = TimeFrame(5, TimeFrameUnit.Minute)
            except Exception:
                tf_obj = "5Min"
        elif s in {"1min", "1m"}:                            # <-- NEW
            try:
                tf_obj = TimeFrame(1, TimeFrameUnit.Minute)
            except Exception:
                tf_obj = "1Min"
    except Exception:
        # Older SDK; keep the string (e.g., "5Min" / "1D")
        tf_obj = timeframe

    # ---- Primary attempt
    try:
        bars = _api.get_bars(
            symbol,
            tf_obj,
            start.isoformat(),
            end.isoformat(),
            adjustment="raw",
            feed=feed,
        )
        # DataFrame path (newer SDK)
        if hasattr(bars, "df"):
            df = bars.df.reset_index()
            tcol = "timestamp" if "timestamp" in df.columns else "time"
            close_col = "close" if "close" in df.columns else "c"
            out = df.rename(columns={tcol: "ts", close_col: "close"})
            if "symbol" in out.columns:
                out = out[out["symbol"].astype(str).str.upper().eq(symbol.upper())]
            out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
            return out[["ts", "close"]].dropna()
        # Iterable path (older SDK)
        rows = []
        for b in bars:
            ts = getattr(b, "t", getattr(b, "timestamp", getattr(b, "time", None)))
            px = getattr(b, "c", getattr(b, "close", None))
            if ts is None or px is None:
                continue
            rows.append({"ts": pd.to_datetime(str(ts), utc=True, errors="coerce"), "close": float(px)})
        if rows:
            return pd.DataFrame(rows).dropna()
    except Exception:
        pass

    # ---- Final fallback: Yahoo Finance (if installed)
    try:
        y = _get_symbol_bars_yahoo(symbol, timeframe, days=days)
        if not y.empty:
            return y
    except Exception:
        pass

    return pd.DataFrame(columns=["ts","close"])

def _symbol_returns_5min_robust(api: Optional[REST], symbol: str, *, days: int = 5) -> pd.DataFrame:
    """
    Try 5Min first; if empty, pull 1Min and resample to 5Min.
    Always returns tidy ['ts','ret'] in UTC (%).
    """
    z = _get_symbol_bars(api, symbol, "5Min", days=days).sort_values("ts")
    if z.empty:
        z = _get_symbol_bars(api, symbol, "1Min", days=days).sort_values("ts")
        if not z.empty:
            z = (z.set_index("ts").resample("5min").last().dropna().reset_index())

    if z.empty:
        return pd.DataFrame(columns=["ts", "ret"])

    z["ts"]  = pd.to_datetime(z["ts"], utc=True, errors="coerce")
    z["ret"] = z["close"].pct_change() * 100.0
    return z[["ts", "ret"]].dropna()

def _daily_returns_from_equity(ph: pd.DataFrame, period: str) -> pd.DataFrame:
    """Return tidy returns with a tz-aware timestamp column 'ts' and 'ret' (%)"""
    if ph is None or ph.empty:
        return pd.DataFrame(columns=["ts", "ret"])
    s = ph[["ts", "equity"]].dropna().sort_values("ts").copy()
    s["ret"] = s["equity"].pct_change() * 100.0
    s["ts"]  = pd.to_datetime(s["ts"], utc=True, errors="coerce")
    return s[["ts", "ret"]].dropna()


def _daily_returns_for_symbol(api: Optional[REST], symbol: str, period: str) -> pd.DataFrame:
    """Symbol returns as tidy frame with 'ts' (UTC) and 'ret' (%)"""
    tf   = _tf_for(period)
    days = _lookback_for(period)
    z = _get_symbol_bars(api, symbol, tf, days=days).sort_values("ts")
    if z.empty:
        return pd.DataFrame(columns=["ts", "ret"])
    z["ts"]  = pd.to_datetime(z["ts"], utc=True, errors="coerce")
    z["ret"] = z["close"].pct_change() * 100.0
    return z[["ts", "ret"]].dropna()

def render_spy_vs_quantml_daily(api: Optional[REST], period: str = "1M") -> None:
    """
    SPY vs QuantML — daily returns
    Top: intraday lines (2D) or 1M close-to-close lines
    Bottom: grouped bars of close-of-business daily returns (QuantML vs SPY),
            plus a thin Δ-line (QuantML − SPY) on a secondary axis.
    """
    import numpy as np
    from zoneinfo import ZoneInfo

    st.markdown("**SPY vs QuantML — daily returns**")

    ui_period = st.radio(
        "X-axis span",
        options=["2D", "1M"],
        horizontal=True,
        label_visibility="collapsed",
        key="spy_vs_qm_period"
    )

    et = ZoneInfo("US/Eastern")

    if ui_period == "2D":
        # ---- Intraday (market hours only), last two ET sessions ----
        lookback_days = _lookback_for("2D")

        # QuantML equity → try 1W/5Min, then 1D/5Min, then with extended_hours=False
        ph = get_portfolio_history_df(api, period="1W", timeframe="5Min")
        if ph.empty:
            ph = get_portfolio_history_df(api, period="1D", timeframe="5Min")
        if ph.empty:
            ph = get_portfolio_history_df(api, period="1W", timeframe="5Min", extended_hours=False)

        qm = _daily_returns_from_equity(ph, ui_period)     # ['ts','ret'] (%)

        # SPY robust intraday fetch (5Min with 1Min→5Min resample fallback)
        spy = _symbol_returns_5min_robust(api, "SPY", days=lookback_days)

        if qm.empty or spy.empty:
            st.info("SPY or portfolio data not available for 2D.")
            return

        # ---- Convert to ET & keep regular session only ----
        qm["ts"]  = qm["ts"].dt.tz_convert(et)
        spy["ts"] = spy["ts"].dt.tz_convert(et)

        mopen, mclose = pd.to_datetime("09:30").time(), pd.to_datetime("16:00").time()
        spy = spy[(spy["ts"].dt.time >= mopen) & (spy["ts"].dt.time <= mclose)]
        qm  = qm[(qm["ts"].dt.time  >= mopen) & (qm["ts"].dt.time  <= mclose)]

        # Keep the last two ET sessions based on SPY calendar
        spy["date_et"] = spy["ts"].dt.date
        last2 = sorted(spy["date_et"].unique())[-2:]
        spy = spy[spy["date_et"].isin(last2)]

        qm["date_et"] = qm["ts"].dt.date
        if qm["date_et"].max() > spy["date_et"].max():
            # Overlay pre-market day onto prior SPY session
            qm["ts"] -= pd.Timedelta(days=1)
            qm["date_et"] = qm["ts"].dt.date

        # Align to SPY 5-min grid and only compare where both exist
        grid = (spy[["ts"]]
                .assign(ts=lambda d: d["ts"].dt.floor("5min"))
                .drop_duplicates()
                .sort_values("ts"))
        qm_g  = grid.merge(qm.rename(columns={"ret": "QuantML"}), on="ts", how="left")
        spy_g = grid.merge(spy.rename(columns={"ret": "SPY"}),      on="ts", how="left")
        merged = (qm_g.merge(spy_g[["ts","SPY"]], on="ts", how="left")
                       .dropna(subset=["QuantML","SPY"])
                       .assign(date_et=lambda d: d["ts"].dt.date)
                       .sort_values("ts"))

        if merged.empty:
            st.info("No overlapping timestamps in market-hours window.")
            return

        # ── Top LINE CHART (intraday, split by session) ─────────────────────
        fig = go.Figure()
        for d, g in merged.groupby("date_et", sort=True):
            fig.add_trace(go.Scatter(
                x=g["ts"], y=g["QuantML"], mode="lines",
                name=f"QuantML (daily %) · {d}",
                line=dict(width=2, color=BRAND['accent']), connectgaps=False,
                hovertemplate="%{x}<br>QuantML % %{y:.2f}<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=g["ts"], y=g["SPY"], mode="lines",
                name=f"SPY (daily %) · {d}",
                line=dict(width=2, color=BRAND['primary']), connectgaps=False,
                hovertemplate="%{x}<br>SPY % %{y:.2f}<extra></extra>"
            ))

        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="rgba(0,0,0,.35)")
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat","mon"]),
                                      dict(pattern="hour", bounds=[16, 9.5])])
        fig.update_layout(height=260, margin=dict(l=8, r=8, t=6, b=6),
                          xaxis_title=None, yaxis_title="Daily return (%)",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig, config={**PLOTLY_CONFIG, "responsive": True}, use_container_width=True)

        # ── Bottom BAR CHART (daily close-of-business return summary) ───────
        daily_summary = (
            merged.groupby("date_et", as_index=False)[["QuantML", "SPY"]].mean()
        )
        x_dates = daily_summary["date_et"]
        y_q = daily_summary["QuantML"]
        y_s = daily_summary["SPY"]

    else:
        # ---- 1-month view (close-to-close daily returns) ----
        ph  = get_portfolio_history_df(api, period="1M")

        qm  = (_ensure_ts_and_date(_daily_returns_from_equity(ph, ui_period))
               .groupby("date", as_index=False).last()
               .rename(columns={"ret": "QuantML"}))

        spy = (_ensure_ts_and_date(_daily_returns_for_symbol(api, "SPY", ui_period))
               .groupby("date", as_index=False).last()
               .rename(columns={"ret": "SPY"}))

        merged = qm.merge(spy, on="date", how="inner").sort_values("date")
        if 'START_1M_FROM' in globals() and START_1M_FROM:
            merged = merged[merged["date"] >= START_1M_FROM]

        if merged.empty:
            st.info("No overlapping business days between SPY and portfolio in 1M window.")
            return

        # ── Top LINE CHART (daily) ───────────────────────────────────────────
        x   = merged["date"]
        y_q = merged["QuantML"]
        y_s = merged["SPY"]

        # Clip outliers for nicer line plot
        def _clip(s, qlo=0.01, qhi=0.99):
            s = pd.to_numeric(s, errors="coerce")
            if s.notna().sum() < 5: return s
            lo, hi = np.nanquantile(s, [qlo, qhi])
            return s.clip(lo, hi)

        y_q_clip = _clip(y_q); y_s_clip = _clip(y_s)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y_q_clip, mode="lines", name="QuantML (daily %)",
            line=dict(width=2, color=BRAND["accent"]),
            hovertemplate="%{x}<br>QuantML % %{y:.2f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y_s_clip, mode="lines", name="SPY (daily %)",
            line=dict(width=2, color=BRAND["primary"]),
            hovertemplate="%{x}<br>SPY % %{y:.2f}<extra></extra>"
        ))
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="rgba(0,0,0,.35)")
        fig.update_layout(
            height=260, margin=dict(l=8, r=8, t=6, b=6),
            xaxis_title=None, yaxis_title="Daily return (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
        )
        st.plotly_chart(fig, config={**PLOTLY_CONFIG, "responsive": True}, use_container_width=True)

        # ── Bottom BAR CHART (daily close-of-business) ───────────────────────
        x_dates = merged["date"]
        y_q = merged["QuantML"]
        y_s = merged["SPY"]

    # === Professional BAR CHART (grouped) with Δ-line ========================
    delta = (pd.to_numeric(y_q, errors="coerce") - pd.to_numeric(y_s, errors="coerce"))

    bar_fig = go.Figure()

    # QuantML bars
    bar_fig.add_trace(go.Bar(
        x=x_dates, y=y_q,
        name="QuantML (daily %)",
        marker_color=BRAND["accent"],
        marker_line_width=1.1, marker_line_color="rgba(0,0,0,0.25)",
        hovertemplate="%{x}<br>QuantML % %{y:.2f}<extra></extra>"
    ))
    # SPY bars
    bar_fig.add_trace(go.Bar(
        x=x_dates, y=y_s,
        name="SPY (daily %)",
        marker_color=BRAND["primary"],
        marker_line_width=1.1, marker_line_color="rgba(0,0,0,0.25)",
        hovertemplate="%{x}<br>SPY % %{y:.2f}<extra></extra>"
    ))
    # Δ-line (secondary axis)
    bar_fig.add_trace(go.Scatter(
        x=x_dates, y=delta,
        name="Δ (QML − SPY)",
        mode="lines+markers",
        line=dict(width=2, color="#64748B"),
        marker=dict(size=5),
        yaxis="y2",
        hovertemplate="%{x}<br>Δ %{y:.2f} pp<extra></extra>"
    ))

    # Layout polish
    bar_fig.update_layout(
        barmode="group",
        bargap=0.25,
        height=260,
        margin=dict(l=8, r=8, t=6, b=6),
        xaxis_title=None,
        yaxis_title="Daily return (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="rgba(0,0,0,0.3)"),
        # Secondary axis for Δ (keep subtle range)
        yaxis2=dict(overlaying="y", side="right", title="Δ (pp)", showgrid=False, zeroline=True,
                    zerolinewidth=1, zerolinecolor="rgba(0,0,0,0.25)"),
    )

    st.plotly_chart(bar_fig, config={**PLOTLY_CONFIG, "responsive": True})
    # --- Caption explaining the SPY vs QuantML comparison ---
    st.caption(
        "Each bar shows the daily percentage return for QuantML (green) and SPY (blue). "
        "The thin grey Δ-line tracks QuantML’s relative outperformance or underperformance "
        "versus SPY for that day."
    )


# ===================== Alpaca Portfolio History =====================
@st.cache_data(ttl=60, show_spinner=False)
def get_portfolio_history_df(_api: Optional[REST],
                             period: str = "1D",
                             timeframe: Optional[str] = None,
                             extended_hours: bool = True) -> pd.DataFrame:
    """
    Wraps Alpaca /v2/account/portfolio/history into a tidy DataFrame.
    """
    if _api is None:
        return pd.DataFrame()

    # 🔧 Normalize "1Y" -> "1A" for Alpaca
    alpaca_period = _normalize_period(period)

    # Sensible timeframe defaults
    if timeframe:
        # Sensible timeframe defaults (use only Alpaca‑valid values)
        tf = timeframe or _default_timeframe_for_period(period)

    else:
        if alpaca_period == "1D":
            tf = "5Min"
        elif alpaca_period.endswith("W"):
            tf = "30Min"
        else:
            tf = "1D"

    try:
        ph = _api.get_portfolio_history(period=alpaca_period, timeframe=tf, extended_hours=extended_hours)
    except TypeError:
        ph = _api.get_portfolio_history(period=alpaca_period, timeframe=tf)
    # Lists/arrays coming from SDK (sometimes numpy arrays)
    ts = (getattr(ph, "timestamp", None) or getattr(ph, "time", None) or
          getattr(ph, "ts", None) or ph.get("timestamp", []))
    eq = (getattr(ph, "equity", None) or ph.get("equity", []))
    pl = (getattr(ph, "profit_loss", None) or ph.get("profit_loss", []))

    df = pd.DataFrame({
        "ts": pd.to_datetime(ts, unit="s", utc=True, errors="coerce"),
        "equity": pd.to_numeric(pd.Series(eq), errors="coerce")   # Series guards dtype
    }).dropna()

    # Use Series to safely call .fillna no matter if source was list/array
    if len(pl) == len(df):
        pl_series = pd.to_numeric(pd.Series(pl), errors="coerce").fillna(0.0)
        df["pl"] = pl_series.to_numpy()
    else:
        # Fallback: compute diff of equity
        df["pl"] = df["equity"].diff().fillna(0.0)

    df["ret"] = (df["equity"] / df["equity"].iloc[0]) - 1.0
    df["ret_pct"] = df["ret"] * 100.0
    return df

@st.cache_data(ttl=300, show_spinner=False)
def compute_period_returns(_api: Optional[REST]) -> dict:
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("US/Eastern")
    except Exception:
        tz = timezone.utc

    df = get_portfolio_history_df(_api, period="1A", timeframe="1D")
    if df.empty:
        return {"MTD": np.nan, "QTD": np.nan, "YTD": np.nan}

    last_ts = df["ts"].iloc[-1].to_pydatetime().astimezone(tz)

    # Start dates
    m_start = last_ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    q_month = ((last_ts.month - 1) // 3) * 3 + 1
    q_start = last_ts.replace(month=q_month, day=1, hour=0, minute=0, second=0, microsecond=0)
    y_start = last_ts.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

    def perf_from(start_dt):
        start_dt = pd.Timestamp(start_dt)
        if start_dt.tz is None:
            start_dt = start_dt.tz_localize(tz)
        start_dt = start_dt.tz_convert("UTC")
        s = df[df["ts"] >= start_dt]
        if len(s) >= 2:
            start, end = float(s["equity"].iloc[0]), float(s["equity"].iloc[-1])
            if start > 0 and np.isfinite(start):
                return float((end / start - 1.0) * 100.0)
        return np.nan  # prevents +inf%

    return {
        "MTD": perf_from(m_start),
        "QTD": perf_from(q_start),
        "YTD": perf_from(y_start),
    }

def _sort_df_green_amber_red(df: pd.DataFrame, pct_col: str) -> pd.DataFrame:
    """
    Sort DataFrame rows by:
      1) Green (≥0%) → descending (best first)
      2) Amber (-0.8% ≤ x < 0%) → ascending (closest to 0 first)
      3) Red (< -0.8%) → descending (worst first)
    """
    z = df.copy()
    v = pd.to_numeric(z[pct_col], errors="coerce").fillna(0.0)
    grp = v.apply(_traffic_group)

    # Within-group order logic
    within = np.select(
        [
            grp.eq(0),  # green
            grp.eq(1),  # amber
            grp.eq(2),  # red
        ],
        [
            -v,          # green: high → low
            v.abs(),     # amber: closest to 0 first
            -v,          # red: low → high (descending negative)
        ],
        default=v,
    )

    z["__grp"] = grp
    z["__within"] = within
    z = z.sort_values(["__grp", "__within"]).drop(columns=["__grp", "__within"])
    return z


def _max_drawdown_pct(equity: pd.Series) -> float:
    """Max drawdown in %, computed on a series of equity values."""
    if equity is None or len(equity) == 0:
        return 0.0
    s = pd.to_numeric(pd.Series(equity), errors="coerce").dropna()
    if s.empty:
        return 0.0
    running_max = s.cummax()
    dd = s / running_max - 1.0
    return float(dd.min() * 100.0)


@st.cache_resource
def _load_clock_sources() -> tuple[Optional[str], Optional[str]]:
    """
    Returns (js_text, singlefile_html_text). Missing items are None.
    """
    js = None
    for p in _CLOCK_JS_CANDIDATES:
        fp = Path(p)
        if fp.exists():
            js = fp.read_text(encoding="utf-8")
            break

    single = None
    for p in _SINGLEFILE_CLOCK_HTML:
        fp = Path(p)
        if fp.exists():
            single = fp.read_text(encoding="utf-8")
            break

    return js, single

from pathlib import Path
import base64, uuid
from textwrap import dedent
import streamlit.components.v1 as components

def render_quantml_clock(
    size: int = 220,
    tz: str = "Europe/Dublin",
    title: str = "Dublin",
    show_seconds: bool = True,
    is_24h: bool = True,
    logo_path: str | None = "Clock/QuantML.png",
    logo_scale: float = 0.55
) -> None:
    """Canvas clock with QUANTML logo. One time sample drives both analog & digital."""
    from textwrap import dedent

    # ---- load logo (inline as base64) ----
    logo_b64 = ""
    try:
        if logo_path and Path(logo_path).exists():
            logo_b64 = base64.b64encode(Path(logo_path).read_bytes()).decode("ascii")
        else:
            logo_b64 = load_logo_b64()  # fall back to candidates / secrets
    except Exception:
        logo_b64 = ""

    canvas_id = f"qmclk_canvas_{st.session_state.get('_clk', 0)}"
    time_id   = f"qmclk_time_{st.session_state.get('_clk', 0)}"
    date_id   = f"qmclk_date_{st.session_state.get('_clk', 0)}"
    st.session_state["_clk"] = st.session_state.get("_clk", 0) + 1

    h = size + 125  # room for labels

    html = dedent(f"""
    <div style="display:flex;flex-direction:column;align-items:center;gap:6px;">
      <canvas id="{canvas_id}" width="{size}" height="{size}" style="
        width:{size}px;height:{size}px;border-radius:16px;background:#0f172a;box-shadow:0 6px 20px rgba(0,0,0,.25);
      "></canvas>
      <div style="font:700 26px system-ui; color:navy; letter-spacing:0.5px" id="{time_id}"></div>
      <div style="font:500 16px system-ui; color:navy">{title}</div>
      <div style="font:500 16px system-ui; color:navy" id="{date_id}"></div>
    </div>
    <script>
    (function(){{
      const cssW = {size}, cssH = {size}, LOGO_SCALE = {logo_scale};
      const tz = "{tz}", showSeconds = {('true' if show_seconds else 'false')}, is24h = {('true' if is_24h else 'false')};
      const canvas = document.getElementById("{canvas_id}");
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      canvas.width = cssW*dpr; canvas.height = cssH*dpr; canvas.style.width = cssW+"px"; canvas.style.height = cssH+"px";
      const ctx = canvas.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
      const cx = cssW/2, cy = cssH/2, R = Math.min(cx,cy)-6, OFF = -Math.PI/2;

      // inline logo
      const logoData = "{logo_b64}";
      let logoImg = null, logoReady = false;
      if (logoData) {{ logoImg = new Image(); logoImg.onload = () => logoReady = true; logoImg.src = "data:image/png;base64," + logoData; }}

      function nowInTZ() {{ return new Date(new Date().toLocaleString('en-US', {{ timeZone: tz }})); }}
      function pad(n) {{ return String(n).padStart(2,'0'); }}

      function sampleParts() {{
        const d = nowInTZ();
        const h = d.getHours(), m = d.getMinutes(), s = d.getSeconds(), ms = d.getMilliseconds();
        const weekday = d.toLocaleString('en-GB', {{ weekday:'short', timeZone: tz }});
        const mon = d.toLocaleString('en-GB', {{ month:'short', timeZone: tz }});
        const day = pad(d.getDate()), year = d.getFullYear();
        return {{ h, m, s, ms, dateStr: `${{weekday}}, ${{day}} ${{mon}} ${{year}}` }};
      }}

      function drawFace() {{
        ctx.clearRect(0,0,cssW,cssH);
        for (let i = 0; i < 60; i++) {{
          const a = (Math.PI/30)*i + OFF;
          const r1 = R*(i%5===0 ? 0.82 : 0.88), r2 = R*0.97;
          ctx.beginPath();
          ctx.moveTo(cx + r1*Math.cos(a), cy + r1*Math.sin(a));
          ctx.lineTo(cx + r2*Math.cos(a), cy + r2*Math.sin(a));
          ctx.lineWidth = (i%5===0) ? 2.4 : 1.2;
          ctx.strokeStyle = "rgba(180,197,255,0.9)";
          ctx.stroke();
        }}
        ctx.beginPath(); ctx.arc(cx,cy,R*0.99,0,Math.PI*2);
        ctx.strokeStyle = "rgba(79,70,229,0.6)"; ctx.lineWidth = 2; ctx.stroke();

        const L = Math.min(cssW, cssH) * LOGO_SCALE;
        if (logoReady) {{
          ctx.save(); ctx.globalAlpha = 0.95; ctx.drawImage(logoImg, cx - L/2, cy - L/2, L, L); ctx.restore();
        }} else {{
          // visible fallback so you know the logo didn't load
          ctx.save();
          ctx.fillStyle = "rgba(226,232,255,0.18)";
          ctx.font = "bold 16px system-ui,-apple-system,Segoe UI,Roboto";
          ctx.textAlign = "center"; ctx.textBaseline = "middle";
          ctx.fillText("QUANTML", cx, cy);
          ctx.restore();
        }}
      }}  // <-- close drawFace()

      function hand(angle, len, w, col) {{
        ctx.save(); ctx.translate(cx,cy); ctx.rotate(angle + OFF);
        ctx.beginPath(); ctx.moveTo(-R*0.08, 0); ctx.lineTo(len, 0);
        ctx.lineWidth = w; ctx.lineCap = "round"; ctx.strokeStyle = col; ctx.stroke(); ctx.restore();
      }}

      function drawHandsFromParts(p) {{
        const pi = Math.PI;
        const hrA  = (pi/6)  * ((p.h % 12) + p.m/60 + p.s/3600);
        const minA = (pi/30) * (p.m + p.s/60);
        const secA = (pi/30) *  p.s;
        hand(hrA,  R*0.50, 5,  "#9DB2FF");
        hand(minA, R*0.72, 3.4,"#9DB2FF");
        if (showSeconds) hand(secA, R*0.78, 2, "#4F7BFF");
        ctx.beginPath(); ctx.arc(cx,cy,6,0,pi*2); ctx.fillStyle="#99A8FF"; ctx.fill();
        ctx.beginPath(); ctx.arc(cx,cy,3,0,pi*2); ctx.fillStyle="#335CFF"; ctx.fill();
      }}

      function drawDigitalFromParts(p) {{
        const hh = is24h ? pad(p.h) : pad(((p.h % 12) || 12));
        const ampm = is24h ? "" : (p.h < 12 ? " AM" : " PM");
        document.getElementById("{time_id}").textContent =
          showSeconds ? `${{hh}}:${{pad(p.m)}}:${{pad(p.s)}}${{ampm}}`
                      : `${{hh}}:${{pad(p.m)}}${{ampm}}`;
        document.getElementById("{date_id}").textContent = p.dateStr;
      }}

      function scheduleNext(ms) {{ setTimeout(tick, 1000 - (ms % 1000)); }}
      function tick() {{
        const p = sampleParts();
        drawFace(); drawHandsFromParts(p); drawDigitalFromParts(p);
        scheduleNext(p.ms);
      }}
      tick();
    }})();
    </script>
    """)
    components.html(html, height=h, scrolling=False)

def render_banner_clock(*,
                        size=200, tz="Europe/Dublin", title="Dublin",
                        show_seconds=True, is_24h=True,
                        logo_path=None, logo_scale: float = 0.55) -> None:
    """
    Analog + digital clock locked to the same TZ sample each tick.
    The analog hands are computed from the SAME h/m/s used to render the digital text.
    """
    import base64, uuid
    from pathlib import Path
    from textwrap import dedent
    import streamlit.components.v1 as components

    # Inline the logo so it renders on Streamlit Cloud
    logo_b64 = ""
    try:
        if logo_path and Path(logo_path).exists():
            logo_b64 = base64.b64encode(Path(logo_path).read_bytes()).decode("ascii")
    except Exception:
        logo_b64 = ""

    h = int(size + 120)
    uid = uuid.uuid4().hex[:8]
    canvas_id = f"qmAnalog-{uid}"
    time_id   = f"qmTime-{uid}"
    date_id   = f"qmDate-{uid}"

    components.html(dedent(f"""
    <div style="display:flex;flex-direction:column;align-items:center;">
      <div style="background:#0b1220;border-radius:18px;padding:12px 16px 18px 16px;box-shadow:0 10px 30px rgba(0,0,0,.35);">
        <canvas id="{canvas_id}" style="display:block;width:{size}px;height:{size}px;"></canvas>
        <div style="margin-top:10px;text-align:center;font-family:ui-sans-serif;color:navy;">
          <div id="{time_id}" style="font-weight:800;font-size:22px;">loading…</div>
          <div id="{date_id}" style="margin-top:2px;font-size:14px;opacity:.85;">—</div>
          <div style="font-size:13px;opacity:.75;">{title}</div>
        </div>
      </div>
    </div>

    <script>
    (function(){{
      const tz = "{tz}";
      const showSeconds = {str(show_seconds).lower()};
      const is24h = {str(is_24h).lower()};
      const cssW = {size}, cssH = {size};
      const LOGO_SCALE = {logo_scale};

      const canvas = document.getElementById("{canvas_id}");
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      canvas.width = cssW*dpr; canvas.height = cssH*dpr;
      canvas.style.width = cssW+"px"; canvas.style.height = cssH+"px";
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr,0,0,dpr,0,0);
      const cx = cssW/2, cy = cssH/2, R = Math.min(cx,cy)-6;

      // Inline logo
      const logoData = "{logo_b64}";
      let logoImg=null, logoReady=false;
      if (logoData) {{ logoImg = new Image(); logoImg.onload=()=>logoReady=true; logoImg.src="data:image/png;base64,"+logoData; }}

      // ---- one true time source (used by both analog & digital) ----
      function nowInTZ(){{
        return new Date(new Date().toLocaleString('en-US', {{ timeZone: tz }}));
      }}
      function pad(n){{return String(n).padStart(2,'0');}}

      function sampleParts(){{
        const d = nowInTZ();                 // <— single sample per tick
        const h = d.getHours(), m = d.getMinutes(), s = d.getSeconds(), ms = d.getMilliseconds();
        const weekday = d.toLocaleString('en-GB', {{ weekday:'short', timeZone: tz }});
        const mon  = d.toLocaleString('en-GB', {{ month:'short',  timeZone: tz }});
        const day  = pad(d.getDate());
        const year = d.getFullYear();
        return {{
          h, m, s, ms,
          dateStr: `${{weekday}}, ${{day}} ${{mon}} ${{year}}`
        }};
      }}

      function drawFace(){{
        ctx.clearRect(0,0,cssW,cssH);
        for(let i=0;i<60;i++) {{
          const a=(Math.PI/30)*i;
          const r1=R*(i%5===0?0.82:0.88), r2=R*0.97;
          ctx.beginPath();
          ctx.moveTo(cx+r1*Math.cos(a), cy+r1*Math.sin(a));
          ctx.lineTo(cx+r2*Math.cos(a), cy+r2*Math.sin(a));
          ctx.lineWidth=(i%5===0)?2.4:1.2;
          ctx.strokeStyle="rgba(180,197,255,0.9)";
          ctx.stroke();
        }}
        ctx.beginPath(); ctx.arc(cx,cy,R*0.99,0,Math.PI*2);
        ctx.strokeStyle="rgba(79,70,229,0.6)"; ctx.lineWidth=2; ctx.stroke();

        const L = Math.min(cssW, cssH) * LOGO_SCALE;
        if (logoReady) {{
          ctx.save(); ctx.globalAlpha=0.95;
          ctx.drawImage(logoImg, cx-L/2, cy-L/2, L, L);
          ctx.restore();
        }} else {{
          ctx.save();
          ctx.fillStyle="rgba(226,232,255,0.18)";
          ctx.font="bold 14px system-ui,-apple-system,Segoe UI,Roboto";
          ctx.textAlign="center"; ctx.textBaseline="middle";
          ctx.fillText("QUANTML", cx, cy);
          ctx.restore();
        }}
      }}

      function hand(a,len,w,col){{ ctx.save(); ctx.translate(cx,cy); ctx.rotate(a);
        ctx.beginPath(); ctx.moveTo(-R*0.08,0); ctx.lineTo(len,0);
        ctx.lineWidth=w; ctx.lineCap="round"; ctx.strokeStyle=col; ctx.stroke(); ctx.restore(); }}

      // compute analog angles **from the same h/m/s used for digital**
      function drawHandsFromParts(p){{
        const pi=Math.PI;
        const h = p.h, m = p.m, s = p.s;
        const hrA  = (pi/6)  * ((h%12) + m/60 + s/3600);
        const minA = (pi/30) * (m + s/60);
        const secA = (pi/30) * s;

        hand(hrA,  R*0.50, 5,  "#9DB2FF");
        hand(minA, R*0.72, 3.4,"#9DB2FF");
        if (showSeconds) hand(secA, R*0.78, 2,  "#4F7BFF");
        ctx.beginPath(); ctx.arc(cx,cy,6,0,pi*2); ctx.fillStyle="#99A8FF"; ctx.fill();
        ctx.beginPath(); ctx.arc(cx,cy,3,0,pi*2); ctx.fillStyle="#335CFF"; ctx.fill();
      }}

      function drawDigitalFromParts(p){{
        const hh = is24h ? pad(p.h) : pad(((p.h%12)||12));
        const ampm = is24h ? "" : (p.h<12? " AM" : " PM");
        const txt = showSeconds ? `${{hh}}:${{pad(p.m)}}:${{pad(p.s)}}${{ampm}}`
                                : `${{hh}}:${{pad(p.m)}}${{ampm}}`;
        document.getElementById("{time_id}").textContent = txt;
        document.getElementById("{date_id}").textContent = p.dateStr;
      }}

      // align ticks to the next second boundary to avoid drift
      function scheduleNext(ms){{
        const delay = 1000 - (ms % 1000);
        setTimeout(tick, delay);
      }}

      function tick(){{
        const p = sampleParts();      // <— ONE sample
        drawFace();                   // analog frame
        drawHandsFromParts(p);        // analog hands from p
        drawDigitalFromParts(p);      // digital from p
        scheduleNext(p.ms);
      }}

      tick();
    }})();
    </script>
    """), height=h, scrolling=False)

def enable_autorefresh(seconds: int = 60) -> None:
    """
    Reruns the Streamlit app every `seconds` seconds.
    Uses a tiny JS snippet so no extra pip package is required.
    """
    components.html(
        f"<script>setTimeout(() => window.parent.location.reload(), {seconds*1000});</script>",
        height=0,
    )

# ========= Transaction history (from Alpaca fills; FIFO realized per fill) =========
from collections import defaultdict, deque
try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("US/Eastern")
except Exception:
    _ET = None

@st.cache_data(ttl=300, show_spinner=False)
def _pull_all_fills_df(_api: Optional[REST], start: datetime | None = None) -> pd.DataFrame:
    """All FILLs since `start` (default 2000-01-01), with fees, paginated."""
    cols = ["ts","symbol","side","qty","price","fee"]
    if _api is None: return pd.DataFrame(columns=cols)
    if start is None: start = datetime(2000, 1, 1, tzinfo=timezone.utc)
    acts = _fetch_fills_paged(_api, after_iso=start.isoformat())

    rows = []
    for a in acts or []:
        sym  = (getattr(a,"symbol",None) or getattr(a,"asset_symbol",None) or "").upper()
        side = (getattr(a,"side","") or "").lower()
        def _fee(obj):
            for k in ("commission","commissions","fee","fees","trade_fees","transaction_fees"):
                v = getattr(obj,k,None)
                if v is not None:
                    try: return float(v)
                    except: pass
            return 0.0
        try:
            price = float(getattr(a,"price", getattr(a,"fill_price",0.0)) or 0.0)
            qty   = float(getattr(a,"qty",   getattr(a,"quantity",0.0)) or 0.0)
            fee   = float(_fee(a))
        except Exception:
            price, qty, fee = 0.0, 0.0, 0.0
        ts = getattr(a,"transaction_time", getattr(a,"timestamp", getattr(a,"date",None)))
        ts = pd.to_datetime(str(ts), utc=True, errors="coerce")
        if pd.isna(ts) or not sym or qty <= 0 or price <= 0: 
            continue
        rows.append({"ts": ts, "symbol": sym, "side": side, "qty": qty, "price": price, "fee": fee})
    if not rows: return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df

def _load_fills_dataframe(api, days: int) -> pd.DataFrame:
    """
    Paginated fills for last `days`, normalized for lot construction (keeps order_id when present).
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days+2)
    acts = _fetch_fills_paged(api, after_iso=start.isoformat(), until_iso=end.isoformat())

    rows = []
    for a in acts or []:
        rows.append({
            "symbol": (getattr(a,"symbol",None) or getattr(a,"asset_symbol",None) or "").upper(),
            "time":   pd.to_datetime(str(getattr(a,"transaction_time", getattr(a,"timestamp", getattr(a,"date", None)))), utc=True, errors="coerce"),
            "side":   str(getattr(a,"side","")).lower(),
            "qty":    float(getattr(a,"qty", getattr(a,"quantity",0.0)) or 0.0),
            "price":  float(getattr(a,"price", getattr(a,"fill_price",0.0)) or 0.0),
            "order_id": getattr(a, "order_id", None),
            "id":       getattr(a, "id", None),
            "fee":     float(getattr(a, "fee", 0.0) or 0.0),
        })
    df = pd.DataFrame(rows).dropna(subset=["time"])
    if df.empty: return df
    df = df.sort_values(["symbol","time"]).reset_index(drop=True)
    df["signed_qty"] = np.where(df["side"].str.contains("buy"), df["qty"], -df["qty"])
    return df

def _fills_for_lot_window(fills: pd.DataFrame,
                          symbol: str,
                          open_ts_utc: pd.Timestamp,
                          close_ts_utc: pd.Timestamp | None) -> pd.DataFrame:
    """
    Slice fills for a given symbol between [open_ts_utc, close_ts_utc] (inclusive).
    If close_ts_utc is None, take fills from open_ts onward (current open lot).
    Returns a tidy per-fill table with ET time, side, qty, price, notional.
    """
    if fills is None or fills.empty or not symbol or pd.isna(open_ts_utc):
        return pd.DataFrame(columns=["Time (ET)","Side","Qty","Price","Notional"])

    d = fills[(fills["symbol"].str.upper() == str(symbol).upper()) &
              (fills["ts"] >= open_ts_utc)]
    if close_ts_utc is not None and pd.notna(close_ts_utc):
        d = d[d["ts"] <= close_ts_utc]

    if d.empty:
        return pd.DataFrame(columns=["Time (ET)","Side","Qty","Price","Notional"])

    # Convert to ET for display
    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("US/Eastern")
        d["Time (ET)"] = d["ts"].dt.tz_convert(et).dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        d["Time (ET)"] = d["ts"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")

    d = d.copy()
    d["Side"]     = d["side"].str.upper()     # BUY / SELL / SELL_SHORT
    d["Qty"]      = pd.to_numeric(d["qty"], errors="coerce")
    d["Price"]    = pd.to_numeric(d["price"], errors="coerce")
    d["Notional"] = d["Qty"] * d["Price"]
    show = d[["Time (ET)","Side","Qty","Price","Notional"]].sort_values("Time (ET)")
    return show


def _fifo_realized_per_fill(fills: pd.DataFrame) -> pd.DataFrame:
    """
    Compute realized P&L per fill using FIFO lots, with correct signs for shorts.
    Returns a DataFrame shaped like the History template.
    """
    if fills is None or fills.empty:
        return pd.DataFrame(columns=["Date","Symbol","Side","Qty","Price","Notional","Realized","Fees","Amount","P&L $","Notes"])

    lots = defaultdict(deque)   # symbol -> deque of open lots: each lot {'qty'(signed), 'price'}
    out_rows = []

    # Ensure proper dtypes/sort
    z = fills.copy().sort_values("ts")
    z["qty"]   = pd.to_numeric(z["qty"], errors="coerce").fillna(0.0)
    z["price"] = pd.to_numeric(z["price"], errors="coerce").fillna(0.0)
    z["fee"]   = pd.to_numeric(z.get("fee", 0.0), errors="coerce").fillna(0.0)

    for _, r in z.iterrows():
        ts     = pd.to_datetime(r["ts"], utc=True, errors="coerce")
        sym    = str(r["symbol"]).upper()
        side   = "Buy" if str(r["side"]).lower() == "buy" else "Sell"
        qty    = float(r["qty"])
        price  = float(r["price"])
        fees   = float(r["fee"])
        sign   = +1.0 if side == "Buy" else -1.0  # buy opens long / closes short; sell opens short / closes long

        # Current net position before this trade
        net_qty_before = sum(l["qty"] for l in lots[sym])

        remaining = qty
        realized  = 0.0

        # Closing if the trade sign opposes current position sign
        if net_qty_before * sign < 0:
            while remaining > 1e-9 and lots[sym]:
                lot = lots[sym][0]
                lot_sign = 1.0 if lot["qty"] > 0 else -1.0
                lot_abs  = abs(lot["qty"])
                take = min(lot_abs, remaining)

                # Realized against this lot:
                # long lot (lot_sign=+1) closed by SELL:  +1*(close - entry)*qty
                # short lot (lot_sign=-1) closed by BUY:  -1*(close - entry)*qty  == (entry - close)*qty
                realized += lot_sign * (price - lot["price"]) * take

                lot["qty"] = lot["qty"] - lot_sign * take
                remaining -= take
                if abs(lot["qty"]) <= 1e-9:
                    lots[sym].popleft()

        # If we crossed through zero, any leftover becomes a NEW lot in the trade direction
        if remaining > 1e-9:
            lots[sym].append({"qty": sign * remaining, "price": price})

        notional = qty * price
        amount   = (price * qty * (1.0 if side == "Sell" else -1.0)) - fees   # cash delta (+ for sells, − for buys)
        pnl_d    = realized - fees

        # Display time in US/Eastern (or UTC if zone unavailable)
        dt = ts
        if _ET is not None:
            try: dt = ts.tz_convert(_ET)
            except Exception: pass

        out_rows.append({
            "Date":     dt.tz_localize(None),
            "Symbol":   sym,
            "Side":     side,
            "Qty":      qty,
            "Price":    price,
            "Notional": notional,
            "Realized": realized,
            "Fees":     fees,
            "Amount":   amount,
            "P&L $":    pnl_d,
            "Notes":    "",
        })

    out = pd.DataFrame(out_rows)
    # newest first for the UI
    return out.sort_values("Date", ascending=False).reset_index(drop=True)


def render_transaction_history_from_alpaca(api: Optional[REST], *, since: datetime | None = None) -> None:
    st.subheader("Transaction History")
    fills = _pull_all_fills_df(api, start=since)
    if fills.empty:
        st.info("No fill activity returned by Alpaca.")
        return

    hist = _fifo_realized_per_fill(fills)

    # Daily roll‑up like the template's summary
    day_group = (hist.assign(Day=hist["Date"].dt.date)
                      .groupby("Day", dropna=True)
                      .agg(**{
                          "Trades": ("Symbol","count"),
                          "Realized $": ("Realized","sum"),
                          "Fees $": ("Fees","sum"),
                          "P&L $": ("P&L $","sum")
                      })
                      .reset_index()
                      .sort_values("Day", ascending=False))

    c1, c2 = st.columns([0.60, 0.40])
    with c1:
        st.markdown("**Latest 200 fills**")
        st.dataframe(
            hist.head(200).style.format({
                "Qty":"{:,.0f}",
                "Price":"{:,.2f}",
                "Notional":"${:,.2f}",
                "Realized":"${:,.2f}",
                "Fees":"${:,.2f}",
                "Amount":"${:,.2f}",
                "P&L $":"${:,.2f}"
            }, na_rep="—"),
            width="stretch", hide_index=True
        )
    with c2:
        st.markdown("**Daily summary**")
        st.dataframe(
            day_group.style.format({"Realized $":"${:,.2f}","Fees $":"${:,.2f}","P&L $":"${:,.2f}"}),
            width="stretch", hide_index=True
        )

def render_ticker_tape(prices: list[dict]) -> None:
    """
    Horizontal scrolling ticker feed for your open symbols.
    """
    items = []
    for p in prices:
        sym = p.get("symbol", "")
        price = p.get("price", 0.0)
        change = p.get("change", 0.0)
        color = "#16A34A" if change >= 0 else "#DC2626"
        items.append(
            f"<span style='margin-right:24px; font-weight:600; font-size:16px;'>"
            f"{sym} <span style='color:{color}'>{price:,.2f} ({change:+.2f}%)</span>"
            f"</span>"
        )

    html = f"""
    <marquee behavior="scroll" direction="left" scrollamount="6"
             style="white-space:nowrap; font-family:system-ui; background:#0B1220; color:white; padding:6px 0; border-radius:6px;">
        {"".join(items)}
    </marquee>
    """
    st.markdown(html, unsafe_allow_html=True)

def _first_existing_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Return the first column name that exists in df (case sensitive)."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


@st.cache_data(ttl=20, show_spinner=False)
def get_open_ticker_prices(_api) -> list[dict]:
    """
    Live prices for currently open tickers from Alpaca, cached for 20s.
    Each item: {"symbol": str, "price": float, "change": float}
    """
    df = pull_live_positions(_api)  # uses the same Alpaca client
    prices = []
    for _, r in df.iterrows():
        prices.append({
            "symbol": str(r["Ticker"]),
            "price": float(r["current_price"]),
            "change": float(r.get("pl_%", 0.0)),
        })
    return prices

# --- Robust QUANTML logo loader (Streamlit Cloud safe) ---
from pathlib import Path
import base64, os
import streamlit as st

@st.cache_resource
def load_logo_b64(candidates: list[str] | None = None) -> str:
    """
    Order:
      1) st.secrets["LOGO_B64"] if present (recommended on Streamlit Cloud)
      2) first existing file from candidates (case differences handled by trying multiple names)
    Returns base64 string WITHOUT "data:image/...;base64," prefix.
    """
    # 1) Secret (paste your base64 once in Streamlit secrets)
    b64 = (st.secrets.get("LOGO_B64", "") or "").strip()
    if b64:
        return b64

    # 2) Files on disk (try several common locations/cases)
    if candidates is None:
        candidates = [
            "Clock/QuantML.png",
            "Clock/QuantML.png",
            "Clock/QuantML.png",
            "assets/QuantML.png",
            "QuantML.png",
            str(Path(__file__).with_name("QuantML.png")),
        ]
    for p in candidates:
        fp = Path(p)
        if fp.exists():
            try:
                return base64.b64encode(fp.read_bytes()).decode("ascii")
            except Exception:
                pass
    return ""  # caller can decide what to do if empty


def render_header(api):
    st.markdown('<div class="header-wrap">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([0.20, 0.65, 0.15], vertical_alignment="center")
    with c1:
        render_quantml_clock(size=200, tz="Europe/Dublin", title="Dublin", show_seconds=True, is_24h=True)
    with c2:
        st.markdown("## QUANTML — Investor Summary (Live)")
        if api is not None:
            render_market_chip(api)
            prices = get_open_ticker_prices(api)
            if prices:
                st.markdown('<div class="ticker-wrap">', unsafe_allow_html=True)
                render_ticker_tape(prices)
                st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown("<div style='text-align:right'>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", key="refresh_live"):
            st.cache_data.clear(); st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ===================== Equity Chart (replicates Alpaca Home) =====================
def render_portfolio_equity_chart(api: Optional[REST]) -> dict:
    """Render Alpaca portfolio equity with period toggles and tight y-range."""
    st.subheader("Your portfolio (Live from Alpaca broker)")

    # Period selector
    period = st.radio(
        "Period",
        options=["1D", "1M", "3M", "all"],
        horizontal=True,
        label_visibility="collapsed",
        key="ph_period",
    )

    # Fetch equity curve
    try:
        df = get_portfolio_history_df(api, period=period)
    except Exception as e:
        st.error(f"Could not load portfolio history: {e}")
        return {"period": period, "error": str(e)}

    if df.empty:
        st.info("No portfolio history is available.")
        return {"period": period}

    # Tight y-range for better readability
    y = pd.to_numeric(df["equity"], errors="coerce").astype(float)
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    span = ymax - ymin
    pad = max(1.0, 0.02 * span) if np.isfinite(span) else 1.0
    yrange = [ymin - pad, ymax + pad] if np.isfinite(ymin) and np.isfinite(ymax) else None

    # --- Main area chart (soft fill) ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["ts"],
        y=y,
        mode="lines",
        name="Equity",
        fill="tozeroy",
        fillcolor="rgba(79,70,229,0.15)",  # soft purple fill
        line=dict(width=2, color=BRAND["primary"]),
        hovertemplate="%{x}<br>Equity %{y:$,.0f}<extra></extra>",
    ))

    # Base layout
    fig.update_layout(
        height=280,
        showlegend=False,
        margin=dict(l=8, r=8, t=6, b=6),
        xaxis_title=None,
        yaxis_title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Apply y-range
    if yrange:
        fig.update_yaxes(range=yrange)

    # --- Fix Y-axis labels ("k k k" issue) ---
    try:
        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            tick_vals = np.linspace(ymin, ymax, 5).tolist()
            tick_texts = [f"${v:,.0f}" for v in tick_vals]
            fig.update_yaxes(tickmode="array", tickvals=tick_vals, ticktext=tick_texts)
        else:
            fig.update_yaxes(tickformat="$,.0f")
    except Exception:
        fig.update_yaxes(tickformat="$,.0f")

    # --- Render chart ---
    st.plotly_chart(fig, config={**PLOTLY_CONFIG, "responsive": True}, use_container_width=True)

    # === Quick summary stats ===
    chg_pct = float(df["ret_pct"].iloc[-1]) if len(df) else float("nan")
    idd_pct = _max_drawdown_pct(y)
    st.caption(
        f"Change {period}: {chg_pct:+.2f}% · Max drawdown over period: {idd_pct:.2f}%"
    )

    return {"period": period, "change_pct": chg_pct, "idd_pct": idd_pct}


def _parse_ts(ts):
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc)

def _fmt_et(ts) -> str:
    try:
        dt = _parse_ts(ts)
        if _NY is not None:
            dt = dt.astimezone(_NY)
        return dt.strftime("%H:%M")
    except Exception:
        return "—:—"

def _style_row_by_pl(row: pd.Series) -> pd.Series:
    """
    Apply traffic-light colors to Current Price, Total P/L $, and Total P/L % 
    following the same scheme used in render_traffic_lights():
      🟢 ≥ 0%
      🟡 -0.8% ≤ x < 0%
      🔴 < -0.8%
    """
    try:
        pct_val = float(row.get("Total P/L (%)", 0) or 0)
    except Exception:
        pct_val = 0.0

    color = _tl_color_for_pct(pct_val)  # reuse the same logic
    s = pd.Series("", index=row.index, dtype="object")
    for c in ("Current Price", "Total P/L ($)", "Total P/L (%)"):
        if c in row.index:
            s[c] = f"color:{color}; font-weight:700;"
    return s

def render_market_chip(api: REST) -> None:
    try:
        clk = api.get_clock()
    except Exception as e:
        st.caption(f"Clock not available: {e}")
        return

    is_open   = bool(getattr(clk, "is_open", False))
    next_open = getattr(clk, "next_open", None)
    next_close= getattr(clk, "next_close", None)

    color = "#16A34A" if is_open else "#DC2626"
    label = "NYSE OPEN" if is_open else "NYSE CLOSED"
    tail  = f"· closes { _fmt_et(next_close) } ET" if is_open else f"· opens { _fmt_et(next_open) } ET"

    st.markdown(
        f'<div class="chip"><span class="dot" style="background:{color}"></span>'
        f'<span class="label">{label} {tail}</span></div>',
        unsafe_allow_html=True
    )

# ---------- BROKER BALANCES & BUYING POWER ----------

def pull_account_snapshot(api: Optional[REST]) -> dict:
    """Grab the key Alpaca fields and compute a few safe derived metrics."""
    if api is None:
        return {}
    try:
        a = api.get_account()
    except Exception as e:
        st.warning(f"Could not fetch account: {e}")
        return {}

    def _f(x):
        try: return float(x)
        except: return None

    data = {
        "regt_buying_power":           _f(getattr(a, "regt_buying_power", None)),
        "daytrading_buying_power":     _f(getattr(a, "daytrading_buying_power", None)),
        "buying_power":                _f(getattr(a, "buying_power", None)),
        "non_marginable_buying_power": _f(getattr(a, "non_marginable_buying_power", None)),
        "initial_margin":              _f(getattr(a, "initial_margin", None)),
        "maintenance_margin":          _f(getattr(a, "maintenance_margin", None)),
        "cash":                        _f(getattr(a, "cash", None)),
        "cash_withdrawable":           _f(getattr(a, "cash_withdrawable", None) or getattr(a, "withdrawable_amount", None)),
        "equity":                      _f(getattr(a, "equity", None)),
        "last_equity":                 _f(getattr(a, "last_equity", None)),
        "long_market_value":           _f(getattr(a, "long_market_value", None)),
        "short_market_value":          _f(getattr(a, "short_market_value", None)),
        "accrued_fees":                _f(getattr(a, "accrued_fees", None)),
        "daytrade_count":              int(getattr(a, "daytrade_count", 0) or 0),
    }

    lm = data.get("long_market_value")  or 0.0
    sm = data.get("short_market_value") or 0.0
    data["position_market_value"] = abs(lm) + abs(sm)

    e, le = data.get("equity"), data.get("last_equity")
    data["equity_delta"] = (e - le) if (e is not None and le is not None) else None

    mm = data.get("maintenance_margin")
    data["margin_util_pct"] = (mm / e * 100.0) if (mm and e and e > 0) else None
    return data

# --- Intraday rolling stats for equity-PnL (robust to 1/5/30 min bars)
def _rolling_window_bars(ts: pd.Series, minutes: int = 10) -> int:
    if ts is None or ts.empty:
        return 2
    diffs = pd.to_datetime(ts, utc=True, errors="coerce").sort_values().diff().dropna()
    if diffs.empty:
        return 2
    # median bar size; default to 5 minutes if weird
    bar = diffs.median()
    try:
        bars = max(2, int(pd.Timedelta(minutes=minutes) / bar))
    except Exception:
        bars = 2
    return bars

def _intraday_rolling_stats(df_1d: pd.DataFrame, minutes: int = 10) -> dict:
    """
    df_1d: result of get_portfolio_history_df(..., period='1D'), with ['ts','equity']
    Returns:
      {
        'roll_mean': float, 'roll_vol': float,
        'ret_series': Series (%),
        'roll_mean_series': Series (%),
        'roll_std_series': Series (%),
        'cum_series': Series (%)   # cumulative intraday %
      }
    """
    if df_1d is None or df_1d.empty:
        return {"roll_mean": float("nan"), "roll_vol": float("nan"),
                "ret_series": pd.Series(dtype=float),
                "roll_mean_series": pd.Series(dtype=float),
                "roll_std_series": pd.Series(dtype=float),
                "cum_series": pd.Series(dtype=float)}

    z = df_1d.copy().sort_values("ts")
    z["ret_pct"] = z["equity"].pct_change() * 100.0
    z["cum_pct"] = z["ret_pct"].fillna(0).cumsum()

    win = _rolling_window_bars(z["ts"], minutes=minutes)
    rmean = z["ret_pct"].rolling(win).mean()
    rstd  = z["ret_pct"].rolling(win).std()

    return {
        "roll_mean": float(rmean.iloc[-1]) if len(rmean) else float("nan"),
        "roll_vol": float(rstd.iloc[-1]) if len(rstd) else float("nan"),
        "ret_series": z["ret_pct"],
        "roll_mean_series": rmean,
        "roll_std_series": rstd,
        "cum_series": z["cum_pct"],
    }

def _render_open_vs_spy_caption(open_pct: float | None, spy_pct: float | None) -> None:
    import numpy as np
    def _signed_pct(v):
        return f"{float(v):+.2f}%" if (v is not None and np.isfinite(float(v))) else "—"
    def _col(v):
        if v is None:
            return "#6B7280"
        try:
            v = float(v)
        except Exception:
            return "#6B7280"
        return BRAND["success"] if v >= 0 else BRAND["danger"]

    open_html = (
        f"<span style='color:{_col(open_pct)};font-weight:800;font-size:1.6rem;'>"
        f"QuantML {_signed_pct(open_pct)}</span>"
    )
    spy_html = (
        f"<span style='color:{_col(spy_pct)};font-weight:800;font-size:1.6rem;'>"
        f"SPY {_signed_pct(spy_pct)}</span>"
    )
    st.markdown(
        f"""
        <div style='text-align:center;margin-top:12px;margin-bottom:4px;'>
            {open_html} &nbsp;&nbsp;vs&nbsp;&nbsp; {spy_html}<br>
            <span style='font-size:1.1rem;color:#64748B;'>(intraday, regular session only)</span>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_perf_and_risk_kpis(api: Optional[REST], positions: pd.DataFrame) -> None:
    """
    Performance KPIs row:
      • Portfolio P&L today (% and $) from Alpaca equity curve
      • Open Positions P&L today (% and $) from position-level intraday P&L
      • Intraday smoothing panel (10-min μ/σ)
      • Compact caption comparing QuantML intraday vs SPY intraday
    """
    # ===== Intraday equity curve (Portfolio P&L today) =====
    intraday = get_portfolio_history_df(api, period="1D")

    # Rolling 10-minute average & volatility (uses equity curve % changes)
    roll_stats = _intraday_rolling_stats(intraday, minutes=10)

    day_pl_usd = day_pl_pct = np.nan
    start_equity = np.nan
    if not intraday.empty:
        start_equity = float(intraday["equity"].iloc[0])
        last_equity  = float(intraday["equity"].iloc[-1])
        day_pl_usd   = last_equity - start_equity
        day_pl_pct   = (day_pl_usd / start_equity * 100.0) if start_equity > 0 else np.nan

    # ===== Open positions P&L today (sum across positions) =====
    today_total_pl_usd = np.nan
    today_total_pl_pct = np.nan
    if positions is not None and not positions.empty:
        z = compute_derived_metrics(positions).copy()

        intraday_cols_usd = [
            "pl_today_usd",                # from compute_derived_metrics()
            "unrealized_intraday_pl",      # Alpaca field
            "intraday_pl_usd",             # any custom alias
        ]
        c_today_usd = next((c for c in intraday_cols_usd if c in z.columns), None)

        if c_today_usd is None and {"current_price","prev_close","qty"}.issubset(z.columns):
            today_vec = (pd.to_numeric(z["current_price"], errors="coerce")
                        - pd.to_numeric(z["prev_close"],  errors="coerce")) \
                        * pd.to_numeric(z["qty"], errors="coerce")
            today_total_pl_usd = float(np.nansum(today_vec.to_numpy()))
        elif c_today_usd is not None:
            today_total_pl_usd = float(pd.to_numeric(z[c_today_usd], errors="coerce").sum())

        if start_equity and np.isfinite(start_equity) and start_equity > 0 and np.isfinite(today_total_pl_usd):
            today_total_pl_pct = float(today_total_pl_usd / start_equity * 100.0)

    # ===== Optional: SPY intraday (for caption) =====
    spy_intraday_pct = np.nan
    try:
        et = ZoneInfo("US/Eastern")
    except Exception:
        et = timezone.utc

    try:
        spy_5 = _symbol_returns_5min_robust(api, "SPY", days=3)
        if not spy_5.empty:
            spy_5["ts"] = spy_5["ts"].dt.tz_convert(et)
            mopen, mclose = pd.to_datetime("09:30").time(), pd.to_datetime("16:00").time()
            spy_5 = spy_5[(spy_5["ts"].dt.time >= mopen) & (spy_5["ts"].dt.time <= mclose)]
            today_et = datetime.now(et).date()
            tday = spy_5[spy_5["ts"].dt.date == today_et]
            if not tday.empty:
                spy_intraday_pct = (np.prod(1.0 + tday["ret"].astype(float).to_numpy()/100.0) - 1.0) * 100.0
    except Exception:
        pass  # keep NaN on any data issue

    # ====================== KPI CARDS ======================
    c1, c2, c3, c4 = st.columns(4)

    # Portfolio-level (from equity curve)
    with c1:
        tone = "pos" if (day_pl_pct or 0) >= 0 else "neg"
        arrow = "▲" if tone == "pos" else "▼"
        _kpi_card("📈 Portfolio P&L (Today, %)",
                  f"{arrow} {(day_pl_pct if np.isfinite(day_pl_pct) else 0):+.2f}%",
                  tone)

    with c2:
        tone = "pos" if (day_pl_usd or 0) >= 0 else "neg"
        arrow = "▲" if tone == "pos" else "▼"
        _kpi_card("💰 Portfolio P&L (Today, $)",
                  f"{arrow} {money(day_pl_usd)}",
                  tone)

    # Open positions (sum of intraday)
    with c3:
        tone = "pos" if (today_total_pl_pct or 0) >= 0 else "neg"
        arrow = "▲" if tone == "pos" else "▼"
        _kpi_card("🟢 Open Positions P&L (Today, %)",
                  f"{arrow} {(today_total_pl_pct if np.isfinite(today_total_pl_pct) else 0):+.2f}%",
                  tone)

    with c4:
        tone = "pos" if (today_total_pl_usd or 0) >= 0 else "neg"
        arrow = "▲" if tone == "pos" else "▼"
        _kpi_card("💹 Open Positions P&L (Today, $)",
                  f"{arrow} {money(today_total_pl_usd)}",
                  tone)

    # === Open Positions vs SPY caption (color-coded) ===
    _render_open_vs_spy_caption(today_total_pl_pct, spy_intraday_pct)

    # === Intraday smoothing (last 10 minutes) ===
    st.markdown("<hr style='margin:10px 0;border:0.5px solid rgba(0,0,0,0.05);'>", unsafe_allow_html=True)
    st.markdown("**Intraday smoothing (last 10 minutes)**")

    # Optional: regular session only
    reg_only = st.toggle("Regular session only (09:30–16:00 ET)", value=True, key="roll_reg_only")

    r_all   = roll_stats["ret_series"]
    rs_all  = roll_stats["roll_mean_series"]
    rv_all  = roll_stats["roll_std_series"]
    cum_all = roll_stats["cum_series"]

    # Session filter (ET)
    r, rs, rv, cum = r_all, rs_all, rv_all, cum_all
    if reg_only and not r_all.empty:
        try:
            et = ZoneInfo("US/Eastern")
        except Exception:
            et = timezone.utc
        idx  = pd.to_datetime(intraday["ts"], utc=True, errors="coerce")
        hhmm = idx.dt.tz_convert(et).dt.time
        mask = (hhmm >= pd.to_datetime("09:30").time()) & (hhmm <= pd.to_datetime("16:00").time())
        r   = r_all.loc[mask.values]
        rs  = rs_all.loc[mask.values]
        rv  = rv_all.loc[mask.values]
        cum = cum_all.loc[mask.values]

    # Focus on a reasonable recent window for readability
    r  = r.tail(60)   # ~ last 60 bars (auto 1m/5m)
    rs = rs.reindex_like(r)
    rv = rv.reindex_like(r)

    cA, cB = st.columns([0.66, 0.34])

    with cA:
        fig_roll = go.Figure()

        # ±σ band around rolling mean
        if len(rs) and len(rv):
            upper = (rs + rv)
            lower = (rs - rv)
            fig_roll.add_trace(go.Scatter(
                x=rs.index, y=upper, mode="lines", line=dict(width=0),
                hoverinfo="skip", showlegend=False, name="+σ"
            ))
            fig_roll.add_trace(go.Scatter(
                x=rs.index, y=lower, mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(16,185,129,0.10)",  # soft green band
                hoverinfo="skip", showlegend=False, name="-σ"
            ))

        # raw intraday Δequity (%)
        if len(r):
            fig_roll.add_trace(go.Scatter(x=r.index, y=r.values, mode="lines",
                                          name="Δ equity (%)", line=dict(width=1.6), marker=dict(size=5)))
        # 10-min rolling mean
        if len(rs):
            fig_roll.add_trace(go.Scatter(x=rs.index, y=rs.values, mode="lines",
                                          name="10-min avg", line=dict(width=2.2)))

        fig_roll.add_hline(y=0, line_dash="dot", line_color="rgba(0,0,0,.35)")

        # right-edge μ / ±σ labels
        if len(rs):
            last_mu = float(rs.iloc[-1])
            fig_roll.add_annotation(x=rs.index[-1], y=last_mu, text=f"μ {last_mu:+.2f}%", showarrow=False,
                                    xanchor="left", yanchor="middle", xshift=6, font=dict(size=11))
        if len(rv):
            last_sig = float(rv.iloc[-1])
            fig_roll.add_annotation(x=rv.index[-1], y=float((rs+rv).iloc[-1]), text=f"+σ {last_sig:.2f}%", showarrow=False,
                                    xanchor="left", yanchor="bottom", xshift=6, font=dict(size=10, color="#6b7280"))
            fig_roll.add_annotation(x=rv.index[-1], y=float((rs-rv).iloc[-1]), text=f"−σ {last_sig:.2f}%", showarrow=False,
                                    xanchor="left", yanchor="top", xshift=6, font=dict(size=10, color="#6b7280"))

        # flag > 2σ moves
        if len(rv) and len(r):
            thresh = (rv * 2.0).reindex_like(r).ffill()
            spikes = r[abs(r) > thresh]
            if not spikes.empty:
                fig_roll.add_trace(go.Scatter(
                    x=spikes.index, y=spikes.values, mode="markers",
                    marker=dict(size=8, symbol="circle"), name=">2σ",
                    hovertemplate="%{x}<br>%{y:.2f}% (>2σ)<extra></extra>"
                ))

        fig_roll.update_layout(
            height=180, margin=dict(l=8, r=8, t=6, b=6), showlegend=False,
            xaxis=dict(visible=False), yaxis_title="Δ equity (%)"
        )
        st.plotly_chart(fig_roll, config={**PLOTLY_CONFIG, "responsive": True}, use_container_width=True)

        # summary chips + one-line explanation UNDER the sparkline
        mu  = roll_stats["roll_mean"]
        sig = roll_stats["roll_vol"]
        cum_today = float(cum.iloc[-1]) if len(cum) else float("nan")
        st.markdown(
            f"<span class='qml-chip {'good' if (mu or 0) >= 0 else 'bad'}'><span class='sw' style='background:#10B981'></span>μ10 {mu:+.2f}%</span> &nbsp;"
            f"<span class='qml-chip'><span class='sw' style='background:#64748B'></span>σ10 {sig:.2f}%</span> &nbsp;"
            f"<span class='qml-chip'><span class='sw' style='background:#4F46E5'></span>Today cum {cum_today:+.2f}%</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            "μ₁₀ is the 10-minute rolling average of intraday P&L; "
            "σ₁₀ is its volatility (standard deviation); "
            "and *Today cum* is your cumulative intraday return since market open."
        )

    with cB:
        # Volatility gauge (10-min σ)
        vol = roll_stats["roll_vol"]
        st.plotly_chart(
            _banded_gauge(vol if np.isfinite(vol) else 0.0,
                        title="P&L vol (10-min σ)",
                        bands=(0.20, 0.50, 1.00),
                        good="low"),
            config={**PLOTLY_CONFIG, "responsive": True}
        )

def render_broker_balances(acct: dict) -> None:
    st.subheader("Broker Balance & Buying Power (Alpaca)")
    if not acct:
        st.info("—"); return

    c1, c2, c3, c4 = st.columns(4)
    with c1: _kpi_card("RegT Buying Power",        money(acct.get("regt_buying_power") or 0.0))
    with c2: _kpi_card("Day Trading Buying Power", money(acct.get("daytrading_buying_power") or 0.0))
    with c3: _kpi_card("Effective Buying Power",   money(acct.get("buying_power") or 0.0))
    with c4: _kpi_card("Non‑Marginable BP",        money(acct.get("non_marginable_buying_power") or 0.0))

    c5, c6, c7, c8 = st.columns(4)
    with c5: _kpi_card("Initial Margin",     money(acct.get("initial_margin") or 0.0))
    with c6: _kpi_card("Maintenance Margin", money(acct.get("maintenance_margin") or 0.0))
    with c7:
        eq, leq = acct.get("equity"), acct.get("last_equity")
        delta = (eq - leq) if (eq is not None and leq is not None) else 0.0
        rel   = (delta / leq * 100.0) if (leq and leq != 0) else 0.0
        tone  = "pos" if (delta or 0) >= 0 else "neg"
        _kpi_card("Equity", money(eq or 0.0), tone, caption=f"vs last close: {money(leq)} ({rel:+.2f}%)")
    with c8: _kpi_card("Position Market Value", money(acct.get("position_market_value") or 0.0))

    c9, c10, c11, c12 = st.columns(4)
    with c9:  _kpi_card("Cash",              money(acct.get("cash") or 0.0))
    with c10: _kpi_card("Cash Withdrawable", money(acct.get("cash_withdrawable") or 0.0))
    with c11: _kpi_card("Accrued Fees",      money(acct.get("accrued_fees") or 0.0))
    with c12: _kpi_card("Day Trade Count (5‑day)", str(int(acct.get("daytrade_count") or 0)))

    if acct.get("margin_util_pct") is not None:
        st.plotly_chart(_banded_gauge(float(acct["margin_util_pct"]), "Margin Utilization",
                                      bands=(25, 50, 100), good="low"),
                        width='stretch', config={**PLOTLY_CONFIG, "responsive": True}, use_container_width=True)
        st.caption("= Maintenance margin ÷ equity. Lower is safer.")

# =============================================================================
# Formatting helpers
# =============================================================================
def money(x) -> str:
    try: return f"${float(x):,.2f}"
    except: return "—"

def money_signed(x) -> str:
    try:
        v = float(x)
        return f"+{money(v)}" if v > 0 else money(v)
    except:
        return "—"

def pct(x) -> str:
    try:
        v = float(x)
    except Exception:
        return "—"
    return "—" if not np.isfinite(v) else f"{v:+.2f}%".replace("+-", "-")


# =============================================================================
# Live math / derivations
# =============================================================================
def compute_derived_metrics(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Normalizes live fields and computes:
      - dir_sign (+1 long, -1 short), P&L $, P&L %
      - risk_$ / reward_$ (from SL/TP if present)
      - R = reward_$ / |risk_$|
    """
    if df is None or df.empty:
        return df

    res = df.copy()

    side_raw = res.get("Trade Action", res.get("Side", "")).astype(str).str.upper()
    res["Side"] = np.where(side_raw.eq("LONG"), "Long",
                    np.where(side_raw.eq("SHORT"), "Short", side_raw.str.title()))
    res["dir_sign"] = np.where(side_raw.eq("LONG"), 1.0,
                         np.where(side_raw.eq("SHORT"), -1.0, np.nan))

    res["qty"]           = pd.to_numeric(res.get("qty", res.get("quantity")), errors="coerce")
    res["entry_price"]   = pd.to_numeric(res.get("entry_price", res.get("avg_entry_price")), errors="coerce")
    res["current_price"] = pd.to_numeric(res.get("current_price", res.get("mark_price")), errors="coerce")

    # P&L
    res["pl_$"] = res["dir_sign"] * (res["current_price"] - res["entry_price"]) * res["qty"].abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        res["pl_%"] = 100.0 * res["dir_sign"] * (res["current_price"] - res["entry_price"]) / res["entry_price"]

    # Optional SL/TP columns (several possible names)
    sl = pd.to_numeric(res.get("sl_price", res.get("SL")), errors="coerce")
    tp = pd.to_numeric(res.get("tp_price", res.get("TP")), errors="coerce")

    risk_per_sh   = res["dir_sign"] * (res["entry_price"] - sl)
    reward_per_sh = res["dir_sign"] * (tp - res["entry_price"])

    res["risk_$"]   = (risk_per_sh   * res["qty"].abs()).where(risk_per_sh.notna())
    res["reward_$"] = (reward_per_sh * res["qty"].abs()).where(reward_per_sh.notna())

    res["R"] = np.where(res["risk_$"].abs().gt(1e-9), res["reward_$"] / res["risk_$"].abs(), np.nan)
    return res

def derive_totals_from_positions(df: pd.DataFrame | None) -> dict:
    """Totals used by KPIs / dials when a broker snapshot isn't present (robust)."""
    if df is None or df.empty:
        return {}

    z = compute_derived_metrics(df).copy()

    # ensure numeric
    qty        = pd.to_numeric(z.get("qty"), errors="coerce").fillna(0.0)
    entry_px   = pd.to_numeric(z.get("entry_price"), errors="coerce").fillna(0.0)
    current_px = pd.to_numeric(z.get("current_price"), errors="coerce").fillna(0.0)

    # notional (always positive)
    notional = pd.to_numeric(z.get("notional"), errors="coerce")
    if notional.isna().any():
        notional = (qty.abs() * entry_px)

    capital_spent = float(np.nansum(np.abs(notional.to_numpy())))
    # P&L $ — prefer our computed column, else fall back to broker's
    pl_cols = ["pl_$", "pl_usd", "unrealized_pl", "unrealized_pl_usd"]
    pl_col  = next((c for c in pl_cols if c in z.columns), None)
    if pl_col is None:
        # compute from entry/current as a last resort
        sign = np.where(z.get("Side","Long").astype(str).str.upper().eq("LONG"), 1.0, -1.0)
        pl_series = sign * (current_px - entry_px) * qty.abs()
    else:
        pl_series = pd.to_numeric(z[pl_col], errors="coerce")

    upnl = float(np.nansum(pl_series.to_numpy()))
    # sum of row percents as a fallback total %
    pl_pct_sum = float(pd.to_numeric(z.get("pl_%"), errors="coerce").sum(skipna=True))

    return {
        "positions": int(len(z)),
        "capital_spent": round(capital_spent, 2),
        "unrealized_pl_$": round(upnl, 2),
        "unrealized_pl_%_weighted": (round(upnl / capital_spent * 100.0, 3) if capital_spent > 0 else 0.0),
        "unrealized_pl_%_sum": round(pl_pct_sum, 2),
        "gross_exposure": float(np.nansum(np.abs((qty.abs() * entry_px).to_numpy()))),
        "net_exposure":   float(np.nansum(np.sign(qty.fillna(0)) * qty.abs() * entry_px)),
        "wins": int((pl_series > 0).sum()),
        "win_rate_%": float(100.0 * (pl_series > 0).sum() / max(1, len(z))),
    }

# =============================================================================
# Alpaca: Live pulls + TP/SL from open orders (cached)
# =============================================================================
_OPEN_STATES = {"new", "accepted", "held", "open", "partially_filled", "replaced"}

def _list_open_orders(_api):
    """Return open orders using whichever Alpaca SDK signature is available."""
    if _api is None:
        return []
    if hasattr(_api, "list_orders"):
        try:
            return _api.list_orders(status="open", nested=True, limit=500)
        except TypeError:
            return _api.list_orders(status="open", limit=500)
    if hasattr(_api, "get_orders"):  # older SDK
        try:
            return _api.get_orders(status="open", nested=True, limit=500)
        except TypeError:
            return _api.get_orders(status="open", limit=500)
    try:
        return _api.get_orders(status="open")  # alpaca‑py duck‑type
    except Exception:
        return []

# ===================== Adaptive ATR — helpers =====================
# ✅ safe to cache: returns plain dicts, not Alpaca Entity objects
@st.cache_data(ttl=120, show_spinner=False)
def _get_order_by_id_robust(_api: Optional[REST], order_id: str):
    """Fetch one order and return a plain-JSON-like dict (pickle-safe for Streamlit cache)."""
    if _api is None or not order_id:
        return None

    def _g(obj, *names):
        for n in names:
            try:
                v = getattr(obj, n)
            except Exception:
                v = None
            if v is not None:
                return v
        return None

    def _plain_order(obj):
        if obj is None:
            return None
        d = {
            "id": _g(obj, "id"),
            "client_order_id": _g(obj, "client_order_id"),
            "symbol": _g(obj, "symbol", "asset_symbol"),
            "side": _g(obj, "side"),
            "type": _g(obj, "type", "order_type"),
            "order_class": _g(obj, "order_class"),
            "status": _g(obj, "status"),
            "limit_price": _g(obj, "limit_price", "price"),
            "stop_price": _g(obj, "stop_price"),
            "submitted_at": _g(obj, "submitted_at", "created_at", "timestamp"),
        }
        tp = _g(obj, "take_profit")
        sl = _g(obj, "stop_loss")
        legs = _g(obj, "legs") or []

        if tp is not None:
            d["take_profit"] = {"limit_price": _g(tp, "limit_price", "price")}
        if sl is not None:
            d["stop_loss"] = {
                "stop_price": _g(sl, "stop_price"),
                "limit_price": _g(sl, "limit_price"),
                "trail_price": _g(sl, "trail_price"),
                "trail_percent": _g(sl, "trail_percent"),
            }
        # Recurse legs to plain dicts too
        d["legs"] = [_plain_order(l) for l in legs] if legs else []
        return d

    # --- fetch with fallbacks ---
    o = None
    try:
        o = _api.get_order(order_id)
    except TypeError:
        try:
            o = _api.get_order_by_id(order_id)
        except Exception:
            o = None
    except Exception:
        o = None

    if o is None:
        # Fallback: scan recent orders and match by id
        try:
            try:
                orders = _api.list_orders(status="all", nested=True, limit=500)
            except TypeError:
                orders = _api.list_orders(status="all", limit=500)
            for cand in orders or []:
                if str(_g(cand, "id")) == str(order_id):
                    o = cand
                    break
        except Exception:
            o = None

    return _plain_order(o)

def _attr_or_key(obj, name, default=None):
    """Get attribute or dict key."""
    try:
        v = getattr(obj, name)
        if v is not None:
            return v
    except Exception:
        pass
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default

def _extract_bracket_prices(order_obj) -> tuple[float | float, float | float]:
    """Return (initial_tp, initial_sl) from an order object; supports parent fields or legs; may return NaN."""
    import math, pandas as pd
    if order_obj is None:
        return (math.nan, math.nan)

    def _attr_or_key(obj, *names, default=None):
        for n in names:
            try:
                v = getattr(obj, n)
            except Exception:
                v = None
            if v is None and isinstance(obj, dict):
                v = obj.get(n)
            if v is not None:
                return v
        return default

    tp_px = math.nan
    sl_px = math.nan

    tp = _attr_or_key(order_obj, "take_profit")
    sl = _attr_or_key(order_obj, "stop_loss")

    if tp is not None:
        v = _attr_or_key(tp, "limit_price", "price")
        try: tp_px = float(v)
        except Exception: pass

    if sl is not None:
        v = _attr_or_key(sl, "stop_price", "trail_price")
        try: sl_px = float(v)
        except Exception: pass

    if (not pd.notna(tp_px)) or (not pd.notna(sl_px)):
        legs = _attr_or_key(order_obj, "legs", []) or []
        for leg in legs:
            ltype = str(_attr_or_key(leg, "type", "order_type", default="")).lower()
            if "limit" in ltype and not pd.notna(tp_px):
                try: tp_px = float(_attr_or_key(leg, "limit_price", "price"))
                except Exception: pass
            if ("stop" in ltype or "trailing" in ltype) and not pd.notna(sl_px):
                try: sl_px = float(_attr_or_key(leg, "stop_price", "trail_price", "price"))
                except Exception: pass

    return (tp_px, sl_px)

@st.cache_data(ttl=90, show_spinner=False)
def _current_open_lot_map(_api: Optional[REST]) -> dict[str, dict]:
    """
    For currently OPEN lots, return {SYM: {"open_ts": pd.Timestamp(UTC), "side": "Long|Short"}}
    Derived from build_position_transaction_history() which you already have.
    """
    out = {}
    try:
        df = build_position_transaction_history(_api, days=180)
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        return out
    open_rows = df[df["status"].astype(str).str.lower().eq("open")]
    for _, r in open_rows.iterrows():
        sym = str(r.get("symbol", "")).upper()
        if not sym:
            continue
        out[sym] = {"open_ts": pd.to_datetime(r.get("open_ts"), utc=True, errors="coerce"),
                    "side":    str(r.get("side", "")).title()}
    return out


@st.cache_data(ttl=90, show_spinner=False)
def _initial_tp_sl_lookup(_api: Optional[REST], positions: pd.DataFrame) -> dict[str, dict]:
    """
    Build {SYM: {"init_tp": float|nan, "init_sl": float|nan}} for the currently open lot,
    by locating the entry order (via first fill at/after lot open) and reading its bracket.
    Falls back to the *nearest* submitted order by timestamp if fills lack order_id.
    """
    import math, pandas as pd
    if _api is None or positions is None or positions.empty:
        return {}

    open_map = _current_open_lot_map(_api)  # {SYM: {"open_ts": ..., "side": ...}}
    fills = _load_fills_dataframe(_api, days=180)
    if fills is None:
        fills = pd.DataFrame(columns=["symbol", "time", "order_id"])

    out = {}
    for _, r in positions.iterrows():
        sym = str(r.get("Ticker", r.get("symbol", ""))).upper()
        if not sym or sym not in open_map:
            continue

        lot_ts = pd.to_datetime(open_map[sym]["open_ts"], utc=True, errors="coerce")
        sub = fills[(fills["symbol"].astype(str).str.upper() == sym)]
        if pd.notna(lot_ts):
            sub = sub[sub["time"] >= lot_ts]
        sub = sub.sort_values("time").head(1)

        order_obj = None
        if not sub.empty:
            oid = str(sub["order_id"].iloc[0]) if "order_id" in sub.columns else None
            if oid and oid != "None":
                order_obj = _get_order_by_id_robust(_api, oid)

        # Fallback: choose the order closest to lot open_ts
        if order_obj is None:
            try:
                def _attr_or_key(obj, name, alt=None):
                    try: v = getattr(obj, name)
                    except Exception: v = None
                    if v is None and isinstance(obj, dict):
                        v = obj.get(name if alt is None else alt)
                    return v

                orders = _api.list_orders(status="all", limit=500)  # nested=True optional
                cand = []
                for o in (orders or []):
                    osym = (getattr(o, "symbol", None) or getattr(o, "asset_symbol", None) or "").upper()
                    if osym != sym:
                        continue
                    ots = _attr_or_key(o, "submitted_at") or _attr_or_key(o, "created_at")
                    ts = pd.to_datetime(str(ots), utc=True, errors="coerce")
                    if pd.isna(ts):
                        continue
                    cand.append((abs((ts - lot_ts).total_seconds()) if pd.notna(lot_ts) else 0, o))
                if cand:
                    order_obj = sorted(cand, key=lambda x: x[0])[0][1]
            except Exception:
                order_obj = None

        init_tp, init_sl = (math.nan, math.nan)
        if order_obj is not None:
            init_tp, init_sl = _extract_bracket_prices(order_obj)

            # Final fallback: parse from client_order_id (QML|tp=..|sl=..|sym=..)
            if not pd.notna(init_tp) or not pd.notna(init_sl):
                try:
                    cid = getattr(order_obj, "client_order_id", None)
                    tp_cid, sl_cid = _parse_init_from_client_id(cid)
                    if pd.notna(tp_cid) and not pd.notna(init_tp):
                        init_tp = tp_cid
                    if pd.notna(sl_cid) and not pd.notna(init_sl):
                        init_sl = sl_cid
                except Exception:
                    pass

        out[sym] = {"init_tp": init_tp, "init_sl": init_sl}

    return out

def _last_exit_update_et(_api: Optional[REST], sym: str, side_long_short: str) -> str:
    """
    Return newest submitted_at (ET) among open exit legs (TP/SL) for a symbol's closing side.
    """
    try:
        exits = _open_exits_df(_api)
    except Exception:
        exits = pd.DataFrame()
    if exits is None or exits.empty:
        return "—"

    want_side = "sell" if str(side_long_short).lower() == "long" else "buy"
    d = exits[(exits["symbol"].astype(str).str.upper() == str(sym).upper()) &
              (exits["side"].astype(str).str.lower() == want_side)]
    if d.empty:
        return "—"

    ts = pd.to_datetime(d["submitted_at"], utc=True, errors="coerce").max()
    if pd.isna(ts):
        return "—"
    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("US/Eastern")
        return ts.tz_convert(et).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts.strftime("%Y-%m-%d %H:%M:%S %Z")

def _open_exits_df(_api) -> pd.DataFrame:
    """
    Flat table of *open* exit legs: symbol, side (closing side), leg_type (limit/stop/stop_limit/trailing_stop),
    limit_price, stop_price, trail_price, trail_percent, submitted_at, status.
    """
    import numpy as np, pandas as pd

    orders = _list_open_orders(_api)
    if not orders:
        return pd.DataFrame()

    def _ts(o):
        for attr in ("submitted_at", "created_at", "timestamp"):
            v = getattr(o, attr, None)
            if v:
                try:
                    return pd.to_datetime(str(v), utc=True, errors="coerce")
                except Exception:
                    pass
        return pd.NaT

    def _closing_side(parent_side: str) -> str:
        return "buy" if str(parent_side).lower() == "sell" else "sell"

    rows = []
    for o in orders or []:
        sym  = (getattr(o, "symbol", None) or getattr(o, "asset_symbol", None) or "").upper()
        if not sym:
            continue
        parent_side = (getattr(o, "side", None) or "").lower()  # entry side
        legs = list(getattr(o, "legs", None) or [])

        # ------------- collect legs that are still open -------------
        for leg in ([o] + legs):
            ltype = (getattr(leg, "type", None) or getattr(leg, "order_type", None) or "").lower()
            lstat = (getattr(leg, "status", None) or "").lower()
            if lstat and lstat not in _OPEN_STATES:
                continue

            raw_leg_side = (getattr(leg, "side", None) or "").lower()
            side = raw_leg_side if raw_leg_side else (
                _closing_side(parent_side) if ("limit" in ltype or "take_profit" in ltype or ltype.startswith("stop")) else parent_side
            )

            def _price(obj, kind):
                if kind == "limit":
                    return float(getattr(obj, "limit_price", getattr(obj, "price", np.nan)) or np.nan)
                return float(getattr(obj, "stop_price",  getattr(obj, "price", np.nan)) or np.nan)

            is_stop     = ltype.startswith("stop")
            is_trailing = ("trail" in ltype) or ("trailing" in ltype)
            is_limit    = ("limit" in ltype) or ("take_profit" in ltype)

            rows.append({
                "symbol": sym,
                "side": side,
                "leg_type": ("trailing_stop" if is_trailing else ("stop" if is_stop else ("limit" if is_limit else ltype))),
                "limit_price": _price(leg, "limit"),
                "stop_price":  _price(leg, "stop"),
                "trail_price":  float(getattr(leg, "trail_price",  np.nan) or np.nan),
                "trail_percent":float(getattr(leg, "trail_percent",np.nan) or np.nan),
                "submitted_at": _ts(leg),
                "status": lstat or (getattr(o, "status", None) or "").lower(),
            })

        # ------------- fallback: some brackets keep TP/SL only on parent -------------
        if not legs:
            def _get_attr_or_key(obj, name):
                try:
                    v = getattr(obj, name)
                except Exception:
                    v = None
                if v is None and isinstance(obj, dict):
                    v = obj.get(name)
                return v

            tp_obj = getattr(o, "take_profit", None)
            sl_obj = getattr(o, "stop_loss", None)
            close_side = _closing_side(parent_side)
            parent_status = (getattr(o, "status", None) or "").lower()
            parent_ts = _ts(o)

            if tp_obj is not None:
                tp_px = _get_attr_or_key(tp_obj, "limit_price") or _get_attr_or_key(tp_obj, "price")
                rows.append({
                    "symbol": sym, "side": close_side, "leg_type": "limit",
                    "limit_price": float(tp_px) if tp_px is not None else np.nan,
                    "stop_price": np.nan, "trail_price": np.nan, "trail_percent": np.nan,
                    "submitted_at": parent_ts, "status": parent_status,
                })

            if sl_obj is not None:
                stop_px = _get_attr_or_key(sl_obj, "stop_price")
                stop_limit_px = _get_attr_or_key(sl_obj, "limit_price")
                trail_px = _get_attr_or_key(sl_obj, "trail_price")
                trail_pct = _get_attr_or_key(sl_obj, "trail_percent")
                leg_t = "trailing_stop" if (trail_px is not None and trail_px != "") else ("stop_limit" if stop_limit_px is not None else "stop")
                rows.append({
                    "symbol": sym, "side": close_side, "leg_type": leg_t,
                    "limit_price": float(stop_limit_px) if stop_limit_px is not None else np.nan,
                    "stop_price":  float(stop_px) if stop_px is not None else np.nan,
                    "trail_price":  float(trail_px) if trail_px is not None else np.nan,
                    "trail_percent":float(trail_pct) if trail_pct is not None else np.nan,
                    "submitted_at": parent_ts, "status": parent_status,
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["submitted_at"] = pd.to_datetime(df["submitted_at"], utc=True, errors="coerce")
    return df

def _pick_tp_sl_for(sym: str, side_long_short: str, exits_df: pd.DataFrame) -> tuple[float, float]:
    """
    LONG  -> want closing side = 'sell' (TP is sell LIMIT, SL is sell STOP/STOP_LIMIT/TRAILING_STOP)
    SHORT -> want closing side = 'buy'
    """
    import numpy as np, pandas as pd
    if exits_df is None or exits_df.empty or not sym:
        return (np.nan, np.nan)

    want_side = "sell" if str(side_long_short).strip().lower() == "long" else "buy"
    d = exits_df[
        (exits_df["symbol"].astype(str).str.upper() == str(sym).upper()) &
        (exits_df["side"].astype(str).str.lower() == want_side)
    ]
    if d.empty:
        return (np.nan, np.nan)

    # TP: newest LIMIT
    tp = np.nan
    lim = d[d["leg_type"].eq("limit")].sort_values("submitted_at").tail(1)
    if not lim.empty:
        v = lim["limit_price"].iloc[0]
        tp = float(v) if pd.notna(v) else np.nan

    # SL: newest STOP / STOP_LIMIT / TRAILING_STOP (prefer stop_price, else trail_price)
    sl = np.nan
    stp = d[d["leg_type"].isin(["stop", "stop_limit", "trailing_stop"])].sort_values("submitted_at").tail(1)
    if not stp.empty:
        v_stop  = stp["stop_price"].iloc[0]  if "stop_price"  in stp.columns else np.nan
        v_trail = stp["trail_price"].iloc[0] if "trail_price" in stp.columns else np.nan
        sl = float(v_stop) if pd.notna(v_stop) else (float(v_trail) if pd.notna(v_trail) else np.nan)

    return (tp, sl)

def merge_tp_sl_from_alpaca_orders(positions: pd.DataFrame, api) -> pd.DataFrame:
    """Attach TP/SL columns from broker open orders (no crash if none exist)."""
    import numpy as np, pandas as pd

    out = positions.copy() if positions is not None else pd.DataFrame()
    if out is None or out.empty or api is None:
        for c in ("TP", "SL", "tp_price", "sl_price"):
            if c not in out.columns:
                out[c] = np.nan
        return out

    exits = _open_exits_df(api)

    # Normalize symbol/side cols
    sym_col = "Ticker" if "Ticker" in out.columns else ("symbol" if "symbol" in out.columns else None)
    if not sym_col:
        out["Ticker"] = out.get("Ticker", out.get("symbol", ""))
        sym_col = "Ticker"
    out[sym_col] = out[sym_col].astype(str).str.upper()

    side_col = "Side" if "Side" in out.columns else ("Trade Action" if "Trade Action" in out.columns else None)
    if side_col:
        s = out[side_col].astype(str).str.upper()
        out["Side"] = np.where(s.eq("LONG"), "Long", np.where(s.eq("SHORT"), "Short", out.get("Side", s.str.title())))
    else:
        out["Side"] = out.get("Side", "Long")

    tp_vals, sl_vals = [], []
    for _, r in out.iterrows():
        tp, sl = _pick_tp_sl_for(str(r[sym_col]), str(r["Side"]), exits)
        tp_vals.append(tp); sl_vals.append(sl)

    out["TP"]       = out.get("TP").combine_first(pd.Series(tp_vals, index=out.index)) if "TP" in out else tp_vals
    out["SL"]       = out.get("SL").combine_first(pd.Series(sl_vals, index=out.index)) if "SL" in out else sl_vals
    out["tp_price"] = out.get("tp_price").combine_first(out["TP"]) if "tp_price" in out else out["TP"]
    out["sl_price"] = out.get("sl_price").combine_first(out["SL"]) if "sl_price" in out else out["SL"]
    return out

# =============================================================================
# Dials (3 core)
# =============================================================================
_DIAL_H = 280           # more vertical room
_NEEDLE = "#374151"
_BAND_GREEN = "rgba(22,163,74,0.22)"
_BAND_AMBER = "rgba(234,179,8,0.26)"
_BAND_RED   = "rgba(220,38,38,0.24)"

def _banded_gauge(percent: float, title: str, bands=(60,80,100), good="low") -> go.Figure:
    val = float(max(0.0, percent))

    # --- Custom scaling for open-position P&L dials ---
    if "open positions" in title.lower() or "total p/l" in title.lower():
        axis_max = 3.0   # cap at 3 %
    else:
        axis_max = min(200.0, max(100.0, (int(val/20.0)+1) * 20.0))

    a, b, c = bands
    steps = ([{"range":[0,a], "color":_BAND_RED},
              {"range":[a,b], "color":_BAND_AMBER},
              {"range":[b,axis_max], "color":_BAND_GREEN}]
             if good=="high" else
             [{"range":[0,a], "color":_BAND_GREEN},
              {"range":[a,b], "color":_BAND_AMBER},
              {"range":[b,axis_max], "color":_BAND_RED}])

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=min(val, axis_max),
        title={"text": title, "font":{"size":13}},
        number={"suffix":"%", "font":{"size":24}},
        gauge={
            "axis":{"range":[0,axis_max], "tickwidth":1, "ticklen":3, "nticks":6, "tickfont":{"size":9}},
            "bar":{"color":_NEEDLE, "thickness":0.22},
            "bgcolor":"white", "borderwidth":0, "steps":steps
        }
    ))
    fig.update_layout(height=_DIAL_H, margin=dict(l=10,r=10,t=72,b=56))
    return fig


def gauge_budget_pct(cap_spent: float, budget: float, title="Capital Deployed") -> go.Figure:
    pct_val = 0.0 if not budget else (100.0 * float(cap_spent) / float(budget))
    return _banded_gauge(pct_val, title, bands=(60, 80, 100), good="low")

def gauge_exposure_pct(gross_exp: float, budget: float, title="Gross Exposure") -> go.Figure:
    pct_val = 0.0 if not budget else (100.0 * float(gross_exp) / float(budget))
    return _banded_gauge(pct_val, title, bands=(60, 80, 100), good="low")

def gauge_win_rate(win_pct: float, title="Win Rate") -> go.Figure:
    return _banded_gauge(float(win_pct), title, bands=(40, 60, 80), good="high")


# =============================================================================
# UI building blocks
# =============================================================================
def _kpi_card(title: str, value_str: str, tone: str = "neutral", caption: str | None = None) -> None:
    tone_cls = {"pos":"positive", "neg":"negative"}.get(tone, "")
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value {tone_cls}">{value_str}</div>
          {f'<div class="kpi-caption">{caption}</div>' if caption else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

# ===================== Traffic Lights (sorted green→amber→red) =====================

def render_traffic_lights(df: pd.DataFrame) -> None:
    st.subheader("Traffic Lights (per open position)")
    if df is None or df.empty:
        st.info("No open positions.")
        return

    z = compute_derived_metrics(df).copy()
    # choose today % if available; else total %
    z["pl_light_%"] = pd.to_numeric(z.get("pl_%"), errors="coerce")
    z = _sort_df_green_amber_red(z, "pl_light_%")

    CHIP = ("display:inline-flex;align-items:center;gap:8px;padding:6px 10px;"
            "border-radius:16px;background:rgba(14,27,58,0.06);border:1px solid rgba(14,27,58,0.20);"
            "margin:3px 4px;")
    chips = []
    for _, r in z.iterrows():
        sym = r.get("Ticker", r.get("symbol", "?"))
        pl_today_pct = float(r.get("pl_light_%", 0.0))
        col = _tl_color_for_pct(pl_today_pct)
        label = f"{sym} · {r.get('Side','—')} · {pl_today_pct:+.2f}% / {money_signed(r.get('pl_$', 0.0))}"
        dot = f"<span style='width:12px;height:12px;border-radius:50%;display:inline-block;border:2px solid {col};background:{col};'></span>"
        chips.append(f"<span style='{CHIP}'>{dot}"
                     f"<span style='font-weight:600;font-size:0.9rem;color:#0E1B3A'>{label}</span></span>")
    st.markdown("<div style='display:flex;flex-wrap:wrap;gap:8px'>" + "".join(chips) + "</div>", unsafe_allow_html=True)

# ===================== Live Positions (sorted green→amber→red) =====================

def render_positions_table(df: pd.DataFrame) -> None:
    st.subheader("Overall Performance vs Entry")
    if df is None or df.empty:
        st.info("—")
        return

    z = compute_derived_metrics(df).copy()
    z["Market Value"] = pd.to_numeric(z["current_price"], errors="coerce") * pd.to_numeric(z["qty"], errors="coerce")
    # Sort by total P/L % using the green→amber→red rule
    z = _sort_df_green_amber_red(z, "pl_%")

    def _status_row(r):
        """
        Status column with traffic-light logic:
        🟢 ≥ 0%
        🟡 -0.8% ≤ x < 0%
        🔴 < -0.8%
        """
        try:
            pct = float(r.get("pl_%", 0.0))
        except Exception:
            pct = 0.0

        color = _tl_color_for_pct(pct)
        # choose emoji consistent with color
        if color == TL_GREEN:
            dot, word = "🟢", "UP"
        elif color == TL_AMBER:
            dot, word = "🟡", "DOWN"
        else:
            dot, word = "🔴", "DOWN"

        return f"{dot} {word} {pct:+.2f}% / {money_signed(r.get('pl_$', 0.0))}"

    show = pd.DataFrame({
        "Asset":         z.get("Ticker", z.get("symbol")),
        "Status":        z.apply(_status_row, axis=1),
        "Qty":           pd.to_numeric(z["qty"], errors="coerce"),
        "Side":          z["Side"],
        "Avg Entry":     pd.to_numeric(z["entry_price"], errors="coerce"),
        "Current Price": pd.to_numeric(z["current_price"], errors="coerce"),
        "TP":            pd.to_numeric(z.get("TP", z.get("tp_price")), errors="coerce"),
        "Market Value":  pd.to_numeric(z["Market Value"], errors="coerce"),
        "Total P/L ($)": pd.to_numeric(z["pl_$"], errors="coerce"),
        "Total P/L (%)": pd.to_numeric(z["pl_%"], errors="coerce"),
    })

    order = ["Asset","Status","Qty","Side","Avg Entry","Current Price","TP",
             "Market Value","Total P/L ($)","Total P/L (%)"]
    show = show[[c for c in order if c in show.columns]]

    styled = (
        show.style
            .format({
                "Qty":            "{:.0f}",
                "Avg Entry":      "{:.2f}",
                "Current Price":  "{:.2f}",
                "TP":             "{:.2f}",
                "Market Value":   "${:,.2f}",
                "Total P/L ($)":  "${:,.2f}",
                "Total P/L (%)":  "{:+.2f}%",
            }, na_rep="—")
            .apply(_style_row_by_pl, axis=1)
    )

    st.dataframe(styled, width="stretch", hide_index=True)

def _gauge_percent(value: float, *, title: str, good: str = "high", bands=(40,60,80)) -> go.Figure:
    return _banded_gauge(float(value), title, bands=bands, good=good)

def _gauge_count(value: int, total: int, *, title: str) -> go.Figure:
    """Gauge showing a count out of total, with bands at 33/66/100% of total."""
    maxv = max(1, int(total))
    v = max(0, int(value))
    a = int(maxv/3); b = int(2*maxv/3); c = maxv
    steps = [{"range":[0,a], "color":_BAND_RED},
             {"range":[a,b], "color":_BAND_AMBER},
             {"range":[b,c], "color":_BAND_GREEN}]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=v,
        title={"text": title, "font":{"size":14}},
        number={"suffix": f" / {maxv}", "font":{"size":30}},
        gauge={"axis":{"range":[0, maxv], "tickwidth":1, "ticklen":4},
               "bar":{"color": _NEEDLE},
               "bgcolor":"white", "borderwidth":0, "steps":steps}
    ))
    fig.update_layout(height=_DIAL_H, margin=dict(l=6,r=6,t=24,b=0))
    return fig

def render_updated_dials(positions: pd.DataFrame, api: Optional[REST]) -> None:
    st.subheader("Performance")

    if positions is None or positions.empty:
        st.info("No open positions.")
        return

    npos = int(len(positions))

    # Dial 1: % up today
    col_today_usd = next((c for c in ["pl_today_usd","pl_today_$","pl_today","pl_today_val"] if c in positions.columns), None)
    if col_today_usd:
        up_today = int((pd.to_numeric(positions[col_today_usd], errors="coerce") > 0).sum())
    else:
        col_total = next((c for c in ["pl_usd","pl_$"] if c in positions.columns), None)
        up_today = int((pd.to_numeric(positions[col_total], errors="coerce") > 0).sum()) if col_total else 0
    up_today_pct = (100.0 * up_today / max(1, npos))

    # Dial 2: still open since SOD
    touched = _symbols_touched_today(api)
    open_syms = set(positions.get("Ticker", pd.Series(dtype=str)).astype(str).str.upper())
    still_open_since_sod = len([s for s in open_syms if s not in touched])

    # Dial 3: weighted total P/L %
    totals = derive_totals_from_positions(positions) if npos else {}
    total_pl_pct_weighted = float(totals.get("unrealized_pl_%_weighted", 0.0))

    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown("<div style='font-weight:600;margin:0 0 4px 2px;'>% of stocks up today</div>", unsafe_allow_html=True)
        st.plotly_chart(_gauge_percent(up_today_pct, title="", good="high", bands=(40,60,80)),
                config={**PLOTLY_CONFIG, "responsive": True})
        st.markdown("<div style='text-align:center;font-size:13px;color:#64748B;'>Dial 1: positive intraday P&L</div>", unsafe_allow_html=True)
    with d2:
        st.markdown("<div style='font-weight:600;margin:0 0 4px 2px;'># open since start of day</div>", unsafe_allow_html=True)
        st.plotly_chart(_gauge_count(still_open_since_sod, max(1, npos), title=""),
                config={**PLOTLY_CONFIG, "responsive": True})
        st.markdown("<div style='text-align:center;font-size:13px;color:#64748B;'>Dial 2: untouched by fills today</div>", unsafe_allow_html=True)

    with d3:
        st.markdown("<div style='font-weight:600;margin:0 0 4px 2px;'>Total P/L % (open positions)</div>", unsafe_allow_html=True)
        st.plotly_chart(
            _gauge_percent(total_pl_pct_weighted,
                        title="Total P/L % (open positions)",
                        good="high",
                        bands=(1.0, 2.0, 3.0)),   # bands relative to 3 %
            config={**PLOTLY_CONFIG, "responsive": True}
        )
        st.markdown("<div style='text-align:center;font-size:13px;color:#64748B;'>Dial 3: weighted open P&L %</div>", unsafe_allow_html=True)

    # tiny spacer to ensure nothing touches the plots
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# =============================================================================
# Data source: Alpaca live snapshot
# =============================================================================
def _load_alpaca_api() -> Optional[REST]:
    """Build a REST client from config/env; return None on failure."""
    try:
        from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, USE_LIVE_TRADING, ALPACA_LIVE_URL, ALPACA_PAPER_URL
    except Exception:
        ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
        ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
        USE_LIVE_TRADING  = bool(int(os.getenv("USE_LIVE_TRADING", "0")))
        ALPACA_LIVE_URL   = os.getenv("ALPACA_LIVE_URL", "https://api.alpaca.markets")
        ALPACA_PAPER_URL  = os.getenv("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        st.warning("Missing Alpaca credentials. Set env vars or config.py to enable live mode.")
        return None
    base = ALPACA_LIVE_URL if USE_LIVE_TRADING else ALPACA_PAPER_URL
    try:
        return REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=base)
    except Exception as e:
        st.error(f"Alpaca REST init failed: {e}")
        return None

def _tl_color_for_pct(pct_val: float) -> str:
    """
    Updated traffic-light logic:
    - Green: ≥ 0%
    - Amber: < 0% and ≥ -0.8%
    - Red: < -0.8%
    """
    try:
        v = float(pct_val)
    except Exception:
        v = 0.0

    if v >= 0.0:
        return TL_GREEN
    elif v >= -0.8:
        return TL_AMBER
    else:
        return TL_RED

def render_color_system_legend() -> None:
    html = f"""
    <strong>Traffic Lights (per open position) — colour system</strong>
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:6px;">
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="width:14px;height:14px;border-radius:50%;background:{TL_GREEN};
              border:2px solid {TL_GREEN};"></span>
        <span>≥ 0%</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="width:14px;height:14px;border-radius:50%;background:{TL_AMBER};
              border:2px solid {TL_AMBER};"></span>
        <span>-0.8% ≤ x &lt; 0%</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="width:14px;height:14px;border-radius:50%;background:{TL_RED};
              border:2px solid {TL_RED};"></span>
        <span>&lt; -0.8%</span>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_current_status_grid(df: pd.DataFrame) -> None:
    """Two-row mini grid: Today P&L % and Start-of-day P&L % (carry),
    each cell shows '(Total %)' and is sorted/styled by the row's metric.
    """
    st.subheader("Current status")
    if df is None or df.empty:
        st.info("No open positions.")
        return

    z = df.copy()

    # --- column detection
    col_sym   = _first_existing_col(z, "Ticker", "symbol") or "Ticker"
    col_today = _first_existing_col(z, "pl_today_%", "pl_today_pc", "pl_today_pct")
    col_carry = _first_existing_col(z, "carry_%", "carry_pl_%", "carry_pc", "carry_pct")
    col_total = _first_existing_col(z, "pl_%", "pl_pc", "pl_pct")

    # --- derive missing pieces when possible
    if col_total is None and (col_today and col_carry):
        z["pl_%"] = pd.to_numeric(z[col_today], errors="coerce") + pd.to_numeric(z[col_carry], errors="coerce")
        col_total = "pl_%"
    if col_carry is None and (col_total and col_today):
        z["carry_%"] = pd.to_numeric(z[col_total], errors="coerce") - pd.to_numeric(z[col_today], errors="coerce")
        col_carry = "carry_%"
    if col_today is None and (col_total and col_carry):
        z["pl_today_%"] = pd.to_numeric(z[col_total], errors="coerce") - pd.to_numeric(z[col_carry], errors="coerce")
        col_today = "pl_today_%"

    if any(c is None for c in [col_today, col_carry, col_total]):
        st.warning("Unable to compute Current status grid (missing today/carry/total %).")
        return

    # --- series keyed by symbol (use total performance since entry)
    s_total = pd.to_numeric(z.set_index(col_sym)[col_total], errors="coerce")

# --- order by performance since entry (Green→Amber→Red)
    order_total = _order_by_green_amber_red(s_total)

    # --- single display row
    def _fmt_cell(v):
        try:
            return f"{float(v):+.2f}%"
        except Exception:
            return "—"

    row_vals = [_fmt_cell(s_total.get(sym, np.nan)) for sym in order_total]
    row_df = pd.DataFrame([row_vals],
                        index=["Performance since entry"],
                        columns=order_total)

    # --- per-row style helpers (same color logic)
    def _style_row_by_metric(row: pd.Series):
        styles = []
        for col in row.index:
            pct = float(s_total.get(col, 0.0))
            c = _tl_color_for_pct(pct)
            styles.append(f"background-color:{c}22; border:1px solid {c}66; font-weight:700;")
        return styles

    st.dataframe(
        row_df.style.apply(_style_row_by_metric, axis=1),
        width="stretch", hide_index=False
    )

# ===================== Adaptive ATR — table =====================
@st.cache_data(ttl=20, show_spinner=False)
def build_adaptive_atr_df(_api: Optional[REST], positions: pd.DataFrame) -> pd.DataFrame:
    """
    Build a tidy table for the 'Dynamic Stop trailing (Adaptive ATR)' section:
      Ticker | Side | Cost per Share | Current Price | Initial TP | Current TP | Initial SL | Current SL | Updated (ET)
    """
    if positions is None or positions.empty:
        return pd.DataFrame(columns=[
            "Ticker","Side","Cost per Share","Current Price","Initial TP","Current TP","Initial SL","Current SL","Updated (ET)"
        ])

    # Start from live positions + any TP/SL you already attached
    z = merge_tp_sl_from_alpaca_orders(positions.copy(), _api)

    # Initial TP/SL from the entry bracket (per current open lot)
    init_map = _initial_tp_sl_lookup(_api, z)

    rows = []
    for _, r in z.iterrows():
        sym  = str(r.get("Ticker", r.get("symbol", ""))).upper()
        side = str(r.get("Side", r.get("Trade Action",""))).title()

        # entry cost (per share)
        try:
            entry = float(r.get("entry_price", r.get("avg_entry_price", 0.0)) or 0.0)
            cost  = entry if np.isfinite(entry) and entry > 0 else float("nan")
        except Exception:
            cost = float("nan")

        # current price
        try:
            cur_px = float(r.get("current_price", np.nan))
        except Exception:
            cur_px = float("nan")

        # --- current TP/SL (robust: read from open orders and keep only the closing side) ---
        # Close side for a LONG is 'sell'; for a SHORT is 'buy'
        side_exit = "sell" if side.lower() == "long" else "buy"
        tp_px = sl_px = np.nan
        tp_side = sl_side = None
        try:
            # Uses list_orders(..., nested=True), classifies, and picks by relation to price
            tp_px, sl_px, tp_side, sl_side = _get_current_exits_with_sides(_api, sym, cur_px)
        except Exception:
            # fallback to whatever merge put in
            tp_px = pd.to_numeric(pd.Series([r.get("TP", r.get("tp_price"))]), errors="coerce").iloc[0]
            sl_px = pd.to_numeric(pd.Series([r.get("SL", r.get("sl_price"))]), errors="coerce").iloc[0]
            tp_side = sl_side = side_exit

        # Keep only exits on the CORRECT close side
        if tp_side and tp_side != side_exit:
            tp_px = np.nan
        if sl_side and sl_side != side_exit:
            sl_px = np.nan

        cur_tp = float(tp_px) if pd.notna(tp_px) else np.nan
        cur_sl = float(sl_px) if pd.notna(sl_px) else np.nan

        # initial TP/SL (from entry bracket)
        im = init_map.get(sym.upper(), {})
        init_tp = pd.to_numeric(pd.Series([im.get("init_tp")]), errors="coerce").iloc[0]
        init_sl = pd.to_numeric(pd.Series([im.get("init_sl")]), errors="coerce").iloc[0]

        updated = _last_exit_update_et(_api, sym, side)

        rows.append({
            "Ticker": sym,
            "Side": side,
            "Cost per Share": cost,
            "Current Price": cur_px,
            "Initial TP": init_tp,
            "Current TP": cur_tp,
            "Initial SL": init_sl,
            "Current SL": cur_sl,
            "Updated (ET)": updated,
        })

    df = pd.DataFrame(rows, columns=[
        "Ticker","Side","Cost per Share","Current Price","Initial TP","Current TP","Initial SL","Current SL","Updated (ET)"
    ])

    # Round numeric columns
    num_cols = ["Cost per Share","Current Price","Initial TP","Current TP","Initial SL","Current SL"]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: round(x, 2) if pd.notna(x) else x)

    return df

def render_adaptive_atr_table(positions: pd.DataFrame, api: Optional[REST]) -> None:
    st.subheader("Dynamic Stop trailing (Adaptive ATR)")

    if positions is None or positions.empty:
        st.info("No open positions.")
        return

    df = build_adaptive_atr_df(api, positions)
    if df.empty:
        st.info("No data available for Adaptive ATR.")
        return

    # fix tiny header typo if present
    if "Upated (ET)" in df.columns:
        df = df.rename(columns={"Upated (ET)":"Updated (ET)"})

    # Existing columns (you already added TP%/SL% elsewhere)
    df = _add_tp_sl_percent_cols(df)          # already in your file
    df = _add_pnl_percent_col(df)             # A) add PnL %
    df = _add_sl_atr_multiple(df)             # B) add SL (ATR ×) if ATR column exists

    # Numeric formatting
    fmt = {
        "Cost per Share": "${:,.2f}",
        "Current Price":  "${:,.2f}",
        "Initial TP":     "{:.2f}",
        "Current TP":     "{:.2f}",
        "TP %":           _fmt_signed_pct,
        "Initial SL":     "{:.2f}",
        "Current SL":     "{:.2f}",
        "SL %":           _fmt_signed_pct,
        "PnL %":          _fmt_signed_pct,
        "SL (ATR ×)":     _fmt_atr_mult,
    }

    # Build styler: keep your row-level style, then neutralize TP/SL %, color PnL %
    styled = (
        df.style
          .format(fmt, na_rep="—")
          .apply(_style_adaptive_atr, axis=1)                # colors Current Price + Side
    )

    # neutral grey for TP % / SL %
    pct_neutral_cols = [c for c in ("TP %", "SL %") if c in df.columns]
    if pct_neutral_cols:
        styled = styled.map(_style_tp_sl_percent, subset=pct_neutral_cols)

    # green/red for PnL %
    if "PnL %" in df.columns:
        styled = styled.map(_color_pct_signed, subset=["PnL %"])


    # C) tooltips (column help). Note: works on recent Streamlit; if not, add st.caption legend instead.
    column_help = {
        "PnL %": "Side-aware return from entry price. Long: (Price/Cost−1). Short: (Cost/Price−1).",
        "TP %":  "Distance from current price to Current TP as a percent of price.",
        "SL %":  "Distance from current price to Current SL as a percent of price.",
        "SL (ATR ×)": "How many ATRs between Current Price and Current SL (uses ATR column if present).",
    }

    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        column_config={k: st.column_config.TextColumn(k, help=v) for k, v in column_help.items() if k in df.columns}
    )


def _symbols_touched_today(api: Optional[REST]) -> set[str]:
    """Symbols with any fills since ET midnight (today)."""
    df = _pull_all_fills_df(api)
    if df.empty:
        return set()

    try:
        tz = ZoneInfo("US/Eastern")
    except Exception:
        tz = timezone.utc

    now_et = datetime.now(tz)
    sod_et = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
    sod_utc = sod_et.astimezone(timezone.utc)

    touched = df[df["ts"] >= sod_utc]["symbol"].astype(str).str.upper().unique().tolist()
    return set(touched)

def pull_live_positions(api: Optional[REST]) -> pd.DataFrame:
    """
    Pulls open positions and computes consistent fields used across the UI.
    """
    if api is None:
        return pd.DataFrame()
    try:
        raw = api.list_positions()
    except Exception as e:
        st.error(f"Could not fetch positions: {e}")
        return pd.DataFrame()

    rows = []
    for p in raw or []:
        try:
            qty            = float(p.qty)
            entry_price    = float(p.avg_entry_price)
            current_price  = float(p.current_price)
            pl_total_usd   = float(getattr(p, "unrealized_pl", 0.0) or 0.0)
            pl_total_pct   = float(getattr(p, "unrealized_plpc", 0.0) or 0.0) * 100.0
            pl_today_usd   = float(getattr(p, "unrealized_intraday_pl", 0.0) or 0.0)
            pl_today_pct   = float(getattr(p, "unrealized_intraday_plpc", 0.0) or 0.0) * 100.0
        except Exception:
            qty = entry_price = current_price = 0.0
            pl_total_usd = pl_total_pct = pl_today_usd = pl_today_pct = 0.0

        entry_notional = abs(qty * entry_price)
        carry_usd  = pl_total_usd - pl_today_usd
        carry_pct  = (carry_usd / entry_notional * 100.0) if entry_notional > 0 else 0.0

        rows.append({
            "Ticker":        p.symbol,
            "Side":          ("Long" if (getattr(p, "side", "long").lower() == "long") else "Short"),
            "qty":           qty,
            "entry_price":   entry_price,
            "current_price": current_price,
            "notional":      float(getattr(p, "market_value", qty * current_price) or qty * current_price),
            "pl_usd":        pl_total_usd,
            "pl_%":          pl_total_pct,
            "pl_today_usd":  pl_today_usd,
            "pl_today_%":    pl_today_pct,
            "carry_usd":     carry_usd,
            "carry_%":       carry_pct,
        })
    return pd.DataFrame(rows)

# ---------- LEDGER SUMMARY TABLE (Totals / Current / History shell) ----------
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
    _DUBLIN = ZoneInfo("Europe/Dublin")
except Exception:
    _DUBLIN = None

def _fmt_dub(dt: datetime) -> str:
    try:
        if _DUBLIN is not None:
            dt = dt.astimezone(_DUBLIN)
        return dt.strftime("%d-%b-%y")
    except Exception:
        return dt.strftime("%Y-%m-%d")

# ---------- HISTORY FROM FILLS (per day) ----------
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

@st.cache_data(ttl=120, show_spinner=False)
def _pull_fills_df(_api: Optional[REST], days: int = 7) -> pd.DataFrame:
    """Account activities → FILLs for the past N days (ascending)."""
    cols = ["ts","date","symbol","side","qty","price"]
    if _api is None: return pd.DataFrame(columns=cols)

    after = (datetime.now(timezone.utc) - timedelta(days=days+2)).isoformat()
    acts  = _fetch_fills_paged(_api, after_iso=after)

    rows = []
    for a in acts or []:
        sym  = (getattr(a, "symbol", None) or getattr(a, "asset_symbol", None) or "").upper()
        side = (getattr(a, "side", "") or "").lower()
        try:
            price = float(getattr(a, "price", getattr(a, "fill_price", 0.0)) or 0.0)
            qty   = float(getattr(a, "qty",   getattr(a, "quantity", 0.0)) or 0.0)
        except Exception:
            price, qty = 0.0, 0.0
        ts = getattr(a, "transaction_time", getattr(a, "timestamp", getattr(a, "date", None)))
        ts = pd.to_datetime(str(ts), utc=True, errors="coerce")
        if pd.isna(ts) or not sym or qty <= 0 or price <= 0:
            continue
        rows.append({"ts": ts, "date": ts.date(), "symbol": sym, "side": side, "qty": qty, "price": price})
    if not rows: return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows).sort_values("ts")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df


def build_history_rows_from_fills(api: Optional[REST],
                                  positions: pd.DataFrame | None,
                                  days: int = 5) -> tuple[pd.DataFrame, float]:
    """
    Cohort-by-open-day ledger (original behaviour) + calendar-day activity rows:
      - If a day had only closes (no opens), we still emit a row using *close-day* realized P&L.
      - Cohort math is preserved for days that opened exposure.

    Returns (history_df, realized_total_lastN).
    """
    fills = _pull_fills_df(api, days=days)
    if fills.empty:
        return pd.DataFrame(), 0.0

    # Current prices for valuing any leftover cohort lots
    curr_px = {}
    if positions is not None and not positions.empty:
        sym_col = "Ticker" if "Ticker" in positions.columns else "symbol"
        for _, r in positions.iterrows():
            s = str(r.get(sym_col, "")).upper()
            if s:
                try:
                    curr_px[s] = float(r.get("current_price"))
                except Exception:
                    pass

    from collections import defaultdict, deque
    lots = defaultdict(deque)           # symbol -> deque of open lots: {'day','qty'(signed), 'price'}
    day_cost_open   = defaultdict(float)  # keyed by OPEN day
    day_liq_notional= defaultdict(float)  # keyed by OPEN day
    day_realized    = defaultdict(float)  # keyed by OPEN day (cohort)
    # NEW: realized by CLOSE (calendar) day — to surface “previous days” with only closes
    day_realized_close = defaultdict(float)

    # Iterate fills in time order
    for _, r in fills.sort_values("ts").iterrows():
        sym  = r["symbol"]
        side = r["side"]                   # 'buy' or 'sell'
        qty  = float(r["qty"])
        px   = float(r["price"])
        day  = r["date"]                   # this is the *calendar day of the fill* (close day for this trade)

        change = qty if side == "buy" else -qty
        prev_qty = sum(l["qty"] for l in lots[sym])
        new_qty  = prev_qty + change
        opens_more = (abs(new_qty) > abs(prev_qty))  # moving away from zero → open/add

        if opens_more:
            lots[sym].append({"day": day, "qty": change, "price": px})
            day_cost_open[day] += abs(change) * px
        else:
            # closing exposure against FIFO open lots
            qty_to_close = abs(change)
            while qty_to_close > 1e-9 and lots[sym]:
                lot = lots[sym][0]
                lot_sign = 1.0 if lot["qty"] > 0 else -1.0
                avail = abs(lot["qty"])
                take = min(avail, qty_to_close)

                open_day = lot["day"]
                realized_slice = lot_sign * (px - lot["price"]) * take
                # Book to OPEN day (cohort) — original behaviour
                day_realized[open_day] += realized_slice
                day_liq_notional[open_day] += px * take
                # ALSO book to CLOSE day (calendar) — to ensure the date shows
                day_realized_close[day] += realized_slice

                lot["qty"] = lot["qty"] - lot_sign * take
                qty_to_close -= take
                if abs(lot["qty"]) <= 1e-9:
                    lots[sym].popleft()

    # Leftover lots → unrealized by their original OPEN day (cohort view)
    day_open_val = defaultdict(float)
    day_unrl_pnl = defaultdict(float)
    for sym, dq in lots.items():
        cp = curr_px.get(sym, np.nan)
        for lot in dq:
            d = lot["day"]
            q_abs = abs(lot["qty"])
            entry = lot["price"]
            sign  = 1.0 if lot["qty"] > 0 else -1.0
            if np.isfinite(cp):
                day_open_val[d] += q_abs * cp
                day_unrl_pnl[d] += sign * (cp - entry) * q_abs

    # Assemble rows for the last N days:
    # - union of (cohort OPEN days) and (calendar CLOSE days with activity)
    cohort_days = set(day_cost_open.keys()) | set(day_open_val.keys()) | set(day_realized.keys())
    close_days  = set(day_realized_close.keys())
    all_days    = sorted(cohort_days | close_days)
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=days))

    rows = []
    for d in sorted([x for x in all_days if x >= cutoff], reverse=True):
        cost_open = day_cost_open.get(d, 0.0)
        open_val  = day_open_val.get(d, 0.0)
        # Prefer cohort P&L when there was an OPEN that day; otherwise show close-day realized only
        if d in cohort_days:
            rlz_cohort = day_realized.get(d, 0.0)
            unrl_cohort = day_unrl_pnl.get(d, 0.0)
            pl_d = rlz_cohort + unrl_cohort
            pl_pct = (pl_d / cost_open * 100.0) if cost_open > 0 else 0.0
            realized_show = rlz_cohort
        else:
            # close-only day: no open cost/value context — show realized only, leave others 0
            realized_show = day_realized_close.get(d, 0.0)
            pl_d = realized_show
            pl_pct = 0.0

        rows.append({
            "date": d.strftime("%d-%b-%y"),
            "cost_open": round(cost_open, 2),
            "open_value": round(open_val, 2),
            "realized": round(realized_show, 2),
            "pl_dollar": round(pl_d, 2),
            "pl_percent": round(pl_pct, 2),
        })

    hist_df = pd.DataFrame(rows, columns=["date","cost_open","open_value","realized","pl_dollar","pl_percent"])
    realized_total = float(sum(day_realized.values()))  # keep the cohort definition for totals
    return hist_df, realized_total


# ---------- OPEN LEDGER (sign-correct for shorts) ----------
def _compute_open_ledger(positions: pd.DataFrame) -> dict:
    """Cost to open, current value, and UNRL P&L (+/−) for all currently open lots."""
    if positions is None or positions.empty:
        return {"cost_open": 0.0, "open_value": 0.0, "unrl": 0.0, "unrl_pct": 0.0}

    z = compute_derived_metrics(positions).copy()
    qty_abs  = pd.to_numeric(z.get("qty"), errors="coerce").abs()
    entry    = pd.to_numeric(z.get("entry_price"), errors="coerce")
    current  = pd.to_numeric(z.get("current_price"), errors="coerce")

    # Notional views (always positive)
    cost_open  = float((qty_abs * entry).sum(skipna=True))
    open_value = float((qty_abs * current).sum(skipna=True))

    # UNRL P&L with correct sign for shorts (sum of row P&L)
    unrl = float(pd.to_numeric(z.get("pl_$"), errors="coerce").sum(skipna=True))
    unrl_pct = (unrl / cost_open * 100.0) if cost_open > 0 else 0.0

    return {
        "cost_open": round(cost_open, 2),
        "open_value": round(open_value, 2),
        "unrl": round(unrl, 2),
        "unrl_pct": round(unrl_pct, 2),
    }

def render_portfolio_ledger_table(positions: pd.DataFrame,
                                  realized_pnl_total: float | None = None,
                                  history_rows: pd.DataFrame | None = None) -> None:
    """
    Ledger where History rows use their cohort math:
      History P&L $ = realized(cohort) + unrealized(cohort)
      History P&L % = History P&L $ / cost_open
    Current row uses live snapshot:
      Current P&L $ = (open_value - cost_open) + realized_now
    Totals are SUMS of History cohorts across the window.
    """
    st.subheader("Portfolio Ledger")

    # ---- Live snapshot for "Current"
    o = _compute_open_ledger(positions)  # cost_open, open_value, unrl (sum of row P&L, sign-correct incl. shorts)
    realized_now = float(realized_pnl_total or 0.0)
    pnl_current  = o["unrl"] + realized_now
    pct_current  = (pnl_current / o["cost_open"] * 100.0) if o["cost_open"] > 0 else np.nan
    # ---- Prepare History slice (already cohort-correct)
    hist = None
    if isinstance(history_rows, pd.DataFrame) and not history_rows.empty:
        hist = history_rows.copy()
        for c in ["cost_open","open_value","realized","pl_dollar","pl_percent"]:
            if c in hist.columns:
                hist[c] = pd.to_numeric(hist[c], errors="coerce").fillna(0.0)

        # Totals from cohorts
        cost_sum = float(hist["cost_open"].sum())
        open_sum = float(hist["open_value"].sum())
        rlz_sum  = float(hist["realized"].sum())
        pnl_sum  = float(hist["pl_dollar"].sum())
        pct_sum  = (pnl_sum / cost_sum * 100.0) if cost_sum > 0 else np.nan
    else:
        cost_sum = open_sum = rlz_sum = pnl_sum = 0.0
        pct_sum  = np.nan

    # ---- Build display rows
    rows = []
    rows.append({
        "Section": "Totals",
        "Date": "",
        "Cost to open positions": cost_sum,
        "Open (current value)":   open_sum,
        "Liquidated":             rlz_sum,
        "P&L $":                  pnl_sum,
        "P&L %":                  pct_sum,
    })

    rows.append({
        "Section": "Current",
        "Date": _fmt_dub(datetime.now(timezone.utc)),
        "Cost to open positions": o["cost_open"],
        "Open (current value)":   o["open_value"],
        "Liquidated":             realized_now,
        "P&L $":                  pnl_current,
        "P&L %":                  pct_current,
    })

    if hist is not None:
        for _, r in hist.iterrows():
            rows.append({
                "Section": "History",
                "Date": str(r.get("date","")),
                "Cost to open positions": float(r.get("cost_open",0.0)),
                "Open (current value)":   float(r.get("open_value",0.0)),
                "Liquidated":             float(r.get("realized",0.0)),
                "P&L $":                  float(r.get("pl_dollar",0.0)),   # <-- use cohort P&L
                "P&L %":                  float(r.get("pl_percent",0.0)), # <-- use cohort %
            })

    df = pd.DataFrame(rows, columns=[
        "Section","Date","Cost to open positions","Open (current value)","Liquidated","P&L $","P&L %"
    ])

    # ---- Styling
    def _style_ledger(s: pd.Series) -> pd.Series:
        sty = pd.Series("", index=s.index, dtype="object")
        v = float(s.get("P&L $", 0) or 0)
        color = "#16A34A" if v >= 0 else "#DC2626"
        sty["P&L $"] = f"color:{color}; font-weight:700;"
        sty["P&L %"] = f"color:{color}; font-weight:700;"
        return sty

    styled = (df.style
                .format({
                    "Cost to open positions": "${:,.2f}",
                    "Open (current value)":   "${:,.2f}",
                    "Liquidated":             "${:,.2f}",
                    "P&L $":                  "${:,.2f}",
                    "P&L %":                  "{:+.2f}%"
                }, na_rep="—")
                .apply(_style_ledger, axis=1))

    st.dataframe(styled, width="stretch", hide_index=True)
    st.caption("“Liquidated” = realized P&L (if provided). History P&L uses cohort math (realized + unrealized). Current P&L = (Open − Cost) + Realized.")


# =============================================================================
# Main — orchestrate sections (no duplicates)
# =============================================================================
def main() -> None:
    api = _load_alpaca_api()
    render_header(api)  # clock + heading + ticker ribbon

    # === Load positions once (and enrich) — use everywhere below ===
    positions = pull_live_positions(api)
    positions = merge_tp_sl_from_alpaca_orders(positions, api)
    positions = compute_derived_metrics(positions)

    # hReview_Summary.py — inside main(), right after positions are built
    tabs = st.tabs(["📈 Portfolio", "📰 Sentiment & News"])

    with tabs[0]:
        # (move the existing sections into this block)
        render_perf_and_risk_kpis(api, positions)
        st.divider()
        render_traffic_lights(positions)
        render_color_system_legend()
        st.divider()
        info = render_portfolio_equity_chart(api)
        st.divider()
        render_spy_vs_quantml_daily(api, period=(info or {}).get("period", "1M"))
        st.divider()
        render_current_status_grid(positions)
        st.divider()
        render_adaptive_atr_table(positions, api)
        st.divider()
        effective_atr = {"mode":"ATR", "step_atr":0.50, "trigger_atr":1.00}
        render_positions_panel(api, positions, atr_mode=effective_atr)
        st.divider()
        render_positions_table(positions)
        st.divider()
        render_updated_dials(positions, api)
        st.divider()

    with tabs[1]:
        render_sentiment_news_panel(positions_df=positions)

    st.divider()

if __name__ == "__main__":
    main()
