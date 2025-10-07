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
import uuid
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from alpaca_trade_api.rest import REST

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
# --- Traffic Light thresholds (percent points, not decimals) ---
TL_THRESH_PCT = 0.10   # ±0.10% band -> amber
TL_GREEN = BRAND["success"]
TL_AMBER = BRAND["warning"]
TL_RED   = BRAND["danger"]


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

# ===================== Sorting + Drawdown helpers =====================

def _traffic_group(value: float, thr: float = TL_THRESH_PCT) -> int:
    """0 = green (>= +thr), 1 = amber (-thr..+thr), 2 = red (<= -thr)"""
    try:
        v = float(value)
    except Exception:
        v = 0.0
    if v >= thr:
        return 0
    if v <= -thr:
        return 2
    return 1

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
    """Return index labels ordered: green (desc), amber (|x| asc), red (asc)."""
    v = pd.to_numeric(series, errors="coerce").fillna(0.0)
    grp = v.apply(_traffic_group)  # 0,1,2
    within = np.where(grp.eq(0), -v, np.where(grp.eq(2), v, v.abs()))
    tmp = pd.DataFrame({"sym": v.index, "grp": grp, "within": within})
    tmp = tmp.sort_values(["grp", "within"])
    return tmp["sym"].tolist()

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

def _load_fills_dataframe(api, days: int) -> pd.DataFrame:
    """
    Pull account activities (FILL) for the last `days` days and normalize.
    Works with Alpaca v2 REST `get_activities('FILL')`.
    """
    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(days=days)
    acts = api.get_activities(activity_types="FILL", after=start.isoformat(), until=end.isoformat())

    rows = []
    for a in acts:
        # a has: symbol, side, qty, price, cum_qty, transaction_time, leaves_qty, order_id, id, ...
        rows.append({
            "symbol": a.symbol,
            "time": _to_dt(a.transaction_time),
            "side": str(a.side).lower(),             # 'buy'/'sell' for equities (or 'sell_short' on some brokers)
            "qty": float(a.qty),                     # always positive in activities
            "price": float(a.price),
            "order_id": getattr(a, "order_id", None),
            "id": getattr(a, "id", None),
            "commission": float(getattr(a, "net_amount", 0.0)) * 0.0,  # placeholder if you don’t have fees field
            "fee": float(getattr(a, "fee", 0.0)) if hasattr(a, "fee") else 0.0,
        })
    df = pd.DataFrame(rows).sort_values(["symbol", "time"]).reset_index(drop=True)
    if df.empty:
        return df

    # Normalize side to signed quantity: buys increase long, sells decrease long (or cover short)
    # Use +qty for buys, -qty for sells
    df["signed_qty"] = np.where(df["side"].str.contains("buy"), df["qty"], -df["qty"])
    return df

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
    df[num_cols] = df[num_cols].applymap(lambda x: round(x, 2) if pd.notna(x) else x)

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
        use_container_width=True, hide_index=True
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
                    }), use_container_width=True, hide_index=True
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
      1) greens first (high→low), then amber (closest to zero first), then red (low→high)
    """
    z = df.copy()
    v = pd.to_numeric(z[pct_col], errors="coerce").fillna(0.0)
    grp = v.apply(_traffic_group)
    within = np.where(grp.eq(0), -v, np.where(grp.eq(2), v, v.abs()))
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
    """
    Pull ALL FILL activities since `start` (default: 2000-01-01), robust to old/new Alpaca SDKs.
    Returns columns: ts (UTC), symbol, side, qty, price, fee.
    """
    cols = ["ts", "symbol", "side", "qty", "price", "fee"]
    if _api is None:
        return pd.DataFrame(columns=cols)

    # go far back by default
    if start is None:
        start = datetime(2000, 1, 1, tzinfo=timezone.utc)
    after_iso = start.isoformat()

    acts = []
    # try the different SDK signatures
    try:
        acts = _api.get_activities(activity_types="FILL", after=after_iso)
    except TypeError:
        try:
            acts = _api.get_activities("FILL", after=after_iso)
        except Exception:
            try:
                acts = _api.get_account_activities("FILL", after=after_iso)
            except Exception:
                acts = []

    rows = []
    for a in acts or []:
        sym  = (getattr(a, "symbol", None) or getattr(a, "asset_symbol", None) or "").upper()
        side = (getattr(a, "side", "") or "").lower()  # 'buy' or 'sell'
        try:
            price = float(getattr(a, "price", getattr(a, "fill_price", 0.0)) or 0.0)
            qty   = float(getattr(a, "qty",   getattr(a, "quantity", 0.0)) or 0.0)
        except Exception:
            price, qty = 0.0, 0.0

        # commission/fees (best‑effort across SDKs)
        fee = None
        for name in ("commission", "commissions", "fee", "fees", "trade_fees", "transaction_fees"):
            v = getattr(a, name, None)
            if v is not None:
                try:
                    fee = float(v)
                    break
                except Exception:
                    pass
        fee = float(fee if fee is not None else 0.0)

        ts = getattr(a, "transaction_time", getattr(a, "timestamp", getattr(a, "date", None)))
        ts = pd.to_datetime(str(ts), utc=True, errors="coerce")
        if pd.isna(ts) or not sym or qty <= 0 or price <= 0:
            continue
        rows.append({"ts": ts, "symbol": sym, "side": side, "qty": qty, "price": price, "fee": fee})

    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df.sort_values("ts").reset_index(drop=True)

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
            use_container_width=True, hide_index=True
        )
    with c2:
        st.markdown("**Daily summary**")
        st.dataframe(
            day_group.style.format({"Realized $":"${:,.2f}","Fees $":"${:,.2f}","P&L $":"${:,.2f}"}),
            use_container_width=True, hide_index=True
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
    st.subheader("Your portfolio (Alpaca)")

    # Period selector
    period = st.radio(
        "Period",
        options=["1D", "1M", "3M", "all"],
        horizontal=True,
        label_visibility="collapsed",
        key="ph_period",
    )

    # Fetch series (get_portfolio_history_df handles 1Y->1A and timeframe defaults)
    try:
        df = get_portfolio_history_df(api, period=period)
    except Exception as e:
        st.error(f"Could not load portfolio history: {e}")
        return {"period": period, "error": str(e)}

    if df.empty:
        st.info("No portfolio history is available.")
        return {"period": period}

    # Tighten the y-range so the area doesn't flood the plot
    y = pd.to_numeric(df["equity"], errors="coerce").astype(float)
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    span = ymax - ymin
    pad = max(1.0, 0.02 * span) if np.isfinite(span) else 1.0
    yrange = [ymin - pad, ymax + pad] if np.isfinite(ymin) and np.isfinite(ymax) else None

    # Main area chart (low-opacity fill)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["ts"],
        y=y,
        mode="lines",
        name="Equity",
        fill="tozeroy",
        fillcolor="rgba(79,70,229,0.15)",  # soft fill
        line=dict(width=2),
    ))
    fig.update_layout(
        height=280,
        showlegend=False,
        margin=dict(l=8, r=8, t=6, b=6),
        yaxis_title=None,
        xaxis_title=None,
    )
    if yrange:
        fig.update_yaxes(range=yrange)

    st.plotly_chart(fig, use_container_width=True)

    # Quick stats from the plotted series
    chg_pct = float(df["ret_pct"].iloc[-1]) if len(df) else float("nan")
    idd_pct = _max_drawdown_pct(y)
    st.caption(f"Change {period}: {chg_pct:+.2f}% · Max drawdown over period: {idd_pct:.2f}%")

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

# ===================== Extra KPIs + Contributors/Detractors =====================
def render_perf_and_risk_kpis(api: Optional[REST], positions: pd.DataFrame) -> None:
    """
    KPIs: Day P&L ($/% from equity curve) and Today total P&L ($/% from positions intraday),
    plus top contributors/detractors. (Exposure/MTD/QTD/YTD computed but not shown.)
    """
    # ===== Intraday equity curve (for Day P&L $/% and the % base) =====
    intraday = get_portfolio_history_df(api, period="1D")
    day_pl_usd = day_pl_pct = np.nan
    start_equity = np.nan
    if not intraday.empty:
        start_equity = float(intraday["equity"].iloc[0])
        last_equity  = float(intraday["equity"].iloc[-1])
        day_pl_usd   = last_equity - start_equity
        day_pl_pct   = (day_pl_usd / start_equity * 100.0) if start_equity > 0 else np.nan

    # ===== Today total P&L from open positions (sum of intraday P&L across positions) =====
    today_total_pl_usd = np.nan
    today_total_pl_pct = np.nan
    if positions is not None and not positions.empty:
        z = compute_derived_metrics(positions).copy()

        # Try multiple possible intraday P&L columns
        intraday_cols_usd = [
            "pl_today_usd",                # from compute_derived_metrics()
            "unrealized_intraday_pl",      # Alpaca field
            "intraday_pl_usd",
        ]
        c_today_usd = next((c for c in intraday_cols_usd if c in z.columns), None)

        if c_today_usd is None:
            # Fallback: (current - prev close) * qty if those fields exist
            if {"current_price", "prev_close", "qty"}.issubset(set(z.columns)):
                today_vec = (pd.to_numeric(z["current_price"], errors="coerce")
                            - pd.to_numeric(z["prev_close"],  errors="coerce")) \
                            * pd.to_numeric(z["qty"], errors="coerce")
                today_total_pl_usd = float(np.nansum(today_vec.to_numpy()))
            else:
                today_total_pl_usd = np.nan
        else:
            today_total_pl_usd = float(pd.to_numeric(z[c_today_usd], errors="coerce").sum())

        if start_equity and np.isfinite(start_equity) and start_equity > 0 and np.isfinite(today_total_pl_usd):
            today_total_pl_pct = float(today_total_pl_usd / start_equity * 100.0)

    # ===== (Optional) Exposure & leverage from account snapshot — computed but not shown =====
    acct = pull_account_snapshot(api)
    equity   = float(acct.get("equity") or 0.0)
    long_mv  = float(acct.get("long_market_value")  or 0.0)
    short_mv = float(acct.get("short_market_value") or 0.0)
    gross = abs(long_mv) + abs(short_mv)
    net   = long_mv + short_mv
    leverage = (gross / equity * 100.0) if equity > 0 else np.nan

    # ===== (Optional) Rolling returns — computed but not shown =====
    rolls = compute_period_returns(api)
    mtd, qtd, ytd = rolls.get("MTD", np.nan), rolls.get("QTD", np.nan), rolls.get("YTD", np.nan)

    # ===== Top movers today by $ P&L =====
    winners = losers = pd.DataFrame()
    if positions is not None and not positions.empty:
        z = compute_derived_metrics(positions).copy()
        base_col = "pl_today_usd" if "pl_today_usd" in z.columns else ("pl_$" if "pl_$" in z.columns else None)
        if base_col:
            tmp = pd.DataFrame({
                "Symbol": z.get("Ticker", z.get("symbol")),
                "P&L $": pd.to_numeric(z[base_col], errors="coerce"),
            })
            winners = tmp.sort_values("P&L $", ascending=False).head(3)
            losers  = tmp.sort_values("P&L $", ascending=True).head(3)

        # ===== KPI cards in requested order (swapped + renamed) =====
    c1, c2, c3, c4 = st.columns(4)

    # Portfolio-level P&L (previously Day P&L)
    with c1:
        _kpi_card("Today Portfolio P&L %",
                  f"{(day_pl_pct if np.isfinite(day_pl_pct) else 0):+.2f}%",
                  "pos" if (day_pl_pct or 0) >= 0 else "neg")

    with c2:
        _kpi_card("Today Portfolio P&L $", money(day_pl_usd),
                  "pos" if (day_pl_usd or 0) >= 0 else "neg")

    with c3:
        _kpi_card("Today Open Positions P&L %",
                  f"{(today_total_pl_pct if np.isfinite(today_total_pl_pct) else 0):+.2f}%",
                  "pos" if (today_total_pl_pct or 0) >= 0 else "neg")

    # Open positions intraday P&L (previously Today total P&L)
    with c4:
        _kpi_card("Today Open Positions P&L $", money(today_total_pl_usd),
                  "pos" if (today_total_pl_usd or 0) >= 0 else "neg")
        

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
                        use_container_width="stretch")
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

@st.cache_data(ttl=15, show_spinner=False)
def _open_exits_df(_api) -> pd.DataFrame:
    """
    Flat table of open legs: symbol, side, leg_type (limit/stop), limit_price, stop_price, submitted_at, status.
    Leading underscore on _api avoids Streamlit hashing the REST client (prevents UnhashableParamError).
    """
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

    def _price(o, kind):
        if kind == "limit":
            return float(getattr(o, "limit_price", getattr(o, "price", np.nan)) or np.nan)
        return float(getattr(o, "stop_price", getattr(o, "price", np.nan)) or np.nan)

    rows = []
    for o in orders or []:
        sym  = (getattr(o, "symbol", None) or getattr(o, "asset_symbol", None) or "").upper()
        parent_side = (getattr(o, "side", None) or "").lower()  # e.g., 'buy' or 'sell' for the entry
        legs = getattr(o, "legs", None) or []

        # Helper: closing side is opposite of parent (for bracket legs)
        def _closing_side(pside: str) -> str:
            return "buy" if str(pside).lower() == "sell" else "sell"

        for leg in ([o] + list(legs)):
            ltype = (getattr(leg, "type", None) or getattr(leg, "order_type", None) or "").lower()
            lstat = (getattr(leg, "status", None) or "").lower()
            if lstat and lstat not in _OPEN_STATES:
                continue

            # Classify leg
            is_stop  = ltype.startswith("stop")
            is_limit = ("limit" in ltype) or ("take_profit" in ltype)

            # Prefer leg.side; if missing on bracket legs, infer the closing side
            raw_leg_side = (getattr(leg, "side", None) or "").lower()
            side = raw_leg_side if raw_leg_side else (_closing_side(parent_side) if (is_stop or is_limit) else parent_side)

            # Prices
            def _price(obj, kind):
                if kind == "limit":
                    return float(getattr(obj, "limit_price", getattr(obj, "price", np.nan)) or np.nan)
                return float(getattr(obj, "stop_price",  getattr(obj, "price", np.nan)) or np.nan)

            # --- in _open_exits_df() right before rows.append({...}) ---
            is_stop     = ltype.startswith("stop")
            is_trailing = ("trail" in ltype) or ("trailing" in ltype)   # NEW
            is_limit    = ("limit" in ltype) or ("take_profit" in ltype)

            rows.append({
                "symbol": sym,
                "side": side,
                "leg_type": ("trailing_stop" if is_trailing else ("stop" if is_stop else ("limit" if is_limit else ltype))),
                "limit_price": _price(leg, "limit"),
                "stop_price":  _price(leg, "stop"),
                # NEW: capture trailing fields if present
                "trail_price":  float(getattr(leg, "trail_price",  np.nan) or np.nan),
                "trail_percent":float(getattr(leg, "trail_percent",np.nan) or np.nan),
                "submitted_at": _ts(leg),
                "status": lstat or (getattr(o, "status", None) or "").lower(),
            })

        # Fallback: Some Alpaca bracket orders keep TP/SL on the parent (no legs)
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
            close_side = "buy" if parent_side == "sell" else "sell"
            parent_status = (getattr(o, "status", None) or "").lower()
            parent_ts = _ts(o)

            if tp_obj is not None:
                tp_px = _get_attr_or_key(tp_obj, "limit_price") or _get_attr_or_key(tp_obj, "price")
                tp_px = float(tp_px) if tp_px is not None else np.nan
                rows.append({
                    "symbol": sym,
                    "side": close_side,
                    "leg_type": "limit",
                    "limit_price": tp_px,
                    "stop_price":  np.nan,
                    "trail_price":  np.nan,
                    "trail_percent":np.nan,
                    "submitted_at": parent_ts,
                    "status": parent_status,
                })

            if sl_obj is not None:
                stop_px = _get_attr_or_key(sl_obj, "stop_price")
                # For stop_limit, also capture limit_price
                stop_limit_px = _get_attr_or_key(sl_obj, "limit_price")
                trail_px = _get_attr_or_key(sl_obj, "trail_price")
                trail_pct = _get_attr_or_key(sl_obj, "trail_percent")
                # Determine leg_type for SL
                leg_t = "trailing_stop" if (trail_px is not None or trail_pct is not None) else ("stop_limit" if stop_limit_px is not None else "stop")
                rows.append({
                    "symbol": sym,
                    "side": close_side,
                    "leg_type": leg_t,
                    "limit_price": float(stop_limit_px) if stop_limit_px is not None else np.nan,
                    "stop_price":  float(stop_px) if stop_px is not None else np.nan,
                    "trail_price":  float(trail_px) if trail_px is not None else np.nan,
                    "trail_percent":float(trail_pct) if trail_pct is not None else np.nan,
                    "submitted_at": parent_ts,
                    "status": parent_status,
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["submitted_at"] = pd.to_datetime(df["submitted_at"], utc=True, errors="coerce")
    return df

def _pick_tp_sl_for(sym: str, side_long_short: str, exits_df: pd.DataFrame) -> tuple[float, float]:
    """LONG → TP = sell LIMIT, SL = sell STOP; SHORT → TP = buy LIMIT, SL = buy STOP (keep newest)."""
    if exits_df is None or exits_df.empty or not sym:
        return (np.nan, np.nan)

    # --- in _pick_tp_sl_for() ---
    want_side = "sell" if str(side_long_short).strip().lower() == "long" else "buy"
    d = exits_df[(exits_df["symbol"].str.upper() == str(sym).upper()) & (exits_df["side"] == want_side)]
    if d.empty:
        return (np.nan, np.nan)

    tp = np.nan; sl = np.nan
    lim = d[d["leg_type"] == "limit"].sort_values("submitted_at").tail(1)
    if not lim.empty:
        v = lim["limit_price"].iloc[0]
        tp = float(v) if pd.notna(v) else np.nan

    # Accept stop, stop_limit or trailing_stop for SL
    stp = d[d["leg_type"].isin(["stop","stop_limit","trailing_stop"])].sort_values("submitted_at").tail(1)
    if not stp.empty:
        v_stop  = stp["stop_price"].iloc[0]  if "stop_price"  in stp.columns else np.nan
        v_trail = stp["trail_price"].iloc[0] if "trail_price" in stp.columns else np.nan
        sl = float(v_stop) if pd.notna(v_stop) else (float(v_trail) if pd.notna(v_trail) else np.nan)

    return (tp, sl)


def merge_tp_sl_from_alpaca_orders(positions: pd.DataFrame, api) -> pd.DataFrame:
    """Attach TP/SL columns from broker open orders (resilient if empty)."""
    out = positions.copy() if positions is not None else pd.DataFrame()
    if out is None or out.empty or api is None:
        for c in ("TP", "SL", "tp_price", "sl_price"):
            if c not in out.columns:
                out[c] = np.nan
        return out

    exits = _open_exits_df(api)
    sym_col = "Ticker" if "Ticker" in out.columns else ("symbol" if "symbol" in out.columns else None)
    if not sym_col:
        for c in ("TP", "SL", "tp_price", "sl_price"):
            if c not in out.columns:
                out[c] = np.nan
        return out

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
        st.plotly_chart(_gauge_percent(up_today_pct, title="", good="high", bands=(40,60,80)), use_container_width=True)
        st.markdown("<div style='text-align:center;font-size:13px;color:#64748B;'>Dial 1: positive intraday P&L</div>", unsafe_allow_html=True)

    with d2:
        st.markdown("<div style='font-weight:600;margin:0 0 4px 2px;'># open since start of day</div>", unsafe_allow_html=True)
        st.plotly_chart(_gauge_count(still_open_since_sod, max(1, npos), title=""), use_container_width=True)
        st.markdown("<div style='text-align:center;font-size:13px;color:#64748B;'>Dial 2: untouched by fills today</div>", unsafe_allow_html=True)

    with d3:
        st.markdown("<div style='font-weight:600;margin:0 0 4px 2px;'>Total P/L % (open positions)</div>", unsafe_allow_html=True)
        st.plotly_chart(
            _gauge_percent(total_pl_pct_weighted,
                        title="Total P/L % (open positions)",
                        good="high",
                        bands=(1.0, 2.0, 3.0)),   # bands relative to 3 %
            use_container_width=True
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
        use_container_width=True, hide_index=False
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
    if _api is None:
        return pd.DataFrame(columns=cols)

    # Timezone-aware 'after' so Alpaca honors the filter
    after = (datetime.now(timezone.utc) - timedelta(days=days+2)).isoformat()

    acts = []
    try:
        acts = _api.get_activities(activity_types="FILL", after=after)
    except TypeError:
        try:
            acts = _api.get_activities("FILL", after=after)
        except Exception:
            acts = []
    except Exception:
        try:
            acts = _api.get_account_activities("FILL", after=after)
        except Exception:
            acts = []

    rows = []
    for a in acts or []:
        sym  = (getattr(a, "symbol", None) or getattr(a, "asset_symbol", None) or "").upper()
        side = (getattr(a, "side", "") or "").lower()
        try:
            price = float(getattr(a, "price", getattr(a, "fill_price", 0.0)) or 0.0)
            qty   = float(getattr(a, "qty", getattr(a, "quantity", 0.0)) or 0.0)
        except Exception:
            price, qty = 0.0, 0.0
        ts = getattr(a, "transaction_time", getattr(a, "timestamp", getattr(a, "date", None)))
        ts = pd.to_datetime(str(ts), utc=True, errors="coerce")
        if pd.isna(ts) or not sym or qty <= 0 or price <= 0:
            continue
        rows.append({"ts": ts, "date": ts.date(), "symbol": sym, "side": side, "qty": qty, "price": price})

    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df.sort_values("ts")


def build_history_rows_from_fills(api: Optional[REST],
                                  positions: pd.DataFrame | None,
                                  days: int = 5) -> tuple[pd.DataFrame, float]:
    """
    FIFO lot tracker per symbol:
      - 'Cost to open positions' (notional added) per day
      - 'Liquidated'  = realized P&L for portions closed that day
      - 'Open (current value)' for that day's remaining lots (valued at *current* price)
      - 'P&L $'       = realized (that day) + unrealized on remaining lots from that day
      - 'P&L %'       = P&L $ / day_cost_open
    Returns (history_df, realized_total_lastN).
    """
    fills = _pull_fills_df(api, days=days)
    if fills.empty:
        return pd.DataFrame(), 0.0

    # current prices for open lots valuation
    curr_px = {}
    if positions is not None and not positions.empty:
        sym_col = "Ticker" if "Ticker" in positions.columns else "symbol"
        for _, r in positions.iterrows():
            s = str(r.get(sym_col, "")).upper()
            if s:
                try: curr_px[s] = float(r.get("current_price"))
                except Exception: pass

    lots = defaultdict(deque)               # symbol -> deque of open lots: {'day','qty'(signed),'price'}
    day_cost_open = defaultdict(float)
    day_liq_notional = defaultdict(float)
    day_realized = defaultdict(float)

    for _, r in fills.iterrows():
        sym, side, qty, px, day = r["symbol"], r["side"], float(r["qty"]), float(r["price"]), r["date"]
        change = qty if side == "buy" else -qty
        prev_qty = sum(l["qty"] for l in lots[sym])
        new_qty  = prev_qty + change
        inc_exp  = (abs(new_qty) > abs(prev_qty))  # moving away from zero → opening/add

        if inc_exp:
            lots[sym].append({"day": day, "qty": change, "price": px})
            day_cost_open[day] += abs(change) * px
        else:
            # closing exposure: FIFO across lots
            qty_to_close = abs(change)
            while qty_to_close > 1e-9 and lots[sym]:
                lot = lots[sym][0]
                lot_sign = 1.0 if lot["qty"] > 0 else -1.0
                avail = abs(lot["qty"])
                take = min(avail, qty_to_close)
                # realized P&L for this slice
                open_day = lot["day"]
                day_realized[open_day] += lot_sign * (px - lot["price"]) * take
                day_liq_notional[open_day] += px * take
                lot["qty"] = lot["qty"] - lot_sign * take
                qty_to_close -= take
                if abs(lot["qty"]) <= 1e-9:
                    lots[sym].popleft()

    # leftover lots → unrealized by original open day
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

    # assemble last N days (descending)
    all_days = sorted(set(list(day_cost_open.keys()) + list(day_open_val.keys()) + list(day_realized.keys())))
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=days))
    rows = []
    for d in sorted([x for x in all_days if x >= cutoff], reverse=True):
        cost_open = day_cost_open.get(d, 0.0)
        open_val  = day_open_val.get(d, 0.0)
        realized  = day_realized.get(d, 0.0)
        pl_d      = realized + day_unrl_pnl.get(d, 0.0)
        pl_pct    = (pl_d / cost_open * 100.0) if cost_open > 0 else 0.0
        rows.append({
            "date": d.strftime("%d-%b-%y"),
            "cost_open": round(cost_open, 2),
            "open_value": round(open_val, 2),
            "realized": round(realized, 2),
            "pl_dollar": round(pl_d, 2),
            "pl_percent": round(pl_pct, 2),
        })

    hist_df = pd.DataFrame(rows, columns=["date","cost_open","open_value","realized","pl_dollar","pl_percent"])
    realized_total = float(sum(day_realized.values()))
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
    pnl_current  = (o["open_value"] - o["cost_open"]) + realized_now
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

    st.dataframe(styled, use_container_width="stretch", hide_index=True)
    st.caption("“Liquidated” = realized P&L (if provided). History P&L uses cohort math (realized + unrealized). Current P&L = (Open − Cost) + Realized.")


# =============================================================================
# Main — orchestrate sections (no duplicates)
# =============================================================================
def main() -> None:
    api = _load_alpaca_api()
    render_header(api)  # clock + heading + ticker ribbon

    # Live positions + attach TP/SL and derived metrics
    positions = pull_live_positions(api)
    positions = merge_tp_sl_from_alpaca_orders(positions, api)
    positions = compute_derived_metrics(positions)

    st.divider()
    render_perf_and_risk_kpis(api, positions)
    st.divider()
    # ----- Traffic Lights
    render_traffic_lights(positions)
    st.divider()
    render_color_system_legend()
    st.divider()

    # === Your portfolio (Alpaca)
    render_portfolio_equity_chart(api)
    st.divider()
    # ----- Current status (mini-grid) + colour legend -----
    render_current_status_grid(positions)
    st.divider()

    # Live Positions table (sorted) -----
    render_positions_table(positions)
    st.divider()

    # ----- Dials (existing) -----
    render_updated_dials(positions, api)
    st.divider()

    # ----- Portfolio Ledger (existing) -----
    hist_df, realized_total = build_history_rows_from_fills(api, positions, days=5)
    render_portfolio_ledger_table(positions, realized_pnl_total=realized_total, history_rows=hist_df)

    st.divider()
    # ----- NEW: Attach your uploaded transaction history file -----
    render_transaction_history_positions(api, days=180)

    # ===== Contributors / Detractors (moved and fixed) =====
    st.divider()
    z = compute_derived_metrics(positions).copy()

    base_col = "pl_$" if "pl_$" in z.columns else (
        "pl_today_usd" if "pl_today_usd" in z.columns else None
    )

    if base_col:
        tmp = pd.DataFrame({
            "Symbol": z.get("Ticker", z.get("symbol")),
            "P&L $": pd.to_numeric(z[base_col], errors="coerce"),
        })

        winners = tmp.sort_values("P&L $", ascending=False).head(3)
        losers = tmp.sort_values("P&L $", ascending=True).head(3)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top contributors (since entry)**")
            st.table(
                winners.assign(**{"P&L $": winners["P&L $"].map(lambda x: f"{x:,.2f}")})
            )
        with c2:
            st.markdown("**Top detractors (since entry)**")
            st.table(
                losers.assign(**{"P&L $": losers["P&L $"].map(lambda x: f"{x:,.2f}")})
            )
    
if __name__ == "__main__":
    main()
