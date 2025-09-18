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

def apply_branding() -> None:
    px.defaults.template = "plotly_white"
    px.defaults.color_discrete_sequence = PLOTLY_SEQ

    st.markdown("""
    <style>
      .block-container{
        padding-top:2.4rem;          /* headroom under Streamlit's toolbar */
        padding-bottom:2rem;
        max-width:1750px;
      }
      @media (min-width: 1900px){ .block-container{ max-width:2000px; } }

      /* Left‑align st.dataframe headers + cells */
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

# ---------- HEADER (clock + market chip) ----------
from pathlib import Path
import base64
import uuid
from textwrap import dedent
import streamlit.components.v1 as components

def render_quantml_clock(*,
                         size: int = 200,
                         tz: str = "Europe/Dublin",
                         title: str = "Dublin",
                         show_seconds: bool = True,
                         is_24h: bool = True,
                         logo_path: str | None = "Clock/quantml_logo_clock.png") -> None:
    render_banner_clock(size=size, tz=tz, title=title,
                        show_seconds=show_seconds, is_24h=is_24h,
                        logo_path=logo_path)

def render_banner_clock(*,
                        size=200, tz="Europe/Dublin", title="Dublin",
                        show_seconds=True, is_24h=True, logo_path=None) -> None:
    """
    Analog + digital clock that ALWAYS renders in the requested timezone.
    - Uses Intl.DateTimeFormat(..., timeZone=tz).formatToParts() for accurate TZ math.
    - Optional center logo: put a square PNG at Clock/quantml_logo_clock.png in the repo.
      (Falls back to a subtle QUANTML wordmark if the file is not present.)
    """
    # Inline the logo (if present) as a base64 data URI so it works on Streamlit Cloud
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

    html = dedent(f"""
    <div style="display:flex;flex-direction:column;align-items:center;">
      <div style="background:#0b1220;border-radius:18px;padding:12px 16px 18px 16px;box-shadow:0 10px 30px rgba(0,0,0,.35);">
        <canvas id="{canvas_id}" style="display:block;width:{size}px;height:{size}px;"></canvas>
        <div style="margin-top:10px;text-align:center;font-family:ui-sans-serif;color:#E5E7EB;">
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
      const canvas = document.getElementById("{canvas_id}");
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      canvas.width = cssW*dpr; canvas.height = cssH*dpr;
      canvas.style.width = cssW+"px"; canvas.style.height = cssH+"px";
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr,0,0,dpr,0,0);
      const cx = cssW/2, cy = cssH/2, R = Math.min(cx,cy)-6;

      // Load the (optional) logo once
      const logoData = "{logo_b64}";
      let logoImg = null, logoReady = false;
      if (logoData) {{
        logoImg = new Image();
        logoImg.onload = function(){{ logoReady = true; }};
        logoImg.src = "data:image/png;base64," + logoData;
      }}

      function getParts(){{
        // Robust TZ parts (works across browsers)
        try {{
          const fmt = new Intl.DateTimeFormat('en-GB', {{
            timeZone: tz, hour:'2-digit', minute:'2-digit', second:'2-digit',
            weekday:'short', day:'2-digit', month:'short', year:'numeric',
            hour12: !is24h
          }});
          const arr = fmt.formatToParts(new Date());
          const m = {{}};
          for (const p of arr) m[p.type] = p.value;
          return {{
            h: parseInt(m.hour)||0,
            m: parseInt(m.minute)||0,
            s: parseInt(m.second)||0,
            dateStr: `${{m.weekday||""}}, ${{m.day||""}} ${{m.month||""}} ${{m.year||""}}`
          }};
        }} catch(e) {{
          // Fallback: coerce Date into tz via toLocaleString
          const d = new Date(new Date().toLocaleString('en-GB', {{ timeZone: tz }}));
          return {{ h:d.getHours(), m:d.getMinutes(), s:d.getSeconds(), dateStr:d.toDateString() }};
        }}
      }}

      function drawFace(){{
        ctx.clearRect(0,0,cssW,cssH);
        // tick marks
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
        // outer ring
        ctx.beginPath(); ctx.arc(cx,cy,R*0.99,0,Math.PI*2);
        ctx.strokeStyle="rgba(79,70,229,0.6)"; ctx.lineWidth=2; ctx.stroke();

        // center logo (if available), else subtle wordmark
        const L = Math.min(cssW, cssH) * 0.28;
        if (logoReady) {{
          ctx.save(); ctx.globalAlpha = 0.92;
          ctx.drawImage(logoImg, cx - L/2, cy - L/2, L, L);
          ctx.restore();
        }} else {{
          ctx.save();
          ctx.fillStyle = "rgba(226,232,255,0.18)";
          ctx.font = "bold 14px system-ui, -apple-system, Segoe UI, Roboto";
          ctx.textAlign="center"; ctx.textBaseline="middle";
          ctx.fillText("QUANTML", cx, cy);
          ctx.restore();
        }}
      }}

      function hand(angle,len,width,color){{
        ctx.save(); ctx.translate(cx,cy); ctx.rotate(angle);
        ctx.beginPath(); ctx.moveTo(-R*0.08,0); ctx.lineTo(len,0);
        ctx.lineWidth=width; ctx.lineCap="round"; ctx.strokeStyle=color; ctx.stroke();
        ctx.restore();
      }}

      function drawHands(h,m,s){{
        const pi=Math.PI;
        const hrA=(pi/6)*((h%12)+m/60+s/3600);
        const minA=(pi/30)*(m+s/60);
        const secA=(pi/30)*s;
        hand(hrA, R*0.50, 5,  "#9DB2FF");
        hand(minA, R*0.72, 3.4,"#9DB2FF");
        if (showSeconds) hand(secA, R*0.78, 2, "#4F7BFF");
        ctx.beginPath(); ctx.arc(cx,cy,6,0,pi*2); ctx.fillStyle="#99A8FF"; ctx.fill();
        ctx.beginPath(); ctx.arc(cx,cy,3,0,pi*2); ctx.fillStyle="#335CFF"; ctx.fill();
      }}

      function tick(){{
        const p = getParts();
        drawFace(); drawHands(p.h,p.m,p.s);
        document.getElementById("{time_id}").textContent =
          new Intl.DateTimeFormat('en-GB', {{
            timeZone: tz, hour:'2-digit', minute:'2-digit', second: showSeconds? '2-digit': undefined, hour12: !is24h
          }}).format(new Date());
        document.getElementById("{date_id}").textContent = p.dateStr;
      }}

      tick(); setInterval(tick, 1000);
    }})();
    </script>
    """)

    components.html(html, height=h, scrolling=False)

def render_header(api: Optional[REST]) -> None:
    c1, c2, c3 = st.columns([0.20, 0.65, 0.15], vertical_alignment="center")
    with c1:
        # Dublin clock with fixed timezone & optional logo
        render_quantml_clock(size=200, tz="Europe/Dublin", title="Dublin",
                             show_seconds=True, is_24h=True,
                             logo_path="Clock/quantml_logo_clock.png")
    with c2:
        st.markdown("## QUANTML — Investor Summary (Live)")
        if api is not None:
            render_market_chip(api)  # NYSE OPEN/CLOSED chip from Alpaca
    with c3:
        st.markdown("<div style='text-align:right'>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", help="Pull latest live data from Alpaca", key="refresh_live"):
            st.cache_data.clear()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


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
    """Color Current Price, P/L $ and P/L % by row P&L sign."""
    color = BRAND["success"] if float(row.get("Total P/L ($)", 0) or 0) >= 0 else BRAND["danger"]
    s = pd.Series("", index=row.index, dtype="object")
    for c in ("Current Price", "Total P/L ($)", "Total P/L (%)"):
        if c in row.index:
            s[c] = f"color: {color}; font-weight: 700;"
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
                        use_container_width=True)
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
    """Totals used by KPIs / dials when a broker snapshot isn't present."""
    if df is None or df.empty:
        return {}
    z = compute_derived_metrics(df).copy()

    qty_abs  = pd.to_numeric(z.get("qty"), errors="coerce").abs()
    entry_px = pd.to_numeric(z.get("entry_price"), errors="coerce")

    notional = pd.to_numeric(z.get("notional"), errors="coerce")
    if notional.isna().any():
        notional = qty_abs * entry_px

    # Capital and unrealized
    capital_spent = float(np.nansum(np.abs(notional)))
    upnl = float(pd.to_numeric(z.get("pl_$"), errors="coerce").sum(skipna=True))
    pl_pct_sum = float(pd.to_numeric(z.get("pl_%"), errors="coerce").sum(skipna=True))

    # Exposure
    qty_signed = pd.to_numeric(z.get("qty"), errors="coerce")
    exp_signed = np.where(qty_signed.notna(), entry_px * qty_signed, z["dir_sign"] * qty_abs * entry_px)
    net_exp    = float(np.nansum(exp_signed))
    gross_exp  = float(np.nansum(np.abs(exp_signed)))

    # Win rate (rows with P&L > 0)
    wins = int((pd.to_numeric(z.get("pl_$"), errors="coerce") > 0).sum())
    npos = int(len(z))
    win_rate = float(100.0 * wins / max(1, npos))

    return {
        "positions": len(z),
        "capital_spent": round(capital_spent, 2),
        "unrealized_pl_$": round(upnl, 2),
        # keep the weighted metric (now correct because capital is positive)
        "unrealized_pl_%_weighted": (round(upnl / capital_spent * 100.0, 3) if capital_spent > 0 else 0.0),
        # NEW: sum of all row P/L %
        "unrealized_pl_%_sum": round(pl_pct_sum, 2),
        "gross_exposure": round(gross_exp, 2),
        "net_exposure": round(net_exp, 2),
        "wins": wins,
        "win_rate_%": round(win_rate, 2),
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
_DIAL_H = 210
_NEEDLE = "#374151"  # gray‑700
_BAND_GREEN = "rgba(22,163,74,0.22)"
_BAND_AMBER = "rgba(234,179,8,0.26)"
_BAND_RED   = "rgba(220,38,38,0.24)"

def _banded_gauge(percent: float, title: str, bands=(60, 80, 100), good="low") -> go.Figure:
    val = float(max(0.0, percent))
    # extend axis to the next 20% band over the current value (cap at, say, 200%)
    axis_max = min(200.0, max(100.0, (int(val/20.0)+1) * 20.0))
    a, b, c = bands
    steps = ([{"range":[0,a], "color":_BAND_GREEN},
              {"range":[a,b], "color":_BAND_AMBER},
              {"range":[b,100], "color":_BAND_RED}]
             if good=="low" else
             [{"range":[0,a], "color":_BAND_RED},
              {"range":[a,b], "color":_BAND_AMBER},
              {"range":[b,100], "color":_BAND_GREEN}])
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=min(val, axis_max),
        title={"text": title, "font": {"size": 14}},
        number={"suffix": "%", "font": {"size": 30}},
        gauge={"axis": {"range": [0, axis_max], "tickwidth": 1, "ticklen": 4},
               "bar": {"color": _NEEDLE}, "bgcolor": "white", "borderwidth": 0,
               "steps": steps}
    ))
    fig.update_layout(height=_DIAL_H, margin=dict(l=6, r=6, t=24, b=0))
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

# ---------- TOP KPI ROW ----------
def render_top_kpis(totals: dict, budget: float, acct_equity: float | None = None) -> None:
    """
    5 cards: Capital Deployed, Current Portfolio Value, Total P/L %, Total P/L $, Win Rate
    Uses broker equity if available so the 'Current Portfolio Value' matches Alpaca.
    """
    cap  = float(totals.get("capital_spent", 0.0))
    upnl = float(totals.get("unrealized_pl_$", 0.0))
    wr   = float(totals.get("win_rate_%", totals.get("win_rate", 0.0)))
    cur_val = float(acct_equity) if acct_equity is not None else (cap + upnl)
    pl_pct_from_kpis = (upnl / cap * 100.0) if cap > 0 else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: _kpi_card("Total Capital Deployed", money(cap))
    with c2:
        tone = "pos" if cur_val >= cap else "neg"
        _kpi_card("Current Portfolio Value", money(cur_val), tone, caption=f"Budget: {money(budget)}")
    with c3:
        tone = "pos" if pl_pct_from_kpis >= 0 else "neg"
        _kpi_card("Total P/L %", f"{pl_pct_from_kpis:+.2f}%", tone)
    with c4:
        tone = "pos" if upnl >= 0 else "neg"
        _kpi_card("Total P/L $", money(upnl), tone)
    with c5:
        _kpi_card("Win Rate", f"{wr:.0f}%", caption="Open positions up ÷ total")


def render_traffic_lights(df: pd.DataFrame) -> None:
    st.subheader("Traffic Lights (per Open position)")
    if df is None or df.empty:
        st.info("—"); return
    z = compute_derived_metrics(df)  # ensure pl_%/$ present
    sort_col = "Ticker" if "Ticker" in z.columns else ("symbol" if "symbol" in z.columns else None)
    if sort_col: z = z.sort_values(sort_col)

    CHIP_CSS = ("display:inline-flex;align-items:center;gap:8px;"
                "padding:6px 10px;border-radius:16px;"
                "background:rgba(14,27,58,0.08);border:1px solid rgba(14,27,58,0.26);"
                "margin:3px 4px;")
    chips = []
    for _, r in z.iterrows():
        col = BRAND["success"] if float(r.get("pl_$", 0) or 0) >= 0 else BRAND["danger"]
        # precedence with R/pct if you prefer:
        col = (BRAND["success"] if pd.to_numeric(pd.Series([r.get("R")]), errors="coerce").iloc[0] >= 1.0
               else BRAND["warning"] if pd.to_numeric(pd.Series([r.get("pl_%")]), errors="coerce").iloc[0] >= -0.5
               else col)
        label = f"{r.get('Ticker', r.get('symbol','?'))} · {r.get('Side','—')} · {pct(r.get('pl_%'))} / {money_signed(r.get('pl_$'))}"
        dot = f"<span style='width:12px;height:12px;border-radius:50%;display:inline-block;border:2px solid {col};background:{col};'></span>"
        chips.append(f"<span style='{CHIP_CSS}'>{dot}<span style='font-weight:600;font-size:0.9rem;color:#0E1B3A'>{label}</span></span>")

    st.markdown("<div style='display:flex;flex-wrap:wrap;gap:8px'>" + "".join(chips) + "</div>", unsafe_allow_html=True)

def render_positions_table(df: pd.DataFrame) -> None:
    st.subheader("Live Positions (Alpaca)")
    if df is None or df.empty:
        st.info("—"); return

    z = compute_derived_metrics(df).copy()
    z["Market Value"] = pd.to_numeric(z["current_price"], errors="coerce") * pd.to_numeric(z["qty"], errors="coerce")

    def _status_row(r):
        up = (r.get("pl_$", 0.0) or 0.0) >= 0
        dot = "🟢" if up else "🔴"
        return f"{dot} {'UP' if up else 'DOWN'} {r.get('pl_%', np.nan):+.2f}% / {money_signed(r.get('pl_$', np.nan))}"

    show = pd.DataFrame({
        "Asset":         z.get("Ticker", z.get("symbol")),
        "Status":        z.apply(_status_row, axis=1),
        "Qty":           pd.to_numeric(z["qty"], errors="coerce"),
        "Side":          z["Side"],
        "Avg Entry":     pd.to_numeric(z["entry_price"], errors="coerce"),
        "Current Price": pd.to_numeric(z["current_price"], errors="coerce"),
        "TP":            pd.to_numeric(z.get("TP", z.get("tp_price")), errors="coerce"),
        "SL":            pd.to_numeric(z.get("SL", z.get("sl_price")), errors="coerce"),
        "Market Value":  pd.to_numeric(z["Market Value"], errors="coerce"),
        # IMPORTANT: keep pl_% as actual percent (already 100*x in compute_derived_metrics)
        "Total P/L ($)": pd.to_numeric(z["pl_$"], errors="coerce"),
        "Total P/L (%)": pd.to_numeric(z["pl_%"], errors="coerce"),
    })

    order = ["Asset","Status","Qty","Side","Avg Entry","Current Price","TP","SL","Market Value","Total P/L ($)","Total P/L (%)"]
    show = show[[c for c in order if c in show.columns]]

    styled = (
        show.style
            .format({
                "Qty":            "{:.0f}",
                "Avg Entry":      "{:.2f}",
                "Current Price":  "{:.2f}",
                "TP":             "{:.2f}",
                "SL":             "{:.2f}",
                "Market Value":   "${:,.2f}",
                "Total P/L ($)":  "${:,.2f}",
                "Total P/L (%)":  "{:+.2f}%",
            }, na_rep="—")
            .apply(_style_row_by_pl, axis=1)
    )

    st.dataframe(styled, width="stretch", hide_index=True)


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

def pull_live_positions(api: Optional[REST]) -> pd.DataFrame:
    if api is None:
        return pd.DataFrame()
    try:
        raw = api.list_positions()
    except Exception as e:
        st.error(f"Could not fetch positions: {e}")
        return pd.DataFrame()

    rows = []
    for p in raw or []:
        rows.append({
            "Ticker":        p.symbol,
            "Side":          ("Long" if (getattr(p, "side", "long").lower() == "long") else "Short"),
            "qty":           float(p.qty),
            "entry_price":   float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "notional":      float(p.market_value),
            "pl_$":          float(getattr(p, "unrealized_pl", 0.0) or 0.0),
            "pl_%":          (float(getattr(p, "unrealized_plpc", 0.0) or 0.0) * 100.0
                              if float(p.avg_entry_price or 0) != 0 else 0.0),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Main — orchestrate sections (no duplicates)
# =============================================================================
def main() -> None:
    api = _load_alpaca_api()
    render_header(api)

    # New top section: broker snapshot grid
    acct = pull_account_snapshot(api)
    render_broker_balances(acct)
    st.divider()

    # Live positions → TP/SL → totals (unchanged)
    positions = pull_live_positions(api)
    positions = merge_tp_sl_from_alpaca_orders(positions, api)
    positions = compute_derived_metrics(positions)
    totals    = derive_totals_from_positions(positions)

    # 5 KPIs row (right above your 3 dials)
    acct_equity = acct.get("equity") if acct else None
    render_top_kpis(totals, _PORTFOLIO_BUDGET, acct_equity=acct_equity)
    st.divider()

    # ↓↓↓ KEEP your existing 3 dials and everything below from here ↓↓↓
    # (No changes required to your dials, traffic lights, table, etc.)


    st.divider()

    # Traffic Lights (chips) + Live Positions table (TP/SL shown after Current Price)
    render_traffic_lights(positions)
    st.divider()
    render_positions_table(positions)

    # Optional: recent closed trades (comment out if not needed)
    # render_closed_trades_history(api, days=7)

    # Footer
    st.caption(f"As of: {datetime.now().strftime('%Y-%m-%d %H:%M')} • Budget: {money(_PORTFOLIO_BUDGET)}")

if __name__ == "__main__":
    main()
