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
    logo_scale: float = 0.55,   # 0.0–1.0 of the clock size
) -> None:
    """Canvas clock with QUANTML logo. One time sample drives both analog & digital."""
    import base64, io
    from textwrap import dedent

    logo_b64 = ""
    if logo_path:
        try:
            with open(logo_path, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode("ascii")
        except Exception:
            logo_b64 = ""

    canvas_id = f"qmclk_canvas_{st.session_state.get('_clk', 0)}"
    time_id   = f"qmclk_time_{st.session_state.get('_clk', 0)}"
    date_id   = f"qmclk_date_{st.session_state.get('_clk', 0)}"
    st.session_state['_clk'] = st.session_state.get('_clk', 0) + 1

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
      const cssW={size}, cssH={size}, LOGO_SCALE={logo_scale};
      const tz="{tz}", showSeconds={str(show_seconds).lower()}, is24h={str(is_24h).lower()};
      const canvas=document.getElementById("{canvas_id}");
      const dpr=Math.max(1, window.devicePixelRatio||1);
      canvas.width=cssW*dpr; canvas.height=cssH*dpr; canvas.style.width=cssW+"px"; canvas.style.height=cssH+"px";
      const ctx=canvas.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
      const cx=cssW/2, cy=cssH/2, R=Math.min(cx,cy)-6, OFF=-Math.PI/2; // << rotate so 0° is at 12

      // inline logo
      const logoData="{logo_b64}";
      let logoImg=null, logoReady=false;
      if (logoData) {{ logoImg=new Image(); logoImg.onload=()=>logoReady=true; logoImg.src="data:image/png;base64,"+logoData; }}

      // one true time source (sample ONCE per tick for analog+digital)
      function nowInTZ() {{
        // build a Date with the *local clock* of tz
        return new Date(new Date().toLocaleString('en-US', {{ timeZone: tz }}));
      }}
      function pad(n){{return String(n).padStart(2,'0');}}

      function sampleParts(){{
        const d=nowInTZ();
        const h=d.getHours(), m=d.getMinutes(), s=d.getSeconds(), ms=d.getMilliseconds();
        const weekday=d.toLocaleString('en-GB', {{ weekday:'short', timeZone: tz }});
        const mon=d.toLocaleString('en-GB', {{ month:'short', timeZone: tz }});
        const day=pad(d.getDate()), year=d.getFullYear();
        return {{ h, m, s, ms, dateStr:`${{weekday}}, ${{day}} ${{mon}} ${{year}}` }};
      }}

      function drawFace(){{
        ctx.clearRect(0,0,cssW,cssH);
        // ticks, with OFF so 0 is at 12 o'clock
        for(let i=0;i<60;i++) {{
          const a=(Math.PI/30)*i + OFF;
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

        const L=Math.min(cssW,cssH)*LOGO_SCALE;
        if(logoReady){{ ctx.save(); ctx.globalAlpha=0.95; ctx.drawImage(logoImg,cx-L/2,cy-L/2,L,L); ctx.restore(); }}
      }}

      function hand(angle,len,w,col) {{
        ctx.save(); ctx.translate(cx,cy); ctx.rotate(angle+OFF);
        ctx.beginPath(); ctx.moveTo(-R*0.08,0); ctx.lineTo(len,0);
        ctx.lineWidth=w; ctx.lineCap="round"; ctx.strokeStyle=col; ctx.stroke(); ctx.restore();
      }}

      function drawHandsFromParts(p) {{
        const pi=Math.PI;
        const hrA  = (pi/6)  * ((p.h%12) + p.m/60 + p.s/3600);
        const minA = (pi/30) * (p.m + p.s/60);
        const secA = (pi/30) *  p.s;
        hand(hrA,  R*0.50, 5,  "#9DB2FF");
        hand(minA, R*0.72, 3.4,"#9DB2FF");
        if(showSeconds) hand(secA,R*0.78, 2,  "#4F7BFF");
        // center pin
        ctx.beginPath(); ctx.arc(cx,cy,6,0,pi*2); ctx.fillStyle="#99A8FF"; ctx.fill();
        ctx.beginPath(); ctx.arc(cx,cy,3,0,pi*2); ctx.fillStyle="#335CFF"; ctx.fill();
      }}

      function drawDigitalFromParts(p) {{
        const hh = is24h ? pad(p.h) : pad(((p.h%12)||12));
        const ampm = is24h ? "" : (p.h<12 ? " AM" : " PM");
        document.getElementById("{time_id}").textContent =
          {str(show_seconds).lower()} ? `${{hh}}:${{pad(p.m)}}:${{pad(p.s)}}${{ampm}}` : `${{hh}}:${{pad(p.m)}}${{ampm}}`;
        document.getElementById("{date_id}").textContent = p.dateStr;
      }}

      function scheduleNext(ms) {{ setTimeout(tick, 1000 - (ms % 1000)); }}

      function tick() {{
        const p = sampleParts();       // one sample → both analog & digital
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



def render_header(api):
    c1, c2, c3 = st.columns([0.20, 0.65, 0.15], vertical_alignment="center")
    with c1:
        render_quantml_clock(size=200, tz="Europe/Dublin", title="Dublin",
                             show_seconds=True, is_24h=True,
                             logo_path="Clock/QuantML.png")
    with c2:
        st.markdown("## QUANTML — Investor Summary (Live)")
        if api is not None:
            render_market_chip(api)  # your existing chip
            prices = get_open_ticker_prices(api)
            if prices:
                render_ticker_tape(prices)

    # make the whole page refresh every 60 seconds
    enable_autorefresh(60)

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
                day_realized[day] += lot_sign * (px - lot["price"]) * take
                day_liq_notional[day] += px * take
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
    cutoff = (datetime.utcnow().date() - timedelta(days=days))
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
    Renders a compact ledger-style table:
      Columns: Cost to open positions | Open (current value) | Liquidated | P&L $ | P&L %
      Sections: Totals, Current (today), History (optional)
    - 'Liquidated' is interpreted as realized P&L (if you pass it).
    - P&L $ = Unrealized + Realized.
    - P&L % uses cost_to_open as the denominator (typical view for open risk).
    """
    st.subheader("Portfolio Ledger")

    o = _compute_open_ledger(positions)
    realized = float(realized_pnl_total or 0.0)
    total_pl = o["unrl"] + realized
    total_pl_pct = (total_pl / o["cost_open"] * 100.0) if o["cost_open"] > 0 else 0.0

    rows = []

    # Totals row (same as Current unless you pass history/realized from logs)
    rows.append({
        "Section": "Totals",
        "Date": "",
        "Cost to open positions": o["cost_open"],
        "Open (current value)":   o["open_value"],
        "Liquidated":             realized,      # realized P&L (if known)
        "P&L $":                  total_pl,
        "P&L %":                  total_pl_pct,
    })

    # Current (today)
    rows.append({
        "Section": "Current",
        "Date": _fmt_dub(datetime.utcnow()),
        "Cost to open positions": o["cost_open"],
        "Open (current value)":   o["open_value"],
        "Liquidated":             realized,
        "P&L $":                  total_pl,
        "P&L %":                  total_pl_pct,
    })

    # Optional history rows you may pass in (columns: date, cost_open, open_value, realized, pl$, pl%)
    if isinstance(history_rows, pd.DataFrame) and not history_rows.empty:
        for _, r in history_rows.iterrows():
            rows.append({
                "Section": "History",
                "Date": str(r.get("date", "")),
                "Cost to open positions": float(r.get("cost_open", 0)),
                "Open (current value)":   float(r.get("open_value", 0)),
                "Liquidated":             float(r.get("realized", 0)),
                "P&L $":                  float(r.get("pl_dollar", 0)),
                "P&L %":                  float(r.get("pl_percent", 0)),
            })

    df = pd.DataFrame(rows, columns=[
        "Section","Date","Cost to open positions","Open (current value)","Liquidated","P&L $","P&L %"
    ])

    # Style: currency/percent formatting + green/red for P&L columns
    def _style_ledger(s: pd.Series) -> pd.Series:
        sty = pd.Series("", index=s.index, dtype="object")
        try:
            c = float(s.get("P&L $", 0) or 0)
            pct = float(s.get("P&L %", 0) or 0)
        except Exception:
            c, pct = 0.0, 0.0
        color = "#16A34A" if c >= 0 else "#DC2626"
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

    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption("“Liquidated” = realized P&L (if provided). P&L $ = Unrealized + Realized. P&L % is relative to cost to open.")

# =============================================================================
# Main — orchestrate sections (no duplicates)
# =============================================================================
def main() -> None:
    api = _load_alpaca_api()
    render_header(api)

    # In main(), right before the 3 dials:
    positions = pull_live_positions(api)
    positions = merge_tp_sl_from_alpaca_orders(positions, api)
    positions = compute_derived_metrics(positions)

    # NEW: build history from fills (last 5 days by default)
    hist_df, realized_total = build_history_rows_from_fills(api, positions, days=5)

    # Render the ledger with History rows included, and correct sign for shorts
    render_portfolio_ledger_table(positions,
                                realized_pnl_total=realized_total,
                                history_rows=hist_df)
    st.divider()

    # Totals for the 3 dials (unchanged)
    totals = derive_totals_from_positions(positions)
    cap   = float(totals.get("capital_spent", 0.0))
    gross = float(totals.get("gross_exposure", 0.0))
    win   = float(totals.get("win_rate_%", 0.0))

    d1, d2, d3 = st.columns(3)
    with d1:
        st.plotly_chart(gauge_budget_pct(cap, _PORTFOLIO_BUDGET, "Capital Deployed"), width="stretch")
        st.caption("How much of your portfolio budget is currently invested (capital_spent ÷ budget). Bands: 60/80/100%.")
    with d2:
        st.plotly_chart(gauge_exposure_pct(gross, _PORTFOLIO_BUDGET, "Gross Exposure"), width="stretch")
        st.caption("Long + |Short| exposure relative to budget (percentage). Shows how levered the book is regardless of netting.")
    with d3:
        st.plotly_chart(gauge_win_rate(win, "Win Rate"), width="stretch")
        st.caption("Share of open positions that are currently up in P&L.")

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
