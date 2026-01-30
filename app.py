# app.py
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from alpaca_trade_api.rest import REST


# -----------------------------
# Config
# -----------------------------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Optional shared secret between QuantML and this API
DASHBOARD_API_KEY = os.getenv("DASHBOARD_API_KEY", "")

# 30s cache TTL
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "30"))

# Lock CORS down to your domains
ALLOWED_ORIGINS = [
    "https://app.base44.com",
    "https://quantml.ai",
    "https://app.quantml.ai",
]

# -----------------------------
# Simple TTL cache
# -----------------------------
@dataclass
class TTLCache:
    value: Optional[Dict[str, Any]] = None
    expires_at: float = 0.0

    def get(self) -> Optional[Dict[str, Any]]:
        if self.value is not None and time.time() < self.expires_at:
            return self.value
        return None

    def set(self, v: Dict[str, Any], ttl_s: int) -> None:
        self.value = v
        self.expires_at = time.time() + ttl_s


CACHE = TTLCache()


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="QuantML Dashboard API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


def require_env() -> None:
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY")


def alpaca_client() -> REST:
    require_env()
    return REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)


def auth_guard(x_api_key: Optional[str]) -> None:
    # If DASHBOARD_API_KEY is empty, auth is disabled (not recommended for production).
    if DASHBOARD_API_KEY:
        if not x_api_key or x_api_key != DASHBOARD_API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")


# -----------------------------
# Data builders
# -----------------------------
def build_account_snapshot(api: REST) -> Dict[str, Any]:
    a = api.get_account()
    # float() conversions to make JSON clean
    return {
        "equity": float(a.equity),
        "cash": float(a.cash),
        "buying_power": float(a.buying_power),
        "portfolio_value": float(a.portfolio_value),
        "daytrade_count": int(getattr(a, "daytrade_count", 0) or 0),
        "pattern_day_trader": bool(getattr(a, "pattern_day_trader", False) or False),
        "last_equity": float(getattr(a, "last_equity", a.equity) or a.equity),
    }


def build_positions(api: REST) -> Dict[str, Any]:
    raw = api.list_positions() or []
    rows: List[Dict[str, Any]] = []

    for p in raw:
        qty = float(p.qty)
        entry = float(p.avg_entry_price)
        cur = float(p.current_price)

        # Alpaca fields can differ by account/type; guard with getattr
        market_value = float(getattr(p, "market_value", qty * cur) or qty * cur)

        pl_total_usd = float(getattr(p, "unrealized_pl", 0.0) or 0.0)
        pl_total_pct = float(getattr(p, "unrealized_plpc", 0.0) or 0.0) * 100.0

        pl_today_usd = float(getattr(p, "unrealized_intraday_pl", 0.0) or 0.0)
        pl_today_pct = float(getattr(p, "unrealized_intraday_plpc", 0.0) or 0.0) * 100.0

        side_raw = str(getattr(p, "side", "long")).lower()
        side = "Long" if side_raw == "long" else "Short"

        # "capital deployed" definition:
        # use notional at entry (abs(qty) * entry). This is stable & matches investor-style reporting.
        deployed = abs(qty) * entry

        rows.append({
            "ticker": str(p.symbol),
            "side": side,
            "qty": qty,
            "entry_price": entry,
            "current_price": cur,
            "market_value": market_value,
            "deployed": float(deployed),

            "pl_usd": pl_total_usd,
            "pl_pct": pl_total_pct,
            "pl_today_usd": pl_today_usd,
            "pl_today_pct": pl_today_pct,
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["deployed", "market_value", "pl_today_usd"])

    capital_deployed = float(df["deployed"].sum()) if not df.empty else 0.0
    open_value = float(df["market_value"].sum()) if not df.empty else 0.0

    open_positions_intraday_pl_usd = float(df["pl_today_usd"].sum()) if not df.empty else 0.0

    return {
        "rows": rows,
        "capital_deployed": capital_deployed,
        "open_value": open_value,
        "open_positions_intraday_pl_usd": open_positions_intraday_pl_usd,
    }


def build_portfolio_history(api: REST) -> Dict[str, Any]:
    # Mirrors your Streamlit usage: 1D, 5Min, RTH only
    ph = api.get_portfolio_history(period="1D", timeframe="5Min", extended_hours=False)

    # alpaca_trade_api returns an object with attributes (or dict-like)
    ts = getattr(ph, "timestamp", None) or ph.get("timestamp", [])
    eq = getattr(ph, "equity", None) or ph.get("equity", [])

    if not ts or not eq:
        return {"points": [], "day_pl_pct": None, "day_pl_usd": None}

    df = pd.DataFrame({
        "ts": pd.to_datetime(ts, unit="s", utc=True, errors="coerce"),
        "equity": pd.to_numeric(pd.Series(eq), errors="coerce"),
    }).dropna()

    if df.empty:
        return {"points": [], "day_pl_pct": None, "day_pl_usd": None}

    start_equity = float(df["equity"].iloc[0])
    last_equity = float(df["equity"].iloc[-1])

    day_pl_usd = last_equity - start_equity
    day_pl_pct = (day_pl_usd / start_equity * 100.0) if start_equity != 0 else None

    points = [{"ts": r.ts.isoformat(), "equity": float(r.equity)} for r in df.itertuples(index=False)]

    return {
        "points": points,
        "start_equity": start_equity,
        "last_equity": last_equity,
        "day_pl_usd": float(day_pl_usd),
        "day_pl_pct": float(day_pl_pct) if day_pl_pct is not None else None,
    }


def compute_quantml_return_on_deployed(capital_deployed: float, open_value: float) -> Optional[float]:
    # Return in % terms based on deployed capital (your investor dashboard concept)
    if capital_deployed <= 0:
        return None
    return (open_value - capital_deployed) / capital_deployed * 100.0


def get_spy_prior_close_to_now_return_pct() -> Optional[float]:
    """
    Plug in your existing 'SPY prior close -> now' method here.
    Keep it server-side.

    Return: percent (e.g. -0.22 for -0.22%)
    """
    # TODO: paste your working SPY logic here.
    return None


# -----------------------------
# The single endpoint
# -----------------------------
@app.get("/api/dashboard")
def dashboard(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    auth_guard(x_api_key)

    cached = CACHE.get()
    if cached is not None:
        return {"cached": True, **cached}

    api = alpaca_client()

    snapshot = build_account_snapshot(api)
    positions = build_positions(api)
    history = build_portfolio_history(api)

    capital_deployed = positions["capital_deployed"]
    open_value = positions["open_value"]

    quantml_on_deployed_pct = compute_quantml_return_on_deployed(capital_deployed, open_value)

    spy_pct = get_spy_prior_close_to_now_return_pct()

    payload = {
        "asof_utc": pd.Timestamp.utcnow().isoformat(),
        "market": {
            # You can also compute open/closed here if you want
            "alpaca_base_url": ALPACA_BASE_URL,
        },
        "snapshot": snapshot,
        "positions": {
            "capital_deployed": capital_deployed,
            "open_value": open_value,
            "open_positions_intraday_pl_usd": positions["open_positions_intraday_pl_usd"],
            "rows": positions["rows"],
        },
        "portfolio_history": history,
        "kpis": {
            "portfolio_pl_today_usd": history.get("day_pl_usd"),
            "portfolio_pl_today_pct": history.get("day_pl_pct"),
            "quantml_return_on_deployed_pct": quantml_on_deployed_pct,
            "spy_return_prior_close_to_now_pct": spy_pct,
        },
    }

    CACHE.set(payload, CACHE_TTL_SECONDS)
    return {"cached": False, **payload}
