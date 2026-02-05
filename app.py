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

from datetime import datetime, timedelta
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, select
from sqlalchemy.orm import declarative_base, sessionmaker


# -----------------------------
# Config
# -----------------------------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
X_API_Key = "quantml-dashboard-2026-secure-key"

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
# Auth / Database
# -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "")
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MIN = int(os.getenv("JWT_EXPIRES_MIN", "720"))

if not DATABASE_URL or not JWT_SECRET:
    raise RuntimeError("DATABASE_URL and JWT_SECRET must be set")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(320), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(pw: str) -> str:
    return pwd_context.hash(pw)

def verify_password(pw: str, pw_hash: str) -> bool:
    return pwd_context.verify(pw, pw_hash)

def create_access_token(email: str) -> str:
    payload = {
        "sub": email,
        "exp": datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MIN)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

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

from pydantic import BaseModel, EmailStr

class RegisterIn(BaseModel):
    email: EmailStr
    password: str

@app.post("/auth/register")
def register_user(data: RegisterIn, db=Depends(get_db)):
    email = data.email.lower().strip()
    if len(data.password) < 10:
        raise HTTPException(status_code=400, detail="Password too short")

    exists = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if exists:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(email=email, password_hash=hash_password(data.password))
    db.add(user)
    db.commit()

    token = create_access_token(email)
    return {"access_token": token, "token_type": "bearer"}


@app.post("/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db=Depends(get_db)):
    email = form.username.lower().strip()
    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()

    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(email)
    return {"access_token": token, "token_type": "bearer"}


@app.get("/auth/me")
def me(user_email: str = Depends(get_current_user)):
    return {"email": user_email}


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

@app.get("/")
def root():
    return {"ok": True, "service": "QuantML Dashboard API"}

@app.get("/health")
def health():
    return {"ok": True}


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

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["deployed", "market_value", "pl_today_usd", "pl_usd"])

    capital_deployed = float(df["deployed"].sum()) if not df.empty else 0.0

    # Keep net open_value if you want (net exposure). Optional.
    open_value_net = float(df["market_value"].sum()) if not df.empty else 0.0

    # NEW: gross open value is more intuitive for "open value" display
    open_value_gross = float(df["market_value"].abs().sum()) if not df.empty else 0.0

    open_positions_intraday_pl_usd = float(df["pl_today_usd"].sum()) if not df.empty else 0.0

    # NEW: total unrealized P&L across positions
    open_positions_unrealized_pl_usd = float(df["pl_usd"].sum()) if not df.empty else 0.0

    return {
        "rows": rows,
        "capital_deployed": capital_deployed,
        "open_value_net": open_value_net,
        "open_value_gross": open_value_gross,
        "open_positions_intraday_pl_usd": open_positions_intraday_pl_usd,
        "open_positions_unrealized_pl_usd": open_positions_unrealized_pl_usd,
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

def compute_quantml_return_on_deployed(capital_deployed: float, unrealized_pl_usd: float) -> Optional[float]:
    if capital_deployed <= 0:
        return None
    return (unrealized_pl_usd / capital_deployed) * 100.0


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
def dashboard(user_email: str = Depends(get_current_user)):

    cached = CACHE.get()
    if cached is not None:
        return {"cached": True, **cached}

    api = alpaca_client()

    snapshot = build_account_snapshot(api)
    positions = build_positions(api)
    history = build_portfolio_history(api)

    capital_deployed = positions["capital_deployed"]
    unrealized_pl_usd = positions["open_positions_unrealized_pl_usd"]

    quantml_on_deployed_pct = compute_quantml_return_on_deployed(capital_deployed, unrealized_pl_usd)
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
            "open_value_net": positions["open_value_net"],
            "open_value_gross": positions["open_value_gross"],
            "open_positions_intraday_pl_usd": positions["open_positions_intraday_pl_usd"],
            "open_positions_unrealized_pl_usd": positions["open_positions_unrealized_pl_usd"],
            "rows": positions["rows"],
        },        "portfolio_history": history,
        "kpis": {
            "portfolio_pl_today_usd": history.get("day_pl_usd"),
            "portfolio_pl_today_pct": history.get("day_pl_pct"),
            "spy_return_prior_close_to_now_pct": spy_pct,
            "quantml_return_on_deployed_pct": quantml_on_deployed_pct,
        },
    }

    CACHE.set(payload, CACHE_TTL_SECONDS)
    return {"cached": False, **payload}
