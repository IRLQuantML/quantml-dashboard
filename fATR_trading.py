"""
Standalone TP extender for Alpaca positions.
- Imports your QuantML config.py for API keys and defaults.
- Extends (raises/lowers) existing TP legs when the trade moves in your favor.
- Can run once, or loop every N seconds for continuous monitoring.

Run examples:
  python quantml_tp_extender.py --once
  python quantml_tp_extender.py --interval 120 --max-min 360
  python quantml_tp_extender.py --mode percent --trigger-pct 0.03 --step-pct 0.01
  python quantml_tp_extender.py --mode atr --trigger-atr 1.0 --step-atr 0.5
"""
# --- Standard library ---
import os
import sys
import time
import math
import logging
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
from typing import Optional, Tuple, Dict
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
# --- Third-party ---
import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame

# Yahoo fallback (optional)
try:
    import yfinance as yf
except Exception:
    yf = None
    
# --- Read QuantML config with graceful fallbacks ---
try:
    import config as CFG
except Exception as e:
    print("❌ Cannot import config.py — place this file in your QuantML repo root.", file=sys.stderr)
    raise

predict_path = getattr(CFG, "predict_path", "6.Predictions")
TICKER_MAP = getattr(CFG, "TICKER_MAP", {})
_failsafe_hits: dict[str, int] = {}

MIN_SL_GAP_PCT = 0.002   # 0.2% min distance from price
FAILSAFE_ALERT_AFTER = 3
_REDUCE_ONLY_WARNED = False

# API / broker
ALPACA_API_KEY   = getattr(CFG, "ALPACA_API_KEY", "")
ALPACA_SECRET_KEY= getattr(CFG, "ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL  = getattr(CFG, "ALPACA_LIVE_URL", "") if getattr(CFG, "USE_LIVE_TRADING", False) \
                   else getattr(CFG, "ALPACA_PAPER_URL", "")
USE_LIVE_TRADING = bool(getattr(CFG, "USE_LIVE_TRADING", False))
ALLOW_TP_CREATE_IF_MISSING = bool(getattr(CFG, "ALLOW_TP_CREATE_IF_MISSING", True))

# === ATR source preferences ===
# Rebalance exits when SL is missing and TP consumes 100% of qty
SL_REBALANCE_IF_MISSING  = bool(getattr(CFG, "SL_REBALANCE_IF_MISSING", True))
SL_MIN_SHARES            = int(getattr(CFG, "SL_MIN_SHARES", 1))

# --- Additional trading/runtime knobs from config ---
ALLOW_FRACTIONAL            = bool(getattr(CFG, "ALLOW_FRACTIONAL", False))
CANCEL_ORDER_TIMEOUT_S      = float(getattr(CFG, "CANCEL_ORDER_TIMEOUT_S", 2.0))
RECHECK_AVAIL_MAX_WAIT_S    = float(getattr(CFG, "RECHECK_AVAIL_MAX_WAIT_S", 1.0))
RECHECK_AVAIL_POLL_S        = float(getattr(CFG, "RECHECK_AVAIL_POLL_S", 0.1))

# Order of sources to try for ATR resolution
ATR_SOURCE_ORDER = list(getattr(CFG, "ATR_SOURCE_ORDER", ["fresh", "folder", "yahoo"]))
ATR_LENGTH = int(getattr(CFG, "ATR_LENGTH", 14))
# --- ATR floors (optional) ---
ATR_DOLLAR_FLOOR         = float(getattr(CFG, "ATR_DOLLAR_FLOOR", 0.25))  # treat ATR as at least this $ value
ATR_STEP_MIN_DOLLARS     = float(getattr(CFG, "ATR_STEP_MIN_DOLLARS", 0.03))  # min $ step per ratchet


# Yahoo fallback controls
YAHOO_FALLBACK_ENABLE     = bool(getattr(CFG, "YAHOO_FALLBACK_ENABLE", True))
YAHOO_LOOKBACK_DAYS       = int(getattr(CFG, "YAHOO_LOOKBACK_DAYS", 120))
YAHOO_INTERVAL            = str(getattr(CFG, "YAHOO_INTERVAL", "1d"))

# Price formatting
PRICE_DECIMALS   = int(getattr(CFG, "PRICE_DECIMALS", 2))

# Ticker remap
TICKER_MAP       = getattr(CFG, "TICKER_MAP", {})
REVERSE_TICKER_MAP = {v: k for k, v in TICKER_MAP.items()}

# Dry-run (note: extender needs REAL open orders to adjust)
DRY_RUN          = bool(getattr(CFG, "dry_run", False))

# === Extender & monitor settings from config (with sane defaults) ===
TP_EXTEND_ENABLE      = bool(getattr(CFG, "TP_EXTEND_ENABLE", True))
TP_EXTEND_MODE        = str(getattr(CFG, "TP_EXTEND_MODE", "percent")).lower()

TP_EXTEND_TRIGGER_PCT = float(getattr(CFG, "TP_EXTEND_TRIGGER_PCT", 0.03))
TP_EXTEND_STEP_PCT    = float(getattr(CFG, "TP_EXTEND_STEP_PCT", 0.01))

TP_EXTEND_TRIGGER_ATR = float(getattr(CFG, "TP_EXTEND_TRIGGER_ATR", 1.0))
TP_EXTEND_STEP_ATR    = float(getattr(CFG, "TP_EXTEND_STEP_ATR", 0.50))
TP_EXTEND_MAX_MULT    = float(getattr(CFG, "TP_EXTEND_MAX_MULT", 2.5))

MONITOR_ENABLE        = bool(getattr(CFG, "MONITOR_ENABLE", True))
MONITOR_INTERVAL_SEC  = int(getattr(CFG, "MONITOR_INTERVAL_SEC", 120))
MONITOR_MAX_MINUTES   = getattr(CFG, "MONITOR_MAX_MINUTES", 360)  # allow None/0
MONITOR_STOP_AT_CLOSE = bool(getattr(CFG, "MONITOR_STOP_AT_CLOSE", True))

REPLACE_ORDER_FIRST   = bool(getattr(CFG, "REPLACE_ORDER_FIRST", True))
LOG_TICKERS_SAMPLES   = int(getattr(CFG, "LOG_TICKERS_SAMPLES", 10))

log_dir = Path(getattr(CFG, "trading_path", "7.Trading"))
log_dir.mkdir(parents=True, exist_ok=True)
ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = log_dir / f"tp_extender_ATR_{ts}.log"

LOG_LEVEL_NAME = str(getattr(CFG, "TP_EXTENDER_LOG_LEVEL", getattr(CFG, "QML_LOG_LEVEL", "INFO"))).upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.INFO)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")]
)
logging.info("Writing logs to %s", log_file)

# === TP extender guardrails (config-overridable) ===
MIN_ABS_STEP = float(getattr(CFG, "TP_EXTEND_MIN_ABS_STEP", 0.05))     # $0.05
MIN_REL_STEP = float(getattr(CFG, "TP_EXTEND_MIN_REL_STEP", 0.001))    # 0.10%
SPREAD_BUFFER_TICKS = int(getattr(CFG, "TP_EXTEND_SPREAD_BUFFER_TICKS", 2))
COOLDOWN_MIN = int(getattr(CFG, "TP_EXTEND_COOLDOWN_MIN", 5))          # 5 minutes

# --- Healthcheck knobs ---
HEALTHCHECK_AT_OPEN        = bool(getattr(CFG, "HEALTHCHECK_AT_OPEN", True))
HEALTHCHECK_AT_CLOSE       = bool(getattr(CFG, "HEALTHCHECK_AT_CLOSE", True))
HEALTHCHECK_OPEN_GRACE_MIN = int(getattr(CFG, "HEALTHCHECK_OPEN_GRACE_MIN", 3))   # run within first N minutes
HEALTHCHECK_CLOSE_GRACE_MIN= int(getattr(CFG, "HEALTHCHECK_CLOSE_GRACE_MIN", 3))  # run within last  N minutes
HEALTHCHECK_BASE_SL_FRAC   = float(getattr(CFG, "HEALTHCHECK_BASE_SL_FRACTION", 0.60))  # when seeding both legs

# per-symbol last-extend timestamps
_last_extend_at: dict[str, float] = {}

# === Apply TP_EXTEND_PROFILE overrides LAST (so they don't get overwritten) ===
try:
    profile_name = str(getattr(CFG, "TP_EXTEND_PROFILE", "")).lower()
    profiles = getattr(CFG, "TP_EXTEND_PROFILES", {})
    prof = profiles.get(profile_name)
    if prof:
        logging.info("⚙️ Applying TP_EXTEND_PROFILE='%s' overrides", profile_name)
        _cast = {
            # --- ratchet / guardrails ---
            "TP_EXTEND_MODE": str,            # we still SKIP applying this key below; mode is chosen by pick_mode
            "TP_EXTEND_TRIGGER_PCT": float,
            "TP_EXTEND_STEP_PCT": float,
            "TP_EXTEND_TRIGGER_ATR": float,
            "TP_EXTEND_STEP_ATR": float,
            "TP_EXTEND_MAX_MULT": float,
            "TP_EXTEND_COOLDOWN_MIN": int,
            "TP_EXTEND_MIN_ABS_STEP": float,
            "TP_EXTEND_MIN_REL_STEP": float,
            "TP_EXTEND_SPREAD_BUFFER_TICKS": int,
            "TP_EXTEND_MAX_REPLACES_PER_SYMBOL": (lambda v: None if v in (None, "None", "", 0) else int(v)),

            # --- baseline & protection ---
            "EXTENDER_BASE_SL_ENABLE": bool,
            "EXTENDER_BASE_SL_ATR_MULT": float,
            "EXTENDER_BASE_SL_PCT": (lambda v: None if v in (None, "None", "", 0) else float(v)),
            "BASELINE_SL_FRACTION": float,
            "ATR_BE_LOCK": float,
            "MIN_SL_GAP_PCT": float,
            "SL_HARD_FAILSAFE": bool,
            "SL_REBALANCE_IF_MISSING": bool,
            "SL_MIN_SHARES": int,
            "SL_REBALANCE_QTY_FRACTION": float,
            "ATR_TRIGGER_BUFFER_MULT": float,
            "ATR_DOLLAR_FLOOR": float,
            "ATR_STEP_MIN_DOLLARS": float,

            # --- monitor cadence / behavior ---
            "MONITOR_INTERVAL_SEC": int,
            "MONITOR_MAX_MINUTES": (lambda v: None if v in (None, "None", "", 0) else int(v)),
            "MONITOR_STOP_AT_CLOSE": bool,
            "REPLACE_ORDER_FIRST": bool,
            "SKIP_GHOST_POSITIONS": bool,
            "LOG_TICKERS_SAMPLES": int,
            "TP_EXTENDER_LOG_LEVEL": str,     # note: logging.basicConfig already ran; keep for future sessions

            # --- ATR source preferences ---
            "ATR_SOURCE_ORDER": list,
            "ATR_LENGTH": int,
            "YAHOO_FALLBACK_ENABLE": bool,
            "YAHOO_LOOKBACK_DAYS": int,
            "YAHOO_INTERVAL": str,

            # --- healthcheck knobs (optional per-profile) ---
            "HEALTHCHECK_AT_OPEN": bool,
            "HEALTHCHECK_AT_CLOSE": bool,
            "HEALTHCHECK_OPEN_GRACE_MIN": int,
            "HEALTHCHECK_CLOSE_GRACE_MIN": int,
            "HEALTHCHECK_BASE_SL_FRACTION": float,
        }
        for k, v in prof.items():
            if k == "TP_EXTEND_MODE":
                continue
            globals()[k] = _cast.get(k, lambda x: x)(v)
        # --- Ensure logger reflects effective profile/config level (do this AFTER overrides) ---
        try:
            _lvl_name = str(
                prof.get("TP_EXTENDER_LOG_LEVEL",
                         getattr(CFG, "TP_EXTENDER_LOG_LEVEL",
                                 getattr(CFG, "QML_LOG_LEVEL", "INFO")))
            ).upper()
            root = logging.getLogger()
            if not root.handlers:
                logging.basicConfig(
                    level=getattr(logging, _lvl_name, logging.INFO),
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")]
                )
            root.setLevel(getattr(logging, _lvl_name, logging.INFO))
        except Exception:
            logging.getLogger().setLevel(logging.INFO)

except Exception as e:
    logging.warning("Profile load error: %s", e)

# Max TP replaces per symbol per session (None/0 = unlimited)
TP_EXTEND_MAX_REPLACES_PER_SYMBOL = globals().get("TP_EXTEND_MAX_REPLACES_PER_SYMBOL",
                                                  getattr(CFG, "TP_EXTEND_MAX_REPLACES_PER_SYMBOL", None))
try:
    TP_EXTEND_MAX_REPLACES_PER_SYMBOL = (int(TP_EXTEND_MAX_REPLACES_PER_SYMBOL)
                                         if TP_EXTEND_MAX_REPLACES_PER_SYMBOL not in (None, "None", "") else None)
except Exception:
    TP_EXTEND_MAX_REPLACES_PER_SYMBOL = None

# per-symbol replace counters for this process/session
_replace_counts: dict[str, int] = {}

from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING

def _tick_for_price(px: float) -> Decimal:
    return Decimal("0.01") if px >= 1 else Decimal("0.0001")

def _refresh_exits_from_baseline(api, pos, live_px: float, atr_val: float | None):
    """
    Replace BOTH exits to a fresh baseline around the current price.
    - Baseline uses EXTENDER_BASE_SL_ATR_MULT × ATR if available, else 1% of price.
    - Keeps correct closing side (sell for longs, buy for shorts).
    - Uses replace-then-cancel patterns already present in helpers.
    """
    direction, side_exit = _position_side_and_exit(pos)  # 'LONG'/'SHORT', 'sell'/'buy'
    # offset: prefer ATR baseline
    eff_atr = None
    try:
        if atr_val and float(atr_val) > 0:
            eff_atr = max(float(atr_val), float(globals().get("ATR_DOLLAR_FLOOR", 0.0)))
    except Exception:
        eff_atr = None
    if eff_atr:
        offset = float(globals().get("EXTENDER_BASE_SL_ATR_MULT", 0.35)) * eff_atr
    else:
        offset = 0.01 * float(live_px)

    base_tp = (live_px + offset) if direction == "LONG" else (live_px - offset)
    base_sl = (live_px - offset) if direction == "LONG" else (live_px + offset)

    # Round for broker and spread/tick sanity
    try:
        base_tp = _round_limit_for_side(base_tp, side_exit)
        base_sl = _round_stop_for_side(base_sl, side_exit)
    except Exception:
        pass

    # Replace or (cancel→create) TP and SL using existing helpers
    # 1) TP
    place_or_amend_tp(api, pos, base_tp)
    # 2) SL (create or amend)
    place_or_amend_sl(api, pos, target_sl_price=base_sl, sl_order=None, side_exit=side_exit)

def _get_current_exits_with_sides(api, symbol: str, live_px: float | None):
    """
    Return (tp_px, sl_px, tp_side, sl_side) for a symbol by scanning open orders.
    Sides are the *exit* sides ('sell' closes LONG, 'buy' closes SHORT).
    Picks reasonable candidates by relation to live_px if available.
    """
    try:
        open_orders = api.list_orders(status="open", nested=True) or []
    except Exception:
        open_orders = []

    sym = str(symbol).upper()
    # collect candidates per side
    cands = {"sell": {"tp": [], "sl": []}, "buy": {"tp": [], "sl": []}}

    for o in open_orders:
        try:
            if str(getattr(o, "symbol", "")).upper() != sym:
                continue
            side = str(getattr(o, "side", "")).lower()          # 'sell' or 'buy'
            typ  = (getattr(o, "type", None) or getattr(o, "order_type", "") or "").lower()
            status = str(getattr(o, "status", "open")).lower()
            if status not in {"open","accepted","new","partially_filled","replaced","held"}:
                continue

            lp = getattr(o, "limit_price", None)
            sp = getattr(o, "stop_price",  None)

            # classify
            if "limit" in typ and lp is not None:
                cands.setdefault(side, cands.get(side, {"tp": [], "sl": []}))
                cands[side]["tp"].append(float(lp))
            elif ("stop" in typ or sp is not None):
                price = float(sp if sp is not None else (lp if lp is not None else 0))
                cands.setdefault(side, cands.get(side, {"tp": [], "sl": []}))
                cands[side]["sl"].append(price)
        except Exception:
            continue

    def _pick_tp(side, arr):
        if not arr:
            return None
        if live_px is None:
            return arr[0]
        if side == "sell":
            above = [p for p in arr if p > live_px]
            return min(above) if above else min(arr, key=lambda p: abs(p - live_px))
        else:
            below = [p for p in arr if p < live_px]
            return max(below) if below else min(arr, key=lambda p: abs(p - live_px))

    def _pick_sl(side, arr):
        if not arr:
            return None
        if live_px is None:
            return arr[0]
        if side == "sell":
            below = [p for p in arr if p < live_px]
            return max(below) if below else min(arr, key=lambda p: abs(p - live_px))
        else:
            above = [p for p in arr if p > live_px]
            return min(above) if above else min(arr, key=lambda p: abs(p - live_px))

    # choose exit side with any signals; prefer the side that yields a valid pair first
    out = (None, None, None, None)
    for sx in ("sell", "buy"):
        tp_px = _pick_tp(sx, cands.get(sx, {}).get("tp", []))
        sl_px = _pick_sl(sx, cands.get(sx, {}).get("sl", []))
        if (tp_px is not None) or (sl_px is not None):
            out = (tp_px, sl_px, sx, sx)
            break

    return out

def run_exits_healthcheck(api: REST, *, label: str = "open") -> None:
    """
    Verify each open position has a protective SL and a TP leg on the correct exit side.
    If one leg is missing (or 100% qty is locked in the other), rebalance and seed the missing leg.
    • Runs only when market is open (for label="open"); at label="close" it seeds an SL only.
    • Skips mutations within the last N minutes before the bell.
    • Debounces per-symbol to avoid thrash on slow snapshots.
    """
    # -------- local helpers (no external deps) --------
    import math, time, logging
    from datetime import datetime, time as _t
    try:
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("US/Eastern")
    except Exception:
        _ET = None

    def _now_et():
        dt = datetime.now()
        if _ET: 
            try: dt = datetime.now(_ET)
            except Exception: pass
        return dt

    def _near_close(minutes=5) -> bool:
        dt = _now_et().time()
        # 15:55 → 16:00 default
        return (dt >= _t(15, max(0, 60 - int(minutes))))

    def _market_open_now(api) -> bool:
        try:
            clk = api.get_clock()
            return bool(getattr(clk, "is_open", False))
        except Exception:
            return True  # best-effort fallback

    def _round_to_tick(px: float, tick: float) -> float:
        try:
            return round(float(px) / float(tick)) * float(tick)
        except Exception:
            return float(px)

    def _hc_log(sym: str, action: str, **kw):
        kv = " ".join(f"{k}={v}" for k, v in kw.items() if v is not None)
        logging.info("🏥 HC %s: %s %s", sym, action, kv)

    def _safe_int(x, default=0):
        try: return int(x)
        except Exception: return default

    # -------- config with safe defaults --------
    DRY_RUN = bool(globals().get("DRY_RUN", False))
    HEALTHCHECK_BASE_SL_FRAC    = float(globals().get("HEALTHCHECK_BASE_SL_FRAC", 0.60))
    HEALTHCHECK_AT_OPEN         = bool(globals().get("HEALTHCHECK_AT_OPEN", True))
    HEALTHCHECK_AT_CLOSE        = bool(globals().get("HEALTHCHECK_AT_CLOSE", True))
    HEALTHCHECK_OPEN_GRACE_MIN  = int(globals().get("HEALTHCHECK_OPEN_GRACE_MIN", 3))
    HEALTHCHECK_CLOSE_GRACE_MIN = int(globals().get("HEALTHCHECK_CLOSE_GRACE_MIN", 3))
    ATR_DOLLAR_FLOOR            = float(globals().get("ATR_DOLLAR_FLOOR", 0.0))
    EXTENDER_BASE_SL_ATR_MULT   = float(globals().get("EXTENDER_BASE_SL_ATR_MULT", 0.35))
    ENABLE_FAILSAFE             = bool(globals().get("ENABLE_FAILSAFE", False))  # used by staleness hook

    if DRY_RUN:
        logging.info("⏭️ DRY_RUN=True — skipping %s healthcheck.", label)
        return

    # market state guards
    if label == "open" and not _market_open_now(api):
        logging.info("ℹ️ Healthcheck(open): market not open — skipped.")
        return
    if label != "close" and _near_close(HEALTHCHECK_CLOSE_GRACE_MIN):
        logging.info("⏳ Near close — skipping %s healthcheck.", label)
        return

    # grace window at open: only run within first few minutes if desired
    if label == "open" and HEALTHCHECK_OPEN_GRACE_MIN > 0:
        try:
            clk = api.get_clock()
            # Some SDKs expose previous open as "next_open" during session; best-effort logic:
            sess_open = getattr(clk, "next_open", None)
            if sess_open:
                opent = pd.to_datetime(str(sess_open), utc=True, errors="coerce")
                if opent is not None and not pd.isna(opent):
                    now_utc = pd.Timestamp.utcnow().tz_localize("UTC")
                    if (now_utc - opent).total_seconds() > HEALTHCHECK_OPEN_GRACE_MIN * 60 + 600:
                        # If we’re *well* past the window, don't spam changes
                        logging.info("ℹ️ Past open-grace window — skipping intensive %s healthcheck.", label)
                        # continue anyway, but no-op if you prefer:
                        # return
        except Exception:
            pass

    # fetch positions
    try:
        positions = api.list_positions() or []
    except Exception as e:
        logging.warning("Healthcheck: cannot list positions: %s", e)
        return

    if not positions:
        logging.info("✅ Healthcheck(%s): no open positions.", label)
        return

    # per-symbol debounce across runs
    if not hasattr(run_exits_healthcheck, "_last_fix"):
        run_exits_healthcheck._last_fix = {}
    _last_fix: dict = run_exits_healthcheck._last_fix
    now_ts = time.time()

    fixed = missing_tp = missing_sl = 0

    for pos in positions:
        try:
            sym = str(getattr(pos, "symbol", "")).upper()
            qty = _safe_int(globals().get("_position_qty_for_submit", lambda p: int(float(getattr(p, "qty", 0))))(pos))
            if not sym or qty <= 0:
                continue

            # debounce (3s per symbol+label)
            key = (sym, label)
            if _last_fix.get(key, 0) and now_ts - float(_last_fix[key]) < 3.0:
                continue

            # side + exit side
            try:
                direction, side_exit = _position_side_and_exit(pos)  # 'LONG'/'SHORT', 'sell'/'buy'
            except Exception:
                side_exit = "sell" if str(getattr(pos, "side", "long")).lower() == "long" else "buy"
                direction = "LONG" if side_exit == "sell" else "SHORT"

            # live price
            live_px = None
            try:
                live_px = float(_latest_trade_price(api, sym))
            except Exception:
                pass
            if not live_px or not math.isfinite(live_px) or live_px <= 0:
                _hc_log(sym, "skip_no_live_price")
                continue

            # current exits
            tp_px = sl_px = None
            tp_side = sl_side = side_exit
            # Try enriched helper that also returns sides
            try:
                tp_px, sl_px, tp_side, sl_side = _get_current_exits_with_sides(api, sym, live_px)  # optional helper
            except Exception:
                try:
                    tp_px, sl_px = _get_current_exits(api, sym, side_exit, live_px)
                except Exception:
                    tp_px = sl_px = None
            # normalize to closing side
            if tp_px is not None and tp_side != side_exit:
                tp_px = None
            if sl_px is not None and sl_side != side_exit:
                sl_px = None

            has_tp = tp_px is not None
            has_sl = sl_px is not None

            # --- Wrong-side SL correction (run for BOTH 'open' and 'close') ---
            if has_sl:
                wrong_side = (
                    (direction == "LONG"  and sl_px is not None and live_px <= sl_px) or
                    (direction == "SHORT" and sl_px is not None and live_px >= sl_px)
                )
                if wrong_side:
                    _refresh_exits_from_baseline(api, pos, live_px, eff_atr)
                    _hc_log(sym, "refresh_wrong_side_sl",
                            live=f"{live_px:.4f}", sl=f"{(sl_px or float('nan')):.4f}")
                    fixed += 1
                    _last_fix[key] = time.time()
                    # On 'close' we still want to ensure an SL exists afterward; don't return yet.
                    if label != "close":
                        continue

            # ---- label=close policy: ensure SL only (no TP changes) ----
            if label == "close":
                if not has_sl:
                    ok = place_or_amend_sl(api, pos, target_sl_price=base_sl,
                                           sl_order=None, side_exit=side_exit)
                    if ok:
                        fixed += 1
                        _hc_log(sym, "seed_sl_only_close", sl=f"{base_sl:.4f}", qty=qty)
                _last_fix[key] = time.time()
                continue

            # ---- Case A: both exits present → optional staleness refresh ----
            if has_tp and has_sl:
                try:
                    score = assess_exit_staleness(
                        side_long_short=("Long" if direction == "LONG" else "Short"),
                        entry_px=float(getattr(pos, "avg_entry_price", live_px)),
                        live_px=live_px, tp_px=tp_px, sl_px=sl_px, atr=eff_atr,
                        submitted_at=_most_recent_exit_ts(api, sym, side_exit),
                        last_fill_ts=_last_fill_ts_for_symbol(api, sym),
                        tick=tick
                    )
                    if score and score.get("stale"):
                        _refresh_exits_from_baseline(api, pos, live_px, eff_atr)
                        _hc_log(sym, "refresh_stale", reasons=",".join(score.get("reasons", [])))
                        fixed += 1
                        _last_fix[key] = time.time()
                        continue
                except Exception:
                    pass
                # all good, move on
                continue

            # ---- Case B: both missing → seed TP first (~40%), then SL ----
            if (not has_tp) and (not has_sl):
                tp_frac = max(0.0, 1.0 - float(HEALTHCHECK_BASE_SL_FRAC))
                tp_req  = max(1, int(tp_frac * qty))
                if avail <= 0:
                    # synthesize from limits to avoid oversizing
                    avail = max(0, int(qty) - int(reserved_limits))
                tp_req = min(tp_req, int(avail))
                if tp_req > 0:
                    _submit_order_safe(
                        api, symbol=sym, qty=tp_req, side=side_exit,
                        type="limit", time_in_force="gtc",
                        limit_price=_round_to_tick(base_tp, tick),
                        reduce_only=True,
                    )
                    time.sleep(0.2)  # let snapshot settle
                place_or_amend_sl(api, pos, target_sl_price=base_sl, sl_order=None, side_exit=side_exit)
                _hc_log(sym, "seed_both", tp=f"{base_tp:.4f}", sl=f"{base_sl:.4f}", qty=qty)
                fixed += 1
                _last_fix[key] = time.time()
                continue

            # ---- Case C: SL missing → create SL (and free qty if all in TP) ----
            if has_tp and not has_sl:
                if _safe_int(reserved_limits) >= int(qty):
                    # free a chunk for SL
                    try:
                        _trim_open_limits_for(api, sym, side_exit,
                                              new_total_limit_qty=max(0, int((1.0 - HEALTHCHECK_BASE_SL_FRAC) * qty)))
                        time.sleep(0.15)
                    except Exception:
                        pass
                ok = place_or_amend_sl(api, pos, target_sl_price=base_sl, sl_order=None, side_exit=side_exit)
                if ok:
                    fixed += 1
                    _hc_log(sym, "seed_sl", sl=f"{base_sl:.4f}")
                else:
                    missing_sl += 1
                    _hc_log(sym, "seed_sl_failed")
                _last_fix[key] = time.time()
                continue

            # ---- Case D: TP missing → shrink SL if it locks 100%, then create TP ----
            if (not has_tp) and has_sl:
                desired_tp_qty = max(1, int((1.0 - float(HEALTHCHECK_BASE_SL_FRAC)) * qty))
                try:
                    _rebalance_sl_to_make_room_for_tp(
                        api=api, symbol=sym, side_exit=side_exit,
                        desired_tp_qty=desired_tp_qty,
                        tp_price=_round_to_tick(base_tp, tick),
                    )
                except Exception:
                    # fallback: just post a small TP if we can
                    free = max(0, int(qty) - int(reserved_limits))
                    if free > 0:
                        _submit_order_safe(
                            api, symbol=sym, qty=max(1, min(desired_tp_qty, free)), side=side_exit,
                            type="limit", time_in_force="gtc",
                            limit_price=_round_to_tick(base_tp, tick),
                            reduce_only=True,
                        )
                # ensure SL still exists afterward
                try:
                    sl_check = _get_symbol_open_sl_order_by_relation(api, sym, side_exit, live_px)
                    if not sl_check:
                        place_or_amend_sl(api, pos, target_sl_price=base_sl, sl_order=None, side_exit=side_exit)
                except Exception:
                    pass
                fixed += 1
                _hc_log(sym, "seed_tp", tp=f"{base_tp:.4f}")
                _last_fix[key] = time.time()
                continue
            # If SL exists but sits on the wrong side of live_px, immediately refresh both exits
            if has_sl:
                wrong_side = False
                if direction == "LONG" and sl_px is not None and live_px <= sl_px:
                    wrong_side = True
                if direction == "SHORT" and sl_px is not None and live_px >= sl_px:
                    wrong_side = True
                if wrong_side:
                    _refresh_exits_from_baseline(api, pos, live_px, eff_atr)
                    _hc_log(sym, "refresh_wrong_side_sl", live=f"{live_px:.4f}", sl=f"{(sl_px or float('nan')):.4f}")
                    fixed += 1
                    _last_fix[key] = time.time()
                    continue

        except Exception as e:
            logging.warning("Healthcheck(%s) error on %s: %s", label, getattr(pos, "symbol", "?"), e)

    logging.info("🏥 Healthcheck(%s) done — fixed=%d, missing_sl=%d, missing_tp=%d",
                 label, fixed, missing_sl, missing_tp)

def _rebalance_sl_to_make_room_for_tp(
    api,
    symbol: str,
    side_exit: str,            # 'sell' for LONG, 'buy' for SHORT (exit side)
    desired_tp_qty: int,
    tp_price: float,           # already rounded for side
) -> bool:
    """
    When SL consumes 100% of the position and no TP exists, cancel the full-qty SL,
    re-post a reduced-qty SL, then submit a new TP using the freed quantity.
    """
    # Find any existing SL
    try:
        open_orders = api.list_orders(status="open", nested=True) or []
    except Exception:
        open_orders = []
    sym = symbol.upper(); sx = side_exit.lower()
    sl_order = None
    for o in open_orders:
        try:
            if str(getattr(o,"symbol","")).upper()!=sym: continue
            if str(getattr(o,"side","")).lower()!=sx:    continue
            typ = (getattr(o,"type","") or getattr(o,"order_type","")).lower()
            if "stop" in typ or getattr(o,"stop_price",None) is not None:
                sl_order = o
                break
        except Exception:
            continue

    if not sl_order:
        logging.warning("Rebalance SL→TP requested, but no SL found for %s.", symbol)
        return False

    # Current SL fields
    try:
        sl_stop_px = float(getattr(sl_order,"stop_price",None) or getattr(sl_order,"limit_price",None) or 0.0)
    except Exception:
        sl_stop_px = 0.0
    raw_q = getattr(sl_order,"qty",None) or getattr(sl_order,"quantity",None)
    try:
        from decimal import Decimal, ROUND_FLOOR
        sl_qty_orig = int(Decimal(str(raw_q or "0")).to_integral_value(rounding=ROUND_FLOOR))
    except Exception:
        sl_qty_orig = 0

    if sl_qty_orig <= 0 or sl_stop_px <= 0:
        logging.warning("Rebalance SL→TP aborted: SL fields invalid for %s (qty=%s, stop=%s)", symbol, sl_qty_orig, sl_stop_px)
        return False

    # Compute reduced SL qty after carving out desired TP qty (leave ≥1 share on SL when possible)
    desired_tp_qty = max(int(desired_tp_qty), int(globals().get("SL_MIN_SHARES", 1)))
    sl_qty_new = max(0, sl_qty_orig - desired_tp_qty)
    if sl_qty_new == sl_qty_orig:
        logging.info("Rebalance SL→TP not needed for %s (sl_qty_new=%s, desired_tp=%s)", symbol, sl_qty_new, desired_tp_qty)
        return False

    # 1) Cancel existing SL and WAIT a beat until it disappears
    ok_cancel = _cancel_order_and_wait(api, getattr(sl_order,"id",None), symbol, side_exit, timeout_s=float(globals().get("CANCEL_ORDER_TIMEOUT_S", 2.0)))
    if not ok_cancel:
        logging.warning("Proceeding with SL→TP rebalance even though SL still appears open for %s.", symbol)

    # 2) Re-post reduced-qty SL (if any)
    ok_sl = True
    if sl_qty_new > 0:
        try:
            _submit_order_safe(
                api,
                symbol=symbol,
                qty=sl_qty_new,
                side=side_exit,
                type="stop",
                time_in_force="gtc",
                stop_price=_round_stop_for_side(sl_stop_px, side_exit),
                reduce_only=True,
            )
            logging.info("🔁 Reposted SL for %s qty=%s @ %s", symbol, sl_qty_new, round(sl_stop_px, int(globals().get("PRICE_DECIMALS", 2))))
        except Exception as e:
            logging.error("Failed to repost reduced SL for %s: %s", symbol, e)
            ok_sl = False

    # tiny settle so availability reflects reduced SL
    try:
        import time; time.sleep(0.2)
    except Exception:
        pass

    # 3) Submit new TP with whatever is now free (cap to availability)
    ok_tp = True
    # position qty
    try:
        pos_qty = _position_qty_for_submit(api.get_position(symbol))
    except Exception:
        # fallback scan
        try:
            positions = api.list_positions() or []
            pos_qty = next(( _position_qty_for_submit(p) for p in positions if str(getattr(p,"symbol","")).upper()==symbol.upper() ), 0)
        except Exception:
            pos_qty = 0

    avail = _available_exit_qty(api, symbol, side_exit, pos_qty)
    req_tp_qty = max(0, min(int(desired_tp_qty), int(avail)))
    if req_tp_qty <= 0:
        logging.warning("No free qty for new TP on %s after SL rebalance (pos=%s).", symbol, pos_qty)
        _debug_dump_open_orders_for_symbol(api, symbol, side_exit)
        return bool(ok_sl and False)

    try:
        _submit_order_safe(
            api,
            symbol=symbol,
            qty=req_tp_qty,
            side=side_exit,
            type="limit",
            time_in_force="gtc",
            limit_price=_round_limit_for_side(tp_price, side_exit),
            reduce_only=True,
        )
        logging.info("🆕 Created TP for %s qty=%s @ %s", symbol, req_tp_qty, round(tp_price, int(globals().get("PRICE_DECIMALS", 2))))
    except Exception as e:
        logging.error("Failed to create TP after SL rebalance for %s: %s", symbol, e)
        ok_tp = False

    return bool(ok_sl and ok_tp)

def _find_sl_by_parent_id(api, parent_id: str | None, symbol: str, side_exit: str):
    """
    If we know the TP's parent_order_id, find the sibling STOP/STOP_LIMIT in the same bracket.
    """
    if not parent_id:
        return None
    try:
        orders = api.list_orders(status="open", nested=True) or []
    except Exception:
        return None
    sym = str(symbol).upper(); side_exit = str(side_exit).lower()
    for o in orders:
        try:
            if str(getattr(o, "symbol", "")).upper() != sym:
                continue
            if str(getattr(o, "side", "")).lower() != side_exit:
                continue
            pid = getattr(o, "parent_order_id", None)
            if str(pid) != str(parent_id):
                continue
            otyp = (getattr(o, "type", None) or getattr(o, "order_type", "") or "").lower()
            if "stop" in otyp:
                return o
        except Exception:
            continue
    return None

# --- Bracket submit helper (place near your other broker helpers) ---
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING

def _tick_for_px(px: float) -> Decimal:
    return Decimal("0.01") if px >= 1 else Decimal("0.0001")

def assess_exit_staleness(*, side_long_short: str, entry_px: float, live_px: float,
                          tp_px: float | None, sl_px: float | None,
                          atr: float | None, submitted_at: "pd.Timestamp|None",
                          last_fill_ts: "pd.Timestamp|None",
                          tick: float = 0.01) -> dict:
    import math, numpy as np, pandas as pd
    reasons = []
    side = (side_long_short or "Long").strip().lower()
    is_long = side == "long"
    atr = float(atr) if atr and atr > 0 else None

    # 1) nonsense geometry
    if tp_px is not None:
        if (is_long and tp_px <= entry_px) or ((not is_long) and tp_px >= entry_px):
            reasons.append("tp_wrong_side_of_entry")
    if sl_px is not None:
        if (is_long and sl_px >= entry_px) or ((not is_long) and sl_px <= entry_px):
            reasons.append("sl_wrong_side_of_entry")

    # 2) live breach desync
    if sl_px is not None:
        if (is_long and live_px <= sl_px) or ((not is_long) and live_px >= sl_px):
            reasons.append("sl_breached_but_open")

    # 3) ATR drift
    if atr:
        if tp_px is not None:
            tp_gap = abs(tp_px - live_px) / atr
            if tp_gap < 0.5:  reasons.append("tp_too_close_vs_atr")
            if tp_gap > 6.0:  reasons.append("tp_too_far_vs_atr")
        if sl_px is not None:
            sl_gap = abs(live_px - sl_px) / atr
            if sl_gap < 0.25: reasons.append("sl_too_close_vs_atr")
            if sl_gap > 3.0:   reasons.append("sl_too_far_vs_atr")

    # 4) spread / tick sanity
    if tp_px is not None and abs(tp_px - live_px) < 2*tick:
        reasons.append("tp_inside_spread_or_too_close")
    if sl_px is not None and abs(sl_px - live_px) < 2*tick:
        reasons.append("sl_inside_spread_or_too_close")

    # 5) too old vs activity
    try:
        now = pd.Timestamp.utcnow().tz_localize("UTC")
        if submitted_at is not None and last_fill_ts is not None:
            age_d = (now - pd.to_datetime(submitted_at, utc=True)).total_seconds()/86400.0
            since_fill_d = (now - pd.to_datetime(last_fill_ts, utc=True)).total_seconds()/86400.0
            if age_d > 2 and since_fill_d < 5:  # exits older than 2d while fills recent-ish
                reasons.append("exit_older_than_2d_vs_activity")
    except Exception:
        pass

    return {
        "stale": len(reasons) > 0,
        "reasons": reasons
    }

def _round_limit_for_side(px: float, exit_side: str) -> float:
    # SELL limit rounds UP (ceiling); BUY limit rounds DOWN (floor)
    is_sell = (str(exit_side).lower().strip() == "sell")
    d = Decimal(str(px)); tick = _tick_for_px(float(d))
    q = (d / tick).to_integral_value(rounding=(ROUND_CEILING if is_sell else ROUND_FLOOR))
    return float(q * tick)

def _round_stop_for_side(px: float, exit_side: str) -> float:
    # SELL stop rounds DOWN; BUY stop rounds UP
    is_sell = (str(exit_side).lower().strip() == "sell")
    d = Decimal(str(px)); tick = _tick_for_px(float(d))
    q = (d / tick).to_integral_value(rounding=(ROUND_FLOOR if is_sell else ROUND_CEILING))
    return float(q * tick)

def submit_bracket_entry(api, *, symbol: str, side: str, qty: float,
                         entry_type: str = "market", entry_price: float | None = None,
                         tp_price: float, sl_price: float, tif: str = "gtc"):
    """
    Force a true Alpaca bracket entry so TP/SL legs exist from second zero.
    side: 'buy' or 'sell' (opening trade). For SHORT entries use side='sell'.
    """
    side = side.lower().strip()
    if side not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")

    # Exit side for the bracket legs
    exit_side = ("sell" if side == "buy" else "buy")

    # Safety rounding
    tp_price = _round_limit_for_side(tp_price, exit_side)
    sl_price = _round_stop_for_side(sl_price, exit_side)

    kwargs = dict(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=tif,
        order_class="bracket",
        take_profit={"limit_price": tp_price},
        stop_loss={"stop_price": sl_price},
    )

    if entry_type.lower() == "limit":
        if entry_price is None:
            raise ValueError("entry_price is required for limit entries")
        kwargs.update(type="limit", limit_price=_round_limit_for_side(entry_price, side))
    else:
        kwargs.update(type="market")

    return api.submit_order(**kwargs)

def _cancel_order_and_wait(api, order_id: str, symbol: str, side_exit: str,
                           timeout_s: float = CANCEL_ORDER_TIMEOUT_S) -> bool:
    """
    Cancel an order and wait until it no longer appears in open orders.
    Returns True if it disappears within timeout, else False.
    """
    try:
        api.cancel_order(order_id)
    except Exception as e:
        logging.warning("Could not cancel order %s for %s: %s", order_id, symbol, e)
        return False

    import time
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            open_orders = api.list_orders(status="open", nested=True) or []
        except Exception:
            open_orders = []
        # disappear check
        if not any(getattr(o, "id", None) == order_id for o in open_orders):
            return True
        time.sleep(0.1)
    logging.warning("Order %s for %s still visible after cancel timeout.", order_id, symbol)
    return False

def _enforce_min_sl_gap(live_px: float, raw_sl: float, direction: str, *, pct: float = 0.002) -> float:
    """
    Ensure the SL sits at least 'pct' away from price (default 0.2%) so we don't
    immediately trigger on tiny ticks. Returns adjusted SL.
    """
    gap = max(pct * float(live_px), 0.0)
    if direction == "LONG":
        return min(raw_sl, float(live_px) - gap)
    else:  # SHORT
        return max(raw_sl, float(live_px) + gap)

def _rebalance_tp_to_make_room_for_sl(
    api,
    symbol: str,
    side_exit: str,                 # 'sell' or 'buy' (exit side)
    desired_sl_qty: int,
    sl_price: float,                # already rounded for side
) -> bool:
    """
    When TP already consumes 100% of the position and no SL exists,
    cancel the full-qty TP, re-post a reduced-qty TP, then submit a new SL
    using the freed quantity. Returns True if both new orders were accepted.
    """
    # --- Helper: wait until a predicate becomes true or timeout
    def _wait_until(pred, timeout_s=2.0, poll_s=0.1):
        import time
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if pred():
                return True
            time.sleep(poll_s)
        return False

    try:
        live_px = _latest_trade_price(api, symbol)
    except Exception:
        live_px = None

    # 1) Find the current TP
    tp_order = _get_symbol_open_tp_order_by_relation(api, symbol, side_exit, live_px or 0.0)
    if not tp_order:
        logging.warning("Rebalance requested, but no TP found to reduce for %s.", symbol)
        return False

    # Extract current TP fields
    try:
        tp_limit_px = float(getattr(tp_order, "limit_price", None) or getattr(tp_order, "stop_price", None) or 0.0)
    except Exception:
        tp_limit_px = 0.0
    raw_q = getattr(tp_order, "qty", None) or getattr(tp_order, "quantity", None)
    try:
        from decimal import Decimal, ROUND_FLOOR
        tp_qty_orig = int(Decimal(str(raw_q or "0")).to_integral_value(rounding=ROUND_FLOOR))
    except Exception:
        tp_qty_orig = 0

    if tp_qty_orig <= 0 or tp_limit_px <= 0:
        logging.warning("Rebalance aborted: TP fields invalid for %s (qty=%s, limit=%s)", symbol, tp_qty_orig, tp_limit_px)
        return False

    # Compute new TP qty after carving out desired SL qty (leave ≥1 share on TP when possible)
    desired_sl_qty = max(int(desired_sl_qty), int(globals().get("SL_MIN_SHARES", 1)))
    tp_qty_new = max(0, tp_qty_orig - desired_sl_qty)
    if tp_qty_new == tp_qty_orig or desired_sl_qty <= 0:
        logging.info("Rebalance not needed for %s (tp_qty_new=%s, desired_sl=%s)", symbol, tp_qty_new, desired_sl_qty)
        return False

    # 2) Cancel existing TP and WAIT for it to disappear to avoid ghosting
    ok_cancel = _cancel_order_and_wait(api, getattr(tp_order, "id", None), symbol, side_exit, timeout_s=2.0)
    if not ok_cancel:
        # still try to proceed, but log a note (we’ll cap qty below anyway)
        logging.warning("Proceeding with rebalance even though TP still appears open for %s.", symbol)

    # 3) Re-post reduced-qty TP (if any)
    ok_tp = True
    if tp_qty_new > 0:
        try:
            _submit_order_safe(
                api,
                symbol=symbol,
                qty=tp_qty_new,
                side=side_exit,
                type="limit",
                time_in_force="gtc",
                limit_price=_round_limit_for_side(tp_limit_px, side_exit),
                reduce_only=True,
            )
            logging.info("🔁 Reposted TP for %s qty=%s @ %s", symbol, tp_qty_new, round(tp_limit_px, PRICE_DECIMALS))
        except Exception as e:
            logging.error("Failed to repost reduced TP for %s: %s", symbol, e)
            ok_tp = False

    # 3b) Wait up to ~2s for the reduced TP to show, so available qty is accurate
    def _reduced_tp_seen():
        o = _get_symbol_open_tp_order_by_relation(api, symbol, side_exit, live_px or 0.0)
        if not o:
            return (tp_qty_new == 0)  # if no TP expected, treat as seen
        oq = getattr(o, "qty", None) or getattr(o, "quantity", None)
        try:
            from decimal import Decimal, ROUND_FLOOR
            oqi = int(Decimal(str(oq or "0")).to_integral_value(rounding=ROUND_FLOOR))
        except Exception:
            oqi = None
        return (oqi == tp_qty_new)

    _wait_until(_reduced_tp_seen, timeout_s=2.0, poll_s=0.1)

    # 4) Submit the new SL with whatever free qty is actually available
    #    (cap the desired qty to availability to avoid 403s)
    ok_sl = True
    try:
        # position qty (whole shares)
        # we ask available after the reduced TP is visible
        pos_qty = _position_qty_for_submit(api.get_position(symbol))
    except Exception:
        # fallback to previous pos object if available
        try:
            positions = api.list_positions() or []
            pos_qty = next(( _position_qty_for_submit(p) for p in positions if str(getattr(p,"symbol","")).upper()==symbol.upper() ), 0)
        except Exception:
            pos_qty = 0

    avail = _available_exit_qty(api, symbol, side_exit, pos_qty)
    if avail <= 0:
        logging.warning("No free qty for new SL on %s after TP rebalance (pos=%s).", symbol, pos_qty)
        _debug_dump_open_orders_for_symbol(api, symbol, side_exit)
        return bool(ok_tp and False)

    req_sl_qty = min(int(desired_sl_qty), int(avail))
    try:
        _submit_order_safe(
            api,
            symbol=symbol,
            qty=req_sl_qty,
            side=side_exit,
            type="stop",
            time_in_force="gtc",
            stop_price=sl_price,
            reduce_only=True,
        )
        logging.info("🆕 Created SL for %s qty=%s @ %s", symbol, req_sl_qty, round(sl_price, PRICE_DECIMALS))
    except Exception as e:
        # If broker still complains (e.g., 403), try once with latest avail
        logging.warning("SL create failed for %s (%s). Retrying with fresh available qty…", symbol, e)
        avail2 = _available_exit_qty(api, symbol, side_exit, pos_qty)
        req2 = max(0, min(req_sl_qty, int(avail2)))
        if req2 <= 0:
            logging.error("Retry aborted for %s: no available qty after refresh.", symbol)
            ok_sl = False
        else:
            try:
                _submit_order_safe(
                    api,
                    symbol=symbol,
                    qty=req2,
                    side=side_exit,
                    type="stop",
                    time_in_force="gtc",
                    stop_price=sl_price,
                    reduce_only=True,
                )
                logging.info("🆕 Created SL (retry) for %s qty=%s @ %s", symbol, req2, round(sl_price, PRICE_DECIMALS))
            except Exception as e2:
                logging.error("Failed to create SL (retry) for %s: %s", symbol, e2)
                ok_sl = False

    return bool(ok_tp and ok_sl)

def _fresh_atr_from_yahoo(symbol: str, length: int = 14, lookback_days: int = 120, interval: str = "1d") -> float | None:
    """
    Pull recent daily bars from Yahoo and compute ATR(length).
    Returns latest ATR or None. Safe when yfinance isn't installed.
    """
    if yf is None:
        return None
    try:
        end = datetime.now().date()
        start = end - timedelta(days=lookback_days)
        # auto_adjust=False to keep raw High/Low/Close
        df = yf.download(
            str(symbol).upper(),
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if df is None or df.empty or len(df) < length + 1:
            logging.debug("ATR yahoo bars insufficient for %s (have=%s, need>=%s)", symbol, 0 if df is None else len(df), length + 1)
            return None

        # Normalize column names (ensure H/L/C present)
        cols = {c.lower(): c for c in df.columns}
        H = cols.get("high"); L = cols.get("low"); C = cols.get("close")
        if not (H and L and C):
            logging.debug("ATR yahoo missing H/L/C for %s", symbol)
            return None

        # Compute Wilder ATR
        _df = df[[H, L, C]].rename(columns={H: "h", L: "l", C: "c"}).astype(float).reset_index(drop=True)
        prev_close = _df["c"].shift(1)
        tr = pd.concat([
            _df["h"] - _df["l"],
            (_df["h"] - prev_close).abs(),
            (_df["l"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean()
        val = float(atr.iloc[-1])
        return val if val > 0 else None
    except Exception as e:
        logging.debug("ATR yahoo exception for %s: %s", symbol, e)
        return None

def _debug_dump_open_orders_for_symbol(api, symbol: str, side_exit: str):
    try:
        orders = api.list_orders(status="open", nested=True) or []
    except Exception:
        return
    sym = str(symbol).upper(); sx = str(side_exit).lower()
    logging.debug("↪ Open orders for %s (%s):", sym, sx)
    for o in orders:
        try:
            if str(getattr(o, "symbol","")).upper() != sym or str(getattr(o, "side","")).lower() != sx:
                continue
            logging.debug(
                "  id=%s type=%s cls=%s qty=%s limit=%s stop=%s coid=%s parent=%s",
                getattr(o,"id",None),
                (getattr(o,"type",None) or getattr(o,"order_type","")),
                getattr(o,"order_class",None),
                getattr(o,"qty",None) or getattr(o,"quantity",None),
                getattr(o,"limit_price",None),
                getattr(o,"stop_price",None),
                getattr(o,"client_order_id",None),
                getattr(o,"parent_order_id",None),
            )
        except Exception:
            continue

# --- Safe submit helper: retries without reduce_only if SDK doesn't support it ---
def _submit_order_safe(api, *, symbol, qty, side, type, time_in_force, **kx):
    """
    Wrapper around submit_order with friendly logging and a one-time downgrade
    when the SDK lacks reduce_only support.
    """
    global _REDUCE_ONLY_WARNED
    from alpaca_trade_api.rest import APIError

    try:
        return api.submit_order(symbol=symbol, qty=qty, side=side, type=type,
                                time_in_force=time_in_force, **kx)
    except Exception as e:
        msg = str(e).lower()
        if "reduce_only" in msg and not _REDUCE_ONLY_WARNED:
            logging.warning("submit_order doesn't support reduce_only in this SDK; retrying without it (was %s).", kx.get("reduce_only"))
            _REDUCE_ONLY_WARNED = True
        elif "reduce_only" in msg:
            # subsequent occurrences stay quiet
            pass
        else:
            raise
    # retry without reduce_only
    kx.pop("reduce_only", None)
    return api.submit_order(symbol=symbol, qty=qty, side=side, type=type,
                            time_in_force=time_in_force, **kx)


from decimal import Decimal, ROUND_FLOOR
def _order_remaining_qty(o) -> int:
    from decimal import Decimal, ROUND_FLOOR
    try:
        q_raw = getattr(o, "qty", None) or getattr(o, "quantity", None) or "0"
        f_raw = getattr(o, "filled_qty", None) or getattr(o, "filled_quantity", None) or "0"
        q = int(Decimal(str(q_raw)).to_integral_value(rounding=ROUND_FLOOR))
        f = int(Decimal(str(f_raw)).to_integral_value(rounding=ROUND_FLOOR))
        return max(0, q - f)
    except Exception:
        return 0

def _get_current_exits(api, symbol: str, side_exit: str, live_px: float | None):
    """
    Returns (tp_price, sl_price) or (None, None) if not visible.
    Robust to SDK differences:
      - uses nested=True
      - accepts stop or stop_limit or any order with stop_price
      - picks reasonable candidate by relation to live price
    """
    try:
        orders = api.list_orders(status="open", nested=True) or []
    except Exception:
        orders = []

    sym = symbol.upper(); sx = side_exit.lower()

    tp_cands = []
    sl_cands = []

    for o in orders:
        try:
            if str(getattr(o,"symbol","")).upper() != sym: continue
            if str(getattr(o,"side","")).lower()   != sx:  continue

            typ = (getattr(o, "type", None) or getattr(o, "order_type","") or "").lower()
            lp  = getattr(o, "limit_price", None)
            sp  = getattr(o, "stop_price",  None)
            status = str(getattr(o, "status", "open")).lower()
            if status not in {"open","accepted","new"}:
                continue

            # TP = LIMIT exits with remaining qty
            if typ == "limit" and lp is not None and _order_remaining_qty(o) > 0:
                tp_cands.append((float(lp), o))

            # SL = STOP/STOP_LIMIT exits (or anything exposing stop_price)
            if ("stop" in typ or sp is not None) and _order_remaining_qty(o) > 0:
                sl_cands.append((float(sp if sp is not None else lp), o))
        except Exception:
            continue

    tp_price = None
    sl_price = None

    # Choose best TP by relation to live price if available; else first
    if tp_cands:
        if live_px is not None:
            if sx == "sell":
                # LONG exit → TP above current price → choose lowest above
                above = [c for c in tp_cands if c[0] > live_px]
                if above:
                    tp_price = min(above, key=lambda t: t[0])[0]
                else:
                    tp_price = min(tp_cands, key=lambda t: abs(t[0]-live_px))[0]
            else:
                # SHORT exit → TP below current price → choose highest below
                below = [c for c in tp_cands if c[0] < live_px]
                if below:
                    tp_price = max(below, key=lambda t: t[0])[0]
                else:
                    tp_price = min(tp_cands, key=lambda t: abs(t[0]-live_px))[0]
        else:
            tp_price = tp_cands[0][0]

    # Choose best SL by relation to live price; else first
    if sl_cands:
        if live_px is not None:
            if sx == "sell":
                # LONG exit → SL below current price → choose highest below
                below = [c for c in sl_cands if c[0] < live_px]
                if below:
                    sl_price = max(below, key=lambda t: t[0])[0]
                else:
                    sl_price = min(sl_cands, key=lambda t: abs(t[0]-live_px))[0]
            else:
                # SHORT exit → SL above current price → choose lowest above
                above = [c for c in sl_cands if c[0] > live_px]
                if above:
                    sl_price = min(above, key=lambda t: t[0])[0]
                else:
                    sl_price = min(sl_cands, key=lambda t: abs(t[0]-live_px))[0]
        else:
            sl_price = sl_cands[0][0]

    return tp_price, sl_price
def _sum_exit_limit_remaining(api, symbol: str, side_exit: str) -> int:
    """Sum remaining qty of ANY open LIMIT orders on the exit side (counts TP legs even if not bracket children)."""
    try:
        orders = api.list_orders(status="open", nested=True) or []
    except Exception:
        return 0
    sym = symbol.upper(); side = side_exit.lower()
    total = 0
    for o in orders:
        try:
            if str(getattr(o, "symbol","")).upper() != sym:      continue
            if str(getattr(o, "side","")).lower() != side:       continue
            typ = (getattr(o, "type", "") or getattr(o, "order_type","")).lower()
            if typ != "limit":                                    continue
            total += _order_remaining_qty(o)
        except Exception:
            continue
    return max(0, total)

def _has_exit_limit(api, symbol: str, side_exit: str) -> bool:
    return _sum_exit_limit_remaining(api, symbol, side_exit) > 0


def _available_exit_qty(api, symbol: str, side_exit: str, pos_qty: int) -> int:
    """
    Free exitable qty = position qty - qty reserved by active exit orders (TP/SL)
    on the same side. Robust to Alpaca 'ghost' parents and partial fills.
    """
    try:
        open_orders = api.list_orders(status="open", nested=True) or []
    except Exception:
        open_orders = []

    # Normalize comparisons once
    sym_u = symbol.upper()
    side_u = side_exit.lower()

    reserved = 0
    for o in open_orders:
        try:
            sym_ok = str(getattr(o, "symbol", "")).upper() == sym_u
            side_ok = str(getattr(o, "side", "")).lower() == side_u
            status  = str(getattr(o, "status", "open")).lower()
            cls     = str(getattr(o, "order_class", "") or getattr(o, "cls", "")).lower()
            typ     = str(getattr(o, "type", "")).lower()
            parent  = getattr(o, "parent_order_id", None)
        except Exception:
            continue

        if not (sym_ok and side_ok):
            continue

        # Ignore clearly inactive
        if status in {"canceled", "expired", "done_for_day", "filled"}:
            continue

        # Count only exit legs: limit TP / stop SL / stop_limit SL
        if typ not in {"limit", "stop", "stop_limit"}:
            continue

        # In bracket flows, ignore the ENTRY parent (which can momentarily appear as limit).
        # Prefer counting only children with a parent_order_id.
        if cls == "bracket" and not parent:
            # Parent without parent_order_id → likely the entry; skip.
            continue

        # Remaining qty (handles partial fills)
        rem = _order_remaining_qty(o)
        if rem <= 0:
            continue

        reserved += rem

    # Don’t let reserved exceed position qty (API duplicates / flicker safety)
    reserved = min(max(0, reserved), max(0, int(pos_qty)))
    free_qty = max(0, int(pos_qty) - reserved)
    return free_qty


# ===== Stop-Loss helpers (add near other helpers) =====
def _round_stop_for_side(price: float, side_or_exit: str) -> float:
    """
    Rounds a stop price cautiously:
      * SELL stop (closing a LONG)  → round DOWN (floor)
      * BUY  stop (closing a SHORT) → round UP   (ceiling)
    """
    s = str(side_or_exit).lower().strip()
    if s in ("sell", "buy"):
        is_sell = (s == "sell")
    elif s in ("long", "short"):
        is_sell = (s == "long")  # long closes with SELL stop
    else:
        is_sell = True

    px = Decimal(str(price))
    tick = _tick_for_price(float(px))
    q = px / tick
    q = q.to_integral_value(rounding=ROUND_FLOOR if is_sell else ROUND_CEILING)
    return float(q * tick)

def _get_symbol_open_sl_order_by_relation(api, symbol: str, side_exit: str, live_px: float | None):
    """
    Find the open STOP/STOP-LIMIT order used as protective SL for this symbol+exit side.
    Pass 1: prefer orders that satisfy the expected relation to the live price
            (LONG→SELL stop below live, SHORT→BUY stop above live).
    Pass 2: if none match, pick the best available stop on that side.
    """
    try:
        # nested=True helps surface bracket legs in some SDK versions
        open_orders = api.list_orders(status="open", nested=True) or []
    except Exception as e:
        logging.warning("Could not list open orders for %s: %s", symbol, e)
        return None

    sym = str(symbol).upper()
    side_exit = str(side_exit).lower().strip()

    # Collect stop / stop-limit candidates (or anything with a stop_price)
    cands = []
    for o in open_orders:
        try:
            if str(getattr(o, "symbol", "")).upper() != sym:
                continue
            if str(getattr(o, "side", "")).lower() != side_exit:
                continue
            otyp = (getattr(o, "type", None) or getattr(o, "order_type", "") or "").lower()
            stop_px = getattr(o, "stop_price", None)
            limit_px = getattr(o, "limit_price", None)

            looks_like_stop = ("stop" in otyp) or (stop_px is not None)
            if not looks_like_stop:
                continue

            # Use stop_price if present, else limit_price to rank proximity
            if stop_px is not None:
                price = float(stop_px)
            elif limit_px is not None:
                price = float(limit_px)
            else:
                continue

            cands.append((o, price))
        except Exception:
            continue

    if not cands:
        return None

    # Pass 1 — enforce relation to live price (if we have it)
    if live_px is not None:
        below = [(o, p) for (o, p) in cands if p <  live_px]
        above = [(o, p) for (o, p) in cands if p >  live_px]

        if side_exit == "sell":
            # LONG → stop should be below live; choose the highest (closest protective)
            if below:
                below.sort(key=lambda t: -t[1])
                return below[0][0]
        else:
            # SHORT → stop should be above live; choose the lowest (closest protective)
            if above:
                above.sort(key=lambda t: t[1])
                return above[0][0]

    # Pass 2 — fallback: pick a reasonable stop even if relation filter failed
    if side_exit == "sell":
        return max(cands, key=lambda t: t[1])[0]  # LONG: highest stop
    else:
        return min(cands, key=lambda t: t[1])[0]  # SHORT: lowest stop

def tighten_stop_loss_to_match_step(
    api,
    pos,
    live_px: float,
    atr_val: float | None,
    mode: str,
    step_pct: float,
    step_atr: float,
    cur_tp_step_value: float | None = None,
    be_lock_atr_mult: float | None = None,
    tp_parent_id: str | None = None,
    tp_qty: int | None = None,
    tp_coid_prefix: str | None = None,
):
    """
    Tighten the SL by the SAME step as TP just moved (ATR or %).
    - Only tightens (never loosens).
    - Optional breakeven lock: once gain >= be_lock_atr_mult × ATR, SL >= entry (LONG) / <= entry (SHORT).
    Returns True if SL was amended/created, else False.
    """
    try:
        symbol = str(getattr(pos, "symbol", "")).upper()
        qty    = float(getattr(pos, "qty", 0) or 0.0)
        side   = str(getattr(pos, "side", "long")).lower()
        entry  = float(getattr(pos, "avg_entry_price", 0) or 0.0)
        if qty == 0 or entry <= 0:
            return False
    except Exception:
        return False

    is_long = (side == "long")
    direction, side_exit = _position_side_and_exit(pos)  # 'LONG'/'SHORT', 'sell'/'buy'
    PRICE_DEC = int(globals().get("PRICE_DECIMALS", 2))

    # --- Find SL: (1) sibling by parent id, (2) fallback by qty/COID, (3) fallback by price-relation
    sl_order = _find_sl_by_parent_id(api, tp_parent_id, symbol, side_exit)
    if sl_order is None:
        sl_order = _find_sl_by_qty_or_coid(api, symbol, side_exit, tp_qty, tp_coid_prefix)
    if sl_order is None:
        sl_order = _get_symbol_open_sl_order_by_relation(api, symbol, side_exit, live_px)
    if sl_order is None:
        sl_order = _find_any_stop_exit(api, symbol, side_exit)

    # Current SL price (after final selection)
    cur_sl = None
    if sl_order:
        try:
            cur_sl = float(getattr(sl_order, "stop_price", None) or getattr(sl_order, "limit_price", None) or 0.0)
        except Exception:
            cur_sl = None

    # Compute same step size as TP
    if cur_tp_step_value is not None:
        step_value = float(cur_tp_step_value)
    else:
        if str(mode).lower() == "atr":
            if not (atr_val and atr_val > 0):
                return False
            eff_atr = max(float(atr_val), float(globals().get("ATR_DOLLAR_FLOOR", 0.0)))
            step_value = float(step_atr) * eff_atr
            _base_for_rel = abs(cur_sl if (cur_sl is not None and cur_sl > 0) else live_px)
            min_step = max(
                float(globals().get("TP_EXTEND_MIN_ABS_STEP", 0.05)),
                float(globals().get("TP_EXTEND_MIN_REL_STEP", 0.001)) * _base_for_rel,
                float(globals().get("ATR_STEP_MIN_DOLLARS", 0.0)),
            )
            if step_value < min_step:
                step_value = min_step
        else:
            step_value = float(step_pct) * live_px

    # Propose tighter SL (never loosen)
    if cur_sl is None or cur_sl <= 0:
        proposed = (entry - step_value) if is_long else (entry + step_value)
    else:
        proposed = (cur_sl + step_value) if is_long else (cur_sl - step_value)

    tick = 10 ** (-PRICE_DEC)
    if is_long:
        proposed = min(proposed, live_px - tick)
        if cur_sl is not None and proposed <= cur_sl:
            return False
    else:
        proposed = max(proposed, live_px + tick)
        if cur_sl is not None and proposed >= cur_sl:
            return False

    # Optional breakeven lock
    if be_lock_atr_mult and atr_val and atr_val > 0:
        delta = (live_px - entry) if is_long else (entry - live_px)
        if delta >= be_lock_atr_mult * float(atr_val):
            proposed = max(proposed, entry) if is_long else min(proposed, entry)

    # Minimum meaningful move
    _ABS_MIN = float(globals().get("TP_EXTEND_MIN_ABS_STEP", 0.05))
    _REL_MIN = float(globals().get("TP_EXTEND_MIN_REL_STEP", 0.001))
    if cur_sl is not None and abs(proposed - cur_sl) < max(_ABS_MIN, _REL_MIN * abs(cur_sl)):
        return False

    rounded_sl = _round_stop_for_side(proposed, side_exit)

    # Prefer replace (keep OCO)
    if sl_order and globals().get("REPLACE_ORDER_FIRST", True):
        try:
            api.replace_order(getattr(sl_order, "id", None), stop_price=rounded_sl)
            logging.info("🔒 Replaced SL for %s %s → %0.*f", symbol, direction, PRICE_DEC, rounded_sl)
            return True
        except Exception as e:
            logging.warning("replace_order failed for SL %s: %s — falling back to cancel+submit", symbol, e)

    # Cancel then create (only if free qty remains)
    if sl_order:
        oid = getattr(sl_order, "id", None)
        try:
            api.cancel_order(oid)
        except Exception as e:
            logging.warning("Could not cancel existing SL for %s: %s", symbol, e)
        # Wait so open-orders snapshot reflects the cancel, then mark as missing.
        _cancel_order_and_wait(api, oid, symbol, side_exit,
                            timeout_s=float(globals().get("CANCEL_ORDER_TIMEOUT_S", 2.0)))
        sl_order = None  # ← IMPORTANT: allow rebalance path below

    pos_qty = _position_qty_for_submit(pos)
    if pos_qty <= 0:
        return False

    avail = _available_exit_qty(api, symbol, side_exit, pos_qty)
    if avail <= 0:
        logging.warning(
            "No free qty for new SL on %s — TP/SL legs already reserve %s of %s.",
            symbol, pos_qty, pos_qty
        )

        # Try to free quantity by shrinking TP regardless of prior SL variable state.
        if bool(globals().get("SL_REBALANCE_IF_MISSING", True)):
            ok = _rebalance_tp_to_make_room_for_sl(
                api=api,
                symbol=symbol,
                side_exit=side_exit,
                desired_sl_qty=max(1, int(pos_qty * float(globals().get("SL_REBALANCE_QTY_FRACTION", 0.50)))),
                sl_price=rounded_sl
            )
            if ok:
                return True

        _debug_dump_open_orders_for_symbol(api, symbol, side_exit)
        return False

    # Otherwise, submit a new SL with the capped free qty...
    _submit_order_safe(
        api,
        symbol=symbol,
        qty=int(avail),
        side=side_exit,
        type="stop",
        time_in_force="gtc",
        stop_price=rounded_sl,
        reduce_only=True,
    )
    logging.info("🔒 Submitted SL for %s %s → %0.*f (qty=%s)", symbol, direction, PRICE_DEC, rounded_sl, int(avail))
    return True

def _fresh_atr_from_bars(api: REST, symbol: str, length: int = 14) -> float | None:
    """
    Fetch recent daily bars and compute ATR(length).
    Handles DF/dict/list returns, forces feed='iex' (paper), logs bar count.
    """
    try:
        sy = str(symbol).upper()

        # Ask for extra bars so we have length TRs after shift
        # IMPORTANT: some SDKs support feed/adjustment; pass feed='iex' for paper accounts
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=90)

        raw = api.get_bars(
            sy,
            TimeFrame.Day,
            start=start.isoformat(),
            end=end.isoformat(),
            limit=length + 200,   # generous ceiling; API will still bound by start/end
            adjustment="raw",
            feed="iex",
        )

        # ---- Normalize to DataFrame with columns [h,l,c]
        df = None

        # Case A: DataFrame
        if isinstance(raw, pd.DataFrame):
            df0 = raw.reset_index(drop=False)
            cmap = {c.lower(): c for c in df0.columns}
            H = cmap.get("high") or cmap.get("h")
            L = cmap.get("low")  or cmap.get("l")
            C = cmap.get("close")or cmap.get("c")
            if H and L and C:
                df = pd.DataFrame({"h": df0[H].astype(float),
                                   "l": df0[L].astype(float),
                                   "c": df0[C].astype(float)})

        # Case B: dict of DataFrames keyed by symbol
        if df is None and isinstance(raw, dict):
            sub = raw.get(sy)
            if isinstance(sub, pd.DataFrame) and not sub.empty:
                df0 = sub.reset_index(drop=False)
                cmap = {c.lower(): c for c in df0.columns}
                H = cmap.get("high") or cmap.get("h")
                L = cmap.get("low")  or cmap.get("l")
                C = cmap.get("close")or cmap.get("c")
                if H and L and C:
                    df = pd.DataFrame({"h": df0[H].astype(float),
                                       "l": df0[L].astype(float),
                                       "c": df0[C].astype(float)})

        # Case C: iterable of Bar objects
        if df is None:
            try:
                rows = list(raw)
            except Exception:
                rows = []
            if rows:
                df = pd.DataFrame({
                    "h": [float(getattr(b, "h", getattr(b, "high", 0))) for b in rows],
                    "l": [float(getattr(b, "l", getattr(b, "low",  0))) for b in rows],
                    "c": [float(getattr(b, "c", getattr(b, "close",0))) for b in rows],
                })

        # Log the bar count so you can see empties in DEBUG
        if df is None or df.empty:
            logging.debug("ATR fresh bars empty for %s", sy)
            return None
        if len(df) < length + 1:
            logging.debug("ATR fresh bars too few for %s: have=%d need>=%d", sy, len(df), length + 1)
            return None

        prev_close = df["c"].shift(1)
        tr = pd.concat([
            df["h"] - df["l"],
            (df["h"] - prev_close).abs(),
            (df["l"] - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean()
        val = float(atr.iloc[-1])
        if val <= 0:
            logging.debug("ATR fresh computed <= 0 for %s", sy)
            return None

        return val
    except Exception as e:
        logging.debug("ATR fresh exception for %s: %s", symbol, e)
        return None

def _resolve_atr_for_symbol(api: REST, symbol: str, hints: dict[str, float]) -> tuple[float | None, str]:
    """
    Try ATR sources in ATR_SOURCE_ORDER (e.g., ["fresh","folder","yahoo"]).
    Returns (atr_value, source_str) where source_str in {"fresh","folder","yahoo","none"}.
    """
    sym = str(symbol).upper()

    for src in ATR_SOURCE_ORDER:
        if src == "fresh":
            val = _fresh_atr_from_bars(api, sym, length=ATR_LENGTH)
            if val and val > 0:
                return val, "fresh"

        elif src == "folder":
            val = _atr_value(hints.get(sym))
            if val and val > 0:
                return val, "folder"

        elif src == "yahoo":
            if not YAHOO_FALLBACK_ENABLE:
                continue
            val = _fresh_atr_from_yahoo(sym, length=ATR_LENGTH, lookback_days=YAHOO_LOOKBACK_DAYS, interval=YAHOO_INTERVAL)
            if val and val > 0:
                return val, "yahoo"

        else:
            logging.debug("Unknown ATR source '%s' in ATR_SOURCE_ORDER; skipping.", src)

    return None, "none"


def _get_symbol_open_tp_order_by_relation(api, symbol: str, side_exit: str, live_px: float):
    try:
        open_orders = api.list_orders(status="open", nested=True)  # <— important
    except Exception as e:
        logging.warning("Could not list open orders for %s: %s", symbol, e)
        return None

    symbol   = symbol.upper()
    side_exit = side_exit.lower().strip()
    candidates = []
    for o in open_orders:
        try:
            if str(getattr(o, "symbol", "")).upper() != symbol:
                continue
            if str(getattr(o, "side", "")).lower() != side_exit:
                continue
            coid = str(getattr(o, "client_order_id", "")).lower()
            ocls = str(getattr(o, "order_class", "")).lower()
            tag_is_tp = ("tp" in coid) or ("take" in coid and "profit" in coid) or ("take_profit" in ocls)
            limit_px = getattr(o, "limit_price", None)
            stop_px  = getattr(o, "stop_price", None)
            price = float(limit_px or stop_px or 0)
            candidates.append((o, tag_is_tp, price))
        except Exception:
            continue

    # Prefer explicit TP tags
    tagged = [o for (o, tag, _) in candidates if tag]
    if tagged:
        return tagged[0]

    # Fallback by relation to live price
    up = [(o, price) for (o, _tag, price) in candidates if price and price > live_px]
    dn = [(o, price) for (o, _tag, price) in candidates if price and price < live_px]
    if side_exit == "sell":
        if up:
            up.sort(key=lambda t: t[1])
            return up[0][0]
    else:
        if dn:
            dn.sort(key=lambda t: -t[1])
            return dn[0][0]
    return None

def _atr_value(atr_hint):
    try:
        if isinstance(atr_hint, (int, float)): return float(atr_hint)
        val = getattr(atr_hint, "atr", None)
        return float(val) if val is not None else None
    except Exception:
        return None

def _round_limit_for_side(price: float, side_or_exit: str) -> float:
    """
    Accepts either:
      - position direction: 'long' / 'short', OR
      - exit side:          'sell' / 'buy'
    Rules:
      * SELL limit (closing a long) → round UP (ceiling)
      * BUY  limit (closing a short) → round DOWN (floor)
    """
    s = str(side_or_exit).lower().strip()
    if s in ("sell", "buy"):
        is_sell = (s == "sell")
    elif s in ("long", "short"):
        is_sell = (s == "long")  # long position closes with SELL
    else:
        # conservative default: treat as SELL
        is_sell = True

    px = Decimal(str(price))
    tick = _tick_for_price(float(px))
    q = px / tick
    q = q.to_integral_value(rounding=ROUND_CEILING if is_sell else ROUND_FLOOR)
    return float(q * tick)


def _meaningful_tp_move(cur_tp: float, new_tp: float) -> bool:
    """Require either an absolute or relative minimum change."""
    if cur_tp is None or new_tp is None:
        return False
    try:
        cur_tp = float(cur_tp); new_tp = float(new_tp)
    except Exception:
        return False
    abs_ok = abs(new_tp - cur_tp) >= MIN_ABS_STEP
    rel_ok = abs((new_tp / cur_tp) - 1.0) >= MIN_REL_STEP
    return abs_ok or rel_ok

# ---------------- Alpaca helpers ----------------
def reason_log(sym, **k):
    kv = " ".join(f"{kk}={vv}" for kk, vv in k.items())
    logging.info(f"└─ {sym}: {kv}")

def _position_side_and_exit(pos):
    qty = float(getattr(pos, "qty", 0) or 0)
    side = str(getattr(pos, "side", "")).lower()
    direction = "LONG" if side == "long" or qty > 0 else "SHORT"
    side_exit = "sell" if direction == "LONG" else "buy"
    return direction, side_exit

def evaluate_tp_extension(
    pos, live_px: float, atr_val: float, mode: str,
    trigger_pct: float, step_pct: float, trigger_atr: float, step_atr: float,
    max_mult: float, cur_tp: float | None = None
):
    """
    Returns (ok: bool, reason_dict: dict, new_tp: Optional[float])

    - In ATR mode: uses ATR-based trigger and step.
    - In percent mode: uses percent-based trigger and step.
    - Handles long vs short properly.
    - Avoids extending on unfavorable/flat moves.
    - Enforces max_mult cap (relative to entry).
    """
    # Basic fields from position (Alpaca Position object semantics)
    try:
        side = str(getattr(pos, "side", "long")).lower()
        qty  = float(getattr(pos, "qty", 0) or getattr(pos, "quantity", 0) or 0)
        entry_px = float(getattr(pos, "avg_entry_price", 0) or getattr(pos, "entry_price", 0) or 0)
    except Exception:
        return False, {"reason": "pos_fields_error"}, None

    if qty == 0 or entry_px <= 0:
        return False, {"reason": "empty_or_bad_position"}, None

    is_long = (side == "long")
    delta   = (live_px - entry_px) if is_long else (entry_px - live_px)  # signed PnL move favorable if > 0
    if delta <= 0:
        return False, {"reason": "unfavorable_or_flat", "delta": round(delta, 6)}, None

    # Current TP level: prefer the value passed from the live TP order
    if cur_tp is None:
        for attr in ("take_profit", "tp", "take_profit_price", "take_profit_limit_price"):
            v = getattr(pos, attr, None)
            if v:
                try:
                    cur_tp = float(v)
                    break
                except Exception:
                    pass
    if cur_tp is None:
        cur_tp = entry_px * (1.02 if is_long else 0.98)
            
    # Compute move magnitudes
    pct_move = delta / entry_px

    move_atr = None
    if atr_val is not None and atr_val > 0:
        move_atr = abs(live_px - entry_px) / atr_val

    if mode == "atr":
        if atr_val is None or atr_val <= 0:
            return False, {"reason": "no_atr_hint_in_atr_mode"}, None

        # === ATR floor applied here ===
        eff_atr = max(float(atr_val), float(ATR_DOLLAR_FLOOR))

        # ✅ Optional log flag: did the floor kick in?
        if eff_atr > float(atr_val):
            floor_applied = True
        else:
            floor_applied = False

        # Trigger readiness based on ATR distance
        move_atr = abs(live_px - entry_px) / eff_atr
        trigger_ready = (move_atr >= float(trigger_atr))

        # Compute base step
        step_value = float(step_atr) * eff_atr

        # Apply minimum step guardrails
        _base_for_rel = abs(cur_tp if (cur_tp is not None and cur_tp > 0) else live_px)
        min_step = max(
            MIN_ABS_STEP,
            MIN_REL_STEP * _base_for_rel,
            float(ATR_STEP_MIN_DOLLARS)
        )
        if step_value < min_step:
            step_value = min_step

        step_type = "ATR"
        step_key  = "atr_step"
        trig_key  = "atr_trigger"
        trig_val  = float(trigger_atr)
        step_val  = float(step_atr)
    else:
        # (percent mode unchanged)
        trigger_ready = pct_move >= float(trigger_pct)
        step_value = float(step_pct) * live_px
        step_type = "%"
        step_key = "step_pct"
        trig_key = "trigger_pct"
        trig_val = float(trigger_pct)
        step_val = float(step_pct)

    if not trigger_ready:
        reason = "trigger_not_met"
        payload = {"pct_move": round(100 * pct_move, 4), trig_key: trig_val}
        if mode == "atr" and move_atr is not None:
            payload["move_atr"] = round(move_atr, 4)
        return False, payload | {"reason": reason}, None

    # Compute proposed new TP one step tighter in the profitable direction
    if is_long:
        proposed = cur_tp + step_value
        # Max cap relative to entry: don't exceed entry * (1 + max_mult * (cur_tp-entry)/???)
        # If you interpret max_mult as "multiple of initial TP distance", we need that baseline.
        # A simpler, common cap is: do not set TP beyond entry*(1 + max_mult * ATR) in ATR mode
        # or entry*(1 + max_mult * step_pct) in percent mode. Keep your existing system if you already enforce elsewhere.
        if max_mult and max_mult > 0 and mode == "atr" and atr_val:
            hard_cap = entry_px + max_mult * atr_val
            if proposed > hard_cap:
                proposed = hard_cap
                capped = True
            else:
                capped = False
        else:
            capped = False
    else:
        proposed = cur_tp - step_value
        if max_mult and max_mult > 0 and mode == "atr" and atr_val:
            hard_cap = entry_px - max_mult * atr_val
            if proposed < hard_cap:
                proposed = hard_cap
                capped = True
            else:
                capped = False
        else:
            capped = False

    # If the step would move TP the wrong way (e.g., due to missing/odd cur_tp), stop
    if is_long and proposed <= cur_tp:
        return False, {"reason": "non_tightening_step_long"}, None
    if (not is_long) and proposed >= cur_tp:
        return False, {"reason": "non_tightening_step_short"}, None
    # Skip tiny adjustments (absolute OR relative threshold)
    if abs(proposed - cur_tp) < max(MIN_ABS_STEP, MIN_REL_STEP * abs(cur_tp)):
        return False, {
            "reason": "tp_step_too_small",
            "cur_tp": round(cur_tp, 6),
            "new_tp": round(proposed, 6),
            "min_abs": MIN_ABS_STEP,
            "min_rel": MIN_REL_STEP,
        }, None

    rsn = {
        "dir": "LONG" if is_long else "SHORT",
        "entry": round(entry_px, 6),
        "live": round(live_px, 6),
        "delta": round(delta, 6),
        "cur_tp": round(cur_tp, 6),
        "new_tp": round(proposed, 6),
        "capped": bool(capped),
        "step_type": step_type,
        step_key: round(step_val, 6),
        "pct_move": round(100 * pct_move, 4),
        # add this line so we can reuse the exact step size:
        "step_value_abs": round(step_value, 6),
    }

    # --- Add these inside the result dict block ---
    if mode == "atr":
        rsn["eff_atr"] = round(eff_atr, 6)
        if floor_applied:
            rsn["atr_floor_applied"] = True

    if mode == "atr" and move_atr is not None:
        rsn["move_atr"] = round(move_atr, 4)

    return True, rsn, proposed

def _position_qty_for_submit(pos) -> float:
    # Alpaca positions often expose qty as a string
    raw = getattr(pos, "qty", None)
    q = Decimal(str(raw if raw is not None else "0"))
    q = abs(q)
    if ALLOW_FRACTIONAL:
        return float(q)  # e.g., 2.37
    # whole shares only
    q_int = q.to_integral_value(rounding=ROUND_FLOOR)
    return int(q_int)

def _fallback_atr_from_df(symbol: str) -> float | None:
    # Try a preloaded, cached DF if you have one; otherwise reuse _build_atr_hints sources.
    try:
        # reuse build function sources cheaply; returns dict[str, float]
        hints2 = _build_atr_hints()
        val = hints2.get(symbol.upper())
        return float(val) if val and val > 0 else None
    except Exception:
        return None

def _find_sl_by_qty_or_coid(api, symbol: str, side_exit: str, tp_qty: int | None, coid_prefix: str | None):
    try:
        orders = api.list_orders(status="open", nested=True) or []
    except Exception:
        return None
    sym = str(symbol).upper(); sx = str(side_exit).lower()
    cands = []
    for o in orders:
        try:
            if str(getattr(o, "symbol","")).upper() != sym: 
                continue
            if str(getattr(o, "side","")).lower() != sx: 
                continue
            typ = (getattr(o, "type", None) or getattr(o, "order_type", "") or "").lower()
            stop_px = getattr(o, "stop_price", None)
            looks_like_stop = ("stop" in typ) or (stop_px is not None)
            if not looks_like_stop:
                continue
            oq = getattr(o, "qty", None) or getattr(o, "quantity", None)
            try:
                from decimal import Decimal
                oqi = int(Decimal(str(oq)))
            except Exception:
                oqi = None
            coid = str(getattr(o,"client_order_id","") or "")
            cands.append((o, oqi, coid))
        except Exception:
            continue
    # Prefer COID prefix, then exact qty match, else first candidate
    if coid_prefix:
        for o, oqi, co in cands:
            if co.startswith(coid_prefix): 
                return o
    if tp_qty is not None:
        for o, oqi, _ in cands:
            if oqi == tp_qty: 
                return o
    return cands[0][0] if cands else None


def _find_any_stop_exit(api, symbol: str, side_exit: str):
    """Forgiving fallback: return any open stop-like order for this symbol+exit side."""
    try:
        orders = api.list_orders(status="open", nested=True) or []
    except Exception:
        return None
    sym = str(symbol).upper(); sx = str(side_exit).lower()
    for o in orders:
        try:
            if str(getattr(o, "symbol","")).upper() != sym:
                continue
            if str(getattr(o, "side","")).lower() != sx:
                continue
            otyp = (getattr(o, "type", None) or getattr(o, "order_type","") or "").lower()
            stop_px = getattr(o, "stop_price", None)
            if ("stop" in otyp) or (stop_px is not None):
                return o
        except Exception:
            continue
    return None

def _coid_prefix(o):
    coid = str(getattr(o, "client_order_id", "") or "")
    # heuristic: keep alnum/hyphen up to first colon/space
    return coid.split(":")[0].split(" ")[0] if coid else None

def place_or_amend_sl(api, pos, target_sl_price: float, sl_order=None, side_exit: str | None = None) -> bool:
    """
    Create or amend the protective Stop-Loss (SL) for a position.

    Behavior:
      • If sl_order exists → try replace; on failure, cancel → fall through to create.
      • Caps SL qty to actual available exit qty (never > position qty).
      • If availability appears stale right after a TP change, waits briefly and rechecks.
      • If still no free qty and no SL exists → rebalance TP to free shares (and place SL).
      • One-shot retry on transient 403/availability flicker.

    Extra safety (this version):
      • BEFORE submitting, we also cap 'avail' by any visible LIMIT exit reservations
        (e.g., TP qty) via _sum_exit_limit_remaining(). This avoids oversize requests like
        "requested: full position, available: 0" during brief broker snapshot races.
    """
    # --- basics ---
    try:
        symbol = str(getattr(pos, "symbol", "")).upper()
        PRICE_DEC = int(globals().get("PRICE_DECIMALS", 2))
        direction, side_exit_auto = _position_side_and_exit(pos)  # 'LONG'/'SHORT', 'sell'/'buy'
        if side_exit is None:
            side_exit = side_exit_auto
    except Exception:
        return False

    rounded_sl = _round_stop_for_side(float(target_sl_price), side_exit)

    # Prefer replace to preserve OCO if an SL already exists
    if sl_order and globals().get("REPLACE_ORDER_FIRST", True):
        try:
            api.replace_order(getattr(sl_order, "id", None), stop_price=rounded_sl)
            logging.info("🔒 Replaced SL for %s %s → %0.*f", symbol, direction, PRICE_DEC, rounded_sl)
            return True
        except Exception as e:
            logging.warning("replace_order failed for SL %s: %s — falling back to cancel+submit", symbol, e)
            oid = getattr(sl_order, "id", None)
            try:
                api.cancel_order(oid)
            except Exception as e2:
                logging.warning("Could not cancel existing SL for %s: %s", symbol, e2)
            _cancel_order_and_wait(
                api, oid, symbol, side_exit,
                timeout_s=float(globals().get("CANCEL_ORDER_TIMEOUT_S", 2.0))
            )
            sl_order = None  # allow rebalance branch below

    # --- compute position & initial availability ---
    try:
        pos_qty = _position_qty_for_submit(pos)
    except Exception:
        pos_qty = 0
    if pos_qty <= 0:
        return False

    # Helper: quick poll to let open-orders snapshot reflect the latest TP replace
    def _rechecked_avail(max_wait_s: float = RECHECK_AVAIL_MAX_WAIT_S,
                         poll_s: float = RECHECK_AVAIL_POLL_S) -> int:
        """
        Recompute free exit qty for up to a short timeout after an exit update.
        If ANY exit limit exists but availability returns == pos_qty, treat it as reserved.
        """
        import time
        t0 = time.time()
        last = None
        while True:
            avail_now = _available_exit_qty(api, symbol, side_exit, pos_qty)
            # If some reservation is visible, return immediately
            if avail_now < pos_qty:
                return avail_now

            # If exit limits exist, synthesize reservation from their summed remaining qty
            sum_exit_limits = _sum_exit_limit_remaining(api, symbol, side_exit)
            if sum_exit_limits > 0:
                return max(0, int(pos_qty) - int(sum_exit_limits))

            # No exit limits at all → avail_now is trustworthy
            if last is not None and avail_now == last:
                pass
            last = avail_now

            if time.time() - t0 >= max_wait_s:
                return avail_now
            time.sleep(poll_s)

    # 1) First availability read; if it looks fully free, recheck briefly
    avail = _available_exit_qty(api, symbol, side_exit, pos_qty)
    if avail >= pos_qty:
        avail = _rechecked_avail()  # may synthesize reservations from visible LIMIT exits

    # 🔒 NEW: cap availability by any visible LIMIT reservations (TP, partials), even if API flickers
    reserved_limits = _sum_exit_limit_remaining(api, symbol, side_exit)
    if reserved_limits > 0:
        avail = max(0, int(pos_qty) - int(reserved_limits))

    # If nothing free, try to free shares by shrinking TP (only if we don't already have an SL)
    if avail <= 0:
        logging.warning(
            "No free qty for new SL on %s — TP/SL legs already reserve %s of %s.",
            symbol, pos_qty, pos_qty
        )
        if bool(globals().get("SL_REBALANCE_IF_MISSING", True)) and (sl_order is None):
            desired_sl_qty = int(globals().get(
                "SL_REBALANCE_DESIRED_QTY",
                max(1, int(pos_qty * float(globals().get("SL_REBALANCE_QTY_FRACTION", 0.50))))
            ))
            ok = _rebalance_tp_to_make_room_for_sl(
                api=api,
                symbol=symbol,
                side_exit=side_exit,    # 'sell' for LONGs, 'buy' for SHORTs
                desired_sl_qty=desired_sl_qty,
                sl_price=rounded_sl
            )
            if ok:
                return True
        _debug_dump_open_orders_for_symbol(api, symbol, side_exit)
        return False

    # ✅ cap request to actual availability (and never > pos size)
    req = max(0, min(int(avail), int(pos_qty)))
    if req <= 0:
        _debug_dump_open_orders_for_symbol(api, symbol, side_exit)
        return False

    # Submit capped SL
    try:
        _submit_order_safe(
            api,
            symbol=symbol,
            qty=req,
            side=side_exit,
            type="stop",
            time_in_force="gtc",
            stop_price=rounded_sl,
            reduce_only=True,
        )
        logging.info("🔒 Submitted SL for %s %s → %0.*f (qty=%s)", symbol, direction, PRICE_DEC, rounded_sl, req)
        return True
    except Exception as e:
        # One-shot retry after a fresh availability check
        logging.warning("Failed to submit SL for %s: %s", symbol, e)
        avail2 = _rechecked_avail()  # uses config defaults
        # Re-apply LIMIT reservation cap on the recheck too
        reserved_limits2 = _sum_exit_limit_remaining(api, symbol, side_exit)
        if reserved_limits2 > 0:
            avail2 = max(0, int(pos_qty) - int(reserved_limits2))

        req2 = max(0, min(int(avail2), int(pos_qty)))
        if req2 <= 0:
            logging.error("Retry aborted for %s — still no available qty.", symbol)
            _debug_dump_open_orders_for_symbol(api, symbol, side_exit)
            return False
        try:
            _submit_order_safe(
                api,
                symbol=symbol,
                qty=req2,
                side=side_exit,
                type="stop",
                time_in_force="gtc",
                stop_price=rounded_sl,
                reduce_only=True,
            )
            logging.info("🔒 Submitted SL (retry) for %s %s → %0.*f (qty=%s)", symbol, direction, PRICE_DEC, rounded_sl, req2)
            return True
        except Exception as e2:
            logging.warning("Failed to submit SL (retry) for %s: %s", symbol, e2)

            # One last quick recheck + optional TP→SL rebalance
            avail2 = _rechecked_avail(max_wait_s=0.8, poll_s=0.1)
            reserved_limits2 = _sum_exit_limit_remaining(api, symbol, side_exit)
            if reserved_limits2 > 0:
                avail2 = max(0, int(pos_qty) - int(reserved_limits2))

            req2 = max(0, min(int(avail2), int(pos_qty)))
            if req2 <= 0 and (sl_order is None) and bool(globals().get("SL_REBALANCE_IF_MISSING", True)):
                desired_sl_qty = int(globals().get(
                    "SL_REBALANCE_DESIRED_QTY",
                    max(1, int(pos_qty * float(globals().get("SL_REBALANCE_QTY_FRACTION", 0.50))))
                ))
                ok = _rebalance_tp_to_make_room_for_sl(
                    api=api,
                    symbol=symbol,
                    side_exit=side_exit,
                    desired_sl_qty=desired_sl_qty,
                    sl_price=rounded_sl
                )
                if ok:
                    return True  # helper placed the SL itself

            if req2 <= 0:
                logging.error("Retry aborted for %s — still no available qty.", symbol)
                _debug_dump_open_orders_for_symbol(api, symbol, side_exit)
                return False

            try:
                _submit_order_safe(
                    api,
                    symbol=symbol,
                    qty=req2,
                    side=side_exit,
                    type="stop",
                    time_in_force="gtc",
                    stop_price=rounded_sl,
                    reduce_only=True,
                )
                logging.info("🔒 Submitted SL (retry) for %s %s → %0.*f (qty=%s)", symbol, direction, PRICE_DEC, rounded_sl, req2)
                return True
            except Exception as e3:
                logging.error("Failed to submit SL (retry) for %s: %s", symbol, e3)
                _debug_dump_open_orders_for_symbol(api, symbol, side_exit)
                return False

def place_or_amend_tp(api, pos, new_tp):
    symbol = str(getattr(pos, "symbol", "")).upper()
    direction, side_exit = _position_side_and_exit(pos)  # 'LONG'/'SHORT', 'sell'/'buy'
    PRICE_DEC = int(globals().get("PRICE_DECIMALS", 2))
    rounded_tp = _round_limit_for_side(float(new_tp), side_exit)

    # Find current TP (your finder should use nested=True internally)
    tp_order = _get_symbol_open_tp_order_by_relation(
        api, symbol, side_exit, live_px=_latest_trade_price(api, symbol)
    )

    def _parent_of(o):
        pid = getattr(o, "parent_order_id", None)
        return str(pid) if pid else None

    def _qty_of(o):
        q = getattr(o, "qty", None) or getattr(o, "quantity", None)
        try:
            from decimal import Decimal
            return int(Decimal(str(q)))
        except Exception:
            return None

    def _coid_prefix(o):
        coid = str(getattr(o, "client_order_id", "") or "")
        return coid.split(":")[0].split(" ")[0] if coid else None

    # === No TP found → create one (with remaining qty only) ===
    if not tp_order:
        logging.warning("No open TP order found for %s (%s) — cannot amend TP.", symbol, direction)
        if not ALLOW_TP_CREATE_IF_MISSING:
            return False, None, None, None

        pos_qty = _position_qty_for_submit(pos)
        if pos_qty <= 0:
            logging.error("Cannot create TP for %s: position qty <= 0 (raw=%r)", symbol, getattr(pos, "qty", None))
            return False, None, None, None

        avail = _available_exit_qty(api, symbol, side_exit, pos_qty)
        if avail <= 0:
            logging.warning("No free qty for new TP on %s — existing exits consume the position.", symbol)
            _debug_dump_open_orders_for_symbol(api, symbol, side_exit)  # DEBUG aid
            return False, None, None, None

        try:
            _submit_order_safe(
                api,
                symbol=symbol,
                qty=avail,
                side=side_exit,
                type="limit",
                time_in_force="gtc",
                limit_price=rounded_tp,
                reduce_only=True,  # will be auto-dropped by helper if SDK doesn’t support it
            )
            logging.info("🆕 Created TP for %s %s → %0.*f (qty=%s)", symbol, direction, PRICE_DEC, rounded_tp, avail)
            try:
                import time
                time.sleep(0.2)  # small settle to avoid immediate snapshot races
            except Exception:
                pass

            # We don't have the order object here; parent/coid unknown
            return True, None, avail, None
        except Exception as e:
            logging.error("Failed to create TP for %s: %s", symbol, e)
            return False, None, None, None

    # === TP exists → amend it (prefer replace to preserve OCO) ===
    pos_qty = _position_qty_for_submit(pos)
    if pos_qty <= 0:
        logging.error("Failed to amend TP for %s: position qty <= 0 (raw=%r)", symbol, getattr(pos, "qty", None))
        return False, _parent_of(tp_order), _qty_of(tp_order), _coid_prefix(tp_order)

    if REPLACE_ORDER_FIRST:
        try:
            api.replace_order(getattr(tp_order, "id", None), limit_price=rounded_tp)
            logging.info("⬆️ Replaced TP for %s %s → %0.*f", symbol, direction, PRICE_DEC, rounded_tp)
            try:
                import time
                time.sleep(0.2)
            except Exception:
                pass
            return True, _parent_of(tp_order), _qty_of(tp_order), _coid_prefix(tp_order)
        except Exception as e:
            logging.warning("replace_order failed for %s: %s — falling back to cancel+submit", symbol, e)

    # Cancel existing TP then submit a fresh one using remaining qty
    try:
        api.cancel_order(getattr(tp_order, "id", None))
    except Exception as e:
        logging.warning("Could not cancel existing TP for %s: %s", symbol, e)
        return False, _parent_of(tp_order), _qty_of(tp_order), _coid_prefix(tp_order)

    avail = _available_exit_qty(api, symbol, side_exit, pos_qty)
    if avail <= 0:
        logging.warning("No free qty for new TP on %s — existing exits consume the position.", symbol)
        _debug_dump_open_orders_for_symbol(api, symbol, side_exit)  # DEBUG aid
        return False, _parent_of(tp_order), _qty_of(tp_order), _coid_prefix(tp_order)

    try:
        _submit_order_safe(
            api,
            symbol=symbol,
            qty=avail,
            side=side_exit,
            type="limit",
            time_in_force="gtc",
            limit_price=rounded_tp,
            reduce_only=True,
        )
        logging.info("⬆️ Extended TP (new order) for %s %s → %0.*f (qty=%s)", symbol, direction, PRICE_DEC, rounded_tp, avail)
        try:
            import time
            time.sleep(0.2)
        except Exception:
            pass
        return True, _parent_of(tp_order), _qty_of(tp_order), _coid_prefix(tp_order)
    except Exception as e:
        logging.error("Failed to submit new TP for %s: %s", symbol, e)
        return False, _parent_of(tp_order), _qty_of(tp_order), _coid_prefix(tp_order)

def _latest_run_dir(root: str | Path) -> Path:
    root = Path(root)
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)

def _read_csv_safe(p: Path):
    try:
        import pandas as pd
        return pd.read_csv(p)
    except Exception:
        return None

def _build_atr_hints(n_runs: int = 5) -> dict[str, float]:
    hints: dict[str, float] = {}
    root = Path(predict_path)
    runs = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")],
        key=lambda p: p.stat().st_mtime, reverse=True
    )[:n_runs]

    tagged_frames = []
    candidates = ("1.Summary_signals_pre_hedge.csv", "signals_top_conf.csv", "ALL_predictions.csv")

    for run in runs:
        mtime = run.stat().st_mtime
        for name in candidates:
            p = run / name
            if not p.exists():
                continue
            df = _read_csv_safe(p)
            if df is None or df.empty:
                continue
            # create a new frame with the extra column in one go
            tagged_frames.append(
                pd.concat([df, pd.DataFrame({"__run_mtime__": [mtime]*len(df)})], axis=1, copy=False)
            )

    if not tagged_frames:
        return hints

    sig = pd.concat(tagged_frames, ignore_index=True)

    # normalize columns (ticker/symbol, atr/atr_14/atr14, date)
    colmap = {c.lower(): c for c in sig.columns}
    tcol = colmap.get("ticker") or colmap.get("symbol")
    acol = colmap.get("atr") or colmap.get("atr_14") or colmap.get("atr14")
    dcol = colmap.get("date")
    if not (tcol and acol):
        return hints

    df = sig[[tcol, acol] + ([dcol] if dcol else []) + ["__run_mtime__"]].copy()
    df[tcol] = df[tcol].astype(str).str.upper().str.strip()
    df[acol] = pd.to_numeric(df[acol], errors="coerce")
    if dcol:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[tcol, acol])
    df = df[df[acol] > 0]

    # latest by (date, then run_mtime) — keep your existing logic
    if dcol:
        idx = df.groupby(tcol)[[dcol, "__run_mtime__"]].transform("max")
        df = df[(df[dcol].eq(idx[dcol])) & (df["__run_mtime__"].eq(idx["__run_mtime__"]))] \
               .drop_duplicates(subset=[tcol], keep="last")
    else:
        df = df.sort_values("__run_mtime__", ascending=False).drop_duplicates(subset=[tcol], keep="first")

    for _, r in df.iterrows():
        hints[r[tcol]] = float(r[acol])

    logging.debug("ATR hints ready for %d symbols across last %d runs", len(hints), len(runs))
    return hints

def _alpaca_client() -> REST:
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise RuntimeError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY.")
    return REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)

def _parse_ts(ts) -> datetime:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc)

def _latest_trade_price(api: REST, symbol: str) -> float | None:
    try:
        trade = api.get_latest_trade(symbol)
        return float(trade.price)
    except Exception as e:
        logging.warning("Could not fetch latest price for %s: %s", symbol, e)
        return None

# ---------------- Core logic ----------------
def _position_direction(pos) -> str:
    """Return 'LONG' or 'SHORT'."""
    qty = float(getattr(pos, "qty", 0) or 0)
    side = str(getattr(pos, "side", "")).lower()
    return "LONG" if side == "long" or qty > 0 else "SHORT"

def _get_symbol_open_tp_order(api: REST, symbol: str, side_for_exit: str):
    """
    Find active TP exit leg for symbol.
    For LONG positions TP leg is SELL limit; for SHORT positions BUY limit.
    """
    try:
        open_orders = api.list_orders(status="open", nested=True)
    except Exception:
        open_orders = []
    side_for_exit = side_for_exit.lower()
    candidates = []
    for o in open_orders or []:
        sym  = getattr(o, "symbol", "")
        otyp = (getattr(o, "type", None) or getattr(o, "order_type", "") or "").lower()
        oside= str(getattr(o, "side", "")).lower()
        lp   = getattr(o, "limit_price", None)
        if sym == symbol and otyp == "limit" and oside == side_for_exit and lp not in (None, ""):
            candidates.append(o)
    if not candidates:
        return None
    # Long→sell: pick HIGHEST; Short→buy: pick LOWEST
    if side_for_exit == "sell":
        return max(candidates, key=lambda x: float(getattr(x, "limit_price", 0) or 0))
    else:
        return min(candidates, key=lambda x: float(getattr(x, "limit_price", 0) or 0))

def _extend_tp_once(api: REST, pos, live_price: float,
                    mode: str,
                    trigger_pct: float, step_pct: float,
                    trigger_atr: float, step_atr: float,
                    max_mult: float,
                    atr_hint: float | None = None) -> bool:
    """Raise/Lower TP if the move from entry meets the trigger. Prefer replace to preserve OCO."""
    # --- basics ---
    symbol = str(getattr(pos, "symbol", "")).upper()
    qty = float(getattr(pos, "qty", 0) or 0.0)
    direction = "LONG" if str(getattr(pos, "side", "")).lower() == "long" or qty > 0 else "SHORT"
    side_exit = "sell" if direction == "LONG" else "buy"

    try:
        entry_px = float(getattr(pos, "avg_entry_price", 0) or 0.0)
    except Exception:
        entry_px = 0.0
    if entry_px <= 0 or live_price is None or live_price <= 0:
        return False

    # --- favorable move from entry ---
    delta = (live_price - entry_px) if direction == "LONG" else (entry_px - live_price)
    if delta <= 0:
        return False

    # --- trigger check (ATR or % ) ---
    triggered = False
    if mode.lower() == "atr" and atr_hint not in (None, ""):
        try:
            atr = float(atr_hint)
        except Exception:
            atr = None
        if atr and atr > 0 and delta >= float(trigger_atr) * atr:
            triggered = True
    else:
        pct_move = delta / entry_px
        if pct_move >= float(trigger_pct):
            triggered = True
    if not triggered:
        return False

    # --- find current TP leg ---
    tp_order = _get_symbol_open_tp_order(api, symbol, side_for_exit=side_exit)
    if not tp_order:
        return False

    try:
        cur_tp = float(getattr(tp_order, "limit_price", 0) or 0.0)
    except Exception:
        cur_tp = 0.0
    if cur_tp <= 0:
        return False

    # --- propose new TP (forward only) ---
    if mode.lower() == "atr" and atr_hint not in (None, ""):
        step = float(step_atr) * float(atr_hint)
        new_tp = (cur_tp + step) if direction == "LONG" else (cur_tp - step)
    else:
        sp = float(step_pct)
        new_tp = (cur_tp * (1.0 + sp)) if direction == "LONG" else (cur_tp * (1.0 - sp))

    # never move TP backwards
    if (direction == "LONG" and new_tp <= cur_tp) or (direction == "SHORT" and new_tp >= cur_tp):
        return False

    # soft cap vs distance from entry (prevents runaway targets)
    original_tp_dist = abs(cur_tp - entry_px)
    max_tp_dist = float(max_mult) * original_tp_dist
    proposed_dist = abs(new_tp - entry_px)
    if proposed_dist > max_tp_dist:
        new_tp = entry_px + max_tp_dist if direction == "LONG" else entry_px - max_tp_dist

    # broker sanity: keep at least a tick away from current price and correct side
    min_tick = 10 ** (-int(PRICE_DECIMALS))
    if direction == "LONG":
        new_tp = max(new_tp, max(cur_tp, live_price) + min_tick)
    else:
        new_tp = min(new_tp, min(cur_tp, live_price) - min_tick)

    # --- NEW: side-aware spread buffer (keeps TP a few ticks from inside quote)
    try:
        _SPREAD_TICKS = SPREAD_BUFFER_TICKS  # if defined at top via config
    except NameError:
        _SPREAD_TICKS = 2  # fallback
    if direction == "LONG":
        new_tp = max(new_tp, live_price + _SPREAD_TICKS * min_tick)
    else:
        new_tp = min(new_tp, live_price - _SPREAD_TICKS * min_tick)

    # --- NEW: skip tiny adjustments (meaningful step check)
    try:
        _ABS_MIN = MIN_ABS_STEP
        _REL_MIN = MIN_REL_STEP
    except NameError:
        _ABS_MIN = 0.05     # $0.05 absolute
        _REL_MIN = 0.001    # 0.10% relative

    # require either absolute or relative minimum change from current TP

    if abs(new_tp - cur_tp) < max(_ABS_MIN, _REL_MIN * abs(cur_tp)):
        return False, {
            "reason": "tp_step_too_small",
            "cur_tp": f"{cur_tp:.4f}",
            "new_tp": f"{new_tp:.4f}",
            "min_abs": f"{_ABS_MIN:.4f}",
            "min_rel": f"{_REL_MIN:.4%}",
        }, None

    new_tp = round(float(new_tp), int(PRICE_DECIMALS))

    # Prefer REPLACE (keeps OCO) if enabled
    if REPLACE_ORDER_FIRST:
        try:
            api.replace_order(getattr(tp_order, "id", None), limit_price=new_tp)
            logging.info("⬆️ Replaced TP for %s %s → %0.*f", symbol, direction, int(PRICE_DECIMALS), new_tp)
            return True
        except Exception as e:
            logging.warning("replace_order failed for %s: %s — falling back to cancel+submit", symbol, e)

    # Fallback (or if REPLACE_ORDER_FIRST=False): cancel & submit new TP
    try:
        api.cancel_order(getattr(tp_order, "id", None))
    except Exception as e:
        logging.warning("Could not cancel existing TP for %s: %s", symbol, e)
        return False

    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side_exit,
            type="limit",
            time_in_force="gtc",
            limit_price=new_tp
        )
        logging.info("⬆️ Extended TP (new order) for %s %s → %0.*f", symbol, direction, int(PRICE_DECIMALS), new_tp)
        return True
    except Exception as e:
        logging.error("Failed to submit new TP for %s: %s", symbol, e)
        return False

def _current_tp_price(api, symbol: str, side_exit: str):
    tp = _get_symbol_open_tp_order(api, symbol, side_for_exit=side_exit)
    if not tp:
        return None
    try:
        # Alpaca orders commonly expose price on .limit_price (string/Decimal)
        return float(getattr(tp, "limit_price", None) or getattr(tp, "limit", None))
    except Exception:
        return None

def run_tp_extender_pass(api: REST, mode: str, trigger_pct: float, step_pct: float,
                         trigger_atr: float, step_atr: float, max_mult: float):
    # --- ⏰ Market-close safety guard (skip updates after 15:55 ET) ---
    from datetime import datetime, time
    from zoneinfo import ZoneInfo

    now_et = datetime.now(ZoneInfo("US/Eastern")).time()
    if now_et >= time(15, 55):
        logging.info("⏳ Market near close — skipping TP/SL updates for this pass.")
        return

    # Master switch + dry-run guard
    if not bool(globals().get("TP_EXTEND_ENABLE", True)):
        logging.info("⏭️ TP_EXTEND_ENABLE=False — skipping extender pass.")
        return
    if DRY_RUN:
        logging.info("⏭️ DRY_RUN=True — no live orders to adjust; skipping extender pass.")
        return

    # Mode normalization
    mode = (mode or "percent").lower().strip()
    if mode not in ("atr", "percent"):
        logging.warning("Invalid mode=%r; defaulting to 'percent'.", mode)
        mode = "percent"

    # Build ATR hints once for the pass
    atr_hints = _build_atr_hints()
    _fallback_cache = None

    # Get positions
    try:
        positions = api.list_positions()
    except Exception as e:
        logging.warning("Cannot list positions: %s", e)
        return

    # 🩶 Optional ghost-position filter (skip zero-qty or zero-value entries)
    open_positions = []
    if bool(globals().get("SKIP_GHOST_POSITIONS", True)):
        for p in positions or []:
            try:
                qty = float(getattr(p, "qty", 0) or 0.0)
                mv  = float(getattr(p, "market_value", 0) or 0.0)
                if abs(qty) < 1e-6 or abs(mv) < 1e-4:
                    logging.debug("🩶 Skipping ghost position %s (qty=%.6f, mv=%.4f)",
                                  getattr(p, "symbol", "?"), qty, mv)
                    continue
                open_positions.append(p)
            except Exception:
                continue
    else:
        open_positions = positions or []

    if not open_positions:
        logging.info("🛌 No real open positions — sleeping until next cycle.")
        return

    # Header log
    syms = [str(getattr(p, "symbol", "?")).upper() for p in open_positions][:LOG_TICKERS_SAMPLES]
    tail = " …" if len(open_positions) > LOG_TICKERS_SAMPLES else ""
    logging.info("🔍 Scanning %d open positions: %s%s", len(open_positions), ", ".join(syms), tail)

    extended = 0
    from time import time as _now

    for pos in open_positions:
        symbol = str(getattr(pos, "symbol", "")).upper()

        live_px = _latest_trade_price(api, symbol)
        if not live_px or live_px <= 0:
            reason_log(symbol, reason="no_live_price")
            continue

        # 1) Resolve ATR according to ATR_SOURCE (fresh → folder → yahoo)
        atr_val, atr_src = _resolve_atr_for_symbol(api, symbol, atr_hints)
        if mode == "atr" and not (atr_val and atr_val > 0):
            reason_log(symbol, reason="no_atr_hint_in_atr_mode", atr_source="none")
        elif atr_src != "none":
            logging.debug("ATR %s used for %s: %.4f (len=%d)", atr_src, symbol, atr_val or -1, ATR_LENGTH)

        # --- Common offset for baseline exits (prefer ATR; else %, else tiny fallback)
        eff_atr = None
        if atr_val and atr_val > 0:
            eff_atr = max(float(atr_val), float(globals().get("ATR_DOLLAR_FLOOR", 0.0)))
        base_pct = globals().get("EXTENDER_BASE_SL_PCT", None)
        if base_pct not in (None, "", 0):
            offset = float(base_pct) * live_px
        elif eff_atr:
            offset = float(globals().get("EXTENDER_BASE_SL_ATR_MULT", 0.35)) * eff_atr
        else:
            offset = 0.01 * live_px  # 1% fallback

        direction, side_exit = _position_side_and_exit(pos)  # 'LONG'/'SHORT', 'sell'/'buy'
        tp_price = (live_px + offset) if direction == "LONG" else (live_px - offset)
        sl_price = (live_px - offset) if direction == "LONG" else (live_px + offset)

        # 2) Baseline exit ensure — runs EVERY pass (even if TP hasn't ratcheted yet)
        try:
            if bool(globals().get("EXTENDER_BASE_SL_ENABLE", True)):
                sl_order = _get_symbol_open_sl_order_by_relation(api, symbol, side_exit, live_px)
                tp_order = _get_symbol_open_tp_order_by_relation(api, symbol, side_exit, live_px)

                pos_qty = _position_qty_for_submit(pos)
                avail0 = _available_exit_qty(api, symbol, side_exit, pos_qty) if pos_qty > 0 else 0

                # a) Both exits missing → seed small TP first, then SL
                if (tp_order is None) and (sl_order is None) and pos_qty > 0:
                    base_sl_frac = float(globals().get("BASELINE_SL_FRACTION", 0.60))
                    base_sl_frac = min(max(base_sl_frac, 0.10), 0.90)
                    tp_frac = 1.0 - base_sl_frac

                    tp_req = max(1, int(tp_frac * pos_qty)) if avail0 > 0 else 0
                    tp_req = min(tp_req, int(avail0))
                    if tp_req > 0:
                        try:
                            _submit_order_safe(
                                api,
                                symbol=symbol,
                                qty=tp_req,
                                side=side_exit,
                                type="limit",
                                time_in_force="gtc",
                                limit_price=_round_limit_for_side(tp_price, side_exit),
                                reduce_only=True,
                            )
                            logging.info("🧩 Baseline TP seeded for %s %s → %.2f (qty=%d)",
                                         symbol, direction, tp_price, tp_req)
                            try:
                                import time; time.sleep(0.2)
                            except Exception:
                                pass
                        except Exception as e:
                            logging.warning("Baseline TP seed failed for %s: %s", symbol, e)

                    # now create SL with remaining qty (capped & rebalance-safe)
                    place_or_amend_sl(api, pos, target_sl_price=sl_price, sl_order=None, side_exit=side_exit)

                # b) SL exists but TP missing → shrink SL to free qty and create TP
                elif (tp_order is None) and (sl_order is not None) and pos_qty > 0:
                    try:
                        desired_tp_qty = max(1, int((1.0 - float(globals().get("BASELINE_SL_FRACTION", 0.60))) * max(1, pos_qty)))
                    except Exception:
                        desired_tp_qty = max(1, int(0.40 * max(1, pos_qty)))

                    _rebalance_sl_to_make_room_for_tp(
                        api=api,
                        symbol=symbol,
                        side_exit=side_exit,
                        desired_tp_qty=desired_tp_qty,
                        tp_price=_round_limit_for_side(tp_price, side_exit),
                    )
                    # Ensure SL still exists after the rebalance
                    try:
                        sl_check = _get_symbol_open_sl_order_by_relation(api, symbol, side_exit, live_px)
                        if not sl_check:
                            place_or_amend_sl(api, pos, target_sl_price=sl_price, sl_order=None, side_exit=side_exit)
                    except Exception as e:
                        logging.warning("Post-rebalance SL ensure failed for %s: %s", symbol, e)

                # c) TP exists but SL missing → just create SL
                elif (tp_order is not None) and (sl_order is None):
                    place_or_amend_sl(api, pos, target_sl_price=sl_price, sl_order=None, side_exit=side_exit)

                # d) Single-share policy — always end the pass with BOTH exits present
                elif pos_qty == 1:
                    tp_missing = (tp_order is None)
                    sl_missing = (sl_order is None)

                    # (i) If BOTH missing, use your baseline seeding (TP first, then SL)
                    if tp_missing and sl_missing:
                        base_sl_frac = float(globals().get("BASELINE_SL_FRACTION", 0.60))
                        base_sl_frac = min(max(base_sl_frac, 0.10), 0.90)
                        tp_px = _round_limit_for_side(tp_price, side_exit)

                        # seed TP for 1 share (we only have qty=1)
                        try:
                            _submit_order_safe(
                                api,
                                symbol=symbol,
                                qty=1,
                                side=side_exit,
                                type="limit",
                                time_in_force="gtc",
                                limit_price=tp_px,
                                reduce_only=True,
                            )
                            logging.info("🧩 Baseline TP seeded for %s (qty=1) @ %.2f", symbol, tp_px)
                        except Exception as e:
                            logging.warning("Baseline TP seed failed for %s: %s", symbol, e)

                        # then create SL
                        place_or_amend_sl(api, pos, target_sl_price=sl_price, sl_order=None, side_exit=side_exit)

                    # (ii) If EXACTLY ONE leg exists, cancel it to free qty and repost BOTH
                    elif tp_missing ^ sl_missing:
                        try:
                            lone = sl_order if tp_missing else tp_order
                            lone_id = getattr(lone, "id", None) if lone else None
                            if lone_id:
                                try:
                                    _cancel_order_safe(api, symbol, lone_id)
                                except NameError:
                                    # fallback if helper not present
                                    try:
                                        api.cancel_order(lone_id)
                                    except Exception:
                                        pass
                                logging.info("♻️  Cancelled lone %s leg for %s to free qty", "SL" if tp_missing else "TP", symbol)
                        except Exception as e:
                            logging.warning("Could not cancel lone leg for %s: %s", symbol, e)

                        # repost both legs fresh (SL first for safety, then TP)
                        place_or_amend_sl(api, pos, target_sl_price=sl_price, sl_order=None, side_exit=side_exit)
                        try:
                            _submit_order_safe(
                                api,
                                symbol=symbol,
                                qty=1,
                                side=side_exit,
                                type="limit",
                                time_in_force="gtc",
                                limit_price=_round_limit_for_side(tp_price, side_exit),
                                reduce_only=True,
                            )
                            logging.info("🆕 Reposted TP for %s (qty=1)", symbol)
                        except Exception as e:
                            logging.warning("Repost TP failed for %s: %s", symbol, e)

                    # (iii) If only TP is missing but we prefer not to cancel, try a tiny TP if free qty exists
                    elif tp_missing:
                        # Always ensure SL present
                        if not sl_order:
                            place_or_amend_sl(api, pos, target_sl_price=sl_price, sl_order=None, side_exit=side_exit)

                        if bool(globals().get("ALLOW_TP_CREATE_IF_MISSING", True)):
                            try:
                                tp_order_check = _get_symbol_open_tp_order_by_relation(api, symbol, side_exit, live_px)
                            except Exception:
                                tp_order_check = None

                            if not tp_order_check:
                                avail = _available_exit_qty(api, symbol, side_exit, pos_qty=1)
                                if avail > 0:
                                    _submit_order_safe(
                                        api,
                                        symbol=symbol,
                                        qty=min(1, int(avail)),
                                        side=side_exit,
                                        type="limit",
                                        time_in_force="gtc",
                                        limit_price=_round_limit_for_side(tp_price, side_exit),
                                        reduce_only=True,
                                    )
                                    logging.info("🆕 Seeded tiny TP for %s (qty=1)", symbol)
                                else:
                                    logging.info("ℹ️ Tiny-TP skipped for %s (qty=1) — no free exit qty available right now.", symbol)
        except Exception as e:
            logging.warning("Baseline exit ensure failed for %s: %s", symbol, e)
        # 3) Evaluate TP extension
        direction, side_exit = _position_side_and_exit(pos)   # re-evaluate
        cur_tp_px = _current_tp_price(api, symbol, side_exit)

        ok, rsn, new_tp = evaluate_tp_extension(
            pos, live_px, atr_val,
            mode,
            trigger_pct, step_pct, trigger_atr, step_atr, max_mult,
            cur_tp=cur_tp_px
        )
        reason_log(symbol, **rsn)

        if ok and new_tp is not None:
            last = _last_extend_at.get(symbol, 0.0)
            elapsed = _now() - last
            if elapsed < COOLDOWN_MIN * 60:
                reason_log(symbol, reason="cooldown_skip", since_sec=int(elapsed), cooldown_min=COOLDOWN_MIN)
                continue

            if TP_EXTEND_MAX_REPLACES_PER_SYMBOL and TP_EXTEND_MAX_REPLACES_PER_SYMBOL > 0:
                cnt = _replace_counts.get(symbol, 0)
                if cnt >= TP_EXTEND_MAX_REPLACES_PER_SYMBOL:
                    reason_log(symbol, reason="max_replaces_reached",
                               count=cnt, cap=TP_EXTEND_MAX_REPLACES_PER_SYMBOL)
                    continue

            # place_or_amend_tp may return bool OR tuple; handle both
            _res = place_or_amend_tp(api, pos, new_tp)
            if isinstance(_res, tuple):
                if len(_res) == 4:
                    ok_tp, parent_id, tp_qty, tp_coid_prefix = _res
                else:
                    ok_tp, parent_id = _res; tp_qty = tp_coid_prefix = None
            else:
                ok_tp, parent_id = bool(_res), None; tp_qty = tp_coid_prefix = None

            if not DRY_RUN and ok_tp:
                _last_extend_at[symbol] = _now()
                reason_log(symbol, tp_parent_id=parent_id, tp_qty_hint=tp_qty, tp_coid_prefix=tp_coid_prefix)
                if TP_EXTEND_MAX_REPLACES_PER_SYMBOL and TP_EXTEND_MAX_REPLACES_PER_SYMBOL > 0:
                    _replace_counts[symbol] = _replace_counts.get(symbol, 0) + 1
                extended += 1

                # 4) Tighten SL by the same step now that TP moved
                try:
                    be_lock = globals().get("ATR_BE_LOCK", None)  # optional breakeven guard
                except Exception:
                    be_lock = None

                tighten_stop_loss_to_match_step(
                    api=api,
                    pos=pos,
                    live_px=live_px,
                    atr_val=atr_val,
                    mode=mode,
                    step_pct=float(step_pct),
                    step_atr=float(step_atr),
                    cur_tp_step_value=(rsn.get("step_value_abs") if isinstance(rsn, dict) else None),
                    be_lock_atr_mult=(float(be_lock) if be_lock not in (None, "", 0) else None),
                    tp_parent_id=parent_id,
                    tp_qty=tp_qty,
                    tp_coid_prefix=tp_coid_prefix,
                )

                # 5) Post-TP ensure (if still no SL for any reason, lay one down using ATR buffer)
                try:
                    sl_order = _get_symbol_open_sl_order_by_relation(api, symbol, side_exit, live_px)
                    if not sl_order:
                        if eff_atr:
                            offset2 = float(globals().get("ATR_TRIGGER_BUFFER_MULT", 0.30)) * eff_atr
                            target_sl_price = (live_px - offset2) if direction == "LONG" else (live_px + offset2)
                        else:
                            target_sl_price = live_px * (0.99 if direction == "LONG" else 1.01)
                        place_or_amend_sl(api, pos, target_sl_price=target_sl_price, sl_order=None, side_exit=side_exit)
                except Exception as e:
                    logging.warning("SL ensure (post-TP) failed for %s: %s", symbol, e)

        # 6) 🔐 HARD FAILSAFE — always finish the cycle with a protective SL
        try:
            if bool(globals().get("SL_HARD_FAILSAFE", True)):
                sl_now = _get_symbol_open_sl_order_by_relation(api, symbol, side_exit, live_px)
                if not sl_now:
                    # Prefer ATR-based offset; fall back to 1% of price
                    if eff_atr:
                        off = float(globals().get("EXTENDER_BASE_SL_ATR_MULT", 0.35)) * eff_atr
                    else:
                        off = 0.01 * live_px
                    fail_safe = (live_px - off) if direction == "LONG" else (live_px + off)

                    # 6a) First attempt: normal SL create (will cap to available & may rebalance)
                    fail_safe = _enforce_min_sl_gap(live_px, fail_safe, direction, pct=float(globals().get("MIN_SL_GAP_PCT", 0.002)))
                    ok_fs = place_or_amend_sl(
                        api, pos, target_sl_price=fail_safe, sl_order=None, side_exit=side_exit
                    )

                    # 6b) If still missing (e.g., availability flicker), try one explicit TP→SL rebalance
                    if not ok_fs:
                        try:
                            pos_qty = _position_qty_for_submit(pos)
                        except Exception:
                            pos_qty = 0
                        desired_sl_qty = int(globals().get(
                            "SL_REBALANCE_DESIRED_QTY",
                            max(1, int(pos_qty * float(globals().get("SL_REBALANCE_QTY_FRACTION", 0.50))))
                        ))
                        ok_rb = _rebalance_tp_to_make_room_for_sl(
                            api=api,
                            symbol=symbol,
                            side_exit=side_exit,
                            desired_sl_qty=desired_sl_qty,
                            sl_price=_round_stop_for_side(fail_safe, side_exit),
                        )
                        ok_fs = bool(ok_fs or ok_rb)
                    # 6c) Re-verify on broker, then log real outcome (with qty/price if found)
                    sl_verify = _get_symbol_open_sl_order_by_relation(api, symbol, side_exit, live_px)
                    ok_fs = bool(sl_verify)

                    # 6d) Track repeat failsafe hits (early warning)
                    global _failsafe_hits
                    if ok_fs:
                        _failsafe_hits[symbol] = 0
                        try:
                            qv = getattr(sl_verify, "qty", None) or getattr(sl_verify, "remaining_qty", None)
                            pv = getattr(sl_verify, "stop_price", None) or getattr(sl_verify, "stop", None)
                            pv = float(pv) if pv is not None else float(fail_safe)
                            logging.info("🛡️ Hard-failsafe SL ACTIVE for %s qty=%s @ %.2f", symbol, qv, pv)
                        except Exception:
                            logging.info("🛡️ Hard-failsafe SL ACTIVE for %s @ %.2f", symbol, float(fail_safe))
                    else:
                        _failsafe_hits[symbol] = _failsafe_hits.get(symbol, 0) + 1
                        hits = _failsafe_hits[symbol]
                        logging.warning("🛡️ Hard-failsafe could not place SL for %s (attempt %d).", symbol, hits)
                        if hits >= int(globals().get("FAILSAFE_ALERT_AFTER", 3)):
                            logging.error("🚨 %s required hard-failsafe %d consecutive cycles — investigate TP/SL reserves.", symbol, hits)
                        _debug_dump_open_orders_for_symbol(api, symbol, side_exit)

        except Exception as e:
            logging.warning("Hard-failsafe SL attempt failed for %s: %s", symbol, e)
    # 🔎 Post-pass audit: every open position must have a visible SL
    try:
        price_cache = {}
        audit_missing = []
        for p in open_positions:
            sym = str(getattr(p, "symbol", "")).upper()
            dirn, side_exit2 = _position_side_and_exit(p)
            px = price_cache.get(sym)
            if px is None:
                px = _latest_trade_price(api, sym)
                price_cache[sym] = px
            sl = _get_symbol_open_sl_order_by_relation(api, sym, side_exit2, px)
            if not sl:
                audit_missing.append(sym)
        if audit_missing:
            logging.critical("🚨 Audit: SL still missing for %d symbols after pass: %s — will retry next cycle.",
                             len(audit_missing), ", ".join(audit_missing))
    except Exception as _e_audit:
        logging.warning("Audit step failed: %s", _e_audit)

    logging.info("📈 Cycle result: extended=%d, skipped=%d",
                 extended, max(0, len(open_positions) - extended))

def monitor_positions_loop(api: REST,
                           mode: str,
                           trigger_pct: float, step_pct: float,
                           trigger_atr: float, step_atr: float,
                           max_mult: float,
                           interval_sec: int,
                           max_minutes: int | None,
                           stop_at_close: bool):
    """Loop to keep extending TPs at a fixed cadence, with clear pass-by-pass logs."""
    if DRY_RUN:
        logging.info("⏭️ DRY_RUN=True — monitor loop will not adjust live orders.")
        return

    t0 = time.time()
    did_open_hc = False
    while True:
        # Snapshot positions BEFORE the pass so we can log what we’re scanning
        try:
            positions = api.list_positions()
        except Exception as e:
            positions = []
            logging.warning("Cannot list positions: %s", e)

        n = len(positions or [])
        if n == 0:
            logging.info("🛌 No open positions — sleeping %ss", int(interval_sec))
        else:
            # Show up to 10 tickers for quick visibility; “…” if more
            syms = [str(getattr(p, "symbol", "?")).upper() for p in positions][:LOG_TICKERS_SAMPLES]
            tail = " …" if n > LOG_TICKERS_SAMPLES else ""

        # --- Run healthcheck shortly after market opens
        if HEALTHCHECK_AT_OPEN:
            try:
                clk = api.get_clock()
                if bool(getattr(clk, "is_open", False)) and not did_open_hc:
                    # within first N minutes after open?
                    opents = _parse_ts(getattr(clk, "next_open", None))  # when previous close set next_open to today’s open
                    nowts  = datetime.now(timezone.utc)
                    # some SDKs expose 'next_open' as NEXT day during open hours; fall back to do-once
                    within = True
                    if opents and opents.tzinfo:
                        within = (0 <= (nowts - opents).total_seconds() <= HEALTHCHECK_OPEN_GRACE_MIN*60)
                    if within:
                        run_exits_healthcheck(api, label="open")
                        did_open_hc = True
            except Exception:
                pass

        # Run one extender pass (this will try to raise TPs where triggers fire)
        try:
            run_tp_extender_pass(
                api,
                mode=mode,
                trigger_pct=trigger_pct, step_pct=step_pct,
                trigger_atr=trigger_atr, step_atr=step_atr,
                max_mult=max_mult
            )
        except Exception as e:
            logging.warning("TP extender pass error: %s", e)

        # time-based stop
        if max_minutes and (time.time() - t0) >= max_minutes * 60:
            logging.info("🛑 Reached max_minutes=%s — exiting monitor loop.", max_minutes)
            break

        # optional: stop at close
        if stop_at_close:
            try:
                clock = api.get_clock()
                if not bool(getattr(clock, "is_open", False)):
                    if HEALTHCHECK_AT_CLOSE:
                        run_exits_healthcheck(api, label="close")
                    logging.info("🔔 Market closed — stopping monitor loop.")
                    break
            except Exception:
                break

        logging.info("🕑 Sleeping %s sec…", int(interval_sec))
        time.sleep(int(interval_sec))

# --- precedence + helpers ---
PREFER_CONFIG = os.getenv("QML_PREFER_CONFIG", "1") == "1"

def pick(arg_val, cfg_val):
    """Generic picker: if preferring config, ignore CLI when CLI provided (unless arg is None)."""
    return cfg_val if (PREFER_CONFIG and arg_val is not None) else (arg_val if arg_val is not None else cfg_val)

# --- precedence + helpers (near your arg/config merge) ---

def _normalize_mode(v):
    if v is None: return None
    s = str(v).strip().lower()
    if s in ("atr","a"): return "atr"
    if s in ("percent","pct","p"): return "percent"
    raise ValueError(f"Invalid mode: {v!r}")

def pick_mode(arg_mode, cfg_mode, prof):
    am = _normalize_mode(arg_mode)
    cm = _normalize_mode(cfg_mode)
    pm = _normalize_mode((prof or {}).get("TP_EXTEND_MODE"))
    return am or pm or cm or "percent"

def main():
    raw_profile = getattr(CFG, "TP_EXTEND_PROFILES", {}).get(getattr(CFG, "TP_EXTEND_PROFILE", ""), {})
    profile = {k: v for k, v in raw_profile.items() if k.lower() not in ("mode", "tp_extend_mode")}

    # 1) Parser
    parser = argparse.ArgumentParser(description="QuantML — TP Extender (standalone)")
    parser.add_argument("--mode", choices=["atr", "percent"],
                        help="Extension mode (atr or percent). CLI overrides config/profile.")
    parser.add_argument("--trigger-pct", type=float, default=None)
    parser.add_argument("--step-pct",    type=float, default=None)
    parser.add_argument("--trigger-atr", type=float, default=None)
    parser.add_argument("--step-atr",    type=float, default=None)
    parser.add_argument("--max-mult",    type=float, default=None)
    parser.add_argument("--interval",    type=int,    default=None, help="Loop cadence in seconds")
    parser.add_argument("--max-min",     type=int,    default=None, help="Stop after N minutes (None = run forever)")
    parser.add_argument("--no-stop-at-close", action="store_true", help="Do not stop when market closes")
    parser.add_argument("--once", action="store_true", help="Run one extender pass and exit")
    args = parser.parse_args()

    logging.info("ARGV → %s", " ".join(sys.argv))

    api = _alpaca_client()

    # 2) Read config defaults up-front
    cfg_mode        = str(getattr(CFG, "TP_EXTEND_MODE", "percent")).lower()
    cfg_trig_pct    = float(getattr(CFG, "TP_EXTEND_TRIGGER_PCT", 0.03))
    cfg_step_pct    = float(getattr(CFG, "TP_EXTEND_STEP_PCT",    0.01))
    cfg_trig_atr    = float(getattr(CFG, "TP_EXTEND_TRIGGER_ATR", 1.00))
    cfg_step_atr    = float(getattr(CFG, "TP_EXTEND_STEP_ATR",    0.50))
    cfg_max_mult    = float(getattr(CFG, "TP_EXTEND_MAX_MULT",    2.50))
    cfg_interval    = int(getattr(CFG, "MONITOR_INTERVAL_SEC",    120))
    cfg_max_min     = getattr(CFG, "MONITOR_MAX_MINUTES",         360)  # None/0 allowed
    cfg_stop_close  = bool(getattr(CFG, "MONITOR_STOP_AT_CLOSE",  True))

    logging.info("🔧 Using config file: %s", getattr(CFG, "__file__", "<unknown>"))
    logging.info(
        "CFG → mode=%s, pct_trigger=%.4f, pct_step=%.4f, atr_trigger=%.2f, atr_step=%.2f, max_mult=%.2f, interval=%ds",
        cfg_mode, cfg_trig_pct, cfg_step_pct, cfg_trig_atr, cfg_step_atr, cfg_max_mult, cfg_interval
    )
    logging.info(
        "ARGS → mode=%s, pct_trigger=%s, pct_step=%s, atr_trigger=%s, atr_step=%s, max_mult=%s, interval=%s",
        args.mode, args.trigger_pct, args.step_pct, args.trigger_atr, args.step_atr, args.max_mult, args.interval
    )

    # 3) Build profiles up-front
    raw_profile = getattr(CFG, "TP_EXTEND_PROFILES", {}).get(getattr(CFG, "TP_EXTEND_PROFILE", ""), {})
    profile = {k: v for k, v in raw_profile.items() if k.lower() not in ("mode", "tp_extend_mode")}  # exclude mode here

    # 4) Precedence helpers
    PREFER_CONFIG = os.getenv("QML_PREFER_CONFIG", "1") == "1"

    def pick(arg_val, cfg_val):
        # if preferring config, ignore CLI when CLI provided (unless arg is None)
        return cfg_val if (PREFER_CONFIG and arg_val is not None) else (arg_val if arg_val is not None else cfg_val)

    def _normalize_mode(v):
        if v is None: return None
        s = str(v).strip().lower()
        if s in ("atr","a"): return "atr"
        if s in ("percent","pct","p"): return "percent"
        raise ValueError(f"Invalid mode: {v!r}")

    def pick_mode(arg_mode, cfg_mode, prof):
        am = _normalize_mode(arg_mode)
        cm = _normalize_mode(cfg_mode)
        pm = _normalize_mode((prof or {}).get("TP_EXTEND_MODE"))
        return am or pm or cm or "percent"

    # 5) Decide mode (CLI > profile > config > default)
    mode = pick_mode(args.mode, cfg_mode, raw_profile)
    mode_src = "CLI" if args.mode else ("profile" if raw_profile.get("TP_EXTEND_MODE") else ("config" if cfg_mode else "default"))

    # 6) Start from config, then profile (excluding mode), then CLI/config pick()
    trigger_pct = cfg_trig_pct
    step_pct    = cfg_step_pct
    trigger_atr = cfg_trig_atr
    step_atr    = cfg_step_atr
    max_mult    = cfg_max_mult
    interval    = cfg_interval
    max_min     = cfg_max_min
    stop_close  = cfg_stop_close

    trigger_pct = pick(args.trigger_pct, profile.get("TP_EXTEND_TRIGGER_PCT", profile.get("pct_trigger", trigger_pct)))
    step_pct    = pick(args.step_pct,    profile.get("TP_EXTEND_STEP_PCT",    profile.get("pct_step",    step_pct)))
    trigger_atr = pick(args.trigger_atr, profile.get("TP_EXTEND_TRIGGER_ATR", profile.get("atr_trigger", trigger_atr)))
    step_atr    = pick(args.step_atr,    profile.get("TP_EXTEND_STEP_ATR",    profile.get("atr_step",    step_atr)))
    max_mult    = pick(args.max_mult,    profile.get("TP_EXTEND_MAX_MULT",    profile.get("max_mult",    max_mult)))
    interval    = pick(args.interval,    profile.get("MONITOR_INTERVAL_SEC",  profile.get("interval",    interval)))
    max_min     = pick(args.max_min,     profile.get("MONITOR_MAX_MINUTES",   profile.get("max_minutes", max_min)))

    # 7) Friendly warnings if mode/params mismatch
    if mode == "atr" and (args.trigger_pct is not None or args.step_pct is not None):
        logging.warning("ATR mode active: percent params (trigger_pct/step_pct) will be ignored.")
    if mode == "percent" and (args.trigger_atr is not None or args.step_atr is not None):
        logging.warning("Percent mode active: ATR params (trigger_atr/step_atr) will be ignored.")

    # 8) EFFECTIVE banner — now that 'mode' and knobs are defined
    logging.info(
        "✅ EFFECTIVE → mode=%s (%s), pct_trigger=%.4f, pct_step=%.4f, atr_trigger=%.2f, atr_step=%.2f, "
        "max_mult=%.2f, interval=%ds",
        mode, mode_src, float(trigger_pct), float(step_pct), float(trigger_atr), float(step_atr),
        float(max_mult), int(interval)
    )
    print(
        f"EFFECTIVE → mode={mode} ({mode_src}), pct_trigger={float(trigger_pct):.4f}, "
        f"pct_step={float(step_pct):.4f}, atr_trigger={float(trigger_atr):.2f}, "
        f"atr_step={float(step_atr):.2f}, max_mult={float(max_mult):.2f}, interval={int(interval)}s"
    )

    # 9) Run
    run_once = bool(getattr(args, "once", False)) or not bool(getattr(CFG, "MONITOR_ENABLE", True))
    if run_once:
        run_tp_extender_pass(
            api,
            mode=mode,
            trigger_pct=float(trigger_pct), step_pct=float(step_pct),
            trigger_atr=float(trigger_atr), step_atr=float(step_atr),
            max_mult=float(max_mult)
        )
        logging.info("✅ One-pass extender finished.")
        print("✅ One-pass extender finished.")
        return

    monitor_positions_loop(
        api,
        mode=mode,
        trigger_pct=float(trigger_pct), step_pct=float(step_pct),
        trigger_atr=float(trigger_atr), step_atr=float(step_atr),
        max_mult=float(max_mult),
        interval_sec=int(interval),
        max_minutes=(None if max_min in (None, 0, "None", "") else int(max_min)),
        stop_at_close=bool(stop_close)
    )
    print("🛑 Monitor loop exited.")

if __name__ == "__main__":
    main()
