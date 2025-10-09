from __future__ import annotations

# === Standard Library ===
import os
import re
import gc
import csv
import math
import time
import copy
import json
import pickle
import logging
from pathlib import Path
from statistics import mean
from datetime import datetime
from collections.abc import Iterable
from typing import Any, Dict, Mapping, Optional, Tuple

# === Third Party ===
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from joblib import load as joblib_load
from openpyxl import load_workbook
from itertools import product

# === Project Config ===
import config as CFG
from config import (
    # === Date Ranges ===
    stack_prob1_col, test_start, test_end,

    # === File Paths ===
    stock_file_path, stock_file, features_test_path, model_path_base,
    model_path_stacked, metrics_folder, backtest_path, predict_path,
    normalize_ticker_for_path,

    # === Model + Feature Settings ===
    ENSEMBLE_WEIGHTS, ENSEMBLE_LONG_THRESHOLD, ENSEMBLE_SHORT_THRESHOLD,
    CLASS_IS_LONG, MIN_CONF_FLOOR, ENSEMBLE_MODEL_PREFIXES,
    BACKTEST_STACKED, FORCE_PERCENT_BRACKETS, USE_PERCENT_BRACKETS,
    PERCENT_SL, PERCENT_TP, PRICE_DECIMALS, ENSEMBLE_TEMP,
    to_prefix, to_title, resolve_stacked_pred_column,
    STACK_BASE_COL_PATTERNS, pred_col, prob1_col, max_prob_col,
    prob_diff_col, stack_pred_col, stack_prob1_col,
    stack_max_prob_col, stack_prob_diff_col, STACKED_PRED_ALIASES,

    # backtest performance tuning
    BACKTEST_BASE, BACKTEST_STACKED, BACKTEST_MODE, WRITE_TRADES, WRITE_FUNNELS,

    # === Trading / Backtest Parameters ===
    initial_capital, transaction_cost, max_open_trades,
    atr_sl_multiplier, atr_tp_multiplier, min_pass,

    # === Signal Filtering Thresholds ===
    xgb_prob_diff_quantile, top_signals_limit,
    rsi_threshold_long, rsi_threshold_short,
    macd_hist_long, macd_hist_short,
    ema_buffer_pct_long, ema_buffer_pct_short,

    # === Technical Filters ===
    volume_min, strict_macd, volume_buffer_pct,
    min_model_return, min_model_sharpe,
)

from cTrain import load_stack_feature_cols

# === Pandas Options ===
pd.options.mode.copy_on_write = True

# === Runtime / Debug Toggles ===
WRITE_DEBUG_CSVS = False       # when True, writes large debug CSVs during backtests
ENABLE_EXCEL_AUTOFIT = False   # when True, openpyxl auto-fits columns (slow for big files)

# === Logging Setup ===
QML_LOG_LEVEL = os.getenv("QML_LOG_LEVEL", "WARNING").upper()

import logging as _logging
_level_name = str(getattr(CFG, "QML_LOG_LEVEL", os.getenv("QML_LOG_LEVEL", "INFO"))).upper()
_level = getattr(_logging, _level_name, _logging.INFO)
_logging.basicConfig(level=_level, format="%(asctime)s - %(levelname)s - %(message)s")

logging.basicConfig(
    level=_level,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Log-noise controls --------------------------------------------------------
_LOG_ONCE_KEYS: set[str] = set()

def _log_once(level: int, key: str, message: str) -> None:
    """Log `message` only once per process for this unique `key`."""
    if key in _LOG_ONCE_KEYS:
        return
    _LOG_ONCE_KEYS.add(key)
    logging.log(level, message)

# --- Selective warning filter --------------------------------------------------
class _ActionableOnlyFilter(logging.Filter):
    ACTIONABLE = (
        "missing", "failed", "error", "cannot", "invalid", "exception",
        "no valid", "memoryerror", "traceback"
    )
    NOISE = (
        "skipping", "diagnostics skipped", "no results to save",
        "no stacked rows produced", "no new base predictions",
        "stacked backtests disabled", "identical to base", "processed in"
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING:
            return True
        msg = (record.getMessage() or "").lower()
        if any(tok in msg for tok in self.ACTIONABLE):
            return True
        if os.getenv("QML_SILENCE_OK_WARNINGS", "1") == "1":
            if any(tok in msg for tok in self.NOISE):
                return False
            return False
        return True

if os.getenv("QML_SILENCE_OK_WARNINGS", "1") == "1":
    logging.getLogger().addFilter(_ActionableOnlyFilter())

# === Risk / Price Tick Setup ===
_TICK = 10.0 ** (-int(PRICE_DECIMALS))  # one price “tick”, e.g., 0.01 for 2 decimals

try:
    RISK_PCT = float(getattr(CFG, "RISK_PER_TRADE", 0.01))
except Exception:
    RISK_PCT = 0.01

def _to_prefix(model_key: str) -> str:
    # config.to_prefix accepts prefix/title/alias/friendly; returns short prefix (e.g., "xgb")
    return CFG.to_prefix(model_key)

# === FIX: class index sanity (engine convention is 0=LONG, 1=SHORT) ===
# Ensures CLASS_IS_SHORT always exists and both indices are valid.
def _class_indices() -> tuple[int, int]:
    """
    Returns (LONG_IDX, SHORT_IDX) using robust fallbacks if config is incomplete.
    - If CLASS_IS_SHORT is missing, infer it as the complement of CLASS_IS_LONG.
    - Coerces to integers and validates membership in {0,1} with distinct values.
    """
    # long index may be provided by config as int/str/bool
    long_idx = globals().get("CLASS_IS_LONG", 0)
    try:
        long_idx = int(long_idx)
    except Exception:
        long_idx = 0

    # short index can be provided by config; otherwise infer complement
    short_idx = globals().get("CLASS_IS_SHORT", None)
    if short_idx is None:
        short_idx = 1 - long_idx
    try:
        short_idx = int(short_idx)
    except Exception:
        short_idx = 1 - long_idx

    # Validate: must be distinct and within {0,1}
    if long_idx == short_idx or {long_idx, short_idx} - {0, 1}:
        logging.warning(
            "Invalid CLASS_IS_* values (%r, %r). Forcing 0=LONG, 1=SHORT.",
            long_idx, short_idx
        )
        long_idx, short_idx = 0, 1

    # Expose normalized globals for the rest of the module
    globals()["CLASS_IS_LONG"] = long_idx
    globals()["CLASS_IS_SHORT"] = short_idx
    return long_idx, short_idx

# Materialize normalized globals once at import time
CLASS_IS_LONG, CLASS_IS_SHORT = _class_indices()

# Progress helpers
try:
    from tqdm import tqdm  # optional; nice live bar if installed
except Exception:
    tqdm = None

def _iter_progress(seq, desc="Backtesting", total=None):
    """
    Yield items from seq while showing a progress bar.
    Uses tqdm if available; otherwise prints a simple bar.
    """
    items = list(seq) if not hasattr(seq, "__len__") else seq
    n = total if total is not None else (len(items) if hasattr(items, "__len__") else None)
    if tqdm and n is not None:
        return tqdm(items, total=n, desc=desc)
    else:
        # Fallback: simple textual bar
        def _gen():
            count = 0
            total_n = n if n is not None else 0
            for x in items:
                count += 1
                if total_n:
                    width = 30
                    done = int(width * count / total_n)
                    bar = "#" * done + "-" * (width - done)
                    print(f"\r{desc}: |{bar}| {count}/{total_n}", end="", flush=True)
                else:
                    # Unknown total
                    print(f"\r{desc}: {count}…", end="", flush=True)
                yield x
            print()  # newline after loop
        return _gen()

# --- Add near top of file ---
def _row_family_short_probs(row, prefer_stacked=True):
    """Collect member P(short) for xgb/lgbm/cat/rf/lr (stacked preferred if present)."""
    fams = [("xgb","XGBoost"),("lgbm","LightGBM"),("cat","CatBoost"),("rf","RandomForest"),("lr","LogisticRegression")]
    out = []
    for pfx, title in fams:
        for tag in (["stack_"] if prefer_stacked else ["", "stack_"]):
            cand = f"{pfx}_{tag}prob_class_1"
            if cand in row.index:
                try:
                    v = float(row[cand])
                    if np.isfinite(v): out.append(v)
                    break
                except Exception:
                    pass
        # fallback: TitleCase base prob_class_1
        tc = f"{title}_prob_class_1"
        if (not out or len(out) < len(fams)) and tc in row.index:
            try:
                v = float(row[tc])
                if np.isfinite(v): out.append(v)
            except Exception:
                pass
    return out


def _hc_export_cols(df: pd.DataFrame) -> list[str]:
    """Columns worth emitting to *_high_confidence_signals.csv (thin, human-readable)."""
    base = ["date", "ticker", "open", "high", "low", "close", "volume",
            "ensemble_pred_engine", "ensemble_conf", "ensemble_max_prob"]
    # Common indicators if present
    extras = [c for c in ["ema_20", "rsi", "macd", "macd_signal", "macd_hist"] if c in df.columns]
    # Per-model signals worth keeping
    sigs = [c for c in df.columns if c.endswith(("_pred",
                                                 "_stack_prediction",
                                                 "_prob_class_1",
                                                 "_max_prob",
                                                 "_prob_diff"))]
    # Preserve order, remove dups, keep only those that exist
    seen, out = set(), []
    for c in base + extras + sigs:
        if c in df.columns and c not in seen:
            seen.add(c); out.append(c)
    return out

def _candidate_pred_cols(prefix: str, stacked: bool) -> list[str]:
    # TitleCase (e.g., XGBoost) via config
    title = CFG.to_title(prefix)
    if stacked:
        return [
            f"{prefix}_stack_prediction",
            f"{title}_stack_prediction",
            f"{title.lower()}_stack_prediction",
        ]
    else:
        cands = [f"{prefix}_pred", f"{title}_pred"]
        if prefix == "xgb":
            cands += ["xgboost_prediction", "xgb_prediction", "xgboost_pred"]
        return cands

# --- dBacktest.py — helpers to sanitize probability column wiring ---

def _sanitize_prob_col(name):
    """Convert sentinel strings like 'none' / '' / NaN into None."""
    if name is None:
        return None
    s = str(name).strip().lower()
    return None if s in {"", "none", "null", "nan"} else name

def unify_pred_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure BOTH ProperCase and lowercase aliases exist for base preds & prob columns.
    Covers: *_pred, *_prob_class_{0,1}, *_max_prob, *_prob_diff for
    XGBoost/LightGBM/CatBoost/RandomForest/LogisticRegression.
    """
    mapping = {
        "xgb": ("XGBoost",),
        "lgbm": ("LightGBM",),
        "cat": ("CatBoost",),
        "rf": ("RandomForest",),
        "lr": ("LogisticRegression",),
    }
    suffixes = [
        ("_pred", "_pred"),
        ("_prob_class_0", "_prob_class_0"),
        ("_prob_class_1", "_prob_class_1"),
        ("_max_prob", "_max_prob"),
        ("_prob_diff", "_prob_diff"),
    ]

    for pfx, titles in mapping.items():
        for title in titles:
            for lo_suf, hi_suf in suffixes:
                lo = f"{pfx}{lo_suf}"
                hi = f"{title}{hi_suf}"
                # If ProperCase exists but lowercase missing, create it
                if hi in df.columns and lo not in df.columns:
                    df[lo] = df[hi]
                # If lowercase exists but ProperCase missing, create it
                if lo in df.columns and hi not in df.columns:
                    df[hi] = df[lo]
    return df

def _has_pred_cols(df) -> bool:
    cols = [c.lower() for c in df.columns]
    return any(c.endswith("_pred") for c in cols) or any("stack_prediction" in c for c in cols)

def ensure_enriched_predictions_file(ticker: str) -> tuple[str, pd.DataFrame]:
    """
    Guarantee that <predict_path>/<T>_test_features_with_predictions.csv exists
    and is fully enriched (base + stacked + ensemble).
    Returns (outfile_path, df).
    """
    tkr = normalize_ticker_for_path(ticker)
    out_path = os.path.join(predict_path, f"{tkr}_test_features_with_predictions.csv")
    os.makedirs(predict_path, exist_ok=True)

    # Fast path: if file exists and has *_pred columns -> use it, else RAISE instead of rebuilding
    if os.path.exists(out_path):
        df_out = pd.read_csv(out_path)
        if not df_out.empty and _has_pred_cols(df_out):
            logging.info(f"✅ Using features from {os.path.basename(out_path)} (already enriched)")
            return out_path, df_out
        raise FileNotFoundError(f"{out_path} exists but lacks prediction columns")

    # --- 2) Try to source from 2.Features_test (with or without predictions)
    src_candidates = [
        os.path.join(features_test_path, f"{tkr}_test_features_with_predictions.csv"),
        os.path.join(features_test_path, f"{tkr}_test_features.csv"),
    ]
    src_path = None
    src_df = None
    for p in src_candidates:
        if os.path.exists(p):
            try:
                tmp = pd.read_csv(p)
                if not tmp.empty:
                    src_path = p
                    src_df = tmp
                    logging.info(f"📥 Sourcing features from {os.path.basename(p)} to build enriched predictions.")
                    break
            except Exception as e:
                logging.info(f"ℹ️ Skipping unreadable source {p}: {e}")

    if src_df is None:
        raise FileNotFoundError(
            f"No usable source in features_test_path for {ticker}. "
            f"Expected {tkr}_test_features(_with_predictions).csv"
        )

    # 3) If source already has base preds, keep; else add base preds
    df = src_df.copy()
    BASE_MODELS = ["xgboost","lightgbm","catboost","randomforest","logisticregression"]
    base_present = any(
        c.lower().endswith("_pred") and ("stack" not in c.lower()) and (not c.lower().startswith("ensemble"))
        for c in df.columns
    )
    if not base_present:
        df = add_all_base_model_predictions(df, ticker, BASE_MODELS)

    # >>>>>>> CRUCIAL: normalize aliases BEFORE stacking <<<<<<<
    df = ensure_xgb_aliases(df)      # your existing helper
    df = ensure_base_aliases(df)     # your existing helper
    df = unify_pred_aliases(df)      # NEW helper (above)

    # 4) Build stacked predictions (now that aliases exist)
    STACK_MODELS = ["lightgbm","catboost","randomforest","logisticregression"]  # no xgb stack
    try:
        df = add_all_stacked_model_predictions(df, ticker, families=STACK_MODELS)
        df = ensure_stack_aliases(df)
        df = _migrate_legacy_stack_proba(df)
    except Exception as e:
        logging.info(f"ℹ️ {ticker} — stacked build failed/partial: {e}")

    # 5) Build ensemble
    try:
        # process_ticker
        df = apply_ensemble_predictions_engine(
            df, out_prefix="ensemble",
            long_thresh=ENSEMBLE_LONG_THRESHOLD,
            short_thresh=ENSEMBLE_SHORT_THRESHOLD,
            prefer_stacked=CFG.PREFER_STACKED_IF_AVAILABLE,       # <-- use config
            weights=CFG.ENSEMBLE_WEIGHTS                          # <-- be explicit
        )

    except Exception as e:
        logging.info(f"ℹ️ {ticker} — ensemble build failed/partial: {e}")

    # (save as you already do)
    to_save = df.reset_index(drop=False).rename(columns={"index": "date"}) \
              if isinstance(df.index, pd.DatetimeIndex) else df
    to_save.to_csv(out_path, index=False, float_format="%.6g")
    logging.info(f"💾 Wrote enriched predictions → {os.path.basename(out_path)}")

    # Optional sanity ping
    sc = [c.lower() for c in to_save.columns]
    logging.info(
        "📌 %s saved → base=%s, stack=%s, ensemble=%s",
        ticker,
        any(c.endswith("_pred") and ("stack" not in c) and (not c.startswith("ensemble")) for c in sc),
        any("stack_prediction" in c for c in sc),
        any(c.startswith("ensemble") for c in sc),
    )

    return out_path, df

def _coalesce_prob_column(df: pd.DataFrame, requested: str | None) -> str | None:
    """
    Return a probability/confidence column that actually exists on df.
    Priority:
      1) the explicitly requested name (if present)
      2) 'ensemble_conf' (if present)
      3) a model prob_class_1 column matching the prediction column family
      4) None (caller handles)
    """
    # 1) explicit request
    if requested and requested in df.columns:
        return requested

    # 2) ensemble confidence, if available
    if "ensemble_conf" in df.columns:
        return "ensemble_conf"

    # 3) scan for any *_prob_class_1 already present (common case)
    for c in df.columns:
        if c.endswith("_prob_class_1"):
            return c

    # 4) nothing usable — let caller handle None
    return None

def resolve_prediction_column(df, model_key: str, stacked: bool = False,
                              explicit: str | None = None) -> tuple[str, list[str]]:
    if explicit and explicit in df.columns:
        return explicit, [explicit]

    # Use canonical helpers from config
    try:
        pfx   = to_prefix(model_key)   # e.g., 'lgbm'
        title = to_title(model_key)    # e.g., 'LightGBM'
    except Exception:
        pfx = str(model_key).lower().strip()
        title = pfx

    tried: list[str] = []
    if stacked:
        candidates = [
            f"{pfx}_stack_prediction",
            f"{title}_stack_prediction",
            f"{title.lower()}_stack_prediction",
        ]
    else:
        candidates = [f"{pfx}_pred", f"{title}_pred"]
        if pfx == "xgb":
            candidates += ["xgboost_prediction", "xgb_prediction", "xgboost_pred"]

    for col in candidates:
        tried.append(col)
        if col in df.columns:
            return col, tried

    raise KeyError(
        f"No prediction column found for model={model_key!r} (stacked={stacked}). Tried: {tried}"
    )

def get_bracket_fractions(price, sl=None, tp=None, atr=None, *, sl_mult_override=None, tp_mult_override=None):
    use_pct = bool(getattr(CFG, "FORCE_PERCENT_BRACKETS", False))
    if sl is not None and tp is not None and price and price > 0:
        return (abs(price - sl) / price, abs(tp - price) / price)
    if use_pct:
        return (float(getattr(CFG, "PERCENT_SL", 0.03)), float(getattr(CFG, "PERCENT_TP", 0.05)))
    if atr and price and price > 0:
        sl_mult = float(sl_mult_override) if sl_mult_override is not None else float(getattr(CFG, "atr_sl_multiplier", 2.0))
        tp_mult = float(tp_mult_override) if tp_mult_override is not None else float(getattr(CFG, "atr_tp_multiplier", 3.0))
        return (max(1e-6, (sl_mult * atr) / price), max(1e-6, (tp_mult * atr) / price))
    if bool(getattr(CFG, "USE_PERCENT_BRACKETS", False)):
        return (float(getattr(CFG, "PERCENT_SL", 0.03)), float(getattr(CFG, "PERCENT_TP", 0.05)))
    return (0.02, 0.03)


def _first_present_base(df, model_type: str) -> str | None:
    """
    Return the first base prediction column present for a given family, preferring ProperCase.
    """
    proper_map = {
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "randomforest": "RandomForest",
        "logisticregression": "LogisticRegression",
    }
    proper = proper_map.get(model_type.lower(), model_type)
    candidates = [f"{proper}_pred", f"{model_type.lower()}_pred"]
    return next((c for c in candidates if c in df.columns), None)


def _enforce_min_distance(
    side: str, base_price: float, sl_price: float, tp_price: float
) -> tuple[float, float]:
    """
    Make sure SL/TP sit strictly 1 tick past the base price on the correct side, then round
    to PRICE_DECIMALS — identical spirit to the live nudging logic.
    """
    base = float(base_price)
    step = _TICK
    d = int(PRICE_DECIMALS)

    side = str(side).lower()
    if side == "long":
        if sl_price >= base - step:
            sl_price = base - step
        if tp_price <= base + step:
            tp_price = base + step
    elif side == "short":
        if sl_price <= base + step:
            sl_price = base + step
        if tp_price >= base - step:
            tp_price = base - step
    else:
        raise ValueError(f"Unknown side '{side}'")

    return round(float(sl_price), d), round(float(tp_price), d)


def compute_bracket_prices_from_fracs(
    side: str, base_price: float, sl_frac: float, tp_frac: float
) -> tuple[float, float]:
    """
    Convert fractions → absolute SL/TP and apply the same rounding/min-distance
    rules as live trading.  This is the exact bridge you should use inside backtests.
    """
    base = float(base_price)
    sl_frac = float(sl_frac)
    tp_frac = float(tp_frac)

    side = str(side).lower()
    if side == "long":
        sl = base * (1.0 - sl_frac)
        tp = base * (1.0 + tp_frac)
    elif side == "short":
        sl = base * (1.0 + sl_frac)
        tp = base * (1.0 - tp_frac)
    else:
        raise ValueError(f"Unknown side '{side}'")

    # Round to configured decimals, then enforce a strict 1-tick separation.
    d = int(PRICE_DECIMALS)
    sl = round(float(sl), d)
    tp = round(float(tp), d)
    sl, tp = _enforce_min_distance(side, base, sl, tp)
    return sl, tp


def _norm_family(name: str) -> str:
    n = (name or "").lower()
    # strip separators and common trailing bits
    n = n.replace("-", "_").replace(" ", "_")
    # strip generic suffixes
    for suf in ("_cls", "_clf", "_model", "_base", "_stack", "_stacked"):
        n = n.removesuffix(suf)
    # collapse repeating underscores
    while "__" in n:
        n = n.replace("__", "_")
    # map aliases -> canonical
    for canon, meta in STACK_FAMILY_CANON.items():
        if n == canon:
            return canon
        if any(n == a for a in meta["aliases"]):
            return canon
    # fallback: first token is usually family
    return n.split("_")[0]

def _drop_identical_stack(df: pd.DataFrame, fam: str) -> None:
    """If <fam>_stack_prediction exists but is identical to base, ignore it downstream."""
    s_col = stack_pred_col(fam)   # e.g., "lgbm_stack_prediction"
    b_col = pred_col(fam)         # e.g., "lgbm_pred"
    if s_col in df.columns and b_col in df.columns:
        try:
            if df[s_col].equals(df[b_col]):
                logging.warning(f"⚠️ {fam} (stacked) identical to base {b_col}: treating as absent.")
                # simplest is to drop; or rename to mark as invalid
                df.drop(columns=[s_col], inplace=True, errors="ignore")
        except Exception:
            pass

def ensure_ensemble_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    if "ensemble_confidence" in out.columns and "ensemble_conf" not in out.columns:
        out["ensemble_conf"] = out["ensemble_confidence"]
    if "ensemble_score" in out.columns and "ensemble_conf" not in out.columns:
        out["ensemble_conf"] = out["ensemble_score"]
    return out

def _maybe_drop_identical_stacks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log if a stacked prediction column is byte-identical to its base, but do not drop.
    Dropping causes alias churn & confusing re-creations later.
    """
    out = df.copy()
    families = [("lgbm","LightGBM"), ("cat","CatBoost"), ("rf","RandomForest"), ("lr","LogisticRegression")]
    for pfx, title in families:
        base_candidates  = [f"{pfx}_pred", f"{title}_pred"]
        stack_pred = f"{pfx}_stack_prediction"
        if stack_pred not in out.columns:
            continue
        for base_col in base_candidates:
            if base_col in out.columns:
                try:
                    if out[base_col].equals(out[stack_pred]):
                        logging.info("ℹ️ Identical stacked/base predictions for %s (%s == %s); keeping both for transparency.",
                                     title, stack_pred, base_col)
                        break
                except Exception:
                    pass
    return out

def _side_prob_cols_for_prediction(df: pd.DataFrame, prediction_column: str) -> tuple[str|None, str|None]:
    LONG_IDX, SHORT_IDX = _class_indices()
    fam = _infer_family_from_name(prediction_column or "")
    is_stack = "stack" in (prediction_column or "").lower()

    if fam == "ensemble":
        lp = f"ensemble_prob_class_{LONG_IDX}"
        sp = f"ensemble_prob_class_{SHORT_IDX}"
        return (lp if lp in df.columns else None, sp if sp in df.columns else None)

    if fam in {"xgb","lgbm","cat","rf","lr"}:
        p = fam
        candL = f"{p}_{'stack_' if is_stack else ''}prob_class_{LONG_IDX}"
        candS = f"{p}_{'stack_' if is_stack else ''}prob_class_{SHORT_IDX}"
        altL  = f"{p}_prob_class_{LONG_IDX}"   # fallback to base
        altS  = f"{p}_prob_class_{SHORT_IDX}"
        lp = candL if candL in df.columns else (altL if altL in df.columns else None)
        sp = candS if candS in df.columns else (altS if altS in df.columns else None)
        return lp, sp

    return (None, None)

def run_sltp_grid_fast(
    prepared_trades: pd.DataFrame,
    *, sl_vals: list[float], tp_vals: list[float],
    min_trades: int = 8, max_combos: int = 9
) -> list[dict]:
    """
    Run a compact grid directly on a pre-built trades_df to avoid repeated filtering.
    """
    combos = [(s, t) for s in sl_vals for t in tp_vals]
    if len(combos) > max_combos:
        combos = combos[:max_combos]

    results = []
    if len(prepared_trades) < min_trades:
        return results

    for slm, tpm in combos:
        ret, sharpe, qml, trades, equity = backtest_engine(
            prepared_trades,
            sl_mult=slm, tp_mult=tpm,
            use_ranking=False,  # already filtered
            min_conf_floor=MIN_CONF_FLOOR
        )
        results.append({"sl_mult": slm, "tp_mult": tpm, "ret": ret, "sharpe": sharpe, "quantml": qml, "trades": trades})
    return results


def _expand_stack_feature_patterns(df: pd.DataFrame, patterns: list[str]) -> list[str]:
    """Return existing columns matching simple wildcard list like '*_max_prob'."""
    import fnmatch
    cols = set()
    for pat in patterns:
        m = [c for c in df.columns if fnmatch.fnmatch(c, pat)]
        cols.update(m)
    return [c for c in df.columns if c in cols]

def build_stack_feature_matrix(
    df: pd.DataFrame,
    expected_features: list[str] | None = None,
    min_required: int = 6,
) -> tuple[pd.DataFrame, list[str]]:
    """
    When `expected_features` is given (from <ticker>_<fam>_stack_scaler_columns.pkl),
    build X with EXACTLY those columns, in that order. Any missing columns are added
    with NaN so a pipeline's SimpleImputer can handle them and the feature count matches.

    If no expected features were saved, fall back to wildcard expansion + numeric picks.
    Never raise: always return a DataFrame (may be empty) and the list of used columns.
    """

    # Case 1 — we KNOW what the pipeline expects: reindex to exact set + order
    if expected_features and isinstance(expected_features, (list, tuple)):
        used = list(expected_features)
        # Create any missing columns as NaN, preserve order exactly as expected
        X = df.reindex(columns=used, fill_value=np.nan).copy()
        # Clean infs
        X = X.replace([np.inf, -np.inf], np.nan)
        # Keep rows with at least some information
        X = X.dropna(how="all", axis=0)
        return X, used

    # Case 2 — we have to infer features from present columns (legacy fallback)
    pattern_cols = _expand_stack_feature_patterns(df, STACK_BASE_COL_PATTERNS)
    ensemble_pref = [c for c in ["ensemble_conf","ensemble_prob_class_1","ensemble_prob_class_0"] if c in df.columns]
    extra = [c for c in pattern_cols if c not in ensemble_pref]
    used = ensemble_pref + extra

    if len(used) < min_required:
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        prob_like = [c for c in numeric if "prob" in c.lower() or "max_prob" in c.lower()]
        rest     = [c for c in numeric if c not in prob_like]
        used = prob_like[:min_required] + rest[: max(min_required, 12)]

    X = df[used].replace([np.inf, -np.inf], np.nan)
    X = X.dropna(how="all", axis=0)
    return X, used

_trade_id_counter = 0
def _next_trade_id():
    global _trade_id_counter
    _trade_id_counter += 1
    return _trade_id_counter

def get_stacked_pred_col(df, model_type: str):
    """
    Return a normalized 0/1 engine column for the stacked model, or None.
    Uses resolve_stacked_pred_column() to support legacy column names.
    """
    col = resolve_stacked_pred_column(df, model_type)
    if not col:
        return None
    # map_predictions_to_binary(df, col) should already create '<col>_engine' 0/1
    return map_predictions_to_binary(df, col)

def _ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def write_stack_vs_base_diagnostics(ticker: str, df: pd.DataFrame, out_dir: str) -> str:
    """
    For each family, compare base vs stacked (labels + probs).
    Writes/append to <out_dir>/stack_vs_base_diagnostics.csv
    Returns the path.
    """
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "stack_vs_base_diagnostics.csv")
    fields = [
        "ticker","family","base_pred_col","stack_pred_col","n_overlap","n_equal",
        "all_equal","pearson_p1","spearman_p1","mean_base_p1","mean_stack_p1"
    ]
    fams = {
        "lightgbm": ("lgbm_pred","lgbm_stack_prediction","LightGBM_pred","LightGBM_stack_prediction",
                     "lgbm_prob_class_1","lgbm_stack_prob_class_1","LightGBM_prob_class_1","LightGBM_stack_prob_class_1"),
        "catboost": ("cat_pred","cat_stack_prediction","CatBoost_pred","CatBoost_stack_prediction",
                     "cat_prob_class_1","cat_stack_prob_class_1","CatBoost_prob_class_1","CatBoost_stack_prob_class_1"),
        "randomforest": ("rf_pred","rf_stack_prediction","RandomForest_pred","RandomForest_stack_prediction",
                          "rf_prob_class_1","rf_stack_prob_class_1","RandomForest_prob_class_1","RandomForest_stack_prob_class_1"),
        "logisticregression": ("lr_pred","lr_stack_prediction","LogisticRegression_pred","LogisticRegression_stack_prediction",
                               "lr_prob_class_1","lr_stack_prob_class_1","LogisticRegression_prob_class_1","LogisticRegression_stack_prob_class_1"),
    }

    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header: w.writeheader()

        for fam, (b1,s1,b2,s2,pb1,ps1,pb2,ps2) in fams.items():
            base_col = b1 if b1 in df.columns else (b2 if b2 in df.columns else None)
            stack_col = s1 if s1 in df.columns else (s2 if s2 in df.columns else None)
            base_p1  = pb1 if pb1 in df.columns else (pb2 if pb2 in df.columns else None)
            stack_p1 = ps1 if ps1 in df.columns else (ps2 if ps2 in df.columns else None)
            if not base_col or not stack_col:
                continue

            a = pd.to_numeric(df[base_col], errors="coerce")
            b = pd.to_numeric(df[stack_col], errors="coerce")
            mask = a.notna() & b.notna()
            n_overlap = int(mask.sum())
            n_equal   = int((a[mask] == b[mask]).sum())
            all_equal = (n_overlap > 0 and n_equal == n_overlap)

            pearson = spearman = mean_bp = mean_sp = None
            if base_p1 and stack_p1 and base_p1 in df.columns and stack_p1 in df.columns:
                bp = pd.to_numeric(df[base_p1], errors="coerce")
                sp = pd.to_numeric(df[stack_p1], errors="coerce")
                mm = bp.notna() & sp.notna()
                if mm.any():
                    pearson = float(bp[mm].corr(sp[mm]))
                    spearman = float(bp[mm].corr(sp[mm], method="spearman"))
                    mean_bp = float(bp[mm].mean())
                    mean_sp = float(sp[mm].mean())

            w.writerow({
                "ticker": ticker, "family": fam,
                "base_pred_col": base_col, "stack_pred_col": stack_col,
                "n_overlap": n_overlap, "n_equal": n_equal, "all_equal": all_equal,
                "pearson_p1": pearson, "spearman_p1": spearman,
                "mean_base_p1": mean_bp, "mean_stack_p1": mean_sp
            })
    return out_path

def canonical_model_variants(model_name: str):
    """
    Returns (base_name, stacked_name) canonically, regardless of input spelling.
    Example: 'stacked_randomforest' -> ('randomforest', 'randomforest_stack')
             'xgb' -> ('xgboost', 'xgboost_stack')
    """
    raw = (model_name or "").lower()
    fam = _norm_family(raw)
    base_name = fam
    stacked_name = f"{fam}_stack"
    return base_name, stacked_name

def resolve_prob_col(df_cols, requested_model: str, prefer_stacked: bool):
    """
    Given available dataframe columns and a 'requested_model' (any variant string),
    returns the *existing* probability column name to use, preferring stacked if requested.
    Expected prob cols: '{name}_prob_class_1' where name is base or '{base}_stack'.
    """
    base, stack = canonical_model_variants(requested_model)
    candidates = []
    if prefer_stacked:
        candidates += [f"{stack}_prob_class_1", f"{stack}_prob"]
    candidates += [f"{base}_prob_class_1", f"{base}_prob"]
    for c in candidates:
        if c in df_cols:
            return c
    # last-chance heuristic: search for any '{base or stack}_prob*'
    for c in df_cols:
        if c.startswith(base) or c.startswith(stack):
            if "prob" in c:
                return c
    return None
# --- end: robust stack alias normalization ---

# ===== Float comparison guard for confidence thresholds =====
EPS = 1e-6

def _passes_conf(conf: float, min_conf: float) -> bool:
    return float(conf) >= float(min_conf) - EPS

def _infer_family_from_name(name: str) -> str | None:
    """
    Map any reasonable column/model text to the canonical short family:
      xgb, lgbm, cat, rf, lr, ensemble
    Works for long names like 'lightgbm_pred_engine' too.
    """
    s = (str(name) or "").lower()
    for key, pfx in (
        ("xgboost","xgb"), ("xgb","xgb"),
        ("lightgbm","lgbm"), ("lgbm","lgbm"),
        ("catboost","cat"), ("cat","cat"),
        ("randomforest","rf"), ("rf","rf"),
        ("logisticregression","lr"), ("logistic","lr"), ("logreg","lr"), ("lr","lr"),
        ("ensemble","ensemble"),
    ):
        if key in s:
            return pfx
    return None


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        logging.error(f"❌ Could not create directory {path}: {e}")
        raise
    
def sanitize_filename(text):
    """Sanitize the filename to remove any special characters."""
    # Replace invalid characters (e.g., ':', '/', etc.)
    return re.sub(r'[\/*?:"<>|]', "", str(text)).replace(":", "_")

# Put near your other small utils
def _is_ensemble_col(name: str) -> bool:
    return isinstance(name, str) and name.lower().startswith("ensemble")

def sharpe_ratio(returns, rf=0.0, periods_per_year=252, *, use_log=False, ddof=1):
    """
    returns: per-period simple returns
    rf:      scalar per-period risk-free (same frequency) or array-like aligned with returns
    use_log: if True, compute Sharpe on log-excess returns: log1p(r) - log1p(rf)
    """
    r = np.asarray(returns, dtype=float).ravel()
    rf_vec = np.asarray(rf, dtype=float).ravel()

    if rf_vec.size == 1:
        rf_vec = np.full_like(r, float(rf_vec))
    elif rf_vec.shape != r.shape:
        raise ValueError("rf must be scalar or the same shape as returns")

    mask = np.isfinite(r) & np.isfinite(rf_vec)
    r, rf_vec = r[mask], rf_vec[mask]
    if r.size < 2:
        return np.nan

    excess = (np.log1p(r) - np.log1p(rf_vec)) if use_log else (r - rf_vec)
    mu = np.nanmean(excess)
    sigma = np.nanstd(excess, ddof=ddof)
    return (mu / sigma) * np.sqrt(periods_per_year) if sigma > 0 else np.nan

def _coalesce_prob_columns(df: pd.DataFrame, prefer_stacked: bool=True) -> tuple[str|None, str|None]:
    """
    Return (long_prob_col, short_prob_col) if available, else (None, None).
    Ensemble first; then stacked; then base.
    """
    LONG_IDX, SHORT_IDX = _class_indices()

    # Ensemble first
    cL = f"ensemble_prob_class_{LONG_IDX}"
    cS = f"ensemble_prob_class_{SHORT_IDX}"
    if cL in df.columns and cS in df.columns:
        return cL, cS

    # Families
    fams = ("xgb","lgbm","cat","rf","lr")
    # Stacked then base
    orders = [("stack", True), ("", False)] if prefer_stacked else [("", False), ("stack", True)]
    for tag, is_stack in orders:
        for fam in fams:
            cL = f"{fam}_{'stack_' if is_stack else ''}prob_class_{LONG_IDX}"
            cS = f"{fam}_{'stack_' if is_stack else ''}prob_class_{SHORT_IDX}"
            if cL in df.columns and cS in df.columns:
                return cL, cS

    return None, None

def save_gridsearch_results(results_df, ticker, model="Ensemble", metric_to_optimize="Sharpe"):
    if results_df.empty:
        logging.warning(f"⚠️ No results to save for {ticker}.")
        return

    results_df = results_df.sort_values(by=metric_to_optimize, ascending=False)

    os.makedirs(metrics_folder, exist_ok=True)

    safe_model_name = str(model).replace("(", "_").replace(")", "").replace("=", "_").replace(",", "").replace(" ", "")
    filename = f"SLTP_GridSearch_{ticker}_{safe_model_name}.xlsx"
    full_path = os.path.join(metrics_folder, filename)

    results_df.to_excel(full_path, index=False)
    logging.info(f"📁 Exported SL/TP grid results to: {full_path}")

# --- small naming helper so all 0/1 engine columns are consistent ---
def engine_col(source_col: str) -> str:
    """
    Return the engine-mapped column name for *source_col*.
    Idempotent: if *source_col* already endswith '_engine', it's returned unchanged.
    Example: 'lgbm_pred' -> 'lgbm_pred_engine', 'ensemble_pred_engine' -> 'ensemble_pred_engine'.
    """
    name = str(source_col).strip()
    return name if name.endswith("_engine") else f"{name}_engine"

def map_predictions_to_binary(df: pd.DataFrame, source_col: str) -> str:
    """
    Normalize a model's prediction column to engine convention: 0=LONG, 1=SHORT.
    Supports {0,1}, {1,2}, {-1,1}. Anything else is coerced to NaN.
    """
    s = pd.to_numeric(df[source_col], errors="coerce")
    vals = set(int(v) for v in s.dropna().unique())

    if vals.issubset({0, 1}):
        mapped = s.astype("float")
    elif vals.issubset({1, 2}):           # <-- handle 1/2 FIRST
        mapped = s.map({1: 0, 2: 1}).astype("float")
    elif vals.issubset({-1, 1}):
        mapped = s.map({+1: 0, -1: 1}).astype("float")
    else:
        mapped = s.where(s.isin([0, 1]), other=np.nan).astype("float")

    out_col = engine_col(source_col)
    df[out_col] = mapped
    return out_col

def map_predictions_to_engine(df, source_col: str, engine_pred_col: str) -> str:
    """
    Backwards-compatible wrapper. Will mirror the result of map_predictions_to_binary(...)
    into the requested engine_pred_col name if it differs.
    """
    created = map_predictions_to_binary(df, source_col)
    if created != engine_pred_col:
        df[engine_pred_col] = df[created]
    return engine_pred_col

def ensure_ensemble_columns(df: pd.DataFrame) -> pd.DataFrame:
    return apply_ensemble_predictions_engine(
        df,
        out_prefix="ensemble",
        long_thresh=CFG.ENSEMBLE_LONG_THRESHOLD,
        short_thresh=CFG.ENSEMBLE_SHORT_THRESHOLD,
        prefer_stacked=CFG.PREFER_STACKED_IF_AVAILABLE,
        weights=CFG.ENSEMBLE_WEIGHTS
    )

# === NEW: side-aware confidence resolver =====================================
def resolve_side_confidence(row: pd.Series, prediction_column: str, side01: int, fallback_col: str | None = None) -> float:
    """
    Return P(side) for the chosen side (0=LONG, 1=SHORT).
    Tries <fam>_prob_class_0/1 (or ensemble_*), falls back to max_prob/ensemble_conf, 
    and finally to 'fallback_col' if provided.
    """

    src = str(prediction_column or "").lower().replace("_engine", "")
    fam = _infer_family_from_name(src)

    def _first_present(_row: pd.Series, candidates: list[str]) -> float | None:
        for c in candidates:
            if c in _row.index:
                try:
                    v = float(_row[c])
                    if np.isfinite(v):
                        return v
                except Exception:
                    pass
        return None

    if fam == "ensemble":
        p0 = _first_present(row, ["ensemble_prob_class_0"])
        p1 = _first_present(row, ["ensemble_prob_class_1"])
        alt = _first_present(row, ["ensemble_conf"])
    else:
        title = {
            "xgb": "XGBoost", "lgbm": "LightGBM", "cat": "CatBoost",
            "rf": "RandomForest", "lr": "LogisticRegression"
        }.get(fam, fam or "")
        p0 = _first_present(row, [f"{fam}_prob_class_0", f"{title}_prob_class_0"])
        p1 = _first_present(row, [f"{fam}_prob_class_1", f"{title}_prob_class_1"])
        alt = _first_present(row, [f"{fam}_max_prob", f"{title}_max_prob"])

    # Use the actual side prob; invert if only the opposite class is available
    if side01 == 0:  # LONG
        if p0 is not None: return float(p0)
        if p1 is not None: return float(1.0 - p1)
    else:            # SHORT
        if p1 is not None: return float(p1)
        if p0 is not None: return float(1.0 - p0)

    if alt is not None:
        return float(alt)

    if fallback_col and fallback_col in row.index:
        try:
            val = float(row[fallback_col])
            if np.isfinite(val):
                return val
        except Exception:
            pass

    return float("nan")
# ==============================================================================

def _has_any_model_artifacts(ticker: str) -> bool:
    """
    Quick existence check for any saved base/stacked model artifacts for this ticker.
    """
    try:
        base = Path(model_path_base)
        stk  = Path(model_path_stacked)
    except NameError:
        return False

    patterns = [
        f"{ticker}_*_pipeline.joblib",
        f"{ticker}_*model.pkl",
        f"{ticker}_*scaler.pkl",
        f"{ticker}_*scaler_columns.pkl",
        f"{ticker}_*stack_*model.pkl",
        f"{ticker}_*stack_*scaler.pkl",
        f"{ticker}_*stack_*scaler_columns.pkl",
    ]
    return any(p for pat in patterns for p in list(base.glob(pat)) + list(stk.glob(pat)))


def load_predictions_for_backtest(ticker: str, predict_path: str) -> pd.DataFrame:
    """
    Load the canonical predictions file emitted by ePredict and normalize columns for backtest.
    """
    tkr = normalize_ticker_for_path(ticker)
    csv = os.path.join(predict_path, f"{tkr}_test_features_with_predictions.csv")
    df = pd.read_csv(csv)
    if "date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # pick the active model cols (prefer XGB if nothing else is obvious)
    pred_col, prob_col = get_pred_and_prob_cols(df, model="xgboost", prefer_stacked=False)
    if pred_col is None or prob_col is None:
        # last resort: leave as-is and let the caller decide
        return df
    return ensure_ensemble_columns(df)

def compute_weighted_ensemble_prediction(row):
    """
    Weighted ensemble over each model's own directional prediction (0=LONG, 1=SHORT).
    Robust to legacy labels (1=LONG, 2=SHORT) and to ±1/0 encodings.
    """
    weights = {
        'XGBoost': ENSEMBLE_WEIGHTS.get('XGBoost', 0),
        'LightGBM': ENSEMBLE_WEIGHTS.get('LightGBM', 0),
        'CatBoost': ENSEMBLE_WEIGHTS.get('CatBoost', 0),
        'RandomForest': ENSEMBLE_WEIGHTS.get('RandomForest', 0),
        'LogisticRegression': ENSEMBLE_WEIGHTS.get('LogisticRegression', 0),
    }
    fallback_map = {
        'XGBoost': ("XGBoost_pred", "xgboost_prediction"),
        'LightGBM': ("lightgbm_stack_prediction", "LightGBM_pred", "lightgbm_prediction"),
        'CatBoost': ("catboost_stack_prediction", "CatBoost_pred", "catboost_prediction"),
        'RandomForest': ("randomforest_stack_prediction", "RandomForest_pred", "randomforest_prediction"),
        'LogisticRegression': ("logisticregression_stack_prediction", "LogisticRegression_pred", "logisticregression_prediction"),
    }

    all_cols = [c for cols in fallback_map.values() for c in cols]
    legacy_here = any((c in row) and pd.notna(row[c]) and int(row[c]) == 2 for c in all_cols)

    def _map_label(raw):
        try:
            r = int(raw)
        except Exception:
            return None
        if legacy_here:          # 1→LONG(0), 2→SHORT(1)
            if r == 1: return 0
            if r == 2: return 1
        if r in (-1, 0, 1):      # +1→0, -1→1, 0→None
            return {+1: 0, -1: 1}.get(r, None)
        return r if r in (0, 1) else None

    preds_bin, active_w = {}, {}
    for model, cols in fallback_map.items():
        raw = next((row.get(c) for c in cols if pd.notna(row.get(c, pd.NA))), None)
        mapped = _map_label(raw)
        if mapped is None:
            continue
        preds_bin[model] = mapped
        active_w[model] = weights.get(model, 0.0)

    if not active_w:
        return pd.NA

    weighted = sum(preds_bin[m] * active_w[m] for m in preds_bin)
    total_w = sum(active_w.values()) or 1.0
    return weighted / total_w

def _first_present_row(row, *names, default=pd.NA):
    for n in names:
        if n in row and not pd.isna(row[n]):
            return row[n]
    return default

# ======= SAFE ALIASES FOR STACK COLUMNS =======================================
def ensure_stack_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create symmetric aliases between TitleCase and short-prefix stack columns.
    We DO NOT fabricate stack columns from base columns; we only alias existing stacks.
    """
    out = df

    title2short = [
        # LightGBM
        ("LightGBM_stack_prediction",        "lgbm_stack_prediction"),
        ("LightGBM_stack_prob_class_1",      "lgbm_stack_prob_class_1"),
        ("LightGBM_stack_prob_class_0",      "lgbm_stack_prob_class_0"),
        ("LightGBM_stack_max_prob",          "lgbm_stack_max_prob"),
        ("LightGBM_stack_prob_diff",         "lgbm_stack_prob_diff"),
        # CatBoost
        ("CatBoost_stack_prediction",        "cat_stack_prediction"),
        ("CatBoost_stack_prob_class_1",      "cat_stack_prob_class_1"),
        ("CatBoost_stack_prob_class_0",      "cat_stack_prob_class_0"),
        ("CatBoost_stack_max_prob",          "cat_stack_max_prob"),
        ("CatBoost_stack_prob_diff",         "cat_stack_prob_diff"),
        # RandomForest
        ("RandomForest_stack_prediction",    "rf_stack_prediction"),
        ("RandomForest_stack_prob_class_1",  "rf_stack_prob_class_1"),
        ("RandomForest_stack_prob_class_0",  "rf_stack_prob_class_0"),
        ("RandomForest_stack_max_prob",      "rf_stack_max_prob"),
        ("RandomForest_stack_prob_diff",     "rf_stack_prob_diff"),
        # LogisticRegression
        ("LogisticRegression_stack_prediction",       "lr_stack_prediction"),
        ("LogisticRegression_stack_prob_class_1",     "lr_stack_prob_class_1"),
        ("LogisticRegression_stack_prob_class_0",     "lr_stack_prob_class_0"),
        ("LogisticRegression_stack_max_prob",         "lr_stack_max_prob"),
        ("LogisticRegression_stack_prob_diff",        "lr_stack_prob_diff"),
    ]

    # TitleCase → short
    for src, dst in title2short:
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src].values

    # short → TitleCase
    short2title = [(b, a) for (a, b) in title2short]
    for src, dst in short2title:
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src].values

    # Guard: warn if any stack == base (check only where both are non‑NaN)
    for base_col, stack_col in (("lgbm_pred","lgbm_stack_prediction"),
                                ("cat_pred","cat_stack_prediction"),
                                ("rf_pred","rf_stack_prediction"),
                                ("lr_pred","lr_stack_prediction")):
        if base_col in out.columns and stack_col in out.columns:
            b = pd.to_numeric(out[base_col],  errors="coerce")
            s = pd.to_numeric(out[stack_col], errors="coerce")
            m = b.notna() & s.notna()
            if m.any() and (b[m] == s[m]).all():
                logging.info("ℹ️ Stacked %s is identical to base %s for this run; skipping stacked for transparency.", stack_col, base_col)
    return out

def _export_hc_compat(ticker, df, out_dir, **kwargs):
    """
    Calls export_high_confidence_signals with the new signature (supports n_top),
    but falls back to the legacy 3-arg version if that's what's on the path.
    """
    try:
        return export_high_confidence_signals(ticker, df, out_dir, **kwargs)
    except TypeError as e:
        msg = str(e)
        if "unexpected keyword argument 'n_top'" in msg or "got an unexpected keyword argument" in msg:
            # Old signature: export_high_confidence_signals(ticker, df, out_dir)
            return export_high_confidence_signals(ticker, df, out_dir)
        raise

def get_pred_and_prob_cols(df: pd.DataFrame, model: str, prefer_stacked: bool = True):
    """Resolve the prediction & prob columns using canonical names from config."""
    # try stacked first if requested
    if prefer_stacked:
        sp = stack_pred_col(model)
        sp1 = stack_prob1_col(model)
        if sp in df.columns and sp1 in df.columns:
            return sp, sp1
        # also accept legacy case/title variants via resolve_stacked_pred_column if you keep it in config

    # fall back to base
    bp = pred_col(model)
    bp1 = prob1_col(model)
    pc = bp if bp in df.columns else (to_title(model) + "_pred" if (to_title(model) + "_pred") in df.columns else None)
    pr = bp1 if bp1 in df.columns else (to_title(model) + "_prob_class_1" if (to_title(model) + "_prob_class_1") in df.columns else None)
    return pc, pr


def compute_ensemble_direction_and_conf(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Returns (direction_series, confidence_series) from the normalized ensemble block.
    Direction is 0/1 (respects CLASS_IS_LONG), confidence is prob of chosen side.
    """
    if df is None or df.empty:
        return pd.Series(dtype="float64"), pd.Series(dtype="float64")
    if "ensemble_pred_engine" not in df.columns or "ensemble_conf" not in df.columns:
        df = ensure_ensemble_columns(df)
    return df["ensemble_pred_engine"], df["ensemble_conf"]

def export_high_confidence_signals(tkr: str, df: pd.DataFrame, out_dir: str, *, pred_col="ensemble_pred_engine", conf_col="ensemble_conf", top_k: int = 200) -> str | None:
    """
    Saves a thin, human-readable CSV of the top signals by confidence.
    Returns the path (string) or None.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    cols = _hc_export_cols(df)
    work = df.copy()
    # rank reliably even if conf is missing (fallback to max_prob)
    if conf_col in work.columns:
        work["_hc_rank"] = pd.to_numeric(work[conf_col], errors="coerce").fillna(0.0)
    elif "ensemble_max_prob" in work.columns:
        work["_hc_rank"] = pd.to_numeric(work["ensemble_max_prob"], errors="coerce").fillna(0.0)
    else:
        work["_hc_rank"] = 0.0
    work = work.sort_values("_hc_rank", ascending=False).head(int(top_k))

    os.makedirs(out_dir, exist_ok=True)
    safe_tkr = sanitize_filename(tkr)
    out_path = os.path.join(out_dir, f"{safe_tkr}_high_confidence_signals.csv")
    work[cols].to_csv(out_path, index=False, float_format="%.6g")
    return out_path

def _prep_and_validate_df(df, ticker):
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.set_index('date', inplace=True)
    df = df.sort_index(ascending=True)
    for col in ['close', 'atr']:
        if col not in df.columns:
            logging.warning(f"⚠️ {ticker} missing required column: {col}")
            return None
    df['atr'] = pd.to_numeric(df['atr'], errors='coerce')
    df = df[df['atr'].notna() & (df['atr'] > 0)].copy()
    for col in ['close', 'volume', 'atr']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df[df[col].notna()]
    if df.empty:
        logging.warning(f"⚠️ No valid data after numeric validation for {ticker}")
        return None
    return df

def _resolve_sl_tp(sl_mult, tp_mult):
    """Coalesce SL/TP multipliers to numeric values, falling back to config."""
    from config import atr_sl_multiplier as _sl_default, atr_tp_multiplier as _tp_default
    try:
        sl_m = float(sl_mult)
    except (TypeError, ValueError):
        sl_m = float(_sl_default)
    try:
        tp_m = float(tp_mult)
    except (TypeError, ValueError):
        tp_m = float(_tp_default)
    return sl_m, tp_m

def _try_open_position(
    row: pd.Series,
    current_capital: float,
    sl_mult: float,
    tp_mult: float,
    tx_cost: float,
    risk_pct: float,
    prediction_column: str,
    *,
    prob_column: str | None = None,
    use_ranking: bool = False,
    min_conf_floor: float | None = None,
):
    """
    Open a position if the row passes gating.
    Returns: (position_dict | None, [reasons], {entry_features})

    Engine convention: 0=LONG, 1=SHORT (prediction_column already normalized).
    We record 'entry_conf' (probability of the chosen side) at entry.
    """
    reasons: list[str] = []
    entry_feats: dict = {}

    # 0=LONG, 1=SHORT
    try:
        side01 = int(pd.to_numeric(row.get(prediction_column), errors="coerce"))
    except Exception:
        side01 = -1
    if side01 not in (0, 1):
        reasons.append("invalid_signal")
        return None, reasons, entry_feats

    action = "long" if side01 == 0 else "short"

    # side-aware entry confidence
    try:
        conf = resolve_side_confidence(
            row=row,
            prediction_column=prediction_column,
            side01=side01,
            fallback_col=(prob_column or "signal_conf")
        )
    except Exception:
        conf = float("nan")

    # In _try_open_position(...) right after computing `conf`:
    date_key = str(pd.to_datetime(row.name).date())
    dyn_floor = _daily_dynamic_floor(date_key, conf)
    if use_ranking and (min_conf_floor is not None):
        eff_floor = max(float(min_conf_floor), dyn_floor)
        if not _passes_conf(conf, eff_floor):
            reasons.append("below_dynamic_floor")
            return None, reasons, entry_feats

    # --- Inside _try_open_position(...) after you compute 'conf' and before sizing ---
    # Disagreement guard: require extra confidence when members disagree
    try:
        prefer_stacked = bool(getattr(CFG, "PREFER_STACKED_IF_AVAILABLE", False))
        ps = _row_family_short_probs(row, prefer_stacked=prefer_stacked)
        if len(ps) >= 3:
            dis_std = float(_np.nanstd(_np.array(ps, dtype=float)))
            cutoff = float(getattr(CFG, "DISAGREE_STD_CUTOFF", 0.20))
            lift   = float(getattr(CFG, "DISAGREE_CONF_LIFT", 0.03))
            if dis_std >= cutoff and use_ranking and (min_conf_floor is not None):
                # require extra confidence
                eff_floor = float(min_conf_floor) + lift
                if not _passes_conf(conf, eff_floor):
                    reasons.append("high_disagreement_low_conf")
                    return None, reasons, entry_feats
    except Exception:
        pass

    # Optional probability floor — only if a floor was explicitly provided
    if use_ranking and (min_conf_floor is not None):
        p = float(pd.to_numeric(conf, errors="coerce"))
        if not np.isfinite(p) or p + EPS < float(min_conf_floor):
            reasons.append("low_conf")
            return None, reasons, entry_feats
        entry_feats["entry_conf"] = float(p)
    else:
        if np.isfinite(conf):
            entry_feats["entry_conf"] = float(conf)


    # price & ATR
    price = float(pd.to_numeric(row.get("close"), errors="coerce"))
    atr   = float(pd.to_numeric(row.get("atr", row.get("ATR", np.nan)), errors="coerce"))
    if not np.isfinite(price):
        reasons.append("no_price")
        return None, reasons, entry_feats

    # brackets from fractions (use the overrides coming from the engine)
    sl_frac, tp_frac = get_bracket_fractions(
        price=price,
        atr=(atr if np.isfinite(atr) and atr > 0 else None),
        sl=None,
        tp=None,
        sl_mult_override=sl_mult,  # ← use caller override if provided
        tp_mult_override=tp_mult,  # ← use caller override if provided
    )
    sl_price, tp_price = compute_bracket_prices_from_fracs(
        side=action, base_price=price, sl_frac=sl_frac, tp_frac=tp_frac
    )


    stop_dist = (price - sl_price) if action == "long" else (sl_price - price)
    if stop_dist <= 0:
        reasons.append("invalid_brackets")
        return None, reasons, {"stop_distance": stop_dist}

    # risk-based position sizing
    risk_dollars = max(1e-9, float(current_capital) * float(risk_pct))
    qty = math.floor(risk_dollars / stop_dist)
    if qty <= 0:
        reasons.append("allocation_too_small")
        return None, reasons, {"stop_distance": stop_dist, "risk_dollars": risk_dollars}

    pos = {
        "entry_date": row.name,
        "type": action,                 # "long" / "short"
        "entry_price": price,
        "stop_loss": sl_price,
        "take_profit": tp_price,
        "qty": float(qty),
        "entry_conf": float(entry_feats.get("entry_conf", np.nan)) if "entry_conf" in entry_feats else np.nan,
        "features": {
            "rsi": float(pd.to_numeric(row.get("rsi"), errors="coerce")) if "rsi" in row else None,
            "macd": float(pd.to_numeric(row.get("macd_hist"), errors="coerce")) if "macd_hist" in row else None,
        },
    }
    return pos, reasons, entry_feats

def build_trade_signals(df: pd.DataFrame, signal_col: str, conf_col: str | None = None) -> pd.DataFrame:
    """
    Prepare a filtered DataFrame for the execution engine.
    Produces:
      • ml_signal (Int64): 0=LONG, 1=SHORT
      • signal_conf ([0,1]): confidence (probability of chosen side)
    """
    work = df.copy()

    # index → datetime if needed
    if not isinstance(work.index, pd.DatetimeIndex):
        idx_col = "date" if "date" in work.columns else None
        if idx_col:
            work[idx_col] = pd.to_datetime(work[idx_col], errors="coerce")
            work = work.dropna(subset=[idx_col]).set_index(idx_col).sort_index()
        else:
            work.index = pd.date_range("2000-01-01", periods=len(work), freq="D")

    # Ensure ATR exists; do NOT recompute RSI/MACD/EMA here
    if "atr" not in work.columns:
        if {"high", "low", "close"}.issubset(set(work.columns)):
            hl = (work["high"] - work["low"]).abs()
            hc = (work["high"] - work["close"].shift(1)).abs()
            lc = (work["low"]  - work["close"].shift(1)).abs()
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            work["atr"] = tr.rolling(14, min_periods=14).mean()
        else:
            work["atr"] = np.nan

    # Normalize the provided signal column to engine convention (0=LONG,1=SHORT)
    if signal_col not in work.columns:
        raise KeyError(f"Signal column '{signal_col}' not found in DataFrame.")
    engine_signal_col = map_predictions_to_binary(work, signal_col)
    work["ml_signal"] = work[engine_signal_col].astype("Int64")

    # Confidence column (normalize to [0,1] if needed)
    if conf_col and conf_col in work.columns:
        conf = pd.to_numeric(work[conf_col], errors="coerce")
        work["signal_conf"] = conf.where((conf >= 0) & (conf <= 1), conf / 100.0)
    else:
        work["signal_conf"] = 0.5

    # cleanup
    if work.index.has_duplicates:
        work = work[~work.index.duplicated(keep="last")]
    if not work.columns.is_unique:
        work = work.loc[:, ~work.columns.duplicated()]

    for c in ("close", "atr", "volume"):
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.dropna(subset=["close", "atr"])
    if "Close" in work.columns and "close" not in work.columns:
        work = work.rename(columns={"Close": "close"})

    return work

def run_sltp_gridsearch(df, sl_range=(0.8, 1.0, 1.2), tp_range=(1.8, 2.2, 2.8)):
    """
    Grid-search SL/TP using the ensemble engine column.
    NOTE: run_backtest returns (capital, trades_df, metrics, trades_file), so we must unpack it.
    """
    best_score = -np.inf
    best_params = None
    rows = []

    for sl in sl_range:
        for tp in tp_range:
            try:
                cap, trades_df, metrics, _ = run_backtest(
                    df.copy(),
                    ticker="GRIDTEST",
                    model="grid",
                    model_type="ensemble",
                    prediction_column="ensemble_pred_engine",
                    sl_mult=sl,
                    tp_mult=tp
                )
                sharpe = metrics.get('Sharpe', -np.inf)
                mdd = metrics.get('Max Drawdown', np.inf)
                score = sharpe - 0.1 * (mdd / 100.0)
            except Exception as e:
                logging.warning(f"⚠️ SL/TP ({sl},{tp}) failed: {e}")
                continue

            rows.append((sl, tp, sharpe, mdd, score))
            if score > best_score:
                best_score = score
                best_params = (sl, tp)

    cols = ["SL", "TP", "Sharpe", "MaxDD", "Score"]
    return best_params, pd.DataFrame(rows, columns=cols)

def _standardize_position(position):
    """
    Ensures position is a dict. If a tuple, extracts the first element.
    Raises ValueError if not a dict after extraction.
    """
    if isinstance(position, tuple):
        position = position[0]
    if not isinstance(position, dict):
        raise ValueError(f"Position must be a dict or tuple containing a dict, got {type(position)}: {position}")
    return position

def _ensure_position_size(position: dict, default_risk_pct: float = 0.02, log_prefix: str = "") -> float:
    """
    Ensure 'size' exists on the position. If missing, reconstruct a sensible quantity
    from capital_at_entry, entry_price, stop_loss and trade type. If reconstruction
    is impossible, set to 0.0 (and log a warning).
    """
    try:
        if "size" in position and position["size"] is not None:
            return float(position["size"])
    except Exception:
        pass

    entry_price = float(position.get("entry_price", float("nan")))
    stop_loss   = float(position.get("stop_loss", float("nan")))
    trade_type  = position.get("type")
    cap_entry   = float(position.get("capital_at_entry", float("nan")))

    if np.isfinite(entry_price) and np.isfinite(stop_loss) and trade_type in ("long", "short") and np.isfinite(cap_entry):
        denom = (entry_price - stop_loss) if trade_type == "long" else (stop_loss - entry_price)
        if denom > 0:
            qty = (default_risk_pct * cap_entry) / denom
            position["size"] = float(qty)
            logging.warning(f"{log_prefix} Reconstructed missing 'size' as {qty:.6f} using risk_pct={default_risk_pct}")
            return qty
        else:
            logging.warning(f"{log_prefix} Cannot reconstruct 'size' because denom<=0 (entry={entry_price:.4f}, sl={stop_loss:.4f})")
    else:
        logging.warning(f"{log_prefix} Cannot reconstruct 'size' — insufficient fields in position: {position}")

    position["size"] = 0.0
    return 0.0
def _update_and_exit_positions(
    positions: list[dict],
    row: pd.Series,
    current_date: pd.Timestamp,
    *,
    use_trailing_stop: bool,
    trail_pct: float,
    sl_mult: float,
    tp_mult: float,
    tx_cost: float,
    prediction_column: str,
):
    """
    Update unrealized PnL, move trailing stops, and close positions on SL/TP/time.
    Returns: (newly_closed, still_open)
      where each element of newly_closed is:
        {"trade": <trade dict>, "trade_return": <float>, "pnl": <float>}
    """

    price = float(pd.to_numeric(row.get("close"), errors="coerce"))
    atr   = float(pd.to_numeric(row.get("atr", row.get("atr14", np.nan)), errors="coerce"))

    newly_closed: list[dict] = []
    still_open:   list[dict] = []

    for pos in positions:
        side = pos.get("type", "").lower()   # 'long' or 'short'
        qty  = float(pos.get("qty", 0.0))
        entry = float(pos.get("entry_price", np.nan))
        sl    = float(pos.get("stop_loss", pos.get("sl", np.nan)))
        tp    = float(pos.get("take_profit", pos.get("tp", np.nan)))

        if not (np.isfinite(price) and np.isfinite(entry) and qty > 0):
            still_open.append(pos)
            continue

        # trailing stop logic
        if use_trailing_stop and np.isfinite(atr) and atr > 0:
            if side == "long" and price > entry:
                new_sl = max(sl, price - trail_pct * atr)
                pos["stop_loss"] = new_sl
                sl = new_sl
            elif side == "short" and price < entry:
                new_sl = min(sl, price + trail_pct * atr)
                pos["stop_loss"] = new_sl
                sl = new_sl

        # exit checks
        hit_tp = (price >= tp) if side == "long" else (price <= tp)
        hit_sl = (price <= sl) if side == "long" else (price >= sl)

        if hit_tp or hit_sl:
            exit_px = price
            if side == "long":
                gross_ret = (exit_px - entry) / entry
                pnl = qty * (exit_px * (1 - tx_cost) - entry * (1 + tx_cost))
            else:
                gross_ret = (entry - exit_px) / entry
                pnl = qty * (entry * (1 - tx_cost) - exit_px * (1 + tx_cost))

            trade = {
                "trade_id": _next_trade_id(),
                "ticker": pos.get("ticker"),
                "type": side,
                "entry_date": pos.get("entry_date"),
                "exit_date": current_date,
                "entry_price": entry,
                "exit_price": exit_px,
                "qty": qty,
                "sl": sl,
                "tp": tp,
                "reason": "TAKE_PROFIT" if hit_tp else "STOP_LOSS",
                "return_pct": gross_ret * 100.0,
                "pnl": float(pnl),
                "days_held": int(pos.get("days_held", 0)) + 1,
                "rsi_at_entry": pos.get("features", {}).get("rsi"),
                "macd_at_entry": pos.get("features", {}).get("macd"),
                "entry_conf": pos.get("entry_conf"),   # <-- NEW
            }
            newly_closed.append({"trade": trade, "trade_return": gross_ret, "pnl": float(pnl)})
        else:
            # continue holding
            pos["days_held"] = int(pos.get("days_held", 0)) + 1
            still_open.append(pos)

    return newly_closed, still_open

def _update_and_exit_positions_core(current_row, open_positions, closed_positions, cfg):
    """
    Update unrealized PnL, advance trailing stops, and close positions
    if SL/TP/time-based criteria are met.
    """
    price = float(current_row["close"])
    atr = float(current_row.get("atr14", current_row.get("atr", 0.0)))
    allow_trailing = np.isfinite(atr) and atr > 0.0

    lock_mult = float(cfg.get("ATR_LOCK_MULTIPLIER", 1.0))
    trail_mult = float(cfg.get("TRAILING_STOP_MULTIPLIER", 1.0))
    max_hold = int(cfg.get("MAX_HOLD_DAYS", 65))

    to_close = []

    for pid, pos in open_positions.items():
        side = pos["side"]
        qty = pos["qty"]
        entry = pos["open_price"]
        stop_loss = pos["stop_loss"]
        take_profit = pos["take_profit"]

        # unrealized PnL (signed)
        if side == "LONG":
            unreal = (price - entry) * qty
            in_profit = price > entry
            hit_tp = price >= take_profit
            hit_sl = price <= stop_loss
            # advanced trailing when in profit
            if allow_trailing and in_profit:
                gain = price - entry
                # lock to BE once gain >= lock_mult * ATR
                if gain >= lock_mult * atr and stop_loss < entry:
                    pos["stop_loss"] = entry
                    stop_loss = entry
                # trail by trail_mult*ATR behind price
                new_trail = price - trail_mult * atr
                if new_trail > stop_loss:
                    pos["stop_loss"] = new_trail
                    stop_loss = new_trail
        else:  # SHORT
            unreal = (entry - price) * qty
            in_profit = price < entry
            hit_tp = price <= take_profit
            hit_sl = price >= stop_loss
            if allow_trailing and in_profit:
                gain = entry - price
                if gain >= lock_mult * atr and stop_loss > entry:
                    pos["stop_loss"] = entry
                    stop_loss = entry
                new_trail = price + trail_mult * atr
                if new_trail < stop_loss:
                    pos["stop_loss"] = new_trail
                    stop_loss = new_trail

        pos["pnl"] = unreal
        pos["days_held"] = pos.get("days_held", 0) + 1

        # exit decision order: TP → SL/Trail → time
        if hit_tp:
            to_close.append((pid, "TAKE_PROFIT"))
            continue
        if hit_sl:
            to_close.append((pid, "STOP_LOSS" if pos["stop_loss"] != entry else "LOCKED_AT_BREAKEVEN"))
            continue
        if pos["days_held"] >= max_hold:
            to_close.append((pid, "MAX_HOLD_DAYS"))

    # realize exits
    for pid, reason in to_close:
        pos = open_positions.pop(pid, None)
        if pos is None:
            continue
        side = pos["side"]
        qty = pos["qty"]
        entry = pos["open_price"]
        # exit at closing price for backtest (your dataset is EOD / close-driven)
        exit_px = price
        pnl = (exit_px - entry) * qty if side == "LONG" else (entry - exit_px) * qty
        pos.update({
            "exit_date": current_row.name,
            "exit_price": exit_px,
            "pnl": pnl,
            "reason": reason
        })
        closed_positions.append(pos)

def _close_position_final(position: dict,
                          df: pd.DataFrame,
                          transaction_cost: float):
    """
    Force-close any open position at the last available bar.
    Robust to:
      • price column casing: 'close' or 'Close'
      • position shape: size/qty, stop_loss/sl, take_profit/tp
      • type casing: 'long'/'short' or 'LONG'/'SHORT'
    Returns: (trade_dict, trade_return_pct, pnl_value)
    """

    if df is None or df.empty:
        return None, 0.0, 0.0

    # --- Resolve price & date from DataFrame
    price_col = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else None)
    if price_col is None:
        raise KeyError("No 'close' or 'Close' column on DataFrame for final close.")

    last_idx = df.index[-1]
    last_close = float(pd.to_numeric(df.loc[last_idx, price_col], errors="coerce"))
    if not np.isfinite(last_close):
        raise ValueError(f"Final close price is NaN at index {last_idx}")

    exit_date = df.loc[last_idx, "date"] if "date" in df.columns else last_idx

    # --- Normalize position fields
    ptype_raw = position.get("type", "")
    ptype = str(ptype_raw).lower()
    if ptype not in ("long", "short"):
        # tolerate legacy upper-case
        if str(ptype_raw).upper() in ("LONG", "SHORT"):
            ptype = str(ptype_raw).upper().lower()
        else:
            raise ValueError(f"Unknown position type: {ptype_raw}")

    entry_price = float(position.get("entry_price", np.nan))
    if not np.isfinite(entry_price):
        raise ValueError(f"Missing/invalid entry_price on position: {position}")

    # prefer normalized keys; fall back to legacy
    qty = position.get("size", position.get("qty", 0.0))
    qty = float(qty) if np.isfinite(float(qty)) else 0.0
    if qty <= 0:
        # reconstruct a sensible quantity if possible
        entry_cap = float(position.get("capital_at_entry", np.nan))
        sl = position.get("stop_loss", position.get("sl", np.nan))
        if np.isfinite(entry_cap) and np.isfinite(entry_price) and np.isfinite(sl):
            denom = (entry_price - float(sl)) if ptype == "long" else (float(sl) - entry_price)
            if np.isfinite(denom) and denom > 0:
                qty = 0.02 * entry_cap / denom  # fallback risk_pct = 2%
        if not np.isfinite(qty) or qty <= 0:
            qty = 0.0  # last resort, close with zero PnL contribution

    # --- Compute PnL net of transaction costs (both legs)
    if ptype == "long":
        gross_ret = (last_close - entry_price) / entry_price
        trade_return = gross_ret
        # apply tx cost on entry and exit
        pnl = qty * (last_close * (1 - transaction_cost) - entry_price * (1 + transaction_cost))
    else:  # short
        gross_ret = (entry_price - last_close) / entry_price
        trade_return = gross_ret
        pnl = qty * (entry_price * (1 - transaction_cost) - last_close * (1 + transaction_cost))

    trade = {
        'trade_id': _next_trade_id(),
        "ticker": position.get("ticker"),
        "type": ptype,
        "entry_date": position.get("entry_date"),
        "exit_date": exit_date,
        "entry_price": entry_price,
        "exit_price": last_close,
        "qty": qty,
        "sl": position.get("stop_loss", position.get("sl")),
        "tp": position.get("take_profit", position.get("tp")),
        "reason": "final_close",
        "return_pct": trade_return * 100.0,
        "pnl": float(pnl),
        "days_held": (pd.Timestamp(exit_date) - pd.Timestamp(position.get("entry_date"))).days
                     if position.get("entry_date") is not None else 0,
        "rsi_at_entry": position.get("rsi_at_entry", position.get("features", {}).get("rsi")),
        "macd_at_entry": position.get("macd_at_entry", position.get("features", {}).get("macd")),
        "entry_conf": position.get("entry_conf"),  # <-- NEW
    }

    return trade, trade_return, float(pnl)

def _build_equity_curve(
    initial_capital: float,
    trades_df: pd.DataFrame,
    price_df: pd.DataFrame
) -> pd.Series:
    """
    Mark-to-market equity curve over the *daily* price timeline (vectorized).
    """
    if price_df is None or price_df.empty:
        return pd.Series([float(initial_capital)], index=pd.Index([pd.Timestamp.today()], name="date"))

    # Build a daily index from the full price_df (not just filtered signals)
    if isinstance(price_df.index, pd.DatetimeIndex):
        idx = price_df.index.sort_values().unique()
    elif "date" in price_df.columns:
        idx = pd.to_datetime(price_df["date"], errors="coerce").dropna().sort_values().unique()
    else:
        idx = pd.date_range("2000-01-01", periods=len(price_df), freq="D")

    equity_index = pd.Index(idx, name="date")
    daily_factor = pd.Series(1.0, index=equity_index)

    if trades_df is not None and not trades_df.empty and "return_pct" in trades_df.columns:
        td = trades_df.copy()
        td["exit_date"] = pd.to_datetime(td.get("exit_date", td.index), errors="coerce")
        td = td.dropna(subset=["exit_date", "return_pct"]).copy()
        if not td.empty:
            td["factor"] = 1.0 + (pd.to_numeric(td["return_pct"], errors="coerce") / 100.0)
            grouped = td.groupby(td["exit_date"].dt.normalize())["factor"].prod()
            # align onto equity_index (missing days → factor 1.0)
            daily_factor = daily_factor.mul(grouped.reindex(equity_index).fillna(1.0), fill_value=1.0)

    # cumulative product of daily factors
    equity_curve = float(initial_capital) * daily_factor.cumprod()
    # hygiene
    equity_curve = equity_curve.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(initial_capital)
    equity_curve = equity_curve.clip(lower=0.0)
    return equity_curve

def _compute_performance_metrics(
    trades: list[dict],
    trades_df: pd.DataFrame,
    initial_capital: float,
    current_capital: float,
    price_df: pd.DataFrame,
    equity_curve_override: pd.Series | None = None,   # <-- NEW
) -> dict:

    # --- Equity & return stats ---
    if equity_curve_override is not None and len(equity_curve_override) > 1:
        equity_curve = equity_curve_override.sort_index()
    else:
        equity_curve = _build_equity_curve(initial_capital, trades_df, price_df)

    model_return_pct = (current_capital - initial_capital) / max(initial_capital, 1e-8) * 100.0

    try:
        rf_annual = float(getenv("RISK_FREE_ANNUAL", "0.015"))
        rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0
    except Exception:
        rf_daily = 0.0001

    daily_return = equity_curve.pct_change().dropna()
    sharpe = sharpe_ratio(daily_return, rf=rf_daily, periods_per_year=252, ddof=1) if not daily_return.empty else 0.0

    dd_series = (equity_curve / equity_curve.cummax()) - 1.0
    max_drawdown_pct = -100.0 * float(dd_series.min()) if len(dd_series) else 0.0

    # --- Trade stats (unchanged) ---
    if trades_df is None or trades_df.empty:
        wins = losses = pd.DataFrame()
        ntrades = 0
    else:
        wins   = trades_df[trades_df.get("pnl", 0) > 0]
        losses = trades_df[trades_df.get("pnl", 0) <= 0]
        ntrades = int(len(trades_df))

    win_rate     = (len(wins) / ntrades * 100.0) if ntrades else 0.0
    avg_gain     = float(wins["return_pct"].mean()) if not wins.empty else 0.0
    avg_loss     = float(losses["return_pct"].mean()) if not losses.empty else 0.0
    avg_duration = float(trades_df["days_held"].mean()) if (ntrades and "days_held" in trades_df.columns) else 0.0

    total_longs = total_shorts = 0
    if ntrades:
        side_col = "type" if "type" in trades_df.columns else ("position_type" if "position_type" in trades_df.columns else None)
        if side_col:
            side = trades_df[side_col].astype(str).str.lower()
            total_longs  = int((side == "long").sum())
            total_shorts = int((side == "short").sum())

    if ntrades and "entry_conf" in trades_df.columns:
        s = pd.to_numeric(trades_df["entry_conf"], errors="coerce")
        avg_prob_pct = float(s.mul(100.0).mean(skipna=True)) if s.notna().any() else 0.0
    else:
        avg_prob_pct = 0.0

    p_win = (len(wins) / ntrades) if ntrades else 0.0
    expectancy = p_win * avg_gain + (1.0 - p_win) * avg_loss

    return {
        "Sharpe": round(float(sharpe or 0.0), 2),
        "Expectancy (%)": round(float(expectancy), 2),
        "Model Return (%)": round(float(model_return_pct), 2),
        "Max Drawdown (%)": round(float(max_drawdown_pct), 2),
        "Win Rate (%)": round(float(win_rate), 2),
        "Avg Gain (%)": round(float(avg_gain), 2),
        "Avg Loss (%)": round(float(avg_loss), 2),
        "Avg Duration (days)": round(float(avg_duration), 2),
        "Average Prediction/Probability (%)": round(float(avg_prob_pct), 2),
        "Total Longs": int(total_longs),
        "Total Shorts": int(total_shorts),
    }

def _empty_metrics():
    return {
        "Sharpe": 0.0,
        "Expectancy (%)": 0.0,
        "Model Return (%)": 0.0,
        "Max Drawdown (%)": 0.0,
        "Win Rate (%)": 0.0,
        "Avg Gain (%)": 0.0,
        "Avg Loss (%)": 0.0,
        "Avg Duration (days)": 0.0,
    }

def preprocess_data(df, ticker):
    """Preprocess the data (filter by date range, remove NaN, set index, ensure columns)."""
    if df is None or df.empty:
        logging.warning(f"No data provided for {ticker}. Skipping.")
        return None

    logging.warning(f"Raw 'close' column before conversion: {df['close'].head() if 'close' in df.columns else 'N/A'}")

    # === Validate 'close' column ===
    if 'close' not in df.columns:
        logging.error(f"'close' column missing in input data for {ticker}.")
        return None

    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df[df['close'].notna()]
    if df.empty:
        logging.warning(f"No valid 'close' data after preprocessing for {ticker}. Skipping.")
        return None

    # === Recover or validate 'date' ===
    if 'date' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df['date'] = df.index
        elif 'Unnamed: 0' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['Unnamed: 0'], errors='coerce')
            except Exception as e:
                logging.error(f"⚠️ Unable to parse date from 'Unnamed: 0' for {ticker}: {e}")
                return None
        else:
            logging.error(f"'date' column missing and index is not datetime for {ticker}. Skipping.")
            return None

    # === Convert and clean 'date' ===
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].notna()]
    df = df.drop_duplicates(subset='date')
    df.set_index('date', inplace=True)

    # Final clean-up
    df = df[~df.index.isnull()]
    if df.empty or df.index.isnull().all():
        logging.error(f"All dates are invalid in the data for {ticker}. Skipping.")
        return None

    df = df.sort_index()

    logging.debug(f"Raw 'close' column before conversion: {df['close'].head() if 'close' in df.columns else 'N/A'}")
    # ...
    logging.debug(f"✅ Preprocessed data for {ticker} — {df.shape[0]} rows:\n{df.head()}")
    logging.debug(f"Date range before filtering for {ticker}: {df.index.min()} to {df.index.max()}")
    logging.debug(f"Filtering data between {test_start} and {test_end}.")
    logging.debug(f"Rows after date filtering for {ticker}: {df.shape[0]}")

    # === Filter by date range ===
    df = df[(df.index >= test_start) & (df.index <= test_end)]
    logging.warning(f"Rows after date filtering for {ticker}: {df.shape[0]}")
    if df.empty:
        logging.warning(f"No valid data after date filtering for {ticker}. Skipping.")
        return None

    # === Ensure required columns ===
    if 'QuantML Score' not in df.columns:
        logging.warning("'QuantML Score' column is missing, filling with 0.")
        df['QuantML Score'] = 0

    logging.warning(f"✅ Preprocessed data for {ticker} — {df.shape[0]} rows:\n{df.head()}")
    return df


def check_missing_features(df, ticker, model_type, use_stack=False):
    """
    Checks if all expected features exist in df. Matches the new lowercase-first column files.
    """
    model_dir = model_path_stacked if use_stack else model_path_base
    lower = model_type.lower()
    proper = (globals().get("model_name_map", {}) or {
        "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
        "randomforest": "RandomForest", "logisticregression": "LogisticRegression",
    }).get(lower, model_type)

    if use_stack:
        cols_candidates = [
            os.path.join(model_dir, f"{ticker}_{lower}_stack_scaler_columns.pkl"),
            os.path.join(model_dir, f"{ticker}_{proper}_stack_scaler_columns.pkl"),
        ]
    else:
        cols_candidates = [
            os.path.join(model_dir, f"{ticker}_{lower}_scaler_columns.pkl"),
            os.path.join(model_dir, f"{ticker}_{proper}_scaler_columns.pkl"),
        ]

    cols_path = next((p for p in cols_candidates if os.path.exists(p)), None)
    if cols_path is None:
        logging.warning(f"⚠️ Feature column file not found (tried): {cols_candidates}")
        return True

    try:
        with open(cols_path, "rb") as f:
            expected = pickle.load(f)
        if not isinstance(expected, (list, tuple, set)):
            logging.error(f"❌ Expected-features file is not a list/tuple/set: {cols_path}")
            return True
    except Exception as e:
        logging.error(f"❌ Error loading expected features from {cols_path}: {e}")
        return True

    missing = [c for c in expected if c not in df.columns]
    if missing:
        found = [c for c in expected if c in df.columns]
        logging.warning(f"⚠️ Missing features for {ticker} ({model_type}): {missing}. {len(found)} found, {len(missing)} missing.")
        return True
    return False

def add_base_prediction_for_family(
    df: pd.DataFrame,
    ticker: str,
    family: str,
    models_dir: Path | str = Path("3.Models_base"),
) -> pd.DataFrame:
    """
    Thin wrapper so run_base_models can stay simple.
    Accepts 'xgb'/'lgbm'/'cat'/'rf'/'lr' or long names; delegates to the robust
    add_base_model_predictions_single() which uses load_model_and_scaler().
    Idempotent: if the base *_pred column already exists, it just returns df.
    """
    fam_key = CFG.to_prefix(family)  # -> 'xgb','lgbm','cat','rf','lr'
    return add_base_model_predictions_single(df, ticker, model_type=fam_key, use_stack=False)

def add_base_model_predictions_single(
    df: pd.DataFrame,
    ticker: str,
    model_type: str | None = None,
    *,
    model_key: str | None = None,          # back‑compat keyword
    use_stack: bool = False
) -> pd.DataFrame:
    """
    Add ONE base model's predictions into `df`.

    Accepts either `model_type` (preferred) or `model_key` (legacy).
    Writes ProperCase columns (e.g., XGBoost_pred, XGBoost_max_prob, ...).
    Also writes xgb_* filter aliases when the model is XGBoost.
    Idempotent: if the *_pred column already exists, returns `df` unchanged.
    """

    # ---- normalize incoming key
    mkey = (model_type or model_key or "").strip()
    if not mkey:
        logging.error("add_base_model_predictions_single: missing model_type/model_key")
        return df

    longlower_map = {
        "xgb": "xgboost", "xgboost": "xgboost",
        "lgbm": "lightgbm", "lightgbm": "lightgbm",
        "cat": "catboost", "catboost": "catboost",
        "rf": "randomforest", "randomforest": "randomforest", "random_forest": "randomforest",
        "lr": "logisticregression", "logreg": "logisticregression", "logistic": "logisticregression",
        "logisticregression": "logisticregression",
    }
    proper_map = {
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "randomforest": "RandomForest",
        "logisticregression": "LogisticRegression",
    }

    key = mkey.lower().replace(" ", "")
    long = longlower_map.get(key, key)
    proper = proper_map.get(long, mkey)

    # ---- idempotent skip
    pred_col = f"{proper}_pred"
    if pred_col in df.columns:
        logging.info(f"✅ Base prediction for {proper} already exists in {ticker} — skipping.")
        return ensure_xgb_aliases(df)

    # ---- load artifacts
    model, scaler, expected_cols = load_model_and_scaler(ticker, long, use_stack=use_stack)
    if model is None or not expected_cols:
        logging.warning(f"⚠️ Missing components/columns for {proper} ({ticker}); skipping.")
        return df

    # strip meta columns for *base* models
    if not use_stack:
        expected_cols = [
            c for c in expected_cols
            if not any(x in str(c).lower() for x in ("_pred", "_prob", "_stack"))
        ]

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        logging.error(f"❌ Missing input features for {proper} ({ticker}): "
                      f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
        return df

    # ---- build X and predict
    X = df[expected_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if X.empty:
        logging.warning(f"⚠️ No valid rows to predict for base {proper} ({ticker})")
        return df

    is_pipeline = hasattr(model, "named_steps")
    X_in = X if is_pipeline else (scaler.transform(X) if scaler is not None else X)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_in)
        preds = model.predict(X_in)
    else:
        preds = model.predict(X_in)
        probs = None

    # ---- write outputs (ProperCase)
    df.loc[X.index, pred_col] = preds
    # --- helper: find a classifier with classes_ (pipeline-safe)
    def _find_classes(model):
        if hasattr(model, "classes_"):
            return list(model.classes_)
        if hasattr(model, "named_steps"):
            # walk from the end; first step with classes_ wins
            for _, step in list(model.named_steps.items())[::-1]:
                if hasattr(step, "classes_"):
                    return list(step.classes_)
        return None

    # ... inside add_base_model_predictions_single(...)
    if probs is not None:
        classes = _find_classes(model)
        LONG_IDX  = int(getattr(CFG, "CLASS_IS_LONG", 0))
        SHORT_IDX = 1 - LONG_IDX

        # derive P(LONG)/P(SHORT) robustly from classes_
        if classes is not None and all(x in classes for x in (LONG_IDX, SHORT_IDX)):
            idx_long  = classes.index(LONG_IDX)
            idx_short = classes.index(SHORT_IDX)
            p_long    = probs[:, idx_long]
            p_short   = probs[:, idx_short]
        else:
            # Fallback: assume column 0 ~= LONG, column 1 ~= SHORT
            p_long = probs[:, 0] if probs.shape[1] >= 1 else np.full(len(X), 0.5)
            p_short = probs[:, 1] if probs.shape[1] >= 2 else (1.0 - p_long)

        # Always persist with engine semantics:
        #   *_prob_class_0 == P(LONG), *_prob_class_1 == P(SHORT)
        df.loc[X.index, f"{proper}_prob_class_0"] = p_long
        df.loc[X.index, f"{proper}_prob_class_1"] = p_short

        # Diagnostics
        df.loc[X.index, f"{proper}_max_prob"]  = np.maximum(p_long, p_short)
        df.loc[X.index, f"{proper}_prob_diff"] = np.abs(p_long - p_short)

    # Normalize predicted labels to engine 0/1 (even if model returns {-1,1} or {1,2})
    df.loc[X.index, pred_col] = _map_pred_to_engine(df.loc[X.index, pred_col])
    # ---- xgb_* filter aliases used by the signal filters
    if long == "xgboost":
        if "xgb_max_prob" not in df.columns and f"{proper}_max_prob" in df.columns:
            df.loc[X.index, "xgb_max_prob"] = df.loc[X.index, f"{proper}_max_prob"]
        if "xgb_prob_diff" not in df.columns and f"{proper}_prob_diff" in df.columns:
            df.loc[X.index, "xgb_prob_diff"] = df.loc[X.index, f"{proper}_prob_diff"]

    return ensure_xgb_aliases(df)

def add_all_base_model_predictions(df, ticker, model_types):
    """
    Adds predictions for all base models to df (needed for stacked model features).
    Skips models that already have predictions.
    """
    for model in model_types:
        df = add_base_model_predictions_single(df, ticker, model_type=model, use_stack=False)
    return df

def _find_classes(model):
    if hasattr(model, "classes_"):
        return list(model.classes_)
    if hasattr(model, "named_steps"):
        # walk from the end; first step with classes_ wins
        for _, step in list(model.named_steps.items())[::-1]:
            if hasattr(step, "classes_"):
                return list(step.classes_)
    return None

def _map_pred_to_engine(y):
    # keep the input index if Series; else create a clean Series
    s = pd.to_numeric(y if isinstance(y, pd.Series) else pd.Series(y), errors="coerce")
    vals = set(int(v) for v in s.dropna().unique())
    if vals.issubset({0, 1}):   return s.astype("Int64")
    if vals.issubset({-1, 1}):  return s.map({+1: 0, -1: 1}).astype("Int64")
    if vals.issubset({1, 2}):   return s.map({1: 0, 2: 1}).astype("Int64")
    return s.astype("Int64")


# --- canonical stacked inference (uses config + loader) ----------------------
def add_stacked_prediction_for_family(
    df: pd.DataFrame,
    ticker: str,
    family: str,
    *,
    models_dir: Path | str = Path("3.Models_stacked")  # kept for signature compat; not directly used
) -> pd.DataFrame:
    """
    Add ONE stacked model's predictions to df using the robust loader that honors
    config.STACKED_FILE_TEMPLATES. Writes canonical columns so downstream filters
    and the ensemble see them.

    Emits (short-prefix canonical):
      <pfx>_stack_prediction
      <pfx>_stack_prob_class_0
      <pfx>_stack_prob_class_1
      <pfx>_stack_max_prob
      <pfx>_stack_prob_diff

    Also, ensure TitleCase aliases exist (via ensure_stack_aliases) so older code still works.
    """

    # Normalize family -> short prefix ('lgbm','cat','rf','lr') and long key ('lightgbm',...)
    pfx = CFG.to_prefix(family)
    long_key = CFG.key_from_prefix(pfx)  # e.g., 'lgbm' -> 'lightgbm'【12:file-4uvgiiHm9nAXk9NufgwgKg†config.py†key_from_prefix…】

    # If we already have the stacked pred column, bail quickly (idempotent)
    stack_pred_name = CFG.stack_pred_col(pfx)  # e.g., 'lgbm_stack_prediction'【1:file-BXAj2ihmkVCx9bfP46HTQA†config.py†stack_pred_col…】
    if stack_pred_name in df.columns:
        return ensure_stack_aliases(df)

    # Load calibrated pipeline (preferred) + expected feature cols
    model, scaler, expected_cols = load_model_and_scaler(ticker, long_key, use_stack=True)
    if model is None:
        # Don't raise; just log so callers can continue on other families
        logging.warning("⚠️ No stacked model found for %s (%s)", ticker, pfx)
        return df

    # Fallback if expected feature list wasn’t persisted with the model
    if not expected_cols:
        expected_cols = load_stack_feature_cols(
            ticker, df,
            default_cols=[
                # Probs/max/diff from base models + base *_pred (common training recipe)
                "xgb_prob_class_1","xgb_prob_diff","xgb_max_prob",
                "lgbm_prob_class_1","lgbm_prob_diff","lgbm_max_prob",
                "cat_prob_class_1","cat_prob_diff","cat_max_prob",
                "rf_prob_class_1","rf_prob_diff","rf_max_prob",
                "lr_prob_class_1","lr_prob_diff","lr_max_prob",
                "xgb_pred","lgbm_pred","cat_pred","rf_pred","lr_pred",
            ]
        )

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        logging.error("❌ Missing features in DataFrame for %s (%s): %s%s",
                      long_key, ticker, missing[:10], "..." if len(missing) > 10 else "")
        return df

    # Prepare inputs
    X_raw = df[expected_cols].replace([np.inf, -np.inf], np.nan)
    is_pipeline = hasattr(model, "named_steps") or hasattr(getattr(model, "base_estimator", None), "named_steps")
    if is_pipeline:
        X_in, X_idx = X_raw, X_raw.index
    else:
        X_clean = X_raw.dropna()
        X_in = scaler.transform(X_clean) if scaler is not None else X_clean
        X_idx = X_clean.index

    # Predict
    preds = model.predict(X_in)
    probs = model.predict_proba(X_in) if hasattr(model, "predict_proba") else None

    # Write canonical short-prefix columns
    df.loc[X_idx, CFG.stack_pred_col(pfx)] = preds
    if probs is not None:
        def _find_classes(model):
            if hasattr(model, "classes_"):
                return list(model.classes_)
            if hasattr(model, "named_steps"):
                for _, step in list(getattr(model, "named_steps", {}).items())[::-1]:
                    if hasattr(step, "classes_"):
                        return list(step.classes_)
            return None

        classes  = _find_classes(model)
        LONG_IDX = int(getattr(CFG, "CLASS_IS_LONG", 0))
        SHORT_IDX = 1 - LONG_IDX

        if classes is not None and all(x in classes for x in (LONG_IDX, SHORT_IDX)):
            idx_long  = classes.index(LONG_IDX)
            idx_short = classes.index(SHORT_IDX)
            p_long    = probs[:, idx_long]
            p_short   = probs[:, idx_short]
        else:
            p_long, p_short = probs[:, 0], probs[:, 1]

        # For stacked, keep the canonical short-prefix names
        df.loc[X_idx, CFG.stack_prob0_col(pfx)] = p_long if LONG_IDX == 0 else p_short
        df.loc[X_idx, CFG.stack_prob1_col(pfx)] = p_short if LONG_IDX == 1 else p_long
        df.loc[X_idx, CFG.stack_max_prob_col(pfx)]  = np.maximum(p_long, p_short)
        df.loc[X_idx, CFG.stack_prob_diff_col(pfx)] = np.abs(p_long - p_short)

    # Maintain aliases (short + TitleCase) so older code keeps working
    df = ensure_stack_aliases(df)

    # Helpful diagnostics (compare base vs stacked direction if base exists)
    base_pred = f"{pfx}_pred"
    if base_pred in df.columns:
        try:
            a = df.loc[X_idx, stack_pred_name].astype("Int64")
            b = df.loc[X_idx, base_pred].astype("Int64")
            mask = a.notna() & b.notna()
            if mask.any():
                ident = float((a[mask] == b[mask]).mean())
                # Demote to INFO and only log once per (ticker,family)
                if ident == 1.0:
                    _log_once(
                        logging.INFO,
                        key=f"ident_stack_base:{ticker}:{pfx}",
                        message=(f"ℹ️ {ticker} — {CFG.to_title(pfx)} (stacked) "
                                 f"is identical to base {base_pred} on this run; "
                                 f"stacked backtest will be skipped.")
                    )
        except Exception:
            pass

    logging.info("✅ Added stacked predictions for %s to DataFrame.", pfx)
    return df


def run_stacked_models(
    ticker: str,
    features_path_or_df: str | Path | pd.DataFrame,
    stacked_models: list[str] | tuple[str, ...] | None = None,
    models_dir: str | Path = "3.Models_stacked",
) -> pd.DataFrame:
    """
    Ensure stacked-model prediction columns exist for the given ticker.
    Uses the canonical loader + canonical column names.
    """
    if isinstance(features_path_or_df, (str, Path)):
        df = pd.read_csv(features_path_or_df)
    else:
        df = features_path_or_df

    if not stacked_models:
        # Only families for which you actually save stacked artifacts
        stacked_models = ["lightgbm", "catboost", "randomforest", "logisticregression"]

    wrote: list[str] = []
    for fam in [CFG.to_prefix(f) for f in stacked_models]:
        before = set(df.columns)
        df = add_stacked_prediction_for_family(df, ticker, fam, models_dir=models_dir)
        if set(df.columns) - before:
            wrote.append(fam)

    if wrote:
        logging.info("✅ %s — added stacked predictions: %s", ticker, ", ".join(wrote))
    else:
        logging.info("ℹ️ %s — no stacked predictions were written", ticker)

    return df

def _migrate_legacy_stack_proba(df: pd.DataFrame) -> pd.DataFrame:
    LONG_IDX  = int(CLASS_IS_LONG)
    SHORT_IDX = 1 - LONG_IDX
    for pfx in ("lgbm","cat","rf","lr"):
        old_long  = f"{pfx}_stack_proba_long"
        old_short = f"{pfx}_stack_proba_short"
        if old_long in df.columns or old_short in df.columns:
            dst_long  = f"{pfx}_stack_prob_class_{LONG_IDX}"
            dst_short = f"{pfx}_stack_prob_class_{SHORT_IDX}"
            if dst_long not in df.columns and old_long in df.columns and old_short in df.columns:
                df[dst_long]  = df[old_long]  if LONG_IDX==0 else df[old_short]
            if dst_short not in df.columns and old_long in df.columns and old_short in df.columns:
                df[dst_short] = df[old_short] if SHORT_IDX==1 else df[old_long]
            if f"{pfx}_stack_max_prob" not in df.columns:
                a = df.get(dst_long,  np.nan)
                b = df.get(dst_short, np.nan)
                df[f"{pfx}_stack_max_prob"]  = np.maximum(a, b)
                df[f"{pfx}_stack_prob_diff"] = (a - b).abs()
    return df

# ======= STACKED PREDICTION WRITER (SAFE) — DROP-IN REPLACEMENT =======
def add_all_stacked_model_predictions(
    df: pd.DataFrame,
    ticker: str,
    families: list[str] | tuple[str, ...] | None = None,
    models_dir: Path | str = Path("3.Models_stacked"),
    **kwargs,
) -> pd.DataFrame:
    """
    Add stacked model predictions for the given ticker and families.
    Falls back to default families if none specified.
    """
    # ---- Normalize families
    if families is None:
        families = kwargs.pop("stacking_model_types", None)
    if not families:
        families = ["lightgbm", "catboost", "randomforest", "logisticregression"]
    families = [ _to_prefix(fam) for fam in families ]  # normalize

    out = df
    wrote = []

    for fam in families:
        prev_cols = set(out.columns)
        try:
            out = add_stacked_prediction_for_family(out, ticker, fam, models_dir=models_dir)
            new_cols = list(set(out.columns) - prev_cols)
            if new_cols:
                wrote.append(fam)
        except FileNotFoundError:
            logging.warning("⚠️ No stacked model found for %s (%s)", ticker, fam)
        except Exception as e:
            logging.error("❌ Error adding stacked prediction for %s (%s): %s", ticker, fam, e)

    if wrote:
        logging.info("✅ %s — added stacked predictions: %s", ticker, ", ".join(wrote))
    else:
        logging.info("ℹ️ %s — no stacked predictions were written", ticker)

    return out

def add_stacked_model_prediction_single(df, ticker, model_key):
    """
    Add one stacked model's predictions to df using saved model/scaler (if needed).
    Produces:
      - {model_key}_stack_prediction
      - {model_key}_stack_prob_class_*
      - ProperCase *_max_prob / *_prob_diff (so filters/ranking use the stacker's own confidence)
    """

    df = copy.deepcopy(df)

    # Ensure canonical lgbm_/cat_/rf_/lr_ aliases exist from ProperCase sources
    # after base + stacked predictions are attached to df
    df = ensure_stack_aliases(df)
    df = _maybe_drop_identical_stacks(df)
    df = apply_ensemble_predictions_engine(df)  # adds ensemble_* columns (prob, conf, pred_engine)

    # Load stacked artifact (calibrated pipeline preferred); scaler may be None by design
    model_instance, scaler, expected_features = load_model_and_scaler(
        ticker, model_key, use_stack=True
    )
    if model_instance is None:
        logging.warning(f"⚠️ Missing stacked model for {model_key} ({ticker}), skipping prediction.")
        return df

    # Fallback: derive stack feature cols from disk / DataFrame if not stored
    if not expected_features:
        logging.warning("⚠️ No expected features found in stacked model; loading fallback STACK_FEATURE_COLS")
        expected_features = load_stack_feature_cols(
            ticker,
            df,
            default_cols=[
                "xgb_prob_class_1","xgb_prob_diff","xgb_max_prob",
                "lgbm_prob_class_1","lgbm_prob_diff","lgbm_max_prob",
                "cat_prob_class_1","cat_prob_diff","cat_max_prob",
                "rf_prob_class_1","rf_prob_diff","rf_max_prob",
                "lr_prob_class_1","lr_prob_diff","lr_max_prob",
                "xgb_pred","lgbm_pred","cat_pred","rf_pred","lr_pred",
            ]
        )

    # Verify features exist
    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        logging.error(f"❌ Missing features in DataFrame for {model_key} ({ticker}): {missing[:10]}{'...' if len(missing)>10 else ''}")
        return df

    # Prepare inputs; let pipelines (incl. calibrated pipelines) handle impute/scale internally
    X_raw = df[expected_features].replace([np.inf, -np.inf], np.nan)

    is_pipeline = (
        hasattr(model_instance, "named_steps") or
        hasattr(getattr(model_instance, "base_estimator", None), "named_steps")
    )

    try:
        if is_pipeline:
            X_in = X_raw                          # pipeline handles NaNs and scaling
            X_index = X_raw.index
        else:
            X_clean = X_raw.dropna()
            X_in = scaler.transform(X_clean) if scaler is not None else X_clean
            X_index = X_clean.index

        preds = model_instance.predict(X_in)
        probs = model_instance.predict_proba(X_in) if hasattr(model_instance, "predict_proba") else None

        # Lowercase stack outputs (always written)
        mk_lower = str(model_key).lower()
        pred_col = f"{mk_lower}_stack_prediction"
        df.loc[X_index, pred_col] = preds

        if probs is not None:
            for i in range(probs.shape[1]):
                df.loc[X_index, f"{mk_lower}_stack_prob_class_{i}"] = probs[:, i]

            # Also emit ProperCase confidence columns expected by filters/ranking
            model_name_map_local = {
                "xgboost": "XGBoost",
                "lightgbm": "LightGBM",
                "catboost": "CatBoost",
                "randomforest": "RandomForest",
                "logisticregression": "LogisticRegression",
            }
            proper = model_name_map_local.get(mk_lower, model_key)

            # max_prob: max class probability
            df.loc[X_index, f"{proper}_max_prob"] = probs.max(axis=1)

            # prob_diff: absolute gap between top2 classes (binary → |p1-p0|)
            if probs.shape[1] >= 2:
                if probs.shape[1] == 2:
                    diff = np.abs(probs[:, 1] - probs[:, 0])
                else:
                    # multiclass: top1 - top2
                    sorted_probs = -np.sort(-probs, axis=1)
                    diff = sorted_probs[:, 0] - sorted_probs[:, 1]
                df.loc[X_index, f"{proper}_prob_diff"] = diff.astype(float)

        logging.info(f"✅ Added stacked predictions for {model_key} to DataFrame.")
    except Exception as e:
        logging.error(f"❌ Error predicting with stacked model {model_key} for {ticker}: {e}")

    return df

# Add to V3dBacktest.py: compute and cache daily quantile floor
_DYN_FLOORS = {}  # {date -> qth}

# Fix: use np (not _np)
def _daily_dynamic_floor(date_key: str, conf_val: float) -> float:
    q = float(getattr(CFG, "DYN_CONF_Q", 0.60))
    n = int(getattr(CFG, "DYN_CONF_TOP_M", 5))
    rec = _DYN_FLOORS.setdefault(date_key, {"vals": []})
    if np.isfinite(conf_val):
        rec["vals"].append(float(conf_val))
    vals = rec["vals"]
    if len(vals) >= max(20, n):  # need some density
        return float(np.nanquantile(np.array(vals), q))
    return 0.0

def safe_col(df, col, default_val=np.nan):
    """Return a Series of col or a default value if col missing."""
    return df[col] if col in df.columns else pd.Series(default_val, index=df.index)

def apply_ensemble_predictions_engine(
    df,
    out_prefix: str = "ensemble",
    long_thresh: float | None = 0.5,
    short_thresh: float | None = 0.5,
    prefer_stacked: bool = True,
    weights: dict | None = None,
    temp: float | None = None
):
    """
    Logit-ensemble over family member probabilities with:
      • Per-family weight cap (ENSEMBLE_WEIGHT_CAP_PER_FAMILY) + renorm
      • Per-ticker 'stack vs base' selection via STACK_VS_BASE_MAP (if present)
      • Temperature on logits (default from CFG.ENSEMBLE_TEMP)
      • Equal-weight fallback when weights are missing
      • Optional calibration + prior-shift on the ensemble probability
    Writes:
      ensemble_prob_class_{SHORT/LONG}, ensemble_max_prob, ensemble_prob_diff,
      ensemble_pred_engine (0=LONG,1=SHORT), ensemble_conf (P(chosen side)).
    """

    # ---------- helpers ----------
    def _safe_float(x, d=1.0):
        try:
            v = float(x)
            return v if np.isfinite(v) and v > 0 else float(d)
        except Exception:
            return float(d)

    def _safe_logit(p: pd.Series, eps: float = 1e-6) -> pd.Series:
        s = pd.to_numeric(p, errors="coerce").clip(eps, 1.0 - eps)
        return np.log(s / (1.0 - s))

    def _safe_expit(z: pd.Series) -> pd.Series:
        return 1.0 / (1.0 + np.exp(-z))

    title_to_prefix = {
        "XGBoost":"xgb","LightGBM":"lgbm","CatBoost":"cat","RandomForest":"rf","LogisticRegression":"lr"
    }
    any_to_title = {
        "xgboost":"XGBoost","lightgbm":"LightGBM","catboost":"CatBoost",
        "randomforest":"RandomForest","logisticregression":"LogisticRegression",
        "xgb":"XGBoost","lgbm":"LightGBM","cat":"CatBoost","rf":"RandomForest","lr":"LogisticRegression",
    }

    def _norm_weight_keys(w: dict | None) -> dict[str, float]:
        if not w:
            return {}
        out = {}
        for k, v in w.items():
            key = any_to_title.get(str(k).strip().lower(), k)
            try:
                val = float(v)
                if np.isfinite(val) and val > 0:
                    out[key] = val
            except Exception:
                pass
        return out

    def _renorm_cap(weights_title: dict[str, float]) -> dict[str, float]:
        """Apply family cap then renormalize; fallback to equal if empty/degenerate."""
        cap = float(getattr(CFG, "ENSEMBLE_WEIGHT_CAP_PER_FAMILY", 1.0))
        w = {k: max(0.0, float(v)) for k, v in (weights_title or {}).items()}
        if not w:
            return {}
        if cap < 1.0:
            w = {k: min(v, cap) for k, v in w.items()}
        s = sum(w.values())
        if s <= 0:
            return {}
        return {k: v / s for k, v in w.items()}

    def _load_stack_vs_base_map() -> dict:
        """Return {TICKER: {family_title: 'stacked'|'base'}} if file exists, else {}."""
        path = getattr(CFG, "STACK_VS_BASE_MAP", None)
        if not path:
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
                return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    # --- calibration / prior-shift stubs (safe no-ops if not provided) ---
    def _load_calibrator(family: str):
        """Return a calibrator with predict_proba(X)->Nx2 or None."""
        try:
            # Example:
            # from joblib import load as _joblib_load
            # calib_dir = getattr(CFG, "CALIBRATOR_DIR", "4.Metrics/calibrators")
            # path = os.path.join(calib_dir, f"{family}_calibrator.joblib")
            # return _joblib_load(path) if os.path.exists(path) else None
            return None
        except Exception:
            return None

    def _load_prevalences(family: str):
        """Return (pi_train, pi_test) or (None, None) if unavailable."""
        try:
            # Example:
            # prev_dir = getattr(CFG, "PREVALENCE_DIR", "4.Metrics/prevalences")
            # path = os.path.join(prev_dir, f"{family}_prevalence.json")
            # if os.path.exists(path):
            #     with open(path, "r", encoding="utf-8") as f:
            #         obj = json.load(f)
            #     return float(obj.get("pi_train")), float(obj.get("pi_test"))
            return None, None
        except Exception:
            return None, None

    def _apply_calibration_and_shift(p_short: pd.Series, *, family: str = "ensemble") -> pd.Series:
        """Apply optional calibration and prior-shift to P(short)."""
        try:
            if bool(getattr(CFG, "USE_CALIBRATION", False)):
                cal = _load_calibrator(family)
                if cal is not None and hasattr(cal, "predict_proba"):
                    X = np.asarray(p_short.values, dtype=float).reshape(-1, 1)
                    out = cal.predict_proba(X)
                    if out is not None and np.ndim(out) == 2 and out.shape[1] >= 2:
                        p_short = pd.Series(out[:, 1], index=p_short.index)

            if bool(getattr(CFG, "PRIOR_SHIFT_ENABLE", False)):
                pi_train, pi_test = _load_prevalences(family)
                thr = float(getattr(CFG, "PRIOR_SHIFT_MIN_ABS_DELTA", 0.07))
                min_obs = int(getattr(CFG, "PRIOR_SHIFT_MIN_OBS", 100))
                if (pi_train is not None and pi_test is not None and
                    np.isfinite(pi_train) and np.isfinite(pi_test) and
                    abs(float(pi_test) - float(pi_train)) >= thr and
                    p_short.dropna().shape[0] >= min_obs):
                    eps = 1e-9
                    odds = p_short.clip(eps, 1 - eps) / (1 - p_short.clip(eps, 1 - eps))
                    odds *= (float(pi_test) / max(float(pi_train), eps))
                    p_short = odds / (1 + odds)
        except Exception:
            pass
        return p_short.clip(0.0, 1.0)

    # ---------- prep ----------
    TEMP = _safe_float(temp if temp is not None else getattr(CFG, "ENSEMBLE_TEMP", 1.0), d=1.0)
    LONG_IDX  = int(CLASS_IS_LONG)       # 0 if LONG=class 0, else 1
    SHORT_IDX = 1 - LONG_IDX

    work = df.copy()
    work = ensure_xgb_aliases(work)
    work = ensure_base_aliases(work)     # ProperCase → alias
    work = ensure_stack_aliases(work)    # Stacked ProperCase/short aliases

    # Determine ticker (if provided) to consult the stack/base map
    ticker_val = None
    if "ticker" in work.columns:
        try:
            uniq = pd.Series(work["ticker"]).dropna().astype(str).unique()
            if len(uniq) == 1:
                ticker_val = uniq[0]
        except Exception:
            pass
    svb_map = _load_stack_vs_base_map()

    # Normalize/merge weights (caller > config), then cap & renormalize
    w_cfg = _norm_weight_keys(ENSEMBLE_WEIGHTS)
    w_usr = _norm_weight_keys(weights)
    w_merged = _renorm_cap({**w_cfg, **w_usr}) if (w_cfg or w_usr) else {}

    members: list[pd.Series] = []
    used_titles: list[str] = []
    used_cols: list[str] = []

    # Deterministic family order
    family_titles = ["XGBoost","LightGBM","CatBoost","RandomForest","LogisticRegression"]

    for title in family_titles:
        pfx = title_to_prefix.get(title)
        if not pfx:
            continue

        # stack vs base decision: per-ticker override wins, else prefer_stacked
        want_stack = prefer_stacked
        if ticker_val and isinstance(svb_map.get(str(ticker_val), {}), dict):
            choice = str(svb_map[str(ticker_val)].get(title, "")).lower()
            if choice in ("stacked", "base"):
                want_stack = (choice == "stacked")

        cand_stack = f"{pfx}_stack_prob_class_{SHORT_IDX}"
        cand_base  = f"{pfx}_prob_class_{SHORT_IDX}"
        col = cand_stack if (want_stack and cand_stack in work.columns) else (cand_base if cand_base in work.columns else None)
        if not col:
            continue

        s = pd.to_numeric(work[col], errors="coerce")
        if not s.notna().any():
            continue

        members.append(s)
        used_titles.append(title)
        used_cols.append(col)

    # If nothing collected yet, fallback to any *_prob_class_{SHORT_IDX} on the frame
    if not members:
        cand_cols = [c for c in work.columns
                     if str(c).lower().endswith(f"_prob_class_{SHORT_IDX}")
                     and not str(c).lower().startswith(out_prefix.lower()+"_")]
        if cand_cols:
            logits = [_safe_logit(pd.to_numeric(work[c], errors="coerce")) for c in cand_cols]
            Z = sum(logits) / float(len(logits))
            Z = Z / max(TEMP, 1e-6)
            p_short = _safe_expit(Z)
        else:
            p_short = pd.Series(0.5, index=work.index)

        # apply calibration/prior shift on ensemble prob
        p_short = _apply_calibration_and_shift(p_short, family="ensemble")

    else:
        # If no usable weights → equal weight over members actually present
        if not w_merged:
            eq = 1.0 / float(len(members))
            used_w = [eq] * len(members)
        else:
            w_present = {t: w_merged.get(t, 0.0) for t in used_titles}
            w_present = _renorm_cap(w_present)
            if not w_present or sum(w_present.values()) <= 0:
                eq = 1.0 / float(len(members))
                used_w = [eq] * len(members)
            else:
                used_w = [float(w_present.get(t, 0.0)) for t in used_titles]
                s = sum(used_w)
                used_w = [v / s for v in used_w] if s > 0 else [1.0/len(members)]*len(members)

        # weighted-logit average, then temperature
        denom = sum(used_w) if sum(used_w) > 0 else float(len(used_w))
        Z = sum(_safe_logit(s) * w for s, w in zip(members, used_w)) / denom
        Z = Z / max(TEMP, 1e-6)
        p_short = _safe_expit(Z)

        # apply calibration/prior shift
        p_short = _apply_calibration_and_shift(p_short, family="ensemble")

        logging.debug(
            "Ensemble families: %s | cols=%s | weights=%s | T=%.4f",
            used_titles, used_cols, {t: round(w, 4) for t, w in zip(used_titles, used_w)}, TEMP
        )

    # compose p_long and guards
    p_short = p_short.clip(0.0, 1.0)
    p_long  = (1.0 - p_short).clip(0.0, 1.0)

    # write probs
    work[f"{out_prefix}_prob_class_{SHORT_IDX}"] = p_short
    work[f"{out_prefix}_prob_class_{LONG_IDX}"]  = p_long
    work[f"{out_prefix}_max_prob"]  = np.maximum(p_long, p_short)
    work[f"{out_prefix}_prob_diff"] = (p_long - p_short).abs()

    # thresholds → engine 0/1 label (keep your engine convention)
    tL = float(ENSEMBLE_LONG_THRESHOLD  if long_thresh  is None else long_thresh)
    tS = float(ENSEMBLE_SHORT_THRESHOLD if short_thresh is None else short_thresh)

    longs  = (p_long  >= tL)
    shorts = (p_short >= tS)
    both   = longs & shorts
    pick_long = longs.copy()
    pick_long[both] = (p_long[both] >= p_short[both])
    pred01 = np.where(pick_long, 0, np.where(shorts, 1, np.nan))
    work[f"{out_prefix}_pred_engine"] = pd.Series(pred01, index=work.index).astype("Int64")

    # confidence = P(chosen side)
    chosen_prob = np.where(pred01 == 0, p_long, np.where(pred01 == 1, p_short, np.nan))
    work[f"{out_prefix}_conf"] = pd.Series(chosen_prob, index=work.index)

    return work


# === dBacktest.py — Add/keep this helper =====================================
def compute_ensemble_signal(df: pd.DataFrame,
                            prediction_column: str = "ensemble_pred_engine",
                            long_thr: float = ENSEMBLE_LONG_THRESHOLD,
                            short_thr: float = ENSEMBLE_SHORT_THRESHOLD) -> pd.DataFrame:
    """
    Symmetric ensemble classifier:
      0 = LONG if P(long) >= long_thr
      1 = SHORT if P(short) >= short_thr
      NaN if neither passes; break ties by larger prob
    """
    LONG_IDX, SHORT_IDX = _class_indices()
    pL = pd.to_numeric(df.get(f"ensemble_prob_class_{LONG_IDX}", np.nan), errors="coerce")
    pS = pd.to_numeric(df.get(f"ensemble_prob_class_{SHORT_IDX}", np.nan), errors="coerce")

    longs  = (pL >= long_thr)
    shorts = (pS >= short_thr)
    both   = longs & shorts

    pick_long = longs.copy()
    pick_long[both] = (pL[both] >= pS[both])

    pred = np.where(pick_long, 0, np.where(shorts, 1, np.nan))
    df[prediction_column] = pred.astype("float")
    return df


def ensure_indicators(df, ticker):
    """
    Ensure DF has macd, macd_signal, macd_hist, ema_20, rsi (14).
    After calculation, sanitize NaN/Inf from startup windows to avoid downstream issues.
    """
    if "close" not in df.columns:
        logging.warning(f"⚠️ Missing 'close' column in DataFrame for {ticker}, cannot compute indicators.")
        return df

    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # MACD + signal
    if "macd" not in df.columns or "macd_signal" not in df.columns:
        exp12 = df["close"].ewm(span=12, adjust=False).mean()
        exp26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp12 - exp26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # MACD hist
    if "macd_hist" not in df.columns:
        df["macd_hist"] = df["macd"] - df["macd_signal"]

    # EMA 20
    if "ema_20" not in df.columns:
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

    # RSI 14
    if "rsi" not in df.columns:
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss.replace(0, np.nan))
        df["rsi"] = 100 - (100 / (1 + rs))

    # --- Clean NaNs from warm-up periods (drop is safer for filters)
    ind_cols = ["macd", "macd_signal", "macd_hist", "ema_20", "rsi"]
    for c in ind_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    #before = len(df)
    #df = df.dropna(subset=[c for c in ind_cols if c in df.columns])
    dropped = before - len(df)
    if dropped > 0:
        logging.warning(f"🧹 Dropped {dropped} warm-up rows with NaN indicators for {ticker}")

    return df

def add_rolling_volume(df):
    if "volume" in df.columns:
        df["volume_rolling_20"] = df["volume"].rolling(20).mean().bfill()
    return df

def check_required_columns(df, required_cols, ticker):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logging.warning(f"⚠️ Missing filter columns for {ticker}: {missing}")
        return False
    return True

def _model_prob_cols(df: pd.DataFrame, prediction_column: str) -> tuple[str | None, str | None]:
    """
    Prefer model-specific *_max_prob first, then *_prob_class_1.
    Robust to long names in prediction_column (e.g., 'LightGBM_pred_engine').
    Never calls config helpers with a None/unknown family.
    """
    if df is None or not len(df.columns):
        return None, None

    fam = _infer_family_from_name(prediction_column)
    if fam == "ensemble":
        # 1) prefer max prob for ranking  2) use real separation for tail gating
        prob = "ensemble_max_prob" if "ensemble_max_prob" in df.columns else (
            "ensemble_prob_class_1" if CLASS_IS_LONG == 1 else "ensemble_prob_class_0")
        diff = "ensemble_prob_diff" if "ensemble_prob_diff" in df.columns else None
        return prob, diff

    title_map = {"xgb":"XGBoost","lgbm":"LightGBM","cat":"CatBoost","rf":"RandomForest","lr":"LogisticRegression"}
    title = title_map.get(fam, None)

    prob_candidates: list[str] = []
    diff_candidates: list[str] = []

    if fam is not None:
        # Use config helpers only when we have a proper family
        try:
            prob_candidates += [
                max_prob_col(fam),                 # e.g., lgbm_max_prob
                f"{title}_max_prob" if title else None,
                prob1_col(fam),                    # e.g., lgbm_prob_class_1
                f"{title}_prob_class_1" if title else None,
            ]
            diff_candidates += [
                prob_diff_col(fam),
                f"{title}_prob_diff" if title else None,
            ]
        except Exception:
            # fall through to generic scan below
            pass

    # Generic fallbacks (work even if fam is unknown)
    prob_candidates += [c for c in df.columns if str(c).endswith("_max_prob")]
    prob_candidates += [c for c in df.columns if str(c).endswith("_prob_class_1")]
    diff_candidates += [c for c in df.columns if str(c).endswith("_prob_diff")]

    # pick the first present
    prob = next((c for c in prob_candidates if c and c in df.columns), None)
    diff = next((c for c in diff_candidates if c and c in df.columns), None)
    return prob, diff

def apply_signal_filters(
    df: pd.DataFrame,
    ticker: str,
    prediction_column: str,
    model_type: str | None = None,
    *,
    apply_volume: bool = True,
    apply_rsi: bool = True,
    apply_macd: bool = True,
    apply_ema_buffer: bool = True,
    prob_column: str | None = None,   # optional coalesced confidence column
    min_conf_floor: float | None = None,
    **_ignored,
) -> tuple[pd.DataFrame, dict[str, pd.Series], pd.Series]:
    """
    Lightweight gating for signals produced in ENGINE space (0=LONG, 1=SHORT).
    Returns: (filtered_df, filters_dict, combined_mask)

    Order of gates (important):
      1) pred validity
      2) Volume
      3) RSI
      4) EMA buffer
      5) MACD
      6) Confidence (LAST, and only if min_conf_floor is provided)
      7) Optional relative-prob separation
    """

    def _as_series(_df: pd.DataFrame, col: str, dtype=float, default=np.nan) -> pd.Series:
        if col in _df.columns:
            return pd.to_numeric(_df[col], errors="coerce").astype(dtype, copy=False)
        return pd.Series(default, index=_df.index, dtype=dtype)

    def _get(name: str, default):
        try:
            return globals().get(name, default)
        except Exception:
            return default

    # --- config knobs (safe defaults if missing)
    volume_min            = float(_get("volume_min", 0))
    vol_buf_pct           = float(_get("volume_buffer_pct", 0.0))  # relative vs 20D volume
    rsi_threshold         = _get("rsi_threshold", 50)
    rsi_threshold_long    = _get("rsi_threshold_long", None)
    rsi_threshold_short   = _get("rsi_threshold_short", None)
    ema_buffer_pct_long   = float(_get("ema_buffer_pct_long", 1.0))
    ema_buffer_pct_short  = float(_get("ema_buffer_pct_short", 1.0))
    macd_hist_long        = float(_get("macd_hist_long", 0.0))
    macd_hist_short       = float(_get("macd_hist_short", 0.0))
    diff_q                = _get("xgb_prob_diff_quantile", None)  # e.g. 0.25 keeps top 75% by diff

    # if caller passes None → disable confidence gate here
    abs_prob_floor = (min_conf_floor 
                  if (min_conf_floor is not None) 
                  else _get("prob_threshold", None))

    work = df.copy()

    # --- normalize legacy caps
    rename_map = {}
    for src, dst in (("Close","close"),("Volume","volume"),("RSI","rsi"),
                     ("EMA20","ema_20"),("MACD_Hist","macd_hist")):
        if src in work.columns and dst not in work.columns:
            rename_map[src] = dst
    if rename_map:
        work = work.rename(columns=rename_map)

    # --- compute indicators if missing (defensive)
    need_inds = any(c not in work.columns for c in ("rsi","ema_20","macd_hist"))
    if need_inds:
        try:
            work = ensure_indicators(work, ticker)  # defined elsewhere
        except Exception:
            pass

    # --- core columns
    pr = _as_series(work, prediction_column, dtype="float")
    pred_gate = pr.isin([0.0, 1.0]).fillna(False)

    # side flags (ENGINE space: 0=LONG, 1=SHORT)
    is_long  = (pr == 0.0)
    is_short = (pr == 1.0)

    # 2) Volume gate
    vol = _as_series(work, "volume")
    if apply_volume:
        if "volume_rolling_20" in work.columns:
            v20 = _as_series(work, "volume_rolling_20")
        else:
            v20 = vol.rolling(20, min_periods=1).mean()
        thr = pd.Series(volume_min, index=work.index, dtype=float)
        if vol_buf_pct > 0:
            rel_thr = (v20 * vol_buf_pct).fillna(0.0)
            thr = pd.Series(np.maximum(thr.values, rel_thr.values), index=work.index)
        volume_gate = (vol >= thr).fillna(False)
    else:
        volume_gate = pd.Series(True, index=work.index)

    # 3) RSI (directional)
    rsi = _as_series(work, "rsi")
    rtl = float(rsi_threshold_long  if rsi_threshold_long  is not None else rsi_threshold or 50)
    rts = float(rsi_threshold_short if rsi_threshold_short is not None else rsi_threshold or 50)
    rsi_gate_long  = (rsi >= rtl)
    rsi_gate_short = (rsi <= rts)
    rsi_gate = (is_long & rsi_gate_long) | (is_short & rsi_gate_short) if apply_rsi else pd.Series(True, index=work.index)

    # 4) EMA (directional)
    close = _as_series(work, "close")
    ema20 = _as_series(work, "ema_20")
    ema_long_level  = ema20 * (ema_buffer_pct_long  if ema_buffer_pct_long  else 1.0)
    ema_short_level = ema20 * (ema_buffer_pct_short if ema_buffer_pct_short else 1.0)
    ema_gate_long   = (close >= ema_long_level)
    ema_gate_short  = (close <= ema_short_level)
    ema_gate = (is_long & ema_gate_long) | (is_short & ema_gate_short) if apply_ema_buffer else pd.Series(True, index=work.index)

    # 5) MACD hist (directional)
    macd_h = _as_series(work, "macd_hist")
    macd_long_ok  = (macd_h >= macd_hist_long)
    macd_short_ok = (macd_h <= -abs(macd_hist_short))
    macd_gate = (is_long & macd_long_ok) | (is_short & macd_short_ok) if apply_macd else pd.Series(True, index=work.index)

    # 6) Confidence gate (LAST; optional)
    lp_col, sp_col = _side_prob_cols_for_prediction(work, prediction_column)
    if not lp_col or not sp_col:
        lp_col, sp_col = _coalesce_prob_columns(work, prefer_stacked=False)

    if abs_prob_floor is not None and lp_col and sp_col:
        lp = pd.to_numeric(work[lp_col], errors="coerce")
        sp = pd.to_numeric(work[sp_col], errors="coerce")
        side_prob_gate = (is_long & (lp >= float(abs_prob_floor))) | (is_short & (sp >= float(abs_prob_floor)))
    else:
        side_prob_gate = pd.Series(True, index=work.index)

    # 7) Relative separation on *_prob_diff (keep top (1-q))
    probdiff_gate = pd.Series(True, index=work.index)
    if diff_q and 0.0 < float(diff_q) < 1.0:
        try:
            _max_col, diff_col = _model_prob_cols(work, prediction_column)
        except Exception:
            _max_col, diff_col = (None, None)
        if diff_col and diff_col in work.columns:
            diff_series = _as_series(work, diff_col)
            if diff_series.notna().any():
                thresh = diff_series.quantile(1.0 - float(diff_q))
                probdiff_gate = (diff_series >= thresh)

    # combine all gates — confidence last
    prob_gate = side_prob_gate & probdiff_gate
    combined = pred_gate & volume_gate & rsi_gate & ema_gate & macd_gate & prob_gate

    filters = {
        "pred_gate":      pred_gate,
        "volume_gate":    volume_gate,
        "rsi_gate":       rsi_gate,
        "ema_gate":       ema_gate,
        "macd_gate":      macd_gate,
        "prob_gate":      prob_gate,
        "side_prob_gate": side_prob_gate,
        "probdiff_gate":  probdiff_gate,
    }

    # logging (optional)
    def _log_side_counts(series: pd.Series, tag: str):
        s = pd.to_numeric(series, errors="coerce")
        nL = int((s == 0).sum()); nS = int((s == 1).sum()); nN = int(s.isna().sum())
        logging.info(f"[{tag}] counts → LONG={nL:,}  SHORT={nS:,}  NaN={nN:,}")

    _log_side_counts(pr, "pre_filters")
    _log_side_counts(pr[combined], "post_filters")

    return work[combined].copy(), filters, combined


def save_entry_rejections_debug(ticker: str, rejections: list[dict]):
    """
    Persist per-bar entry rejections for later inspection.
    Skips writing when CFG.WRITE_DEBUG_CSVS is False (default).
    """
    if not getattr(CFG, "WRITE_DEBUG_CSVS", True):
        return
    if not rejections:
        return
    try:
        os.makedirs(metrics_folder, exist_ok=True)
    except Exception:
        pass
    out = pd.DataFrame(rejections)
    out.sort_values("date", inplace=True)
    out_path = os.path.join(metrics_folder, f"debug_entries_rejected_{ticker}.csv")
    out.to_csv(out_path, index=False)
    logging.warning(f"💡 Saved {len(out)} entry rejections to {out_path}")

def save_dropped_signals(df_before, df_kept, filters, ticker, prediction_column="prediction"):
    """
    Save rows removed by gates with human-readable reasons.
    Skips writing when CFG.WRITE_DEBUG_CSVS is False (default).
    """
    if not getattr(CFG, "WRITE_DEBUG_CSVS", True):
        return
    if df_before is None or df_before.empty:
        return

    dropped_idx = df_before.index.difference(df_kept.index)
    if len(dropped_idx) == 0:
        return

    dropped = df_before.loc[dropped_idx].copy()

    max_prob_col, prob_diff_col = _model_prob_cols(df_before, prediction_column)

    reasons_out = []
    for i in dropped.index:
        rs = []
        for name, gate in filters.items():
            try:
                if not bool(gate.get(i, True)):
                    rs.append(name.replace("_", " "))
            except Exception:
                pass
        # Helpful context on probs if present
        try:
            if max_prob_col and max_prob_col in dropped.columns and "prob_gate" in filters and not bool(filters["prob_gate"].get(i, True)):
                v = float(dropped.at[i, max_prob_col])
                rs.append(f"{max_prob_col}={v:.3f} below threshold")
            if prob_diff_col and prob_diff_col in dropped.columns and "probdiff_gate" in filters and not bool(filters["probdiff_gate"].get(i, True)):
                v = float(dropped.at[i, prob_diff_col])
                rs.append(f"{prob_diff_col}={v:.3f} below tail gate")
        except Exception:
            pass

        reasons_out.append("; ".join(rs) if rs else "unknown")

    dropped["filter_reasons"] = reasons_out
    if max_prob_col and max_prob_col in dropped.columns:
        dropped["signal_rank"] = pd.to_numeric(dropped[max_prob_col], errors="coerce").rank(ascending=False, method="first")

    for col in ("date", "close", "ema_20", "rsi", "macd_hist", "volume"):
        if col not in dropped.columns and col.capitalize() in dropped.columns:
            dropped[col] = dropped[col.capitalize()]

    dropped["ticker"] = ticker
    out_path = os.path.join(metrics_folder, f"debug_signals_filtered_out_{ticker}.csv")
    try:
        dropped.reset_index().rename(columns={"index": "date"}).to_csv(out_path, index=False)
        logging.warning(f"💡 Saved {len(dropped)} filtered-out signals with reasons → {out_path}")
    except Exception as e:
        logging.warning(f"⚠️ Could not save filtered-out signals for {ticker}: {e}")

def filter_stale_signals(df, max_days: int = 30):
    if df is None or df.empty:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        df["days_since_signal"] = (df.index[-1] - df.index).days
        df = df[df["days_since_signal"] <= max_days]
    return df

def rank_and_trim_signals(df, prediction_column, limit=None):

    limit = int(limit or top_signals_limit)
    work = df.copy()

    src = str(prediction_column).lower()
    is_ensemble = "ensemble" in src

    if is_ensemble and "ensemble_conf" in work.columns and not work["ensemble_conf"].isna().all():
        work["rank_score"] = pd.to_numeric(work["ensemble_conf"], errors="coerce")
    else:
        max_prob_col, _ = _model_prob_cols(work, prediction_column)  # ✅ use resolver
        if max_prob_col and max_prob_col in work.columns:
            work["rank_score"] = pd.to_numeric(work[max_prob_col], errors="coerce")
        else:
            # graceful fallback (unchanged)
            work["rank_score"] = 0.5

    return work.sort_values(by="rank_score", ascending=False).head(limit)


def normalize_metric_keys(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Map loose metric names into the normalized keys used by the summary."""
    def g(*keys: str, default: Any = 0.0) -> Any:
        for k in keys:
            if k in metrics:
                return metrics[k]
        return default

    out = {
        "Model Return (%)": g("Model Return (%)", "Return (%)", "Total Return (%)", "Return"),
        "Sharpe": g("Sharpe", "Sharpe Ratio"),
        "Win Rate (%)": g("Win Rate (%)", "Winrate (%)", "Win Rate"),
        "Expectancy (%)": g("Expectancy (%)", "Expectancy"),
        "Max Drawdown (%)": g("Max Drawdown (%)", "MDD (%)", "MaxDD (%)", "Max Drawdown"),
        "Avg Duration (days)": g("Avg Duration (days)", "Average Days Held", "Avg Days Held"),
        "Average Prediction/Probability (%)": g("Average Prediction/Probability (%)", "Avg Prob (%)", "Avg Prediction (%)"),
        "QuantML Score": g("QuantML Score", "QuantML"),
        # new counts
        "Total Longs": int(g("Total Longs", "Num Longs", "Longs", default=0)),
        "Total Shorts": int(g("Total Shorts", "Num Shorts", "Shorts", default=0)),
    }
    return out

def make_excel_hyperlink(path: str, label: str = "View Trades") -> str:
    """
    Return an Excel HYPERLINK() formula that always works (absolute file:// URI).
    """
    if not path:
        return ""
    p = Path(path).resolve()
    uri = "file:///" + str(p).replace("\\", "/")
    return f'=HYPERLINK("{uri}", "{label}")'

def save_trade_log(trades_df: pd.DataFrame, ticker: str, model_type: str, suffix: str | None) -> str:
    """
    Save per-ticker trade log and return an *absolute* path so we can
    embed a file:// URI in Excel (reliable hyperlinks).

    Respects CFG.WRITE_TRADES — if False, returns "" and skips writing.
    """
    # Respect config toggle
    if not bool(getattr(CFG, "WRITE_TRADES", True)):
        return ""

    os.makedirs(backtest_path, exist_ok=True)
    suffix = suffix or "None"
    fname = f"{ticker}_{model_type}_{suffix}_trades.csv"
    full = os.path.join(backtest_path, fname)

    # Ensure ticker column
    if "ticker" not in trades_df.columns:
        trades_df = trades_df.copy()
        trades_df["ticker"] = ticker
    else:
        trades_df = trades_df.copy()
        trades_df["ticker"] = trades_df["ticker"].fillna(ticker)

    trades_df.to_csv(full, index=False)
    return os.path.abspath(full)

def _ensure_lowercase_tech_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Map legacy capitalized technical columns to lowercase if the lowercase ones are missing."""
    mapping = {"Close": "close", "Volume": "volume", "RSI": "rsi", "EMA20": "ema_20"}
    rename = {src: dst for src, dst in mapping.items() if src in df.columns and dst not in df.columns}
    return df.rename(columns=rename) if rename else df

def add_ranking_columns(df: pd.DataFrame, prediction_column: str | None, ticker: str = "") -> pd.DataFrame:
    """
    Create/refresh a 'rank_score' column used to rank candidate signals.
    Priority:
      1) Ensemble: 'ensemble_conf' if present
      2) Model-specific: '<model>_max_prob' resolved via _model_prob_cols
      3) Fallback: 0.5
    """

    work = df.copy()

    src = (prediction_column or "").lower()
    is_ensemble = "ensemble" in src

    # Ensemble confidence first
    if is_ensemble and "ensemble_conf" in work.columns and not work["ensemble_conf"].isna().all():
        work["rank_score"] = pd.to_numeric(work["ensemble_conf"], errors="coerce").clip(0, 1)
        return work

    # Model-specific prob column (resolver understands xgboost/lgbm/cat/rf/lr and stack variants)
    max_prob_col, _ = _model_prob_cols(work, prediction_column or "")
    if max_prob_col and max_prob_col in work.columns:
        work["rank_score"] = pd.to_numeric(work[max_prob_col], errors="coerce")
        return work

    # Graceful fallback
    work["rank_score"] = 0.5
    return work

def backtest_engine(
    trades_df: pd.DataFrame,
    ticker: str,
    *,
    price_df: pd.DataFrame,                 # full daily timeline (close/atr)
    model_type: str = "Model",
    suffix: str = "",
    initial_capital: float = 100_000.0,
    transaction_cost: float = 0.001,
    sl_mult: float = 1.5,
    tp_mult: float = 2.5,
    prediction_column: str,                 # should be 'ml_signal' from build_trade_signals()
    prob_column: str | None = None,
    risk_pct: float | None = None,          # <- will default from CFG.RISK_PER_TRADE
    use_trailing_stop: bool = True,
    trail_pct: float | None = None,         # <- will default from CFG.TRAIL_ATR_MULT
    use_ranking: bool = True,
    min_conf_floor: float | None = None,
) -> tuple[float, pd.DataFrame, dict, str]:
    """
    Execute trades over the full daily timeline:
      • iterate daily over price_df
      • open entries only on dates present in trades_df
      • update/trail/exit ALL open positions daily
      • build a daily mark-to-market equity curve
    """
    from config import max_open_trades as _MAX_OPEN

    # ---- defaults from config (safe fallbacks) ----
    if risk_pct is None:
        risk_pct = float(getattr(CFG, "RISK_PER_TRADE", 0.02))
    if trail_pct is None:
        trail_pct = float(getattr(CFG, "TRAIL_ATR_MULT", 1.0))

    # ---- hygiene: canonical indices/columns ----
    px = price_df.copy()
    if not isinstance(px.index, pd.DatetimeIndex):
        if "date" in px.columns:
            px["date"] = pd.to_datetime(px["date"], errors="coerce")
            px = px.dropna(subset=["date"]).set_index("date")
        else:
            raise ValueError("price_df must have a DatetimeIndex or a 'date' column")

    for c in ("close", "atr"):
        if c not in px.columns:
            raise KeyError(f"price_df missing required column '{c}'")

    # signals -> dict keyed by date
    sig = trades_df.copy() if trades_df is not None else pd.DataFrame(columns=["ml_signal"])
    if not isinstance(sig.index, pd.DatetimeIndex):
        if "date" in sig.columns:
            sig["date"] = pd.to_datetime(sig["date"], errors="coerce")
            sig = sig.dropna(subset=["date"]).set_index("date")
        else:
            # allow empty
            sig.index = pd.DatetimeIndex([])
    sig = sig.sort_index()
    # keep only what we need for entries
    keep_cols = [prediction_column, "signal_conf"]
    sig = sig[[c for c in keep_cols if c in sig.columns]]

    # state
    cash = float(initial_capital)
    open_positions: list[dict] = []
    closed_trades: list[dict] = []

    equity_dates = []
    equity_vals  = []

    # helper for unrealized PnL given today’s price
    def _unrealized_sum(price_today: float) -> float:
        tot = 0.0
        for p in open_positions:
            if p.get("type","").lower() == "long":
                tot += p.get("qty", 0.0) * (price_today - p.get("entry_price", 0.0))
            else:
                tot += p.get("qty", 0.0) * (p.get("entry_price", 0.0) - price_today)
        return float(tot)

    # ---- main daily loop ----
    for dt, row_px in px.iterrows():
        price_today = float(row_px["close"])

        # 1) open new position(s) if this day has a signal
        if dt in sig.index and len(open_positions) < int(_MAX_OPEN):
            row_sig = sig.loc[[dt]] if isinstance(sig.loc[dt], pd.Series) else sig.loc[[dt]]
            # support one signal per day (typical per ticker), but handle multiple gracefully
            for _, sigrow in row_sig.iterrows():
                entry_row = row_px.copy()
                entry_row[prediction_column] = sigrow.get(prediction_column, np.nan)
                if "signal_conf" in sigrow.index:
                    entry_row["signal_conf"] = sigrow["signal_conf"]
                pos, reasons, _ = _try_open_position(
                    row=entry_row,
                    current_capital=cash,
                    sl_mult=sl_mult, tp_mult=tp_mult,
                    tx_cost=transaction_cost,
                    risk_pct=risk_pct,
                    prediction_column=prediction_column,
                    prob_column=prob_column,
                    use_ranking=use_ranking,
                    min_conf_floor=min_conf_floor
                )
                if pos:
                    pos["ticker"] = pos.get("ticker", ticker)
                    open_positions.append(pos)

                if len(open_positions) >= int(_MAX_OPEN):
                    break

        # 2) daily update/trailing and exits (TP/SL)
        trail_mult = float(getattr(CFG, "TRAIL_ATR_MULT", 1.0))
        newly_closed, still_open = _update_and_exit_positions(
            positions=open_positions,
            row=row_px,
            current_date=pd.Timestamp(dt),
            use_trailing_stop=use_trailing_stop,
            trail_pct=float(getattr(CFG, "TRAIL_ATR_MULT", 1.0)),
            sl_mult=sl_mult,
            tp_mult=tp_mult,
            tx_cost=transaction_cost,
            prediction_column=prediction_column,
        )
        # realize exits → add to cash
        for c in newly_closed:
            closed_trades.append(c["trade"])
            cash += float(c["pnl"])
        open_positions = still_open

        # 3) mark-to-market equity
        equity = cash + _unrealized_sum(price_today)
        equity_dates.append(pd.Timestamp(dt))
        equity_vals.append(float(equity))

    # 4) force-close leftovers on the last bar (for realized PnL / CSV)
    if open_positions:
        for pos in open_positions:
            trade, trade_return, pnl = _close_position_final(pos, px, transaction_cost)
            if trade is not None:
                closed_trades.append(trade)
                cash += float(pnl)

    trades_out = pd.DataFrame(closed_trades)
    if not trades_out.empty:
        trades_out["trade_id"] = range(1, len(trades_out) + 1)

    # 5) metrics from DAILY equity (not just exit days)
    equity_curve = pd.Series(equity_vals, index=pd.DatetimeIndex(equity_dates, name="date"))
    metrics = _compute_performance_metrics(
        closed_trades,
        trades_out,
        initial_capital,
        cash,
        price_df=px,
        equity_curve_override=equity_curve  # << use daily MTM
    )
    trades_path = save_trade_log(trades_out, ticker, model_type, suffix)
    return float(cash), trades_out, metrics, trades_path

def run_backtest(
    df: pd.DataFrame,
    ticker: str,
    *,
    model: str,
    model_type: str,
    prediction_column: str,
    suffix: str = "base",
    prob_column: str | None = None,
    initial_capital: float = initial_capital,
    transaction_cost: float = transaction_cost,
    sl_mult: float = atr_sl_multiplier,
    tp_mult: float = atr_tp_multiplier,
    risk_pct: float | None = None,                  # default from CFG.RISK_PER_TRADE
    use_trailing_stop: bool = False,
    use_ranking: bool = True,
    min_conf_floor: float | None = None,
    price_df_override: pd.DataFrame | None = None,  # <-- NEW: pass the unfiltered daily timeline
) -> tuple[float, pd.DataFrame, dict, str]:
    """
    Wrapper: filter -> build trade signals -> call execution engine on FULL timeline.
    """
    log = logging.getLogger(__name__)

    # 0) Resolve risk_pct from config if not provided
    if risk_pct is None:
        risk_pct = getattr(CFG, "RISK_PER_TRADE", 0.01)

    # Keep a copy of the full price timeline BEFORE filtering
    price_df_full = price_df_override.copy() if price_df_override is not None else df.copy()

    # 1) Normalize the prediction column into 0/1 engine space (0=LONG, 1=SHORT)
    engine_pred_col = map_predictions_to_binary(df, prediction_column)

    # 2) Resolve confidence columns (model-specific prob or ensemble_conf)
    prob_col, probdiff_col = _model_prob_cols(df, prediction_column)
    if prob_column is None:
        prob_column = prob_col

    # 2b) (Optional) compute 'confidence' for late-floor enforcement
    #     Use max(p, 1-p) if we only have a probability.
    conf_series = None
    if prob_column is not None and prob_column in df.columns:
        p = pd.to_numeric(df[prob_column], errors="coerce")
        conf_series = np.maximum(p, 1.0 - p)
    elif "ensemble_conf" in df.columns:
        conf_series = pd.to_numeric(df["ensemble_conf"], errors="coerce")

    # 3) Rank score once (used by salvage/top-K)
    try:
        df = add_ranking_columns(df, prediction_column, ticker=ticker)
    except Exception as e:
        log.warning(f"{ticker} — ranking columns skipped: {e}")

    # 4) Apply signal filters (RSI/EMA/MACD/volume + prob/prob-diff if present)
    filtered_df, filters_info, combined_mask = apply_signal_filters(
        df=df,
        ticker=ticker,
        prediction_column=engine_pred_col,
        prob_column=prob_column,
        apply_volume=bool(getattr(CFG, "APPLY_VOLUME", True)),
        apply_rsi=bool(getattr(CFG, "APPLY_RSI", True)),
        apply_macd=bool(getattr(CFG, "APPLY_MACD", True)),
        apply_ema_buffer=bool(getattr(CFG, "APPLY_EMA_BUFFER", True)),
        min_conf_floor=min_conf_floor,
    )

    # 4b) Last-chance confidence floor (defensive; in case callers bypass filters)
    if min_conf_floor is not None and conf_series is not None:
        before = len(filtered_df)
        conf_aligned = conf_series.reindex(filtered_df.index)
        keep = (conf_aligned >= float(min_conf_floor))
        filtered_df = filtered_df.loc[keep]
        dropped = before - len(filtered_df)
        if dropped > 0:
            log.debug(f"{ticker} — late confidence floor {min_conf_floor:.3f} dropped {dropped} rows (kept {len(filtered_df)}).")

    # 5) Early out if nothing left after filtering
    if filtered_df.empty:
        empty_trades = pd.DataFrame(columns=["date","side","entry","exit","pnl","reason"])
        return 0.0, empty_trades, {"n_trades": 0, "ticker": ticker, "model": model}, suffix

    # 6) Build entry/exit signals ONLY from filtered_df (correct signature)
    entries = build_trade_signals(
        filtered_df,
        signal_col=engine_pred_col,                                   # 0=LONG,1=SHORT
        conf_col=("ensemble_conf" if "ensemble_conf" in filtered_df.columns else prob_column)
    )

    if entries.empty:
        empty_trades = pd.DataFrame(columns=["date","side","entry","exit","pnl","reason"])
        return 0.0, empty_trades, {"n_trades": 0, "ticker": ticker, "model": model}, suffix

    # 7) EXECUTE on the FULL unfiltered price timeline
    final_capital, trades_df, stats, _ = backtest_engine(
        trades_df=entries,
        ticker=ticker,
        price_df=price_df_full,                 # full unfiltered OHLCV timeline
        model_type=model_type,
        suffix=suffix,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        sl_mult=sl_mult,
        tp_mult=tp_mult,
        prediction_column="ml_signal",          # from build_trade_signals()
        prob_column="signal_conf",              # from build_trade_signals()
        risk_pct=risk_pct,
        use_trailing_stop=use_trailing_stop,
        trail_pct=(getattr(CFG, "TRAIL_ATR_MULT", 0.0) if use_trailing_stop else 0.0),
        use_ranking=use_ranking,
        min_conf_floor=min_conf_floor,
    )
    equity = final_capital

    # 8) Return
    return equity, trades_df, stats, suffix


def compute_quantml_score(metrics: dict, verbose: bool = False):
    """
    QuantML Score = 2*Sharpe + Expectancy + (Return% - Drawdown%)/100
    Accepts either 'Max Drawdown (%)' or 'Max Drawdown' in input.
    """
    if not isinstance(metrics, dict):
        return None

    mm = normalize_metric_keys(metrics)
    try:
        sharpe      = float(mm.get("Sharpe", 0.0))
        expectancy  = float(mm.get("Expectancy (%)", 0.0))
        model_ret   = float(mm.get("Model Return (%)", 0.0))
        # Tolerate either label
        max_dd      = float(mm.get("Max Drawdown (%)", mm.get("Max Drawdown", 0.0)))

        score = sharpe * 2 + expectancy + (model_ret - max_dd) / 100.0
        if verbose:
            logging.info(
                f"QuantML Score calculation: 2*{sharpe} + {expectancy} + ({model_ret}-{max_dd})/100 = {score}"
            )
        return round(score, 2)
    except (ValueError, TypeError) as e:
        if verbose:
            logging.error(f"QuantML score computation error: {e}")
        return None

def record_quality_gate_skip(ticker: str,
                             model: str,
                             reason: str,
                             tm_stamp: str | None = None,
                             link_target: str | None = None) -> None:
    """
    Append a row to 6.Predictions/metrics/quality_gate_skips.csv.
    Creates the folder and header if missing.
    """
    out_path = Path(metrics_folder) / "quality_gate_skips.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": tm_stamp or datetime.now().isoformat(timespec="seconds"),
        "ticker": ticker,
        "model": model,
        "reason": reason,
        "link": link_target or "",
    }

    write_header = not out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def create_result(
    *,
    ticker: str,
    model_name: str,
    prediction_type: str,
    prediction_source: str,
    trades_df: pd.DataFrame,
    metrics: dict,
    trades_filename: str | None,
    initial_capital: float,
    final_capital: float,
    passed: bool = True,  # <- default so legacy callers don't break
) -> dict | None:

    """
    Build one summary-row dict for backtest_summary.xlsx and apply the quality gate.
    """
    mm = normalize_metric_keys(metrics)

    ret_pct  = float(mm.get("Model Return (%)", 0) or 0)
    sharpe   = float(mm.get("Sharpe", 0) or 0)
    win_rate = float(mm.get("Win Rate (%)", 0) or 0)
    expect   = float(mm.get("Expectancy (%)", 0) or 0)
    mdd      = float(mm.get("Max Drawdown (%)", 0) or 0)

    trades_n = int(len(trades_df) if trades_df is not None else 0)

    # NEW: totals by side (use normalized metrics first; fall back to the df)
    total_longs  = int(mm.get("Total Longs", 0) or 0)
    total_shorts = int(mm.get("Total Shorts", 0) or 0)
    if (total_longs == 0 and total_shorts == 0) and trades_df is not None and not trades_df.empty:
        pt = trades_df.get("position_type")
        if pt is not None:
            total_longs  = int(pt.astype(str).str.lower().eq("long").sum())
            total_shorts = int(pt.astype(str).str.lower().eq("short").sum())

    avg_days = float(mm.get("Avg Duration (days)", 0) or 0)

    # Average entry probability (%)
    avg_prob = float(mm.get("Average Prediction/Probability (%)", 0) or 0)
    if (avg_prob == 0) and trades_df is not None and "entry_conf" in trades_df.columns:
        s = pd.to_numeric(trades_df["entry_conf"], errors="coerce")
        if s.notna().any():
            avg_prob = float(s.mul(100.0).mean(skipna=True))

    row = {
        "Ticker": ticker,
        "Model": model_name,
        "Prediction Type": prediction_type,
        "Prediction Source": prediction_source,
        "Start Date": str(test_start),
        "End Date": str(test_end),
        "Initial Capital": round(float(initial_capital), 2),
        "Final Capital": round(float(final_capital), 2),
        "Model Return (%)": round(ret_pct, 2),
        "Sharpe": round(sharpe, 2),
        "Win Rate (%)": round(win_rate, 2),
        "Expectancy (%)": round(expect, 2),
        "Max Drawdown (%)": round(mdd, 2),
        "QuantML Score": compute_quantml_score(mm) or 0.0,   # keep only QuantML Score
        "Trades": trades_n,
        "Total Longs": total_longs,                          # NEW
        "Total Shorts": total_shorts,                        # NEW
        "Average Days Held": round(avg_days, 2),
        "Average Prediction/Probability (%)": round(avg_prob, 2),
        "Trades File": make_excel_hyperlink(trades_filename) if trades_filename else "",
        # NOTE: "Passed Gate(s)" column is intentionally removed from the output
    }
    return row if passed else None
    
def save_backtest_summary(summary_rows):
    """
    Persist an Excel backtest summary (sorted by Final Capital) under config.backtest_path.
    Cleans duplicate/empty columns, filters rows with low returns, and sets a stable column order.
    Also removes 'Passed Gate' and adds 'Total Longs' / 'Total Shorts'.
    """
    os.makedirs(backtest_path, exist_ok=True)

    df = pd.DataFrame(summary_rows or [])
    out_path = os.path.join(backtest_path, "backtest_summary.xlsx")

    if df.empty:
        pd.DataFrame(columns=[
            "Ticker",
            "Model",
            "Prediction Type",
            "Prediction Source",
            "Start Date",
            "End Date",
            "Initial Capital",
            "Final Capital",
            "Model Return (%)",
            "Sharpe",
            "Win Rate (%)",
            "Expectancy (%)",
            "Max Drawdown (%)",
            "QuantML Score",
            "Trades",
            "Total Longs",                                 # NEW
            "Total Shorts",                                # NEW
            "Average Days Held",
            "Average Prediction/Probability (%)",
            "Trades File",
            "Comments",
        ]).to_excel(out_path, index=False)
        print(f"📤 Backtest summary saved → {out_path}")
        return out_path

    # Drop duplicate/legacy/empty columns
    drop_cols = []

    if "QuantML" in df.columns:
        drop_cols.append("QuantML")
    if "Max Drawdown" in df.columns:
        drop_cols.append("Max Drawdown")
    if "Passed Gate" in df.columns:
        drop_cols.append("Passed Gate")   # <-- remove the column entirely

    # Also remove any entirely empty columns
    for c in df.columns:
        if df[c].isna().all() or (isinstance(df[c].dtype, type(object)) and (df[c] == "").all()):
            drop_cols.append(c)

    if drop_cols:
        df = df.drop(columns=list(set(drop_cols)), errors="ignore")

    # Ensure totals exist even if missing from some rows
    for c in ("Total Longs","Total Shorts"):
        if c not in df.columns:
            df[c] = 0

    # Filter rows with Model Return < 20%
    if "Model Return (%)" in df.columns:
        df["Model Return (%)"] = pd.to_numeric(df["Model Return (%)"], errors="coerce")
        df = df[df["Model Return (%)"] >= 20.0]

    # Fill Start/End if missing; coerce Final Capital for sorting
    for col in ["Start","End"]:
        if col not in df.columns:
            df[col] = pd.NA
    df["Start"] = df["Start"].fillna(pd.Timestamp(test_start).date() if pd.notna(pd.Timestamp(test_start)) else "")
    df["End"]   = df["End"].fillna(pd.Timestamp(test_end).date()   if pd.notna(pd.Timestamp(test_end))   else "")

    df["Final Capital"] = pd.to_numeric(df.get("Final Capital", pd.Series(dtype=float)), errors="coerce")

    # Stable, explicit order (NO 'Passed Gate'; WITH totals)
    desired_cols = [
        "Ticker",
        "Model",
        "Prediction Type",
        "Prediction Source",
        "Start Date",
        "End Date",
        "Initial Capital",
        "Final Capital",
        "Model Return (%)",
        "Sharpe",
        "Win Rate (%)",
        "Expectancy (%)",
        "Max Drawdown (%)",
        "QuantML Score",
        "Trades",
        "Total Longs",                                 # NEW
        "Total Shorts",                                # NEW
        "Average Days Held",
        "Average Prediction/Probability (%)",
        "Trades File",
        "Comments",
    ]
    ordered = [c for c in desired_cols if c in df.columns] + [c for c in df.columns if c not in desired_cols]
    df = df[ordered]

    # Sort & write
    df = df.sort_values(by="Final Capital", ascending=False)
    df.to_excel(out_path, index=False)

    # (optional) auto-fit widths
    try:
        if len(df) <= 2000:
            wb = load_workbook(out_path)
            ws = wb.active
            for col in ws.columns:
                max_len = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 80)
            wb.save(out_path)
    except Exception:
        pass

    print(f"📤 Backtest summary saved → {out_path}")
    return out_path

def run_backtest_base_models(ticker, df, model_types):
    results = []

    proper_fallback = {
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "randomforest": "RandomForest",
        "logisticregression": "LogisticRegression",
    }

    available_pred_cols = [c for c in df.columns if c.endswith("_pred")]
    # Demote chatty availability line (and log only once per ticker)
    _log_once(logging.INFO, f"available_pred:{ticker}",
              f"{ticker} — Available *_pred columns: {available_pred_cols}")

    # Early exit: if there are no *_pred columns at all, don't loop and warn per-family
    if not available_pred_cols:
        _log_once(logging.INFO, f"no_base_preds:{ticker}",
                  f"ℹ️ {ticker} — no base prediction columns present; skipping base backtests.")
        return results

    for model_type in model_types:
        lower = model_type.lower()
        name_map = (globals().get("model_name_map", {}) or {
            "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
            "randomforest": "RandomForest", "logisticregression": "LogisticRegression",
        })
        proper = name_map.get(lower, model_type)

        # Prefer ProperCase base preds, fall back to lowercase if present
        candidates = [f"{proper}_pred", f"{lower}_pred"]
        prediction_column = next((c for c in candidates if c in df.columns), None)
        if not prediction_column:
            # This is not an error — it simply wasn't produced for this ticker
            _log_once(logging.INFO, f"no_base_pred:{ticker}:{model_type}",
                      f"ℹ️ No base prediction found for {model_type} ({ticker}) "
                      f"(tried {', '.join(candidates)})")
            continue

        df = ensure_xgb_aliases(df)  # ensure filters see xgb_* aliases

        try:
            final_capital, trades_df, metrics, trades_filename = run_backtest(
                df.copy(),
                ticker,
                model=proper,
                model_type=model_type,
                prediction_column=prediction_column,
                suffix="base",
                sl_mult=atr_sl_multiplier,
                tp_mult=atr_tp_multiplier,
                use_trailing_stop=bool(getattr(CFG, "BACKTEST_USE_TRAILING_STOP", True))
            )
            result = create_result(
                ticker=ticker,
                model_name=proper,
                prediction_type="Base",
                prediction_source="preds",
                trades_df=trades_df,
                metrics=metrics,
                trades_filename=trades_filename,
                initial_capital=initial_capital,
                final_capital=final_capital,
                passed=True,
            )
            if result:
                result["Model Type"] = f"Base - {proper}"
                results.append(result)

        except Exception as e:
            logging.error(f"❌ Error backtesting base model {model_type} for {ticker}: {e}")

    return results

def _warn_if_identical(df, col_a, col_b, label_a, label_b):
    if col_a in df.columns and col_b in df.columns and df[col_a].equals(df[col_b]):
        logging.info("ℹ️ %s and %s predictions are identical for this set; leaving both.", label_a, label_b)

def run_backtest_stacked_models(ticker, df, model_types):
    results = []

    for model_type in model_types:
        if str(model_type).lower() == "xgboost":
            logging.info(f"⏭️ Skipping stacked XGBoost for {ticker}")
            continue

        lower = str(model_type).lower()
        name_map = (globals().get("model_name_map", {}) or {
            "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
            "randomforest": "RandomForest", "logisticregression": "LogisticRegression",
        })
        proper = name_map.get(lower, model_type)

        # Resolve the stacked prediction column
        try:
            prediction_column, tried = resolve_prediction_column(df, model_type, stacked=True)
        except KeyError:
            logging.warning(f"⚠️ No stacked prediction found for {model_type} ({ticker}) "
                            f"(tried {', '.join(_candidate_pred_cols(_to_prefix(model_type), True))})")
            continue

        logging.info(f"{ticker} — STACKED {model_type}: using '{prediction_column}'")

        # Normalize aliases & proactively drop identical stacks
        df = ensure_stack_aliases(df)
        df = ensure_xgb_aliases(df)
        df = _maybe_drop_identical_stacks(df)
        df = ensure_ensemble_aliases(df)

        # If the stack was dropped above, try to resolve again; if absent, skip
        if prediction_column not in df.columns:
            try:
                prediction_column, tried = resolve_prediction_column(df, model_type, stacked=True)
            except KeyError:
                logging.info(f"ℹ️ {ticker} — no stacked rows produced (missing preds or all filtered)")
                continue

        # === Skip stacked if it adds no information vs base ===
        base_col = _first_present_base(df, model_type)
        if base_col and prediction_column in df.columns:
            s = pd.to_numeric(df[prediction_column], errors="coerce")
            b = pd.to_numeric(df[base_col],          errors="coerce")
            mask = s.notna() & b.notna()
            if mask.any():
                n_overlap = int(mask.sum())
                n_equal   = int((s[mask] == b[mask]).sum())
                if n_overlap > 0 and n_equal == n_overlap:
                    _log_once(
                        logging.WARNING,
                        key=f"skip_stack:{ticker}:{model_type}",
                        message=(f"⚠️ {ticker} — {model_type} (stacked) identical to base {base_col} "
                                 f"(overlap={n_overlap}, equal={n_equal}); skipping stacked backtest.")
                    )
                    continue

        try:
            final_capital, trades_df, metrics, trades_filename = run_backtest(
                df.copy(),
                ticker,
                model=proper,
                model_type=model_type,
                prediction_column=prediction_column,
                suffix="stack",
                sl_mult=atr_sl_multiplier,
                tp_mult=atr_tp_multiplier,
                use_trailing_stop=bool(getattr(CFG, "BACKTEST_USE_TRAILING_STOP", True))
            )
            result = create_result(
                ticker=ticker,
                model_name=proper,
                prediction_type="Stacked",
                prediction_source="preds",
                trades_df=trades_df,
                metrics=metrics,
                trades_filename=trades_filename,
                initial_capital=initial_capital,
                final_capital=final_capital,
                passed=True,
            )
            if result:
                result["Model Type"] = f"Stacked - {proper}"
                results.append(result)


        except Exception as e:
            logging.error(f"❌ Error in stacked model {model_type} for {ticker}: {e}")

    return results

def filter_ensemble_by_conf(
    df: pd.DataFrame,
    pred_col: str,
    conf_col: str,
    min_conf: float | None = None,
    threshold: float | None = None,
    fallback_low: float = 0.55,
    topk: int = 15,
    recent_days: int = 30,
) -> pd.DataFrame:

    floor = float(min_conf if min_conf is not None else (threshold if threshold is not None else 0.60))
    work = df.copy()

    pred = pd.to_numeric(work.get(pred_col, np.nan), errors="coerce")
    conf = pd.to_numeric(work.get(conf_col, np.nan), errors="coerce")

    keep   = pred.isin([0, 1]) & ((conf + EPS) >= floor)
    if int(keep.sum()) > 0:
        return work.loc[keep].copy()

    # Lower the bar once before giving up
    fb = min(float(fallback_low), max(0.0, floor - 0.10))
    keep_fb = pred.isin([0, 1]) & ((conf + EPS) >= fb)
    if int(keep_fb.sum()) > 0:
        return work.loc[keep_fb].copy()

    # Final fallback: Top‑K by confidence with recency preference if index is datetime
    work["_conf"] = conf
    if isinstance(work.index, pd.DatetimeIndex) and len(work) > 0:
        last = work.index.max()
        recent = work[(last - work.index).days <= int(recent_days)]
        if not recent.empty:
            return recent.sort_values("_conf", ascending=False).head(int(topk)).drop(columns=["_conf"])
    return work.sort_values("_conf", ascending=False).head(int(topk)).drop(columns=["_conf"])

def run_sltp_grid_for_ticker(
    df: pd.DataFrame,
    ticker: str,
    prediction_column: str,
    sl_range: Iterable[float] | None = None,
    tp_range: Iterable[float] | None = None,
) -> tuple[float, float]:
    """
    Grid-search best (sl_mult, tp_mult) for the given prediction column.
    Returns (best_sl, best_tp). Falls back to config ranges if none supplied.
    Skips combos that error or produce zero trades.
    Respects GRIDSEARCH_ENABLED / GRIDSEARCH_MAX_COMBOS / MIN_TRADES_FOR_GRIDSEARCH.
    """


    # If disabled, just return config defaults immediately (fast path)
    if not bool(GRIDSEARCH_ENABLED):
        return float(atr_sl_multiplier), float(atr_tp_multiplier)

    # Defaults from config if caller doesn't provide ranges
    if sl_range is None:
        sl_range = [float(atr_sl_multiplier) * k for k in (0.5, 0.75, 1.0, 1.25, 1.5)]
    if tp_range is None:
        tp_range = [float(atr_tp_multiplier) * k for k in (0.75, 1.0, 1.25, 1.5, 2.0)]

    pairs = list(product(sl_range, tp_range))

    # Cap the total number of combos (centered subset) to keep runs fast
    try:
        maxc = int(GRIDSEARCH_MAX_COMBOS)
    except Exception:
        maxc = 9
    if maxc > 0 and len(pairs) > maxc:
        # pick inner (more moderate) values preferentially
        sl_sorted = sorted(sl_range)
        tp_sorted = sorted(tp_range)
        center_sl = sl_sorted[1:-1] if len(sl_sorted) >= 3 else sl_sorted
        center_tp = tp_sorted[1:-1] if len(tp_sorted) >= 3 else tp_sorted
        pairs = list(product(center_sl, center_tp))[:maxc] or pairs[:maxc]

    best_sl, best_tp, best_score = float(atr_sl_multiplier), float(atr_tp_multiplier), -1e9

    # If too few trades, bail to defaults (saves time on sparse tickers)
    try:
        if len(df) < int(MIN_TRADES_FOR_GRIDSEARCH):
            return best_sl, best_tp
    except Exception:
        pass

    for sl, tp in pairs:
        try:
            cap, trades_df, metrics, _ = run_backtest(
                df.copy(),
                ticker=ticker,
                model="Ensemble",
                model_type="ensemble",
                prediction_column=prediction_column,   # use the one we were passed
                suffix="grid",
                sl_mult=float(sl),
                tp_mult=float(tp),
                use_trailing_stop=bool(getattr(CFG, "BACKTEST_USE_TRAILING_STOP", True)),
                use_ranking=True,
                # IMPORTANT: don’t re-apply min_conf here; df is already filtered
                min_conf_floor=MIN_CONF_FLOOR,
            )
            # score: Sharpe penalized by drawdown (mild)
            sharpe = float(metrics.get("Sharpe", 0.0)) if isinstance(metrics, dict) else 0.0
            mdd    = float(metrics.get("Max Drawdown (%)", 0.0) or metrics.get("Max Drawdown", 0.0))
            score  = sharpe - 0.1 * (mdd / 100.0)
            if score > best_score:
                best_score, best_sl, best_tp = score, float(sl), float(tp)
        except Exception:
            # any failed combo → skip quietly
            continue

    return best_sl, best_tp

def _accumulate_global_funnel(stats: Optional[Dict[str, float]]) -> None:
    if stats:
        _FUNNEL_GLOBAL.append(stats)

def _print_global_funnel(save_csv_path: Optional[str] = None) -> None:
    if not _FUNNEL_GLOBAL:
        logging.info("Global funnel: no data collected.")
        return
    cols = ["total","pred","volume","rsi","ema","macd","prob","final"]
    agg = {c: 0.0 for c in cols}
    for row in _FUNNEL_GLOBAL:
        for c in cols:
            agg[c] += float(row.get(c, 0.0))

    # compute pass rates
    def _rate(num, den): return (100.0 * num / den) if den else 0.0
    total = agg["total"]
    rates = {c: _rate(agg[c], total) for c in cols if c != "total"}

    # identify biggest drop (between consecutive stages)
    stages = ["pred","volume","rsi","ema","macd","prob","final"]
    drops = []
    prev = "total"
    for s in stages:
        den = agg[prev] if prev in agg else total
        num = agg[s]
        drop = (1.0 - (num / den)) if den else 0.0
        drops.append((s, drop))
        prev = s
    worst_stage, worst_drop = max(drops, key=lambda x: x[1])

    logging.info(
        "GLOBAL funnel: total=%.0f → pred=%.0f (%.1f%%) → vol=%.0f (%.1f%%) → rsi=%.0f (%.1f%%) → "
        "ema=%.0f (%.1f%%) → macd=%.0f (%.1f%%) → prob=%.0f (%.1f%%) → final=%.0f (%.1f%%) | biggest drop at %s (%.1f%%)",
        total,
        agg["pred"],   rates["pred"],
        agg["volume"], rates["volume"],
        agg["rsi"],    rates["rsi"],
        agg["ema"],    rates["ema"],
        agg["macd"],   rates["macd"],
        agg["prob"],   rates["prob"],
        agg["final"],  rates["final"],
        worst_stage, worst_drop * 100.0
    )

    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        
        with open(save_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["stage","rows_kept","kept_pct_of_total"])
            w.writerow(["total", int(total), 100.0])
            for s in stages:
                w.writerow([s, int(agg[s]), rates[s]])


# Collect global funnel stats here (list of dicts, one per ticker)
_FUNNEL_GLOBAL: list[Dict[str, float]] = []

def _log_filter_funnel(
    *,
    ticker: str,
    df: pd.DataFrame,
    filters_info: dict | None,
    combined_mask: pd.Series | None,
    logger: logging.Logger = logging.getLogger(__name__),
    save_csv_dir: str | None = None,
) -> Optional[Dict[str, float]]:
    """
    Print a per-gate survivor count + % funnel for one ticker and RETURN a stats dict:
    {'ticker', 'total','pred','volume','rsi','ema','macd','prob','final'}.
    """
    try:
        total = int(len(df) if df is not None else 0)
        if total == 0:
            logger.info("%s — funnel: no rows", ticker)
            return None

        fi = filters_info or {}
        m_pred   = fi.get("pred_gate",   pd.Series(True, index=df.index))
        m_vol    = fi.get("volume_gate", pd.Series(True, index=df.index))
        m_rsi    = fi.get("rsi_gate",    pd.Series(True, index=df.index))
        m_ema    = fi.get("ema_gate",    pd.Series(True, index=df.index))
        m_macd   = fi.get("macd_gate",   pd.Series(True, index=df.index))
        m_prob   = fi.get("prob_gate",   pd.Series(True, index=df.index))

        steps = []
        cur = pd.Series(True, index=df.index)

        def _push(name, mask):
            nonlocal cur
            cur = cur & pd.Series(mask).reindex(df.index, fill_value=False)
            kept = int(cur.sum())
            pct = 100.0 * kept / total if total else 0.0
            steps.append((name, kept, pct))

        _push("pred",   m_pred)
        _push("volume", m_vol)
        _push("rsi",    m_rsi)
        _push("ema",    m_ema)
        _push("macd",   m_macd)
        _push("prob",   m_prob)

        final_mask = combined_mask if combined_mask is not None else cur
        final_kept = int(pd.Series(final_mask).reindex(df.index, fill_value=False).sum())
        final_pct  = 100.0 * final_kept / total if total else 0.0

        logging.info(
            "%s — funnel: total=%d → pred=%d → vol=%d → rsi=%d → ema=%d → macd=%d → prob=%d → final=%d (%.1f%%)",
            ticker, total,
            steps[0][1], steps[1][1], steps[2][1], steps[3][1], steps[4][1], steps[5][1],
            final_kept, final_pct
        )
        for name, kept, pct in steps:
            logging.debug("%s — %-6s kept=%4d (%.1f%% of total)", ticker, name, kept, pct)

        # Only write funnel CSVs if enabled
        want_funnels = bool(getattr(CFG, "WRITE_FUNNELS", True))
        if save_csv_dir and want_funnels:
            os.makedirs(save_csv_dir, exist_ok=True)
            funnel_df = pd.DataFrame(
                steps + [("final", final_kept, final_pct)],
                columns=["gate", "kept_rows", "kept_pct_of_total"]
            )
            funnel_df.insert(0, "ticker", ticker)
            outp = os.path.join(save_csv_dir, f"{normalize_ticker_for_path(ticker)}_filter_funnel.csv")
            funnel_df.to_csv(outp, index=False)

        # return stats for global aggregation
        return {
            "ticker": ticker,
            "total": float(total),
            **{name: float(kept) for name, kept, _ in steps},
            "final": float(final_kept),
        }
    except Exception as e:
        logger.info("%s — funnel logger skipped: %s", ticker, e)
        return None

def process_ticker(
    ticker,
    features_path_or_df,
    *,
    min_conf_override: float | None = None,
    sl_mult_override: float | None = None,
    tp_mult_override: float | None = None,
):

    """
    Run base, stacked, and ensemble backtests for one ticker.
    Returns a list of summary rows.
    """
    results = []
    BASE_MODELS  = ["xgboost", "lightgbm", "catboost", "randomforest", "logisticregression"]
    STACK_MODELS = ["lightgbm", "catboost", "randomforest", "logisticregression"]  # no xgb in stacker

    # ---------- 1) Ingest + ensure base predictions (idempotent) ----------
    if isinstance(features_path_or_df, (str, os.PathLike)):
        # When a path is given, run the standard base-model attach flow
        df = run_base_models(ticker, features_path_or_df, BASE_MODELS)
    else:
        # A DataFrame was passed in
        df = features_path_or_df.copy()
        has_base = any(c in df.columns for c in (
            "XGBoost_pred","LightGBM_pred","CatBoost_pred","RandomForest_pred","LogisticRegression_pred",
            "xgb_pred","lgbm_pred","cat_pred","rf_pred","lr_pred"
        ))
        if not has_base:
            df = add_all_base_model_predictions(df, ticker, BASE_MODELS)

    # ---------- 1a) Early-skip if truly nothing to work with ----------
    try:
        # do we have any *_pred or *_stack_prediction columns at all?
        has_any_pred_cols = any(str(c).endswith("_pred") or str(c).endswith("_stack_prediction") for c in df.columns)
    except Exception:
        has_any_pred_cols = False

    if not has_any_pred_cols and not _has_any_model_artifacts(ticker):
        _log_once(logging.INFO, f"skip_no_artifacts:{ticker}",
                  f"ℹ️ {ticker} — no saved model artifacts and no prediction columns; skipping.")
        return results  # nothing to do for this ticker

    # Normalize canonical aliases once (no-op if already present)
    df = ensure_xgb_aliases(df)
    df = ensure_base_aliases(df)
    df = ensure_ensemble_aliases(df)

    # --- stacked predictions ONLY if enabled
    if getattr(CFG, "BACKTEST_STACKED", False):
        df = add_all_stacked_model_predictions(df, ticker, families=STACK_MODELS)
        df = ensure_stack_aliases(df)
        df = _migrate_legacy_stack_proba(df)
        # ... and only then run stacked backtests
        # (wrap the stacked backtest loop in the same if)

    # Drop exact-duplicates between base and stack to avoid double counting / confusion
    df = _maybe_drop_identical_stacks(df)

    # ---------- 2.5) Build ENSEMBLE ----------
    df = apply_ensemble_predictions_engine(
        df, out_prefix="ensemble",
        long_thresh=ENSEMBLE_LONG_THRESHOLD,
        short_thresh=ENSEMBLE_SHORT_THRESHOLD,
        prefer_stacked=CFG.PREFER_STACKED_IF_AVAILABLE,       # <-- use config
        weights=CFG.ENSEMBLE_WEIGHTS                          # <-- be explicit
    )

    ENSEMBLE_COL = "ensemble_pred_engine"

    # Optional: debug base vs stack agreement
    for fam, pfx in [("lightgbm","lgbm"),("catboost","cat"),("randomforest","rf"),("logisticregression","lr")]:
        base  = f"{pfx}_pred"
        stack = f"{pfx}_stack_prediction"
        if base in df.columns and stack in df.columns:
            try:
                b = pd.to_numeric(df[base], errors="coerce")
                s = pd.to_numeric(df[stack], errors="coerce")
                mask = b.notna() & s.notna()
                if mask.any():
                    same_frac = float((b[mask] == s[mask]).mean())
                    logging.info("%s — STACKED %s: '%s' vs base '%s', identical_frac=%.3f",
                                 ticker, fam, stack, base, same_frac)
                    if same_frac == 1.0:
                        _log_once(logging.WARNING, f"ident_stack:{ticker}:{fam}",
                                  f"⚠️ {ticker} — {fam} (stacked) identical to base {base}; skipping stacked backtest.")
            except Exception:
                pass

    try:
        metrics_dir = os.path.join(predict_path, "metrics")
        write_stack_vs_base_diagnostics(ticker, df, metrics_dir)
    except Exception as e:
        logging.info(f"ℹ️ {ticker} — diagnostics skipped: {e}")

    mode = getattr(CFG, "BACKTEST_MODE", "full").lower()
    # base
    if mode in ("full", "base_only"):
        if getattr(CFG, "BACKTEST_BASE", mode == "base_only"):
            results.extend(run_backtest_base_models(ticker, df.copy(), BASE_MODELS))

    # stacked
    if mode in ("full", "stacked_only"):
        if getattr(CFG, "BACKTEST_STACKED", mode == "stacked_only"):
            stack_bt = run_backtest_stacked_models(ticker, df.copy(), STACK_MODELS)
            if stack_bt:
                results.extend(stack_bt)

    # ---------- 5) Filter & Backtest ENSEMBLE ----------
    # Allow caller to override the min confidence used for the *pre-filter*
    try:
        conf_threshold = float(min_conf_override) if (min_conf_override is not None) else float(MIN_CONF_FLOOR)
    except Exception:
        conf_threshold = 0.0

    filtered = filter_ensemble_by_conf(
        df, pred_col=ENSEMBLE_COL, conf_col="ensemble_conf", min_conf=conf_threshold
    )

    valid_mask = pd.to_numeric(filtered.get(ENSEMBLE_COL), errors="coerce").isin([0, 1])
    if filtered.empty or not valid_mask.any():
        logging.info("Ensemble signals after side-conf ≥ %.2f: LONG=%d, SHORT=%d — skipping %s.",
                     conf_threshold, 0, 0, ticker)
        return results

    filtered["rank_score"] = pd.to_numeric(filtered.get("ensemble_conf", 0.0), errors="coerce")

    # Respect caller overrides first; else try grid; else fall back to config
    best_sl = float(sl_mult_override) if (sl_mult_override is not None) else float(atr_sl_multiplier)
    best_tp = float(tp_mult_override) if (tp_mult_override is not None) else float(atr_tp_multiplier)

    if sl_mult_override is None and tp_mult_override is None:
        try:
            # Only run grid if no overrides were provided
            best_sl, best_tp = run_sltp_grid_for_ticker(filtered, ticker, prediction_column=ENSEMBLE_COL)
        except Exception as e:
            logging.info(f"{ticker} — SL/TP grid not available/failed ({e}); using current best_sl/best_tp.")

    # >>> changed: pass FULL df via price_df_override while using 'filtered' as signals
    final_capital, trades_df, metrics, trades_filename = run_backtest(
        filtered, ticker, model="Ensemble", model_type="ensemble",
        prediction_column=ENSEMBLE_COL, suffix="ensemble",
        sl_mult=best_sl, tp_mult=best_tp,
        use_trailing_stop=bool(getattr(CFG, "BACKTEST_USE_TRAILING_STOP", True)),
        use_ranking=True,
        risk_pct=float(getattr(CFG, "RISK_PER_TRADE", RISK_PCT)),
        price_df_override=df,
        # IMPORTANT: we've already applied conf_threshold above — don't gate twice
        min_conf_floor=MIN_CONF_FLOOR,
    )

    row = create_result(
        ticker=ticker,
        model_name="Ensemble",
        prediction_type="Ensemble",
        prediction_source="preds",
        trades_df=trades_df,
        metrics=metrics,
        trades_filename=trades_filename,
        initial_capital=initial_capital,
        final_capital=final_capital,
        passed=True,
    )
    if row:
        row["Model Type"] = "Ensemble"
        results.append(row)

    # ---------- 6) Persist enriched predictions ----------
    try:
        os.makedirs(predict_path, exist_ok=True)
        tkr_fs = normalize_ticker_for_path(ticker)

        to_save = df.reset_index(drop=False).rename(columns={"index": "date"}) \
                  if isinstance(df.index, pd.DatetimeIndex) else df.copy()
        to_save.to_csv(os.path.join(predict_path, f"{tkr_fs}_test_features_with_predictions.csv"),
                       index=False, float_format="%.6g")

        hc = filtered.reset_index(drop=False).rename(columns={"index": "date"}) \
             if isinstance(filtered.index, pd.DatetimeIndex) else filtered.copy()
        hc_path = _export_hc_compat(ticker, hc, predict_path, top_k=200)
        if hc_path:
            logging.info("💾 Wrote high-confidence rows → %s", hc_path)

    except Exception as e:
        logging.warning(f"⚠️ Could not persist predictions for {ticker}: {e}")

    return results

def _normalize_prediction_columns(df):
    """
    Create canonical lowercase aliases and drop the TitleCase duplicates
    to prevent ambiguity later.
    """
    # create canonical aliases
    df = ensure_base_aliases(df)         # maps XGBoost_pred → xgb_pred, etc.
    df = ensure_stack_aliases(df)        # maps CatBoost_stack_prediction → cat_stack_prediction, etc.

    # Drop non-canonical prediction/probability columns
    drop_patterns = [
        r'^(XGBoost|LightGBM|CatBoost|RandomForest|LogisticRegression)_(pred|prob_class_[01]|max_prob|prob_diff)$',
        r'^(XGBoost|LightGBM|CatBoost|RandomForest|LogisticRegression)_stack_prediction$'
    ]
    to_drop = []
    for pat in drop_patterns:
        to_drop.extend([c for c in df.columns if re.match(pat, c)])
    # keep canonical lowercase
    keep = set([
        'xgb_pred','lgbm_pred','cat_pred','rf_pred','lr_pred',
        'xgb_prob_class_0','xgb_prob_class_1','xgb_max_prob','xgb_prob_diff',
        'lgbm_prob_class_0','lgbm_prob_class_1','lgbm_max_prob','lgbm_prob_diff',
        'cat_prob_class_0','cat_prob_class_1','cat_max_prob','cat_prob_diff',
        'rf_prob_class_0','rf_prob_class_1','rf_max_prob','rf_prob_diff',
        'lr_prob_class_0','lr_prob_class_1','lr_max_prob','lr_prob_diff',
        'xgb_stack_prediction','lgbm_stack_prediction','cat_stack_prediction','rf_stack_prediction','lr_stack_prediction',
    ])
    to_drop = [c for c in to_drop if c not in keep]
    if to_drop:
        df = df.drop(columns=to_drop, errors='ignore')
    return df

def run_base_models(
    ticker: str,
    features_path_or_df: str | Path | pd.DataFrame,
    base_models: list[str] | tuple[str, ...] | None = None,
    models_dir: str | Path = "3.Models_base",
) -> pd.DataFrame:
    """
    Ensure base-model prediction columns exist for the given ticker by either
    using what's already in the file/df, or loading the saved base models and
    generating predictions.

    - `base_models` may be long/short names (e.g., 'LightGBM', 'lgbm').
    - If a family's base columns already exist, it is skipped.
    """

    # ---- Load df if a path is given
    if isinstance(features_path_or_df, (str, Path)):
        df = pd.read_csv(features_path_or_df)
    else:
        df = features_path_or_df

    # ---- Normalize family list
    if not base_models:
        base_models = ["xgboost", "lightgbm", "catboost", "randomforest", "logisticregression"]
    families = [_to_prefix(f) for f in base_models]

    models_dir = Path(models_dir)
    wrote = []

    for fam in families:
        prev_cols = set(df.columns)
        try:
            # If you have a base-version helper similar to the stacked one:
            # add_base_prediction_for_family() should create:
            #   <prefix>_prediction, <prefix>_proba_long, <prefix>_proba_short
            df = add_base_prediction_for_family(df, ticker, fam, models_dir=models_dir)
            if set(df.columns) - prev_cols:
                wrote.append(fam)
        except FileNotFoundError:
            logging.warning("⚠️ Base model not found for %s (%s) in %s", ticker, fam, models_dir)
        except Exception as e:
            logging.error("❌ Error adding base prediction for %s (%s): %s", ticker, fam, e)

    if wrote:
        logging.info("✅ %s — added base predictions: %s", ticker, ", ".join(wrote))
    else:
        logging.info("ℹ️ %s — no new base predictions were written (likely already present).", ticker)

    return df
def ensure_base_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Symmetric aliases for base (non-stacked) columns of all families.
    Bridges ProperCase (<Model>_*) and short-prefix (lgbm_*, cat_*, rf_*, lr_*).
    """
    out = df

    pairs = [
        # LightGBM
        ("LightGBM_pred",            "lgbm_pred"),
        ("LightGBM_prob_class_1",    "lgbm_prob_class_1"),
        ("LightGBM_prob_class_0",    "lgbm_prob_class_0"),
        ("LightGBM_max_prob",        "lgbm_max_prob"),
        ("LightGBM_prob_diff",       "lgbm_prob_diff"),

        # CatBoost
        ("CatBoost_pred",            "cat_pred"),
        ("CatBoost_prob_class_1",    "cat_prob_class_1"),
        ("CatBoost_prob_class_0",    "cat_prob_class_0"),
        ("CatBoost_max_prob",        "cat_max_prob"),
        ("CatBoost_prob_diff",       "cat_prob_diff"),

        # RandomForest
        ("RandomForest_pred",        "rf_pred"),
        ("RandomForest_prob_class_1","rf_prob_class_1"),
        ("RandomForest_prob_class_0","rf_prob_class_0"),
        ("RandomForest_max_prob",    "rf_max_prob"),
        ("RandomForest_prob_diff",   "rf_prob_diff"),

        # LogisticRegression
        ("LogisticRegression_pred",         "lr_pred"),
        ("LogisticRegression_prob_class_1", "lr_prob_class_1"),
        ("LogisticRegression_prob_class_0", "lr_prob_class_0"),
        ("LogisticRegression_max_prob",     "lr_max_prob"),
        ("LogisticRegression_prob_diff",    "lr_prob_diff"),
    ]
    for a, b in pairs:
        if a in out.columns and b not in out.columns:
            out[b] = out[a].values
        elif b in out.columns and a not in out.columns:
            out[a] = out[b].values
    return out

def ensure_xgb_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create symmetric aliases between TitleCase XGBoost columns and short 'xgb_*' names.
    Idempotent: only creates the missing side, never overwrites existing data.
    Covers base and stacked outputs.
    """
    out = df

    # --- base ----
    pairs = [
        ("XGBoost_pred",             "xgb_pred"),
        ("XGBoost_prob_class_1",     "xgb_prob_class_1"),
        ("XGBoost_prob_class_0",     "xgb_prob_class_0"),
        ("XGBoost_max_prob",         "xgb_max_prob"),
        ("XGBoost_prob_diff",        "xgb_prob_diff"),

        # common legacy spellings → short alias
        ("xgboost_prediction",       "xgb_pred"),
        ("xgboost_pred",             "xgb_pred"),

        # --- stacked ---
        ("XGBoost_stack_prediction",       "xgb_stack_prediction"),
        ("XGBoost_stack_prob_class_1",     "xgb_stack_prob_class_1"),
        ("XGBoost_stack_prob_class_0",     "xgb_stack_prob_class_0"),
        ("XGBoost_stack_max_prob",         "xgb_stack_max_prob"),
        ("XGBoost_stack_prob_diff",        "xgb_stack_prob_diff"),
    ]

    for a, b in pairs:
        if a in out.columns and b not in out.columns:
            out[b] = out[a].values
        elif b in out.columns and a not in out.columns:
            out[a] = out[b].values

    return out

# === Canonical ensemble builder ==============================================
def run_ensemble_model(
    df: pd.DataFrame,
    *,
    out_prefix: str = "ensemble",
    prefer_stacked: bool = True
) -> pd.DataFrame:
    """
    Build engine-ready ensemble columns on the given DataFrame and return it.

    Emits:
      <out>_prob_class_0, <out>_prob_class_1, <out>_max_prob, <out>_prob_diff,
      <out>_conf, and <out>_pred_engine (Int64; 0=LONG, 1=SHORT).
      Also creates a convenience alias <out>_signal → <out>_pred_engine.
    """
    work = df.copy()

    # normalize expected base/stack columns (safe if already present)
    work = ensure_xgb_aliases(work)
    work = ensure_stack_aliases(work)

    # build ensemble with thresholds/weights from config
    work = apply_ensemble_predictions_engine(
        work,
        out_prefix=out_prefix,
        long_thresh=CFG.ENSEMBLE_LONG_THRESHOLD,
        short_thresh=CFG.ENSEMBLE_SHORT_THRESHOLD,
        prefer_stacked=CFG.PREFER_STACKED_IF_AVAILABLE,
        weights=CFG.ENSEMBLE_WEIGHTS
    )

    # Alias commonly used elsewhere
    pe_col = f"{out_prefix}_pred_engine"
    work[f"{out_prefix}_signal"] = work[pe_col]
    return work

def load_model_and_scaler(ticker, model_type, use_stack=False):
    """
    Robust loader for base and stacked artifacts.

    Base (preferred): 3.Models_base/{ticker}_{TitleCase}_pipeline.joblib
                      + {ticker}_{TitleCase}_scaler_columns.pkl
      Fallbacks: also try {ticker}_{longlower}_pipeline.joblib, {ticker}_{prefix}_pipeline.joblib,
                 and legacy triplets: *_model.pkl + *_scaler.pkl + *_scaler_columns.pkl

    Stack: 3.Models_stacked/{ticker}_{longlower}_stack_{model|scaler|scaler_columns}.pkl
           (also try TitleCase and short prefix variants)
    Returns: (model_or_pipeline, scaler_or_None, expected_columns) or (None, None, None)
    """
    
    # --- normalize model key into 3 canonical forms
    # short prefix -> title -> long lowercase used in your stacked filenames
    prefix_aliases = {
        "xgboost": "xgb", "xgb": "xgb",
        "lightgbm": "lgbm", "lgbm": "lgbm",
        "catboost": "cat", "cat": "cat",
        "randomforest": "rf", "rf": "rf", "random_forest": "rf",
        "logisticregression": "lr", "logreg": "lr", "logistic": "lr", "lr": "lr",
    }
    title_from_prefix = {
        "xgb": "XGBoost",
        "lgbm": "LightGBM",
        "cat": "CatBoost",
        "rf": "RandomForest",
        "lr": "LogisticRegression",
    }
    longlower_from_prefix = {
        "xgb": "xgboost",
        "lgbm": "lightgbm",
        "cat": "catboost",
        "rf": "randomforest",
        "lr": "logisticregression",
    }

    key = (model_type or "").strip().lower().replace(" ", "")
    pfx = prefix_aliases.get(key, key)  # tolerate inputs like 'LightGBM', 'lgbm', 'logreg'
    if pfx not in title_from_prefix:
        logging.error(f"Unknown model_type '{model_type}'.")
        return None, None, None

    Title = title_from_prefix[pfx]
    longlower = longlower_from_prefix[pfx]

    # These should already be imported from your config module
    try:
        base_dir = model_path_base
        stack_dir = model_path_stacked
    except NameError:
        logging.error("model_path_base / model_path_stacked not in scope.")
        return None, None, None

    # helper: strip meta columns when loading *base* expected columns
    def _clean_cols(cols, keep_stack=False):
        if not cols:
            return cols
        if keep_stack:
            return cols
        bad = ("_pred", "_prob", "stack")
        return [c for c in cols if not any(t in c.lower() for t in bad)]

    # === Build candidate specs in preferred order ===
    candidates = []

    if use_stack:
        # Your screenshots match the longlower form first
        candidates += [
            {
                "kind": "stack-longlower",
                "dir": stack_dir,
                "model":   f"{ticker}_{longlower}_stack_model.pkl",
                "scaler":  f"{ticker}_{longlower}_stack_scaler.pkl",
                "columns": f"{ticker}_{longlower}_stack_scaler_columns.pkl",
            },
            {
                "kind": "stack-title",
                "dir": stack_dir,
                "model":   f"{ticker}_{Title}_stack_model.pkl",
                "scaler":  f"{ticker}_{Title}_stack_scaler.pkl",
                "columns": f"{ticker}_{Title}_stack_scaler_columns.pkl",
            },
            {
                "kind": "stack-prefix",
                "dir": stack_dir,
                "model":   f"{ticker}_{pfx}_stack_model.pkl",
                "scaler":  f"{ticker}_{pfx}_stack_scaler.pkl",
                "columns": f"{ticker}_{pfx}_stack_scaler_columns.pkl",
            },
        ]
    else:
        # Prefer modern pipeline artifacts (TitleCase first)
        candidates += [
            {
                "kind": "base-pipeline-title",
                "dir": base_dir,
                "pipeline": f"{ticker}_{Title}_pipeline.joblib",
                "columns":  f"{ticker}_{Title}_scaler_columns.pkl",
                "columns_alt": f"{ticker}_{longlower}_scaler_columns.pkl",
            },
            {
                "kind": "base-pipeline-longlower",
                "dir": base_dir,
                "pipeline": f"{ticker}_{longlower}_pipeline.joblib",
                "columns":  f"{ticker}_{longlower}_scaler_columns.pkl",
                "columns_alt": f"{ticker}_{Title}_scaler_columns.pkl",
            },
            {
                "kind": "base-pipeline-prefix",
                "dir": base_dir,
                "pipeline": f"{ticker}_{pfx}_pipeline.joblib",
                "columns":  f"{ticker}_{pfx}_scaler_columns.pkl",
                "columns_alt": f"{ticker}_{Title}_scaler_columns.pkl",
            },
            # Legacy triplets (TitleCase → longlower → prefix)
            {
                "kind": "base-triplet-title",
                "dir": base_dir,
                "model":   f"{ticker}_{Title}_model.pkl",
                "scaler":  f"{ticker}_{Title}_scaler.pkl",
                "columns": f"{ticker}_{Title}_scaler_columns.pkl",
            },
            {
                "kind": "base-triplet-longlower",
                "dir": base_dir,
                "model":   f"{ticker}_{longlower}_model.pkl",
                "scaler":  f"{ticker}_{longlower}_scaler.pkl",
                "columns": f"{ticker}_{longlower}_scaler_columns.pkl",
            },
            {
                "kind": "base-triplet-prefix",
                "dir": base_dir,
                "model":   f"{ticker}_{pfx}_model.pkl",
                "scaler":  f"{ticker}_{pfx}_scaler.pkl",
                "columns": f"{ticker}_{pfx}_scaler_columns.pkl",
            },
        ]

    # === Try to load in order ===
    for spec in candidates:
        try:
            k = spec["kind"]
            root = spec["dir"]

            # Pipelines (modern)
            if "pipeline" in spec:
                pipe_path = os.path.join(root, spec["pipeline"])
                if not os.path.exists(pipe_path):
                    continue
                pipe = joblib_load(pipe_path)

                cols_main = os.path.join(root, spec["columns"])
                cols_alt  = os.path.join(root, spec.get("columns_alt", spec["columns"]))
                cols_path = cols_main if os.path.exists(cols_main) else cols_alt
                if not os.path.exists(cols_path):
                    raise FileNotFoundError(f"Columns file missing for {k}: {cols_main} / {cols_alt}")

                with open(cols_path, "rb") as f:
                    expected_cols = pickle.load(f)

                scaler = getattr(getattr(pipe, "named_steps", {}), "get", lambda *_: None)("scaler")
                return pipe, scaler, _clean_cols(expected_cols, keep_stack=False)



            # Triplets (.pkl)
            model_path   = os.path.join(root, spec["model"])
            scaler_path  = os.path.join(root, spec["scaler"])
            columns_path = os.path.join(root, spec["columns"])
            if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(columns_path)):
                continue

            with open(model_path, "rb") as f:   model = pickle.load(f)
            with open(scaler_path, "rb") as f:  scaler = pickle.load(f)
            with open(columns_path, "rb") as f: expected_cols = pickle.load(f)

            # keep stack feature columns intact when use_stack=True
            return model, scaler, _clean_cols(expected_cols, keep_stack=use_stack)

        except Exception as e:
            logging.error(f"❌ Failed to load {spec.get('kind')} for {ticker}/{model_type}: {e}")

    # Reaching here means none of the candidate files existed for this (ticker, model_type)
    logging.info(f"ℹ️ No matching artifacts found for {ticker} ({model_type}) [use_stack={use_stack}]")
    return None, None, None

def _load_tickers_robust():
    """
    Load tickers from 1.Stocks/SP500_Companies.xlsx using the same normalization
    training uses: prefer 'Ticker', else fall back to 'Symbol' (case-insensitive).
    """

    full_path = os.path.join(stock_file_path, stock_file)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Stocks file not found: {full_path}")

    df = pd.read_excel(full_path)
    # case-insensitive, trimmed headers
    cols = {c.strip().lower(): c for c in df.columns}
    if "ticker" in cols:
        col = cols["ticker"]
    elif "symbol" in cols:
        col = cols["symbol"]
    else:
        raise ValueError(f"❌ Neither 'Ticker' nor 'Symbol' column found in {full_path}")
    # uppercase, unique
    return (
        df[col]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )

def _file_has_any_pred_cols(path: str) -> bool:
    try:
        if not os.path.exists(path):
            return False
        tmp = pd.read_csv(path, nrows=5)
        cols = [c.lower() for c in tmp.columns]
        return any(c.endswith("_pred") or c.endswith("_stack_prediction") for c in cols)
    except Exception:
        return False

def _select_features_file(ticker: str) -> tuple[str, bool]:
    out_path, _ = ensure_enriched_predictions_file(ticker)
    return out_path, True

def _print_pre_save_summary(all_results, tickers_total: int, run_started_at: float, *, out_dir: str):
    elapsed = time.time() - float(run_started_at)
    N = len(all_results or [])
    def _safe(v, d=0.0):
        try:
            return float(v)
        except Exception:
            return d

    # Counts by type
    base_n = sum(1 for r in (all_results or []) if str(r.get("Prediction Type","")).lower().startswith("base"))
    stack_n = sum(1 for r in (all_results or []) if str(r.get("Prediction Type","")).lower().startswith("stack"))
    ens_n   = sum(1 for r in (all_results or []) if "ensemble" in str(r.get("Prediction Type","")).lower()
                                       or "ensemble" in str(r.get("Model Type","")).lower())

    # Aggregates
    rets    = [_safe(r.get("Model Return (%)")) for r in (all_results or [])]
    sharpes = [_safe(r.get("Sharpe")) for r in (all_results or [])]
    mdds    = [_safe(r.get("Max Drawdown (%)")) for r in (all_results or [])]
    trades  = sum(int(_safe(r.get("Trades"), 0)) for r in (all_results or []))

    # Bests (guarded)
    def _best(key, default=0):
        try:
            return max(all_results or [], key=lambda r: _safe(r.get(key), default))
        except ValueError:
            return None

    best_qml    = _best("QuantML Score", 0)
    best_ret    = _best("Model Return (%)", 0)
    best_sharpe = _best("Sharpe", 0)

    # Output path (known)
    out_path = os.path.join(out_dir, "backtest_summary.xlsx")

    lines = []
    lines.append("──────── Backtest Run Summary ────────")
    lines.append(f"Tickers processed       : {tickers_total}")
    lines.append(f"Models evaluated        : {N} (Base: {base_n}, Stacked: {stack_n}, Ensemble: {ens_n})")
    if rets:
        lines.append(f"Avg Return / Sharpe / MDD: {mean(rets):.2f}% / {mean(sharpes or [0]):.2f} / {mean(mdds or [0]):.2f}%")
    if best_qml:
        lines.append(f"Best QuantML            : {best_qml.get('Ticker','?')} — {best_qml.get('Model Type', best_qml.get('Model','?'))} (QML {best_qml.get('QuantML Score')})")
    if best_sharpe:
        lines.append(f"Best Sharpe             : {best_sharpe.get('Ticker','?')} — {best_sharpe.get('Model Type', best_sharpe.get('Model','?'))} (Sharpe {best_sharpe.get('Sharpe')})")
    if best_ret:
        lines.append(f"Best Return             : {best_ret.get('Ticker','?')} — {best_ret.get('Model Type', best_ret.get('Model','?'))} ({best_ret.get('Model Return (%)')}%)")
    lines.append(f"Trades (total)          : {trades:,}")
    lines.append(f"Elapsed time            : {elapsed:.1f}s")
    lines.append(f"Output (next)           : {out_path}")
    lines.append("───────────────────────────────────────")
    print("\n".join(lines))

def main():
    """
    Orchestrate backtests for all tickers in the stock list.

    For each ticker we:
      • Prefer an existing *_test_features_with_predictions.csv if it exists and is readable,
        otherwise fall back to *_test_features.csv.
      • Call process_ticker(ticker, features_path) which may return a list[dict] of results.
      • Accumulate ALL per-model results (base, stacked, ensemble) into all_results.
      • Persist a single summary via save_backtest_summary(all_results) at the end.
    """

    # ---- Paths / setup
    full_stock_file_path = os.path.join(stock_file_path, stock_file)
    os.makedirs(backtest_path, exist_ok=True)

    if not os.path.exists(full_stock_file_path):
        logging.warning(f"Stock file {stock_file} not found in {stock_file_path}, skipping.")
        return

    # ---- Load tickers (support either 'Ticker' or 'Symbol'), clean & dedup
    tickers_df = pd.read_excel(full_stock_file_path)
    col = None
    if "Ticker" in tickers_df.columns:
        col = "Ticker"
    elif "Symbol" in tickers_df.columns:
        col = "Symbol"
    else:
        logging.warning(f"No 'Ticker' or 'Symbol' column found in {full_stock_file_path}.")
        return

    tickers = (
        tickers_df[col]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": np.nan})  # in case "nan" strings slipped in
        .dropna()
        .unique()
        .tolist()
    )
    if not tickers:
        logging.warning("No tickers to process after cleaning.")
        return

    logging.info(f"🧾 Loaded {len(tickers)} unique tickers from {os.path.basename(full_stock_file_path)}")

    # ---- Aggregate all per-ticker results here
    all_results: list[dict] = []
    run_started_at = time.time()  # at top of main(), before the loop

    for ticker in _iter_progress(tickers, desc="Backtesting tickers", total=len(tickers)):
        t0 = time.time()
        try:
            features_path, used_preds = _select_features_file(ticker)
            res = process_ticker(ticker, features_path)

            # ---- Normalize and collect all result dicts
            if res:
                if isinstance(res, dict):
                    all_results.append(res)
                elif isinstance(res, list):
                    for r in res:
                        if isinstance(r, dict):
                            all_results.append(r)

                # Helpful per-ticker log line(s)
                for r in (res if isinstance(res, list) else [res]):
                    try:
                        logging.info(
                            f"{r.get('Ticker', ticker)} "
                            f"({r.get('Model Type', r.get('Model',''))}) "
                            f"Ret: {r.get('Model Return (%)', 0)}% | "
                            f"Sharpe: {r.get('Sharpe', 0)} | "
                            f"QuantML: {r.get('QuantML', r.get('QuantML Score', 0))} | "
                            f"Src: {'preds' if used_preds else 'raw'}"
                        )
                    except Exception:
                        pass
            else:
                logging.warning(f"⚠️ No results for {ticker}, skipping.")

            logging.info(f"⏱️ {ticker} processed in {time.time() - t0:.2f}s")

        except FileNotFoundError as e:
            logging.warning(f"⚠️ {ticker}: {e}")
        except MemoryError:
            logging.error(f"❌ MemoryError while processing {ticker}. Consider reducing dataset size.")
        except Exception as e:
            logging.error(f"❌ Error processing {ticker}: {e}", exc_info=True)
        finally:
            gc.collect()

    # ---- Save one consolidated summary
    if all_results:
        try:
            # NEW: print the pre-save 10–12 line summary
            _print_pre_save_summary(
                all_results,
                tickers_total=len(tickers),
                run_started_at=run_started_at,
                out_dir=backtest_path
            )
            # Existing save (this prints the 📤 line)
            save_backtest_summary(all_results)
        except Exception as e:
            logging.error(f"❌ Failed to save backtest summary: {e}", exc_info=True)
    else:
        logging.warning("No results to save.")
    _print_global_funnel(
        save_csv_path=os.path.join(predict_path, "metrics", "filter_funnels", "GLOBAL_filter_funnel.csv")
    )

if __name__ == "__main__":
    main()
