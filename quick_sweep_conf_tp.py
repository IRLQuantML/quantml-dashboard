# quick_sweep_conf_tp.py — Robust coordinate-ascent optimizer (multi-ticker, folds, restarts)
# - Multi-ticker, k-fold mean-Sharpe objective (with small MDD & low-trade penalties + hard trade floors)
# - Ensemble thresholds/confidence/tech filters/SLTP/trailing/risk optimization
# - Optional ENSEMBLE_WEIGHTS optimization with simplex + per-family caps
# - Caching, patience & adaptive step sizes (+ intra-pass full climb)
# - Optional tied thresholds & local-polish micro-grid
# - NEW: tunes volume gate (volume_min / volume_buffer_pct) and max_open_trades
# - Uses real ticker names in all backtest calls & artifacts

import os, math, copy, warnings, json, random
import pandas as pd, numpy as np

import config as CFG
import dBacktest as BT

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------- USER CONTROLS ---------------------------------
TICKERS    = ["BEN", "SLB", "PG", "VMC", "TRMB", "ADI", "ECL", "CMG", "WMT"]   # basket for robustness
N_RESTARTS = 5                 # random restarts around initial state
MAX_PASSES = 8                 # coord-ascent passes per restart
PATIENCE   = 5                 # patience inside a pass (after the ensemble block)
KFOLDS     = 3                 # walk-forward folds over the test window
SEED       = 1337

# Objective shaping (keeps Sharpe honest)
MIN_TRADES_FOR_GOOD = 25       # soft penalty when a segment has low trades
LOW_TRADES_PENALTY  = 0.20     # subtract from Sharpe if trades < MIN_TRADES_FOR_GOOD
MDD_PENALTY_LAMBDA  = 0.20     # subtract 0.1 * (MaxDD% / 100)

# Hard trade constraints (reject configs that are too thin)
HARD_MIN_TRADES_PER_SEGMENT = 5    # drop a segment if trades < 5
HARD_MIN_TOTAL_TRADES       = 150  # reject whole config if total trades < 100 across all segments

# Tie-break rule (if mean Sharpe equal within EPS)
TIEBREAK_ON_RETURN = True     # prefer higher mean return % on ties

# Ensemble thresholds coupling (optional)
USE_TIED_THRESHOLDS = True   # if True, drive both thresholds by ENSEMBLE_TIED

# Local “polish” micro-grid around the best state each restart
LOCAL_POLISH = True

# Ensemble-weight search (optional; set DO_WEIGHT_SEARCH=False to skip)
DO_WEIGHT_SEARCH       = True
WEIGHT_STEP            = 0.02      # coordinate step for weights (use 0.01 when close)
WEIGHT_CAP_PER_FAMILY  = float(getattr(CFG, "ENSEMBLE_WEIGHT_CAP_PER_FAMILY", 0.50))
WEIGHT_FAMILIES_TITLES = ["RandomForest", "LightGBM", "XGBoost", "CatBoost", "LogisticRegression"]

# IO
OUT_DIR   = getattr(CFG, "metrics_folder", "4.Metrics")
HIST_CSV  = os.path.join(OUT_DIR, "opt_coordinate_history.csv")
BEST_JSON = os.path.join(OUT_DIR, "best_state.json")
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-9  # numerical guard

# ---------------------- Small helpers ---------------------------------

# --- near the top, after imports and your sweep arrays ---
import config as CFG

CONF_FLOORS = [0.55, 0.56, 0.58, 0.60]
TP_MULTS    = [2.5, 2.8, 3.0]
SL_MULTS    = [1.5, 1.6, 1.65, 1.7]
TRAIL_MULTS = [1.0, 1.2, 1.3, 1.35]
RISK_LEVELS = [0.015, 0.018, 0.020]

def log_baseline_and_sweep():
    print("📌 Baseline config values:")
    print(f"   ATR SL Multiplier : {CFG.atr_sl_multiplier}")
    print(f"   ATR TP Multiplier : {CFG.atr_tp_multiplier}")
    print(f"   TRAIL ATR Mult    : {CFG.TRAIL_ATR_MULT}")
    print(f"   Min Conf Floor    : {CFG.MIN_CONF_FLOOR}")
    print(f"   Ensemble Temp     : {CFG.ENSEMBLE_TEMP}")
    print(f"   Ensemble LongThr  : {CFG.ENSEMBLE_LONG_THRESHOLD}")
    print(f"   Ensemble ShortThr : {CFG.ENSEMBLE_SHORT_THRESHOLD}")
    print(f"   Gridsearch Enabled: {CFG.GRIDSEARCH_ENABLED}\n")

    print("🎯 Sweep settings in this run:")
    print(f"   CONF_FLOORS = {CONF_FLOORS}")
    print(f"   TP_MULTS    = {TP_MULTS}")
    print(f"   SL_MULTS    = {SL_MULTS}")
    print(f"   TRAIL_MULTS = {TRAIL_MULTS}")


def _safe_float(val, default=0.0):
    """Coerce to float; treat None/NaN as default."""
    try:
        if val is None:
            return float(default)
        f = float(val)
        if math.isnan(f):
            return float(default)
        return f
    except Exception:
        return float(default)

def _bounded(val, meta):
    if "min" in meta: val = max(meta["min"], val)
    if "max" in meta: val = min(meta["max"], val)
    return val

def _clone_dict(d: dict) -> dict:
    return copy.deepcopy(d)

# ---------------------- Data helpers ----------------------------------
def _pred_path(tkr: str) -> str:
    fn = f"{BT.normalize_ticker_for_path(tkr)}_test_features_with_predictions.csv"
    return os.path.join(CFG.predict_path, fn)

def _load_pred_df(tkr: str) -> pd.DataFrame | None:
    p = _pred_path(tkr)
    if not os.path.exists(p):
        print(f"❌ Missing predictions file: {p}")
        return None
    df = pd.read_csv(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
    return df

def _compute_ensemble(df: pd.DataFrame, long_thr: float, short_thr: float, temp: float, weights: dict | None) -> pd.DataFrame:
    tmp = df.copy()
    tmp = BT.ensure_xgb_aliases(tmp)
    tmp = BT.ensure_base_aliases(tmp)
    try:
        tmp = BT.apply_ensemble_predictions_engine(
            tmp, out_prefix="ensemble",
            long_thresh=float(long_thr), short_thresh=float(short_thr),
            prefer_stacked=getattr(CFG, "PREFER_STACKED_IF_AVAILABLE", False),
            weights=(weights if isinstance(weights, dict) else getattr(CFG, "ENSEMBLE_WEIGHTS", {})),
            temp=float(temp),
        )
    except TypeError:
        old = getattr(CFG, "ENSEMBLE_TEMP", 1.0)
        try:
            setattr(CFG, "ENSEMBLE_TEMP", float(temp))
            tmp = BT.apply_ensemble_predictions_engine(
                tmp, out_prefix="ensemble",
                long_thresh=float(long_thr), short_thresh=float(short_thr),
                prefer_stacked=getattr(CFG, "PREFER_STACKED_IF_AVAILABLE", False),
                weights=(weights if isinstance(weights, dict) else getattr(CFG, "ENSEMBLE_WEIGHTS", {})),
            )
        finally:
            setattr(CFG, "ENSEMBLE_TEMP", old)
    if "ensemble_pred_engine" not in tmp.columns or "ensemble_conf" not in tmp.columns:
        raise KeyError("Ensemble outputs missing")
    return tmp

# ---------------------- Param space -------------------
def _initial_weights_from_config() -> dict:
    w = dict(getattr(CFG, "ENSEMBLE_WEIGHTS", {}))
    s = sum(max(float(v), 0.0) for v in w.values()) or 1.0
    w = {k: max(float(v), 0.0) / s for k, v in w.items() if k in WEIGHT_FAMILIES_TITLES}
    if len(w) != len(WEIGHT_FAMILIES_TITLES):
        even = 1.0 / len(WEIGHT_FAMILIES_TITLES)
        w = {k: even for k in WEIGHT_FAMILIES_TITLES}
    return _project_to_simplex_with_caps(w, WEIGHT_CAP_PER_FAMILY)

def _initial_weights_from_summary(path: str) -> dict | None:
    try:
        df = pd.read_excel(path)
    except Exception:
        return None
    if df is None or df.empty:
        return None

    fams = {"XGBoost","LightGBM","CatBoost","RandomForest","LogisticRegression"}
    out = {k: 0.0 for k in fams}
    if "Model" in df.columns and "Sharpe" in df.columns:
        for fam in fams:
            mask = df["Model"].astype(str).str.contains(fam, case=False, na=False)
            s = pd.to_numeric(df.loc[mask, "Sharpe"], errors="coerce")
            if s.notna().any():
                out[fam] = max(float(s.mean()), 0.0)
    if sum(out.values()) <= 0:
        return None
    s = sum(out.values())
    w = {k: v / s for k, v in out.items()}
    return _project_to_simplex_with_caps(w, WEIGHT_CAP_PER_FAMILY)

def _project_to_simplex_with_caps(w: dict, cap: float) -> dict:
    keys = list(w.keys())
    x = np.array([max(0.0, float(w[k])) for k in keys], dtype=float)
    if x.sum() <= 0: x[:] = 1.0
    x = x / x.sum()
    for _ in range(50):
        over = x > cap
        if not over.any():
            break
        x[over] = cap
        rem = 1.0 - x.sum()
        if rem <= 0:
            x = x / x.sum()
            break
        free = ~over
        if free.any():
            x[free] += rem / free.sum()
    if x.sum() != 0:
        x = x / x.sum()
    return {k: float(v) for k, v in zip(keys, x)}

def _initial_state() -> dict:
    long_thr  = _safe_float(getattr(CFG, "ENSEMBLE_LONG_THRESHOLD", 0.60))
    short_thr = _safe_float(getattr(CFG, "ENSEMBLE_SHORT_THRESHOLD", 0.60))
    tied_thr  = _safe_float(getattr(CFG, "ENSEMBLE_TIED", (long_thr + short_thr) / 2.0))

    return {
        # Ensemble & confidence
        "ENSEMBLE_LONG_THRESHOLD":   long_thr,
        "ENSEMBLE_SHORT_THRESHOLD":  short_thr,
        "ENSEMBLE_TIED":             tied_thr,   # <<< add this line
        "ENSEMBLE_TEMP":             _safe_float(getattr(CFG, "ENSEMBLE_TEMP", 0.85)),
        "MIN_CONF_FLOOR":            _safe_float(getattr(CFG, "MIN_CONF_FLOOR", 0.58)),

        # Technical gates
        "APPLY_RSI":           bool(getattr(CFG, "APPLY_RSI", True)),
        "rsi_threshold_long":  _safe_float(getattr(CFG, "rsi_threshold_long",  getattr(BT, "rsi_threshold_long", 50))),
        "rsi_threshold_short": _safe_float(getattr(CFG, "rsi_threshold_short", getattr(BT, "rsi_threshold_short", 50))),
        "APPLY_EMA_BUFFER":    bool(getattr(CFG, "APPLY_EMA_BUFFER", True)),
        "ema_buffer_pct_long": _safe_float(getattr(CFG, "ema_buffer_pct_long",  getattr(BT, "ema_buffer_pct_long", 1.000))),
        "ema_buffer_pct_short":_safe_float(getattr(CFG, "ema_buffer_pct_short", getattr(BT, "ema_buffer_pct_short", 1.000))),
        "APPLY_MACD":          bool(getattr(CFG, "APPLY_MACD", True)),
        "macd_hist_long":      _safe_float(getattr(CFG, "macd_hist_long",  getattr(BT, "macd_hist_long", 0.00))),
        "macd_hist_short":     _safe_float(getattr(CFG, "macd_hist_short", getattr(BT, "macd_hist_short", 0.00))),
        "APPLY_VOLUME":        bool(getattr(CFG, "APPLY_VOLUME", True)),

        # Tail gate
        "xgb_prob_diff_quantile": _safe_float(getattr(CFG, "xgb_prob_diff_quantile", 0.0), default=0.0),
        "USE_TAIL_GATE": bool(getattr(CFG, "xgb_prob_diff_quantile", 0.0) not in (None,)),

        # SL/TP, trailing, risk
        "atr_sl_multiplier": _safe_float(getattr(CFG, "atr_sl_multiplier", 1.3)),
        "atr_tp_multiplier": _safe_float(getattr(CFG, "atr_tp_multiplier", 2.2)),
        "TRAIL_ATR_MULT":    _safe_float(getattr(CFG, "TRAIL_ATR_MULT", 1.1)),
        "RISK_PER_TRADE":    _safe_float(getattr(CFG, "RISK_PER_TRADE", 0.0125)),

        # Volume gate thresholds (NEW)
        "volume_min":        _safe_float(getattr(CFG, "volume_min", 150_000)),
        "volume_buffer_pct": _safe_float(getattr(CFG, "volume_buffer_pct", 0.10)),

        # Capacity (NEW)
        "max_open_trades":   _safe_float(getattr(CFG, "max_open_trades", 6)),

        # Ensemble weights
        "ENSEMBLE_WEIGHTS": _initial_weights_from_config(),
    }

PARAMS = [
    # Ensemble & confidence (coupled block first)
    ("ENSEMBLE_LONG_THRESHOLD",  {"type":"float","step":0.01, "min":0.50,"max":0.70}),
    ("ENSEMBLE_SHORT_THRESHOLD", {"type":"float","step":0.01, "min":0.45,"max":0.65}),
    ("MIN_CONF_FLOOR",           {"type":"float","step":0.02, "min":0.48,"max":0.70}),

    # Temperature
    ("ENSEMBLE_TEMP",            {"type":"float","step":0.05, "min":0.70,"max":1.10}),

    # Optional tied thresholds (single knob drives both)
    *([("ENSEMBLE_TIED",         {"type":"float","step":0.01, "min":0.50,"max":0.70})] if USE_TIED_THRESHOLDS else []),

    # Tail gate on/off + quantile when on
    ("USE_TAIL_GATE",            {"type":"bool"}),
    ("xgb_prob_diff_quantile",   {"type":"float","step":0.05, "min":0.00,"max":0.80}),

    # Technical gates
    ("APPLY_RSI",                {"type":"bool"}),
    ("rsi_threshold_long",       {"type":"float","step":1.0,  "min":40,  "max":65}),
    ("rsi_threshold_short",      {"type":"float","step":1.0,  "min":35,  "max":60}),
    ("APPLY_EMA_BUFFER",         {"type":"bool"}),
    ("ema_buffer_pct_long",      {"type":"float","step":0.002,"min":0.990,"max":1.020}),
    ("ema_buffer_pct_short",     {"type":"float","step":0.002,"min":0.980,"max":1.010}),
    ("APPLY_MACD",               {"type":"bool"}),
    ("macd_hist_long",           {"type":"float","step":0.01, "min":-0.05,"max":0.10}),
    ("macd_hist_short",          {"type":"float","step":0.01, "min":0.00, "max":0.10}),
    ("APPLY_VOLUME",             {"type":"bool"}),

    # Volume gate thresholds (NEW)
    ("volume_min",        {"type":"float","step":50_000, "min":50_000, "max":400_000}),
    ("volume_buffer_pct", {"type":"float","step":0.05,   "min":0.10,   "max":0.50}),

    # Stops / trailing / risk (Sharpe-friendly floors)
    ("atr_sl_multiplier",  {"type":"float","step":0.10, "min":1.30,"max":2.00}),  # was min 0.70
    ("atr_tp_multiplier",  {"type":"float","step":0.20, "min":2.40,"max":3.20}),  # was min 1.40
    ("TRAIL_ATR_MULT",     {"type":"float","step":0.10, "min":1.00,"max":1.60}),  # was 0.8–1.7
    ("RISK_PER_TRADE",     {"type":"float","step":0.0025,"min":0.0125,"max":0.020}),  # was up to 0.025

    # Capacity (NEW)
    ("max_open_trades",          {"type":"float","step":1.0, "min":2.0, "max":12.0}),
]

ENSEMBLE_BLOCK = ["ENSEMBLE_LONG_THRESHOLD","ENSEMBLE_SHORT_THRESHOLD","MIN_CONF_FLOOR"]

# ---------------------- Overrides & evaluation -------------------------
def _apply_overrides_local(overrides: dict):
    # technical thresholds used by filters
    for k in ("rsi_threshold_long","rsi_threshold_short","ema_buffer_pct_long","ema_buffer_pct_short",
              "macd_hist_long","macd_hist_short","xgb_prob_diff_quantile",
              "volume_min","volume_buffer_pct"):
        if k in overrides:
            setattr(BT, k, overrides[k])

    # apply_* and runtime knobs
    for k in ("APPLY_RSI","APPLY_EMA_BUFFER","APPLY_MACD","APPLY_VOLUME"):
        if k in overrides:
            setattr(CFG, k, bool(overrides[k]))

    # risk & trailing
    if "RISK_PER_TRADE" in overrides:
        setattr(CFG, "RISK_PER_TRADE", float(overrides["RISK_PER_TRADE"]))
    if "TRAIL_ATR_MULT" in overrides:
        setattr(CFG, "TRAIL_ATR_MULT", float(overrides["TRAIL_ATR_MULT"]))

    # capacity
    if "max_open_trades" in overrides:
        setattr(CFG, "max_open_trades", int(round(float(overrides["max_open_trades"]))))

    # Tail gate toggle
    if "USE_TAIL_GATE" in overrides:
        if not overrides["USE_TAIL_GATE"]:
            setattr(BT, "xgb_prob_diff_quantile", None)
        else:
            setattr(BT, "xgb_prob_diff_quantile", float(overrides.get("xgb_prob_diff_quantile", 0.0)))

def _objective_score_from_metrics(metrics: dict, trades_df: pd.DataFrame | None) -> tuple[float,int,float]:
    """Return (score_with_penalties, trades_count, avg_return_pct) for a segment."""
    s   = float(metrics.get("Sharpe", float("nan")))
    if not np.isfinite(s):
        return float("nan"), 0, float("nan")
    mdd = float(metrics.get("Max Drawdown (%)", metrics.get("Max Drawdown", 0.0)))
    trades = int(len(trades_df)) if trades_df is not None else 0
    avg_ret = float(metrics.get("Model Return (%)", 0.0))

    # hard floor for segment sampling
    if trades < HARD_MIN_TRADES_PER_SEGMENT:
        return float("nan"), trades, avg_ret

    s -= MDD_PENALTY_LAMBDA * (mdd / 100.0)
    if trades < MIN_TRADES_FOR_GOOD:
        s -= LOW_TRADES_PENALTY
    return float(s), trades, avg_ret

def _sharpe_for_one(df_base, state, ticker):
    _apply_overrides_local(state)

    long_thr  = state["ENSEMBLE_LONG_THRESHOLD"]
    short_thr = state["ENSEMBLE_SHORT_THRESHOLD"]
    if USE_TIED_THRESHOLDS:
        t = float(state.get("ENSEMBLE_TIED", (long_thr + short_thr)/2.0))
        long_thr = short_thr = t
        # keep state in sync so subsequent steps see the effect
        state["ENSEMBLE_LONG_THRESHOLD"]  = t
        state["ENSEMBLE_SHORT_THRESHOLD"] = t

    weights = state.get("ENSEMBLE_WEIGHTS", None)
    ens_df = _compute_ensemble(df_base, long_thr, short_thr, state["ENSEMBLE_TEMP"], weights=weights)

    cap, trades_df, metrics, _ = BT.run_backtest(
        ens_df, ticker=ticker, model="Ensemble", model_type="ensemble",
        prediction_column="ensemble_pred_engine", suffix="coord",
        sl_mult=float(state["atr_sl_multiplier"]), tp_mult=float(state["atr_tp_multiplier"]),
        use_trailing_stop=True, use_ranking=True, min_conf_floor=float(state["MIN_CONF_FLOOR"]),
        price_df_override=df_base
    )
    return _objective_score_from_metrics(metrics, trades_df)

def _kfold_indices(idx: pd.DatetimeIndex, k: int) -> list[tuple[pd.Timestamp,pd.Timestamp]]:
    """Split index into k contiguous folds."""
    n = len(idx)
    if k <= 1 or n < k:
        return [(idx.min(), idx.max())]
    bounds = [int(round(i * n / k)) for i in range(k + 1)]
    out = []
    for i in range(k):
        lo = idx[bounds[i]]
        hi = idx[bounds[i+1] - 1]
        out.append((lo, hi))
    return out

# ---------- caching ----------
_CACHE = {}
_CACHE_META = {}  # key -> {"total_trades": int, "mean_ret": float}

def _key(state):
    # round floats to cut cache size; skip weights JSON (handled separately)
    def r(v):
        if isinstance(v, float): return round(v, 6)
        return v
    import hashlib
    w = state.get("ENSEMBLE_WEIGHTS", None)
    wsig = ""
    if isinstance(w, dict):
        items = sorted((k, round(float(v),6)) for k,v in w.items())
        wsig = hashlib.md5(str(items).encode("utf-8")).hexdigest()[:8]
    base = tuple(sorted((k, r(v)) for k, v in state.items() if k != "ENSEMBLE_WEIGHTS"))
    return (base, wsig)

def _eval_objective(state: dict, dfs: dict[str,pd.DataFrame]) -> float:
    """Mean Sharpe across tickers and K folds (walk-forward); fills _CACHE_META for tie-breaks."""
    scores, trades_list, rets = [], [], []
    for tkr, df in dfs.items():
        if df is None or df.empty:
            continue
        for lo, hi in _kfold_indices(df.index, KFOLDS):
            seg = df.loc[lo:hi].copy()
            if len(seg) < 50:
                continue
            sc, tr, ar = _sharpe_for_one(seg, state, ticker=tkr)
            if np.isfinite(sc):
                scores.append(sc); trades_list.append(tr); rets.append(ar)

    # ... after computing scores, trades_list, rets ...
    if not scores:
        return float("nan")

    # Penalize fold-to-fold instability (Sharpe std across folds)
    sharpe_std = float(np.nanstd(scores)) if len(scores) > 1 else 0.0
    STD_PENALTY_LAMBDA = 0.10  # <= tune 0.05–0.15
    mean_sharpe = float(np.nanmean(scores)) - STD_PENALTY_LAMBDA * sharpe_std
    total_trades = int(np.nansum(trades_list))
    mean_ret = float(np.nanmean(rets)) if rets else float("nan")

    # hard total-trade constraint
    if total_trades < HARD_MIN_TOTAL_TRADES:
        return float("nan")

    mean_sharpe = float(np.mean(scores))

    # store meta for tie-break
    _CACHE_META[_key(state)] = {"total_trades": total_trades, "mean_ret": mean_ret}
    return mean_sharpe

def _eval_objective_cached(state, dfs):
    k = _key(state)
    if k in _CACHE:
        return _CACHE[k]
    val = _eval_objective(state, dfs)
    _CACHE[k] = val
    return val

# ---------- search steps ----------
def _hill_step(state, name, meta, dfs, best_score):
    base_step = float(meta["step"])
    if name not in state:
        # initialize missing scalar params from config or a sensible default
        if name == "ENSEMBLE_TIED":
            state[name] = float(state.get("ENSEMBLE_LONG_THRESHOLD", 0.60) + state.get("ENSEMBLE_SHORT_THRESHOLD", 0.60)) / 2.0
        else:
            state[name] = float(meta.get("min", 0.0))
    cur_val = float(state[name])

    def eval_at(x):
        s2 = copy.deepcopy(state)
        s2[name] = _bounded(float(x), meta)
        return _eval_objective_cached(s2, dfs)

    trace = []
    cur_best = best_score
    best_val = cur_val

    def ascend_from(start_val, step):
        v = start_val
        s = cur_best
        st = step
        improved_any = False
        while True:
            nxt = _bounded(v + st, meta)
            if abs(nxt - v) < 1e-12:
                break
            sc = eval_at(nxt); trace.append((nxt, sc))
            if np.isfinite(sc) and sc > s + EPS:
                v, s = nxt, sc
                st = st * 2.0  # grow step if we keep improving
                improved_any = True
            else:
                break
        return v, s, improved_any

    # + direction climb
    v_up, s_up, ok_up = ascend_from(best_val, base_step)
    if ok_up:
        best_val, cur_best = v_up, s_up

    # - direction climb (also check minus even if plus improved)
    if not ok_up or True:
        v_dn, s_dn, ok_dn = ascend_from(best_val, -base_step)
        if ok_dn and s_dn > cur_best + EPS:
            best_val, cur_best = v_dn, s_dn

    # If nothing worked, try half-step both ways once
    if best_val == cur_val:
        for st in (0.5*base_step, -0.5*base_step):
            nxt = _bounded(cur_val + st, meta)
            if abs(nxt - cur_val) < 1e-12:
                continue
            sc = eval_at(nxt); trace.append((nxt, sc))
            if np.isfinite(sc) and sc > cur_best + EPS:
                best_val, cur_best = nxt, sc
                break

    improved = (best_val != cur_val)
    return (best_val if improved else cur_val), (cur_best if improved else best_score), trace

def _try_bool(state, name, dfs, best_score):
    tried = []
    for v in (False, True):
        s2 = _clone_dict(state); s2[name] = v
        sh = _eval_objective_cached(s2, dfs)
        tried.append((v, sh))
    tried.sort(key=lambda x: (x[1] if np.isfinite(x[1]) else -1e9), reverse=True)
    best_v, best_s = tried[0]
    improved = (np.isfinite(best_s) and best_s > best_score + EPS)
    return best_v, (best_s if improved else best_score), tried

# ---------- ensemble weight search ----------
def _seed_weights_from_summary_if_available(weights: dict) -> dict:
    xl = os.path.join(getattr(CFG, "backtest_path", "5.Backtest"), "backtest_summary.xlsx")
    if os.path.exists(xl):
        w = _initial_weights_from_summary(xl)
        if isinstance(w, dict):
            return w
    return weights

def _tweak_weight_once(w: dict, fam: str, delta: float) -> dict:
    new = dict(w)
    new[fam] = max(0.0, float(new.get(fam, 0.0)) + float(delta))
    return _project_to_simplex_with_caps(new, WEIGHT_CAP_PER_FAMILY)

def _weight_step_coordinate_ascent(state: dict, dfs: dict[str,pd.DataFrame], best_score: float, step: float) -> tuple[dict, float, list]:
    trace = []
    w = dict(state.get("ENSEMBLE_WEIGHTS", _initial_weights_from_config()))
    improved = False
    cur_best = best_score

    for fam in WEIGHT_FAMILIES_TITLES:
        w_up = _tweak_weight_once(w, fam, +step)
        s2 = _clone_dict(state); s2["ENSEMBLE_WEIGHTS"] = w_up
        sc_up = _eval_objective_cached(s2, dfs)

        w_dn = _tweak_weight_once(w, fam, -step)
        s3 = _clone_dict(state); s3["ENSEMBLE_WEIGHTS"] = w_dn
        sc_dn = _eval_objective_cached(s3, dfs)

        cand = [(w, cur_best), (w_up, sc_up), (w_dn, sc_dn)]
        cand.sort(key=lambda kv: (kv[1] if np.isfinite(kv[1]) else -1e9), reverse=True)
        best_w, best_sc = cand[0]
        trace.append((fam, float(step), float(best_sc)))

        if np.isfinite(best_sc) and (best_sc > cur_best + EPS):
            w = best_w
            cur_best = best_sc
            improved = True

    return (w if improved else state["ENSEMBLE_WEIGHTS"]), (cur_best if improved else best_score), trace

# ---------------------- Param order -----------------------------------
def _build_param_order(rng: random.Random) -> list[tuple[str, dict]]:
    name_to_meta = {k: v for k, v in PARAMS}
    order = [(n, name_to_meta[n]) for n in ENSEMBLE_BLOCK if n in name_to_meta]
    rest = [(k, v) for k, v in PARAMS if k not in ENSEMBLE_BLOCK]
    rng.shuffle(rest)
    order.extend(rest)
    return order

def _random_perturb(s: dict, scale=1.0) -> dict:
    out = _clone_dict(s)
    rng = random.Random(SEED + random.randint(0, 10_000))
    for name, meta in PARAMS:
        if meta["type"] == "bool":
            if rng.random() < 0.10:
                out[name] = not bool(out[name])
        else:
            step = float(meta["step"])
            jitter = rng.uniform(-2*step, 2*step) * scale
            out[name] = _bounded(_safe_float(out[name]) + jitter, meta)
    # small weight jitter, then project
    ww = dict(out.get("ENSEMBLE_WEIGHTS", _initial_weights_from_config()))
    for fam in WEIGHT_FAMILIES_TITLES:
        ww[fam] = max(0.0, ww.get(fam, 0.0) + rng.uniform(-0.02, 0.02))
    out["ENSEMBLE_WEIGHTS"] = _project_to_simplex_with_caps(ww, WEIGHT_CAP_PER_FAMILY)
    return out

# ---------------------- Local polish (micro-grid) ----------------------
def _local_polish(state, dfs):
    """Small micro-grid around current best to polish a few sensitive knobs."""
    center = {
        "ENSEMBLE_LONG_THRESHOLD": state["ENSEMBLE_LONG_THRESHOLD"],
        "ENSEMBLE_SHORT_THRESHOLD": state["ENSEMBLE_SHORT_THRESHOLD"],
        "MIN_CONF_FLOOR": state["MIN_CONF_FLOOR"],
        "atr_sl_multiplier": state["atr_sl_multiplier"],
        "atr_tp_multiplier": state["atr_tp_multiplier"],
        "TRAIL_ATR_MULT": state["TRAIL_ATR_MULT"],
        "rsi_threshold_long": state["rsi_threshold_long"],
        "rsi_threshold_short": state["rsi_threshold_short"],
        "ema_buffer_pct_long": state["ema_buffer_pct_long"],
        "ema_buffer_pct_short": state["ema_buffer_pct_short"],
    }
    steps = {
        "ENSEMBLE_LONG_THRESHOLD": [ -0.01, 0.0, +0.01 ],
        "ENSEMBLE_SHORT_THRESHOLD":[ -0.01, 0.0, +0.01 ],
        "MIN_CONF_FLOOR":          [ -0.02, 0.0, +0.02 ],
        "atr_sl_multiplier":       [ -0.10, 0.0, +0.10 ],
        "atr_tp_multiplier":       [ -0.20, 0.0, +0.20 ],
        "TRAIL_ATR_MULT":          [ -0.10, 0.0, +0.10 ],
        "rsi_threshold_long":      [ -1.0,  0.0, +1.0  ],
        "rsi_threshold_short":     [ -1.0,  0.0, +1.0  ],
        "ema_buffer_pct_long":     [ -0.002,0.0, +0.002],
        "ema_buffer_pct_short":    [ -0.002,0.0, +0.002],
    }

    best = _clone_dict(state)
    best_val = _eval_objective_cached(best, dfs)

    for k, offs in steps.items():
        base = center[k]
        for d in offs:
            trial = _clone_dict(best)
            trial[k] = base + d
            val = _eval_objective_cached(trial, dfs)
            if np.isfinite(val) and val > best_val + EPS:
                best, best_val = trial, val
    return best, best_val

# ---------------------- Main optimize loop ------------------------------------
def main():
    random.seed(SEED); np.random.seed(SEED)

    # Load data once per ticker
    dfs = {t: _load_pred_df(t) for t in TICKERS}
    for t, d in dfs.items():
        if d is None or d.empty:
            print(f"⚠️ {t}: no data")

    # Print baselines & sweep ranges exactly once per run
    log_baseline_and_sweep()
    # --- Optional brute-force sweep of floor/SL/TP/Trail combos ---
    print("\n🔎 Running brute-force sweep of CONF_FLOORS × TP × SL × TRAIL × RISK...")
    for conf in CONF_FLOORS:
        for tp in TP_MULTS:
            for sl in SL_MULTS:
                for trail in TRAIL_MULTS:
                    for risk in RISK_LEVELS:
                        test_state = _clone_dict(_initial_state())
                        test_state["MIN_CONF_FLOOR"]   = conf
                        test_state["atr_tp_multiplier"] = tp
                        test_state["atr_sl_multiplier"] = sl
                        test_state["TRAIL_ATR_MULT"]    = trail
                        test_state["RISK_PER_TRADE"]    = risk
                        score = _eval_objective_cached(test_state, dfs)
                        if np.isfinite(score):
                            print(f"  conf={conf:.3f} tp={tp:.2f} sl={sl:.2f} trail={trail:.2f} risk={risk:.3f} | Sharpe={score:.3f}")
    print("✅ Sweep done\n")

    hist_rows = []
    global_best = None
    global_best_state = None
    
    for r in range(N_RESTARTS):
        print(f"\n=== Restart {r+1}/{N_RESTARTS} ===")
        state = _random_perturb(_initial_state(), scale=1.0) if r > 0 else _initial_state()
        if DO_WEIGHT_SEARCH:
            state["ENSEMBLE_WEIGHTS"] = _seed_weights_from_summary_if_available(state["ENSEMBLE_WEIGHTS"])
        _CACHE.clear(); _CACHE_META.clear()
        best_score = _eval_objective_cached(state, dfs)
        print(f"Start score (mean Sharpe): {best_score:.4f}")

        for p in range(1, MAX_PASSES + 1):
            changed = False
            stall = 0
            rng = random.Random(SEED + r*1000 + p)
            param_order = _build_param_order(rng)
            print(f"  — Pass {p}")

            for name, meta in param_order:
                old = state[name] if name in state else None
                improved_here = False
                prev_score = best_score

                if meta["type"] == "bool":
                    best_v, best_s, tried = _try_bool(state, name, dfs, best_score)
                    if np.isfinite(best_s) and (best_s > best_score + EPS):
                        state[name] = best_v
                        best_score = best_s
                        changed = True
                        improved_here = True

                    hist_rows.append({
                        "restart": r + 1,
                        "pass": p,
                        "param": name,
                        "old": old,
                        "new": state[name],
                        "score": best_score,
                        "delta": (best_score - prev_score) if (np.isfinite(best_score) and np.isfinite(prev_score)) else np.nan,
                        "detail": json.dumps(tried, default=str),
                    })
                    if improved_here:
                        print(f"    {name}: {old} -> {state[name]} | {best_score:.4f}")

                else:
                    best_v, best_s, trace = _hill_step(state, name, meta, dfs, best_score)
                    if np.isfinite(best_s) and (best_s > best_score + EPS) and (not isinstance(old, float) or abs(best_v - _safe_float(old)) > 1e-12):
                        state[name] = best_v
                        best_score = best_s
                        changed = True
                        improved_here = True

                    hist_rows.append({
                        "restart": r + 1,
                        "pass": p,
                        "param": name,
                        "old": old,
                        "new": state[name],
                        "score": best_score,
                        "delta": (best_score - prev_score) if (np.isfinite(best_score) and np.isfinite(prev_score)) else np.nan,
                        "detail": json.dumps(trace, default=str),
                    })
                    if improved_here:
                        print(f"    {name}: {old} -> {state[name]} | {best_score:.4f}")

                # patience: disabled for the coupled ensemble block
                if name in ENSEMBLE_BLOCK:
                    continue
                stall = 0 if improved_here else (stall + 1)
                if stall >= PATIENCE:
                    print(f"    Patience reached after '{name}' — early stop within pass.")
                    break

            # Ensemble weight search at the end of each pass (optional)
            if DO_WEIGHT_SEARCH:
                old_w = _clone_dict(state["ENSEMBLE_WEIGHTS"])
                new_w, new_score, w_trace = _weight_step_coordinate_ascent(state, dfs, best_score, WEIGHT_STEP)
                hist_rows.append({
                    "restart": r + 1,
                    "pass": p,
                    "param": "ENSEMBLE_WEIGHTS",
                    "old": old_w,
                    "new": new_w,
                    "score": new_score,
                    "delta": (new_score - best_score) if (np.isfinite(new_score) and np.isfinite(best_score)) else np.nan,
                    "detail": json.dumps(w_trace, default=str),
                })
                if np.isfinite(new_score) and new_score > best_score + EPS:
                    state["ENSEMBLE_WEIGHTS"] = new_w
                    best_score = new_score
                    changed = True
                    print(f"    ENSEMBLE_WEIGHTS updated | {best_score:.4f}")

            if not changed:
                print("  No improvement; early stop.")
                break

        # Local micro-polish near the current state
        if LOCAL_POLISH:
            polished_state, polished_score = _local_polish(state, dfs)
            if np.isfinite(polished_score) and polished_score > best_score + EPS:
                state, best_score = polished_state, polished_score
                print(f"    Local polish improved score → {best_score:.4f}")

        # Choose global best (with tie-break on mean return if enabled)
        k = _key(state)
        state_meta = _CACHE_META.get(k, {"mean_ret": -1e9})
        best_meta  = _CACHE_META.get(_key(global_best_state) if global_best_state else ("", ""), {"mean_ret": -1e9})

        if (global_best is None) or \
           (np.isfinite(best_score) and best_score > global_best + EPS) or \
           (TIEBREAK_ON_RETURN and np.isfinite(best_score) and global_best is not None and abs(best_score - global_best) <= EPS and
            float(state_meta.get("mean_ret", -1e9)) > float(best_meta.get("mean_ret", -1e9))):
            global_best = best_score
            global_best_state = _clone_dict(state)
            print(f"  ✅ New global best: {global_best:.4f}")

    # Save artifacts
    if hist_rows:
        pd.DataFrame(hist_rows).to_csv(HIST_CSV, index=False)
        print(f"📄 History saved → {HIST_CSV}")

    if global_best_state:
        with open(BEST_JSON, "w", encoding="utf-8") as f:
            json.dump({"best_score": global_best, "state": global_best_state}, f, indent=2)
        print(f"🏆 Best saved → {BEST_JSON}")
        print("Best settings:")
        for k in sorted(global_best_state.keys()):
            v = global_best_state[k]
            if k == "ENSEMBLE_WEIGHTS":
                print("  ENSEMBLE_WEIGHTS:")
                for fam in WEIGHT_FAMILIES_TITLES:
                    print(f"    {fam}: {v.get(fam, 0.0):.5f}")
            else:
                print(f"  {k}: {v}")
    else:
        print("No valid improvements found.")

if __name__ == "__main__":
    main()
