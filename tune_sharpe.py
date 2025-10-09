# tune_sharpe_thresholds.py
import os, json, itertools, time, importlib
import pandas as pd

import config as CFG

RESULTS_CSV = os.path.join(CFG.backtest_path, "threshold_sweep_results.csv")
BEST_JSON   = os.path.join(CFG.backtest_path, "threshold_sweep_best.json")
SUMMARY_XLSX= os.path.join(CFG.backtest_path, "backtest_summary.xlsx")

# ---- small, targeted grid around current values ----
BASE = {
    "ENSEMBLE_LONG_THRESHOLD":  getattr(CFG, "ENSEMBLE_LONG_THRESHOLD", 0.53),
    "ENSEMBLE_SHORT_THRESHOLD": getattr(CFG, "ENSEMBLE_SHORT_THRESHOLD", 0.53),
    "prob_threshold":           getattr(CFG, "prob_threshold", 0.52),
    "rsi_threshold_long":       getattr(CFG, "rsi_threshold_long", 43),
    "rsi_threshold_short":      getattr(CFG, "rsi_threshold_short", 57),
    "macd_hist_long":           getattr(CFG, "macd_hist_long", 0.80),
    "macd_hist_short":          getattr(CFG, "macd_hist_short", -0.16),
}

grid = {
    # ---- Ensemble direction thresholds (tied for symmetry) ----
    "ENSEMBLE_TIED": [0.55, 0.57, 0.59],

    # ---- Confidence floors / separation ----
    "prob_threshold":         [0.49, 0.50, 0.51],
    "xgb_prob_diff_quantile": [None, 0.20, 0.30],
    "MIN_CONF_FLOOR":         [0.58, 0.60, 0.62],

    # ---- Technical filters ----
    "rsi_threshold_long":  [43, 45, 47],
    "rsi_threshold_short": [53, 55, 57],
    "macd_hist_long":      [0.08, 0.12],
    "macd_hist_short":     [-0.12, -0.16],
    "volume_min":          [50_000, 150_000, 300_000],
    "volume_buffer_pct":   [0.20, 0.35, 0.50],
    "ema_buffer_pct_long": [1.000, 1.001, 1.002],
    "ema_buffer_pct_short":[0.999, 0.999, 0.998],  # mirror around 1.0 (<=)

    # ---- Brackets + trailing ----
    "atr_sl_multiplier":   [0.8, 1.1, 1.5],
    "atr_tp_multiplier":   [1.8, 2.3, 3.0],
    "TRAIL_ATR_MULT":      [0.5, 1.0, 1.5],

    # ---- Ensemble calibration ----
    "ENSEMBLE_TEMP":       [0.95, 1.00, 1.08],

    # ---- Capacity / risk ----
    "max_open_trades":     [10, 20, 30],
    "RISK_PER_TRADE":      [0.01, 0.02, 0.03],
}


# === Backtest engine knobs (also sweepable in Sharpe tuner) ===
RISK_PER_TRADE   = 0.02   # fraction of capital risked per trade (e.g., 0.005–0.03)
TRAIL_ATR_MULT   = 1.0    # trail stop by this many ATR when in profit (0.5–1.5 good)
ATR_BE_LOCK      = 1.0    # lock stop to breakeven once gain >= this * ATR (unused if trailing off)
MAX_HOLD_DAYS    = 65     # time-based exit safety (only used in the core variant)

# constraint: target Sharpe subject to enough trades
MIN_TOTAL_TRADES = 100        # adjust to your tolerance
MIN_TRADES_PER_ROW = 5        # ignore rows with <5 trades when averaging Sharpe
PRIMARY_METRIC = "Sharpe"
TIEBREAK_METRIC = "Model Return (%)"

def iter_param_dicts(grid):
    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        yield {k: v for k, v in zip(keys, values)}

def coerce_summary(df: pd.DataFrame) -> pd.DataFrame:
    # normalize column names expected from dBacktest writer
    m = {c.lower(): c for c in df.columns}
    # required fields
    req = {
        "ticker": m.get("ticker", "Ticker"),
        "sharpe": m.get("sharpe", "Sharpe"),
        "trades": m.get("trades", "Trades"),
        "ret":    m.get("model return (%)","Model Return (%)")
    }
    for need in req.values():
        if need not in df.columns:
            raise ValueError(f"Missing column in summary: {need}")
    return df.rename(columns={req["ret"]:"Model Return (%)",
                              req["sharpe"]:"Sharpe",
                              req["trades"]:"Trades",
                              req["ticker"]:"Ticker"})

def score_summary(df: pd.DataFrame):
    # filter out microscopically-sampled rows to avoid noisy Sharpe
    df2 = df[df["Trades"] >= MIN_TRADES_PER_ROW].copy()
    if df2.empty:
        return {"avg_sharpe": -999, "total_trades": 0, "avg_ret": -999}
    return {
        "avg_sharpe": float(df2["Sharpe"].mean()),
        "total_trades": int(df["Trades"].sum()),
        "avg_ret": float(df2["Model Return (%)"].mean())
    }

def run_once(params: dict):
    # apply overrides
    for k,v in params.items():
        setattr(CFG, k, v)
    # (optional) ensure prob-diff tail is disabled for sweep
    setattr(CFG, "xgb_prob_diff_quantile", None)

    # reload dBacktest to pick up new CFG values
    import dBacktest as BT
    importlib.reload(BT)

    # run backtest
    t0 = time.time()
    try:
        BT.main()   # uses current CFG
    except SystemExit:
        pass
    dur = time.time() - t0

    # read summary and compute metrics
    if not os.path.exists(SUMMARY_XLSX):
        raise FileNotFoundError(SUMMARY_XLSX)
    df = pd.read_excel(SUMMARY_XLSX)
    df = coerce_summary(df)
    s = score_summary(df)
    s["runtime_sec"] = round(dur, 1)
    return s

def main():
    results = []
    best = None

    for params in iter_param_dicts(grid):
        label = json.dumps(params, sort_keys=True)
        print(f"▶ sweep {label}")
        try:
            s = run_once(params)
            row = {"label": label, **params, **s}
            results.append(row)

            # trade constraint first
            if s["total_trades"] < MIN_TOTAL_TRADES:
                print(f"  ✖ trades={s['total_trades']} (<{MIN_TOTAL_TRADES})")
                continue

            # select best by avg_sharpe, then avg_ret
            if not best:
                best = row
            else:
                if (s["avg_sharpe"] > best["avg_sharpe"] + 1e-9) or \
                   (abs(s["avg_sharpe"] - best["avg_sharpe"]) < 1e-9 and s["avg_ret"] > best["avg_ret"]):
                    best = row

            print(f"  ✓ sharpe={s['avg_sharpe']:.2f}, trades={s['total_trades']}, ret={s['avg_ret']:.2f}, t={s['runtime_sec']}s")
        except Exception as e:
            results.append({"label": label, "error": str(e), **params})

    if results:
        os.makedirs(CFG.backtest_path, exist_ok=True)
        pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
        print(f"\n📄 wrote sweep results → {RESULTS_CSV}")

    if best:
        print("\n🏆 BEST (meets min trades):")
        keep = {k: best[k] for k in grid.keys()}
        print(json.dumps(keep, indent=2))
        with open(BEST_JSON, "w", encoding="utf-8") as f:
            json.dump(keep, f, indent=2)
        print(f"💾 saved best overrides → {BEST_JSON}")
    else:
        print("\n⚠ no parameter set met the trade constraint; see results CSV for diagnostics.")

if __name__ == "__main__":
    main()
