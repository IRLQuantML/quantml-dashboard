# dBacktest_refine.py
# Unified launcher for targeted sweeps & tuners with a simple menu.
# - Keeps everything in one file (you can delete older sweep/tuner scripts if you wish)
# - Calls dBacktest.py in a subprocess with environment overrides
# - Writes/extends 5.Backtest/auto_runs_summary.csv and saves 5.Backtest/auto_leaderboard.csv

import os
import sys
import json
import time
import math
import random
from pathlib import Path
from itertools import product, islice
from datetime import datetime

import pandas as pd

# --- Paths / constants --------------------------------------------------------
PY = sys.executable
ROOT = Path(__file__).resolve().parent
DBACKTEST = ROOT / "dBacktest.py"

# Output dirs
BACKTEST_DIR = ROOT / "5.Backtest"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
AUTO_SUMMARY = BACKTEST_DIR / "auto_runs_summary.csv"
AUTO_LEADER  = BACKTEST_DIR / "auto_leaderboard.csv"

# Optional: cap very large grids during quick checks
DEFAULT_MAX_COMBOS_QUICK = 80          # cap for "Quick sweep"
DEFAULT_MIN_TRADES_ROW   = 6           # leaderboard filter
SLEEP_BETWEEN_RUNS_SEC   = 0.30        # avoid same-second run_ids

# Composite ranking weights (Return ↑ / Sharpe ↑ / MDD ↓)
SCORE_W_RETURN = 0.5
SCORE_W_SHARPE = 0.4
SCORE_W_MDD    = 0.1


# --- Core: run one backtest with env overrides -------------------------------
def _run_one(run_id: str, env_overrides: dict) -> dict:
    """
    Launch one backtest with the given (ENV) config overrides.
    Returns a dict with run metadata + key metrics parsed from backtest_summary.xlsx.
    """
    env = os.environ.copy()
    env["BACKTEST_RUN_ID"] = run_id
    env["PYTHONIOENCODING"] = "utf-8"

    # Apply each override (strings expected by config)
    for k, v in env_overrides.items():
        env[k] = str(v)

    # Ensure run subdir is enabled (automation expects subfolders)
    env.setdefault("BACKTEST_USE_RUN_SUBDIR", "True")

    # Launch
    print(f"\n=== RUN {run_id} ===")
    print("Overrides:", json.dumps(env_overrides, indent=2))
    t0 = time.time()
    proc = None
    try:
        proc = __import__("subprocess").run(
            [PY, str(DBACKTEST)], cwd=str(ROOT), env=env, capture_output=True, text=True
        )
    except Exception as e:
        return {"run_id": run_id, "status": "error", "stderr": f"launch failed: {e}"}
    dur_s = time.time() - t0

    if proc.returncode != 0:
        # Surface child output for quick diagnosis
        print(proc.stdout)
        print(proc.stderr)
        return {
            "run_id": run_id, "status": "error",
            "returncode": proc.returncode, "stderr": proc.stderr[:2000]
        }

    # Locate run dir (we forced the name) & parse summary
    run_dir = BACKTEST_DIR / run_id
    snap = run_dir / "run_config_snapshot.json"
    summary_xlsx = run_dir / "backtest_summary.xlsx"

    row = {"run_id": run_id, "status": "ok", "duration_s": round(dur_s, 2)}
    if summary_xlsx.exists():
        df = pd.read_excel(summary_xlsx)
        # Coerce numeric columns if present
        for c in ["Model Return (%)", "Sharpe", "Max Drawdown (%)", "Trades", "Final Capital"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Use Ensemble rows if present
        if "Prediction Type" in df.columns:
            ens = df[df["Prediction Type"].str.contains("Ensemble", case=False, na=False)].copy()
        else:
            ens = df.copy()

        row["tickers"]        = int(ens["Ticker"].nunique()) if "Ticker" in ens.columns else len(ens)
        row["trades_total"]   = int(ens.get("Trades", pd.Series(dtype=float)).fillna(0).sum())
        row["avg_return_pct"] = float(ens.get("Model Return (%)", pd.Series(dtype=float)).mean() or 0.0)
        row["median_sharpe"]  = float(ens.get("Sharpe", pd.Series(dtype=float)).median() or 0.0)
        row["mean_mdd_pct"]   = float(ens.get("Max Drawdown (%)", pd.Series(dtype=float)).mean() or 0.0)
    else:
        row.update({"status": "no_summary"})

    # Snapshot path for repro
    row["snapshot_path"] = str(snap) if snap.exists() else ""

    # Echo key overrides into the row so CSV is self-contained (only if present)
    important = (
        "ENSEMBLE_LONG_THRESHOLD","ENSEMBLE_SHORT_THRESHOLD",
        "MIN_CONF_FLOOR","ENSEMBLE_TEMP",
        "atr_tp_multiplier","atr_sl_multiplier","TRAIL_ATR_MULT",
        "RISK_PER_TRADE","DAILY_TOP_K",
        "EV_MAX_WEIGHT_PER_NAME","PER_NAME_CAP","PORTFOLIO_TARGET_NET",
        "MIN_PRICE","MIN_AVG_DOLLAR_VOL","MIN_ATR_DOLLARS",
        "MAX_AVG_PAIRWISE_CORR","MAX_PER_SECTOR",
        "DISAGREE_STD_CUTOFF","DISAGREE_CONF_LIFT",
        "BACKTEST_MODE","BACKTEST_BASE","BACKTEST_STACKED","LIMIT_TICKERS"
    )
    for k in important:
        if k in env_overrides:
            row[k] = env_overrides[k]

    print(f"→ {row}")
    return row


def _append_summary(rows):
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if AUTO_SUMMARY.exists():
        old = pd.read_csv(AUTO_SUMMARY)
        out = pd.concat([old, df_new], ignore_index=True)
    else:
        out = df_new
    out.to_csv(AUTO_SUMMARY, index=False)
    print(f"[AUTO] Updated {AUTO_SUMMARY}")


def _print_and_save_leaderboard(min_trades=DEFAULT_MIN_TRADES_ROW, top_n=12):
    if not AUTO_SUMMARY.exists():
        print("(no auto_runs_summary.csv yet)")
        return
    df = pd.read_csv(AUTO_SUMMARY)
    req = {"avg_return_pct", "median_sharpe", "mean_mdd_pct", "trades_total"}
    if not req.issubset(df.columns):
        print("\n(Insufficient columns to build leaderboard)")
        return

    clean = (df.dropna(subset=list(req)).query(f"trades_total >= {min_trades}").copy())
    if clean.empty:
        print("\n(No qualifying runs with trades_total >= {min_trades})")
        return

    clean["score"] = (
        SCORE_W_RETURN * clean["avg_return_pct"] +
        SCORE_W_SHARPE * clean["median_sharpe"] -
        SCORE_W_MDD    * clean["mean_mdd_pct"]
    )

    top = clean.sort_values(
        ["score", "avg_return_pct", "median_sharpe"],
        ascending=[False, False, False]
    ).head(top_n)

    cols = [
        "run_id",
        "avg_return_pct","median_sharpe","mean_mdd_pct","trades_total",
        "ENSEMBLE_LONG_THRESHOLD","ENSEMBLE_SHORT_THRESHOLD",
        "MIN_CONF_FLOOR","ENSEMBLE_TEMP",
        "atr_tp_multiplier","atr_sl_multiplier","TRAIL_ATR_MULT",
        "RISK_PER_TRADE","DAILY_TOP_K",
        "snapshot_path",
    ]
    cols = [c for c in cols if c in top.columns]

    print("\n=== Leaderboard (score = 0.5*Return + 0.4*Sharpe - 0.1*MDD) ===")
    print(top[cols].to_string(index=False))

    top.to_csv(AUTO_LEADER, index=False)
    print(f"[AUTO] Leaderboard saved → {AUTO_LEADER}")


def _run_plan(plan, label: str):
    print(f"\n▶ Starting plan: {label} (total combos: {len(plan)})")
    runs = []
    for i, p in enumerate(plan):
        run_id = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{i:03d}"
        row = _run_one(run_id, p)
        runs.append(row)
        time.sleep(SLEEP_BETWEEN_RUNS_SEC)
    _append_summary(runs)
    _print_and_save_leaderboard()


# --- Plans / Menus ------------------------------------------------------------

def plan_tight_high_yield():
    """
    Tight/high‑yield sweep (best performer set).
    Focused around your proven good zone; fast/clean for weekly use or after regime changes.
    """
    longs_shorts   = [(0.57, 0.56), (0.58, 0.57)]   # ensemble thresholds
    tps            = [2.0, 2.2]
    sls            = [0.6, 0.7]
    trails         = [1.1, 1.3]
    conf_floors    = [0.54, 0.56, 0.58]
    temps          = [0.9, 1.0]
    risks          = [0.010, 0.015, 0.018]
    topk           = [6, 8, 10]
    plan = []
    for (lt, st), tp, sl, trail, floor, temp, rpt, k in product(
        longs_shorts, tps, sls, trails, conf_floors, temps, risks, topk
    ):
        plan.append({
            "ENSEMBLE_LONG_THRESHOLD": lt,
            "ENSEMBLE_SHORT_THRESHOLD": st,
            "MIN_CONF_FLOOR": floor,
            "ENSEMBLE_TEMP": temp,
            "atr_tp_multiplier": tp,
            "atr_sl_multiplier": sl,
            "TRAIL_ATR_MULT": trail,
            "RISK_PER_TRADE": rpt,
            "DAILY_TOP_K": k,
            "LIMIT_TICKERS": 30,
            "BACKTEST_MODE": "ensemble_only",
            "BACKTEST_BASE": "False",
            "BACKTEST_STACKED": "False",
        })
    return plan


def plan_quick_sweep(max_combos=DEFAULT_MAX_COMBOS_QUICK, seed=42):
    """
    Quick sweep (wide sanity check).
    Samples a broader config grid to catch regressions or new sweet spots. Monthly or after feature/model changes.
    """
    # Pull grid from config if present; otherwise fall back to a modest default
    try:
        import config as CFG
        grid = getattr(CFG, "backtest_optimize_param_grid", None)
    except Exception:
        grid = None

    if not grid:
        grid = {
            "ENSEMBLE_LONG_THRESHOLD":  [0.56, 0.58, 0.60],
            "ENSEMBLE_SHORT_THRESHOLD": [0.56, 0.58, 0.60],
            "MIN_CONF_FLOOR":           [0.54, 0.58, 0.60],
            "ENSEMBLE_TEMP":            [0.90, 1.00],
            "atr_tp_multiplier":        [2.0, 2.2, 2.5],
            "atr_sl_multiplier":        [0.6, 0.8, 1.0],
            "TRAIL_ATR_MULT":           [1.0, 1.2],
            "RISK_PER_TRADE":           [0.010, 0.015],
        }

    # Random sample across the Cartesian product without exploding
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    # Build all combos lazily; sample indices
    sizes = [len(v) for v in values]
    total = 1
    for s in sizes: total *= s

    rnd = random.Random(seed)
    want = min(max_combos, total)
    picks = set()
    while len(picks) < want:
        idx = tuple(rnd.randrange(n) for n in sizes)
        picks.add(idx)

    plan = []
    for idx in picks:
        d = {k: values[i][idx[i]] for i, k in enumerate(keys)}
        d.update({
            "LIMIT_TICKERS": 30,
            "BACKTEST_MODE": "ensemble_only",
            "BACKTEST_BASE": "False",
            "BACKTEST_STACKED": "False",
        })
        plan.append(d)
    return plan


def plan_sharpe_tuner_local():
    """
    Local Sharpe tuner (small halo around current config).
    Weekly or when drift logs highlight degradation; keeps trade counts in a healthy range.
    """
    try:
        import config as CFG
        base_lt  = float(getattr(CFG, "ENSEMBLE_LONG_THRESHOLD", 0.57))
        base_st  = float(getattr(CFG, "ENSEMBLE_SHORT_THRESHOLD", 0.57))
        base_tp  = float(getattr(CFG, "atr_tp_multiplier", 2.0))
        base_sl  = float(getattr(CFG, "atr_sl_multiplier", 0.7))
        base_trl = float(getattr(CFG, "TRAIL_ATR_MULT", 1.2))
        base_rpt = float(getattr(CFG, "RISK_PER_TRADE", 0.015))
        base_top = int(getattr(CFG, "DAILY_TOP_K", 8))
    except Exception:
        base_lt, base_st, base_tp, base_sl, base_trl, base_rpt, base_top = 0.57, 0.57, 2.0, 0.7, 1.2, 0.015, 8

    lts = [round(base_lt + d, 3) for d in (-0.01, 0.0, +0.01)]
    sts = [round(base_st + d, 3) for d in (-0.01, 0.0, +0.01)]
    tps = [round(base_tp + d, 2) for d in (-0.2, 0.0, +0.2)]
    sls = [0.6, 0.7, 0.8]
    trl = [max(0.8, round(base_trl + d, 2)) for d in (-0.2, 0.0, +0.2)]
    rpt = sorted(set([0.012, base_rpt, 0.015, 0.018]))
    topk= sorted(set([max(5, base_top-2), base_top, base_top+2]))

    plan = []
    for lt, st, tp, sl, trail, risk, k in product(lts, sts, tps, sls, trl, rpt, topk):
        plan.append({
            "ENSEMBLE_LONG_THRESHOLD": lt,
            "ENSEMBLE_SHORT_THRESHOLD": st,
            "MIN_CONF_FLOOR": 0.54,            # gentle floor
            "ENSEMBLE_TEMP":  1.00,            # neutral temperature
            "atr_tp_multiplier": tp,
            "atr_sl_multiplier": sl,
            "TRAIL_ATR_MULT": trail,
            "RISK_PER_TRADE": risk,
            "DAILY_TOP_K": k,
            "LIMIT_TICKERS": 30,
            "BACKTEST_MODE": "ensemble_only",
            "BACKTEST_BASE": "False",
            "BACKTEST_STACKED": "False",
        })
    return plan


def plan_portfolio_risk_concentration():
    """
    Portfolio risk / concentration sweep.
    Monthly or after meaningful drawdowns; checks robustness of sizing/diversity constraints.
    """
    risks   = [0.010, 0.015, 0.018]
    topk    = [6, 8, 10]
    wcap    = [0.20, 0.25, 0.30]   # EV_MAX_WEIGHT_PER_NAME
    pcap    = [0.12, 0.15, 0.20]   # PER_NAME_CAP
    pnet    = [0.10, 0.20, 0.30]   # PORTFOLIO_TARGET_NET
    corrmax = [0.55, 0.65, 0.75]   # MAX_AVG_PAIRWISE_CORR
    persec  = [2, 3, 4]            # MAX_PER_SECTOR

    # Keep exit shape in the known good zone
    lt_st   = [(0.57, 0.56), (0.58, 0.57)]
    tp_sl   = [(2.0, 0.6), (2.2, 0.6), (2.0, 0.7)]
    trail   = [1.1, 1.3]

    plan = []
    for (lt, st), (tp, sl), tr, r, k, w, pc, pn, cm, ms in product(
        lt_st, tp_sl, trail, risks, topk, wcap, pcap, pnet, corrmax, persec
    ):
        plan.append({
            "ENSEMBLE_LONG_THRESHOLD": lt,
            "ENSEMBLE_SHORT_THRESHOLD": st,
            "MIN_CONF_FLOOR": 0.56,
            "ENSEMBLE_TEMP":  1.00,
            "atr_tp_multiplier": tp,
            "atr_sl_multiplier": sl,
            "TRAIL_ATR_MULT": tr,
            "RISK_PER_TRADE": r,
            "DAILY_TOP_K": k,
            "EV_MAX_WEIGHT_PER_NAME": w,
            "PER_NAME_CAP": pc,
            "PORTFOLIO_TARGET_NET": pn,
            "MAX_AVG_PAIRWISE_CORR": cm,
            "MAX_PER_SECTOR": ms,
            "LIMIT_TICKERS": 30,
            "BACKTEST_MODE": "ensemble_only",
            "BACKTEST_BASE": "False",
            "BACKTEST_STACKED": "False",
        })
    return plan


def run_best_from_leaderboard():
    """
    Re-run best from leaderboard snapshot (fast).
    Daily (or before live) to confirm stability on latest data slice.
    """
    if not AUTO_LEADER.exists():
        print(f"No {AUTO_LEADER} found. Run a sweep first.")
        return
    df = pd.read_csv(AUTO_LEADER)
    if df.empty:
        print("Leaderboard is empty.")
        return

    best = df.iloc[0].to_dict()
    overrides = {}
    # Prefer to read the stored keys directly (they’re already flattened)
    keys = [
        "ENSEMBLE_LONG_THRESHOLD","ENSEMBLE_SHORT_THRESHOLD",
        "MIN_CONF_FLOOR","ENSEMBLE_TEMP",
        "atr_tp_multiplier","atr_sl_multiplier","TRAIL_ATR_MULT",
        "RISK_PER_TRADE","DAILY_TOP_K",
        "EV_MAX_WEIGHT_PER_NAME","PER_NAME_CAP","PORTFOLIO_TARGET_NET",
        "MIN_PRICE","MIN_AVG_DOLLAR_VOL","MIN_ATR_DOLLARS",
        "MAX_AVG_PAIRWISE_CORR","MAX_PER_SECTOR",
        "DISAGREE_STD_CUTOFF","DISAGREE_CONF_LIFT",
        "LIMIT_TICKERS","BACKTEST_MODE","BACKTEST_BASE","BACKTEST_STACKED",
    ]
    for k in keys:
        if k in best and not (isinstance(best[k], float) and math.isnan(best[k])):
            overrides[k] = best[k]

    # Fallback to snapshot JSON for completeness (if missing keys)
    snap = best.get("snapshot_path") or ""
    if isinstance(snap, str) and snap and Path(snap).exists():
        try:
            with open(snap, "r", encoding="utf-8") as f:
                snap_cfg = json.load(f)
            for k, v in snap_cfg.items():
                overrides.setdefault(k, v)
        except Exception as e:
            print(f"(snapshot read warning: {e})")

    run_id = f"rerun_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    row = _run_one(run_id, overrides)
    _append_summary([row])
    _print_and_save_leaderboard()


def manual_one_off():
    """
    One‑off manual overrides.
    Ad‑hoc checks; run on demand. Enter KEY=val pairs separated by commas.
    """
    print("\nEnter overrides as comma‑separated KEY=val pairs (e.g., ENSEMBLE_LONG_THRESHOLD=0.58, atr_tp_multiplier=2.2)")
    line = input("Overrides> ").strip()
    if not line:
        print("No overrides entered.")
        return

    def _coerce(v: str):
        v = v.strip()
        if v.lower() in ("true", "false"):
            return v  # keep as string so config’s bool caster can handle it
        try:
            if "." in v:
                return float(v)
            return int(v)
        except Exception:
            return v

    overrides = {}
    for part in line.split(","):
        if not part.strip(): continue
        if "=" not in part:
            print(f"Skipping malformed token: {part}")
            continue
        k, v = part.split("=", 1)
        overrides[k.strip()] = _coerce(v)

    run_id = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    row = _run_one(run_id, overrides)
    _append_summary([row])
    _print_and_save_leaderboard()


# --- Menu / entrypoint --------------------------------------------------------

MENU = [
    ("1", "Tight/high‑yield sweep",
     "Focused combos around your best zone; run weekly or after regime changes.",
     plan_tight_high_yield),
    ("2", "Quick sweep",
     f"Broad sanity check (random‑sampled grid ~{DEFAULT_MAX_COMBOS_QUICK}); run monthly or after feature/model changes.",
     lambda: plan_quick_sweep(DEFAULT_MAX_COMBOS_QUICK)),
    ("3", "Local Sharpe tuner",
     "Small halo around current config for Sharpe/return balance; run weekly or on drift warnings.",
     plan_sharpe_tuner_local),
    ("4", "Portfolio risk / concentration sweep",
     "Stress‑test sizing/diversity caps; run monthly or after drawdowns.",
     plan_portfolio_risk_concentration),
    ("5", "Re‑run best from leaderboard (fast)",
     "Daily or pre‑live confirmation using prior best snapshot.",
     None),
    ("6", "One‑off manual overrides",
     "Ad‑hoc single run for a specific hypothesis.",
     None),
    ("7", "Exit",
     "Quit the launcher.",
     None),
]

def print_menu():
    print("\n================ dBacktest_refine — Menu ================\n")
    for key, title, expl, _maker in MENU:
        print(f"{key}) {title}: {expl}")
    print("\n=========================================================\n")

def main():
    # Allow non‑interactive selection: REFINE_MODE or first CLI arg
    mode = os.environ.get("REFINE_MODE") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not mode:
        print_menu()
        mode = input("Select an option [1-7] (default=1): ").strip() or "1"

    if mode == "1":
        plan = plan_tight_high_yield()
        _run_plan(plan, "Tight/high‑yield sweep")
    elif mode == "2":
        plan = plan_quick_sweep(DEFAULT_MAX_COMBOS_QUICK)
        _run_plan(plan, "Quick sweep")
    elif mode == "3":
        plan = plan_sharpe_tuner_local()
        _run_plan(plan, "Local Sharpe tuner")
    elif mode == "4":
        plan = plan_portfolio_risk_concentration()
        _run_plan(plan, "Portfolio risk / concentration sweep")
    elif mode == "5":
        run_best_from_leaderboard()
    elif mode == "6":
        manual_one_off()
    elif mode == "7":
        print("Bye.")
    else:
        print(f"Unknown option '{mode}'.")
        print_menu()

if __name__ == "__main__":
    main()
