import os, sys, json, time, subprocess, shutil
from pathlib import Path
import pandas as pd
from datetime import datetime
from itertools import product

PY = sys.executable
ROOT = Path(__file__).resolve().parent

# Paths used in your repo
DBACKTEST = ROOT / "dBacktest.py"
BACKTEST_DIR = ROOT / "5.Backtest"
PRED_DIR = ROOT / "6.Predictions"

# Where to store the automation summary (now under 5.Backtest/)
AUTO_SUMMARY = BACKTEST_DIR / "auto_runs_summary.csv"


# ---- Helpers ----------------------------------------------------------------
def run_one(run_id: str, env_overrides: dict) -> dict:
    """
    Launch one backtest with the given (ENV) config overrides.
    Returns a dict with run metadata + key metrics parsed from backtest_summary.xlsx.
    """
    # Ensure UTF-8 so unicode banners never crash on Windows pipes
    env = os.environ.copy()
    env["BACKTEST_RUN_ID"] = run_id
    env["PYTHONIOENCODING"] = "utf-8"

    # Apply each override (strings expected)
    for k, v in env_overrides.items():
        env[k] = str(v)

    # Ensure run subdir is enabled (automation expects subfolders)
    env.setdefault("BACKTEST_USE_RUN_SUBDIR", "True")

    # Launch
    print(f"\n=== RUN {run_id} ===")
    print("Overrides:", json.dumps(env_overrides, indent=2))
    t0 = time.time()
    proc = subprocess.run([PY, str(DBACKTEST)], cwd=str(ROOT), env=env, capture_output=True, text=True)
    dur_s = time.time() - t0

    # Show errors if any
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        return {"run_id": run_id, "status": "error", "returncode": proc.returncode, "stderr": proc.stderr[:2000]}

    # Locate run dir (we forced the name)
    run_dir = BACKTEST_DIR / run_id
    snap = run_dir / "run_config_snapshot.json"
    summary_xlsx = run_dir / "backtest_summary.xlsx"

    # Parse summary
    row = {"run_id": run_id, "status": "ok", "duration_s": round(dur_s, 2)}
    if summary_xlsx.exists():
        df = pd.read_excel(summary_xlsx)
        # Coerce numeric columns if present
        for c in ["Model Return (%)", "Sharpe", "Max Drawdown (%)", "Trades", "Final Capital"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Compute aggregates (focus on ensemble rows if present)
        if "Prediction Type" in df.columns:
            ens = df[df["Prediction Type"].str.contains("Ensemble", case=False, na=False)].copy()
        else:
            ens = df.copy()

        row["tickers"] = int(ens["Ticker"].nunique()) if "Ticker" in ens.columns else len(ens)
        row["trades_total"] = int(ens.get("Trades", pd.Series(dtype=float)).fillna(0).sum())
        row["avg_return_pct"] = float(ens.get("Model Return (%)", pd.Series(dtype=float)).mean() or 0.0)
        row["median_sharpe"] = float(ens.get("Sharpe", pd.Series(dtype=float)).median() or 0.0)
        row["mean_mdd_pct"] = float(ens.get("Max Drawdown (%)", pd.Series(dtype=float)).mean() or 0.0)
    else:
        row.update({"status": "no_summary"})

    # Attach snapshot + echo key overrides into the row so CSV is self-contained
    row["snapshot_path"] = str(snap) if snap.exists() else ""
    for k in (
        "ENSEMBLE_LONG_THRESHOLD","ENSEMBLE_SHORT_THRESHOLD",
        "MIN_CONF_FLOOR","ENSEMBLE_TEMP",
        "atr_tp_multiplier","atr_sl_multiplier","TRAIL_ATR_MULT",
        "RISK_PER_TRADE","DAILY_TOP_K",
        "EV_MAX_WEIGHT_PER_NAME","PER_NAME_CAP","PORTFOLIO_TARGET_NET",
        "MIN_PRICE","MIN_AVG_DOLLAR_VOL","MIN_ATR_DOLLARS",
        "MAX_AVG_PAIRWISE_CORR","MAX_PER_SECTOR",
        "DISAGREE_STD_CUTOFF","DISAGREE_CONF_LIFT",
        "BACKTEST_MODE","BACKTEST_BASE","BACKTEST_STACKED","LIMIT_TICKERS"
    ):
        if k in env_overrides:
            row[k] = env_overrides[k]

    print(f"→ {row}")
    return row


def append_summary(rows):
    if not rows:
        return
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    if AUTO_SUMMARY.exists():
        old = pd.read_csv(AUTO_SUMMARY)
        out = pd.concat([old, df_new], ignore_index=True)
    else:
        out = df_new
    out.to_csv(AUTO_SUMMARY, index=False)
    print(f"[AUTO] Updated {AUTO_SUMMARY}")


# ---- Search strategies -------------------------------------------------------
def grid_params():
    # Suggested next sweep (tight, high-yield)
    longs_shorts   = [(0.57, 0.56), (0.58, 0.57)]  # (ENSEMBLE_LONG_THRESHOLD, ENSEMBLE_SHORT_THRESHOLD)
    tps            = [2.0, 2.2]                     # atr_tp_multiplier
    sls            = [0.6, 0.7]                     # atr_sl_multiplier
    trails         = [1.1, 1.3]                     # TRAIL_ATR_MULT
    conf_floors    = [0.54, 0.56, 0.58]             # MIN_CONF_FLOOR
    temps          = [0.9, 1.0]                     # ENSEMBLE_TEMP
    risk_per_trade = [0.010, 0.015, 0.018]          # RISK_PER_TRADE
    daily_top_k    = [6, 8, 10]                      # DAILY_TOP_K

    for (lt, st) in longs_shorts:
        for tp in tps:
            for sl in sls:
                for trail in trails:
                    for floor in conf_floors:
                        for temp in temps:
                            for rpt in risk_per_trade:
                                for topk in daily_top_k:
                                    yield {
                                        "ENSEMBLE_LONG_THRESHOLD": lt,
                                        "ENSEMBLE_SHORT_THRESHOLD": st,
                                        "MIN_CONF_FLOOR": floor,
                                        "ENSEMBLE_TEMP": temp,
                                        "atr_tp_multiplier": tp,
                                        "atr_sl_multiplier": sl,
                                        "TRAIL_ATR_MULT": trail,
                                        "RISK_PER_TRADE": rpt,
                                        "DAILY_TOP_K": topk,
                                        # constants
                                        "LIMIT_TICKERS": 5,
                                        "BACKTEST_MODE": "ensemble_only",
                                        "BACKTEST_BASE": "False",
                                        "BACKTEST_STACKED": "False",
                                    }


def bayes_suggestions(n=12):
    import random
    for _ in range(n):
        yield {
            # classifier / confidence
            "ENSEMBLE_LONG_THRESHOLD": round(random.uniform(0.54, 0.60), 3),
            "ENSEMBLE_SHORT_THRESHOLD": round(random.uniform(0.54, 0.60), 3),
            "MIN_CONF_FLOOR":          round(random.uniform(0.52, 0.60), 3),
            "ENSEMBLE_TEMP":           round(random.uniform(0.80, 1.10), 2),

            # exits / trailing
            "atr_tp_multiplier":       round(random.uniform(1.8, 2.4), 2),
            "atr_sl_multiplier":       round(random.uniform(0.6, 0.75), 2),  # avoid the 0-trade choke at 0.9
            "TRAIL_ATR_MULT":          round(random.uniform(1.0, 1.5), 2),

            # risk & concentration
            "RISK_PER_TRADE":          round(random.uniform(0.008, 0.020), 4),
            "DAILY_TOP_K":             random.choice([5, 6, 7, 8, 10]),
            "EV_MAX_WEIGHT_PER_NAME":  random.choice([0.20, 0.25, 0.30]),
            "PER_NAME_CAP":            random.choice([0.12, 0.15, 0.20]),
            "PORTFOLIO_TARGET_NET":    random.choice([0.10, 0.20, 0.25, 0.30]),

            # quality / liquidity gates
            "MIN_PRICE":               random.choice([0.0, 2.0, 5.0, 10.0]),
            "MIN_AVG_DOLLAR_VOL":      random.choice([0, 1_000_000, 2_000_000]),
            "MIN_ATR_DOLLARS":         random.choice([0.0, 0.15, 0.25]),

            # correlation / sector constraints
            "MAX_AVG_PAIRWISE_CORR":   random.choice([0.55, 0.65, 0.75]),
            "MAX_PER_SECTOR":          random.choice([2, 3, 4]),

            # disagreement guard
            "DISAGREE_STD_CUTOFF":     random.choice([0.20, 0.25, 0.30]),
            "DISAGREE_CONF_LIFT":      random.choice([0.03, 0.04, 0.05]),

            # constants for this sweep
            "LIMIT_TICKERS":           5,
            "BACKTEST_MODE":           "ensemble_only",
            "BACKTEST_BASE":           "False",
            "BACKTEST_STACKED":        "False",
        }


# ---- Main driver -------------------------------------------------------------
if __name__ == "__main__":
    runs = []

    # Materialize plan so we can compute indices and print stable run_ids
    plan = list(grid_params()) + list(bayes_suggestions(10))

    for i, p in enumerate(plan):
        run_id = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{i:03d}"
        rows = run_one(run_id, p)
        runs.append(rows)
        time.sleep(0.3)  # tiny stagger to avoid same-second run_id collisions

    append_summary(runs)

    # Leaderboard: filter noise and compute composite score
    df = pd.read_csv(AUTO_SUMMARY)
    req = {"avg_return_pct", "median_sharpe", "mean_mdd_pct", "trades_total"}
    if req.issubset(df.columns):
        # 1) Filter first: valid metrics + enough trading activity
        clean = (
            df.dropna(subset=list(req))
              .query("trades_total >= 6")  # tighten to taste (e.g., 8–10 for stability)
              .copy()
        )

        if not clean.empty:
            # 2) Composite score (Return ↑, Sharpe ↑, MDD ↓)
            clean["score"] = (
                0.5 * clean["avg_return_pct"]
                + 0.4 * clean["median_sharpe"]
                - 0.1 * clean["mean_mdd_pct"]
            )

            # 3) Rank and select top rows
            top = clean.sort_values(
                ["score", "avg_return_pct", "median_sharpe"],
                ascending=[False, False, False]
            ).head(12)

            # 4) Columns to show (only those present will be printed)
            cols = [
                "run_id",
                "avg_return_pct", "median_sharpe", "mean_mdd_pct", "trades_total",
                "ENSEMBLE_LONG_THRESHOLD", "ENSEMBLE_SHORT_THRESHOLD",
                "MIN_CONF_FLOOR", "ENSEMBLE_TEMP",
                "atr_tp_multiplier", "atr_sl_multiplier", "TRAIL_ATR_MULT",
                "RISK_PER_TRADE", "DAILY_TOP_K",
                "snapshot_path",
            ]
            cols = [c for c in cols if c in top.columns]

            # 5) Print nicely
            print("\n=== Leaderboard (score = 0.5*Return + 0.4*Sharpe - 0.1*MDD) ===")
            print(top[cols].to_string(index=False))

            # 6) Save leaderboard next to summaries
            lb_path = BACKTEST_DIR / "auto_leaderboard.csv"
            top.to_csv(lb_path, index=False)
            print(f"[AUTO] Leaderboard saved → {lb_path}")
        else:
            print("\n(No qualifying runs with trades_total >= 6)")
    else:
        print("\n(Insufficient columns to build leaderboard)")
