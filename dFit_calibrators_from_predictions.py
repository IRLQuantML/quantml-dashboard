import os, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from joblib import dump

# ---- PATHS (edit if needed) ----
PREDICTIONS_ROOT = r"D:\QuantML\VS_QuantMLV6\6.Predictions"  # where past runs live
CALIBRATORS_DIR  = r"D:\QuantML\VS_QuantMLV6\4.Metrics\calibrators"
os.makedirs(CALIBRATORS_DIR, exist_ok=True)

# Families and their column prefixes in your pipeline
FAM_PREFIXES = {
    "xgboost": "xgb",
    "lightgbm": "lgbm",
    "catboost": "cat",
    "randomforest": "rf",
    "logisticregression": "lr",
}

# LONG=0, SHORT=1 — we calibrate P(class_1) and reconstruct P(class_0)=1-P1
CLASS_IS_LONG = 0  # keep consistent with your log

# ---------- Helpers ----------
def _canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    # minimal canonicalization for family prob columns
    # (your ePredict has a richer version; here we just ensure presence)
    return df

def _find_runs(root):
    for run in sorted(Path(root).glob("run_*")):
        f = run / "All_predictions.csv"
        if f.exists():
            yield f

def _extract_label(df: pd.DataFrame) -> pd.Series:
    """
    Try to get the true label per row for calibration.
    Priority:
      1) existing 'label'/'y' column (0/1)
      2) target/next_day_return>0 → LONG=0 else SHORT=1
    """
    for c in ["label", "y", "target", "Target", "LABEL"]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.dropna().isin([0,1]).any():
                return s.astype("Int64")

    # Fallback: compute simple next-day sign from adj_close/close if present
    price_col = next((c for c in ["adj_close","Adj Close","close","Close","Price"] if c in df.columns), None)
    date_col = next((c for c in ["date","Date"] if c in df.columns), None)
    if price_col and date_col:
        z = df[[date_col, price_col]].copy()
        z[date_col] = pd.to_datetime(z[date_col], errors="coerce")
        z.sort_values(date_col, inplace=True)
        ret = z[price_col].pct_change(-1)  # next-period change aligned to current row
        lab = (ret <= 0).astype(int)  # up => LONG(0), down/flat => SHORT(1)
        lab = lab.reindex(df.index).astype("Int64")
        return lab

    return pd.Series(pd.NA, index=df.index, dtype="Int64")

def fit_iso(x, y):
    # x: raw prob (class_1); y: true label in {0,1}
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    m.fit(x, y)
    return m

# ---------- Main ----------
def main():
    # Collect (raw_prob_class_1, true_label) per ticker+family
    pairs = {}  # (ticker, fam_prefix) -> list of (p1, y)

    for all_preds in _find_runs(PREDICTIONS_ROOT):
        df = pd.read_csv(all_preds)
        if df.empty: 
            continue
        df = _canonicalize(df)
        # label
        y = _extract_label(df)
        if y.isna().all():
            continue

        # ensure Ticker/date
        tcol = "Ticker" if "Ticker" in df.columns else ("ticker" if "ticker" in df.columns else None)
        if not tcol:
            continue

        for fam_long, pfx in FAM_PREFIXES.items():
            c1 = f"{pfx}_prob_class_1"
            if c1 not in df.columns:
                continue
            sub = df[[tcol, c1]].copy()
            sub["y"] = y
            sub = sub.dropna()
            if sub.empty:
                continue
            for tk, g in sub.groupby(tcol):
                k = (str(tk).upper(), pfx)
                if k not in pairs: pairs[k] = []
                pairs[k].extend(list(zip(g[c1].tolist(), g["y"].tolist())))

    # Fit and save per (ticker,family)
    saved = 0
    for (tk, pfx), lst in pairs.items():
        if len(lst) < 100:  # need a bit of data to fit smoothly
            continue
        arr = np.asarray(lst, dtype=float)
        x = arr[:,0]  # raw P(class_1)
        y = arr[:,1]  # true label (0/1)
        try:
            iso = fit_iso(x, y)
        except Exception:
            continue
        out_dir = Path(CALIBRATORS_DIR) / tk
        out_dir.mkdir(parents=True, exist_ok=True)
        dump(iso, out_dir / f"{pfx}_isotonic.pkl")
        saved += 1

    print(f"✅ Saved {saved} calibrators into {CALIBRATORS_DIR}")

if __name__ == "__main__":
    main()
