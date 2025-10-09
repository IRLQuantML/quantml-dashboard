import os
import datetime as dt
from datetime import datetime
from pandas.tseries.offsets import BDay
import re
from typing import Dict, List
# Pandas copy-on-write is already set in dBacktest, keep here for safety if any code imports config first.
import pandas as _pd
try:
    _pd.options.mode.copy_on_write = True
except Exception:
    pass

# === Alpaca config ===
ALPACA_API_KEY   = 'PK76N760G44THHX73Z7G'     # consider env vars in production
ALPACA_SECRET_KEY = 'IGMBbTL79AWv6GJj1kLdmaDVH5Mgi2nW5K1q6RWs'
ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL  = "https://api.alpaca.markets"
USE_LIVE_TRADING = False   # True = live, False = paper
ALPACA_BASE_URL = ALPACA_LIVE_URL if USE_LIVE_TRADING else ALPACA_PAPER_URL

# Prediction
ALLOCATOR_KIND = "advanced"          # use the new branch
TARGET_SHORT_SLEEVE = 0.35          # optional (defaults shown above)
PER_NAME_CAP = 0.15

# ===== Calibration & temperature =====
USE_CALIBRATION = True                   # turn on if you have saved calibrators
CALIBRATION_METHOD = "isotonic"          # "isotonic" | "platt"

# Side-aware selection
ALLOW_SIDE_QUOTAS = True            # turn on side quotas in daily summary
SIDE_QUOTA_LONG   = None            # leave None to compute from target_net
SIDE_QUOTA_SHORT  = None            # leave None to compute from target_net

# ===== Prior shift (prevalence) correction =====
PRIOR_SHIFT_ENABLE = True
PRIOR_SHIFT_MIN_ABS_DELTA = 0.07         # only correct if |pi_test - pi_train| >= this
PRIOR_SHIFT_MIN_OBS = 100                # need this many recent scores to trust pi_test

# ===== Prefer stacked only when truly better =====
STACK_VS_BASE_MAP = "metrics/stack_vs_base.json"  # { "TICKER": { "xgboost":"stacked"/"base", ... } }

# ===== Disagreement guard for Top-K =====
DISAGREE_STD_CUTOFF = 0.20               # std-dev of family P(LONG) across models
DISAGREE_CONF_LIFT = 0.03                # require this extra confidence when high disagreement

# ===== Sector cap (already supported by allocate_portfolio_advanced) =====
# Set these to enable sector caps in advanced allocator
DIVERSITY_CLUSTER_COL = "Sector"         # make sure your features/preds include this column
DIVERSITY_MAX_PER_CLUSTER = 3

# Capacity for research
top_signals_limit         = 300
min_pass                  = 1

dry_run          = False    # set to False to enable live Alpaca trading

transaction_cost            = 0.001  # 0.1% per trade
ATR_BE_LOCK                 = 1.0    # lock stop to breakeven once gain >= this * ATR (unused if trailing off)
MAX_HOLD_DAYS               = 30     # time-based exit safety (only used in the core variant)
USE_LIVE_PRICE_ONLY         = True   # force live price for sizing and brackets
DRIFT_LOG_THRESHOLD         = 0.01   # 1% drift (only used for logging)
APPLY_PRIOR_SHIFT_AT_PREDICT = False
QML_LOG_LEVEL = "WARNING"   # Keep log noise down by default (can override via env QML_LOG_LEVEL)
WRITE_DEBUG_CSVS = False    # IMPORTANT: dBacktest falls back to True if this is absent. Set it explicitly.
PORTFOLIO_BUDGET            = 20000.0
DAILY_BUDGET                = 20000.0
MIN_DOLLARS_PER_TRADE       = 25.0
VERY_STRONG_CONF_FLOOR      = 0.20   # starting floor aimed by the sweep (fixes crash path)
CONF_FLOOR_STEP             = 0.02
SELECTION_WINNERS_K         = 50
SELECTION_PERFORMERS_K      = 50
SELECTION_POOL_CAP          = 100
USE_PREFERRED_MODEL_FOR_DIRECTION = True
MAJORITY_SCORE_LONG         = 0.50
MAJORITY_SCORE_SHORT        = 0.50
ALWAYS_RUN_ALL_FAMILIES     = False  # Always emit full prediction blocks for all 5 families (base + stacked)
PORTFOLIO_TARGET_NET        = 0.0 # target net exposure (0.0 = neutral)
EV_OPTIMIZE_BY              = "per_dollar"      # or "per_share"
EV_SECOND_PASS              = "round_robin"     # spreads remaining cash across names
EV_MAX_SHARES_PER_NAME      = 0         # 0 = no cap; else integer cap
WAIT_FOR_OPEN               = True  # wait for market open (else use last close price)
WAIT_LOG_INTERVAL_SEC       = 30 # log every 30 sec while waiting
USE_PERCENT_BRACKETS        = True   # prefer ATR-based sizing across tickers
FORCE_PERCENT_BRACKETS      = False   # allow ATR to drive SL/TP
PERCENT_TP                  = 0.02 # 2%â€“5% reasonable
PERCENT_SL                  = 0.015 # 1%â€“3% reasonable
PRICE_DECIMALS              = 2 # round prices to this many decimals
IS_STREAMLIT                = False # set True when running inside Streamlit app
CLASS_IS_LONG               = 0  # LONG=0, SHORT=1 (ePredict uses this mapping)
CLASS_IS_SHORT              = 1   # keep as-is unless your labels differ
initial_capital             = 100_000 # backtest starting capital
BACKTEST_STACKED            = True # backtest stacked models (else base only)
allocation_fraction         = 0.25 # fraction of available cash to allocate per trade
RANK_BOOST_SMA50_LONG      = 1.05 # boost rank if above SMA50
RANK_BOOST_TOP_TERCILE     = 1.03 # boost rank if in top tercile by momentum
PORTFOLIO_MIN_WEIGHT       = 0.00 # minimum 5% weight if selected
PORTFOLIO_MAX_WEIGHT       = 0.33 # maximum 15% weight if selected
PORTFOLIO_K_MAX            = 15   # max names in portfolio
GRIDSEARCH_ENABLED          = False  # enable SL/TP grid search
MIN_TRADES_FOR_GRIDSEARCH   = 12    # skip SL/TP grid if not enough trades
GRIDSEARCH_MAX_COMBOS       = 9    # cap  sl x tp  to keep runs fast (3x3)
DYN_CONF_Q                  = 0.60 # 60th percentile
DYN_CONF_TOP_M              = 5    # average of top 5
min_model_return            = 0.0  # legacy synonym, keep for compatibility
min_model_sharpe            = 0.05 # legacy synonym, keep for compatibility
allocation_mode             = "dynamic"  # or "fixed"
REBALANCE_MIN_NAMES_PER_SIDE = 99 # minimum names per side to rebalance (else skip)
strict_macd                 = False # require MACD signal line confirmation
MIN_ROWS                    = 100  # minimum rows of historical data to consider a ticker
MIN_VOLATILITY              = 0.005 # 0.5%
VOLATILITY_THRESHOLD        = 0.015 # 1.5%
MAX_VOLATILITY              = 0.07      # was 0.10
use_ranking                 = True # rank signals by prob*conf
FILL_TO_TOP_K               = True     # was True
FILL_BY_DAY_PERCENTILE      = 0.70      # was 0.70
DAILY_TOP_K                 = 15         # was 12
TARGET_MIN_NAMES            = 12         # was 10
EV_CONF_POWER               = 2.0       # was 2.0
EV_CONF_MULT                = 0.50      # was 0.50
EV_MAX_WEIGHT_PER_NAME      = 0.25      # was 0.25
EV_SEED_ONE_SHARE           = False     # was True
WAIT_OPEN_OFFSET_MIN        = 2         # was 2
ENSEMBLE_POLICY               = "blend"   # was "winner"
PREFER_STACKED_IF_AVAILABLE   = False     # let data decide; stacked underperformed in Sharpe on avg.

#Speed flags for back testing and prediction
BACKTEST_BASE    = True   # new: skip per-family base runs
BACKTEST_STACKED = True   # already present; keep False to skip stacked
BACKTEST_MODE = "full"   # "full" | "ensemble_only" | "base_only" | "stacked_only"
WRITE_TRADES = False
WRITE_FUNNELS = False

# === Ensemble thresholds & confidence (latest sweep best) ===
ENSEMBLE_LONG_THRESHOLD  = 0.5051058198083329
ENSEMBLE_SHORT_THRESHOLD = 0.5051058198083329
ENSEMBLE_TEMP            = 0.9241685777738983
MIN_CONF_FLOOR           = 0.48197825392001525
# Optional: keep the tied value in config so runs are reproducible
ENSEMBLE_TIED            = 0.5351058198083329

APPLY_VOLUME     = True
APPLY_RSI        = False
APPLY_EMA_BUFFER = False
APPLY_MACD       = False

# Keep thresholds in case you re-enable gates later
rsi_threshold_long   = 55.691156447322655
rsi_threshold_short  = 43.64063219061841
ema_buffer_pct_long  = 1.0094547205579374
ema_buffer_pct_short = 1.002438089436719
macd_hist_long       = 0.08897825145536896
macd_hist_short      = 0.055759346401892694

# Tail gate disabled
xgb_prob_diff_quantile = 0.06402049979658236

# === Exits / trailing / risk ===
atr_sl_multiplier = 1.8
atr_tp_multiplier = 3.2
TRAIL_ATR_MULT    = 1.5106491815783136
RISK_PER_TRADE    = 0.02

# === Volume gate ===
volume_min        = 100_000     # round to a clean, liquid floor
volume_buffer_pct = 0.12817361620656242

# === Capacity ===
max_open_trades   = 9          # cast optimized ~8.64 to int

# === Ensemble weights (cap 0.50 per family) ===
ENSEMBLE_WEIGHTS = {
    "RandomForest":       0.21962,
    "LightGBM":           0.17744,
    "XGBoost":            0.20225,
    "CatBoost":           0.21570,
    "LogisticRegression": 0.18498,
}
ENSEMBLE_WEIGHT_CAP_PER_FAMILY = 0.50

# =======================
# Date ranges (train/test)
# =======================
train_start_str = '2018-01-01'
train_end_str   = '2024-08-31'
test_start_str  = '2024-09-01'
test_end_str    = '2025-09-16'

train_start = datetime.strptime(train_start_str, "%Y-%m-%d")
train_end   = datetime.strptime(train_end_str,   "%Y-%m-%d")
test_start  = datetime.strptime(test_start_str,  "%Y-%m-%d")
test_end    = datetime.strptime(test_end_str,    "%Y-%m-%d")

today_str     = (dt.date.today() - BDay(1)).strftime("%Y-%m-%d")
yesterday_str = (dt.date.today() - BDay(2)).strftime("%Y-%m-%d")

TICKER_MAP = {'BRK.B': 'BRK-B', 'BF.B': 'BF-B'}
ensemble_models_used = ["lightgbm", "logisticregression", "catboost", "xgboost", "randomforest"]

# ==========
# Paths/IO
# ==========
stock_file_path     = "1.Stocks"
stock_file          = "SP500_Companies.xlsx"
features_train_path = "2.Features_train"
features_test_path  = "2.Features_test"
model_path_base     = "3.Models_base"
model_path_stacked  = "3.Models_stacked"
metrics_folder      = "4.Metrics"
METRIC_OUTPUT       = "4.Metrics/stacking_accuracy_summary.csv"
backtest_path       = "5.Backtest"
predict_path        = "6.Predictions"
trading_path        = "7.Trading"
review_path         = "8.Review"

enable_plotting = True

def normalize_ticker_for_path(ticker: str) -> str:
    return TICKER_MAP.get(ticker, ticker)

# Prediction input discovery
PREFER_BASE_TEST_FILES = True
TEST_FILE_SUFFIXES = (
    "_test_features.csv",
    "_test_features_with_predictions.csv",
    "_with_predictions.csv",
)
MIRROR_PREDICTIONS_TO_TEST = True
AUTO_REFRESH_TEST_FEATURES = True
PREDICT_USE_RUN_SUBDIR = True
PREDICT_CLEAN_OLD      = False
PREDICT_PARALLEL       = True
PREDICT_N_JOBS   = max(2, (os.cpu_count() // 4) or 2)
# Targets/features
TARGET_COL  = "target"
XGB_EXCLUDE = ["xgboost_pred"] + [f"xgb_prob_class_{i}" for i in range(3)]

# canonical lists used by loops (keeps ensemble separate)
ENSEMBLE_MODEL_KEYS = ("lightgbm", "catboost", "randomforest", "logisticregression")
ENSEMBLE_MODEL_PREFIXES = ["xgb", "lgbm", "cat", "rf", "lr"]

# Base features
BASE_FEATURE_COLS = [
    'open','high','low','close','volume',
    'return_1d','rsi','macd','macd_signal',
    'upper_band','lower_band','atr','ema_10',
    'volatility','rel_volume','ema_dist','bb_width',
    'roll_max_ratio','roll_min_ratio',
    'sma_50','sma_200','price_sma_50','price_sma_200',
    'volume_change','momentum_10',
    'return_5d','return_10d','rsi_z',
    'vol_std_10','vol_ewma_20','volatility_regime',
    'bb_mid','atr_14',
    'vol_parkinson',
    'adj_close'
]
# Training/tuning controls
ENABLE_TUNING = {
    'xgboost': True,
    'lightgbm': True,
    'catboost': True,
    'randomforest': True,
    'logisticregression': True
}

model_param_grids = {
    'xgb_model.pkl': {
        'n_estimators': [200, 400, 600],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.005, 0.01, 0.03],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_lambda': [0.1, 1.0, 3.0],
        'reg_alpha': [0, 0.1, 0.5],
    },
    'lgb_model.pkl': {
        'n_estimators': [200, 400, 600],
        'max_depth': [6, 10, 14],
        'learning_rate': [0.005, 0.01, 0.03],
        'num_leaves': [15, 31, 63],
        'min_child_samples': [5, 10, 20],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_lambda': [0.1, 1.0, 3.0],
        'reg_alpha': [0, 0.1, 0.5],
    },
    'cb_model.pkl': {
        'iterations': [300, 500, 700],
        'depth': [6, 8, 10],
        'learning_rate': [0.005, 0.01, 0.03],
        'l2_leaf_reg': [1, 5, 10],
        'bagging_temperature': [0.5, 1, 2],
    },
    'rf_model.pkl': {
        'n_estimators': [200, 500, 1000],
        'max_depth': [5, 10, None],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
    },
    'lr_model.pkl': {
        'C': [0.01, 0.1, 1.0, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear'],
    },
}

backtest_optimize_param_grid = {
    # 1) Ensemble classification & confidence
    "ENSEMBLE_LONG_THRESHOLD":  [0.58, 0.60, 0.62, 0.64],
    "ENSEMBLE_SHORT_THRESHOLD": [0.60, 0.62, 0.64],
    "MIN_CONF_FLOOR":           [0.58, 0.60, 0.62, 0.64],
    "ENSEMBLE_TEMP":            [0.75, 0.85, 1.00],

    # 2) Technical regime filters (biggest impact on your funnels)
    "APPLY_RSI":                [False, True],
    "rsi_threshold_long":       [50],
    "rsi_threshold_short":      [50],
    "ema_buffer_pct_long":      [0.995, 1.000, 1.005, 1.010],
    "ema_buffer_pct_short":     [0.995, 1.000, 1.005, 1.010],
    "macd_hist_long":           [0.00, 0.02, 0.05],
    "macd_hist_short":          [0.00, 0.02, 0.05],   # used as -abs(...)

    # 3) Relative margin gate (correct key; q ≈ fraction kept)
    "xgb_prob_diff_quantile":   [0.40, 0.50, 0.60],   # keep top 40–60% by margin

    # 4) Exits (SL/TP) – strong Sharpe lever
    "atr_tp_multiplier":        [2.0, 2.2, 2.5],
    "atr_sl_multiplier":        [0.8, 1.0, 1.2],
    "TRAIL_ATR_MULT":           [0.8, 1.0, 1.2],

    # 5) Volatility / portfolio shaping
    "RISK_PER_TRADE":           [0.0075, 0.01, 0.0125, 0.015],
}

# ===== Single source of truth: models, aliases, prefixes, display names =====
def _norm_token(s: str) -> str:
    """lower + remove non-alnum; used to normalize alias lookups."""
    return re.sub(r"[^a-z0-9]", "", str(s).strip().lower())

MODEL_REGISTRY: Dict[str, Dict] = {
    # canonical key     prefix  TitleCase             Friendly   also-accept-as alias tokens
    "xgboost": {
        "prefix":   "xgb",
        "title":    "XGBoost",
        "friendly": "Char",
        "aliases":  ["xgb", "xg_boost", "XGBoost", "Char"],
    },
    "lightgbm": {
        "prefix":   "lgbm",
        "title":    "LightGBM",
        "friendly": "Sophie",
        "aliases":  ["lgbm", "light_gbm", "LightGBM", "Sophie", "lgb"],
    },
    "catboost": {
        "prefix":   "cat",
        "title":    "CatBoost",
        "friendly": "Chloe",
        "aliases":  ["cat", "cat_boost", "CatBoost", "Chloe", "catb"],
    },
    "randomforest": {
        "prefix":   "rf",
        "title":    "RandomForest",
        "friendly": "Daniel",
        "aliases":  ["rf", "random_forest", "RandomForest", "Daniel", "randforest"],
    },
    "logisticregression": {
        "prefix":   "lr",
        "title":    "LogisticRegression",
        "friendly": "Lera",
        "aliases":  ["lr", "logreg", "logistic", "LogisticRegression", "Lera"],
    },
    "ensemble": {
        "prefix":   "ens",
        "title":    "Ensemble",
        "friendly": "Ensemble",
        "aliases":  ["Ensemble"],
    },
}

CANONICAL_MODEL_KEYS = tuple(k for k in MODEL_REGISTRY.keys() if k != "ensemble")

# config.py  (Base artifacts live under 3.Models_base\..., no _stack_ in names)
MODEL_FILE_TEMPLATES = {
    "xgboost": {
        "dir": model_path_base,
        "model": "{ticker}_xgb_model.pkl",
        "scaler": "{ticker}_xgb_scaler_columns.pkl",
        "features": "{ticker}_xgb_features.txt",
    },
    "lightgbm": {
        "dir": model_path_base,
        "model": "{ticker}_lightgbm_model.pkl",
        "scaler": "{ticker}_lightgbm_scaler_columns.pkl",
        "features": "{ticker}_lightgbm_features.txt",
    },
    "catboost": {
        "dir": model_path_base,
        "model": "{ticker}_catboost_model.pkl",
        "scaler": "{ticker}_catboost_scaler_columns.pkl",
        "features": "{ticker}_catboost_features.txt",
    },
    "randomforest": {
        "dir": model_path_base,
        "model": "{ticker}_rf_model.pkl",
        "scaler": "{ticker}_rf_scaler_columns.pkl",
        "features": "{ticker}_rf_features.txt",
    },
    "logisticregression": {
        "dir": model_path_base,
        "model": "{ticker}_logreg_model.pkl",
        "scaler": "{ticker}_logreg_scaler_columns.pkl",
        "features": "{ticker}_logreg_features.txt",
    },
}

# Keep stacked artifacts in the stacked dir, with _stack_ filenames
STACKED_FILE_TEMPLATES = {
    "lightgbm": {
        "dir": model_path_stacked,
        "model": "{ticker}_lightgbm_stack_model.pkl",
        "scaler": "{ticker}_lightgbm_stack_scaler_columns.pkl",
        "features": "{ticker}_lightgbm_stack_features.txt",
    },
    "catboost": {
        "dir": model_path_stacked,
        "model": "{ticker}_catboost_stack_model.pkl",
        "scaler": "{ticker}_catboost_stack_scaler_columns.pkl",
        "features": "{ticker}_catboost_stack_features.txt",
    },
    "randomforest": {
        "dir": model_path_stacked,
        "model": "{ticker}_rf_stack_model.pkl",
        "scaler": "{ticker}_rf_stack_scaler_columns.pkl",
        "features": "{ticker}_rf_stack_features.txt",
    },
    "logisticregression": {
        "dir": model_path_stacked,
        "model": "{ticker}_lr_stack_model.pkl",
        "scaler": "{ticker}_lr_stack_scaler_columns.pkl",
        "features": "{ticker}_lr_stack_features.txt",
    },
}

# Canonical alias map for STACKED prediction columns (KEYED BY SHORT PREFIX)
# Accept both short and TitleCase variants so older files remain readable.
STACKED_PRED_ALIASES: Dict[str, List[str]] = {
    "xgb":  ["xgb_stack_prediction",  "xgboost_stack_prediction",  "XGBoost_stack_prediction"],
    "lgbm": ["lgbm_stack_prediction", "lightgbm_stack_prediction", "LightGBM_stack_prediction"],
    "cat":  ["cat_stack_prediction",  "catboost_stack_prediction", "CatBoost_stack_prediction"],
    "rf":   ["rf_stack_prediction",   "randomforest_stack_prediction", "RandomForest_stack_prediction"],
    "lr":   ["lr_stack_prediction",   "logisticregression_stack_prediction", "LogisticRegression_stack_prediction"],
}

# Build reverse lookups once
_ALIAS_TO_KEY: Dict[str, str] = {}
_PREFIX_TO_KEY: Dict[str, str] = {}
for _key, _meta in MODEL_REGISTRY.items():
    _ALIAS_TO_KEY[_norm_token(_key)] = _key
    _ALIAS_TO_KEY[_norm_token(_meta["title"])] = _key
    _ALIAS_TO_KEY[_norm_token(_meta["friendly"])] = _key
    _ALIAS_TO_KEY[_norm_token(_meta["prefix"])] = _key
    for a in _meta.get("aliases", []):
        _ALIAS_TO_KEY[_norm_token(a)] = _key
    _PREFIX_TO_KEY[_meta["prefix"]] = _key

# ---- helpers used across train/backtest/predict/trade ----
def norm_key(name: str) -> str:
    """Return canonical key (e.g., 'lightgbm') for any alias/friendly/prefix/title."""
    key = _ALIAS_TO_KEY.get(_norm_token(name))
    if not key:
        raise KeyError(f"Unknown model name '{name}'")
    return key

def to_prefix(name: str) -> str:
    n = str(name).lower()
    if "xgb" in n: return "xgb"
    if "lightgbm" in n or "lgbm" in n: return "lgbm"
    if "catboost" in n or n == "cat": return "cat"
    if "randomforest" in n or "rf" == n: return "rf"
    if "logistic" in n or "lr" == n: return "lr"
    return n

def to_title(name: str) -> str:
    return MODEL_TITLE_BY_PREFIX.get(to_prefix(name), name)

def to_friendly(name: str) -> str:
    p = to_prefix(name)
    return {
        "xgb": "XGBoost",
        "lgbm": "LightGBM",
        "cat": "CatBoost",
        "rf": "RandomForest",
        "lr": "LogisticRegression",
    }.get(p, name)

# Base column builders (pred/prob columns emitted by model inference)
def pred_col(model: str) -> str:
    p = to_prefix(model); return {
        "xgb":  "xgb_pred",
        "lgbm": "lgbm_pred",
        "cat":  "cat_pred",
        "rf":   "rf_pred",
        "lr":   "lr_pred",
    }[p]

def prob1_col(model: str) -> str:
    p = to_prefix(model); return {
        "xgb":  "xgb_prob_class_1",
        "lgbm": "lgbm_prob_class_1",
        "cat":  "cat_prob_class_1",
        "rf":   "rf_prob_class_1",
        "lr":   "lr_prob_class_1",
    }[p]

def max_prob_col(model: str) -> str:
    p = to_prefix(model); return {
        "xgb":  "xgb_max_prob",
        "lgbm": "lgbm_max_prob",
        "cat":  "cat_max_prob",
        "rf":   "rf_max_prob",
        "lr":   "lr_max_prob",
    }[p]

def prob_diff_col(model: str) -> str:
    p = to_prefix(model); return {
        "xgb":  "xgb_prob_diff",
        "lgbm": "lgbm_prob_diff",
        "cat":  "cat_prob_diff",
        "rf":   "rf_prob_diff",
        "lr":   "lr_prob_diff",
    }[p]

# Stacked column canonical names
def stack_pred_col(model: str) -> str:
    p = to_prefix(model)
    return f"{p}_stack_prediction"

def stack_prob1_col(model: str) -> str:
    p = to_prefix(model)
    return f"{p}_stack_prob_class_1"

def stack_prob0_col(model: str) -> str:
    p = to_prefix(model)
    return f"{p}_stack_prob_class_0"

def stack_max_prob_col(model: str) -> str:
    p = to_prefix(model)
    return f"{p}_stack_max_prob"

def stack_prob_diff_col(model: str) -> str:
    p = to_prefix(model)
    return f"{p}_stack_prob_diff"

MODEL_TITLE_BY_PREFIX = {
    "xgb":  "XGBoost",
    "lgbm": "LightGBM",
    "cat":  "CatBoost",
    "rf":   "RandomForest",
    "lr":   "LogisticRegression",
}

def resolve_stacked_pred_column(df, model: str) -> str | None:
    """
    Return the *first* stacked prediction column that exists for the model (never a base column).
    """
    fam = to_prefix(model)
    candidates = [stack_pred_col(fam)] + STACKED_PRED_ALIASES.get(fam, [])
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _safe_norm(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", str(s or "").strip().lower())

def key_from_prefix(prefix: str) -> str:
    """
    Return the long canonical model key (e.g., 'lightgbm') from a short prefix (e.g., 'lgbm').
    Falls back to the prefix itself if the registry isn't available.
    """
    p = to_prefix(prefix)
    # If MODEL_REGISTRY exists, invert it; else use common defaults
    reg: Dict[str, Dict] = globals().get("MODEL_REGISTRY", {})
    for k, meta in (reg or {}).items():
        if str(meta.get("prefix", "")).lower() == p:
            return k
    inv = {"xgb": "xgboost", "lgbm": "lightgbm", "cat": "catboost", "rf": "randomforest", "lr": "logisticregression"}
    return inv.get(p, p)

def stacked_pred_aliases(name: str) -> List[str]:
    """
    Return all accepted stacked prediction column aliases for a given model name.
    You can pass 'lgbm', 'LightGBM', or 'lightgbm' â€” resolution is robust.
    """
    key = key_from_prefix(name)
    return STACKED_PRED_ALIASES.get(key, [])

# Optional: convenience to list *all* stacked columns we may generate/expect for a family
def stacked_all_prob_cols(name: str) -> List[str]:
    p = to_prefix(name)
    return [
        stack_prob0_col(p),
        stack_prob1_col(p),
        stack_max_prob_col(p),
        stack_prob_diff_col(p),
    ]
# Centralized patterns used by stacked feature selection in backtest/predict
STACK_BASE_COL_PATTERNS = {
    "base": {
        "pred":      [pred_col(m)      for m in ENSEMBLE_MODEL_PREFIXES],   # e.g., xgb_pred
        "prob1":     [prob1_col(m)     for m in ENSEMBLE_MODEL_PREFIXES],   # e.g., xgb_prob_class_1
        "max_prob":  [max_prob_col(m)  for m in ENSEMBLE_MODEL_PREFIXES],   # e.g., xgb_max_prob
        "prob_diff": [prob_diff_col(m) for m in ENSEMBLE_MODEL_PREFIXES],   # e.g., xgb_prob_diff
    },
    "stack": {
        # canonical stack_pred first for each family, then all accepted aliases
        "pred": (
            [stack_pred_col(m) for m in ENSEMBLE_MODEL_PREFIXES] +
            [alias for m in ENSEMBLE_MODEL_PREFIXES for alias in STACKED_PRED_ALIASES.get(m, [])]
        ),
        "prob1":     [stack_prob1_col(m)     for m in ENSEMBLE_MODEL_PREFIXES],
        "max_prob":  [stack_max_prob_col(m)  for m in ENSEMBLE_MODEL_PREFIXES],
        "prob_diff": [stack_prob_diff_col(m) for m in ENSEMBLE_MODEL_PREFIXES],
    }
}