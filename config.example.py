# ==========================================================
# config.example.py
# Copy this file to config.py and fill values locally.
# NEVER commit config.py (contains secrets).
# ==========================================================

import os
from datetime import datetime

# =====================
# Alpaca (BROKER)
# =====================
# Use environment variables in production
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_PAPER_URL  = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL   = "https://api.alpaca.markets"

USE_LIVE_TRADING  = False
ALPACA_BASE_URL   = ALPACA_LIVE_URL if USE_LIVE_TRADING else ALPACA_PAPER_URL


# =====================
# Core budgets
# =====================
PORTFOLIO_BUDGET = 100_000.0
DAILY_BUDGET     = 100_000.0
USE_FULL_DAILY_BUDGET = True


# =====================
# Session-aware risk policy
# =====================
SESSION_POLICY_ENABLED = True

# Highly liquid symbols allowed for overnight / AH handling
OVERNIGHT_LIQUID_ALLOWLIST = {
    "SPY","QQQ","IWM","DIA",
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA",
}

# Overnight exposure caps
MAX_OVERNIGHT_GROSS_EXPOSURE_PCT = 0.35
MAX_OVERNIGHT_NEW_TRADES = 0

# Position sizing overnight
OVERNIGHT_POSITION_SIZE_MULT = 0.50

# Stop behaviour
OVERNIGHT_STOP_MODE = "soft"          # "soft" | "hard"
OVERNIGHT_SOFT_STOP_TRIGGER_ATR = 1.2
RTH_HARD_STOP_AT_OPEN = True

# Extended hours
ALLOW_EXTENDED_HOURS_ORDERS = True
EXTENDED_HOURS_ONLY_IF_LIQUID = True


# =====================
# Trade execution guards
# =====================
MIN_SHARES_PER_TRADE = 5
MAX_TRADES_PER_RUN  = 8

USE_QUOTE_PRICE_FOR_SIZING = True
QUOTE_SIDE_FOR_LONG  = "ask"
QUOTE_SIDE_FOR_SHORT = "bid"

WAIT_FOR_OPEN = True
WAIT_OPEN_OFFSET_MIN = 5


# =====================
# Files & folders
# =====================
features_train_path = "2.Features_train"
features_test_path  = "2.Features_test"
model_path_base     = "3.Models_base"
model_path_stacked  = "3.Models_stacked"
metrics_folder      = "4.Metrics"
backtest_path       = "5.Backtest"
predict_path        = "6.Predictions"
trading_path        = "7.Trading"
review_path         = "8.Review"
governance_path     = "7.Governance"


# =====================
# Misc
# =====================
RUN_MODE = os.getenv("RUN_MODE", "research").lower()
IS_LIVE = RUN_MODE == "live"
VERBOSE = False
