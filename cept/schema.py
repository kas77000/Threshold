"""Canonical column names. Only load.py ever sees a vendor column name."""

ORDER_ID = "order_id"
ALGO = "algo"
SYMBOL = "symbol"
MARKET = "market"
ORDER_DATE = "order_date"
BENCHMARK = "benchmark"

METRIC = "perf_in_spreads"          # ePvwap/Sprd, as supplied

SLIPPAGE_BPS = "slippage_bps"       # Pvwap
SPREAD_BPS = "spread_bps"           # Sprd
PCT_ADV = "pct_adv"                 # executed size / ADV, percent
VOLATILITY = "volatility"           # single-day Parkinson, daily fraction
PARTICIPATION = "participation"     # PR, percent
DURATION_MIN = "duration_min"       # Dur, minutes
OPEN_PCT = "open_pct"
CLOSE_PCT = "close_pct"
SIDE = "side"

FILL_RATE = "fill_rate"             # FR, percent -- an OUTCOME, never a covariate
QUANTITY = "quantity"               # #Shares, ordered quantity
NOTIONAL = "notional"
PASSIVE_FILL = "passive_fill_pct"   # outcome, never a covariate
REVERSION = "reversion"             # outcome, never a covariate

REQUIRED = [ORDER_ID, ALGO, SYMBOL, ORDER_DATE, METRIC]

# Inputs to the expected-performance model. A row missing any of these is a
# data exception (dq.py), not a silent drop.
CONDITIONS = [VOLATILITY, DURATION_MIN, SPREAD_BPS, PCT_ADV, PARTICIPATION,
              OPEN_PCT, CLOSE_PCT, SIDE]

NUMERIC = [METRIC, SLIPPAGE_BPS, SPREAD_BPS, PCT_ADV, VOLATILITY,
           PARTICIPATION, DURATION_MIN, OPEN_PCT, CLOSE_PCT, FILL_RATE,
           QUANTITY, NOTIONAL, PASSIVE_FILL, REVERSION]

# Medians stamped at fit and compared at score.
DRIFT_FEATURES = [VOLATILITY, DURATION_MIN, SPREAD_BPS, PCT_ADV,
                  PARTICIPATION, CLOSE_PCT, FILL_RATE]
