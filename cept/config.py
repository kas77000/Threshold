"""The only file edited to point CEPT at real data, and the policy record.

POLICY VALUES are fixed before any data is consulted. They are copied into
every model file with their rationale. Changing one is a governance event,
not a tuning step.
"""

from __future__ import annotations

from cept import schema

# ---- performance metric: the ONE place it is named -----------------------
# To measure another strategy family on another metric, change these lines
# only. The model file records the metric, and `score` refuses a model that
# was fitted on a different one.
METRIC_COLUMN = "ePvwap/Sprd"   # performance, in METRIC_UNITS
METRIC_UNITS = "spreads"
# The same performance in bps, used by the data-quality rule
# METRIC_COLUMN == METRIC_BPS_COLUMN / Sprd. None disables that rule.
METRIC_BPS_COLUMN = "Pvwap"

COLUMN_MAP = {
    "aggrTgtId": schema.ORDER_ID, "Strategy": schema.ALGO,
    "Sym": schema.SYMBOL, "Date": schema.ORDER_DATE,
    METRIC_COLUMN: schema.METRIC,
    **({METRIC_BPS_COLUMN: schema.SLIPPAGE_BPS} if METRIC_BPS_COLUMN else {}),
    "Sprd": schema.SPREAD_BPS, "%Adv": schema.PCT_ADV,
    "Vol": schema.VOLATILITY, "PR": schema.PARTICIPATION,
    "Dur": schema.DURATION_MIN, "%OPEN": schema.OPEN_PCT,
    "%CLOSE": schema.CLOSE_PCT, "Side": schema.SIDE,
    "FR": schema.FILL_RATE, "#Shares": schema.QUANTITY,
    "$Mln": schema.NOTIONAL, "%POST": schema.PASSIVE_FILL,
    "Rev30min": schema.REVERSION,
}

# ---- strategy groups: ONE model and ONE threshold per group ---------------
# Strategies that share the same performance behaviour are pooled into one
# group. To add a strategy, map it to a group; to split or merge groups,
# change the right-hand side. A strategy not listed is excluded from the run
# and reported by name -- never guessed into a group.
# The cohort test still checks every strategy separately inside its group.
STRATEGY_GROUP = {
    "VWAP": "TWAP_VWAP",
    "TWAP": "TWAP_VWAP",
    "TMX": "TWAP_VWAP",
}

DEFAULT_MARKET = "HK"

# ---- POLICY (spec section 3.3, 3.4) --------------------------------------
RARITY_A = 1e-4      # two-sided: flag if rarer than 1 in 10,000 given conditions
FLOOR_A = 4.0        # never tighter than 4 conditional sigma
RARITY_B = 1e-3      # watch list: rarer than 1 in 1,000
FLOOR_B = 3.0
COHORT_T = 4.0       # |t| of a cohort's mean z
POLICY_RATIONALE = (
    "Tier A flags an order whose performance is rarer than 1 in 10,000 given "
    "its own conditions, and never closer than 4 conditional standard "
    "deviations. On a book whose conditional residuals are near-normal this is "
    "exactly mean +/- 4 sigma. Values were fixed before the data was examined "
    "and are not selected from the alert count.")

# ---- estimation knobs (not policy: they affect precision, not stringency) -
TAIL_PROBS = (0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995)
MIN_CELL_N = 5000
MIN_SYMBOL_N = 20
COHORT_FEATURES = [schema.PARTICIPATION, schema.PCT_ADV, schema.VOLATILITY,
                   schema.DURATION_MIN, schema.SPREAD_BPS, schema.CLOSE_PCT,
                   schema.FILL_RATE]
COHORT_BINS = 5
COHORT_MIN_N = 100
INJECT_SIZES = (5, 8, 12, 20)
INJECT_N = 500
DRIFT_THRESHOLD_PCT = 25.0

# ---- data-quality bounds -------------------------------------------------
SPREAD_RANGE_BPS = (0.5, 500.0)
METRIC_CONSISTENCY_TOL = 0.01

# ---- explanation tags ----------------------------------------------------
PARTIAL_FILL_BELOW = 80.0
VOL_SURPRISE_ABOVE = 2.0
AUCTION_HEAVY_ABOVE = 50.0
HIGH_PR_ABOVE = 30.0
HIGH_ADV_PCT = 10.0


def group_for(strategy) -> str | None:
    if strategy is None:
        return None
    key = str(strategy).strip().upper()
    if not key or key in {"NAN", "NONE"}:
        return None
    return STRATEGY_GROUP.get(key)
