"""Record-level data quality. Rules test the RECORD, never the performance.

A rule of the form "performance too large" would be a hidden second
threshold. Every rule here is about whether the record can be true at all.
Rules are checked in order; a row is reported under its first failure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cept import config, schema as s

_NUM_CONDITIONS = [c for c in s.CONDITIONS if c != s.SIDE]


def _col(df, c):
    return df[c] if c in df.columns else pd.Series(np.nan, index=df.index)


RULES = {
    "condition_missing": lambda d: d[_NUM_CONDITIONS].isna().any(axis=1)
                                   | d[s.SIDE].isna(),
    "spread_implausible": lambda d: ~d[s.SPREAD_BPS].between(*config.SPREAD_RANGE_BPS),
    "metric_inconsistent": lambda d: (
        _col(d, s.SLIPPAGE_BPS).notna()
        & ((_col(d, s.SLIPPAGE_BPS) / d[s.SPREAD_BPS] - d[s.METRIC]).abs()
           > config.METRIC_CONSISTENCY_TOL * np.maximum(1.0, d[s.METRIC].abs()))),
    "participation_impossible": lambda d: ~((d[s.PARTICIPATION] > 0)
                                            & (d[s.PARTICIPATION] <= 100)),
    "duration_nonpositive": lambda d: ~(d[s.DURATION_MIN] > 0),
    "auction_share_impossible": lambda d: (d[s.OPEN_PCT] + d[s.CLOSE_PCT]) > 100.0,
    "adv_nonpositive": lambda d: ~(d[s.PCT_ADV] > 0),
    "volatility_nonpositive": lambda d: ~(d[s.VOLATILITY] > 0),
    "fill_rate_impossible": lambda d: _col(d, s.FILL_RATE).notna()
                                      & ~((_col(d, s.FILL_RATE) > 0)
                                          & (_col(d, s.FILL_RATE) <= 100)),
}


def reasons(df: pd.DataFrame) -> pd.Series:
    out = pd.Series("", index=df.index, dtype=object)
    for name, rule in RULES.items():
        hit = rule(df).fillna(True).astype(bool) & (out == "")
        out[hit] = name
    return out


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = reasons(df)
    exc = df[r != ""].copy()
    exc["dq_reason"] = r[r != ""]
    return df[r == ""].reset_index(drop=True), exc.reset_index(drop=True)
