"""Order conditions -> model inputs.

Volatility is split into the stock's TYPICAL level (median of its own
same-day Parkinson vol over the fit year) and the day's SURPRISE relative to
it. Same-day vol is partly caused by a large order itself; keeping the
surprise as its own term is what lets sensitivity.py show exactly which
alerts it clears.

Strategy: a group pools several strategies under ONE threshold. Each strategy
other than the most frequent (the reference) gets an indicator, so the model
can still learn that one strategy normally runs slightly differently. If the
strategies in a group really perform alike, these coefficients come out near
zero -- and that number is the evidence the pooling is justified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cept import schema as s

FEATURE_NAMES = ["log_vol_typ", "log_vol_surprise", "log_dur", "log_spread",
                 "log_pct_adv", "sqrt_pr", "open_frac", "close_frac", "is_buy"]


def feature_names(include_surprise: bool, strategies=()) -> list[str]:
    return ([n for n in FEATURE_NAMES
             if include_surprise or n != "log_vol_surprise"]
            + [f"strategy_{a}" for a in list(strategies)[1:]])


def strategy_levels(df: pd.DataFrame) -> list[str]:
    """Strategies in the group, most frequent first (the reference)."""
    vc = _algo(df).value_counts()
    return sorted(vc.index, key=lambda a: (-vc[a], a))


def _algo(df: pd.DataFrame) -> pd.Series:
    return df[s.ALGO].astype(str).str.strip().str.upper()


class VolTypical:
    def __init__(self, by_symbol: dict, fallback: float):
        self.by_symbol = dict(by_symbol)
        self.fallback = float(fallback)

    @classmethod
    def fit(cls, df: pd.DataFrame, min_n: int) -> "VolTypical":
        g = df.groupby(s.SYMBOL)[s.VOLATILITY]
        n, med = g.size(), g.median()
        keep = med[n >= min_n]
        return cls({str(k): float(v) for k, v in keep.items()},
                   float(df[s.VOLATILITY].median()))

    def lookup(self, df: pd.DataFrame) -> np.ndarray:
        return (df[s.SYMBOL].astype(str).map(self.by_symbol)
                .fillna(self.fallback).to_numpy(dtype=float))

    def to_dict(self) -> dict:
        return {"by_symbol": self.by_symbol, "fallback": self.fallback}

    @classmethod
    def from_dict(cls, d: dict) -> "VolTypical":
        return cls(d["by_symbol"], d["fallback"])


def design(df: pd.DataFrame, voltyp: VolTypical,
           include_surprise: bool = True, strategies=()) -> np.ndarray:
    vt = voltyp.lookup(df)
    vol = df[s.VOLATILITY].to_numpy(dtype=float)
    cols = [np.log(vt)]
    if include_surprise:
        cols.append(np.log(vol / vt))
    cols += [
        np.log(df[s.DURATION_MIN].to_numpy(dtype=float)),
        np.log(df[s.SPREAD_BPS].to_numpy(dtype=float)),
        np.log(df[s.PCT_ADV].to_numpy(dtype=float)),
        np.sqrt(df[s.PARTICIPATION].to_numpy(dtype=float) / 10.0),
        df[s.OPEN_PCT].to_numpy(dtype=float) / 100.0,
        df[s.CLOSE_PCT].to_numpy(dtype=float) / 100.0,
        df[s.SIDE].astype(str).str.upper().str.startswith("B")
          .to_numpy(dtype=float),
    ]
    if len(strategies) > 1:
        algo = _algo(df).to_numpy()
        cols += [(algo == a).astype(float) for a in list(strategies)[1:]]
    return np.column_stack(cols)
