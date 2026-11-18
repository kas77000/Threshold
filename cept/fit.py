"""Fit one frozen model per strategy group (one threshold per group)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cept import cohort, config, schema as s
from cept.features import VolTypical, design, strategy_levels
from cept.model import LocationScale
from cept.tail import TailFit, exceed_prob, fit_side, threshold

TIERS = {"A": (config.RARITY_A, config.FLOOR_A),
         "B": (config.RARITY_B, config.FLOOR_B)}


@dataclass
class CellModel:
    group: str
    n: int
    include_surprise: bool
    voltyp: VolTypical
    ls: LocationScale
    z_med: float
    z_mad: float
    tail_hi: TailFit
    tail_lo: TailFit
    thr: dict
    cohort_edges: dict = field(default_factory=dict)
    medians: dict = field(default_factory=dict)
    cohort_base: dict = field(default_factory=dict)
    strategies: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"group": self.group, "n": self.n,
                "include_surprise": self.include_surprise,
                "voltyp": self.voltyp.to_dict(), "ls": self.ls.to_dict(),
                "z_med": self.z_med, "z_mad": self.z_mad,
                "tail_hi": self.tail_hi.to_dict(),
                "tail_lo": self.tail_lo.to_dict(), "thr": self.thr,
                "cohort_edges": self.cohort_edges, "medians": self.medians,
                "cohort_base": self.cohort_base,
                "strategies": self.strategies}

    @classmethod
    def from_dict(cls, d: dict) -> "CellModel":
        return cls(d["group"], int(d["n"]), bool(d["include_surprise"]),
                   VolTypical.from_dict(d["voltyp"]),
                   LocationScale.from_dict(d["ls"]),
                   float(d["z_med"]), float(d["z_mad"]),
                   TailFit.from_dict(d["tail_hi"]),
                   TailFit.from_dict(d["tail_lo"]), d["thr"],
                   d.get("cohort_edges", {}), d.get("medians", {}),
                   d.get("cohort_base", {}), d.get("strategies", []))


def fit_cell(df: pd.DataFrame, group: str,
             include_surprise: bool = True) -> CellModel:
    vt = VolTypical.fit(df, config.MIN_SYMBOL_N)
    strategies = strategy_levels(df)
    X = design(df, vt, include_surprise, strategies)
    y = df[s.METRIC].to_numpy(dtype=float)
    ls = LocationScale().fit(X, y)
    z0 = ls.z(X, y)
    med = float(np.median(z0))
    mad = float(1.4826 * np.median(np.abs(z0 - med)))
    z = (z0 - med) / mad
    t_hi, t_lo = fit_side(z), fit_side(-z)
    thr = {tier: {"hi": threshold(t_hi, r, f), "lo": threshold(t_lo, r, f)}
           for tier, (r, f) in TIERS.items()}
    qs = np.linspace(0, 1, config.COHORT_BINS + 1)[1:-1]
    edges = {f: np.unique(np.quantile(df[f].dropna(), qs)).tolist()
             for f in config.COHORT_FEATURES
             if f in df.columns and df[f].notna().any()}
    medians = {f: float(df[f].median()) for f in s.DRIFT_FEATURES
               if f in df.columns and df[f].notna().any()}
    base = cohort.baselines(edges, df, z)
    return CellModel(group, len(df), include_surprise, vt, ls, med, mad,
                     t_hi, t_lo, thr, edges, medians, base, strategies)


def fit_all(df: pd.DataFrame, include_surprise: bool = True,
            min_cell_n: int = config.MIN_CELL_N):
    cells, skipped = {}, {}
    for bench, g in df.groupby(s.GROUP):
        if len(g) < min_cell_n:
            skipped[str(bench)] = int(len(g))
            continue
        cells[str(bench)] = fit_cell(g, str(bench), include_surprise)
    return cells, skipped


def zscores(cm: CellModel, df: pd.DataFrame, y=None):
    X = design(df, cm.voltyp, cm.include_surprise, cm.strategies)
    mu, sc = cm.ls.predict(X)
    y = df[s.METRIC].to_numpy(dtype=float) if y is None else np.asarray(y, float)
    z = ((y - mu) / sc - cm.z_med) / cm.z_mad
    return z, mu + cm.z_med * sc, sc * cm.z_mad


def predicted_count(cm: CellModel, n: int, tier: str) -> float:
    t = cm.thr[tier]
    return float(n * (exceed_prob(cm.tail_hi, t["hi"])
                      + exceed_prob(cm.tail_lo, t["lo"])))
