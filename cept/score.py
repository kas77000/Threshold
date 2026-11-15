"""Apply frozen cell models to a month. Refuse what would be meaningless."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cept import config, schema as s
from cept.fit import zscores


class GuardError(ValueError):
    """A refusal. Never downgraded to a warning."""


def check_metric(meta: dict) -> None:
    """Refuse a model fitted on another performance metric or unit.

    A z-score is only meaningful against the metric the model was fitted on;
    a VWAP-relative model applied to arrival-price performance would produce
    plausible-looking, wrong alerts.
    """
    fitted = (meta.get("metric"), meta.get("metric_units"))
    now = (config.METRIC_COLUMN, config.METRIC_UNITS)
    if fitted != now:
        raise GuardError(
            f"Model was fitted on metric '{fitted[0]}' in '{fitted[1]}', but "
            f"cept/config.py now names '{now[0]}' in '{now[1]}'. Refit on the "
            f"new metric, or restore the config that matches this model.")


def check_no_overlap(fit_start, fit_end, df: pd.DataFrame) -> None:
    d = df[s.ORDER_DATE].dropna()
    if d.empty or fit_start is None or fit_end is None:
        return
    lo, hi = d.min(), d.max()
    if lo <= pd.Timestamp(fit_end) and pd.Timestamp(fit_start) <= hi:
        raise GuardError(
            f"Scored window {lo.date()}..{hi.date()} overlaps the fit window "
            f"{fit_start}..{fit_end}; the flag rate would be circular.")


def _tags(df: pd.DataFrame, voltyp_lookup: np.ndarray) -> list[str]:
    out = []
    fr = df[s.FILL_RATE] if s.FILL_RATE in df else pd.Series(np.nan, index=df.index)
    surprise = df[s.VOLATILITY].to_numpy(float) / voltyp_lookup
    for i, (_, r) in enumerate(df.iterrows()):
        t = []
        if pd.notna(fr.iloc[i]) and fr.iloc[i] < config.PARTIAL_FILL_BELOW:
            t.append(f"partial fill (FR {fr.iloc[i]:.0f}%)")
        if surprise[i] > config.VOL_SURPRISE_ABOVE:
            t.append(f"volatile day ({surprise[i]:.1f}x typical)")
        if r[s.OPEN_PCT] + r[s.CLOSE_PCT] > config.AUCTION_HEAVY_ABOVE:
            t.append("auction heavy")
        if r[s.PARTICIPATION] > config.HIGH_PR_ABOVE:
            t.append(f"high participation ({r[s.PARTICIPATION]:.0f}%)")
        if r[s.PCT_ADV] > config.HIGH_ADV_PCT:
            t.append(f"large vs ADV ({r[s.PCT_ADV]:.0f}%)")
        out.append("; ".join(t))
    return out


def score_orders(cells: dict, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("expected_perf", "expected_disp", "z", "excess"):
        out[c] = np.nan
    out["tier"] = "NO_MODEL"
    out["tags"] = ""
    for bench, g in df.groupby(s.BENCHMARK):
        cm = cells.get(str(bench))
        if cm is None:
            continue
        z, mu, disp = zscores(cm, g)
        a, b = cm.thr["A"], cm.thr["B"]
        tier = np.where((z > a["hi"]) | (z < -a["lo"]), "A",
                        np.where((z > b["hi"]) | (z < -b["lo"]), "B", "IN_RANGE"))
        excess = np.where(tier == "A",
                          np.where(z > 0, z - a["hi"], -z - a["lo"]), np.nan)
        out.loc[g.index, "expected_perf"] = mu
        out.loc[g.index, "expected_disp"] = disp
        out.loc[g.index, "z"] = z
        out.loc[g.index, "tier"] = tier
        out.loc[g.index, "excess"] = excess
        out.loc[g.index, "tags"] = _tags(g, cm.voltyp.lookup(g))
    return out


def drift(cells: dict, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bench, cm in cells.items():
        g = df[df[s.BENCHMARK] == bench]
        for f, fit_med in cm.medians.items():
            mm = float(g[f].median()) if f in g and g[f].notna().any() else np.nan
            pct = (100.0 * (mm - fit_med) / abs(fit_med)
                   if np.isfinite(mm) and fit_med else np.nan)
            rows.append({"benchmark": bench, "feature": f, "fit_median": fit_med,
                         "month_median": mm, "pct_change": pct,
                         "warn": bool(np.isfinite(pct)
                                      and abs(pct) > config.DRIFT_THRESHOLD_PCT)})
    return pd.DataFrame(rows, columns=["benchmark", "feature", "fit_median",
                                       "month_median", "pct_change", "warn"])
