"""Systematic deterioration that no single order reveals.

Cohorts: each strategy in the group, and each quintile of each condition
(quintile edges from the FIT year). For each cohort, the mean conditional z
(winsorised at +/-4, so one outlier cannot drive it) is compared with the SAME
cohort's mean z in the fit year. The question is "has this cohort got worse
than it was", not "is it different from the group": strategies pooled in one
group may legitimately sit slightly apart, and that is not an alert.

The model is frozen at fit, so execution that gets worse afterwards -- even
when correlated with a covariate -- shows up here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cept import config, schema as s

COLS = ["group", "feature", "bucket", "n", "mean_z", "baseline_mean_z", "t",
        "alert"]
WINSOR = 4.0


def memberships(edges: dict, g: pd.DataFrame):
    """(feature, bucket, mask) for every cohort of these rows."""
    out = [("ALL", "all", np.ones(len(g), bool))]
    if s.ALGO in g:
        algo = g[s.ALGO].astype(str).str.strip().str.upper().to_numpy()
        out += [("strategy", a, algo == a) for a in sorted(set(algo))]
    for f, e in edges.items():
        if f not in g:
            continue
        bins = np.digitize(g[f].to_numpy(float), e)
        out += [(f, str(int(i)), bins == i) for i in np.unique(bins)]
    return out


def _stats(z: np.ndarray) -> dict:
    return {"mean": float(z.mean()), "sd": float(z.std(ddof=1)), "n": int(len(z))}


def baselines(edges: dict, g: pd.DataFrame, z: np.ndarray) -> dict:
    """Fit-year mean z per cohort, stored in the model file."""
    z = np.clip(np.asarray(z, float), -WINSOR, WINSOR)
    return {f"{f}|{b}": _stats(z[m]) for f, b, m in memberships(edges, g)
            if m.sum() >= 2}


def cohort_test(cells: dict, scored: pd.DataFrame,
                t_crit: float = config.COHORT_T) -> pd.DataFrame:
    rows = []
    for grp, g in scored[scored["tier"] != "NO_MODEL"].groupby(s.GROUP):
        cm = cells[str(grp)]
        z = np.clip(g["z"].to_numpy(float), -WINSOR, WINSOR)
        for f, b, mask in memberships(cm.cohort_edges, g):
            n = int(mask.sum())
            if n < config.COHORT_MIN_N:
                continue
            zz = z[mask]
            base = cm.cohort_base.get(f"{f}|{b}")
            var = zz.var(ddof=1) / n
            if base and base["n"] >= 2:
                mb = base["mean"]
                var += base["sd"] ** 2 / base["n"]
            else:               # a cohort absent from the fit year: vs zero
                mb = 0.0
            t = float((zz.mean() - mb) / np.sqrt(var))
            rows.append({"group": str(grp), "feature": f, "bucket": b,
                         "n": n, "mean_z": float(zz.mean()),
                         "baseline_mean_z": mb, "t": t,
                         "alert": abs(t) > t_crit})
    return pd.DataFrame(rows, columns=COLS)
