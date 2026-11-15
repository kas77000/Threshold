"""Systematic deterioration that no single order reveals.

Mean conditional z (winsorised at +/-4, so one outlier cannot drive a cohort)
per strategy and per quintile of each condition, quintile edges from the FIT
year, tested against zero. The model is frozen at fit, so execution that gets
worse afterwards -- even when correlated with a covariate -- shows up here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cept import config, schema as s

COLS = ["benchmark", "feature", "bucket", "n", "mean_z", "t", "alert"]


def cohort_test(cells: dict, scored: pd.DataFrame,
                t_crit: float = config.COHORT_T) -> pd.DataFrame:
    rows = []
    for bench, g in scored[scored["tier"] != "NO_MODEL"].groupby(s.BENCHMARK):
        cm = cells[str(bench)]
        z = np.clip(g["z"].to_numpy(float), -4.0, 4.0)
        groups = [("ALL", "all", np.ones(len(g), bool))]
        for f, edges in cm.cohort_edges.items():
            if f not in g:
                continue
            bins = np.digitize(g[f].to_numpy(float), edges)
            groups += [(f, int(i), bins == i) for i in np.unique(bins)]
        for f, b, mask in groups:
            n = int(mask.sum())
            if n < config.COHORT_MIN_N:
                continue
            zz = z[mask]
            t = float(zz.mean() / (zz.std(ddof=1) / np.sqrt(n)))
            rows.append({"benchmark": str(bench), "feature": f, "bucket": b,
                         "n": n, "mean_z": float(zz.mean()), "t": t,
                         "alert": abs(t) > t_crit})
    return pd.DataFrame(rows, columns=COLS)
