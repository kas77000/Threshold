"""Is the stated rarity TRUE, and what does each rarity cost in detection?

For M8 (robust conditional z + quantile-matched t tail, floor 4):
  - for each policy rarity p (one-sided), threshold per side = MAX(4, q_t(p))
  - predicted false alerts/month = expected clean exceedances under the fitted
    tail; realised = clean orders actually flagged in the held-out month
  - detection: orders made bad by exactly m TRUE conditional sigmas

    python -m lab.frontier [--seeds 5]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from lab import methods, scenarios
from lab.methods import METRIC, LocationScale, dq_mask, features, \
    t_tail_quantile_match

warnings.filterwarnings("ignore")
RARITIES = {"1/1k": 1e-3 / 2, "1/10k": 1e-4 / 2, "4sig(1/16k)": 3.17e-5,
            "1/100k": 1e-5 / 2}
SIZES = (5, 8, 12, 20)


def fit_cell(g):
    X, y = features(g), g[METRIC].to_numpy()
    ls = LocationScale(robust=True).fit(X, y)
    z = ls.z(X, y)
    med = np.median(z)
    mad = 1.4826 * np.median(np.abs(z - med))
    z = (z - med) / mad
    tails = {}
    for side, t in (("hi", z), ("lo", -z)):
        _, df_, sc = t_tail_quantile_match(t, 1e-4)
        tails[side] = (df_, sc)
    return ls, med, mad, tails


def zscore(c, g, y=None):
    ls, med, mad, _ = c
    y = g[METRIC].to_numpy() if y is None else y
    return (ls.z(features(g), y) - med) / mad


def run(scs, seeds):
    rows = []
    for sc in scs:
        for seed in range(seeds):
            year, month = scenarios.make(sc, seed=seed)
            year = year[~dq_mask(year)]
            month = month[~dq_mask(month)]
            for b, g in month.groupby("Strategy"):
                c = fit_cell(year[year.Strategy == b])
                clean = g[g.truth == "clean"]
                z = zscore(c, clean)
                for label, p in RARITIES.items():
                    thr = {s: max(4.0, stats.t.isf(p, *([c[3][s][0], 0,
                                                        c[3][s][1]])))
                           for s in ("hi", "lo")}
                    # predicted exceedance prob per side at that threshold
                    pred = sum(stats.t.sf(thr[s], c[3][s][0], 0, c[3][s][1])
                               for s in ("hi", "lo")) * len(clean)
                    real = int(((z > thr["hi"]) | (z < -thr["lo"])).sum())
                    det = {}
                    for m in SIZES:
                        y_bad = clean["_mu"] - m * clean["_sd"]
                        zb = zscore(c, clean, y_bad.to_numpy())
                        det[f"det_{m}sd"] = float((zb < -thr["lo"]).mean())
                    rows.append({"scenario": sc, "rarity": label, "seed": seed,
                                 "bench": b, "pred_false": pred,
                                 "real_false": real,
                                 "thr_lo": thr["lo"], **det})
    df = pd.DataFrame(rows)
    # per month = sum over the two benchmarks
    per = df.groupby(["scenario", "rarity", "seed"], sort=False).agg(
        pred_false=("pred_false", "sum"), real_false=("real_false", "sum"),
        thr_lo=("thr_lo", "mean"),
        **{f"det_{m}sd": (f"det_{m}sd", "mean") for m in SIZES})
    return per.groupby(["scenario", "rarity"], sort=False).mean().round(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--only", default="S2_mixture,S3_fat_residual,"
                    "S9_misspecified,S11_misspec_fat,S10_thin_book")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    print(run(a.only.split(","), a.seeds).to_string())
