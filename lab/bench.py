"""Score every method on every scenario, several seeds, against ground truth.

    python -m lab.bench [--seeds 5] [--n 2500] [--only S2_mixture,...]

Per (scenario, method), averaged over seeds, on the held-out score month:
    false_alerts   flags on clean orders          (what you must justify)
    bad_caught     planted bad orders flagged / planted (6 per month)
    dq_to_queue    data errors that reached the EXECUTION queue
    dq_caught      data errors routed to the data-exception queue
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from lab import methods, scenarios

warnings.filterwarnings("ignore")


def run(seeds=5, n=2500, only=None):
    rows = []
    names = only or list(scenarios.SCENARIOS)
    for sc in names:
        for seed in range(seeds):
            year, month = scenarios.make(sc, seed=seed, n_per_month=n)
            truth = month["truth"]
            cm = methods.CondRobust().fit(year)
            hits = methods.cohort_test(cm, year, month)
            rows.append({"scenario": sc, "method": "COHORT_test", "seed": seed,
                         "false_alerts": len(hits),
                         "cohorts": ";".join(f"{h.strategy}:{h.feature}{h.bucket}"
                                             for h in hits.itertuples())
                         if len(hits) else ""})
            for make in methods.ALL:
                m = make()
                m.fit(year)
                flags, _, dq = m.score(month)
                rows.append({
                    "scenario": sc, "method": m.name, "seed": seed,
                    "false_alerts": int((flags & (truth == "clean")).sum()),
                    "bad_caught": int((flags & (truth == "bad")).sum())
                    / max(1, int((truth == "bad").sum())),
                    "drift_flagged": int((flags & (truth == "drift")).sum()),
                    "dq_to_queue": int((flags & (truth == "dq")).sum()),
                    "dq_caught": int((dq & (truth == "dq")).sum()),
                    "dq_false": int((dq & (truth != "dq")).sum()),
                })
    return pd.DataFrame(rows)


def summarise(df):
    g = df.groupby(["scenario", "method"], sort=False)
    out = g.agg(false_mean=("false_alerts", "mean"),
                false_max=("false_alerts", "max"),
                bad_caught=("bad_caught", "mean"),
                drift_flagged=("drift_flagged", "mean"),
                dq_to_queue=("dq_to_queue", "mean"),
                dq_caught=("dq_caught", "mean"),
                dq_false=("dq_false", "mean"),
                cohorts=("cohorts", lambda c: " | ".join(sorted({x for x in c.dropna() if x}))[:90]))
    return out.round(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    res = run(a.seeds, a.n, a.only.split(",") if a.only else None)
    pd.set_option("display.width", 200)
    print(summarise(res).to_string())
