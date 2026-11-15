"""Head-to-head: legacy PerfThreshold vs CEPT, through their real CLIs.

For each synthetic scenario and seed: write the fit year and score month as
CSVs (ground-truth columns removed), run `legacy/run.py fit/score` and
`python -m cept fit/score`, then compare each method's alert list with the
hidden truth.

    python -m lab.compare_legacy [--seeds 2] [--only S2_mixture,...]

false = clean orders flagged; caught = share of planted bad orders flagged;
dq_flagged = planted data errors that reached the review queue as if they
were performance alerts.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import pandas as pd

from lab import scenarios

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def _run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode not in (0,):
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stdout}\n{r.stderr}")


def _ids(path):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(df["order_id"].astype(str)) if len(df) else set()


def one(sc, seed, work):
    year, month = scenarios.make(sc, seed=seed)
    hidden = [c for c in year.columns if c.startswith("_") or c == "truth"]
    yp, mp = os.path.join(work, "year.csv"), os.path.join(work, "month.csv")
    year.drop(columns=hidden).to_csv(yp, index=False)
    month.drop(columns=hidden).to_csv(mp, index=False)
    truth = month.set_index("aggrTgtId")["truth"]

    lf, ls = os.path.join(work, "lfit"), os.path.join(work, "lscore")
    _run([PY, "legacy/run.py", "fit", "--csv", yp, "--out", lf, "--no-plots"])
    _run([PY, "legacy/run.py", "score", "--csv", mp, "--bands",
          os.path.join(lf, "bands.json"), "--out", ls, "--no-plots"])
    cf, cs = os.path.join(work, "cfit"), os.path.join(work, "cscore")
    # Thin books (S10) are below CEPT's production MIN_CELL_N of 5000; use
    # the legacy default of 2000 so both methods fit the same cells.
    min_n = ["--min-cell-n", "2000"] if sc == "S10_thin_book" else []
    _run([PY, "-m", "cept", "fit", "--csv", yp, "--out", cf, "--no-lomo"] + min_n)
    _run([PY, "-m", "cept", "score", "--csv", mp, "--model",
          os.path.join(cf, "model.json"), "--out", cs])

    sets = {"legacy": _ids(os.path.join(ls, "outliers.csv")),
            "cept_A": _ids(os.path.join(cs, "alerts_tierA.csv")),
            "cept_B": _ids(os.path.join(cs, "watchlist_tierB.csv"))}
    n_bad = int((truth == "bad").sum())
    rows = []
    for m, ids in sets.items():
        t = truth.reindex(list(ids))
        rows.append({"scenario": sc, "seed": seed, "method": m,
                     "flagged": len(ids), "false": int((t == "clean").sum()),
                     "caught": (int((t == "bad").sum()) / n_bad) if n_bad else float("nan"),
                     "dq_flagged": int((t == "dq").sum())})
    coh = os.path.join(cs, "cohort.csv")
    n_coh = 0
    if os.path.exists(coh):
        c = pd.read_csv(coh)
        n_coh = int(c["alert"].sum()) if "alert" in c and len(c) else 0
    rows.append({"scenario": sc, "seed": seed, "method": "cept_cohort",
                 "flagged": n_coh})
    return rows


def _job(arg):
    sc, seed = arg
    with tempfile.TemporaryDirectory() as w:
        return one(sc, seed, w)


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--only", default=",".join(scenarios.SCENARIOS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--raw", default=os.path.join(ROOT, "lab", "results",
                                                  "compare_legacy_raw.csv"))
    a = ap.parse_args()
    done = pd.read_csv(a.raw) if os.path.exists(a.raw) else pd.DataFrame(
        columns=["scenario", "seed"])
    have = set(zip(done["scenario"], done["seed"]))
    todo = [(sc, s) for sc in a.only.split(",") for s in range(a.seeds)
            if (sc, s) not in have]
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_job, t): t for t in todo}
        for f in as_completed(futs):
            sc, seed = futs[f]
            try:
                part = pd.DataFrame(f.result())
            except Exception as e:  # keep the others going; report the failure
                print(f"FAILED {sc} seed {seed}: {e}", file=sys.stderr, flush=True)
                continue
            part.to_csv(a.raw, mode="a", index=False,
                        header=not os.path.exists(a.raw))
            print(f"done {sc} seed {seed}", file=sys.stderr, flush=True)
    df = pd.read_csv(a.raw)
    df = df[df["scenario"].isin(a.only.split(","))]
    out = df.groupby(["scenario", "method"])[
        ["flagged", "false", "caught", "dq_flagged"]].mean().round(2)
    pd.set_option("display.width", 200)
    print(out.to_string())


if __name__ == "__main__":
    main()
