"""Is the stated rarity true? Predicted vs realised exceedances.

LOMO: each month of the fit year is scored by models fitted on the other
months -- out-of-sample, as in production. Kupiec's proportion-of-failures
likelihood ratio gives a p-value for realised vs stated rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from cept import config, schema as s
from cept.fit import fit_all, predicted_count
from cept.score import score_orders

COLS = ["group", "n", "pred_A", "real_A", "p_A", "pred_B", "real_B", "p_B"]


def kupiec_pvalue(n: int, x: int, p: float) -> float:
    if n <= 0:
        return float("nan")
    phat = x / n

    def ll(q):
        a = (n - x) * np.log1p(-q) if x < n else 0.0
        b = x * np.log(q) if x > 0 else 0.0
        return a + b
    lr = -2.0 * (ll(p) - (ll(phat) if 0 < phat < 1 else 0.0))
    return float(stats.chi2.sf(max(lr, 0.0), 1))


def month_table(cells: dict, scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bench, g in scored[scored["tier"] != "NO_MODEL"].groupby(s.GROUP):
        cm = cells[str(bench)]
        n = len(g)
        pa, pb = predicted_count(cm, n, "A"), predicted_count(cm, n, "B")
        ra = int((g["tier"] == "A").sum())
        rb = int(g["tier"].isin(["A", "B"]).sum())
        rows.append({"group": str(bench), "n": n,
                     "pred_A": pa, "real_A": ra,
                     "p_A": kupiec_pvalue(n, ra, pa / n),
                     "pred_B": pb, "real_B": rb,
                     "p_B": kupiec_pvalue(n, rb, pb / n)})
    return pd.DataFrame(rows, columns=COLS)


def lomo(df: pd.DataFrame, include_surprise: bool = True,
         min_cell_n: int = config.MIN_CELL_N) -> pd.DataFrame:
    month = df[s.ORDER_DATE].dt.to_period("M").astype(str)
    out = []
    for label in sorted(month.unique()):
        held = month == label
        cells, _ = fit_all(df[~held], include_surprise, min_cell_n)
        t = month_table(cells, score_orders(cells, df[held]))
        t.insert(0, "month", label)
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["month"] + COLS)
