"""Which alerts does the same-day volatility term clear?

Same-day Parkinson vol is partly caused by the order itself. This refits the
model WITHOUT the day-surprise term and lists every order that would then be
Tier A but is not under the production model -- with the large-vs-ADV ones
marked, so it is visible whether any order excused itself.
"""

from __future__ import annotations

import pandas as pd

from cept import config, schema as s
from cept.fit import fit_all
from cept.score import score_orders

COLS = ["order_id", "group", "perf_in_spreads", "z_with", "z_without",
        "vol_surprise", "pct_adv", "high_adv"]


def vol_sensitivity(fit_df: pd.DataFrame, target_df: pd.DataFrame,
                    cells: dict) -> pd.DataFrame:
    without, _ = fit_all(fit_df, include_surprise=False, min_cell_n=1)
    a = score_orders(cells, target_df)
    b = score_orders({k: without[k] for k in cells if k in without}, target_df)
    hit = (b["tier"] == "A") & (a["tier"] != "A")
    rows = []
    for i in target_df.index[hit.to_numpy()]:
        cm = cells[target_df.loc[i, s.GROUP]]
        vt = cm.voltyp.lookup(target_df.loc[[i]])[0]
        adv = float(target_df.loc[i, s.PCT_ADV])
        rows.append({"order_id": target_df.loc[i, s.ORDER_ID],
                     "group": target_df.loc[i, s.GROUP],
                     "perf_in_spreads": float(target_df.loc[i, s.METRIC]),
                     "z_with": float(a.loc[i, "z"]),
                     "z_without": float(b.loc[i, "z"]),
                     "vol_surprise": float(target_df.loc[i, s.VOLATILITY] / vt),
                     "pct_adv": adv, "high_adv": adv > config.HIGH_ADV_PCT})
    return pd.DataFrame(rows, columns=COLS)


def summary(df: pd.DataFrame) -> dict:
    return {"n_cleared_by_surprise": int(len(df)),
            "n_high_adv": int(df["high_adv"].sum()) if len(df) else 0}
