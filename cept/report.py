"""Readable summaries. The fit summary is the model card's core."""

from __future__ import annotations

import pandas as pd

from cept.features import feature_names
from cept.fit import predicted_count


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) if len(df) else "_none_"


def _cells_table(cells: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "benchmark": b, "n": cm.n,
        "tail_df_lo": round(cm.tail_lo.df, 1), "tail_df_hi": round(cm.tail_hi.df, 1),
        "Tier A lo/hi": f"{cm.thr['A']['lo']:.2f} / {cm.thr['A']['hi']:.2f}",
        "Tier B lo/hi": f"{cm.thr['B']['lo']:.2f} / {cm.thr['B']['hi']:.2f}",
        "expected A per 1,000 orders": round(predicted_count(cm, 1000, "A"), 3),
    } for b, cm in cells.items()])


def _coef_table(cm) -> pd.DataFrame:
    return pd.DataFrame({"condition": feature_names(cm.include_surprise),
                         "centre": cm.ls.loc_coef.round(4),
                         "log dispersion": cm.ls.sc_coef.round(4)})


def write_fit_summary(path, cells, meta, lomo_df, power_df, sens_df,
                      clean_rep, dq_counts) -> None:
    pol = meta.get("policy", {})
    lines = ["# CEPT fit summary (model card)", "",
             f"- market: {meta.get('market')}, fit window "
             f"{meta.get('fit_start')} .. {meta.get('fit_end')}, "
             f"orders {meta.get('n_orders'):,}",
             f"- policy: Tier A rarity {pol.get('RARITY_A')}, floor "
             f"{pol.get('FLOOR_A')}; Tier B rarity {pol.get('RARITY_B')}, "
             f"floor {pol.get('FLOOR_B')}; cohort |t| > {pol.get('COHORT_T')}",
             f"- rationale: {pol.get('POLICY_RATIONALE', '')}", "",
             "## Cells and thresholds (in conditional sigma)", "",
             _md(_cells_table(cells)), "",
             "A tail df of 15 or more means conditional residuals are "
             "near-normal and Tier A is exactly 4 sigma.", ""]
    for b, cm in cells.items():
        lines += [f"## {b}: model coefficients", "", _md(_coef_table(cm)), ""]
    lines += ["## Backtest, leave-one-month-out (predicted vs realised)", "",
              _md(lomo_df), "",
              "## Injected detection (share of planted misses caught)", "",
              _md(power_df), "",
              "## Same-day volatility sensitivity", "",
              f"Orders that would be Tier A without the day-surprise term: "
              f"{len(sens_df)}", "", _md(sens_df), "",
              "## Data quality and cleaning", "",
              _md(pd.DataFrame(list(dq_counts.items()),
                               columns=["rule", "rows"])), "",
              "```", clean_rep.summary(), "```", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_score_summary(path, meta, scored, cohort_df, bt_df, drift_df,
                        dq_counts) -> None:
    a = scored[scored["tier"] == "A"].sort_values("excess", ascending=False)
    b = scored[scored["tier"] == "B"]
    cols = [c for c in ["order_id", "order_date", "benchmark", "symbol",
                        "perf_in_spreads", "expected_perf", "expected_disp",
                        "z", "excess", "tags"] if c in scored]
    alerts = cohort_df[cohort_df["alert"]] if len(cohort_df) else cohort_df
    lines = ["# CEPT review summary", "",
             f"- model fitted {meta.get('fit_start')} .. {meta.get('fit_end')}",
             f"- orders scored: {len(scored):,}",
             f"- **Tier A (must review): {len(a)}**; Tier B (watch list): {len(b)}",
             f"- cohort alerts: {len(alerts)}",
             f"- data exceptions: {sum(dq_counts.values())}", "",
             "## Tier A", "", _md(a[cols]), "",
             "## Cohort alerts", "", _md(alerts), "",
             "## Backtest this month", "", _md(bt_df), "",
             "## Drift vs fit window", "", _md(drift_df), "",
             "## Data exceptions by rule", "",
             _md(pd.DataFrame(list(dq_counts.items()), columns=["rule", "rows"])),
             ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
