"""Readable summaries. The fit summary is the model card's core."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cept.features import feature_names
from cept.fit import predicted_count


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) if len(df) else "_none_"


def _cells_table(cells: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "group": b, "n": cm.n,
        "tail_df_lo": round(cm.tail_lo.df, 1), "tail_df_hi": round(cm.tail_hi.df, 1),
        "Tier A lo/hi": f"{cm.thr['A']['lo']:.2f} / {cm.thr['A']['hi']:.2f}",
        "Tier B lo/hi": f"{cm.thr['B']['lo']:.2f} / {cm.thr['B']['hi']:.2f}",
        "expected A per 1,000 orders": round(predicted_count(cm, 1000, "A"), 3),
    } for b, cm in cells.items()])


def _coef_table(cm) -> pd.DataFrame:
    return pd.DataFrame({"condition": feature_names(cm.include_surprise, cm.strategies),
                         "centre": cm.ls.loc_coef.round(4),
                         "log dispersion": cm.ls.sc_coef.round(4)})


NEAR_NORMAL_DF = 15.0


def _threshold_table(cells: dict) -> pd.DataFrame:
    rows = []
    for g, cm in cells.items():
        a, b = cm.thr["A"], cm.thr["B"]
        rows.append({
            "strategy group": g,
            "strategies": ", ".join(cm.strategies) or "-",
            "OUTLIER if below expected by more than": f"{a['lo']:.2f} sigma",
            "OUTLIER if above expected by more than": f"{a['hi']:.2f} sigma",
            "watch list (below / above)": f"{b['lo']:.2f} / {b['hi']:.2f} sigma",
            "expected outliers per month (clean book)":
                round(predicted_count(cm, cm.n / 12.0, "A"), 2),
        })
    return pd.DataFrame(rows)


def _spread_examples(scored: pd.DataFrame) -> pd.DataFrame:
    """The band in spreads for an easy, a typical and a hard order."""
    rows = []
    sc = scored.dropna(subset=["expected_disp"])
    by = "algo" if "algo" in sc else "group"
    for strat, g in sc.groupby(by):
        d = g["expected_disp"].to_numpy()
        for label, q in (("easy order (10th pct)", 0.10),
                         ("typical order (median)", 0.50),
                         ("hard order (90th pct)", 0.90)):
            i = g.index[int(np.abs(d - np.quantile(d, q)).argmin())]
            r = g.loc[i]
            rows.append({"strategy": strat, "order": label,
                         "sigma (spreads)": round(r["expected_disp"], 2),
                         "expected perf (spreads)": round(r["expected_perf"], 2),
                         "OUTLIER below (spreads)": round(r["band_lo"], 2),
                         "OUTLIER above (spreads)": round(r["band_hi"], 2)})
    return pd.DataFrame(rows)


def _checks(cells, lomo_df, power_df) -> list[str]:
    out = []
    for g, cm in cells.items():
        dfs = (cm.tail_lo.df, cm.tail_hi.df)
        if min(dfs) >= NEAR_NORMAL_DF:
            verdict = ("near-normal, so the threshold is 4 sigma. A value "
                       "slightly above 4.00 means the fitted tail is "
                       "marginally heavier than a perfect bell curve, so 4 "
                       "sigma alone would be a little more frequent than 1 "
                       "in 10,000.")
        else:
            verdict = ("FATTER than normal: the threshold was widened above 4 "
                       "sigma so that 1-in-10,000 stays true. Check the "
                       "detection test below.")
        out.append(f"- **{g} tails** (df low {dfs[0]:.0f} / high {dfs[1]:.0f}): "
                   f"{verdict}")
        names = feature_names(cm.include_surprise, cm.strategies)
        for j, n in enumerate(names):
            if n.startswith("strategy_"):
                out.append(
                    f"- **{g} pooling check**: {n[9:]} vs {cm.strategies[0]} "
                    f"under the same conditions: expected perf differs by "
                    f"{cm.ls.loc_coef[j]:+.3f} spreads, sigma by a factor "
                    f"{float(np.exp(cm.ls.sc_coef[j])):.2f}. Near 0 and 1 means "
                    f"the strategies behave alike; either way the model "
                    f"accounts for it and the threshold stays single.")
    if lomo_df is not None and len(lomo_df):
        out.append(f"- **Backtest** (each month scored by a model fitted on "
                   f"the other 11): predicted {lomo_df['pred_A'].sum():.1f} "
                   f"outliers over the year, realised "
                   f"{int(lomo_df['real_A'].sum())}. Realised includes genuinely "
                   f"bad orders, so it can exceed predicted.")
    if power_df is not None and len(power_df):
        lo = power_df[power_df["side"] == "lo"].groupby("size")["detected_A"].mean()
        out.append("- **Detection test** (planted underperformance caught as "
                   "outlier): " + ", ".join(f"{int(k)} sigma: {v:.0%}"
                                            for k, v in lo.items()))
    return out


def write_fit_summary(path, cells, meta, lomo_df, power_df, sens_df,
                      clean_rep, dq_counts, scored=None) -> None:
    pol = meta.get("policy", {})
    lines = [
        "# CEPT model card", "",
        f"Market {meta.get('market')}, fitted on {meta.get('fit_start')} .. "
        f"{meta.get('fit_end')}, {meta.get('n_orders'):,} orders.", "",
        "## 1. The threshold", "",
        "**Rule, identical for every order in a strategy group:** an order is "
        "an OUTLIER when its performance (`ePvwap/Sprd`) is further from the "
        "performance expected *for that order's conditions* than the number of "
        "sigma below, where sigma is the normal variation *for that order's "
        "conditions*. It is mean +/- 4 sigma, with the mean and the sigma "
        "computed per order instead of once for the whole book.", "",
        _md(_threshold_table(cells)), "",
        f"Policy, fixed in advance: never closer than {pol.get('FLOOR_A')} "
        f"sigma, and rarer than 1 in {1 / float(pol.get('RARITY_A', 1e-4)):,.0f} "
        f"for a normally executed order. Watch list: {pol.get('FLOOR_B')} "
        f"sigma / 1 in {1 / float(pol.get('RARITY_B', 1e-3)):,.0f}.", ""]
    if scored is not None and len(scored):
        lines += [
            "### What that means in spreads", "",
            "Each order gets its own band. Three real orders from the fit "
            "year, chosen as easy, typical and hard:", "",
            _md(_spread_examples(scored)), "",
            "Every order's own band is in `scored_orders.csv`: columns "
            "`band_lo` / `band_hi` (outlier) and `watch_lo` / `watch_hi`.", ""]
    lines += ["## 2. Can it be trusted? (checks run at every fit)", ""]
    lines += _checks(cells, lomo_df, power_df) + [""]
    lines += ["---", "", "## Appendix (detail for model validation)", "",
              f"Policy rationale: {pol.get('POLICY_RATIONALE', '')}", "",
              "### A1. Thresholds and tails", "", _md(_cells_table(cells)), ""]
    for g, cm in cells.items():
        lines += [f"### A2. {g}: model coefficients", "", _md(_coef_table(cm)), ""]
    lines += ["### A3. Backtest by month (predicted vs realised)", "",
              _md(lomo_df), "",
              "### A4. Detection test", "", _md(power_df), "",
              "### A5. Same-day volatility sensitivity", "",
              f"Orders that would be outliers without the day-surprise term: "
              f"{len(sens_df)}", "", _md(sens_df), "",
              "### A6. Data quality and cleaning", "",
              _md(pd.DataFrame(list(dq_counts.items()),
                               columns=["rule", "rows"])), "",
              "```", clean_rep.summary(), "```", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_score_summary(path, meta, scored, cohort_df, bt_df, drift_df,
                        dq_counts) -> None:
    a = scored[scored["tier"] == "A"].sort_values("excess", ascending=False)
    b = scored[scored["tier"] == "B"]
    cols = [c for c in ["order_id", "order_date", "group", "symbol",
                        "perf_in_spreads", "expected_perf", "band_lo",
                        "band_hi", "z", "excess", "tags"] if c in scored]
    alerts = cohort_df[cohort_df["alert"]] if len(cohort_df) else cohort_df
    lines = ["# CEPT review summary", "",
             f"- model fitted {meta.get('fit_start')} .. {meta.get('fit_end')}",
             f"- orders scored: {len(scored):,}",
             f"- **Tier A (must review): {len(a)}**; Tier B (watch list): {len(b)}",
             f"- cohort alerts: {len(alerts)}",
             f"- data exceptions: {sum(dq_counts.values())}", "",
             "## Tier A outliers", "",
             "Each order was outside its own band: `perf_in_spreads` below "
             "`band_lo` or above `band_hi` (all in spreads). `z` is the "
             "distance from expected in sigma; `excess` is how far beyond the "
             "threshold.", "", _md(a[cols]), "",
             "## Cohort alerts", "", _md(alerts), "",
             "## Backtest this month", "", _md(bt_df), "",
             "## Drift vs fit window", "", _md(drift_df), "",
             "## Data exceptions by rule", "",
             _md(pd.DataFrame(list(dq_counts.items()), columns=["rule", "rows"])),
             ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
