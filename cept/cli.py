"""check / fit / score. Fit and score are separate commands with a file
between them: that separation is the leakage guard."""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from cept import (backtest, cohort, config, dq, fit, load, persist, plots,
                  power, report, schema as s, score, sensitivity)
from cept.features import feature_names


def _dq_counts(exc: pd.DataFrame) -> dict:
    return exc["dq_reason"].value_counts().to_dict() if len(exc) else {}


def cmd_check(a) -> int:
    raw = load.read_any(a.csv)
    print(f"file: {a.csv}   rows: {len(raw):,}")
    print("\n--- STRATEGY -> BENCHMARK ---")
    for k, n in raw["Strategy"].astype(str).str.strip().value_counts().items():
        print(f"  {k:<20}{n:>9,}  -> {config.benchmark_for(k) or '*** UNMAPPED ***'}")
    print("\n--- MARKETS ---")
    print(raw["Sym"].astype(str).str[-2:].str.upper().value_counts().to_string())
    df, rep = load.prepare(raw, market=a.market)
    print("\n--- LOADING ---\n" + rep.summary())
    clean, exc = dq.split(df)
    print("\n--- DATA QUALITY ---")
    print(pd.Series(_dq_counts(exc), dtype=int).to_string() if len(exc)
          else "  no data exceptions")
    print("\n--- CONDITIONS (clean rows) ---")
    cols = [c for c in s.CONDITIONS if c != s.SIDE] + [s.FILL_RATE]
    print(clean[[c for c in cols if c in clean]].describe(
        percentiles=[.01, .5, .99]).T.to_string())
    return 0


def cmd_fit(a) -> int:
    df, rep = load.load(a.csv, market=a.market)
    print(rep.summary())
    clean, exc = dq.split(df)
    cells, skipped = fit.fit_all(clean, min_cell_n=a.min_cell_n)
    if not cells:
        print(f"No cell has >= {a.min_cell_n} clean orders: {skipped}",
              file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    meta = {"market": a.market,
            "fit_start": str(clean[s.ORDER_DATE].min().date()),
            "fit_end": str(clean[s.ORDER_DATE].max().date()),
            "n_orders": int(len(clean)),
            "source_file": os.path.abspath(a.csv),
            "source_hash": persist.file_hash(a.csv), "skipped": skipped,
            "policy": persist.policy_snapshot()}
    persist.write_model(os.path.join(a.out, "model.json"), cells, meta)

    lomo_df = (pd.DataFrame() if a.no_lomo
               else backtest.lomo(clean, min_cell_n=a.min_cell_n))
    power_df = power.injected_detection(cells, clean)
    sens_df = sensitivity.vol_sensitivity(clean, clean, cells)
    scored = score.score_orders(cells, clean)
    coef = pd.concat([pd.DataFrame({"benchmark": b,
                                    "condition": feature_names(cm.include_surprise),
                                    "centre": cm.ls.loc_coef,
                                    "log_dispersion": cm.ls.sc_coef})
                      for b, cm in cells.items()])
    for name, frame in [("lomo_backtest.csv", lomo_df),
                        ("detection_power.csv", power_df),
                        ("vol_sensitivity.csv", sens_df),
                        ("coefficients.csv", coef),
                        ("data_exceptions.csv", exc),
                        ("cleaning_report.csv", rep.to_frame()),
                        ("in_sample_scored.csv", scored)]:
        frame.to_csv(os.path.join(a.out, name), index=False)
    for b, cm in cells.items():
        plots.qq_plot(scored.loc[scored[s.BENCHMARK] == b, "z"],
                      os.path.join(a.out, f"qq_{b}.png"), f"{b} conditional z")
        plots.coef_plot(cm, os.path.join(a.out, f"coef_{b}.png"))
    report.write_fit_summary(os.path.join(a.out, "summary.md"), cells, meta,
                             lomo_df, power_df, sens_df, rep, _dq_counts(exc))
    print(f"\nWrote {os.path.join(a.out, 'model.json')}")
    return 0


def cmd_score(a) -> int:
    cells, meta = persist.read_model(a.model)
    score.check_metric(meta)
    df, rep = load.load(a.csv, market=meta.get("market"))
    print(rep.summary())
    score.check_no_overlap(meta.get("fit_start"), meta.get("fit_end"), df)
    clean, exc = dq.split(df)
    scored = score.score_orders(cells, clean)
    coh = cohort.cohort_test(cells, scored)
    bt = backtest.month_table(cells, scored)
    dr = score.drift(cells, clean)
    os.makedirs(a.out, exist_ok=True)
    a_df = scored[scored["tier"] == "A"].sort_values("excess", ascending=False)
    for name, frame in [("alerts_tierA.csv", a_df),
                        ("watchlist_tierB.csv", scored[scored["tier"] == "B"]),
                        ("scored_orders.csv", scored), ("cohort.csv", coh),
                        ("backtest.csv", bt), ("drift.csv", dr),
                        ("data_exceptions.csv", exc)]:
        frame.to_csv(os.path.join(a.out, name), index=False)
    report.write_score_summary(os.path.join(a.out, "summary.md"), meta, scored,
                               coh, bt, dr, _dq_counts(exc))
    n_coh = int(coh["alert"].sum()) if len(coh) else 0
    print(f"\n  TIER A: {len(a_df)}   TIER B: {int((scored['tier'] == 'B').sum())}"
          f"   COHORT ALERTS: {n_coh}   DATA EXCEPTIONS: {len(exc)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cept")
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check")
    c.add_argument("--csv", required=True)
    c.add_argument("--market", default=config.DEFAULT_MARKET)
    c.set_defaults(func=cmd_check)
    f = sub.add_parser("fit")
    f.add_argument("--csv", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--market", default=config.DEFAULT_MARKET)
    f.add_argument("--min-cell-n", type=int, default=config.MIN_CELL_N,
                   dest="min_cell_n")
    f.add_argument("--no-lomo", action="store_true", dest="no_lomo")
    f.set_defaults(func=cmd_fit)
    sc = sub.add_parser("score")
    sc.add_argument("--csv", required=True)
    sc.add_argument("--model", required=True)
    sc.add_argument("--out", required=True)
    sc.set_defaults(func=cmd_score)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (score.GuardError, ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
