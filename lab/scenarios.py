"""Synthetic HK TWAP/VWAP books with ground truth, for comparing methods.

Every order carries a hidden `truth` label:
    clean   ordinary execution, generated from the conditional model
    bad     genuinely abnormal execution for its conditions (planted)
    dq      a data error (planted)

The generator is a location-scale model on order conditions. That is the
claim being tested: much of the "fat tail" of a pooled book is a mixture of
orders with very different, perfectly ordinary, dispersion. Scenarios then
break that claim in specific ways (genuine fat tails, regime shifts, a strategy
that really deteriorates) so the methods are judged on the cases where they
could fail, not only on the case they were designed for.

Column names are the raw extract's, so the lab exercises the real contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HK_MINUTES = 330.0
# Daily-fraction equivalent of 25% annualised: the reference volatility.
VOL_REF = 0.25 / np.sqrt(252.0)
N_SYMBOLS = 300

# Per-benchmark behaviour. TWAP ignores the volume curve, so it tracks
# interval VWAP worse (higher tracking scale) and pays more when the close is
# a big share of the day.
BENCH = {
    "VWAP": {"mu0": -0.05, "impact": 0.10, "close_pen": 0.20, "track": 0.10},
    "TWAP": {"mu0": -0.12, "impact": 0.14, "close_pen": 0.35, "track": 0.14},
}


def _conditions(n: int, rng: np.random.Generator, vol_mult: float = 1.0,
                spread_mult: float = 1.0) -> pd.DataFrame:
    """Order conditions, driven by a latent liquidity factor.

    Small, illiquid names have wider spreads, higher volatility and larger
    orders as a share of ADV -- the correlation structure that makes a pooled
    distribution look fat-tailed.
    """
    illiq = rng.normal(0.0, 1.0, n)
    e = rng.normal(0.0, 1.0, (n, 4))
    spread = np.clip(8.0 * np.exp(0.55 * illiq + 0.25 * e[:, 0]), 1.0, 200.0)
    vol = np.clip(25.0 * np.exp(0.25 * illiq + 0.25 * e[:, 1]), 8.0, 150.0)
    pct_adv = np.clip(1.5 * np.exp(0.6 * illiq + 0.9 * e[:, 2]), 0.01, 60.0)
    dur = np.clip(90.0 * np.exp(0.8 * e[:, 3]), 5.0, HK_MINUTES)
    pr = np.clip(pct_adv * (HK_MINUTES / dur) * rng.uniform(0.6, 1.4, n),
                 0.1, 60.0)
    open_pct = np.where(rng.random(n) < 0.15, rng.beta(2, 12, n) * 100, 0.0)
    close_pct = np.where(rng.random(n) < 0.35,
                         rng.beta(2, 5, n) * (100.0 - open_pct), 0.0)
    return pd.DataFrame({
        "Sprd": spread * spread_mult,
        "Vol": vol * vol_mult / 100.0 / np.sqrt(252.0),
        "%Adv": pct_adv, "Dur": dur, "PR": pr,
        "%OPEN": open_pct, "%CLOSE": close_pct,
        "Side": np.where(rng.random(n) < 0.5, "Buy", "Sell"),
        "_illiq": illiq,
    })


def true_location_scale(c: pd.DataFrame, bench: str
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Expected performance and dispersion, IN SPREADS, for these conditions.

    Dispersion is the price risk over the order's life, a fraction of which
    leaks into VWAP-relative performance, divided by the spread. That ratio is
    what varies by an order of magnitude across a book.
    """
    p = BENCH[bench]
    move_bps = c["Vol"] * 1e4 * np.sqrt(c["Dur"] / HK_MINUTES)
    sd_bps = (p["track"] * move_bps * (1.0 + 0.5 * np.sqrt(c["PR"] / 10.0))
              + 1.0)
    sd = sd_bps / c["Sprd"]
    mu = (p["mu0"] - p["impact"] * np.sqrt(c["PR"] / 10.0)
          - p["close_pen"] * c["%CLOSE"] / 100.0)
    return mu.to_numpy(), sd.to_numpy()


def _noise(n, rng, tail_df):
    if tail_df is None:
        return rng.normal(0.0, 1.0, n)
    # Unit-variance Student-t, so tail_df changes the tail, not the scale.
    return rng.standard_t(tail_df, n) * np.sqrt((tail_df - 2.0) / tail_df)


def _month(n, start, rng, *, tail_df=None, homoskedastic=False, unmodelled=False,
           vol_mult=1.0, spread_mult=1.0, bad_n=0, dq_n=0,
           shift=None, catastrophic=0, id0=0):
    rows = []
    for bench in BENCH:
        m = n // 2
        c = _conditions(m, rng, vol_mult, spread_mult)
        mu, sd = true_location_scale(c, bench)
        if homoskedastic:
            mu = np.full(m, BENCH[bench]["mu0"])
            sd = np.full(m, 0.6)
        if unmodelled:
            # What the extract cannot see: news / event days (4% of orders at
            # 2.5x dispersion), a convex volatility effect and a vol x close
            # interaction the linear model does not contain.
            event = rng.random(m) < 0.04
            sd = sd * np.where(event, 2.5, 1.0)
            sd = sd * (c["Vol"].to_numpy() / VOL_REF) ** 0.4
            sd = sd * (1.0 + 0.8 * (c["%CLOSE"].to_numpy() / 100.0)
                       * (c["Vol"].to_numpy() / VOL_REF))
            mu = mu - 0.15 * (c["Vol"].to_numpy() / VOL_REF) ** 2
        eps = _noise(m, rng, tail_df)
        perf = mu + sd * eps
        truth = np.array(["clean"] * m, dtype=object)
        if shift and shift.get("bench") == bench:
            hit = rng.random(m) < shift["frac"]
            perf[hit] -= shift["size"] * sd[hit]
            if shift.get("label_bad"):
                truth[hit] = "drift"
        c["Strategy"] = bench
        c["ePvwap/Sprd"] = perf
        c["_sd"], c["_mu"] = sd, mu
        c["truth"] = truth
        rows.append(c)
    df = pd.concat(rows, ignore_index=True)
    k = len(df)

    # Genuinely bad execution: abnormal FOR ITS CONDITIONS, mostly a miss.
    if bad_n:
        idx = rng.choice(np.flatnonzero(df["truth"] == "clean"), bad_n,
                         replace=False)
        sign = np.where(rng.random(bad_n) < 0.8, -1.0, 1.0)
        df.loc[idx, "ePvwap/Sprd"] = (df.loc[idx, "_mu"] + sign
                                      * rng.uniform(4.5, 7.0, bad_n)
                                      * df.loc[idx, "_sd"])
        df.loc[idx, "truth"] = "bad"

    if catastrophic:
        idx = rng.choice(np.flatnonzero(df["truth"] == "clean"), catastrophic,
                         replace=False)
        df.loc[idx, "ePvwap/Sprd"] = -rng.uniform(40.0, 120.0, catastrophic)
        df.loc[idx, "truth"] = "bad"

    df["Pvwap"] = df["ePvwap/Sprd"] * df["Sprd"]

    # Data errors, of the kinds real extracts carry.
    if dq_n:
        idx = rng.choice(np.flatnonzero(df["truth"] == "clean"), dq_n,
                         replace=False)
        kinds = rng.integers(0, 4, dq_n)
        for i, kind in zip(idx, kinds):
            if kind == 0:     # spread in wrong units -> ratio x100
                df.loc[i, "Sprd"] /= 100.0
                df.loc[i, "ePvwap/Sprd"] *= 100.0
                df.loc[i, "Pvwap"] = df.loc[i, "ePvwap/Sprd"] * df.loc[i, "Sprd"]
            elif kind == 1:   # metric from another benchmark: inconsistent
                df.loc[i, "ePvwap/Sprd"] = -df.loc[i, "ePvwap/Sprd"] * 8 - 5
            elif kind == 2:   # impossible participation
                df.loc[i, "PR"] = 150.0 + rng.uniform(0, 200)
                df.loc[i, "ePvwap/Sprd"] -= 12 * df.loc[i, "_sd"]
                df.loc[i, "Pvwap"] = df.loc[i, "ePvwap/Sprd"] * df.loc[i, "Sprd"]
            else:             # zero / negative duration
                df.loc[i, "Dur"] = 0.0
                df.loc[i, "ePvwap/Sprd"] += 15 * df.loc[i, "_sd"]
                df.loc[i, "Pvwap"] = df.loc[i, "ePvwap/Sprd"] * df.loc[i, "Sprd"]
            df.loc[i, "truth"] = "dq"

    start = pd.Timestamp(start)
    df["Date"] = start + pd.to_timedelta(
        rng.integers(0, start.days_in_month, k), unit="D")
    df["aggrTgtId"] = [f"O{id0 + i:08d}" for i in range(k)]
    sym_id = np.clip(((df["_illiq"] + 3.0) / 6.0 * N_SYMBOLS).astype(int),
                     0, N_SYMBOLS - 1) + 1
    df["Sym"] = [f"{s:04d} HK" for s in sym_id]
    df["FR"] = np.where(rng.random(k) < 0.88, 100.0,
                        rng.uniform(20.0, 99.0, k))
    df["#Shares"] = np.round(rng.lognormal(10.0, 1.0, k))
    return df


SCENARIOS = {
    "S1_clean_normal": "homoskedastic Gaussian book; nothing to explain",
    "S2_mixture": "heteroskedastic mix of ordinary orders (the main case)",
    "S3_fat_residual": "mixture AND genuinely fat conditional tails (t4)",
    "S4_data_errors": "S2 plus data errors in fit year and score month",
    "S5_volatile_month": "S2, score month 1.7x volatility (market got harder)",
    "S6_strategy_decay": "S2, TWAP degrades in score month (we got worse)",
    "S7_catastrophic": "S2, catastrophic orders in the fit year (masking)",
    "S8_feature_decay": "S2, score month: high-PR orders systematically worse",
    "S9_misspecified": "dispersion the columns cannot explain + nonlinearity",
    "S10_thin_book": "S2 with 600 orders/month instead of 2500",
    "S11_misspec_fat": "S9 and t5 residuals: the hardest realistic case",
}


def make(scenario: str, seed: int = 0, n_per_month: int = 2500
         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(fit_year, score_month) for one scenario."""
    rng = np.random.default_rng(seed)
    base = dict(tail_df=None, homoskedastic=False)
    if scenario == "S1_clean_normal":
        base["homoskedastic"] = True
    if scenario == "S3_fat_residual":
        base["tail_df"] = 4.0
    if scenario in ("S9_misspecified", "S11_misspec_fat"):
        base["unmodelled"] = True
    if scenario == "S11_misspec_fat":
        base["tail_df"] = 5.0
    if scenario == "S10_thin_book":
        n_per_month = 600

    # The fit year itself is not clean: real years contain bad orders.
    year_bad = max(1, n_per_month // 1000)
    months = []
    for m in range(12):
        start = pd.Timestamp("2025-07-01") + pd.DateOffset(months=m)
        months.append(_month(
            n_per_month, start, rng, bad_n=year_bad,
            dq_n=(3 if scenario == "S4_data_errors" else 0),
            catastrophic=(1 if scenario == "S7_catastrophic" and m % 2 == 0
                          else 0),
            id0=m * 100_000, **base))
    year = pd.concat(months, ignore_index=True)

    kw = dict(base)
    kw.update(bad_n=6, id0=9_000_000)
    if scenario == "S4_data_errors":
        kw["dq_n"] = 6
    if scenario == "S5_volatile_month":
        kw.update(vol_mult=1.7, spread_mult=1.2)
    if scenario == "S6_strategy_decay":
        kw["shift"] = {"bench": "TWAP", "frac": 1.0, "size": 0.6,
                       "label_bad": False}
    month = _month(n_per_month, pd.Timestamp("2026-07-01"), rng, **kw)
    if scenario == "S8_feature_decay":
        hi = month["PR"] > month["PR"].quantile(0.9)
        month.loc[hi, "ePvwap/Sprd"] -= 2.0 * month.loc[hi, "_sd"]
        month.loc[hi, "Pvwap"] = month.loc[hi, "ePvwap/Sprd"] * month.loc[hi, "Sprd"]
        month.loc[hi & (month["truth"] == "clean"), "truth"] = "drift"
    return year, month
