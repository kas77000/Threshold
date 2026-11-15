"""Candidate outlier methods, all behind one interface.

    m = Method(); m.fit(year_df); flags, score = m.score(month_df)

Each method is fitted per benchmark (the cell, HK only). A data-quality layer
is applied before any statistical method, except where noted, so that broken
records go to a data-exception queue instead of the execution queue.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import HuberRegressor, LinearRegression

METRIC = "ePvwap/Sprd"
P_4SIGMA_ONE_SIDE = 0.5 * math.erfc(4.0 / math.sqrt(2.0))   # 3.17e-5

# --------------------------------------------------------------------------
# Data quality: rules about the RECORD, never about the performance value.
# --------------------------------------------------------------------------
DQ_RULES = {
    "spread_implausible": lambda d: ~d["Sprd"].between(0.5, 500.0),
    "metric_inconsistent": lambda d: (
        (d["Pvwap"] / d["Sprd"] - d[METRIC]).abs()
        > 0.01 * np.maximum(1.0, d[METRIC].abs())),
    "participation_impossible": lambda d: ~d["PR"].between(0.0, 100.0,
                                                           inclusive="right"),
    "duration_nonpositive": lambda d: ~(d["Dur"] > 0),
    "auction_share_impossible": lambda d: (d["%OPEN"] + d["%CLOSE"]) > 100.0,
    "adv_nonpositive": lambda d: ~(d["%Adv"] > 0),
    "volatility_nonpositive": lambda d: ~(d["Vol"] > 0),
}


def dq_mask(df: pd.DataFrame) -> pd.Series:
    bad = pd.Series(False, index=df.index)
    for rule in DQ_RULES.values():
        bad |= rule(df).fillna(True)
    return bad


# --------------------------------------------------------------------------
def features(df: pd.DataFrame) -> np.ndarray:
    """Order conditions, transformed to enter roughly linearly."""
    return np.column_stack([
        np.log(df["Vol"].clip(lower=1e-3)),
        np.log(df["Dur"].clip(lower=1e-3)),
        np.log(df["Sprd"].clip(lower=1e-3)),
        np.log(df["%Adv"].clip(lower=1e-4)),
        np.sqrt(df["PR"].clip(lower=0) / 10.0),
        df["%OPEN"] / 100.0,
        df["%CLOSE"] / 100.0,
        (df["Side"].astype(str).str.upper().str.startswith("B")).astype(float),
    ])


class Method:
    name = "base"
    uses_dq = True

    def fit(self, year):
        if self.uses_dq:
            year = year[~dq_mask(year)]
        self.cells = {b: self._fit_cell(g) for b, g in year.groupby("Strategy")}
        return self

    def score(self, month):
        flags = pd.Series(False, index=month.index)
        z = pd.Series(np.nan, index=month.index)
        dq = dq_mask(month) if self.uses_dq else pd.Series(False, index=month.index)
        for b, g in month[~dq].groupby("Strategy"):
            f, s = self._score_cell(self.cells[b], g)
            flags.loc[g.index], z.loc[g.index] = f, s
        return flags, z, dq


# --- raw metric -----------------------------------------------------------
class RawSigma(Method):
    """The current rule: MAX(mean + 4sd, P99.5), MIN(mean - 4sd, P0.5)."""
    name, uses_dq = "M0_raw_4sigma", False

    def __init__(self, k=4.0, uses_dq=False):
        self.k, self.uses_dq = k, uses_dq
        if uses_dq:
            self.name = "M0b_raw_4sigma+DQ"

    def _fit_cell(self, g):
        x = g[METRIC].to_numpy()
        m, s = x.mean(), x.std(ddof=1)
        return (min(m - self.k * s, np.percentile(x, 0.5)),
                max(m + self.k * s, np.percentile(x, 99.5)), m, s)

    def _score_cell(self, c, g):
        x = g[METRIC].to_numpy()
        return (x < c[0]) | (x > c[1]), (x - c[2]) / c[3]


class RawRobust(Method):
    name = "M1_raw_median_4MAD"

    def _fit_cell(self, g):
        x = g[METRIC].to_numpy()
        med = np.median(x)
        return med, 1.4826 * np.median(np.abs(x - med))

    def _score_cell(self, c, g):
        z = (g[METRIC].to_numpy() - c[0]) / c[1]
        return np.abs(z) > 4.0, z


class Bucketed(Method):
    """4 sigma within %ADV x volatility x duration terciles (27 buckets)."""
    name = "M2_bucket_4sigma"
    cols = ("%Adv", "Vol", "Dur")

    def _fit_cell(self, g):
        edges = {c: np.quantile(g[c], [1 / 3, 2 / 3]) for c in self.cols}
        key = self._key(g, edges)
        bands = {}
        for k, h in g.groupby(key):
            x = h[METRIC].to_numpy()
            bands[k] = (x.mean(), x.std(ddof=1))
        return edges, bands

    def _key(self, g, edges):
        parts = [np.digitize(g[c], edges[c]).astype(str) for c in self.cols]
        return pd.Series(["".join(p) for p in zip(*parts)], index=g.index)

    def _score_cell(self, c, g):
        edges, bands = c
        key = self._key(g, edges)
        mean = key.map(lambda k: bands.get(k, (np.nan, np.nan))[0]).to_numpy()
        sd = key.map(lambda k: bands.get(k, (np.nan, np.nan))[1]).to_numpy()
        z = (g[METRIC].to_numpy() - mean) / sd
        return np.abs(z) > 4.0, z


# --- conditioned: location-scale model -----------------------------------
class LocationScale:
    """Expected performance and expected dispersion given order conditions.

    centre: Huber regression (bounded influence, so a bad order cannot drag
            the expectation toward itself).
    scale : regression of log|residual| on the same conditions -- the
            multiplicative heteroskedasticity model (Harvey 1976) -- then one
            constant rescaling so median |z| equals the normal 0.6745.
    Two passes: the second centre fit is weighted by the first scale fit.
    """

    def __init__(self, robust=True):
        self.robust = robust

    def fit(self, X, y):
        self.mu_x, self.sd_x = X.mean(0), X.std(0) + 1e-12
        Xs = (X - self.mu_x) / self.sd_x
        w = np.ones(len(y))
        for _ in range(2):
            loc = (HuberRegressor(epsilon=1.35, max_iter=500, alpha=0.0)
                   if self.robust else LinearRegression())
            loc.fit(Xs, y, sample_weight=w)
            r = y - loc.predict(Xs)
            # Huber on log|r| too: a bad order should not move the scale fit.
            sc = (HuberRegressor(epsilon=1.35, max_iter=500, alpha=0.0)
                  if self.robust else LinearRegression())
            sc.fit(Xs, np.log(np.abs(r) + 1e-9))
            scale = np.exp(sc.predict(Xs))
            w = 1.0 / scale ** 2
            w = w / w.mean()
        self.loc, self.sc = loc, sc
        z = r / scale
        self.c = (np.median(np.abs(z - np.median(z))) / 0.6745
                  if self.robust else z.std(ddof=1))
        return self

    def predict(self, X):
        Xs = (X - self.mu_x) / self.sd_x
        return self.loc.predict(Xs), np.exp(self.sc.predict(Xs)) * self.c

    def z(self, X, y):
        mu, s = self.predict(X)
        return (y - mu) / s


class CondSigma(Method):
    """Literal mean +/- 4 sd, applied to the conditional z of the fit year."""
    name = "M3_cond_classical_4sigma"
    robust = False

    def _fit_cell(self, g):
        X, y = features(g), g[METRIC].to_numpy()
        ls = LocationScale(robust=self.robust).fit(X, y)
        z = ls.z(X, y)
        return ls, z.mean(), z.std(ddof=1)

    def _score_cell(self, c, g):
        ls, m, s = c
        z = (ls.z(features(g), g[METRIC].to_numpy()) - m) / s
        return np.abs(z) > 4.0, z


class CondRobust(CondSigma):
    """Robust centre and scale throughout; flag |z| > 4 robust sigmas."""
    name = "M4_cond_robust_4sigma"
    robust = True

    def _fit_cell(self, g):
        X, y = features(g), g[METRIC].to_numpy()
        ls = LocationScale(robust=True).fit(X, y)
        z = ls.z(X, y)
        med = np.median(z)
        return ls, med, 1.4826 * np.median(np.abs(z - med))


class CondEVT(CondRobust):
    """Robust conditional z, then a tail probability from a GPD fit.

    Per side, fit a generalised Pareto to the exceedances of |z| over its
    97.5th percentile (Pickands-Balkema-de Haan). Flag an order whose one-sided
    tail probability is below that of a NORMAL 4-sigma event (3.17e-5): the
    threshold keeps the meaning "as rare as 4 sigma" and that coverage claim is
    then true of this data rather than of a Gaussian.
    """
    name = "M5_cond_robust_EVT"
    u_q = 0.975
    p_star = P_4SIGMA_ONE_SIDE

    def _fit_cell(self, g):
        ls, med, mad = super()._fit_cell(g)
        z = (ls.z(features(g), g[METRIC].to_numpy()) - med) / mad
        tails = {}
        for side, t in (("hi", z), ("lo", -z)):
            u = np.quantile(t, self.u_q)
            exc = t[t > u] - u
            xi, _, beta = stats.genpareto.fit(exc, floc=0.0)
            tails[side] = (u, xi, beta, 1.0 - self.u_q)
        return ls, med, mad, tails

    @staticmethod
    def tail_p(t, u, xi, beta, zeta):
        """P(T > t) from the GPD tail; the empirical rate below u."""
        t = np.asarray(t, dtype=float)
        p = np.full(t.shape, 1.0)
        above = t > u
        p[above] = zeta * stats.genpareto.sf(t[above] - u, xi, 0.0, beta)
        return p

    def pvalues(self, c, g):
        ls, med, mad, tails = c
        z = (ls.z(features(g), g[METRIC].to_numpy()) - med) / mad
        p_hi = self.tail_p(z, *tails["hi"])
        p_lo = self.tail_p(-z, *tails["lo"])
        return z, p_hi, p_lo

    def threshold(self, c):
        """The z at which each side's tail probability hits p_star."""
        _, _, _, tails = c
        out = {}
        for side, (u, xi, beta, zeta) in tails.items():
            out[side] = u + stats.genpareto.isf(self.p_star / zeta, xi, 0, beta)
        return out

    def _score_cell(self, c, g):
        z, p_hi, p_lo = self.pvalues(c, g)
        return (p_hi < self.p_star) | (p_lo < self.p_star), z


class CondFDR(CondEVT):
    """EVT p-values, then Benjamini-Hochberg across the month at q.

    Controls the expected share of false alerts among those raised, and in a
    month with no genuine outliers keeps P(any alert) <= q.
    """
    name = "M6_cond_EVT_BH"
    q = 0.05

    def score(self, month):
        flags = pd.Series(False, index=month.index)
        z = pd.Series(np.nan, index=month.index)
        p = pd.Series(1.0, index=month.index)
        dq = dq_mask(month)
        for b, g in month[~dq].groupby("Strategy"):
            zz, ph, pl = self.pvalues(self.cells[b], g)
            z.loc[g.index] = zz
            p.loc[g.index] = np.minimum(1.0, 2.0 * np.minimum(ph, pl))
        ok = ~dq
        pv = p[ok].to_numpy()
        order = np.argsort(pv)
        m = len(pv)
        passed = pv[order] <= self.q * np.arange(1, m + 1) / m
        if passed.any():
            kmax = np.flatnonzero(passed).max()
            idx = p[ok].index[order[:kmax + 1]]
            flags.loc[idx] = True
        return flags, z, dq


class CondTailT(CondRobust):
    """Robust conditional z; threshold = MAX(4, Student-t quantile at p*).

    The policy is a probability, fixed in advance: flag an order whose
    execution is rarer, GIVEN ITS CONDITIONS, than a normal 4-sigma event
    (one-sided 3.17e-5). A Student-t fitted by maximum likelihood to the fit
    year's conditional z carries the tail: one shape parameter estimated from
    the whole sample, so it is far less variable than an extreme-tail GPD, and
    the t likelihood itself down-weights the year's own outliers. The floor of
    4 means a near-normal book gets exactly mean +/- 4 sigma.
    """
    name = "M7_cond_robust_tTail"
    p_star = P_4SIGMA_ONE_SIDE
    floor = 4.0

    def _fit_cell(self, g):
        ls, med, mad = super()._fit_cell(g)
        z = (ls.z(features(g), g[METRIC].to_numpy()) - med) / mad
        thr = {}
        for side, t in (("hi", z), ("lo", -z)):
            # Fit each side's tail on its own half, reflected, so a skewed
            # book gets a different bound on each side.
            half = t[t > 0]
            sym = np.concatenate([half, -half])
            df_, _, sc = stats.t.fit(sym, floc=0.0)
            thr[side] = max(self.floor, stats.t.isf(self.p_star, df_, 0, sc))
        return ls, med, mad, thr

    def _score_cell(self, c, g):
        ls, med, mad, thr = c
        z = (ls.z(features(g), g[METRIC].to_numpy()) - med) / mad
        return (z > thr["hi"]) | (z < -thr["lo"]), z


def t_tail_quantile_match(t: np.ndarray, p_star: float,
                          probs=(0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995)):
    """Threshold on one side's |z| from a t fitted by QUANTILE MATCHING.

    Uses only the body and moderate tail (up to the 99.5th percentile of the
    side), so the year's own extreme orders -- the ones the band exists to
    catch -- cannot widen the threshold that judges them. Returns
    (threshold, df, scale).
    """
    from scipy.optimize import minimize
    half = t[t > 0]
    q_emp = np.quantile(half, probs)
    # Half-t: P(T > x | T > 0) = 2 * sf(x)  ->  quantile at 1 - (1-p)/2.
    pp = 1.0 - (1.0 - np.asarray(probs)) / 2.0

    def loss(theta):
        df_, sc = np.exp(theta)
        q = stats.t.ppf(pp, df_ + 2.0, 0, sc)
        return float(np.sum((np.log(q) - np.log(q_emp)) ** 2))

    best = minimize(loss, x0=np.log([8.0, 1.0]), method="Nelder-Mead")
    df_, sc = np.exp(best.x)
    df_ += 2.0
    # half holds P(T>0) ~ 1/2 of the sample; p_star is a one-sided overall rate.
    thr = stats.t.isf(p_star, df_, 0, sc)
    return thr, df_, sc


class CondTailQM(CondRobust):
    """Robust conditional z; threshold = MAX(4, quantile-matched t at p*)."""
    name = "M8_cond_robust_tQM"
    p_star = P_4SIGMA_ONE_SIDE
    floor = 4.0
    spline = False

    def _fit_cell(self, g):
        X, y = self._X(g), g[METRIC].to_numpy()
        ls = LocationScale(robust=True).fit(X, y)
        z = ls.z(X, y)
        med = np.median(z)
        mad = 1.4826 * np.median(np.abs(z - med))
        z = (z - med) / mad
        thr = {}
        for side, t in (("hi", z), ("lo", -z)):
            v, df_, sc = t_tail_quantile_match(t, self.p_star)
            thr[side] = max(self.floor, v)
        self.last_tail = thr
        return ls, med, mad, thr

    def _X(self, g):
        return features(g)

    def _score_cell(self, c, g):
        ls, med, mad, thr = c
        z = (ls.z(self._X(g), g[METRIC].to_numpy()) - med) / mad
        return (z > thr["hi"]) | (z < -thr["lo"]), z


class SplineFeatures:
    """Additive cubic B-splines per condition + one vol x close interaction."""

    def __init__(self):
        from sklearn.preprocessing import SplineTransformer
        self.st = SplineTransformer(n_knots=5, degree=3,
                                    extrapolation="linear")

    def fit(self, X):
        self.st.fit(X[:, :5])
        return self

    def transform(self, X):
        return np.column_stack([self.st.transform(X[:, :5]), X[:, 5:],
                                X[:, 0] * X[:, 6]])


class CondSplineRobust(CondRobust):
    """M4 with additive splines (GAMLSS-like centre and scale)."""
    name = "M9_spline_robust_4sigma"

    def fit(self, year):
        clean = year[~dq_mask(year)]
        self.sf = {b: SplineFeatures().fit(features(g))
                   for b, g in clean.groupby("Strategy")}
        self._b = None
        return super().fit(year)

    def _fit_cell(self, g):
        b = g["Strategy"].iloc[0]
        X = self.sf[b].transform(features(g))
        y = g[METRIC].to_numpy()
        ls = LocationScale(robust=True).fit(X, y)
        z = ls.z(X, y)
        med = np.median(z)
        return ls, med, 1.4826 * np.median(np.abs(z - med)), b

    def _score_cell(self, c, g):
        ls, med, mad, b = c
        X = self.sf[b].transform(features(g))
        z = (ls.z(X, g[METRIC].to_numpy()) - med) / mad
        return np.abs(z) > 4.0, z


class CondSplineQM(CondSplineRobust):
    """M9 centre/scale + M8 quantile-matched tail threshold."""
    name = "M10_spline_robust_tQM"
    p_star = P_4SIGMA_ONE_SIDE

    def _fit_cell(self, g):
        ls, med, mad, b = super()._fit_cell(g)
        X = self.sf[b].transform(features(g))
        z = (ls.z(X, g[METRIC].to_numpy()) - med) / mad
        thr = {side: max(4.0, t_tail_quantile_match(t, self.p_star)[0])
               for side, t in (("hi", z), ("lo", -z))}
        return ls, med, mad, b, thr

    def _score_cell(self, c, g):
        ls, med, mad, b, thr = c
        X = self.sf[b].transform(features(g))
        z = (ls.z(X, g[METRIC].to_numpy()) - med) / mad
        return (z > thr["hi"]) | (z < -thr["lo"]), z


ALL = [RawSigma, lambda: RawSigma(uses_dq=True), RawRobust, Bucketed,
       CondSigma, CondRobust, CondEVT, CondFDR, CondTailT, CondTailQM,
       CondSplineRobust, CondSplineQM]


# --------------------------------------------------------------------------
# Cohort test: systematic deterioration that no single order reveals.
# --------------------------------------------------------------------------
COHORT_FEATURES = ("PR", "%Adv", "Vol", "Dur", "Sprd", "%CLOSE")


def cohort_test(model: CondRobust, year: pd.DataFrame, month: pd.DataFrame,
                t_crit: float = 4.0, n_bins: int = 5, min_n: int = 100):
    """Mean conditional z per strategy and per condition bucket, vs zero.

    z is winsorised at +/-4 so a single outlier (already handled by the order
    test) cannot drive a cohort alarm. Bucket edges come from the FIT year.
    Returns the cohorts whose |t| exceeds t_crit.
    """
    year = year[~dq_mask(year)]
    month = month[~dq_mask(month)]
    hits = []
    for b, g in month.groupby("Strategy"):
        c = model.cells[b]
        z = (c[0].z(features(g), g[METRIC].to_numpy()) - c[1]) / c[2]
        z = np.clip(z, -4, 4)
        groups = [("ALL", "all", np.ones(len(g), bool))]
        yb = year[year["Strategy"] == b]
        for f in COHORT_FEATURES:
            edges = np.unique(np.quantile(yb[f], np.linspace(0, 1, n_bins + 1)[1:-1]))
            bins = np.digitize(g[f], edges)
            groups += [(f, int(i), bins == i) for i in np.unique(bins)]
        for f, i, mask in groups:
            n = int(mask.sum())
            if n < min_n:
                continue
            zz = z[mask]
            t = zz.mean() / (zz.std(ddof=1) / np.sqrt(n))
            if abs(t) > t_crit:
                hits.append({"strategy": b, "feature": f, "bucket": i, "n": n,
                             "mean_z": round(float(zz.mean()), 3),
                             "t": round(float(t), 2)})
    return pd.DataFrame(hits)
