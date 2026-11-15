"""One side's tail, from a Student-t fitted by QUANTILE MATCHING.

Only the body and the moderate tail (up to the 99.5th percentile of the
side) inform the fit, so the year's own extreme orders -- the ones the band
exists to catch -- cannot widen the threshold that judges them. In the lab,
maximum-likelihood t and GPD peaks-over-threshold fits were both inflated by
0.1% contamination; this one was not.

Convention: t is the side's signed distance (z for the high side, -z for the
low side), centred on the median, so P(t > 0) ~ 1/2. P(T > x | T > 0) is
2 * sf(x), hence matching at probability 1 - (1 - p)/2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from cept import config


@dataclass(frozen=True)
class TailFit:
    df: float
    scale: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TailFit":
        return cls(float(d["df"]), float(d["scale"]))


def fit_side(t, probs=config.TAIL_PROBS) -> TailFit:
    t = np.asarray(t, float)
    half = t[t > 0]
    q_emp = np.quantile(half, probs)
    pp = 1.0 - (1.0 - np.asarray(probs)) / 2.0

    def loss(theta):
        nu, sc = np.exp(theta)
        q = stats.t.ppf(pp, nu + 2.0, 0.0, sc)
        return float(np.sum((np.log(q) - np.log(q_emp)) ** 2))

    best = minimize(loss, x0=np.log([8.0, 1.0]), method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 4000})
    nu, sc = np.exp(best.x)
    return TailFit(df=float(nu + 2.0), scale=float(sc))


def threshold(tf: TailFit, rarity: float, floor: float) -> float:
    return float(max(floor, stats.t.isf(rarity / 2.0, tf.df, 0.0, tf.scale)))


def exceed_prob(tf: TailFit, thr: float) -> float:
    return float(stats.t.sf(thr, tf.df, 0.0, tf.scale))
