"""Expected performance and expected dispersion given order conditions.

centre : Huber M-regression (Huber 1964). Bounded influence, so a bad order
         cannot drag the expectation toward itself.
scale  : Huber regression of log|residual| on the same conditions -- the
         multiplicative heteroskedasticity model (Harvey 1976). One constant
         then rescales so that median |z| matches the normal (0.6745).
Two passes; the second centre fit is weighted by 1/scale^2.

Coefficients are stored as plain arrays so the frozen model is a JSON file
that any reader can apply with a dot product.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import HuberRegressor


def _huber():
    return HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=1000)


class LocationScale:
    def fit(self, X: np.ndarray, y: np.ndarray, n_pass: int = 2):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.x_mean = X.mean(axis=0)
        self.x_sd = X.std(axis=0)
        self.x_sd[self.x_sd == 0] = 1.0
        Xs = (X - self.x_mean) / self.x_sd
        w = np.ones(len(y))
        for _ in range(n_pass):
            loc = _huber().fit(Xs, y, sample_weight=w)
            r = y - loc.predict(Xs)
            sc = _huber().fit(Xs, np.log(np.abs(r) + 1e-9))
            s = np.exp(sc.predict(Xs))
            w = 1.0 / s ** 2
            w /= w.mean()
        self.loc_coef = loc.coef_.copy()
        self.loc_intercept = float(loc.intercept_)
        self.sc_coef = sc.coef_.copy()
        self.sc_intercept = float(sc.intercept_)
        z = r / s
        self.c = float(np.median(np.abs(z - np.median(z))) / 0.6745)
        return self

    def predict(self, X):
        Xs = (np.asarray(X, float) - self.x_mean) / self.x_sd
        mu = Xs @ self.loc_coef + self.loc_intercept
        s = np.exp(Xs @ self.sc_coef + self.sc_intercept) * self.c
        return mu, s

    def z(self, X, y):
        mu, s = self.predict(X)
        return (np.asarray(y, float) - mu) / s

    def to_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in {"x_mean": self.x_mean, "x_sd": self.x_sd,
                             "loc_coef": self.loc_coef,
                             "loc_intercept": self.loc_intercept,
                             "sc_coef": self.sc_coef,
                             "sc_intercept": self.sc_intercept,
                             "c": self.c}.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "LocationScale":
        m = cls()
        for k in ("x_mean", "x_sd", "loc_coef", "sc_coef"):
            setattr(m, k, np.asarray(d[k], float))
        m.loc_intercept, m.sc_intercept, m.c = (float(d["loc_intercept"]),
                                                float(d["sc_intercept"]),
                                                float(d["c"]))
        return m
