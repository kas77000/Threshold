import numpy as np

from cept.model import LocationScale


def _data(n=20000, seed=0, contaminate=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    mu = 0.5 - 0.3 * X[:, 0]
    sd = np.exp(0.2 + 0.6 * X[:, 1])
    y = mu + sd * rng.normal(size=n)
    if contaminate:
        y[:contaminate] -= 50 * sd[:contaminate]
    return X, y, mu, sd


def test_recovers_centre_and_scale():
    X, y, mu, sd = _data()
    ls = LocationScale().fit(X, y)
    m, s = ls.predict(X)
    assert np.corrcoef(m, mu)[0, 1] > 0.98
    assert np.corrcoef(np.log(s), np.log(sd))[0, 1] > 0.98
    z = ls.z(X, y)
    assert abs(np.median(z)) < 0.05
    assert abs(1.4826 * np.median(np.abs(z - np.median(z))) - 1.0) < 0.05


def test_robust_to_contamination():
    X, y, mu, sd = _data(contaminate=40)
    m, s = LocationScale().fit(X, y).predict(X)
    assert np.corrcoef(m, mu)[0, 1] > 0.97
    assert np.corrcoef(np.log(s), np.log(sd))[0, 1] > 0.97


def test_round_trip():
    X, y, *_ = _data(n=3000)
    ls = LocationScale().fit(X, y)
    ls2 = LocationScale.from_dict(ls.to_dict())
    assert np.allclose(ls.z(X, y), ls2.z(X, y))
