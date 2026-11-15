import numpy as np
from scipy import stats

from cept import tail


def test_normal_data_gives_the_floor():
    z = np.random.default_rng(0).normal(size=40000)
    tf = tail.fit_side(z)
    assert tf.df > 15
    assert tail.threshold(tf, 1e-4, 4.0) == 4.0          # q(5e-5) ~ 3.9 < floor


def test_fat_data_widens_threshold_to_true_quantile():
    t = np.random.default_rng(1).standard_t(4, size=200000)
    tf = tail.fit_side(t)
    thr = tail.threshold(tf, 1e-4, 4.0)
    true_q = stats.t.isf(5e-5, 4)
    assert abs(thr - true_q) / true_q < 0.25
    assert abs(tail.exceed_prob(tf, thr) - 5e-5) < 1e-6


def test_contamination_does_not_widen():
    z = np.random.default_rng(2).normal(size=30000)
    z[:30] = -np.random.default_rng(3).uniform(6, 10, 30)    # 0.1% bad
    tf = tail.fit_side(-z)                                   # the bad side
    assert tail.threshold(tf, 1e-4, 4.0) < 4.5


def test_round_trip():
    tf = tail.TailFit(df=7.5, scale=0.9)
    assert tail.TailFit.from_dict(tf.to_dict()) == tf
