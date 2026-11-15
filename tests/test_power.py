from cept import dq, fit, load, power
from lab import scenarios


def test_detection_increases_with_size_and_is_high_on_mixture():
    year, _ = scenarios.make("S2_mixture", seed=0, n_per_month=1500)
    y = dq.split(load.prepare(year)[0])[0]
    cells, _ = fit.fit_all(y, min_cell_n=1000)
    p = power.injected_detection(cells, y, n=300)
    lo = p[p["side"] == "lo"].groupby("size")["detected_A"].mean()
    assert lo.is_monotonic_increasing
    assert lo.loc[8] > 0.95 and lo.loc[5] > 0.6
