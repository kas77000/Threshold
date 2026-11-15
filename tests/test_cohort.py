from cept import cohort, dq, fit, load, score
from lab import scenarios


def _run(scn, seed=0):
    year, month = scenarios.make(scn, seed=seed, n_per_month=1500)
    y = dq.split(load.prepare(year)[0])[0]
    m = dq.split(load.prepare(month)[0])[0]
    cells, _ = fit.fit_all(y, min_cell_n=1000)
    return cohort.cohort_test(cells, score.score_orders(cells, m))


def test_silent_on_ordinary_month():
    r = _run("S2_mixture")
    assert not r["alert"].any()
    assert {"ALL"} <= set(r["feature"])


def test_fires_on_strategy_decay():
    r = _run("S6_strategy_decay")
    hit = r[r["alert"]]
    assert ((hit["benchmark"] == "TWAP") & (hit["feature"] == "ALL")).any()
    assert (hit["benchmark"] == "VWAP").sum() == 0


def test_fires_on_high_pr_decay():
    r = _run("S8_feature_decay")
    hit = r[r["alert"] & (r["feature"] == "participation")]
    assert len(hit) >= 1
