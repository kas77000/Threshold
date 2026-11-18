import numpy as np

from cept import dq, fit, load, schema
from lab import scenarios


def _clean(scn, seed=0, n=1500):
    year, month = scenarios.make(scn, seed=seed, n_per_month=n)
    y = dq.split(load.prepare(year)[0])[0]
    m = dq.split(load.prepare(month)[0])[0]
    return y, m


def test_fit_all_cells_and_thresholds_on_normal_book():
    y, _ = _clean("S2_mixture")
    cells, skipped = fit.fit_all(y, min_cell_n=1000)
    assert set(cells) == {"TWAP_VWAP"} and skipped == {}     # one group
    for cm in cells.values():
        assert cm.thr["A"]["hi"] >= 4.0 and cm.thr["A"]["lo"] >= 4.0
        assert cm.thr["A"]["lo"] < 5.0          # near-normal after conditioning
        assert 3.0 <= cm.thr["B"]["lo"] < cm.thr["A"]["lo"]
        assert set(cm.cohort_edges) >= {schema.PARTICIPATION, schema.FILL_RATE}
        assert sorted(cm.strategies) == ["TWAP", "VWAP"]     # strategy is a condition


def test_thin_cell_is_skipped_not_fitted():
    y, _ = _clean("S2_mixture", n=300)
    cells, skipped = fit.fit_all(y, min_cell_n=5000)
    assert cells == {} and set(skipped) == {"TWAP_VWAP"}


def test_zscores_and_predicted_count():
    y, m = _clean("S2_mixture")
    cells, _ = fit.fit_all(y, min_cell_n=1000)
    cm = cells["TWAP_VWAP"]
    g = m
    z, mu, disp = fit.zscores(cm, g)
    assert z.shape == (len(g),) and (disp > 0).all()
    pred = fit.predicted_count(cm, 10000, "A")
    assert 0.2 < pred < 1.5                     # ~1 per 10,000 by construction


def test_round_trip():
    y, m = _clean("S2_mixture")
    cm = fit.fit_all(y, min_cell_n=1000)[0]["TWAP_VWAP"]
    cm2 = fit.CellModel.from_dict(cm.to_dict())
    g = m
    assert np.allclose(fit.zscores(cm, g)[0], fit.zscores(cm2, g)[0])
