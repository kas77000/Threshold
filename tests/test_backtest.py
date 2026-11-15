import numpy as np

from cept import backtest, dq, load
from lab import scenarios


def test_kupiec():
    assert backtest.kupiec_pvalue(10000, 1, 1e-4) > 0.5
    assert backtest.kupiec_pvalue(10000, 12, 1e-4) < 0.001
    assert backtest.kupiec_pvalue(10000, 0, 1e-4) > 0.1   # LR = 2np = 2 -> p = 0.157


def test_lomo_predicted_matches_realised_on_mixture():
    year, _ = scenarios.make("S2_mixture", seed=0, n_per_month=600)
    y = dq.split(load.prepare(year)[0])[0]
    t = backtest.lomo(y, min_cell_n=1000)
    assert set(t["month"]) == set(y["order_date"].dt.to_period("M").astype(str))
    # Tier B is frequent enough to test calibration: about 1/1000
    pred, real = t["pred_B"].sum(), t["real_B"].sum()
    clean_n = t["n"].sum()
    assert real <= pred + 4 * np.sqrt(pred) + 12   # planted bad (12/yr) allowed
