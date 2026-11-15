from cept import dq, fit, load, schema, sensitivity
from lab import scenarios


def test_lists_orders_cleared_only_by_vol_surprise():
    year, month = scenarios.make("S2_mixture", seed=0, n_per_month=1500)
    y = dq.split(load.prepare(year)[0])[0]
    m = dq.split(load.prepare(month)[0])[0].copy()
    cells, _ = fit.fit_all(y, min_cell_n=1000)
    # an order with a bad miss on a 4x-volatile day
    i = m.index[0]
    m.loc[i, schema.VOLATILITY] *= 4.0
    m.loc[i, schema.METRIC] -= 8.0
    r = sensitivity.vol_sensitivity(y, m, cells)
    assert set(r.columns) >= {"z_with", "z_without", "high_adv", "vol_surprise"}
    s = sensitivity.summary(r)
    assert s["n_cleared_by_surprise"] == len(r)
