import pandas as pd
import pytest

from cept import dq, fit, load, schema, score
from lab import scenarios


def _setup(scn="S2_mixture", seed=0, n=1500):
    year, month = scenarios.make(scn, seed=seed, n_per_month=n)
    y = dq.split(load.prepare(year)[0])[0]
    mdf = load.prepare(month)[0]
    mdf["truth"] = month["truth"].to_numpy()        # same row order
    m = dq.split(mdf)[0]
    cells, _ = fit.fit_all(y, min_cell_n=1000)
    return y, m, cells


def test_tiers_on_mixture_month():
    y, m, cells = _setup()
    out = score.score_orders(cells, m)
    assert set(out["tier"]) <= {"A", "B", "IN_RANGE"}
    a = out[out["tier"] == "A"]
    assert (a["truth"] == "bad").sum() >= 5          # 6 planted
    assert (a["truth"] == "clean").sum() <= 1
    assert (a["excess"] > 0).all()


def test_no_model_for_unfitted_cell():
    y, m, cells = _setup()
    out = score.score_orders({"VWAP": cells["VWAP"]}, m)
    assert (out.loc[out[schema.BENCHMARK] == "TWAP", "tier"] == "NO_MODEL").all()


def test_tags():
    y, m, cells = _setup()
    m = m.copy()
    m.loc[m.index[0], schema.FILL_RATE] = 50.0
    m.loc[m.index[0], schema.CLOSE_PCT] = 60.0
    out = score.score_orders(cells, m)
    tags = out.loc[out.index[0], "tags"]
    assert "partial fill" in tags and "auction heavy" in tags


def test_leakage_guard():
    y, m, cells = _setup()
    with pytest.raises(score.GuardError):
        score.check_no_overlap("2025-07-01", "2026-07-15", m)
    score.check_no_overlap("2025-07-01", "2026-06-30", m)   # no raise


def test_drift_flags_volatile_month():
    year, month = scenarios.make("S5_volatile_month", seed=0, n_per_month=1500)
    y = dq.split(load.prepare(year)[0])[0]
    m = dq.split(load.prepare(month)[0])[0]
    cells, _ = fit.fit_all(y, min_cell_n=1000)
    d = score.drift(cells, m)
    assert d.loc[d["feature"] == schema.VOLATILITY, "warn"].all()


def test_check_metric():
    from cept import config
    ok = {"metric": config.METRIC_COLUMN, "metric_units": config.METRIC_UNITS}
    score.check_metric(ok)                                   # no raise
    with pytest.raises(score.GuardError, match="Refit"):
        score.check_metric({**ok, "metric": "eParr/Sprd"})
    with pytest.raises(score.GuardError):
        score.check_metric({**ok, "metric_units": "bps"})
    with pytest.raises(score.GuardError):
        score.check_metric({})                               # unrecorded = refuse


def test_metric_is_named_once_in_config():
    from cept import config, schema as s
    inv = {v: k for k, v in config.COLUMN_MAP.items()}
    assert inv[s.METRIC] == config.METRIC_COLUMN
    assert inv[s.SLIPPAGE_BPS] == config.METRIC_BPS_COLUMN
