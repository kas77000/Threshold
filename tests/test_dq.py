import numpy as np

from cept import dq, load, schema
from lab import scenarios


def _df(scn="S2_mixture"):
    year, _ = scenarios.make(scn, seed=0, n_per_month=300)
    return load.prepare(year)[0]


def test_clean_book_has_no_exceptions():
    clean, exc = dq.split(_df())
    assert len(exc) == 0


def test_each_rule_fires():
    df = _df()
    df.loc[0, schema.SPREAD_BPS] = 0.05
    df.loc[1, schema.METRIC] = df.loc[1, schema.METRIC] * 3 + 7
    df.loc[2, schema.PARTICIPATION] = 150
    df.loc[3, schema.DURATION_MIN] = 0
    df.loc[4, [schema.OPEN_PCT, schema.CLOSE_PCT]] = [60, 60]
    df.loc[5, schema.PCT_ADV] = 0
    df.loc[6, schema.VOLATILITY] = 0
    df.loc[7, schema.FILL_RATE] = 120
    df.loc[8, schema.DURATION_MIN] = np.nan
    r = dq.reasons(df)
    assert list(r.iloc[:9]) == [
        "spread_implausible", "metric_inconsistent", "participation_impossible",
        "duration_nonpositive", "auction_share_impossible", "adv_nonpositive",
        "volatility_nonpositive", "fill_rate_impossible", "condition_missing"]
    clean, exc = dq.split(df)
    assert len(exc) == 9 and len(clean) == len(df) - 9
    assert "dq_reason" in exc


def test_planted_data_errors_are_caught():
    year, _ = scenarios.make("S4_data_errors", seed=0, n_per_month=600)
    df = load.prepare(year)[0]
    year_truth = year["truth"].to_numpy()
    flagged = (dq.reasons(df) != "").to_numpy()
    assert flagged[year_truth == "dq"].all()
    assert not flagged[year_truth != "dq"].any()
