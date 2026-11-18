import pandas as pd
import pytest

from cept import load, schema
from lab import scenarios


def _raw():
    year, _ = scenarios.make("S2_mixture", seed=0, n_per_month=300)
    return year


def test_renames_and_derives():
    df, rep = load.prepare(_raw())
    assert schema.METRIC in df and schema.FILL_RATE in df
    assert set(df[schema.MARKET]) == {"HK"}
    assert set(df[schema.GROUP]) == {"TWAP_VWAP"}
    assert rep.rows_kept == len(df)


def test_every_row_is_accounted_for():
    raw = _raw()
    raw.loc[raw.index[:5], "Strategy"] = "PART"
    raw.loc[raw.index[5:8], "ePvwap/Sprd"] = None
    raw["Date"] = raw["Date"].astype(object)
    raw.loc[raw.index[8:10], "Date"] = "not a date"
    df, rep = load.prepare(raw)
    assert rep.rows_in == rep.rows_kept + sum(rep.dropped.values())
    assert rep.unmapped_strategies == {"PART": 5}
    assert rep.dropped["missing metric"] == 3
    assert rep.dropped["missing or unparseable date"] == 2


def test_market_filter_is_counted():
    raw = _raw()
    raw.loc[raw.index[:7], "Sym"] = "7203 JP"
    df, rep = load.prepare(raw, market="HK")
    assert rep.dropped["other market"] == 7
    assert rep.rows_in == rep.rows_kept + sum(rep.dropped.values())


def test_missing_required_column_names_raw_column():
    raw = _raw().drop(columns=["ePvwap/Sprd"])
    with pytest.raises(ValueError, match="ePvwap/Sprd"):
        load.prepare(raw)


def test_missing_condition_column_is_named_not_keyerror():
    raw = _raw().drop(columns=["Vol"])
    with pytest.raises(ValueError, match="Vol"):
        load.prepare(raw)
