import numpy as np
import pandas as pd

from cept import features, schema


def _frame():
    return pd.DataFrame({
        schema.SYMBOL: ["A HK"] * 25 + ["B HK"] * 3,
        schema.VOLATILITY: [0.02] * 25 + [0.05] * 3,
        schema.DURATION_MIN: 60.0, schema.SPREAD_BPS: 8.0,
        schema.PCT_ADV: 2.0, schema.PARTICIPATION: 10.0,
        schema.OPEN_PCT: 0.0, schema.CLOSE_PCT: 20.0,
        schema.SIDE: ["Buy"] * 14 + ["Sell"] * 14,
    })


def test_vol_typical_uses_symbol_median_and_fallback():
    df = _frame()
    vt = features.VolTypical.fit(df, min_n=20)
    assert vt.by_symbol == {"A HK": 0.02}           # B has only 3 orders
    assert vt.fallback == np.median(df[schema.VOLATILITY])
    looked = vt.lookup(df)
    assert looked[0] == 0.02 and looked[-1] == vt.fallback
    assert features.VolTypical.from_dict(vt.to_dict()).by_symbol == vt.by_symbol


def test_design_shapes_and_values():
    df = _frame()
    vt = features.VolTypical.fit(df, min_n=20)
    X = features.design(df, vt, include_surprise=True)
    assert X.shape == (28, 9)
    assert np.allclose(X[0, 0], np.log(0.02))       # log vol_typ
    assert np.allclose(X[0, 1], 0.0)                # surprise = log(1)
    assert np.allclose(X[0, 5], 1.0)                # sqrt(10/10)
    assert X[0, 8] == 1.0 and X[-1, 8] == 0.0       # is_buy
    X8 = features.design(df, vt, include_surprise=False)
    assert X8.shape == (28, 8)
    assert features.feature_names(False) == [n for n in features.FEATURE_NAMES
                                             if n != "log_vol_surprise"]
