import numpy as np

from lab import scenarios


def test_generator_has_extract_columns_and_units():
    year, month = scenarios.make("S2_mixture", seed=1, n_per_month=600)
    for col in ["aggrTgtId", "Strategy", "Sym", "Date", "ePvwap/Sprd", "Pvwap",
                "Sprd", "%Adv", "Vol", "PR", "Dur", "%OPEN", "%CLOSE", "Side",
                "FR", "#Shares"]:
        assert col in year.columns, col
    # Vol is a daily Parkinson fraction, not annualised percent
    assert 0.003 < year["Vol"].median() < 0.05
    # FR is percent, mostly complete, some partial
    assert year["FR"].between(0, 100).all()
    assert 0.05 < (year["FR"] < 100).mean() < 0.25
    # a symbol pool: symbols repeat across the year
    assert year["Sym"].nunique() <= 300
    assert year.groupby("Sym").size().median() >= 10


def test_auction_shares_feasible():
    year, _ = scenarios.make("S2_mixture", seed=2, n_per_month=600)
    assert ((year["%OPEN"] + year["%CLOSE"]) <= 100.0 + 1e-9).all()
