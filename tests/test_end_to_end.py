import pandas as pd

from cept import cli
from lab import scenarios


def _files(tmp_path, scn):
    year, month = scenarios.make(scn, seed=0, n_per_month=1500)
    hidden = ["truth", "_mu", "_sd", "_illiq"]
    yp, mp = tmp_path / "year.csv", tmp_path / "month.csv"
    year.drop(columns=hidden).to_csv(yp, index=False)
    month.drop(columns=hidden).to_csv(mp, index=False)
    return yp, mp, month


def test_fit_then_score_mixture(tmp_path):
    yp, mp, month = _files(tmp_path, "S2_mixture")
    fit_dir, rev_dir = tmp_path / "fit", tmp_path / "rev"
    assert cli.main(["fit", "--csv", str(yp), "--out", str(fit_dir),
                     "--min-cell-n", "1000"]) == 0
    for f in ["model.json", "summary.md", "lomo_backtest.csv",
              "detection_power.csv", "vol_sensitivity.csv", "coefficients.csv"]:
        assert (fit_dir / f).exists(), f
    assert cli.main(["score", "--csv", str(mp), "--model",
                     str(fit_dir / "model.json"), "--out", str(rev_dir)]) == 0
    a = pd.read_csv(rev_dir / "alerts_tierA.csv")
    bad_ids = set(month.loc[month["truth"] == "bad", "aggrTgtId"])
    assert len(set(a["order_id"]) & bad_ids) >= 5
    assert len(set(a["order_id"]) - bad_ids) <= 1
    for f in ["watchlist_tierB.csv", "cohort.csv", "backtest.csv", "drift.csv",
              "data_exceptions.csv", "summary.md"]:
        assert (rev_dir / f).exists(), f


def test_score_refuses_overlap(tmp_path):
    yp, _, _ = _files(tmp_path, "S2_mixture")
    fit_dir = tmp_path / "fit"
    cli.main(["fit", "--csv", str(yp), "--out", str(fit_dir),
              "--min-cell-n", "1000", "--no-lomo"])
    assert cli.main(["score", "--csv", str(yp), "--model",
                     str(fit_dir / "model.json"), "--out",
                     str(tmp_path / "r")]) == 2


def test_check_runs(tmp_path, capsys):
    yp, _, _ = _files(tmp_path, "S4_data_errors")
    assert cli.main(["check", "--csv", str(yp)]) == 0
    out = capsys.readouterr().out
    assert "STRATEGY" in out and "DATA QUALITY" in out


def test_score_refuses_model_fitted_on_another_metric(tmp_path, monkeypatch):
    from cept import config
    yp, mp, _ = _files(tmp_path, "S2_mixture")
    fit_dir = tmp_path / "fit"
    assert cli.main(["fit", "--csv", str(yp), "--out", str(fit_dir),
                     "--min-cell-n", "1000", "--no-lomo"]) == 0
    # The config now names another metric: the old model must be refused.
    monkeypatch.setattr(config, "METRIC_COLUMN", "eParr/Sprd")
    assert cli.main(["score", "--csv", str(mp), "--model",
                     str(fit_dir / "model.json"), "--out",
                     str(tmp_path / "r")]) == 2
