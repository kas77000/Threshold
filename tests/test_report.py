import numpy as np
import pandas as pd

from cept import dq, fit, load, plots, report
from lab import scenarios


def test_reports_and_plots_write(tmp_path):
    year, _ = scenarios.make("S2_mixture", seed=0, n_per_month=1500)
    y, rep = load.prepare(year)
    y = dq.split(y)[0]
    cells, _ = fit.fit_all(y, min_cell_n=1000)
    plots.qq_plot(np.random.default_rng(0).normal(size=500), tmp_path / "qq.png", "t")
    plots.coef_plot(cells["TWAP_VWAP"], tmp_path / "coef.png")
    meta = {"market": "HK", "fit_start": "2025-07-01", "fit_end": "2026-06-30",
            "n_orders": len(y), "policy": {"RARITY_A": 1e-4}}
    empty = pd.DataFrame()
    report.write_fit_summary(tmp_path / "fit.md", cells, meta, empty, empty,
                             empty, rep, {})
    txt = (tmp_path / "fit.md").read_text(encoding="utf-8")
    assert "TWAP_VWAP" in txt
    # the threshold comes first, the detail in an appendix
    assert txt.index("## 1. The threshold") < txt.index("## Appendix")
    assert "OUTLIER if below expected by more than" in txt
    assert (tmp_path / "qq.png").stat().st_size > 1000
