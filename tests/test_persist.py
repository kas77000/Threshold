import json

import numpy as np
import pytest

from cept import dq, fit, load, persist, schema
from lab import scenarios


def test_round_trip_and_policy(tmp_path):
    year, month = scenarios.make("S2_mixture", seed=0, n_per_month=1500)
    y = dq.split(load.prepare(year)[0])[0]
    m = dq.split(load.prepare(month)[0])[0]
    cells, skipped = fit.fit_all(y, min_cell_n=1000)
    meta = {"market": "HK", "fit_start": "2025-07-01", "fit_end": "2026-06-30",
            "n_orders": len(y), "source_file": "x.csv", "source_hash": "abc",
            "skipped": skipped}
    p = tmp_path / "model.json"
    persist.write_model(p, cells, meta)
    blob = json.loads(p.read_text())
    assert blob["policy"]["RARITY_A"] == 1e-4 and blob["version"] == persist.MODEL_VERSION
    cells2, meta2 = persist.read_model(p)
    g = m
    assert np.allclose(fit.zscores(cells["TWAP_VWAP"], g)[0],
                       fit.zscores(cells2["TWAP_VWAP"], g)[0])
    assert meta2["fit_end"] == "2026-06-30"


def test_rejects_other_version(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"version": 99}))
    with pytest.raises(ValueError, match="version"):
        persist.read_model(p)
