"""Injected detection: what size of miss does the frozen model catch?

Take real orders, worsen (or improve) each by exactly m of its OWN expected
dispersion, and score. Detection then depends on each order's actual noise,
not on an assumed distribution -- this is the synthetic-data calibration test
FCA Market Watch 86 describes, run on the firm's own book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cept import config, schema as s
from cept.fit import zscores

COLS = ["benchmark", "size", "side", "detected_A", "detected_B"]


def injected_detection(cells: dict, df: pd.DataFrame,
                       sizes=config.INJECT_SIZES, n: int = config.INJECT_N,
                       seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for bench, cm in cells.items():
        g = df[df[s.BENCHMARK] == bench]
        g = g.iloc[rng.choice(len(g), min(n, len(g)), replace=False)]
        _, _, disp = zscores(cm, g)
        y = g[s.METRIC].to_numpy(float)
        for m in sizes:
            for side, sign in (("lo", -1.0), ("hi", 1.0)):
                z, _, _ = zscores(cm, g, y + sign * m * disp)
                if side == "lo":
                    da, db = z < -cm.thr["A"]["lo"], z < -cm.thr["B"]["lo"]
                else:
                    da, db = z > cm.thr["A"]["hi"], z > cm.thr["B"]["hi"]
                rows.append({"benchmark": bench, "size": int(m), "side": side,
                             "detected_A": float(da.mean()),
                             "detected_B": float(db.mean())})
    return pd.DataFrame(rows, columns=COLS)
