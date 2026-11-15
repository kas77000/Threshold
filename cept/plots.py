"""Charts that let a reviewer see the method rather than take it on trust."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from cept.features import feature_names  # noqa: E402


def qq_plot(z, path, title: str) -> None:
    z = np.sort(np.asarray(z, float)[np.isfinite(z)])
    q = stats.norm.ppf((np.arange(len(z)) + 0.5) / len(z))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(q, z, ".", ms=2)
    lim = max(abs(q).max(), 4.5)
    ax.plot([-lim, lim], [-lim, lim], lw=1, color="grey")
    ax.axhline(4, ls="--", lw=0.8, color="red")
    ax.axhline(-4, ls="--", lw=0.8, color="red")
    ax.set(xlabel="normal quantile", ylabel="conditional z", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def coef_plot(cm, path) -> None:
    names = feature_names(cm.include_surprise)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, coef, ttl in ((axes[0], cm.ls.loc_coef, "expected performance"),
                          (axes[1], cm.ls.sc_coef, "log expected dispersion")):
        ax.barh(names, coef)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_title(f"{cm.benchmark}: {ttl}\n(effect of +1 sd of condition)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
