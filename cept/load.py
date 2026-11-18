"""Read the extract and account for every row that leaves."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cept import config, schema

MISSING_ORDER_ID = "missing order id"
MISSING_SYMBOL = "missing symbol"
MISSING_DATE = "missing or unparseable date"
UNMAPPED_STRATEGY = "unmapped strategy"
OTHER_MARKET = "other market"
MISSING_METRIC = "missing metric"
_REASONS = [MISSING_ORDER_ID, MISSING_SYMBOL, MISSING_DATE,
            UNMAPPED_STRATEGY, OTHER_MARKET, MISSING_METRIC]


@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_kept: int = 0
    dropped: dict = field(default_factory=dict)
    unmapped_strategies: dict = field(default_factory=dict)
    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None

    def to_frame(self) -> pd.DataFrame:
        rows = [{"reason": r, "rows": n} for r, n in self.dropped.items()]
        return pd.DataFrame(rows, columns=["reason", "rows"])

    def summary(self) -> str:
        lines = [f"rows in  : {self.rows_in:,}", f"rows kept: {self.rows_kept:,}"]
        lines += [f"  dropped ({r}): {n:,}" for r, n in self.dropped.items() if n]
        if self.unmapped_strategies:
            lines.append("  UNMAPPED STRATEGIES (excluded, not defaulted): "
                         + ", ".join(f"{s} ({n})" for s, n in
                                     sorted(self.unmapped_strategies.items())))
        if self.date_min is not None:
            lines.append(f"date range: {self.date_min.date()} .. "
                         f"{self.date_max.date()}")
        return "\n".join(lines)


def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported extension '{ext}'. Use .csv, .xlsx or .parquet.")


def prepare(raw: pd.DataFrame, market: str | None = None
            ) -> tuple[pd.DataFrame, CleaningReport]:
    inverse = {v: k for k, v in config.COLUMN_MAP.items()}
    missing = [inverse[c] for c in schema.REQUIRED + schema.CONDITIONS
               if inverse[c] not in raw.columns]
    if missing:
        raise ValueError(
            "Extract is missing required column(s): " + ", ".join(missing)
            + ". The model needs every condition column; see README "
            "'Input file'. Rename columns in cept/config.py COLUMN_MAP if your "
            "extract uses other names.")

    rep = CleaningReport(rows_in=len(raw), dropped={r: 0 for r in _REASONS})
    present = {k: v for k, v in config.COLUMN_MAP.items() if k in raw.columns}
    df = raw.rename(columns=present)[list(present.values())].copy()

    sym = df[schema.SYMBOL].astype(str).str.strip()
    df[schema.MARKET] = sym.str[-2:].str.upper()
    df.loc[sym.isin(["", "nan", "None"]), schema.MARKET] = pd.NA
    df[schema.GROUP] = df[schema.ALGO].map(config.group_for)
    df[schema.ORDER_DATE] = pd.to_datetime(df[schema.ORDER_DATE],
                                           errors="coerce").dt.normalize()
    for col in schema.NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df[schema.METRIC] = df[schema.METRIC].replace([np.inf, -np.inf], np.nan)

    oid = df[schema.ORDER_ID].astype(str).str.strip()
    other_mkt = (pd.Series(False, index=df.index) if market is None
                 else df[schema.MARKET] != str(market).upper())
    checks = [
        (MISSING_ORDER_ID, df[schema.ORDER_ID].isna() | oid.isin(["", "nan", "None"])),
        (MISSING_SYMBOL, df[schema.MARKET].isna()),
        (MISSING_DATE, df[schema.ORDER_DATE].isna()),
        (UNMAPPED_STRATEGY, df[schema.GROUP].isna()),
        (OTHER_MARKET, other_mkt),
        (MISSING_METRIC, df[schema.METRIC].isna()),
    ]
    alive = pd.Series(True, index=df.index)
    for reason, failed in checks:
        hit = alive & failed.fillna(True).astype(bool)
        rep.dropped[reason] = int(hit.sum())
        if reason == UNMAPPED_STRATEGY and hit.any():
            vc = df.loc[hit, schema.ALGO].astype(str).str.strip().value_counts()
            rep.unmapped_strategies = {str(k): int(v) for k, v in vc.items()}
        alive &= ~hit

    df = df[alive].reset_index(drop=True)
    rep.rows_kept = len(df)
    if len(df):
        rep.date_min = df[schema.ORDER_DATE].min()
        rep.date_max = df[schema.ORDER_DATE].max()
    return df, rep


def load(path: str, market: str | None = None):
    return prepare(read_any(path), market=market)
