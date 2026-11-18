"""The frozen model file: everything needed to score -- and to explain why."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone

from cept import config
from cept.fit import CellModel

MODEL_VERSION = 2   # 2: strategy groups, cohort baselines
_POLICY_KEYS = ["RARITY_A", "FLOOR_A", "RARITY_B", "FLOOR_B", "COHORT_T",
                "POLICY_RATIONALE", "MIN_CELL_N", "MIN_SYMBOL_N", "TAIL_PROBS"]


def policy_snapshot() -> dict:
    return {k: getattr(config, k) for k in _POLICY_KEYS}


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if hasattr(o, "item"):            # numpy scalar
        return _clean(o.item())
    return o


def write_model(path, cells: dict, meta: dict) -> None:
    blob = {"version": MODEL_VERSION,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "metric": config.METRIC_COLUMN, "metric_units": config.METRIC_UNITS,
            "policy": policy_snapshot(), **meta,
            "cells": {k: v.to_dict() for k, v in cells.items()}}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_clean(blob), fh, indent=2, allow_nan=False)


def read_model(path):
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    if int(blob.get("version", 0)) != MODEL_VERSION:
        raise ValueError(f"Model file version {blob.get('version')} != "
                         f"{MODEL_VERSION}; refit rather than guess.")
    cells = {k: CellModel.from_dict(v) for k, v in blob.pop("cells").items()}
    return cells, blob
