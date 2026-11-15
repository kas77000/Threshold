# Legacy: PerfThreshold (raw mean ± 4σ)

Copied unchanged from `Desktop/Projects/PerfThreshold` (commit 8724d48) so the
old method can be run side by side with CEPT on the same extract. It is the
baseline CEPT is compared against (lab method M0). The original project's
README is `README.md` in this folder.

Run (from the Thresholdv2 root):

    python legacy/run.py check --csv year.csv
    python legacy/run.py fit   --csv year.csv --out fits/legacy_2025-07_2026-06/
    python legacy/run.py score --csv month.csv --bands fits/legacy_2025-07_2026-06/bands.json --out review/legacy_2026-07/

or `cd legacy && python -m perfthreshold ...`.

Tests: `python -m pytest legacy` (from the root) or `python -m pytest` inside
`legacy/`. The root `python -m pytest` runs CEPT's tests only.
