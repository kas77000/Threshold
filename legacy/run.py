"""Run the legacy PerfThreshold CLI from anywhere.

    python legacy/run.py check --csv year.csv
    python legacy/run.py fit   --csv year.csv --out fits/2025-07_2026-06/
    python legacy/run.py score --csv aug2026.csv --bands fits/2025-07_2026-06/bands.json --out review/

Equivalent to `cd legacy && python -m perfthreshold ...`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perfthreshold.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
