# CEPT: Conditional Expected-Performance Threshold

Flags HK TWAP/VWAP orders whose `ePvwap/Sprd` is abnormal **given the
order's own conditions**, at policy-fixed rarities. The previous method
(PerfThreshold, raw mean ± 4σ) is kept in `legacy/` and runs on the same file.

Documents:
- `docs/cept-briefing.md`: management briefing on the new method
- `docs/legacy-vs-cept.md`: technical comparison with measured results
- `docs/superpowers/specs/2026-11-15-conditional-threshold-design.md`: full design and evidence
- `legacy/docs/method.md`: the legacy method, stage by stage

---

## 1. Setup (once)

Requirements: Python 3.11 or later (developed on 3.13). Windows, macOS or Linux.

```bash
cd Thresholdv2
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# Windows Git Bash:    source .venv/Scripts/activate
# macOS/Linux:         source .venv/bin/activate
pip install -r requirements.txt
```

With Anaconda you can skip the venv and use the base environment, since
everything in `requirements.txt` ships with Anaconda.

Check the installation:

```bash
python -m pytest -q              # CEPT tests (about 1 minute)
python -m pytest legacy -q       # legacy tests (about 2 minutes)
```

Always run commands **from the `Thresholdv2` folder**.

### Pointing it at your extract

Everything a user edits is in **`cept/config.py`**:

| Setting | What it is | Default |
|---|---|---|
| `METRIC_COLUMN`, `METRIC_UNITS`, `METRIC_BPS_COLUMN` | The performance metric: the **only** place it is named. To use another strategy family, change these three lines, e.g. an arrival-price metric and its bps column. Set `METRIC_BPS_COLUMN = None` if there is no bps partner column. `score` refuses (exit 2) a model fitted on a different metric, so refit after changing it. | `ePvwap/Sprd`, `spreads`, `Pvwap` |
| `COLUMN_MAP` | Your extract's column names → internal names. Edit the **left** side if your headers differ. | The names in §2 |
| `ALGO_BENCHMARK` | Which `Strategy` values are in scope, and their benchmark. A strategy not listed is **excluded and reported by name**, never guessed. | `VWAP→VWAP`, `TWAP→TWAP`, `TMX→TWAP` |
| `DEFAULT_MARKET` | Market kept (last 2 characters of `Sym`, e.g. `0700 HK`). Can also be set per run with `--market`. | `HK` |
| Policy values (`RARITY_A`, `FLOOR_A`, `RARITY_B`, `FLOOR_B`, `COHORT_T`) | The thresholds. **Do not change them to get fewer alerts**: they are the governance record and are copied into every model file. | 1/10,000 & 4σ; 1/1,000 & 3σ; \|t\| > 4 |

The legacy program has its own copy of these settings in
`legacy/perfthreshold/config.py`.

---

## 2. Input file

One row per order. Formats: `.csv`, `.xlsx` or `.parquet`. Use one file for the
fit year (12 months) and a separate file for each month to score.

| Column | Meaning / units | CEPT | Legacy |
|---|---|---|---|
| `aggrTgtId` | Order ID | **required** | **required** |
| `Strategy` | Algo name (TWAP, VWAP, TMX…), mapped by `ALGO_BENCHMARK` | **required** | **required** |
| `Sym` | Symbol with market suffix, e.g. `0700 HK` | **required** | **required** |
| `Date` | Order date | **required** | **required** |
| `ePvwap/Sprd` | Performance vs interval VWAP, in spreads (the metric) | **required** | **required** |
| `Vol` | Same-day Parkinson volatility, daily fraction: `sqrt(1/(4 ln 2)) · ln(High/Low)` | **required** | optional (drift) |
| `Dur` | Order duration, minutes | **required** | optional (drift) |
| `Sprd` | Average spread, bps | **required** | optional (drift) |
| `%Adv` | Executed size as % of ADV | **required** | optional (drift) |
| `PR` | Participation rate, % | **required** | optional (drift) |
| `%OPEN` | % of execution in the opening auction | **required** | optional |
| `%CLOSE` | % of execution in the closing auction | **required** | optional |
| `Side` | `Buy` / `Sell` | **required** | optional |
| `Pvwap` | Performance vs VWAP in bps (checks `ePvwap/Sprd = Pvwap/Sprd`) | recommended | optional |
| `FR` | Fill rate, % of ordered quantity. An **outcome**: used for tags and the cohort test, never in the model | recommended | — |
| `#Shares` | Ordered quantity | optional | optional |
| `$Mln`, `%POST`, `Rev30min` | Notional, passive %, 30-min reversion: carried through, never in the model | optional | optional |

- If a **required column is absent**, the program stops and names the column.
- If a **value is empty or impossible** in a row, the row goes to
  `data_exceptions.csv` with the reason: for example PR > 100, Dur ≤ 0,
  `%OPEN + %CLOSE > 100`, or a metric inconsistent with `Pvwap/Sprd`. Such rows
  are never silently dropped and never sent to the alert queue.
- Every row is accounted for: `rows in = rows kept + Σ dropped by reason`,
  printed on every run and written to `cleaning_report.csv`.

---

## 3. Running CEPT (the new method)

```bash
# 1. Inspect the extract: columns, row accounting, data exceptions, conditions
python -m cept check --csv year.csv

# 2. Fit on a year and freeze the model (about 30 s for 60k orders)
python -m cept fit   --csv year.csv --out fits/HK_2025-07_2026-06

# 3. Score a later month with the frozen model
python -m cept score --csv month.csv --model fits/HK_2025-07_2026-06/model.json --out review/HK_2026-07
```

Options:
- `--market HK`: market to keep (`check`, `fit`)
- `fit --min-cell-n 5000`: minimum clean orders per benchmark
- `fit --no-lomo`: skip the leave-one-month-out back-test (faster, for trials only)

`score` refuses, with exit code 2, a month that overlaps the fit year,
because the result would be circular.

**`fit` writes:**
- `model.json`: the frozen model with its policy values
- `summary.md`: the model card. Read it first: tail df per side, thresholds, coefficients, back-test.
- `lomo_backtest.csv`: predicted vs realised alerts per month, with Kupiec p-values
- `detection_power.csv`: share of planted bad orders caught, by size
- `vol_sensitivity.csv`: results with the same-day volatility term removed
- `coefficients.csv`, `qq_*.png`, `coef_*.png`
- `in_sample_scored.csv`, `data_exceptions.csv`, `cleaning_report.csv`

**`score` writes:**
- `alerts_tierA.csv`: orders to justify
- `watchlist_tierB.csv`: light-touch review
- `cohort.csv`: strategy and segment deterioration test
- `drift.csv`: this month's conditions vs the fit year's
- `scored_orders.csv`: every order with z, expected performance and tags; sort by `z` for sampling
- `data_exceptions.csv`, `backtest.csv`, `summary.md`

---

## 4. Running the legacy method (PerfThreshold, raw mean ± 4σ)

```bash
python legacy/run.py check --csv year.csv
python legacy/run.py fit   --csv year.csv  --out fits/legacy_2025-07_2026-06
python legacy/run.py score --csv month.csv --bands fits/legacy_2025-07_2026-06/bands.json --out review/legacy_2026-07
```

Note that legacy `score` takes `--bands`, where CEPT takes `--model`.
Equivalent: `cd legacy` then `python -m perfthreshold ...`.

Legacy `fit` options:
- `--k 4`: fix k (the default comes from policy)
- `--scope all|markets|groups|group:NAME`
- `--benchmark VWAP`
- `--percentile 99.5`
- `--min-cell-n 2000`
- `--no-plots`

Legacy outputs:
- `bands.json`, `summary.md`, `calibration.csv`, `normality.csv`
- `outliers.csv`: the queue
- `scored_orders.csv`, `drift.csv`

Details: `legacy/README.md` and `legacy/LEGACY.md`.

### Running both on the same extract

```bash
python -m cept fit         --csv year.csv --out fits/cept
python legacy/run.py fit   --csv year.csv --out fits/legacy
python -m cept score       --csv month.csv --model fits/cept/model.json   --out review/cept
python legacy/run.py score --csv month.csv --bands fits/legacy/bands.json --out review/legacy
```

Then compare `review/legacy/outliers.csv` with `review/cept/alerts_tierA.csv`.
For each order the legacy method flags and CEPT does not,
`review/cept/scored_orders.csv` shows its expected performance given its
conditions: the explanation of why it is not an outlier.

---

## 5. Synthetic lab (evidence, no real data needed)

```bash
python -m lab.compare_legacy --seeds 2   # legacy vs CEPT on 11 scenarios (about 15 min)
python -m lab.bench                      # all 11 candidate methods (slow, more than 10 min)
python -m lab.frontier                   # calibration and detection power
```

Results are saved in `lab/results/`.

---

## 6. What to send back after the first real run

Send diagnostics only, not order data:
- the cell table from `fits/.../summary.md` (n, tail df, thresholds)
- `lomo_backtest.csv`
- `detection_power.csv`
- `vol_sensitivity.csv`
- the Tier A, Tier B and cohort counts printed by `score`
