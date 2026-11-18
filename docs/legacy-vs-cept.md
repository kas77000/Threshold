# Legacy PerfThreshold vs CEPT — technical comparison

Audience: whoever runs, reviews or validates the outlier control.
Companion documents:
- `docs/superpowers/specs/2026-11-15-conditional-threshold-design.md` (the full CEPT design and evidence)
- `legacy/docs/method.md` (the legacy method, stage by stage)
- `docs/cept-briefing.md` (non-technical summary)

Both programs are in this workspace and run on the same extract:

```
python legacy/run.py fit   --csv year.csv  --out fits/legacy
python legacy/run.py score --csv month.csv --bands fits/legacy/bands.json --out review/legacy
python -m cept fit         --csv year.csv  --out fits/cept
python -m cept score       --csv month.csv --model fits/cept/model.json --out review/cept
```

---

## 1. The two methods in one line each

| | Legacy PerfThreshold | CEPT |
|---|---|---|
| Question asked | Is this order's `ePvwap/Sprd` far from **the book's** average? | Is this order's `ePvwap/Sprd` far from **what was expected for this order's conditions**? |
| Band | Per benchmark (TWAP and VWAP each get their own): `MAX(mean + 4σ, P99.5)` upper, mirrored lower, on the raw metric | **One threshold per strategy group** (TWAP and VWAP together by default), applied per order: `expected(conditions) ± threshold × expected dispersion(conditions)`. Each order's band is in `scored_orders.csv` as `band_lo`/`band_hi` |
| Threshold | k = 4 on the raw σ | Per side `MAX(4, Student-t quantile at 1 in 10,000)` (Tier A); `MAX(3, 1 in 1,000)` (Tier B, watch list) |

## 2. Stage by stage

| Stage | Legacy | CEPT | Why it matters |
|---|---|---|---|
| Loading and row accounting | `rows_in = kept + Σ dropped`, named reasons, unmapped strategies excluded and named | Same design, plus an explicit market filter counted as a drop reason | Unchanged guarantee: no row disappears silently |
| Data quality | Only missing ID/symbol/date/metric | Record rules: spread implausible, metric ≠ Pvwap/Sprd, PR ∉ (0,100], Dur ≤ 0, %OPEN+%CLOSE > 100, %Adv ≤ 0, Vol ≤ 0, FR ∉ (0,100], missing condition. Rules test the **record**, never the performance | A data error is routed to a data-exception file, not into the review queue; it also no longer inflates σ for everyone else |
| Expected performance | One mean per benchmark | Robust (Huber) regression on difficulty: typical vol, vol surprise, duration, spread, %ADV, √PR, %OPEN, %CLOSE, side, and strategy within the group (so pooling TWAP and VWAP under one threshold does not blur their normal difference) | A long, high-participation order in a volatile stock is compared with what such orders normally get |
| Dispersion | One σ per benchmark | Robust regression of log\|residual\| on the same conditions | A 5-spread miss is routine on a 300-minute, 40%-PR order and extreme on a 20-minute, 2%-PR one |
| Standardisation | `(x − mean) / σ`, classical, so the year's own outliers widen σ | Conditional z, then re-centred and rescaled by median/MAD | Catastrophic orders in the fit year no longer mask others |
| Tail | Assumes 4σ is rare; P99.5 floor | Student-t fitted by quantile matching on the 50–99.5th percentiles of \|z\| (robust to contamination). Threshold never below 4σ | The "1 in 10,000" claim is true, not assumed; on a near-normal book Tier A is exactly 4σ |
| Second line | — | Cohort test: mean z per strategy and per quintile of each condition (incl. fill rate), compared with the **same cohort's level in the fit year**, alert if \|t\| > 4 | Catches gradual or segment-wide deterioration that never produces a single extreme order |
| Excluded from model | n/a | `FR`, `%POST`, `Rev30min` (outcomes) | The model cannot "explain away" bad execution with variables the algo controls |
| Monitoring | LOMO calibration curve, normality table, drift | LOMO backtest with predicted vs realised counts and Kupiec p-values, injected-order detection power, same-day-vol sensitivity refit, drift | Every fit produces the evidence a reviewer will ask for |
| Leakage guard | Refuses to score a window overlapping the fit | Same | Unchanged |
| Output per alert | Value, band, excess | z, expected performance, expected dispersion, excess, plus explanation tags (partial fill, vol surprise, auction-heavy, high PR, high %ADV) | Each alert arrives with its first-line explanation |

## 3. Why the legacy method produces too many alerts

The legacy band is correct arithmetic on the wrong population. A year of
TWAP/VWAP orders mixes orders whose ordinary dispersion differs by an order of
magnitude (spread, volatility, duration and participation all differ). A
mixture of normal distributions with different variances **is** fat-tailed
(`legacy/docs/fat-tails.md` §6 shows this on the legacy project's own data).
So the orders beyond `mean ± 4σ` are mostly the high-dispersion orders doing
exactly what they normally do, not bad executions. Three consequences follow:

1. **False alerts**: ordinary hard orders are flagged every month.
2. **Missed alerts**: a genuinely bad order on an easy stock (tight spread,
   short duration) can be 6 conditional σ off and still sit inside the pooled
   band.
3. **Volatile months flood the queue**: the band is fixed in raw spreads, so
   when the market gets harder everything moves outwards together.

Lowering k or adding a percentile floor cannot fix this, because the problem
is *who* is compared with whom, not how wide the band is. CEPT changes the
comparison.

## 4. Measured head-to-head (synthetic, both real CLIs)

Both programs were run through their real command-line interfaces
(`python -m lab.compare_legacy --seeds 2`). Raw results:
`lab/results/compare_legacy_raw.csv`, summary:
`lab/results/compare_legacy_2026-11-18.txt`. CEPT runs with its default single strategy group (TWAP and VWAP under one threshold). Legacy keeps its own design, one band per benchmark. The earlier run with CEPT split per benchmark gave the same picture: `compare_legacy_2026-11-16_per_benchmark.txt`.

Setup:
- Each scenario has a 12-month fit year and one held-out month of 2,500 orders, averaged over 2 seeds.
- Each month contains 6 planted bad orders, 4.5–7σ off for their own conditions.
- Ground-truth labels are removed before either program sees the file.

How to read the columns:
- **Flagged** is the size of the review queue.
- **False** counts clean orders in the queue.
- **Caught** is the share of the 6 planted bad orders that were flagged.
- **Tier B** is CEPT's watch list.
  - About 2.5 entries a month are expected by construction (1 in 1,000 × 2,500).
  - The watch list gets a light-touch review, not a full justification.
- **Cohort** counts cohort cells in alert.

| Scenario | Legacy flagged | Legacy false | Legacy caught | **CEPT Tier A flagged** | **Tier A false** | **Tier A caught** | Tier B | Cohort |
|---|---|---|---|---|---|---|---|---|
| S1 clean, near-normal book | 6.0 | 0.0 | 100% | 6.0 | 0.0 | 100% | 3.5 | 0 |
| S2 ordinary mixture (main case) | 20.5 | 16.5 | 67% | **6.0** | **0.0** | **100%** | 2.5 | 0 |
| S4 data errors in the book | 3.5 | 1.0 | **0%** (2.5 data errors in queue) | 6.0 | 0.0 | 100% | 2.5 | 0 |
| S5 volatile month (1.7× vol) | **52.0** | 48.0 | 67% | 6.0 | 0.0 | 100% | 2.5 | 0 |
| S6 TWAP strategy decays | 25.5 | 23.0 | 42% | 7.0 | 1.0 | 100% | 5.0 | **30.5** (incl. the TWAP strategy cohort) |
| S7 catastrophic orders in fit year | 8.0 | 6.0 | 33% | 6.0 | 0.0 | 100% | 3.0 | 0 |
| S8 high-PR orders decay | 22.5 | 15.5 | 67% | 8.0 | 0.0 | 100% | 17.5 | **12.5** |
| S10 thin book (600 orders/month)* | 6.0 | 3.0 | 50% | 5.5 | 0.0 | 92% | 0.5 | 0 |
| S3 genuinely fat tails (t4) † | 24.5 | 21.5 | 50% | 0.0 | 0.0 | 0% | 2.5 (17% caught) | 0 |
| S9 model misspecified † | 17.0 | 16.5 | 8% | 2.0 | 2.0 | 0% | 8.0 (67% caught) | 0 |
| S11 misspecified + fat † | 23.5 | 21.0 | 42% | 0.0 | 0.0 | 0% | 2.5 | 0 |

\* S10 has 7,200 orders a year. Pooling TWAP and VWAP into one group keeps
it above CEPT's production minimum of 5,000 per group, which split per
benchmark (3,600 each) it was not. A group below the minimum is refused, not
fitted silently.

† Hard cases. In these books, a 4.5–7σ deviation is not rare among ordinary
orders, so no method can flag it without over-alerting. Legacy does not catch
more here: it catches fewer bad orders than its own false alerts. CEPT stays
calibrated, keeping its Tier A count near the predicted level. The bad orders
it catches there are found on the watch list, and the `detection_power.csv`
file reports this loss of sensitivity at every fit.

What the table shows:
1. **Legacy over-alerts wherever the book is a mixture.** That covers every
   realistic case: 16–48 false alerts a month, and up to 52 queue entries in a
   volatile month.
2. **Legacy also misses.** In the main case it catches 33–67% of the planted
   bad orders, because they hide inside the wide pooled band. With data
   errors present it catches **0%**: the bad records inflate σ and then take
   the queue places themselves.
3. **CEPT Tier A is the planted bad orders and almost nothing else** in every
   well-specified scenario. That is 0–1 false alerts a month with 92–100% of
   bad orders caught.
4. **Gradual decay (S6, S8) is caught by the cohort test.** Legacy has no
   equivalent; it only shows a larger queue.
5. **On a clean, near-normal book (S1) the two agree.** CEPT reduces to plain
   mean ± 4σ when the book's conditions do not vary.

## 5. What does not change

- Fit once, freeze to a file, score unseen months: same as legacy.
- Policy values fixed before data is seen, never tuned to the alert count.
  Legacy has a `--target` mode that searches k to hit an alert count. CEPT
  deliberately has none, because a threshold chosen from the count is exactly
  what an SFC reviewer would refute.
- Row accounting, unmapped-strategy exclusion, the overlap guard.

## 6. When to prefer which

- **CEPT** is the proposed control for SFC purposes.
- **Legacy** stays available as the benchmark: run both on the same extract to
  show the reviewer what the old method would have flagged and why those
  orders are explained by their conditions. It is also the fall-back if a
  market lacks the condition columns CEPT needs.

## 7. Known limits of CEPT (stated, not hidden)

- If conditional residuals are genuinely fat-tailed (lab scenario S3, t4),
  Tier A honestly widens and catches only very large misses; detection then
  relies on Tier B and the cohort test. The detection-power file reports the
  real sensitivity at every fit.
- The model is only as good as its covariates. Misspecification (S9, S11) was
  tested: CEPT stays calibrated (it widens rather than floods) and the lost
  power is reported.
- Results here are synthetic. The decisive test is the real HK extract.
