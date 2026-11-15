# Conditional performance threshold — design

**Date:** 2026-11-15
**Status:** draft for review. Lab-validated on synthetic data; not yet run on a real extract.
**Scope of v1:** Hong Kong, TWAP and VWAP (metric `ePvwap/Sprd`). Other markets and
strategies extend the same method (§9).
**Predecessor:** `Desktop/Projects/PerfThreshold` (raw `mean ± 4σ` per benchmark cell).
Evidence: `lab/` in this folder (`python -m lab.bench`, `python -m lab.frontier`),
results frozen in `lab/results/*_2026-11-15.txt`.

---

## 1. The problem, stated so it can be defended

The current rule applies `mean ± 4σ` to a whole year of HK orders at once. It
produces too many outliers. This is not because the book executes badly. It is
because **one band is being applied to orders with very different, perfectly
ordinary, dispersion**:

- A 40-minute VWAP in a liquid large cap with a wide spread moves very little
  in spread units.
- A full-day order in a volatile small cap at 20% participation moves a lot.

Pooling them produces an apparently fat tail even when neither population has
one (`PerfThreshold/docs/fat-tails.md` §6). In the lab, the raw rule flags
about 10 clean orders a month on an ordinary mixed book (S2). In a month that
is merely more volatile (S5) it flags 38, and it catches only 50–63% of
genuinely bad orders.

So the goal is **not "fewer outliers"**. It is fewer *false positives*, meaning
orders flagged for a reason already visible in their own conditions, while
proving that genuinely bad execution is still caught. That framing is the one
regulators endorse:

- **HKMA (Feb 2023, TM guidance §2.8–2.9):** segmentation to *"generate more
  targeted, higher-quality alerts"*, and statistical tools such as
  above-/below-the-line testing *"to reduce the number of false positive
  alerts."*
- **SFC Best Execution thematic review (Jan 2018):** expects *"clearly defined
  deviation thresholds"* and TCA *"calibrated… based on their clients' trading
  strategies"*.
- **SFC ALP review (Apr 2018), the failure to avoid:** *"The analysis and
  justification of the appropriateness of the criteria and thresholds used for
  the post-trade reviews were not documented, nor were they subject to
  periodic review."*
- **FCA TR14/13:** the criticism is *"unsuitably wide tolerances"*. Raising k
  until the count is comfortable is exactly what gets refuted. Conditioning is
  not.
- **SFC Code of Conduct, Schedule 7 §3.3.2:** firms *"should regularly conduct
  post-trade reviews"* of algorithmic trading.

**The design principle that follows:** every parameter is fixed by policy
*before* the data is consulted. The alert count is a measured consequence,
predicted in advance and back-tested every month.

---

## 2. The method in one view

```
 extract ──► 1 DATA QUALITY ──► data-exception queue (fixed, not justified as execution)
                 │ clean records
                 ▼
             2 EXPECTED PERFORMANCE MODEL   (fitted on the year, frozen)
               expected centre  μ(x)   and   expected dispersion  s(x)
               x = order conditions: Vol, Dur, Sprd, %Adv, PR, %OPEN, %CLOSE, Side
                 │
                 ▼
             3 CONDITIONAL z = (ePvwap/Sprd − μ(x)) / s(x)    robustly standardised
                 │
        ┌────────┴──────────┐
        ▼                   ▼
   4 ORDER TEST          5 COHORT TEST
   Tier A: rarer than    mean z per strategy and per condition bucket;
   1 in 10,000 given     |t| > 4 → systematic deterioration alert
   its conditions,
   never below 4σ
   Tier B: rarer than 1 in 1,000 → watch list
        │                   │
        └────────┬──────────┘
                 ▼
             6 MONITORING: predicted vs realised exceedances (backtest),
               drift, detection power on injected orders, annual review
```

---

## 3. Stage by stage

### 3.1 Data quality (record rules, never performance rules)

A record that fails any rule goes to a **data-exception queue**. It is counted,
named and corrected at source. It never enters the fit, and it is never
justified as an execution outlier.

| Rule | Condition |
|---|---|
| spread implausible | `Sprd` outside [0.5, 500] bps |
| metric inconsistent | `Pvwap/Sprd` differs from `ePvwap/Sprd` by > 1% |
| participation impossible | `PR` ≤ 0 or > 100 |
| duration non-positive | `Dur` ≤ 0 |
| auction share impossible | `%OPEN + %CLOSE` > 100 |
| ADV non-positive | `%Adv` ≤ 0 |
| volatility non-positive | `Vol` ≤ 0 (High = Low on a traded day) |
| fill rate impossible | `FR` ≤ 0 or > 100 |

**Why it matters (lab S4):**
- Without this layer, six data errors in a year inflate σ enough that the raw
  rule catches **3%** of genuinely bad orders.
- With it, all data errors are routed correctly and detection returns to 100%.

The rules test the *record*, never the performance value. A data-quality rule
of the form "performance too large" would be a hidden second threshold, and
easy to attack.

### 3.2 Expected performance model

For each benchmark cell (TWAP, VWAP), the model is fitted on the year after
data quality:

- **Centre** `μ(x)`: Huber M-regression (Huber 1964) of `ePvwap/Sprd` on
  `[log Vol, log Dur, log Sprd, log %Adv, √(PR/10), %OPEN, %CLOSE, side]`.
  Bounded influence means a bad order cannot pull the expectation toward
  itself.
- **Dispersion** `s(x)`: Huber regression of `log|residual|` on the same
  conditions. This is the multiplicative heteroskedasticity model (Harvey
  1976), the location–scale structure of GAMLSS (Rigby & Stasinopoulos 2005).
  A second pass reweights the centre fit by `1/s²`.
- **Standardisation:** `z = (y − μ(x)) / s(x)`, then centred on its median and
  scaled by 1.4826·MAD. On normal data, "1" equals one σ.

**Why these covariates and no others.** They describe the order's *difficulty*,
meaning conditions set by the market and by the client's instruction (size,
horizon, auction participation). They do not describe the algo's own outcome.
Outcome variables (`%POST`, `Rev30min`) are excluded on purpose: conditioning
on them would explain bad execution away. This is the logic of pre-trade cost
models (Almgren et al. 2005). The industry analogue is the Kissell z-score,
(performance − expected cost) / timing risk.

**`Vol` is the single-day Parkinson volatility of the order's own day**
(confirmed 2026-11-15): `sqrt(1/(4·ln 2)) · ln(PX_HIGH/PX_LOW)`, as a daily
fraction. It captures news and event days, the main source of unexplained
dispersion in S9 and S11. It is partly endogenous: a large order can widen the
day's range and so excuse itself. No extra data is needed to control for this:
- **Typical volatility** `Vol_typ(sym)` is the median `Vol` of that symbol's
  orders in the fit year. Symbols with fewer than 20 orders fall back to the
  cell median.
- The model uses `log Vol_typ` and `log(Vol / Vol_typ)` (the day's
  surprise) as two covariates.
- **Sensitivity (`vol_sensitivity.csv`):** the model is refitted without the
  day-surprise term, and every Tier A alert cleared *only* by it is listed. The
  high-`%Adv` orders among them are shown explicitly, so it is visible that no
  order excused itself.
- **Data quality:** `Vol ≤ 0` (High = Low on a traded day) is a data exception.

Units do not affect the model, because Vol enters as a log.

**Size and completion** (confirmed 2026-11-15):
- `%Adv` is the *executed* size as a percentage of ADV.
- `#Shares` is the *ordered* quantity.
- `FR` is the fill rate, executed as a percentage of ordered. Partial fills
  occur, and there is no limit price in the extract.

**`FR` is deliberately NOT a model covariate.** Like `%POST` or reversion,
completion is partly an *outcome* of the algo: an order that failed to finish
because of an algo or routing problem would be excused if the model expected
it. Instead:
- every alert carries an explanation tag `partial fill (FR < 80%)`;
- `FR` is a cohort-test feature, so systematic under-completion raises a
  cohort alert;
- `FR` outside (0, 100] is a data exception.

If real data shows partial fills dominating Tier A, the decision is revisited
on evidence (for example, by separating client cancels from algo shortfall),
not by quietly conditioning them away.

**Why log-linear and not flexible.** The dispersion of VWAP-relative
performance scales like `Vol·√Dur / Sprd`, which is linear in logs.
- In the lab, additive splines (M9/M10) did **not** reduce misspecification
  error (S9).
- They also extrapolated badly into a volatile month: 7–24 false alerts on S5,
  against 0 for log-linear.

The simpler model generalises better, and each coefficient can be read and
challenged.

### 3.3 Order test: rarity fixed by policy

For each cell and each side separately (performance is skewed), the threshold
is:

```
Tier A (must review):  T_A = MAX( 4 ,  q_t(1/20,000) )   one-sided → 1 in 10,000 two-sided
Tier B (watch list):   T_B = MAX( 3 ,  q_t(1/2,000) )    one-sided → 1 in 1,000 two-sided
```

- **The floor of 4 keeps the language the SFC already understands.** When
  conditional residuals are near-normal, which is the case whenever the
  conditions explain the mixture, `T_A` is **exactly 4σ of the order's own
  expected dispersion**.
- **The rarity term makes the coverage claim true when tails are fat.** "4σ
  covers 99.994%" holds only for a Gaussian. When the residuals are fat-tailed,
  the threshold widens to where *this data's* tail really reaches 1 in 10,000.
  The same MAX structure as the current rule then does its job honestly.
- **The tail is estimated by quantile matching, not maximum likelihood.** A
  Student-t `(df, scale)` is fitted per side to the 50th–99.5th percentiles of
  `|z|`, then extrapolated to the policy rarity (McNeil & Frey 2000 is the
  conditional-EVT template).
  - Only the body and the moderate tail inform the fit, so the year's own
    extreme orders, the ones the band exists to catch, cannot widen the
    threshold that judges them.
  - In the lab, the ML t-fit (M7) and a GPD peaks-over-threshold fit (M5) were
    both inflated by 0.1% contamination. They caught only 23–87% of bad orders
    on books where the quantile-matched version (M8) caught 97–100%.

*Lab status:* Tier A is validated as specified. The Tier B figures in §4.2
were measured with a floor of 4. The floor of 3, which sets Tier B at 3.29σ on
a near-normal book (about 2.5 watch-list entries a month at 2,500 orders), is
to be verified by the implementation's tests.

**Expected false alerts per month are known in advance:** about `n × 1/10,000`
for Tier A. At 2,500 HK orders a month that is about 0.25, roughly 3 a year.
The monthly backtest (§3.5) verifies it.

**Why 1 in 10,000:**
- It is a round, policy-level number close to the Gaussian 4σ rate (1 in
  15,787).
- Together with the floor, it reduces exactly to 4σ on a well-behaved book.
- The frontier (§4.2) shows what each alternative would cost.
- It is fixed before the year is examined and recorded in the band file with
  its rationale.

### 3.4 Cohort test: deterioration no single order reveals

If a whole strategy degrades by 0.6σ (lab S6), or only high-participation
orders degrade (S8), no individual order crosses any threshold. The cohort
test catches this:

- **What it tests:** the month's mean conditional z, winsorised at ±4, for
  each strategy and for each quintile of `PR, %Adv, Vol, Dur, Sprd, %CLOSE, FR`.
  Quintile edges come from the fit year. Each mean is tested against zero.
  `|t| > 4` raises a **cohort alert**.
- **Lab result:** it fired on every seed of S6 and S8. Across the 45 control
  runs (S1–S5, S7, S9–S11, 5 seeds each) it raised exactly one spurious cohort
  alert, in S7.

The cohort test is also the answer to the obvious challenge, *"your model
explains bad execution away"*. The model is frozen at fit time. Execution that
gets worse afterwards, systematically, shows up here, even when it is
correlated with a covariate.

### 3.5 Monitoring and governance outputs

| Control | What it shows | Basis |
|---|---|---|
| **Backtest** | Realised Tier A/B exceedances on clean orders vs predicted, with a binomial (Kupiec) p-value. Persistent excess means recalibrate. | SR 26-2 outcomes analysis |
| **Above/below-the-line sample** | Each month, review a sample of orders just below T_A, to evidence that nothing material sits under the line | HKMA §2.9, FCA MW86 |
| **Injected detection test** | At each fit, plant orders at 5σ, 8σ, 12σ and 20σ true conditional deviation, and report the share caught | FCA MW86 (synthetic data) |
| **Drift** | This month's condition medians vs the fit year's (from PerfThreshold) | separates "harder market" from "worse execution" |
| **Model card** | Coefficients, diagnostics (tail df per side, sd/MAD), policy parameters with rationale, fit window, source hash, approver, review date | SFC Sched 7 §3.4.1, ALP 2018 |
| **Annual refit and review** | Policy rarity, covariates and rules reconsidered on evidence | RTS 6 Art 13(6), SFC 2018 |

---

## 4. Evidence (synthetic lab, 5 seeds, 2,500 orders/month, held-out month)

### 4.1 Head to head

False alerts are flags on clean orders, per month. "Bad caught" is the share
of 6 planted bad orders (4.5–7σ for their conditions) that were flagged.

| Scenario | Raw 4σ (today): false | bad caught | **Proposed (M8)**: false | bad caught |
|---|---|---|---|---|
| S1 clean normal | 0 | 100% | 0 | 100% |
| S2 ordinary mixture | 9.6 | 63% | **0** | **100%** |
| S4 + data errors | 0.4 | **3%** | 0 | 100% |
| S5 volatile month | 38.4 | 87% | **0** | **100%** |
| S6 strategy decay | 15.2 | 53% | 0.2 | 100% (+ cohort alert) |
| S7 catastrophic orders in year | 6.6 | 50% | 0 | 97% |
| S8 high-PR decay | 9.4 | 63% | 0 | 97% (+ cohort alert) |
| S10 thin book (600/month) | 2.0 | 50% | 0 | 97% |
| S3 fat residuals (t4) | 18.4 | 60% | 0 | 0% † |
| S9 unmodelled dispersion | 14.6 | 60% | 0.4 | 3% † |
| S11 unmodelled + fat | 16.6 | 50% | 0.2 | 0% † |

† In these books a 4.5–7σ deviation is **not rare**. Under t4 it happens about
once in a few hundred clean orders, so no method can flag it without also
flagging ordinary orders: M4 (robust |z| > 4) catches them at a cost of 10–30
false alerts a month. These rows define the limit of the method. §4.2 says
what the method *does* catch there.

### 4.2 Calibration and detection frontier (M8)

| Scenario | Rarity | Predicted false/month | Realised | Detects 8σ | Detects 12σ | Detects 20σ |
|---|---|---|---|---|---|---|
| S2 mixture | 1/10k | 0.23 | 0.0 | 100% | 100% | 100% |
| S10 thin | 1/10k | 0.05 | 0.0 | 100% | 100% | 100% |
| S9 misspecified | 1/10k | 0.25 | 0.4 | 50% | 100% | 100% |
| S3 fat (t4) | 1/10k | 0.25 | 0.2 | 0% | 22% | 100% |
| S3 fat (t4) | 1/1k (Tier B) | 2.5 | 2.0 | 100% | 100% | 100% |

1. **The stated rarity holds out of sample.** Predicted and realised false
   alerts agree. One exception: S9 at 1/1k (2.5 predicted vs 4.8 realised),
   which is the kind of drift the monthly backtest exists to catch.
2. **Tier B is why the fat-tail case is still covered.** On a t4 book, Tier A
   alone catches only 20σ events. Tier B catches ≥ 8σ at about 2.5 watch-list
   entries a month. Those are reviewed light-touch, each with its automatic
   explanation attached, so they do not each need a full justification.

### 4.3 What was tried and rejected

| Candidate | Why rejected |
|---|---|
| Raw mean ± 4σ (M0) | Flags the mixture, not bad execution; fails in volatile months; masked by data errors |
| Raw median ± 4·MAD (M1) | Robust, but 4× more false alerts than M0 (39/month on S2) |
| %ADV × Vol × Dur buckets (M2) | Barely better than raw (6.6/month); thin buckets |
| Conditional, classical mean ± 4 sd (M3) | Masked by catastrophic orders (83% on S7) |
| Conditional robust \|z\| > 4, no tail term (M4) | Excellent on near-normal books; 10–30 false/month on fat ones, so its coverage claim is untrue there |
| GPD peaks-over-threshold (M5) and BH-FDR on it (M6) | Tail fit inflated by the year's own bad orders: 13–30% detection on clean books |
| Student-t by maximum likelihood (M7) | Same contamination problem, milder (50–87%) |
| Additive splines (M9, M10) | No gain on misspecification; extrapolate badly in volatile months |

---

## 5. Self-challenge: the objections a reviewer will raise

1. **"You tuned it to get few alerts."**
   - No parameter was chosen from the alert count: the rarity (1/10k, 1/1k),
     the floor (4) and the cohort |t| (4) are policy values set beforehand.
   - The count is predicted from the rarity *before* scoring and back-tested
     afterwards.
   - The lab shows the same method raises alerts when execution genuinely
     worsens (S6, S8 cohort alerts).
2. **"The model explains bad execution away."**
   - Covariates are difficulty only; outcome variables are excluded.
   - The model is frozen, so later deterioration is visible.
   - The cohort test covers deterioration correlated with covariates.
3. **"Your z is only as good as your model."**
   - True, which is why misspecification was tested (S9, S11).
   - The method stays calibrated when the model is wrong (it widens rather than
     flooding), and the detection power it loses is *reported*, not hidden.
4. **"A fat-tailed book gets a very wide Tier A band."**
   - Correct, and it is stated: Tier A is then a 20σ net.
   - Tier B (1/1k) and the cohort test exist precisely for this case, and the
     injected detection test reports the real sensitivity at every fit.
5. **"Why trust the tail extrapolation?"**
   - The quantile-matched tail is checked against realised exceedances every
     month.
   - The floor means it can never be *tighter* than 4σ.
6. **"Synthetic data proves nothing about our book."**
   - Agreed. The lab proves the method's properties under controlled truth.
   - The real-data checks in §8 decide whether HK behaves like S2 or like
     S3/S9.

---

## 6. Outputs

**fit:**
- `model.json`: frozen coefficients, standardisation, per-side tail
  parameters, T_A and T_B, policy values with rationale, provenance.
- `model_card.md`
- `coefficients.csv`
- `diagnostics.csv`: tail df, sd/MAD of z, R² of the scale model.
- `detection_power.csv`
- `backtest_lomo.csv`: leave-one-month-out predicted vs realised exceedances.
- Partial-effect plots per covariate.
- QQ plot of z.

**score:**
- `alerts_tierA.csv`: ranked, with expected performance, expected dispersion,
  z, and the conditions that made it unusual.
- `watchlist_tierB.csv`
- `cohort_alerts.csv`
- `data_exceptions.csv`
- `backtest.csv`: this month, predicted vs realised.
- `drift.csv`
- `summary.md`

Reused from PerfThreshold unchanged in spirit: row accounting, the leakage
guard (no scoring of the fit window), unit and scope guards, the frozen
artifact with source hash, and the `check` command.

---

## 7. Code shape

```
Thresholdv2/
  cept/                       # Conditional Expected-Performance Threshold
    schema.py, config.py      # config: covariates, policy rarities, floors, DQ bounds
    load.py                   # from PerfThreshold (row accounting)
    dq.py                     # record rules -> data-exception queue
    model.py                  # LocationScale (Huber centre, log|r| scale)
    tail.py                   # quantile-matched t per side, thresholds
    cohort.py                 # cohort test
    backtest.py               # LOMO + monthly predicted vs realised, Kupiec
    power.py                  # injected detection test
    persist.py, report.py, plots.py, cli.py   # check / fit / score
  lab/                        # this evidence; kept, re-runnable
  tests/
```

---

## 8. To run on the real HK extract (the decisive test)

1. **`fit` on one HK year.** Read `diagnostics.csv`:
   - the tail df per side (≥ 15 means near-normal after conditioning, so the
     S2 regime and a 4σ band);
   - the sd/MAD of z;
   - the R² of the dispersion model.
2. **Compare raw-rule flags with Tier A flags** on the same held-out month.
3. **`backtest_lomo.csv`:** does predicted ≈ realised hold on real months?
4. **`detection_power.csv`:** what size of miss is reliably caught?
5. **Review a sample of the orders the raw rule flagged and the new one
   cleared.** If they are explained by their conditions, that is the evidence
   pack. If any are genuinely bad, the model is missing a covariate.

Column semantics to confirm before fitting:
- Confirmed: `Vol` is same-day Parkinson (daily fraction), `Dur` is minutes,
  `PR` is percent, VWAP is about 47k orders a year in HK (TWAP extra), and
  partial fills occur.
- Confirmed: `%Adv` is executed size / ADV (%), `#Shares` is the ordered
  quantity, `FR` is the fill rate (%), and there is no limit price.
- Optional enrichment, at your site: HK volatility control mechanism
  cooling-off periods during the order (adapted from
  `Threshold/other/detect_limit_orders.py`) as an explanation tag.

---

## 9. Extension

- **Other markets:** one model per (market, benchmark), same policy values. A
  market with too few orders for a stable dispersion model pools with
  comparable markets, using the market as a covariate.
- **Arrival-benchmarked strategies** (`eIS/Sprd`): same method, separate
  model, with arrival-price risk features.
- No change to the policy layer.

---

## References

Huber (1964) Ann. Math. Stat. · Harvey (1976) Econometrica ·
Rigby & Stasinopoulos (2005) JRSS-C (GAMLSS) · McNeil & Frey (2000) J. Empirical Finance ·
Rousseeuw & van Zomeren (1990) JASA (masking) · Almgren et al. (2005) Risk ·
Benjamini & Hochberg (1995) JRSS-B · SFC Code of Conduct §3.2, §18, Sched 7 ·
SFC Best Execution Thematic Review (2018) · SFC ALP Review (2018) ·
HKMA TM Guidance Paper (2023) · FCA TR14/13, Market Watch 86 · Fed SR 26-2 (2026).
