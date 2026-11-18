# Execution outlier detection: method note

*For discussion with the SFC. HK equities, TWAP/VWAP algorithmic orders.*

---

## 1. The rule: mean ± 4 sigma

An order is an **outlier** when its execution performance (versus interval
VWAP, measured in spreads) is more than **4 standard deviations** from the
mean.

This is the same rule as before. The thresholds are fixed in advance, before
the data is examined, and are never adjusted to obtain a particular number of
outliers.

## 2. What we observed

When we applied the rule to a full year of orders, **too many orders were
flagged, and on review most of them were not poor executions.** We traced
this noise to four sources:

| # | Source of noise | Effect |
|---|---|---|
| 1 | **Data errors**: a small number of impossible records (e.g. participation above 100%, zero duration, a performance figure inconsistent with its slippage) | They appear as outliers themselves, and they inflate the sigma for every other order |
| 2 | **Easy and difficult orders measured on one scale**: a 10-minute order in a liquid stock and a full-day order at 40% of volume in an illiquid stock were compared with the same mean and sigma | Difficult orders are flagged routinely even when well executed, while a poor execution on an easy order stays hidden |
| 3 | **A few extreme orders in the reference year** | They pull the mean and widen the sigma, so the band they produce no longer reflects normal execution |
| 4 | **Market conditions**: in a volatile month, all orders naturally disperse more | The number of outliers rises with the market, not with execution quality |

## 3. What we did: one correction for each source of noise

The 4-sigma rule is unchanged. We changed what the mean and the sigma are
computed on.

**1. Check the data before any statistics.**
Every record is tested for plausibility before the calculation. Impossible
records go to a separate data-exception list for operations to correct; they
are neither counted as outliers nor used to compute the band. The tests check
whether the **record** is possible. They never test whether the performance
is good or bad, so they cannot be used to remove a genuine outlier.

**2. Compare each order with comparable orders.**
The mean and the sigma are no longer one number for the whole book. They are
estimated **for each order's own conditions**:
- stock volatility, both its usual level and that day's
- order duration
- bid-ask spread
- order size relative to daily volume
- participation rate
- share executed in the opening and closing auctions
- side
- strategy

The order is then flagged if it is more than 4 sigma from *its* expected
performance. It is still mean ± 4σ, applied like-for-like. An analogy: a
delivery is judged against the expected time for *its* route, not against
the average of all deliveries.

Only factors that make an order harder to execute are used. Outcomes such as
fill rate are deliberately **not** used, so the model cannot explain away a
poor execution.

**3. Estimate the mean and sigma robustly.**
We use standard robust statistics (Huber regression, median-based scaling).
A handful of extreme orders in the reference year therefore cannot distort
the band for everyone else, and genuine outliers stay visible.

**4. Verify that "4 sigma" really is rare on our data.**
Under a normal distribution, a 4-sigma event occurs about once in 16,000
cases. We check on our own data that orders beyond the threshold are indeed
rarer than **1 in 10,000**. If our data shows fatter tails, the threshold is
widened just enough to keep that rarity. It is **never narrower than 4
sigma**. On our test data, the threshold came out at 4.0–4.2 sigma.

## 4. Safeguards that stay in place

- **One threshold per strategy group.** TWAP and VWAP are assessed with one
  threshold. Strategies with similar behaviour can be added to a group later.
- **Fixed for a year.** The model is fitted once on a full year and frozen.
  Each later month is scored against it, so any deterioration shows up.
- **Gradual deterioration is monitored separately.** A monthly test checks
  whether a whole strategy, or a whole type of order, has become worse than
  it was in the reference year, even when no single order is extreme.
- **Checked every time.** Each fit produces:
  - a back-test of predicted against actual outliers, month by month;
  - a detection test, in which deliberately bad orders are planted and we
    measure how many are caught;
  - a sensitivity check.
- **Full traceability.** Every order in the extract is accounted for. Each
  outlier is reported with its expected performance, its own band and the
  context (e.g. partial fill, volatility spike).

## 5. Effect

We tested both methods on simulated order books where the answer is known.
Each test month contained 6 deliberately bad executions:

| Test month | Before: orders flagged | of which not bad | bad ones caught | **Now: orders flagged** | of which not bad | bad ones caught |
|---|---|---|---|---|---|---|
| Normal month | 20.5 | 16.5 | 67% | **6** | **0** | **100%** |
| Volatile month | 52 | 48 | 67% | **6** | **0** | **100%** |
| Month with data errors | 3.5 | 1 | 0% | **6** | **0** | **100%** |

The figures are averages over two simulated books. Fewer orders to justify,
and more of the genuinely bad ones are caught. The decisive check is the run
on our real HK data.

## 6. In one sentence

> We still flag orders beyond 4 sigma. We first remove data errors, then
> measure each order against the mean and sigma of comparable orders,
> estimated robustly, and we verify that the threshold is as rare as it
> claims to be.
