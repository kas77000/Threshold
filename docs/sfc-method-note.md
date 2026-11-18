# Execution outlier detection: method note

*HK equities, TWAP/VWAP algorithmic orders.*

---

## 1. The rule has not changed

> An order is an **outlier** if its performance is more than **4 sigma** away
> from the **mean**.

Performance is measured against the interval VWAP, in spreads. The threshold
is fixed in advance and is never adjusted to obtain a particular number of
outliers.

## 2. What we found

We first used **one mean and one sigma for all orders**. That flagged many
orders that, on review, were not poor executions.

The reason is that our orders are not alike. A 20-minute order in a calm
stock and a full-day order at 30% of volume in a volatile stock behave very
differently, even when both are executed well. The longer the order and the
more the price moves during it, the further its result can land from VWAP. A
miss of 5 spreads is:

- **normal** for a full-day, high-participation order in a volatile stock;
- **abnormal** for a short order in a calm stock.

With one mean and one sigma for everyone:

- difficult orders are flagged **even when well executed** (noise);
- a poor execution on an easy order **goes unnoticed**.

## 3. What we changed: the mean and sigma come from comparable orders

The threshold is still **4 sigma**. The only change is that each order is
compared with the mean and sigma **of orders like it**.

Illustration (test data, VWAP orders):

| Order | Mean of comparable orders | Sigma of comparable orders | Outlier if outside (4 sigma) | An order at −5 spreads is… |
|---|---|---|---|---|
| Easy | −0.2 | 0.8 | −3.6 to +3.1 | **an outlier** |
| Typical | −0.3 | 1.6 | −6.9 to +6.0 | normal |
| Difficult | −0.1 | 3.1 | −13.0 to +12.3 | normal |

"Comparable" means similar conditions:
- the stock's volatility;
- the order's duration;
- the bid-ask spread;
- the order's size relative to daily volume;
- the participation rate;
- the share executed in the opening and closing auctions;
- side and strategy.

We use only factors that make an order harder to execute. Results of the
execution, such as fill rate, are deliberately **not** used, so poor execution
cannot be explained away.

### Worked example

Three VWAP orders, judged first by the old method and then by the new one.
The figures are illustrative, taken from test data.

| | Order 1 | Order 2 | Order 3 |
|---|---|---|---|
| Profile | 20 minutes, calm stock, 5% of volume | Full day, volatile stock, 30% of volume | Full day, volatile stock, 30% of volume |
| Performance vs VWAP | **−5 spreads** | **−5 spreads** | **−11 spreads** |

**Old method: one mean and one sigma for all orders.**
The mean of all orders is −0.2 and the sigma is 2.6, so every order has the
same band: −0.2 ± 4 × 2.6, i.e. **−10.6 to +10.2 spreads**.

| | Order 1 | Order 2 | Order 3 |
|---|---|---|---|
| Inside −10.6 / +10.2? | yes | yes | **no** |
| Result | not flagged | not flagged | **flagged** |

**New method: the mean and sigma of comparable orders.**
Each order's distance from its mean is counted in sigma:
(performance − mean) ÷ sigma. It is flagged beyond 4.

| | Order 1 | Order 2 | Order 3 |
|---|---|---|---|
| Mean of comparable orders | −0.2 | −0.1 | −0.1 |
| Sigma of comparable orders | 0.8 | 3.1 | 3.1 |
| Distance from the mean | (−5 + 0.2) ÷ 0.8 = **−6.0 sigma** | (−5 + 0.1) ÷ 3.1 = −1.6 sigma | (−11 + 0.1) ÷ 3.1 = −3.5 sigma |
| Its band (4 sigma) | −3.4 to +3.0 | −12.5 to +12.3 | −12.5 to +12.3 |
| Result | **flagged** | not flagged | not flagged |

What changed:
- **Order 1** is a genuinely poor execution. A calm 20-minute order should
  never be 5 spreads off. The old method **missed it**; the new method
  catches it.
- **Order 3** looked extreme only because it was compared with easy orders.
  For a full-day, high-participation order in a volatile stock, −11 spreads is
  within normal range. The old method **flagged it** (noise); the new method
  does not.
- **Order 2** is normal under both methods.

Same rule (4 sigma), same orders. The new method compares each order with
the right reference.

## 4. Three additional clean-ups

1. **Data errors are removed first.** Impossible records (e.g. participation
   above 100%, zero duration) are sent to operations for correction. They are
   neither outliers nor used to compute the mean and sigma. The checks are on
   the record, never on whether performance is good or bad.
2. **A few extreme orders cannot distort the mean and sigma.** We use standard
   statistical methods that are not pulled by a handful of extreme values.
3. **We check that 4 sigma is really rare on our data.** If needed, the
   threshold is widened slightly, but **it is never below 4 sigma**. On test
   data it came out at 4.0–4.2.

## 5. What stays in place

- **One threshold per strategy group:** TWAP and VWAP together.
- **Fixed for a year:** the mean and sigma are estimated once on a full year,
  then applied to each later month.
- **Checked every month:** predicted against actual number of outliers.
- **Gradual decline is also monitored:** for example, a whole strategy getting
  worse even though no single order is extreme.
- **Traceable:** every outlier is reported with its mean, its sigma and its
  band.

## 6. Effect on test data

Each test month contained 6 deliberately poor executions. The figures are
averages over two simulated order books.

| Test month | Before: orders flagged | of which not poor | poor ones caught | **Now: orders flagged** | of which not poor | poor ones caught |
|---|---|---|---|---|---|---|
| Normal month | 20.5 | 16.5 | 67% | **6** | **0** | **100%** |
| Volatile month | 52 | 48 | 67% | **6** | **0** | **100%** |
| Month with data errors | 3.5 | 1 | 0% | **6** | **0** | **100%** |

The decisive check will be the run on our real HK data.

## 7. In one sentence

> We still flag orders beyond 4 sigma. We now compare each order with the mean
> and sigma of comparable orders, after removing data errors.
