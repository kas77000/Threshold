# Execution outlier detection for the SFC — the new method (CEPT)

*Management briefing, November 2026. Technical detail:
`docs/legacy-vs-cept.md` and the design specification.*

## The short version

- The SFC expects us to detect execution outliers on our algorithmic orders
  efficiently, with thresholds we can justify and review.
- Our current approach (average ± 4 standard deviations over a year of
  orders) flags too many orders, and **most of those orders are not bad
  executions**. They are difficult orders (illiquid stocks, large size,
  long duration) behaving normally.
- The new method, **CEPT** (Conditional Expected-Performance Threshold), asks
  a better question: *was this order's performance abnormal given how hard
  this order was?* It still uses the familiar "4 standard deviations" rule,
  but measured against what was expected for that order.
- On synthetic test books with known answers, CEPT cut false alerts from
  roughly 16–48 a month to 0–1, **while catching more of the genuinely
  bad orders**, not fewer.
- The next step is to run it on the real HK TWAP/VWAP extract.

## Why the current method over-alerts

Comparing every order with one book-wide average is like judging every
delivery driver by the same delivery time, whether they drive across town or
across the country. The long trips look like "outliers" every day, and a slow
short trip is never noticed.

Our book is like that. A 5-minute order in a liquid stock and a full-day order
at 40% of volume in an illiquid stock are measured against the same band. The
result:

1. Many false alerts, each needing a written justification.
2. Some genuinely bad orders are **missed**, because they were easy orders and
   their miss looks small next to the book's average.
3. In a volatile month, the alert count jumps for reasons that have nothing to
   do with our algorithms.

Making the band narrower or wider does not solve this. The comparison itself
is wrong.

## What the new method does

For each order, in four steps:

1. **Check the record.** Impossible records (e.g. participation above 100%,
   zero duration, performance inconsistent with the slippage figure) go to a
   data-exception list for operations, not to the review queue.
2. **Estimate what was expected.** A statistical model, fitted on a full year
   of our own orders, gives the expected performance and the normal range of
   variation for an order with these characteristics: volatility, duration,
   spread, size versus daily volume, participation rate, auction share and side.
3. **Flag only what is rare.** An order is a Tier A alert if its performance
   was rarer than **1 in 10,000** for its conditions, and never closer than
   **4 standard deviations** from what was expected. A Tier B watch list
   (1 in 1,000) gets a lighter review.
4. **Look for patterns.** A monthly cohort test checks whether a whole strategy
   or a whole type of order (e.g. high participation, partial fills) has drifted
   worse, even if no single order is extreme.

Each alert comes with its context: expected vs actual performance, and tags
such as "partial fill", "volatility spike" or "auction-heavy". That gives the
reviewer a starting point for the justification.

## Why the SFC should find it hard to challenge

| Likely challenge | Our answer |
|---|---|
| "You set the threshold to get few alerts." | The thresholds (1 in 10,000; never below 4σ; cohort t > 4) were **fixed in advance** and written into every model file. They are never adjusted to the alert count. The expected count is predicted before each month and checked after. |
| "Your model explains bad execution away." | The model uses only factors that make an order hard (market conditions and order size). Outcome measures such as fill rate are deliberately excluded. The model is frozen for the year, so any later deterioration shows up. |
| "How do you know it works?" | Every fit produces a back-test (predicted vs actual alerts by month), a detection test (planted bad orders: how many are caught), and a sensitivity check. |
| "Is this an accepted approach?" | It applies standard, published statistics: robust regression, the conditional z-score used in transaction-cost analysis, and tail modelling. It answers the documentation and periodic-review points the SFC raised in its 2018 algorithmic trading review. |
| "What about gradual decline?" | The cohort test is designed for exactly that, and it detected the planted gradual-decline cases in testing. |

## Evidence so far (synthetic data with known answers)

We built 11 synthetic books covering normal months, volatile months, data
errors, catastrophic orders, gradual strategy decay, and deliberately "wrong
model" cases. Each contained planted bad orders, so the right answer is known.
Both the current program and CEPT were run on each.

Each test month contained 6 planted bad orders. Results are averaged over
two runs of each test book.

| Test book | Current method: orders to justify | of which not bad | bad orders caught | **CEPT: orders to justify** | of which not bad | bad orders caught |
|---|---|---|---|---|---|---|
| Normal month, typical mix of orders | 20.5 | 16.5 | 67% | **6** | **0** | **100%** |
| Volatile month | 52 | 48 | 67% | **6** | **0** | **100%** |
| Data errors in the extract | 3.5 | 1 | **0%** | **6** | **0** | **100%** |
| A strategy slowly getting worse | 25.5 | 23 | 42% | **7**, plus a strategy-level alert | 1 | **100%** |
| A few disastrous orders in the training year | 8 | 6 | 33% | **6** | **0** | **100%** |
| Simple, uniform book | 6 | 0 | 100% | 6 | 0 | 100% |

CEPT also keeps a short watch list, typically 2–4 orders a month, which gets a
lighter review. The detailed table, including the hard cases, is in
`docs/legacy-vs-cept.md`.

Where CEPT is weaker: if a book's normal behaviour is genuinely erratic, no
method can separate a moderately bad order from an ordinary one without
over-alerting. In that case CEPT says so openly: it reports its detection
power, and relies on the watch list and the cohort test.

## What happens next

1. **Run on the real HK TWAP/VWAP year** (about 47k VWAP orders plus TWAP).
   This decides whether our real book behaves like the well-behaved test cases.
2. **Review the outputs**: model card, back-test, detection power,
   sensitivity, and the first month's alerts against the current method's.
3. **Governance sign-off**: approve the policy values and the annual
   refit/review cycle.
4. **Extend** to other markets, then other strategies, with the same rules.

The current method remains available alongside, so that for any month we can
show the regulator what the old method would have flagged and why the
difference is justified.
