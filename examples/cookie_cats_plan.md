# Experiment plan: Cookie Cats gate at level 40

Written 2026-10-24 by Matin, before any results were looked at. If the plan changes after launch, add a line under Changes below with the date and the reason.

## Question

Moving the first gate from level 30 to level 40 raises 7 day retention.

## Metrics

| Role | Metric | Good direction | How it is judged |
|---|---|---|---|
| Primary | retention_7 | up | its own p value at alpha 0.05, chosen before the test |
| Secondary | sum_gamerounds | up | Benjamini Hochberg adjusted p value across secondary metrics |
| Guardrail | retention_1 | up | non-inferiority, worst allowed change 1.0% lower |

## Size and length

* Smallest change worth finding on retention_7: 0.0095 (5.0% relative to 0.19), with 80% power at alpha 0.05, two sided.
* Baseline retention_7: 0.19, so the per user sd is 0.3923.
* Split: 50% control, 50% treatment.
* Fixed sample: 26,769 control and 26,769 treatment users (53,538 in all).
* Planned looks: 4 evenly spaced, O'Brien Fleming boundaries (Lan DeMets). The most it can need is 54,590 users, 1.020 times the fixed sample. If the effect is the size above it uses about 82% of that on average, and about 99% when nothing changes.

| Look | Share of maximum sample | Stop if abs(z) is at least |
|---|---|---|
| 1 | 25% | 4.333 |
| 2 | 50% | 2.963 |
| 3 | 75% | 2.359 |
| 4 | 100% | 2.014 |

* At 3,000 new users a day and 100% of traffic in the test, that is 19 days. Plan for 21 so every day of the week shows up equally often.

## Analysis

* Primary test: the Welch z score at each planned look against the boundaries above, with repeated confidence intervals that stay valid at every look.
* Secondary metrics get Benjamini Hochberg adjusted p values. Only the primary metric makes the decision.
* Nothing outside this list is part of the plan. Anything else that gets looked at is labeled exploratory.

## Checks before reading any result

* Sample ratio check on the planned split. A p value under 0.01 means hold and investigate.
* Balance on pre period columns.
* Missing values: does one group lose more of a metric than the other.

## Decision rule

* Split check fails: hold.
* retention_7 significantly better and every guardrail passes: ship.
* retention_7 better but a guardrail not shown safe: look closer before shipping.
* retention_7 significantly worse: don't ship.
* Anything else: no clear effect. The test can only rule out changes bigger than about 0.0095.

## Changes

None yet.
