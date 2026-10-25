# Cookie Cats: moving the first gate from level 30 to 40

**Decision: Hold.** The split check failed (p = 0.0086). That alone doesn't show the test is broken, but find out why before acting on the numbers below.

## Health checks

| Check | Result |
|---|---|
| Users | 44,700 gate_30, 45,489 gate_40 |
| Split check (SRM) | p = 0.0086  (failed, investigate) |

## Results

| Metric | gate_30 | gate_40 | Change | 95% CI | p | Adjusted p |
|---|---|---|---|---|---|---|
| retention_7 (primary) | 0.1902 | 0.182 | -4.31% | -6.92% to -1.70% | 0.0016 | 0.0047 |
| retention_1 | 0.4482 | 0.4423 | -1.32% | -2.76% to +0.12% | 0.0744 | 0.1116 |
| rounds_capped | 49.14 | 48.85 | -0.57% | -2.80% to +1.66% | 0.6152 | 0.6152 |

## Guardrails (non-inferiority)

| Metric | gate_30 | gate_40 | Change | Worst allowed | p | Result |
|---|---|---|---|---|---|---|
| retention_1 | 0.4482 | 0.4423 | -1.32% | 1.00% worse | 0.6664 | not shown |

## Caveats

* Sample ratio mismatch: check assignment, logging, bots and anything that drops users from one group. By chance alone 1 healthy test in 100 lands below 0.01.
* 3 metrics were tested, so their p values are also shown adjusted (Benjamini Hochberg). The primary metric is judged on its own p value because it was picked before the test.

## How the decision was made

Split check first: if it fails, hold. Otherwise the primary metric decides: significantly better and every guardrail passes means ship, significantly worse means don't ship, anything else is no clear effect. Made with abkit.
