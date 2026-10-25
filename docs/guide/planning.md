# Planning a test

Before a test starts you want three numbers, which are how many users, how long, and the smallest effect it can find.

```python
import abkit as ab

n, _ = ab.proportion_sample_size(0.19, 0.01)        # 19% baseline, detect 1 point
print(n)
print(ab.run_length(n, users_per_day=4000, traffic_share=0.5))
print(ab.proportion_mde(0.19, 20000))               # what 20,000 per group can find
print(ab.mde(sd=30.0, n_control=50000))             # an average with sd 30
```

`run_length` rounds up to whole weeks so every weekday is counted equally.

From the command line:

```bash
abkit size --baseline 0.19 --mde 0.01 --daily-users 4000 --traffic-share 0.5
abkit size --baseline 0.19 --users 20000
abkit size --pilot pilot.csv --numerator clicks --denominator sessions --mde 0.01
```

For ratio metrics see [ratio metrics](ratio_metrics.md), because the per session shortcut asks for too few users. With CUPED, plan with the reduced spread, which is sd times sqrt(1-rho^2). `abkit size --cuped-rho 0.5` does that, and `--relative-mde 0.05` takes the effect as a share of the baseline.

## Write the plan down first

A result can pick its own story if nothing was written before it. `abkit.build_plan` works out the sample size and collects the choices into one document with the question, which metric decides, the secondary metrics and guardrails, the size and length, how it will be analyzed, the checks, and the decision rule. The sample size follows the same formulas as above, and the CUPED line is the plain `1 - rho^2` cut.

```python
import abkit as ab

plan = ab.build_plan("Shorter checkout", primary="orders", baseline=0.10, relative_mde=0.05,
                     daily_users=3000, guardrails={"load_ms": "latency"}, secondary=["revenue"],
                     covariate="orders_last_month", cuped_rho=0.5, sequential="group", looks=4,
                     date="2026-10-24")
print(plan["n_control"], plan["looks"]["n_control"])
text = ab.plan_markdown(plan)
print(text[:300])
```

`sequential="group"` adds the planned looks, their boundaries and the bigger maximum sample that keeps the power. `"msprt"` plans for checking whenever anyone likes. Unequal splits work through `split=`. If the plan changes after launch, add a dated line under Changes in the file. [examples/cookie_cats_plan.md](../../examples/cookie_cats_plan.md) is one for the Cookie Cats test, written as if before it ran. From the command line it's `abkit plan`, see [the command line](command_line.md).

## A/A test on your own data

The formulas assume the metric behaves. `aa_test(values)` checks that on real data. It splits the data at random many times with no treatment and reports how often the test calls a difference (should be about 5%) and whether the p values are spread evenly. Heavy tails, too few users, or rows that aren't independent show up here before they cost you a real decision.

```python
import numpy as np
import abkit as ab

rng = np.random.default_rng(15)
revenue = rng.lognormal(0, 1.5, 20000)
print(ab.aa_test(revenue, sims=500))
```

`abkit aa data.csv --metric revenue` does the same from the command line.
