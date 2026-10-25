# Health checks: the split, balance and missing values

Run these before reading any result. `Experiment.check()` does all of them, and the command line report prints them first.

## Sample ratio mismatch (SRM)

If you meant to split 50/50 and got 50.4/49.6 on 90,000 users, something may be dropping users from one group. It could be a redirect that loses some of them, bots filtered unevenly, a logging bug or a crash in one version. That kind of loss can bias every metric, because the users who go missing are rarely a random sample.

`srm_test(n_control, n_treatment, expected_share)` is a chi square test of the counts against the intended split.

A small p value is a warning sign, not proof. By chance alone 1 healthy test in 100 lands below 0.01, and 1 in 1,000 below 0.001. Teams that run hundreds of tests often alert only below 0.001 for that reason. What a failed check really tells you is to go look. The [Cookie Cats case study](../cookie_cats_case_study.md) walks through what that looking can find.

Count the unit that was randomized. If users were randomized, count users, not sessions. If cities were randomized, count cities (`exp.srm(unit="city")`).

```python
import abkit as ab

print(ab.srm_test(44700, 45489))          # Cookie Cats: p = 0.0086
print(ab.srm_test(9000, 1000, 0.1))       # a 90/10 design that came out right
```

## Balance

Columns measured before the test started (last month's activity, country, device) should look the same in both groups. `exp.check(pre_period=["visits_last_month"])` tests each one. A tiny p value on a pre period column means randomization or logging may be off, or the column isn't really pre period.

## Missing values

If a metric is missing more often in one group, comparing only the users who have it can be badly biased. In the stress tests, a treatment that made low value users drop out of logging produced a false "win" 67% of the time. abkit refuses missing values by default. `exp.analyze(m, dropna=True)` drops them and notes how many went missing in each group, and `exp.check(metrics=[...])` warns when one group loses more.

A missing rate that's equal in both groups is not a guarantee. When low values go missing under the same rule in both groups, the treatment can still change who goes missing, and that biased the estimate by 21% of the true effect in the stress tests while the missing rate check noticed in only 13% of them. If the reason values go missing is related to the outcome, no check on the missing rate can rescue the comparison.

```python
import numpy as np
import pandas as pd
import abkit as ab

rng = np.random.default_rng(6)
y = rng.normal(10, 2, 8000)
y[4000:][rng.random(4000) < 0.2] = np.nan        # treatment loses 20% of its values
df = pd.DataFrame({"group": ["A"] * 4000 + ["B"] * 4000, "y": y})
exp = ab.Experiment(df, "group", "A", "B")
print(exp.missing("y"))
print(exp.analyze("y", dropna=True).notes)
```
