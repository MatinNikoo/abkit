# Effects that change over time

A lift measured over two weeks is an average over those two weeks. Two things can make it a bad guess for what happens after launch.

* Novelty or learning. Users click a new button because it's new and the lift fades, or they need time to learn a new flow and the lift grows.
* Day of week. Weekend users often respond differently. A 10 day test counts some weekdays twice, which tilts the average. This is why `run_length` rounds up to whole weeks.

`time_checks(control, treatment, time_c, time_t)` (or `exp.over_time(metric, time_column)`) computes the lift for each calendar day, tests for a linear trend in the daily lifts (weighted least squares, weights 1 / se^2) and tests whether weekdays differ (Cochran's Q on the lift pooled by weekday).

```python
import numpy as np
import pandas as pd
import abkit as ab

rng = np.random.default_rng(14)
days = pd.date_range("2026-03-02", periods=14, freq="D")
rows = []
for i, day in enumerate(days):
    for group in ("A", "B"):
        lift = (0.05 + 0.15 * np.exp(-i / 3)) if group == "B" else 0.0   # fades over the first days
        rows.append(pd.DataFrame({"group": group, "day": day, "y": rng.normal(lift, 1, 1500)}))
df = pd.concat(rows)
checks = ab.Experiment(df, "group", "A", "B").over_time("y", "day")
print(checks)
print(checks.daily[["day", "lift", "se"]].head())
```

## Assumption

Both tests treat the daily lifts as independent. That holds when each day brings different users, which is the usual setup when the time column is the day a user entered the test. When the same users come back every day their daily lifts are correlated and the p values are only approximate. The data needs real timestamps; a shuffled order has no time in it to check.
