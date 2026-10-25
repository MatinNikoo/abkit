# Skewed metrics like revenue

Revenue is the classic hard metric. Most users spend nothing, a few spend a lot, and one whale can move the average. The stress tests in `simulations/stress.py` found something more specific than "skew breaks the t test".

* With same size groups the t test held up. Lognormal revenue with skewness over 400, Pareto with infinite variance and 95% zeros all kept the false positive rate near 5% and each tail near 2.5%. With equal sizes the skew of the two group means cancels in the difference.
* With uneven groups it broke. With 1,000 users in control and 19,000 in treatment, the false positive rate was 11.4%, and all of it came from one side. A 10/90 split is common in practice (a careful rollout gives the new version 10%).

`mean_test` measures this directly. It estimates the skewness of the difference between the two means and adds a note to the result when it's past the same cutoff as the "n > 355 s^2" rule of thumb (Kohavi et al., 2014), applied to the difference instead of to each group. For equal sizes the number is exactly 0.

## Tools, and what each one actually answers

| Function | Question it answers | Cost |
|---|---|---|
| `mean_test` | did the mean move | breaks with skew plus uneven groups |
| `bootstrap_test(method="studentized")` | did the mean move | fixes part of the uneven group problem, slower |
| `winsorize` then `mean_test` | did the capped mean move | far less noise, but biased: misses lift that comes from big spenders |
| `quantile_test(q=0.5)` | did the median move | blind on metrics where most users are at 0 |
| `mann_whitney` | is a random treatment user likely to beat a random control user | not a test of means at all |

`Experiment.analyze_skewed(metric)` puts all of them side by side. If they disagree, the effect lives in the tail.

```python
import numpy as np
import pandas as pd
import abkit as ab

rng = np.random.default_rng(12)
n = 8000
group = rng.choice(["A", "B"], n)
revenue = rng.lognormal(0, 1.3, n) * np.where(group == "B", 1.06, 1.0)
exp = ab.Experiment(pd.DataFrame({"group": group, "revenue": revenue}), "group", "A", "B")
print(exp.analyze_skewed("revenue", n_boot=500).round(4))
```

## What capping costs

Capping at the 99th percentile cut the noise a lot in the stress tests, but the capped answer missed a big share of the real lift when the treatment scaled everyone's spend. It missed 10.1% of the lift for moderately skewed revenue (lognormal sd 1) and 36.5% to 37.3% for very skewed revenue (sd 2), and the 95% interval around the capped estimate covered the true lift only 21% and 7% of the time. Use it when the question really is about typical users, and say so.

## Mann-Whitney is not a test of revenue

Two groups with exactly the same mean revenue but different shapes get flagged by Mann-Whitney 99.6% of the time in the simulations, while the t test flagged them 4.8% of the time. And revenue can go up overall while most users spend a little less. Report it as "chance a treatment user beats a control user", never as "revenue went up".
