# Many metrics, many arms, many segments

Every extra comparison is another chance for a fake win. Check 20 metrics with nothing going on and you'll usually see at least one "significant" result.

## Many metrics: Holm and Benjamini Hochberg

* `holm(p)` controls the chance of **any** false positive. Use it when a single false claim is costly.
* `benjamini_hochberg(p)` controls the **share** of false positives among the wins. Usually the better trade for a dashboard. It's the default in `analyze_many` and the report.

```python
import abkit as ab

p = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
     0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000]
print((ab.benjamini_hochberg(p) <= 0.05).sum())   # 4, as in Benjamini and Hochberg (1995)
print((ab.holm(p) <= 0.05).sum())                 # 3
```

## Several treatments: Dunnett

An A/B/C/D test compares three treatments to one control. All three comparisons share the same control group, so they're correlated, and Dunnett's method uses that to be less strict than Bonferroni while keeping the chance of any false win at alpha. abkit's version allows each group its own variance. The intervals hold for all arms at once.

```python
import numpy as np
import pandas as pd
import abkit as ab

rng = np.random.default_rng(9)
arm = rng.choice(["control", "b", "c", "d"], 12000)
y = rng.normal(5, 2, 12000) + 0.15 * (arm == "d")
exp = ab.Experiment(pd.DataFrame({"arm": arm, "y": y}), "arm", "control", "b")
print(exp.compare_arms("y"))
```

## Segments and screening

`exp.segments(metric, by="country")` gives the effect in each segment with adjusted p values and one Cochran's Q test of whether the segments differ by more than noise. If that test isn't significant, the honest read is "the same effect everywhere", whatever one segment looks like.

`exp.screen(metric, features)` runs that test for many user traits at once (numbers are cut into quartiles) and adjusts across features. Treat a hit as a lead to confirm in a new test.

```python
import numpy as np
import pandas as pd
import abkit as ab

rng = np.random.default_rng(10)
n = 20000
df = pd.DataFrame({"group": rng.choice(["A", "B"], n), "age": rng.integers(18, 70, n),
                   "device": rng.choice(["ios", "android", "web"], n)})
df["y"] = rng.normal(0, 1, n) + 0.3 * (df.group == "B") * (df.device == "ios")
exp = ab.Experiment(df, "group", "A", "B")
table, p_differ = exp.segments("y", by="device")
print(table, p_differ)
print(exp.screen("y", ["age", "device"]))
```
