# When rows aren't independent

Two common cases:

* Several rows per user. Users were randomized, but the data has one row per session or page view.
* Clustered randomization. Whole cities, schools or stores were put in one group together, often because users in the same place talk to each other.

A row level t test treats every row as new information. It isn't, so the interval is too narrow and false positives pile up. In the stress tests a row level test on 50 cities per group called a difference that didn't exist 41.7% of the time. `cluster_test` did it 7.4% of the time, and 5.4% with `permutations=999`.

`cluster_test(y_c, y_t, clusters_c, clusters_t)` sums rows per cluster and uses the delta method on sum(y) / rows. That's the same as a cluster robust standard error. It uses a t distribution with degrees of freedom counted in clusters, because with 20 clusters you have about 20 pieces of information, not 20,000.

## Few clusters

With only a handful of clusters, or clusters of very different sizes, even that runs a bit liberal. `permutations=2000` swaps the p value for a randomization test. It reshuffles which clusters got treatment, recomputes the same statistic each time, and sees where the real one falls. That p value is exact for the sharp null (treatment changes no cluster's outcome) and only approximate for the null of no average effect. In the stress tests it gave 5.4% false positives with 50 clusters per group and 5.0% with 8. The interval stays the t interval, and with 8 clusters it still covered only 87.9% of the time.

## Example

```python
import numpy as np
import pandas as pd
import abkit as ab

rng = np.random.default_rng(5)
rows = []
for city in range(40):
    group = "B" if city % 2 else "A"
    size = int(rng.lognormal(4, 0.8))
    city_effect = rng.normal(0, 0.3)
    y = rng.normal(city_effect + 0.1 * (group == "B"), 1, size)
    rows.append(pd.DataFrame({"city": city, "group": group, "y": y}))
df = pd.concat(rows)

exp = ab.Experiment(df, "group", "A", "B")
print(exp.analyze("y"))                   # wrong: treats rows as independent
print(exp.analyze("y", cluster="city"))   # right
print(exp.srm(unit="city"))               # split check on cities

a = df[df.group == "A"]; b = df[df.group == "B"]
print(ab.cluster_test(a.y, b.y, a.city, b.city, permutations=2000).p_value)
```

## The split check needs the same care

If cities of different sizes were randomized, the number of **users** per group will be uneven even when nothing is wrong. In the stress tests a user level split check flagged 74% to 87% of perfectly healthy clustered tests. Check the split on the unit that was randomized with `exp.srm(unit="city")`.

## Check your setup with an A/A test

`aa_test(values, clusters=ids)` splits your real data by cluster many times with no treatment and reports how often the test calls a difference. Pass `test=ab.mean_test` to see what a row level analysis would have done.
