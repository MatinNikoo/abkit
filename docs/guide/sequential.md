# Checking results while the test runs

If you look at an ordinary p value every day and stop the first time it dips under 0.05, your false positive rate isn't 5%. In my simulations it was 21% after 10 looks and 39% after 100. abkit has two fixes, built for different situations.

| | mSPRT (`msprt`, `sequential_test`) | Group sequential (`group_sequential_design`, `group_sequential_test`) |
|---|---|---|
| How often you can look | any time, as often as you like | a number of looks fixed in advance (say 5) |
| Maximum sample size | none needed, can run open ended | must be set in advance |
| Power at a fixed maximum | lower | close to a single look test |
| Typical use | dashboards people refresh daily | planned interim reviews |

## mSPRT

The mixture sequential probability ratio test (Johari, Koomen, Pekelis and Walsh, 2017) keeps a running likelihood ratio that stays valid at every look. The matching confidence sequence is an interval that holds at all looks at once. `tau` sets the scale of effects you expect; a good default is the smallest effect you care about. Validity holds for any tau, it only changes how fast a real effect gets caught.

```python
import numpy as np
import abkit as ab

rng = np.random.default_rng(7)
a, b = rng.normal(0, 1, 10000), rng.normal(0.06, 1, 10000)   # in arrival order
r = ab.msprt(a, b, tau=0.05, n_looks=50)
print(r)                 # when it stopped and the always valid interval
print(r.p_value[-5:])    # always valid p value, never goes up
```

The variance is plugged in from the data, so the guarantee is for large samples. On rare conversions (0.1%) the early looks often have zero conversions in a group, which gives a zero variance. abkit skips those looks; without that, the interval collapsed to a point and missed the truth at some look in 33.8% of simulated tests (0.7% with the skip).

## Group sequential designs

Decide the number of looks up front and use a stricter cutoff at each one so the total false positive rate stays at alpha. O'Brien Fleming spends almost nothing early, so its last look is close to an ordinary test. Pocock uses about the same cutoff every time, which stops early more often but costs more at the end. The Lan DeMets spending versions (the default) let the looks happen at any time, as long as the decision to look doesn't depend on the data.

```python
import numpy as np
import abkit as ab

d = ab.group_sequential_design(n_looks=5, kind="obrien_fleming")
print(d)                          # the z cutoff at each look
print(round(d.inflation(0.8), 3)) # how many more users than a single look test for the same power

rng = np.random.default_rng(8)
a, b = rng.normal(0, 1, 10000), rng.normal(0.05, 1, 10000)
print(ab.group_sequential_test(a, b, n_looks=5))
```

The boundaries match the published tables (Jennison and Turnbull, 2000) and the R package gsDesign to 4 decimals; tests/test_reference.py checks them. [Validation](../validation.md) has the simulated power and stopping times next to the mSPRT.

## Replaying a finished test

`Experiment.sequential(metric, mde, order="timestamp")` and `Experiment.group_sequential(metric, order="timestamp")` replay a finished test as if someone had looked along the way. Without a real arrival order, a replay in shuffled order shows only how the decision depends on luck in who showed up when; it can't show anything that depends on time.
