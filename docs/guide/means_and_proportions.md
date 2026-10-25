# Means and conversion rates

`mean_test(control, treatment)` is the default test in abkit. It's Welch's t test on the difference in means, and a conversion rate is just the mean of a 0/1 column, so the same function covers both.

## When to use it

Any metric with one number per user, like revenue per user, sessions per user or converted or not. If a user can have several rows (sessions, page views), see [clusters](clusters.md) or [ratio metrics](ratio_metrics.md) instead.

## Assumptions

* Users are independent and were randomized one by one.
* The groups are big enough for the normal approximation. For balanced groups this holds surprisingly early, even for very skewed revenue. It fails when the metric is skewed **and** one group is much smaller than the other. abkit adds a note to the result when that happens (see [skewed metrics](skewed_metrics.md)).
* Welch doesn't assume equal variances, which matters because a treatment often changes the spread as well as the mean.

## Example

```python
import numpy as np
import abkit as ab

rng = np.random.default_rng(0)
control = rng.binomial(1, 0.10, 20000)
treatment = rng.binomial(1, 0.11, 20000)

r = ab.mean_test(control, treatment, metric="converted")
print(r)                  # lift, relative lift with a delta method interval, p value
print(r.significant)      # p < alpha
```

The relative lift (treatment / control minus 1) gets its own interval from the delta method, which accounts for the noise in the control mean too.

## Sample size

```python
import abkit as ab

print(ab.proportion_sample_size(0.10, 0.01))   # users per group for a 1 point lift on 10%
print(ab.sample_size(sd=25.0, mde=0.5))         # an average with sd 25, detect 0.5
print(ab.proportion_mde(0.10, 20000))           # smallest lift 20,000 per group can find
```

## What the checks showed

Welch held its 5% false positive rate on normal, lognormal, Pareto, t, zero inflated and rare binary metrics (down to 0.1% conversion) whenever the groups were about the same size. Details in [validation](../validation.md).
