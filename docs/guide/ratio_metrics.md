# Ratio metrics

Clicks per session, revenue per order, time per page view. The unit that was randomized (the user) isn't the unit in the denominator (the session), and sessions from the same user are correlated. Treating every session as independent makes the interval too narrow.

`ratio_test(num_c, den_c, num_t, den_t)` takes one row per **user** (their total clicks and total sessions) and uses the delta method (Deng, Knoblich and Lu, 2018) to get the variance of sum(clicks) / sum(sessions) right.

## Example

```python
import numpy as np
import abkit as ab

rng = np.random.default_rng(3)
def users(n, shift=0.0):
    rate = np.clip(rng.beta(2, 8, n) + shift, 0, 1)
    sessions = rng.poisson(4, n) + 1
    return rng.binomial(sessions, rate), sessions

clicks_c, sessions_c = users(5000)
clicks_t, sessions_t = users(5000, 0.01)
print(ab.ratio_test(clicks_c, sessions_c, clicks_t, sessions_t, metric="clicks per session"))
```

## Planning a ratio metric test

The usual mistake is to plan with the per session standard deviation, sqrt(p(1-p)), and divide by sessions per user. That ignores the correlation inside users and promises more power than you get. `ratio_sd` measures the right per user spread from pilot data:

```python
import numpy as np
import abkit as ab

rng = np.random.default_rng(4)
sessions = rng.poisson(4, 20000) + 1
clicks = rng.binomial(sessions, np.clip(rng.beta(2, 8, 20000), 0, 1))

print(ab.ratio_sd(clicks, sessions))                   # per user sd of the ratio
print(ab.ratio_sample_size(clicks, sessions, mde=0.01))  # users per group
print(ab.ratio_mde(clicks, sessions, n_control=8000))  # what 8,000 per group can find
```

From the command line: `abkit size --pilot pilot.csv --numerator clicks --denominator sessions --mde 0.01`.

## What the checks showed

In a simulation with a 19.8% click rate and a true lift of 0.0099 (the 0.01 target after clipping), the delta method asked for 7,006 users per group and reached 77.1% power (1.3 points of Monte Carlo error). The exact calculation for that lift gives 77.3%, so the shortfall from 80% comes from the lift landing just under 0.01 and the treatment group being a bit noisier. The per session shortcut asked for 4,980 users and reached 66.3% (1.5 points). The numbers are in `results/new_methods.json`.
