# Chance to win

Product managers usually ask "what's the chance B is better, and what do I lose if I ship it and I'm wrong?" `bayes_test` answers that alongside the regular test.

* 0/1 metrics get a Beta posterior from a flat Beta(1, 1) prior.
* Averages get a normal approximate posterior.
* `prior_sd` adds a normal prior centered at zero on the lift itself. Set it to how much lifts usually vary across your past experiments.

## Why the prior matters

With a flat prior the chance to win is honest when real effects are big compared to the noise. When most real effects are tiny, which is the usual case in product tests, it's overconfident. In the simulations, when the flat prior said 85% the better version won only 73% of the time. With a prior matching the spread of past effects, it said 85% and was right 86% of the time.

## Example

```python
import numpy as np
import abkit as ab

rng = np.random.default_rng(11)
a, b = rng.binomial(1, 0.10, 5000), rng.binomial(1, 0.106, 5000)
print(ab.bayes_test(a, b, metric="converted"))                 # flat prior
print(ab.bayes_test(a, b, metric="converted", prior_sd=0.004)) # prior from past tests
```

The chance to win is estimated by drawing from the posteriors (200,000 draws by default), so it moves in the third decimal from run to run unless you fix `seed`. With `prior_sd` the posterior is normal and tests/test_reference.py checks the draws against the closed form.
