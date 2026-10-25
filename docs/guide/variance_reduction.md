# Variance reduction: CUPED, regression adjustment and CUPAC

Most of the noise in an experiment comes from users being different from each other. If you know something about each user from before the test started, you can subtract the part of the outcome it predicts. You get the same answer with a tighter interval, which means you need fewer users.

abkit has three versions, from simplest to most flexible.

| Function | What it uses | When |
|---|---|---|
| `cuped_adjust` / `Experiment.analyze(m, covariate=...)` | one or a few pre period columns, one shared slope | the default |
| `adjusted_test` | same columns, a separate slope in each group (Lin, 2013) | uneven splits where the effect varies by user |
| `cupac_adjust` / `Experiment.analyze_cupac` | a model's prediction from many pre period features, cross fitted | lots of features, nonlinear patterns |

## Assumptions

* The covariates were measured **before** the test started. A covariate the treatment can move will bias the result. `Experiment.analyze` warns when a covariate is unbalanced between groups.
* For CUPAC, the prediction for each user must come from a model that never saw that user's outcome. `cupac_adjust` does this with cross fitting. Fitting a flexible model and predicting the same rows it learned from leaks part of each user's outcome, treatment effect included, into the covariate. In the simulations that shrank the estimated effect by about a quarter, while the cross fitted version showed no detectable bias and removed about half the variance (see [validation](../validation.md)).

## Example

```python
import numpy as np
import pandas as pd
import abkit as ab

rng = np.random.default_rng(1)
n = 20000
before = rng.gamma(2, 10, n)                       # last month's spend
group = rng.choice(["A", "B"], n)
spend = 0.8 * before + rng.normal(0, 8, n) + 0.4 * (group == "B")
df = pd.DataFrame({"group": group, "spend": spend, "before": before})

exp = ab.Experiment(df, "group", "A", "B")
print(exp.analyze("spend"))                        # plain
print(exp.analyze("spend", covariate="before"))    # CUPED, note says how much variance it removed
```

Separate slopes and a model based covariate:

```python
import numpy as np
import abkit as ab

rng = np.random.default_rng(2)
xc, xt = rng.normal(0, 1, 1000), rng.normal(0, 1, 9000)
yc = 0.7 * xc + rng.normal(0, 0.7, 1000)
yt = 0.1 + 1.2 * xt + rng.normal(0, 0.7, 9000)     # the effect grows with x
print(ab.adjusted_test(yc, yt, xc, xt))

X = rng.normal(0, 1, (4000, 4))
y = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2] + rng.normal(0, 1, 4000)
y_adj, theta, pred = ab.cupac_adjust(y, X)          # needs scikit-learn, or pass model=
print(round(1 - y_adj.var() / y.var(), 2))         # share of variance removed
```

## How much it helps

CUPED removes about rho squared of the variance, where rho is the correlation between the covariate and the outcome. That's 25% at rho 0.5 and 81% at rho 0.9, which turns straight into users, since an 81% cut means the same power with about a fifth of the sample.
