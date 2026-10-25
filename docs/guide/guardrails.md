# Guardrail metrics

Guardrails are metrics you don't expect to improve but can't afford to hurt, like page load time, crash rate, unsubscribes, refunds.

The common mistake is to check them with an ordinary test and call "not significant" safe. It isn't. A small test can miss a real drop simply because it has too few users. In the simulations, an ordinary test missed a drop exactly as big as the margin about a third of the time (34%).

`noninferiority_test(control, treatment, margin)` flips the question. The null is "the treatment is worse by at least `margin`", so a small p value is positive evidence that it isn't. One sided Welch test. `relative=True` reads the margin as a share of the control mean. `higher_is_better=False` for metrics where down is good.

```python
import numpy as np
import pandas as pd
import abkit as ab

rng = np.random.default_rng(13)
n = 10000
group = rng.choice(["A", "B"], n)
df = pd.DataFrame({"group": group,
                   "load_ms": rng.normal(800, 120, n) * np.where(group == "B", 1.002, 1.0),
                   "retained": rng.binomial(1, 0.4, n)})
exp = ab.Experiment(df, "group", "A", "B")
print(exp.guardrails({"load_ms": 0.02, "retained": 0.03}, higher_is_better={"load_ms": False}))
```

A margin can also be a preset name, which sets the direction too, as in `exp.guardrails({"load_ms": "latency", "retained": "retention"})` uses 5% for latency (up is bad) and 1% for retention (down is bad). `abkit presets` lists them. They are starting points to edit, not rules.

The result's interval is the 90% interval when alpha is 0.05, so "passes" means the whole interval stays on the safe side of the margin. Picking the margin is a product decision, and it should be written down before the test starts.
