# Charts

`abkit.plots` has five small charts. Each returns the matplotlib figure, so `fig.savefig("effects.png", dpi=150)` keeps it. They need matplotlib (`pip install "abkit[plots]"`).

| Function | Draws | From |
|---|---|---|
| `effects(results)` | relative change and interval for each metric, blue when the interval clears zero | a list of results, `analyze_many`, or readout rows |
| `segments(table)` | the lift in each segment | `Experiment.segments` |
| `confidence_sequence(result)` | the estimate and its interval at every look, with the stop marked | `Experiment.sequential` or `group_sequential` |
| `daily_lifts(checks)` | lift per day with the fitted trend | `Experiment.over_time` |
| `forest(labels, estimates, low, high)` | the plain version, for numbers you computed yourself | anything |

```python
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import abkit as ab
from abkit import plots

rng = np.random.default_rng(21)
n = 8000
group = rng.choice(["A", "B"], n)
df = pd.DataFrame({"group": group,
                   "orders": rng.binomial(1, np.where(group == "B", 0.12, 0.10)),
                   "revenue": rng.gamma(2, 5, n) * np.where(group == "B", 1.03, 1.0),
                   "day": pd.Timestamp("2026-03-01") + pd.to_timedelta(rng.integers(0, 14, n), unit="D")})
exp = ab.Experiment(df, "group", "A", "B")

plots.effects([exp.analyze("orders"), exp.analyze("revenue")], title="Shorter checkout").savefig("effects.png", dpi=150)
plots.confidence_sequence(exp.group_sequential("orders", n_looks=5)).savefig("looks.png", dpi=150)
plots.daily_lifts(exp.over_time("orders", "day")).savefig("daily.png", dpi=150)
```

A band that crosses zero at every look in `confidence_sequence` means the test never reached a decision, and a dashed line marks the first look that did. The `effects` chart only colors by whether the interval excludes zero. It doesn't adjust for testing several metrics, so check the adjusted p values in the table before reading a blue bar as a win.
