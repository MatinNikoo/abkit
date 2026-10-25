# Experiment readout

`abkit report` with `--markdown` or `--html` writes the whole analysis as one document a product manager can read, with health checks first, then results, a chart of every metric's change, guardrails, segments, a decision and its caveats. `--json` writes every number the report is built from, for a pipeline or a dashboard, and `--plot` saves the chart on its own.

```bash
abkit report results.csv --variant group --control A --treatment B \
    --metrics orders revenue --primary orders \
    --guardrail load_ms:0.02 --lower-is-better load_ms \
    --covariate orders_last_month --segment country \
    --markdown readout.md --html readout.html --json readout.json --plot effects.png
```

From Python:

```python
import numpy as np
import pandas as pd
import abkit as ab
from abkit.report import readout

rng = np.random.default_rng(16)
n = 10000
group = rng.choice(["A", "B"], n)
df = pd.DataFrame({"group": group,
                   "orders": rng.binomial(1, np.where(group == "B", 0.115, 0.10)),
                   "load_ms": rng.normal(800, 100, n)})
exp = ab.Experiment(df, "group", "A", "B")
text = readout(exp, ["orders"], guardrails={"load_ms": 0.02}, higher_is_better={"load_ms": False})
print(text[:400])
numbers = exp.readout(["orders"], guardrails={"load_ms": "latency"}, fmt="json")   # presets work here too
```

`fmt` is `"markdown"`, `"html"` or `"json"`. The HTML file is self contained, with the chart embedded when matplotlib is installed.

## How the decision is made

The rule is written into every report so nobody has to guess.

1. If the split check fails, the decision is Hold. Find out why before acting. A fair test can still do this once in a while.
2. If the primary metric is significantly better and every guardrail passes its non-inferiority test, the decision is Ship.
3. If the primary is better but a guardrail couldn't be shown safe, the decision is Look closer.
4. If the primary is significantly worse, the decision is Don't ship.
5. Otherwise the decision is No clear effect, with the smallest change the test had 80% power to find.

The primary metric is judged on its own p value because it was picked before the test. Other metrics get Benjamini Hochberg adjusted p values next to the raw ones.

The chance B is better only appears when you pass `--prior-sd` (or `prior_sd=`). With a flat prior it overstates how sure you can be when real effects are small, so the report leaves it out unless you say what past tests look like.

[examples/cookie_cats_readout.html](../../examples/cookie_cats_readout.html) is the report for the Cookie Cats test. GitHub shows that file as source, so download it and open it in a browser.
