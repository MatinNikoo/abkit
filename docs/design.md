# abkit design doc

## Goal

Give an analyst one place to go from "here's the experiment data" to a decision they can defend, without having to know which statistical traps apply to their test.

## Who uses it

* An analyst reading out a finished experiment. They want the lift, an interval, a p value, and a warning if something looks broken.
* A product manager who wants to check results while the test is still running. They need numbers that stay valid no matter how often they look.
* Someone planning a test who needs to know how many users it will take.

## Requirements

1. Every method must be shown to work on simulated data where the true answer is known, before it's used on real data.
2. Health checks run before results. A broken split should be loud, not a footnote.
3. Plain inputs. Arrays in, small result objects out, and a thin `Experiment` wrapper on top for DataFrames. No framework to learn.
4. Few dependencies: numpy, scipy and pandas. scikit-learn only if you use CUPAC's default model.
5. Every method has tests, including the command line tool.

## Key decisions

Welch's t test as the default. Treatment often changes the variance as well as the mean, so assuming equal variance is wrong more often than people think. At online sample sizes Welch costs nothing.

Proportions are just means. A conversion rate is the mean of a 0/1 column, so one test covers both. That keeps the API small.

CUPED's theta is estimated on both groups pooled. Estimating it separately per group would let the treatment effect leak into the adjustment. The `Experiment` class also checks whether the covariate is balanced between groups and warns if it isn't, because a covariate the treatment affects will bias the result.

mSPRT for sequential testing, rather than group sequential designs like O'Brien Fleming. Group sequential designs need the number and timing of looks fixed in advance. Real teams don't work that way. The mSPRT is valid at any look, any number of times, which matches how people actually behave. The cost is some power at the very end, measured in the simulations.

tau defaults to the smallest effect you care about. Validity holds for any tau. It only changes how quickly a real effect gets detected, so tying it to the effect that matters to the business is a sensible default someone can explain.

Segments come with a warning built in. Breaking results down by country or platform is where most fake findings come from. So `segments` always adjusts the per segment p values and also returns one heterogeneity test (Cochran's Q) that asks whether the segments differ by more than noise. If that test isn't significant, the honest read is "same effect everywhere", whatever one segment looks like.

Bayesian readout next to the p value, not instead of it. Product managers ask "what's the chance B is better?" and that's a fair question. But with a flat prior the answer is overconfident when real effects are tiny, which is most product tests. The simulations show it saying 85% and being right 73% of the time. So `bayes` takes an optional prior built from the spread of past experiments, which pulls small noisy wins back toward zero and makes the number honest again.

A browser version. The Experiment Lab on my site reimplements the core math in JavaScript so anyone can run an analysis without installing anything, and files never leave their device. I checked its numbers against the Python library on the same data.

A command line tool so someone can get a full readout from a CSV without writing Python. It runs the health checks first and prints a mismatch warning before any result.

Benjamini Hochberg as the default correction for many metrics. Holm protects against even one false win, which is too strict for a dashboard of 20 metrics. The false discovery rate is the trade most teams actually want. Holm is still available.

Stress test before claiming anything. Version 0.2 added `simulations/stress.py`, which runs every method on messy data where I know the answer. It found two real bugs in the sequential test, showed that skewed revenue only breaks the t test when groups are uneven, and showed that row level tests on clustered data are badly wrong. Each finding became either a fix or a guardrail with its own test (`tests/test_guardrails.py`).

Check against other people's code too. A simulation only shows a method behaves like I think it should. `tests/test_reference.py` compares every estimator with scipy, statsmodels or a published table, so a mistake in my understanding can't hide behind a matching mistake in my simulation.

Group sequential designs alongside the mSPRT. When the number of looks really is fixed in advance, an O'Brien Fleming design keeps almost all the power of a single look test. When people refresh a dashboard whenever they like, only the mSPRT stays valid. Both are there, with the trade measured in the simulations.

Cross fitting for CUPAC. A flexible model that predicts the same rows it learned from partly memorizes the outcome, and subtracting that prediction eats into the treatment effect. `cupac_adjust` only predicts each user from a model fit on the other fold.

Count the randomized unit. When users or cities are randomized as a whole, the split check counts them, and `cluster_test` analyzes at that level. Counting rows makes a healthy clustered test look broken most of the time.

A failed split check is a warning sign. The library says "investigate", the report says "hold", and neither says "broken". The Cookie Cats case study shows the kind of digging that should follow.

The report states its decision rule. A PM gets one word (ship, hold, don't ship, look closer, no clear effect) and the rule that produced it, so the decision can be argued with.

Golden vectors for the browser version. `results/golden_vectors.json` holds 197 inputs and abkit's exact outputs, including edge cases, so the JavaScript in the Experiment Lab can be checked against Python on many inputs, edge cases included.

## What it doesn't do (yet)

* Network effects, where one user's treatment affects another's.
* Effects that vary continuously with a user feature are only screened by quantile bins.
* Sequential testing on clustered data. `msprt` treats rows as independent.
* Small sample corrections for `cluster_test` beyond the t reference and the randomization test.

## How I'd know it's working

* Every row in [validation.md](validation.md) either passes or is a known, documented failure with a guardrail.
* All tests pass, on Python 3.10 to 3.13, and the built package passes them in a clean environment.
* A real public experiment runs end to end, the result lines up with other analyses of that data, and the split check's warning gets investigated rather than ignored.
