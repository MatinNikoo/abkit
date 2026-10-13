# abkit design doc

## Goal

Give an analyst one place to go from "here's the experiment data" to a decision they can defend, without having to know which statistical traps apply to their test.

## Who uses it

* **An analyst** reading out a finished experiment. They want the lift, an interval, a p value, and a warning if something looks broken.
* **A product manager** who wants to check results while the test is still running. They need numbers that stay valid no matter how often they look.
* **Someone planning a test** who needs to know how many users it will take.

## Requirements

1. Every method must be shown to work on simulated data where the true answer is known, before it's used on real data.
2. Health checks run before results. A broken split should be loud, not a footnote.
3. Plain inputs. Arrays in, small result objects out, and a thin `Experiment` wrapper on top for DataFrames. No framework to learn.
4. Few dependencies: numpy, scipy and pandas only.
5. Every method has tests, including the command line tool.

## Key decisions

**Welch's t test as the default.** Treatment often changes the variance as well as the mean, so assuming equal variance is wrong more often than people think. At online sample sizes Welch costs nothing.

**Proportions are just means.** A conversion rate is the mean of a 0/1 column, so one test covers both. That keeps the API small.

**CUPED's theta is estimated on both groups pooled.** Estimating it separately per group would let the treatment effect leak into the adjustment. The `Experiment` class also checks whether the covariate is balanced between groups and warns if it isn't, because a covariate the treatment affects will bias the result.

**mSPRT for sequential testing,** rather than group sequential designs like O'Brien Fleming. Group sequential designs need the number and timing of looks fixed in advance. Real teams don't work that way. The mSPRT is valid at any look, any number of times, which matches how people actually behave. The cost is some power at the very end, measured in the simulations.

**tau defaults to the smallest effect you care about.** Validity holds for any tau. It only changes how quickly a real effect gets detected, so tying it to the effect that matters to the business is a sensible default someone can explain.

**Segments come with a warning built in.** Breaking results down by country or platform is where most fake findings come from. So `segments` always adjusts the per segment p values and also returns one heterogeneity test (Cochran's Q) that asks whether the segments differ by more than noise. If that test isn't significant, the honest read is "same effect everywhere", whatever one segment looks like.

**Bayesian readout next to the p value, not instead of it.** Product managers ask "what's the chance B is better?" and that's a fair question. But with a flat prior the answer is overconfident when real effects are tiny, which is most product tests. The simulations show it saying 85% and being right 73% of the time. So `bayes` takes an optional prior built from the spread of past experiments, which pulls small noisy wins back toward zero and makes the number honest again.

**A browser version.** The Experiment Lab on my site reimplements the core math in JavaScript so anyone can run an analysis without installing anything, and files never leave their device. I checked its numbers against the Python library on the same data.

**A command line tool** so someone can get a full readout from a CSV without writing Python. It runs the health checks first and prints a mismatch warning before any result.

**Benjamini Hochberg as the default correction** for many metrics. Holm protects against even one false win, which is too strict for a dashboard of 20 metrics. The false discovery rate is the trade most teams actually want. Holm is still available.

## What it doesn't do (yet)

* Variance reduction with machine learning predictions as the covariate (CUPAC).
* Effects that vary continuously with a user feature, rather than by fixed segments.
* Network effects, where one user's treatment affects another's.

## How I'd know it's working

* The simulation suite reproduces the published properties of each method (false positive rate, coverage, variance reduction).
* All 23 tests pass, including the command line tool.
* A real public experiment runs end to end and the result lines up with other analyses of that data, and the sample ratio check flags the uneven split.
