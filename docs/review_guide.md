# For a reviewer

I'd like someone who didn't build this (a statistics professor or an experimentation engineer) to check the methods themselves. This page is meant to make that fast. Nobody has reviewed it yet.

## What to look at first

1. [validation.md](validation.md): every method, the conditions it was simulated under, and pass or fail on false positives and coverage with Monte Carlo error. The pass rule is written at the top.
2. [tests/test_reference.py](../tests/test_reference.py): each estimator against scipy, statsmodels, published tables (Jennison and Turnbull, gsDesign, Benjamini and Hochberg 1995, Kohavi, Tang and Xu) or a hand computation.
3. [cookie_cats_case_study.md](cookie_cats_case_study.md): a real analysis where the data pushed back.

## Claims I'd most like checked

* The mSPRT plug in variance (`abkit/sequential.py`). The always valid guarantee is for a known variance; abkit plugs in the running sample variance and skips looks where a group has no spread yet. In the stress tests it stayed at or under 1.4% false positives on every data type except sessions, where it treats rows of the same user as independent and reached 7.1%. The guarantee is asymptotic.
* The skew note on `mean_test` (`abkit/core.py`, `_diff_skewness`). It estimates the skewness of the difference in means assuming both groups share a shape, and flags it past 1/sqrt(355), the cutoff behind the "n > 355 s^2" rule. Is this the right generalization of that rule to two groups?
* `cluster_test` with few clusters. Welch degrees of freedom counted in clusters plus an optional randomization test. With very uneven cluster sizes the t version still runs liberal in simulation; is a CR2 / Bell McCaffrey correction worth adding?
* `adjusted_test` standard error. The usual robust standard error for Lin's estimator leaves out the variance from the slopes differing when users are a sample from a population. I add it back. It brought the simulated false positive rate from about 7% back to 5%.
* The Cookie Cats placebo argument. It assumes a level takes at least one round, so players under 29 rounds never saw a gate.

## How to rerun everything

```bash
pip install -e ".[dev]"
make all        # data, tests, all simulations, the Cookie Cats analysis, the paper
```

Feedback through GitHub issues, or by email.
