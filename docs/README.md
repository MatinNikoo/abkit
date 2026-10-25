# abkit docs

Start with the guide for the problem you have. Each one says when to use the method, what it assumes, and has an example you can paste and run (the test suite runs every one of them).

| Guide | For |
|---|---|
| [Means and conversion rates](guide/means_and_proportions.md) | the default test, relative lift, sample sizes |
| [Health checks](guide/health_checks.md) | the split check, balance, missing values |
| [Variance reduction](guide/variance_reduction.md) | CUPED, separate slopes, CUPAC |
| [Ratio metrics](guide/ratio_metrics.md) | clicks per session and planning for them |
| [Clusters](guide/clusters.md) | sessions of a user, randomized cities, few clusters |
| [Skewed metrics](guide/skewed_metrics.md) | revenue, bootstrap, capping, quantiles, Mann-Whitney |
| [Sequential testing](guide/sequential.md) | checking every day (mSPRT) or at planned looks (group sequential) |
| [Many comparisons](guide/many_comparisons.md) | many metrics, several treatments, segments, screening |
| [Guardrails](guide/guardrails.md) | non-inferiority for metrics you can't afford to hurt |
| [Over time](guide/over_time.md) | novelty effects and day of week |
| [Bayes](guide/bayes.md) | chance to win, and why the prior matters |
| [Planning](guide/planning.md) | users, days, smallest detectable effect, A/A tests |
| [Readout](guide/readout.md) | the markdown, HTML or JSON report and its decision rule |
| [Charts](guide/plots.md) | effect, segment, look and daily lift charts |
| [Command line](guide/command_line.md) | reading files, config files, presets, JSON output |

Also here:

* [validation.md](validation.md): every method, every condition it was simulated under, pass or fail.
* [api.md](api.md): every public function in one place.
* [cookie_cats_case_study.md](cookie_cats_case_study.md): what to do when the split check fails on a real test.
* [design.md](design.md): why the library is built the way it is.
* [review_guide.md](review_guide.md): for anyone reviewing the methods.
