# API reference

Every public name in `abkit`, with a link to the guide that explains when to use it. Each function also has a docstring (`help(ab.mean_test)`).

Tests return a `TestResult` with `lift` (treatment minus control), `se`, `ci`, `p_value`, `rel_lift`, `rel_ci`, group means and sizes, `notes`, and `significant`. Printing one gives a short readout.

## Fixed horizon tests

| Function | What it does | Guide |
|---|---|---|
| `mean_test(control, treatment, alpha=0.05)` | Welch t test on the difference in means; works for 0/1 metrics | [means](guide/means_and_proportions.md) |
| `ratio_test(num_c, den_c, num_t, den_t)` | ratio metric with one row per user, delta method | [ratio metrics](guide/ratio_metrics.md) |
| `cluster_test(y_c, y_t, clusters_c, clusters_t, permutations=0)` | rows grouped in users or randomized clusters | [clusters](guide/clusters.md) |
| `noninferiority_test(control, treatment, margin, higher_is_better=True, relative=False)` | guardrail: not worse by more than margin | [guardrails](guide/guardrails.md) |
| `bootstrap_test(control, treatment, stat="mean", method="percentile")` | bootstrap interval for any statistic; `"studentized"` for means | [skewed metrics](guide/skewed_metrics.md) |
| `quantile_test(control, treatment, q=0.5)` | change in a quantile | [skewed metrics](guide/skewed_metrics.md) |
| `mann_whitney(control, treatment)` | chance a treatment user beats a control user | [skewed metrics](guide/skewed_metrics.md) |
| `bayes_test(control, treatment, prior_sd=None)` | chance to win and expected loss | [Bayes](guide/bayes.md) |

## Variance reduction

| Function | What it does | Guide |
|---|---|---|
| `cuped_adjust(y, x, theta=None)` | CUPED with one or more pre period covariates; returns `(y_adj, theta)` | [variance reduction](guide/variance_reduction.md) |
| `adjusted_test(y_c, y_t, x_c, x_t)` | regression adjustment with separate slopes per group (Lin, 2013) | [variance reduction](guide/variance_reduction.md) |
| `cupac_adjust(y, features, model=None, folds=2)` | cross fitted model prediction as the covariate; returns `(y_adj, theta, prediction)` | [variance reduction](guide/variance_reduction.md) |
| `winsorize(control, treatment, upper=0.99, lower=None)` | cap extreme values at pooled quantiles (biased, see guide) | [skewed metrics](guide/skewed_metrics.md) |

## Checks

| Function | What it does | Guide |
|---|---|---|
| `srm_test(n_control, n_treatment, expected_share=0.5)` | sample ratio mismatch p value | [health checks](guide/health_checks.md) |
| `heterogeneity_test(estimates, ses)` | Cochran's Q: do segment effects differ by more than noise | [many comparisons](guide/many_comparisons.md) |
| `time_checks(y_c, y_t, time_c, time_t)` | daily lifts, novelty trend, day of week | [over time](guide/over_time.md) |
| `trend_test(estimates, ses, x=None)` | weighted slope of estimates over time | [over time](guide/over_time.md) |
| `aa_test(values, sims=1000, clusters=None, test=None)` | A/A simulation on your own data | [planning](guide/planning.md) |

## Sequential testing

| Function | What it does | Guide |
|---|---|---|
| `msprt(control, treatment, tau, n_looks=50)` | always valid p values and confidence sequence | [sequential](guide/sequential.md) |
| `sequential_test(control, treatment, mde)` | `msprt` with tau set to the smallest effect you care about | [sequential](guide/sequential.md) |
| `group_sequential_design(n_looks=5, kind="obrien_fleming", spending=True)` | boundaries, power, inflation factor, expected sample | [sequential](guide/sequential.md) |
| `group_sequential_test(control, treatment, n_looks=5, kind=...)` | replay a group sequential test with repeated confidence intervals | [sequential](guide/sequential.md) |

## Many comparisons

| Function | What it does | Guide |
|---|---|---|
| `holm(p)` | family wise error rate adjusted p values | [many comparisons](guide/many_comparisons.md) |
| `benjamini_hochberg(p)` | false discovery rate adjusted p values | [many comparisons](guide/many_comparisons.md) |
| `dunnett(control, treatments)` | several treatments vs one control, simultaneous intervals | [many comparisons](guide/many_comparisons.md) |

## Planning

| Function | What it does | Guide |
|---|---|---|
| `sample_size(sd, mde, alpha=0.05, power=0.8, ratio=1)` | users per group for an average | [planning](guide/planning.md) |
| `proportion_sample_size(p, mde, ...)` | users per group for a conversion rate | [planning](guide/planning.md) |
| `mde(sd, n_control, n_treatment=None)`, `proportion_mde(p, n_control, ...)` | smallest detectable change | [planning](guide/planning.md) |
| `ratio_sd(num, den)`, `ratio_sample_size(num, den, mde)`, `ratio_mde(num, den, n_control)` | planning for ratio metrics from pilot data | [ratio metrics](guide/ratio_metrics.md) |
| `run_length(n_per_group, users_per_day, traffic_share=1, whole_weeks=True)` | how many days | [planning](guide/planning.md) |

## The Experiment class

`Experiment(df, variant, control, treatment, expected_share=0.5, alpha=0.05)` wraps a DataFrame with one row per user (or per row of a cluster, with `cluster=`).

| Method | What it does |
|---|---|
| `check(pre_period=None, unit=None, metrics=None)` | split check, balance on pre period columns, missing values |
| `srm(unit=None)`, `missing(metric)` | the split check alone; missing share per group |
| `analyze(metric, covariate=None, cluster=None, dropna=False)` | one metric, optional CUPED and cluster level test |
| `analyze_ratio(numerator, denominator)` | ratio metric |
| `analyze_many(metrics, correction="bh")` | several metrics with adjusted p values |
| `analyze_cupac(metric, features, model=None)` | CUPAC |
| `analyze_skewed(metric)` | mean, bootstrap, capped mean, quantiles and Mann-Whitney side by side |
| `guardrails(margins, higher_is_better=None)` | non-inferiority for each guardrail |
| `compare_arms(metric)` | every arm against control, Dunnett adjusted |
| `segments(metric, by)`, `screen(metric, features)` | effects by segment; which traits change the effect |
| `over_time(metric, time)` | novelty and day of week |
| `bayes(metric, prior_sd=None)` | chance to win |
| `sequential(metric, mde, order=None)`, `group_sequential(metric, n_looks=5)` | replays with continuous or planned looks |
| `aa(metric, sims=1000, cluster=None)` | A/A simulation on the control group |

## Input

`Experiment(data, ...)` takes a pandas or polars DataFrame, a dict of columns, a list of records or a path. `abkit.read_table(path)` reads CSV, TSV, Parquet, Excel, JSON and JSON lines (and `.gz`). Bad columns, a metric that isn't numeric or a group that doesn't exist raise `abkit.InputError`, a `ValueError` whose message says what to fix. `TestResult.to_dict()` gives plain numbers for `json.dumps`.

## Plans, presets and charts

| Call | What it does |
|---|---|
| `build_plan(name, primary, baseline=None, sd=None, mde=None, relative_mde=None, ...)` | works out the design and returns it as a dict, see [planning](guide/planning.md) |
| `plan_markdown(plan)` | the written plan |
| `GUARDRAILS` | named guardrail margins, usable wherever a margin goes, see [guardrails](guide/guardrails.md) |
| `abkit.plots.effects`, `segments`, `confidence_sequence`, `daily_lifts`, `forest` | charts, see [charts](guide/plots.md) |

## Reports

`abkit.report.readout(exp, metrics, primary=None, guardrails=None, covariate=None, segment=None, cluster=None, higher_is_better=None, fmt="markdown")` returns a full readout as markdown, HTML or JSON (`fmt="json"`). `Experiment.readout(metrics, ...)` is the same call. `abkit.report.build(...)` returns the same content as plain data. See [readout](guide/readout.md).

## Command line

```
abkit report [data] --variant COL --control A --treatment B --metrics M [M ...]
      [--config FILE] [--sheet NAME] [--where QUERY]
      [--covariate C ...] [--segment S] [--cluster U] [--primary M] [--guardrail M:MARGIN|M:PRESET ...]
      [--lower-is-better M ...] [--correction bh|holm|none] [--split 0.5] [--alpha 0.05]
      [--prior-sd S] [--srm-threshold 0.01] [--title T]
      [--markdown FILE] [--html FILE] [--json FILE] [--plot FILE]
abkit plan --name N --primary M (--baseline P | --sd S | --pilot FILE --numerator N --denominator D)
      (--mde D | --relative-mde R) [--config FILE] [--hypothesis TEXT] [--secondary M ...]
      [--guardrail M:MARGIN ...] [--covariate C ...] [--cuped-rho R] [--cluster U] [--segment S ...]
      [--sequential group|msprt] [--looks 5] [--split 0.5] [--daily-users N] [--traffic-share F]
      [--power 0.8] [--alpha 0.05] [--owner NAME] [--date YYYY-MM-DD] [--out FILE] [--json FILE]
abkit size (--baseline P | --sd S | --pilot FILE --numerator N --denominator D)
      (--mde D | --relative-mde R | --users N) [--cuped-rho R] [--daily-users N] [--traffic-share F]
      [--power 0.8] [--alpha 0.05] [--json FILE]
abkit aa [data] --metric M [--cluster U] [--sims 1000] [--seed 0] [--where QUERY] [--json FILE]
abkit presets
abkit --version
```

A dash as a file name prints to the terminal. See [the command line](guide/command_line.md).
