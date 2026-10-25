# Changelog

Versions follow semantic versioning. `abkit.__version__` is the single source for the version number.

## 0.3.0

No statistical method changed, so every method returns the same numbers as 0.2.0 (the golden vector and reference tests pass untouched, and rerunning the Cookie Cats analysis gives an identical `results/cookie_cats.json`). The one behavior change is in the terminal output of `abkit report`, listed under Changed.

### Added

* Input from CSV, TSV, Parquet, Excel, JSON and JSON lines (and `.gz`), and `Experiment` takes polars frames, dicts and paths directly. `abkit.read_table`.
* Clear errors for a misspelled column (with the closest match), a group that isn't in the variant column (with the groups that are), and a metric that isn't numeric. `abkit.InputError` is a `ValueError`. The command line prints one line and exits with 2.
* `abkit plan` and `abkit.build_plan`: a written plan with the question, metrics, sample size (fixed, group sequential or mSPRT, with expected CUPED savings and unequal splits), analysis, checks and decision rule.
* `--config` for every command, read from TOML or JSON, with the command line overriding the file.
* `--json` for `report`, `size`, `aa` and `plan`, `fmt="json"` for `readout`, `Experiment.readout`, and `TestResult.to_dict`. A dash as the file name prints to the terminal.
* Guardrail presets (`abkit.GUARDRAILS`, `abkit presets`, `--guardrail METRIC:PRESET`) that carry a margin and a direction.
* `abkit.plots`: effects, segments, confidence sequence and daily lift charts. `abkit report --plot`, and the HTML readout now embeds the effects chart.
* `abkit size --relative-mde` and `--cuped-rho`. `abkit report --where`, `--sheet` and `--title`.
* Tests for inputs, plans (checked against statsmodels power and a CUPED simulation), charts and every new command line option. `examples/cookie_cats.toml` and `examples/cookie_cats_plan.md`.

### Changed

* `abkit report` with no output file now ends with the guardrails and the decision, and honors `--primary`, `--guardrail`, `--cluster` and `--srm-threshold` there too (before, they only took effect with `--markdown` or `--html`). It no longer prints a chance to win from a flat prior. That number only shows up with `--prior-sd`, because a flat prior overstates it when effects are small.
* Missing values in the terminal report are dropped with a note instead of stopping with an error, the same as in the written readout.
* A p value below 0.0001 prints as `p < 0.0001` in the terminal, the same as in the written readout, instead of `p = 0.0000`.

## 0.2.0

Everything from 0.1.0 still works the same way, with the same numbers, except for the three fixes listed under Fixed.

### Added

* Stress tests (`simulations/stress.py`): 22 kinds of messy data (skewed and heavy tailed revenue, zero inflated spend, rare conversions down to 0.1%, sessions per user, clustered randomization, four kinds of missing data, effects that vary by user), measuring false positives, coverage, power, bias and run time for every method.
* `docs/validation.md`: one table with every method, the conditions it was tested under, and a pass or fail with Monte Carlo error.
* Group sequential designs: `group_sequential_design`, `group_sequential_test`, `Experiment.group_sequential` (O'Brien Fleming and Pocock, classic and Lan DeMets spending). Boundaries match Jennison and Turnbull's tables and gsDesign.
* `cluster_test` for rows that aren't independent (sessions of a user, randomized cities), with an optional randomization test for few clusters. `Experiment.analyze(cluster=...)`, `Experiment.srm(unit=...)`.
* `adjusted_test`: regression adjustment with a separate slope per group (Lin, 2013).
* `cupac_adjust` and `Experiment.analyze_cupac`: a model's prediction as the CUPED covariate, cross fitted.
* Planning: `mde`, `proportion_mde`, `ratio_sd`, `ratio_sample_size`, `ratio_mde`, `run_length`, `aa_test`.
* Skewed metrics: `bootstrap_test` (percentile and studentized), `quantile_test`, `mann_whitney`, `winsorize`, `Experiment.analyze_skewed`.
* `noninferiority_test` and `Experiment.guardrails`.
* `dunnett` and `Experiment.compare_arms` for tests with several treatments.
* `time_checks`, `trend_test`, `Experiment.over_time`: novelty and day of week checks.
* `Experiment.screen`: which user traits change the effect, adjusted for screening many.
* `Experiment.missing`, `Experiment.check(metrics=...)`, `Experiment.analyze(dropna=True)`.
* `abkit.report`: markdown or HTML readout with a written decision rule. CLI: `abkit report --markdown/--html --primary --guardrail`, `abkit size --pilot/--users/--daily-users`, `abkit aa`.
* A note on `mean_test` results when a skewed metric meets uneven group sizes.
* Reference tests against scipy, statsmodels and published tables, property based tests with hypothesis, CLI tests, tests that run every docs example, golden test vectors (`results/golden_vectors.json`) for checking other implementations.
* Cookie Cats: `analysis/cookie_cats_srm.py` investigates the sample ratio mismatch; the sequential replay is labeled as shuffled order and shows its spread over 1,000 shuffles.
* `simulations/msprt_rare_events.py` reproduces the rare event fix and saves `results/msprt_rare_events.json`. `docs/validation.md` marks which failing rows are baselines.
* CI on Python 3.10 to 3.13 plus a release check that installs the built wheel in a clean environment. Method guides in `docs/guide/`.

### Fixed

* `msprt` and `naive_peeking`: running variances came from sums of squares, which lose all precision when a metric's mean is huge next to its spread. A metric around 1e8 with sd 1 gave p = 0 and an interval with its ends swapped. The data is now shifted before summing. Normal metrics change only in the 15th digit.
* `msprt`: on rare conversions an early look could have zero conversions in a group, so the plugged in variance was 0 and the confidence sequence collapsed to a point that later looks inherited. In simulation (`simulations/msprt_rare_events.py`) it missed the true effect at some look in 33.8% of tests at a 0.1% conversion rate, and 0.7% after the fix. Looks where a group has no spread yet are now skipped. Nothing changes for data where every look has some spread (Cookie Cats is unchanged).
* `ratio_test`: the delta method variance is computed from the linearized values instead of the expanded formula. Same math; it no longer leaves rounding noise behind when every user has the same ratio (found by a property based test).

### Changed

* `srm_test` and `Experiment.check` docs now say what a failed split check means. It is a reason to investigate, and the experiment may still be fine.
* `abkit report` shows the chance B is better only when a prior is given (`--prior-sd`), since a flat prior overstates it on small effects.
* `TestResult` has an optional `ci_level` field for intervals that aren't 1 minus alpha.
* Requires scipy 1.11 or newer.

## 0.1.0

First release: Welch t test, delta method for ratio metrics and relative lift, CUPED with one or more covariates, sample ratio check, segment heterogeneity test, sample size, mSPRT with confidence sequences, Holm and Benjamini Hochberg, Bayesian chance to win with an optional prior, the `Experiment` class, the `abkit` command line tool, simulations and the Cookie Cats analysis.
