# Validation

Every method, every kind of data it was simulated on, and whether it held up. This file is written by `tools/validation_table.py` from `results/simulations.json`, `results/stress.json` and `results/new_methods.json`, so nothing here is typed by hand.

**The rule.** A row passes when the false positive rate is not above the nominal 5% by more than 2 Monte Carlo standard errors, the 95% interval covers the truth at least 95% of the time minus 2 standard errors, neither tail gets more than 2.5% plus 2 standard errors of the false positives (when tails were recorded), and the estimate's bias is under 5% of the effect (or within 3 standard errors). Conservative rows (well under 5%) pass and are marked.

**How to read it.** Type I is the share of simulated tests with no real effect that came out significant (for sequential tests, at any look). Coverage is the share of 95% intervals that contained the true effect when there was one (for sequential tests, at every look at once). Numbers are percent ± one Monte Carlo standard error. Bias is percent of the true effect ± one standard error. Rows marked with a remark like "the mistake ... fixes" are deliberate baselines that are supposed to fail.

91 of 127 rows pass. 36 fail, and 18 of those are baselines (marked in the Baseline column) that exist to show the mistake or the different estimate. The other 18 are real limits of the library, each with its reason in the Note column.

| Method | Data | Sims (null + effect) | Type I % | Coverage % | Bias % | Verdict | Baseline | Note |
|---|---|---|---|---|---|---|---|---|
| msprt, 100 looks (any look) | normal, 10,000 per group | 2,000 | 2.2 ± 0.3 |  |  | pass |  | conservative |
| ordinary p value checked 100 times | normal, 10,000 per group | 2,000 | 39.4 ± 1.1 |  |  | fail | yes | too many false positives; the mistake the sequential tests exist to fix |
| CUPED | normal, covariate correlation 0.0 | 800 |  | 94.1 ± 0.8 |  | pass |  |  |
| CUPED | normal, covariate correlation 0.3 | 800 |  | 95.8 ± 0.7 |  | pass |  |  |
| CUPED | normal, covariate correlation 0.5 | 800 |  | 94.6 ± 0.8 |  | pass |  |  |
| CUPED | normal, covariate correlation 0.7 | 800 |  | 94.8 ± 0.8 |  | pass |  |  |
| CUPED | normal, covariate correlation 0.9 | 800 |  | 94.9 ± 0.8 |  | pass |  |  |
| ratio_test (delta method) | clicks per session, 2,000 users per group | 1,000 |  | 96.2 ± 0.6 |  | pass |  |  |
| sessions treated as independent | clicks per session, 2,000 users per group | 1,000 |  | 88.0 ± 1.0 |  | fail | yes | interval too short; the mistake ratio_test fixes |
| holm (any false win) | 20 metrics, 15 with no effect | 2,000 | 4.4 ± 0.5 |  |  | pass |  |  |
| benjamini_hochberg (false discovery rate) | 20 metrics, 15 with no effect | 2,000 | 3.8 ± 0.2 |  |  | pass |  | conservative |
| heterogeneity_test (Cochran's Q) | 10 segments, same effect everywhere | 1,000 | 6.1 ± 0.8 |  |  | pass |  |  |
| segments with BH (any segment significant) | 10 segments, no effect | 1,000 | 5.8 ± 0.7 |  |  | pass |  |  |
| mean_test (Welch) | normal, 2,000 per group | 2,000 + 1,000 | 4.8 ± 0.5 | 96.1 ± 0.6 | -0.1 ± 1.1 | pass |  |  |
| bayes_test, flat prior (95% interval) | normal, 2,000 per group | 2,000 + 1,000 | 4.5 ± 0.5 | 96.3 ± 0.6 | -0.5 ± 1.1 | pass |  |  |
| msprt, 50 looks (any look) | normal, 2,000 per group | 2,000 + 1,000 | 1.3 ± 0.3 | 98.7 ± 0.4 |  | pass |  | conservative |
| mean_test (Welch) | lognormal sd 1, 2,000 per group | 2,000 + 1,000 | 5.3 ± 0.5 | 95.4 ± 0.7 | -1.0 ± 1.2 | pass |  |  |
| mean_test after winsorize(0.99) | lognormal sd 1, 2,000 per group | 2,000 + 1,000 | 6.0 ± 0.5 | 94.0 ± 0.8 | -10.1 ± 1.0 | fail | yes | biased (-10% of the effect); estimates the capped mean, so this is the bias of capping, by design |
| CUPED | lognormal sd 1, 2,000 per group | 2,000 + 1,000 | 5.0 ± 0.5 | 95.6 ± 0.6 | -2.0 ± 0.9 | pass |  |  |
| adjusted_test (Lin) | lognormal sd 1, 2,000 per group | 2,000 + 1,000 | 5.0 ± 0.5 | 95.6 ± 0.6 | -1.8 ± 0.9 | pass |  |  |
| bayes_test, flat prior (95% interval) | lognormal sd 1, 2,000 per group | 2,000 + 1,000 | 5.1 ± 0.5 | 95.4 ± 0.7 | -1.6 ± 1.2 | pass |  |  |
| msprt, 50 looks (any look) | lognormal sd 1, 2,000 per group | 2,000 + 1,000 | 0.8 ± 0.2 | 99.4 ± 0.2 |  | pass |  | conservative |
| mean_test (Welch) | lognormal sd 2, 2,000 per group | 2,000 + 1,000 | 3.9 ± 0.4 | 94.2 ± 0.7 | -0.5 ± 1.5 | pass |  | conservative |
| mean_test after winsorize(0.99) | lognormal sd 2, 2,000 per group | 2,000 + 1,000 | 5.4 ± 0.5 | 21.1 ± 1.3 | -36.5 ± 0.4 | fail | yes | interval too short; biased (-37% of the effect); estimates the capped mean, so this is the bias of capping, by design |
| CUPED | lognormal sd 2, 2,000 per group | 2,000 + 1,000 | 4.5 ± 0.5 | 92.9 ± 0.8 | -1.5 ± 1.4 | fail |  | interval too short; slightly short at this sample size, fine at 20,000 per group |
| adjusted_test (Lin) | lognormal sd 2, 2,000 per group | 2,000 + 1,000 | 5.2 ± 0.5 | 92.9 ± 0.8 | +0.0 ± 1.5 | fail |  | interval too short; slightly short at this sample size, fine at 20,000 per group |
| bayes_test, flat prior (95% interval) | lognormal sd 2, 2,000 per group | 2,000 + 1,000 | 4.2 ± 0.4 | 94.1 ± 0.7 | -0.7 ± 1.5 | pass |  |  |
| msprt, 50 looks (any look) | lognormal sd 2, 2,000 per group | 2,000 + 1,000 | 0.0 ± 0.1 | 94.5 ± 0.7 |  | pass |  | conservative |
| mean_test (Welch) | lognormal sd 2, 20,000 per group | 2,000 + 1,000 | 4.8 ± 0.5 | 95.8 ± 0.6 | -2.6 ± 1.2 | pass |  |  |
| mean_test after winsorize(0.99) | lognormal sd 2, 20,000 per group | 2,000 + 1,000 | 5.1 ± 0.5 | 6.8 ± 0.8 | -37.3 ± 0.3 | fail | yes | interval too short; biased (-37% of the effect); estimates the capped mean, so this is the bias of capping, by design |
| CUPED | lognormal sd 2, 20,000 per group | 2,000 + 1,000 | 4.5 ± 0.5 | 95.6 ± 0.6 | -2.3 ± 1.1 | pass |  |  |
| adjusted_test (Lin) | lognormal sd 2, 20,000 per group | 2,000 + 1,000 | 4.6 ± 0.5 | 95.2 ± 0.7 | -1.8 ± 1.2 | pass |  |  |
| bayes_test, flat prior (95% interval) | lognormal sd 2, 20,000 per group | 2,000 + 1,000 | 4.6 ± 0.5 | 95.7 ± 0.6 | -3.1 ± 1.2 | pass |  |  |
| msprt, 50 looks (any look) | lognormal sd 2, 20,000 per group | 2,000 + 1,000 | 0.0 ± 0.1 | 99.2 ± 0.3 |  | pass |  | conservative |
| mean_test (Welch) | lognormal sd 2, 1,000 vs 19,000 | 2,000 + 1,000 | 11.4 ± 0.7 | 90.2 ± 0.9 | -0.1 ± 1.1 | fail |  | too many false positives; interval too short; errors pile up on one side; the skew note fires here |
| mean_test after winsorize(0.99) | lognormal sd 2, 1,000 vs 19,000 | 2,000 + 1,000 | 4.9 ± 0.5 | 21.1 ± 1.3 | -32.9 ± 0.4 | fail | yes | interval too short; errors pile up on one side; biased (-33% of the effect); estimates the capped mean, so this is the bias of capping, by design |
| CUPED | lognormal sd 2, 1,000 vs 19,000 | 2,000 + 1,000 | 10.1 ± 0.7 | 95.2 ± 0.7 | -0.4 ± 1.0 | fail |  | too many false positives; errors pile up on one side; same pile up as the t test, adjusting does not fix uneven groups |
| adjusted_test (Lin) | lognormal sd 2, 1,000 vs 19,000 | 2,000 + 1,000 | 12.2 ± 0.7 | 91.4 ± 0.9 | -3.5 ± 1.2 | fail |  | too many false positives; interval too short; errors pile up on one side; same pile up as the t test, adjusting does not fix uneven groups |
| bayes_test, flat prior (95% interval) | lognormal sd 2, 1,000 vs 19,000 | 2,000 + 1,000 | 11.3 ± 0.7 | 90.4 ± 0.9 | -0.8 ± 1.1 | fail |  | too many false positives; interval too short; errors pile up on one side; same normal approximation as the t test |
| mean_test (Welch) | Pareto shape 2.5, 5,000 per group | 2,000 + 1,000 | 5.5 ± 0.5 | 94.7 ± 0.7 | -0.1 ± 1.1 | pass |  |  |
| mean_test after winsorize(0.99) | Pareto shape 2.5, 5,000 per group | 2,000 + 1,000 | 5.5 ± 0.5 | 94.5 ± 0.7 | -6.5 ± 0.7 | fail | yes | biased (-6% of the effect); estimates the capped mean, so this is the bias of capping, by design |
| bayes_test, flat prior (95% interval) | Pareto shape 2.5, 5,000 per group | 2,000 + 1,000 | 5.5 ± 0.5 | 94.9 ± 0.7 | -0.6 ± 1.1 | pass |  |  |
| msprt, 50 looks (any look) | Pareto shape 2.5, 5,000 per group | 2,000 + 1,000 | 0.8 ± 0.2 | 98.6 ± 0.4 |  | pass |  | conservative |
| mean_test (Welch) | Pareto shape 1.5 (infinite variance), 5,000 per group | 2,000 + 1,000 | 4.0 ± 0.4 | 96.1 ± 0.6 | +1.0 ± 5.6 | pass |  | conservative |
| mean_test after winsorize(0.99) | Pareto shape 1.5 (infinite variance), 5,000 per group | 2,000 + 1,000 | 4.9 ± 0.5 | 81.5 ± 1.2 | -21.6 ± 0.7 | fail | yes | interval too short; biased (-22% of the effect); estimates the capped mean, so this is the bias of capping, by design |
| bayes_test, flat prior (95% interval) | Pareto shape 1.5 (infinite variance), 5,000 per group | 2,000 + 1,000 | 4.1 ± 0.4 | 96.0 ± 0.6 | +0.0 ± 5.6 | pass |  | conservative |
| msprt, 50 looks (any look) | Pareto shape 1.5 (infinite variance), 5,000 per group | 2,000 + 1,000 | 0.0 ± 0.1 | 99.6 ± 0.2 |  | pass |  | conservative |
| mean_test (Welch) | t with 3 df, 2,000 per group | 2,000 + 1,000 | 4.4 ± 0.5 | 94.5 ± 0.7 | -0.4 ± 1.2 | pass |  |  |
| bayes_test, flat prior (95% interval) | t with 3 df, 2,000 per group | 2,000 + 1,000 | 4.4 ± 0.5 | 94.6 ± 0.7 | -1.0 ± 1.2 | pass |  |  |
| msprt, 50 looks (any look) | t with 3 df, 2,000 per group | 2,000 + 1,000 | 1.1 ± 0.2 | 98.8 ± 0.3 |  | pass |  | conservative |
| mean_test (Welch) | 5% buyers, lognormal spend, 10,000 per group | 2,000 + 1,000 | 5.5 ± 0.5 | 94.1 ± 0.7 | -2.1 ± 2.9 | pass |  |  |
| mean_test after winsorize(0.99) | 5% buyers, lognormal spend, 10,000 per group | 2,000 + 1,000 | 4.8 ± 0.5 | 89.3 ± 1.0 | -29.1 ± 1.5 | fail | yes | interval too short; biased (-29% of the effect); estimates the capped mean, so this is the bias of capping, by design |
| CUPED | 5% buyers, lognormal spend, 10,000 per group | 2,000 + 1,000 | 5.4 ± 0.5 | 94.1 ± 0.7 | -2.4 ± 2.9 | pass |  |  |
| adjusted_test (Lin) | 5% buyers, lognormal spend, 10,000 per group | 2,000 + 1,000 | 5.4 ± 0.5 | 94.1 ± 0.7 | -2.4 ± 2.9 | pass |  |  |
| bayes_test, flat prior (95% interval) | 5% buyers, lognormal spend, 10,000 per group | 2,000 + 1,000 | 5.5 ± 0.5 | 94.2 ± 0.7 | -3.4 ± 2.9 | pass |  |  |
| msprt, 50 looks (any look) | 5% buyers, lognormal spend, 10,000 per group | 2,000 + 1,000 | 1.0 ± 0.2 | 100.0 ± 0.1 |  | pass |  | conservative |
| mean_test (Welch) | 0.1% conversion, 20,000 per group | 2,000 + 1,000 | 5.0 ± 0.5 | 95.0 ± 0.7 | -2.2 ± 1.3 | pass |  |  |
| bayes_test, flat prior (95% interval) | 0.1% conversion, 20,000 per group | 2,000 + 1,000 | 5.0 ± 0.5 | 95.0 ± 0.7 | -3.0 ± 1.3 | pass |  |  |
| msprt, 50 looks (any look) | 0.1% conversion, 20,000 per group | 2,000 + 1,000 | 0.7 ± 0.2 | 98.7 ± 0.4 |  | pass |  | conservative |
| mean_test (Welch) | 0.3% conversion, 20,000 per group | 2,000 + 1,000 | 4.9 ± 0.5 | 95.9 ± 0.6 | -0.2 ± 1.2 | pass |  |  |
| bayes_test, flat prior (95% interval) | 0.3% conversion, 20,000 per group | 2,000 + 1,000 | 4.6 ± 0.5 | 95.7 ± 0.6 | -0.8 ± 1.2 | pass |  |  |
| msprt, 50 looks (any look) | 0.3% conversion, 20,000 per group | 2,000 + 1,000 | 1.4 ± 0.3 | 98.9 ± 0.3 |  | pass |  | conservative |
| mean_test (Welch) | 1% conversion, 20,000 per group | 2,000 + 1,000 | 4.5 ± 0.5 | 94.8 ± 0.7 | +0.1 ± 1.2 | pass |  |  |
| bayes_test, flat prior (95% interval) | 1% conversion, 20,000 per group | 2,000 + 1,000 | 4.5 ± 0.5 | 94.3 ± 0.7 | -0.7 ± 1.2 | pass |  |  |
| msprt, 50 looks (any look) | 1% conversion, 20,000 per group | 2,000 + 1,000 | 1.3 ± 0.3 | 98.7 ± 0.4 |  | pass |  | conservative |
| mean_test (Welch) | 0.1% conversion, 200,000 per group | 2,000 + 1,000 | 5.2 ± 0.5 | 94.9 ± 0.7 | +2.7 ± 1.2 | pass |  |  |
| bayes_test, flat prior (95% interval) | 0.1% conversion, 200,000 per group | 2,000 + 1,000 | 5.6 ± 0.5 | 94.7 ± 0.7 | +1.6 ± 1.2 | pass |  |  |
| msprt, 50 looks (any look) | 0.1% conversion, 200,000 per group | 2,000 + 1,000 | 1.1 ± 0.2 | 99.2 ± 0.3 |  | pass |  | conservative |
| mean_test (Welch) | 2,000 users per group, about 5 sessions each | 2,000 + 1,000 | 11.3 ± 0.7 | 89.8 ± 1.0 | -1.2 ± 1.1 | fail | yes | too many false positives; interval too short; errors pile up on one side; rows of the same user treated as independent |
| cluster_test | 2,000 users per group, about 5 sessions each | 2,000 + 1,000 | 5.5 ± 0.5 | 94.0 ± 0.8 | -1.2 ± 1.1 | pass |  |  |
| msprt, 50 looks (any look) | 2,000 users per group, about 5 sessions each | 2,000 + 1,000 | 7.1 ± 0.6 | 93.2 ± 0.8 |  | fail | yes | too many false positives; interval too short; rows of the same user treated as independent |
| mean_test (Welch) | 50 clusters per group randomized whole, ICC 0.05 | 2,000 + 1,000 | 41.7 ± 1.1 | 60.6 ± 1.5 | -0.8 ± 1.6 | fail | yes | too many false positives; interval too short; errors pile up on one side; rows of the same cluster treated as independent |
| cluster_test | 50 clusters per group randomized whole, ICC 0.05 | 2,000 + 1,000 | 7.4 ± 0.6 | 94.0 ± 0.8 | -0.8 ± 1.6 | fail |  | too many false positives; errors pile up on one side |
| cluster_test, permutations | 50 clusters per group randomized whole, ICC 0.05 | 2,000 + 1,000 | 5.4 ± 0.5 | 94.0 ± 0.8 | -0.8 ± 1.6 | pass |  | p value from the randomization test, interval is still t based |
| ratio_test on cluster sums (normal) | 50 clusters per group randomized whole, ICC 0.05 | 2,000 + 1,000 | 7.6 ± 0.6 | 93.5 ± 0.8 | -0.8 ± 1.6 | fail |  | too many false positives; errors pile up on one side |
| mean_test (Welch) | 8 clusters per group randomized whole, ICC 0.05 | 2,000 + 1,000 | 66.3 ± 1.1 | 32.1 ± 1.5 | +2.4 ± 1.4 | fail | yes | too many false positives; interval too short; errors pile up on one side; rows of the same cluster treated as independent |
| cluster_test | 8 clusters per group randomized whole, ICC 0.05 | 2,000 + 1,000 | 11.3 ± 0.7 | 87.9 ± 1.0 | +2.4 ± 1.4 | fail |  | too many false positives; interval too short; errors pile up on one side |
| cluster_test, permutations | 8 clusters per group randomized whole, ICC 0.05 | 2,000 + 1,000 | 5.0 ± 0.5 | 87.9 ± 1.0 | +2.4 ± 1.4 | fail |  | interval too short; p value from the randomization test, the interval is still t based |
| ratio_test on cluster sums (normal) | 8 clusters per group randomized whole, ICC 0.05 | 2,000 + 1,000 | 15.2 ± 0.8 | 83.8 ± 1.2 | +2.4 ± 1.4 | fail |  | too many false positives; interval too short; errors pile up on one side |
| mean_test (Welch) | 20% missing completely at random | 2,000 + 1,000 | 5.3 ± 0.5 | 95.3 ± 0.7 | +0.5 ± 1.1 | pass |  |  |
| CUPED | 20% missing completely at random | 2,000 + 1,000 | 5.0 ± 0.5 | 95.8 ± 0.6 | +0.0 ± 0.6 | pass |  |  |
| mean_test (Welch) | missing more for low pre period activity | 2,000 + 1,000 | 5.2 ± 0.5 | 96.4 ± 0.6 | +0.1 ± 1.1 | pass |  |  |
| CUPED | missing more for low pre period activity | 2,000 + 1,000 | 5.3 ± 0.5 | 94.8 ± 0.7 | -0.2 ± 0.7 | pass |  |  |
| mean_test (Welch) | low outcomes go missing, same rule both groups | 2,000 + 1,000 | 5.1 ± 0.5 | 92.1 ± 0.9 | -20.8 ± 1.0 | fail |  | interval too short; biased (-21% of the effect); missingness depends on the outcome |
| CUPED | low outcomes go missing, same rule both groups | 2,000 + 1,000 | 5.1 ± 0.5 | 93.2 ± 0.8 | -7.5 ± 0.6 | fail |  | interval too short; biased (-8% of the effect); missingness depends on the outcome |
| mean_test (Welch) | treatment makes low users drop out of logging | 2,000 + 1,000 | 67.2 ± 1.0 | 39.5 ± 1.5 | +75.0 ± 1.0 | fail |  | too many false positives; interval too short; errors pile up on one side; biased (+75% of the effect); no method can fix this; the missing data check fired, see below |
| CUPED | treatment makes low users drop out of logging | 2,000 + 1,000 | 27.2 ± 1.0 | 74.9 ± 1.4 | +24.8 ± 0.6 | fail |  | too many false positives; interval too short; errors pile up on one side; biased (+25% of the effect); no method can fix this |
| mean_test (Welch) | effect varies with a pre period trait, 50/50 | 2,000 + 1,000 | 5.4 ± 0.5 | 95.2 ± 0.7 | -1.9 ± 1.4 | pass |  |  |
| CUPED | effect varies with a pre period trait, 50/50 | 2,000 + 1,000 | 4.9 ± 0.5 | 93.7 ± 0.8 | -0.7 ± 0.9 | pass |  |  |
| adjusted_test (Lin) | effect varies with a pre period trait, 50/50 | 2,000 + 1,000 | 4.9 ± 0.5 | 93.5 ± 0.8 | -0.7 ± 0.9 | pass |  |  |
| msprt, 50 looks (any look) | effect varies with a pre period trait, 50/50 | 2,000 + 1,000 | 1.1 ± 0.2 | 98.6 ± 0.4 |  | pass |  | conservative |
| mean_test (Welch) | effect varies with a pre period trait, 10/90 | 2,000 + 1,000 | 4.3 ± 0.5 | 94.7 ± 0.7 | +1.0 ± 1.1 | pass |  |  |
| CUPED | effect varies with a pre period trait, 10/90 | 2,000 + 1,000 | 4.5 ± 0.5 | 95.0 ± 0.7 | +0.9 ± 0.9 | pass |  |  |
| adjusted_test (Lin) | effect varies with a pre period trait, 10/90 | 2,000 + 1,000 | 4.5 ± 0.5 | 94.3 ± 0.7 | +1.0 ± 0.7 | pass |  |  |
| group_sequential_test, O'Brien Fleming, 5 looks | normal, 10,000 per group; coverage at effect 0.03 sd | 4,000 + 1,000 | 4.5 ± 0.3 | 96.0 ± 0.6 |  | pass |  |  |
| group_sequential_test, Pocock, 5 looks | normal, 10,000 per group; coverage at effect 0.03 sd | 4,000 + 1,000 | 4.9 ± 0.3 | 94.9 ± 0.7 |  | pass |  |  |
| msprt, 50 looks (any look) | normal, 10,000 per group; coverage at effect 0.03 sd | 4,000 + 1,000 | 0.9 ± 0.2 |  |  | pass |  | conservative |
| mean_test, one look | normal, 10,000 per group; coverage at effect 0.03 sd | 4,000 + 1,000 | 4.7 ± 0.3 |  |  | pass |  |  |
| mean_test, no adjustment | nonlinear outcome, 1,500 per group | 300 + 300 | 2.7 ± 0.9 | 94.7 ± 1.3 | -1.6 ± 4.4 | pass |  | conservative |
| CUPED, 5 linear covariates | nonlinear outcome, 1,500 per group | 300 + 300 | 3.7 ± 1.1 | 92.0 ± 1.6 | -0.7 ± 4.4 | pass |  |  |
| cupac_adjust (cross fit) | nonlinear outcome, 1,500 per group | 300 + 300 | 4.7 ± 1.2 | 92.3 ± 1.5 | -0.5 ± 3.2 | pass |  |  |
| CUPAC without cross fitting (boosting) | nonlinear outcome, 1,500 per group | 300 + 300 | 2.7 ± 0.9 | 90.7 ± 1.7 | -26.2 ± 2.2 | fail | yes | interval too short; biased (-26% of the effect); conservative; the mistake cross fitting prevents |
| CUPAC without cross fitting (5 neighbors) | nonlinear outcome, 1,500 per group | 300 + 300 | 3.0 ± 1.0 | 91.3 ± 1.6 | -21.9 ± 2.7 | fail | yes | interval too short; biased (-22% of the effect); conservative; the mistake cross fitting prevents |
| dunnett (any false win) | 3 arms vs control, no effect | 4,000 | 5.2 ± 0.4 |  |  | pass |  | scipy.stats.dunnett on the same data: 5.1% |
| scipy.stats.dunnett (any false win), reference | same simulated data | 4,000 | 5.1 ± 0.3 |  |  | pass |  |  |
| bonferroni (any false win) | 3 arms vs control, no effect | 4,000 | 4.4 ± 0.3 |  |  | pass |  |  |
| no correction (any false win) | 3 arms vs control, no effect | 4,000 | 12.6 ± 0.5 |  |  | fail | yes | too many false positives; the problem dunnett fixes |
| noninferiority_test (passes a bad treatment) | true change exactly at the margin | 4,000 | 4.6 ± 0.3 |  |  | pass |  |  |
| mean_test (Welch) | lognormal sd 1, 2,000 vs 2,000 | 600 | 5.5 ± 0.9 | 95.0 ± 0.9 |  | pass |  |  |
| bootstrap_test, percentile | lognormal sd 1, 2,000 vs 2,000 | 600 | 5.8 ± 1.0 | 93.8 ± 1.0 |  | pass |  |  |
| bootstrap_test, studentized | lognormal sd 1, 2,000 vs 2,000 | 600 | 5.7 ± 1.0 | 93.7 ± 1.0 |  | pass |  |  |
| quantile_test, q = 0.5 | lognormal sd 1, 2,000 vs 2,000 | 600 |  | 93.5 ± 1.0 |  | pass |  |  |
| quantile_test, q = 0.9 | lognormal sd 1, 2,000 vs 2,000 | 600 |  | 95.2 ± 0.9 |  | pass |  |  |
| mann_whitney | lognormal sd 1, 2,000 vs 2,000, same distribution | 600 | 4.3 ± 0.8 |  |  | pass |  |  |
| mean_test (Welch) | lognormal sd 1.5, 1,000 vs 9,000 | 600 | 7.7 ± 1.1 | 92.5 ± 1.1 |  | fail |  | too many false positives; interval too short; errors pile up on one side |
| bootstrap_test, percentile | lognormal sd 1.5, 1,000 vs 9,000 | 600 | 7.7 ± 1.1 | 92.0 ± 1.1 |  | fail |  | too many false positives; interval too short; errors pile up on one side |
| bootstrap_test, studentized | lognormal sd 1.5, 1,000 vs 9,000 | 600 | 6.3 ± 1.0 | 90.8 ± 1.2 |  | fail |  | interval too short |
| quantile_test, q = 0.5 | lognormal sd 1.5, 1,000 vs 9,000 | 600 |  | 95.2 ± 0.9 |  | pass |  |  |
| quantile_test, q = 0.9 | lognormal sd 1.5, 1,000 vs 9,000 | 600 |  | 96.5 ± 0.8 |  | pass |  |  |
| mann_whitney | lognormal sd 1.5, 1,000 vs 9,000, same distribution | 600 | 2.8 ± 0.7 |  |  | pass |  | conservative |
| mann_whitney (flags a difference) | same mean, different shape | 1,000 | 99.6 ± 0.2 |  |  | fail | yes | too many false positives; not a test of means; this is the documented trap |
| time_checks trend test | 14 days, effect constant | 1,000 | 4.2 ± 0.6 |  |  | pass |  |  |
| time_checks day of week test | 14 days, effect constant | 1,000 | 4.0 ± 0.6 |  |  | pass |  |  |
| Experiment.screen (any false lead) | 8 features, none change the effect | 500 | 3.2 ± 0.8 |  |  | pass |  | conservative |
| aa_test on clean data | normal, 20,000 rows | 2,000 | 5.5 ± 0.5 |  |  | pass |  |  |
| aa_test, users split, rows tested | real style session data | 1,000 | 12.5 ± 0.7 |  |  | fail | yes | too many false positives; shows the row level problem on your own data |
| aa_test, users split, cluster_test | real style session data | 1,000 | 4.7 ± 0.7 |  |  | pass |  |  |

## Power checks

The ratio check plans for a lift of 0.01 from pilot data. In the simulation, clipping rates at 1 makes the true lift a little smaller and the treatment group a little noisier, so the third column is the exact power the delta method number should deliver here.

| Check | Promised | Expected here | Got |
|---|---|---|---|
| ratio_sample_size, delta method: 7,006 users per group | 80% | 77.3% | 77.1 ± 1.3% |
| per session shortcut: 4,980 users per group | 80% | 62.7% | 66.3 ± 1.5% |

| Check | Numerical integration | Simulation |
|---|---|---|
| O'Brien Fleming, 5 looks, power at effect 0.02 sd | 28.6% | 29.9 ± 1.4% |
| O'Brien Fleming, 5 looks, power at effect 0.03 sd | 55.3% | 53.8 ± 1.6% |
| O'Brien Fleming, 5 looks, power at effect 0.04 sd | 79.8% | 79.0 ± 1.3% |
| O'Brien Fleming, 5 looks, power at effect 0.05 sd | 93.8% | 93.6 ± 0.8% |
| Pocock, 5 looks, power at effect 0.02 sd | 23.2% | 24.4 ± 1.4% |
| Pocock, 5 looks, power at effect 0.03 sd | 46.9% | 47.3 ± 1.6% |
| Pocock, 5 looks, power at effect 0.04 sd | 72.4% | 71.3 ± 1.4% |
| Pocock, 5 looks, power at effect 0.05 sd | 89.9% | 90.2 ± 0.9% |

## Other checks from the stress tests

| Check | Result |
|---|---|
| skew note on mean_test, lognormal sd 2, 1,000 vs 19,000 | shown on 100.0% of tests |
| split check counting rows, 2,000 users per group, about 5 sessions each (no real problem) | flagged 0.2% of tests |
| split check counting clusters, same data | flagged 0.0% of tests |
| split check counting rows, 50 clusters per group randomized whole, ICC 0.05 (no real problem) | flagged 73.7% of tests |
| split check counting clusters, same data | flagged 0.0% of tests |
| split check counting rows, 8 clusters per group randomized whole, ICC 0.05 (no real problem) | flagged 87.3% of tests |
| split check counting clusters, same data | flagged 0.0% of tests |
| missing value check, 20% missing completely at random | fired in 0.5% of tests |
| missing value check, missing more for low pre period activity | fired in 0.6% of tests |
| missing value check, low outcomes go missing, same rule both groups | fired in 12.8% of tests |
| missing value check, treatment makes low users drop out of logging | fired in 93.0% of tests |
| skew note on mean_test, every other scenario | never shown (21 scenarios) |

## Run time

Median milliseconds per call on one shared core.

| Method | 10,000 per group | 100,000 | 1,000,000 |
|---|---|---|---|
| mean_test | 0.2 | 0.8 | 7 |
| cuped | 0.4 | 7.2 | 81 |
| ratio_test | 0.2 | 1.2 | 11 |
| cluster_test | 0.8 | 7.9 | 142 |
| msprt_50_looks | 0.4 | 3.1 | 28 |
| bayes_200k_draws | 14.2 | 13.9 | 18 |
