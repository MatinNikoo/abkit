"""
Every estimator checked against an independent implementation (scipy,
statsmodels), a hand computation, or a number published in a paper or
textbook. Tolerances are tight unless one side is Monte Carlo.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

import abkit as ab
from abkit.sequential import group_sequential_design

sm = pytest.importorskip("statsmodels.api")
from statsmodels.stats.meta_analysis import combine_effects  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from statsmodels.stats.nonparametric import rank_compare_2indep  # noqa: E402
from statsmodels.stats.power import NormalIndPower  # noqa: E402
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW, ttest_ind  # noqa: E402

rng = np.random.default_rng(2024)


# ------------------------------------------------------------- Welch t test
def test_welch_matches_scipy_and_statsmodels():
    a, b = rng.normal(5, 1, 300), rng.normal(5.2, 2.5, 450)
    r = ab.mean_test(a, b)
    ref = stats.ttest_ind(b, a, equal_var=False)
    ci = ref.confidence_interval(0.95)
    assert r.p_value == pytest.approx(ref.pvalue, rel=1e-10)
    assert r.ci[0] == pytest.approx(ci.low, rel=1e-10)
    assert r.ci[1] == pytest.approx(ci.high, rel=1e-10)
    cm = CompareMeans(DescrStatsW(b), DescrStatsW(a))
    lo, hi = cm.tconfint_diff(alpha=0.05, usevar="unequal")
    assert (r.ci[0], r.ci[1]) == (pytest.approx(lo, rel=1e-10), pytest.approx(hi, rel=1e-10))
    assert r.se == pytest.approx(cm.std_meandiff_separatevar, rel=1e-12)


def test_welch_at_other_alphas():
    a, b = rng.normal(0, 1, 80), rng.normal(0.3, 1, 60)
    for alpha in (0.01, 0.1, 0.2):
        r = ab.mean_test(a, b, alpha=alpha)
        ci = stats.ttest_ind(b, a, equal_var=False).confidence_interval(1 - alpha)
        assert r.ci == (pytest.approx(ci.low, rel=1e-10), pytest.approx(ci.high, rel=1e-10))


def test_proportion_se_is_the_unpooled_wald_with_n_minus_1():
    a, b = rng.binomial(1, 0.2, 5000), rng.binomial(1, 0.22, 6000)
    r = ab.mean_test(a, b)
    pa, pb = a.mean(), b.mean()
    assert r.se == pytest.approx(np.sqrt(pa * (1 - pa) / (len(a) - 1) + pb * (1 - pb) / (len(b) - 1)), rel=1e-12)


def test_relative_lift_delta_method_by_hand():
    a, b = rng.normal(10, 3, 1000), rng.normal(10.5, 3, 1200)
    r = ab.mean_test(a, b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(vb / ma**2 + mb**2 * va / ma**4)
    assert r.rel_lift == pytest.approx(mb / ma - 1, rel=1e-12)
    assert r.rel_ci[0] == pytest.approx(mb / ma - 1 - 1.959963984540054 * se, rel=1e-10)


# ------------------------------------------------- delta method and clusters
def _sessions(n_users, rate_shift=0.0):
    k = rng.poisson(3, n_users) + 1
    users = np.repeat(np.arange(n_users), k)
    rate = np.clip(rng.beta(2, 6, n_users) + rate_shift, 0, 1)
    return (rng.random(k.sum()) < rate[users]).astype(float), users


def test_ratio_and_cluster_test_match_statsmodels_cluster_robust():
    # session level regression on a treatment dummy with CR0 cluster robust
    # errors. the delta method uses ddof = 1 per user, so it's CR0 * G / (G - 1)
    # when both groups have G users
    G = 800
    yc, gc = _sessions(G)
    yt, gt = _sessions(G, 0.03)
    y = np.r_[yc, yt]
    d = np.r_[np.zeros(len(yc)), np.ones(len(yt))]
    groups = np.r_[gc, gt + G]
    fit = sm.OLS(y, sm.add_constant(d)).fit(cov_type="cluster", cov_kwds={"groups": groups, "use_correction": False})
    r = ab.cluster_test(yc, yt, gc, gt)
    assert r.lift == pytest.approx(fit.params[1], rel=1e-12)
    assert r.se**2 == pytest.approx(fit.bse[1] ** 2 * G / (G - 1), rel=1e-10)
    sc, kc = np.bincount(gc, weights=yc), np.bincount(gc).astype(float)
    st, kt = np.bincount(gt, weights=yt), np.bincount(gt).astype(float)
    rr = ab.ratio_test(sc, kc, st, kt)
    assert rr.se == pytest.approx(r.se, rel=1e-12)
    assert rr.lift == pytest.approx(r.lift, rel=1e-12)


def test_ratio_test_by_hand():
    x, n = rng.poisson(3, 500).astype(float), rng.poisson(5, 500).astype(float) + 1
    x2, n2 = rng.poisson(3, 400).astype(float), rng.poisson(5, 400).astype(float) + 1
    r = ab.ratio_test(x, n, x2, n2)
    def v(x, n):
        rr = x.sum() / n.sum()
        return np.sum((x - rr * n) ** 2) / (len(x) - 1) / len(x) / n.mean() ** 2
    assert r.se == pytest.approx(np.sqrt(v(x, n) + v(x2, n2)), rel=1e-10)


# ------------------------------------------------------------------- CUPED
def test_cuped_theta_is_the_pooled_ols_slope():
    x = rng.normal(0, 1, (3000, 2))
    y = 1 + x @ [0.7, -0.4] + rng.normal(0, 1, 3000)
    _, theta = ab.cuped_adjust(y, x)
    fit = sm.OLS(y, sm.add_constant(x)).fit()
    assert np.allclose(theta, fit.params[1:], rtol=1e-10)
    _, th1 = ab.cuped_adjust(y, x[:, 0])
    assert th1 == pytest.approx(np.cov(y, x[:, 0])[0, 1] / x[:, 0].var(ddof=1), rel=1e-10)


def test_cuped_lift_is_raw_lift_minus_theta_times_covariate_gap():
    n = 2000
    x = rng.normal(0, 1, 2 * n)
    y = 0.6 * x + rng.normal(0, 1, 2 * n)
    y[n:] += 0.1
    y_adj, theta = ab.cuped_adjust(y, x)
    r = ab.mean_test(y_adj[:n], y_adj[n:])
    raw = y[n:].mean() - y[:n].mean()
    assert r.lift == pytest.approx(raw - theta * (x[n:].mean() - x[:n].mean()), rel=1e-10)


def test_adjusted_test_matches_interacted_ols():
    nc, nt = 700, 1300
    xc, xt = rng.normal(0, 1, nc), rng.normal(0.0, 1, nt)
    yc = 0.5 * xc + rng.normal(0, 1, nc)
    yt = 0.2 + 0.9 * xt + rng.normal(0, 1, nt)
    r = ab.adjusted_test(yc, yt, xc, xt)
    x = np.r_[xc, xt]
    xcen = x - x.mean()
    d = np.r_[np.zeros(nc), np.ones(nt)]
    X = np.column_stack([np.ones(nc + nt), d, xcen, d * xcen])
    fit = sm.OLS(np.r_[yc, yt], X).fit(cov_type="HC0")
    assert r.lift == pytest.approx(fit.params[1], rel=1e-10)
    # the robust SE plus the term for the slopes differing
    extra = (fit.params[3]) ** 2 * x.var(ddof=1) / len(x)
    assert r.se**2 == pytest.approx(fit.bse[1] ** 2 + extra, rel=0.01)


# ----------------------------------------------- sample ratio mismatch
def test_srm_is_the_two_sided_binomial_z():
    for a, b in ((44700, 45489), (5000, 5100), (1003, 997)):
        n = a + b
        z = (b - n / 2) / np.sqrt(n / 4)
        assert ab.srm_test(a, b) == pytest.approx(2 * stats.norm.sf(abs(z)), rel=1e-9)
    # close to the exact binomial test at these sizes
    assert ab.srm_test(44700, 45489) == pytest.approx(stats.binomtest(45489, 90189).pvalue, rel=0.01)


def test_srm_textbook_example():
    # Kohavi, Tang and Xu (2020), Trustworthy Online Controlled Experiments:
    # 821,588 vs 815,482 users on a 50/50 split, p about 1.8e-6
    assert 1.7e-6 < ab.srm_test(821588, 815482) < 1.9e-6


def test_srm_uneven_design():
    # 10% intended in treatment
    p = ab.srm_test(9000, 1000, expected_share=0.1)
    assert p == pytest.approx(1.0)
    n = 10200
    z = (1200 - n * 0.1) / np.sqrt(n * 0.1 * 0.9)
    assert ab.srm_test(9000, 1200, 0.1) == pytest.approx(2 * stats.norm.sf(abs(z)), rel=1e-9)


# ---------------------------------------------------- multiple testing
P15 = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
       0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000]


def test_benjamini_hochberg_1995_example():
    # the 15 p values from Benjamini and Hochberg (1995): at 0.05 the FDR
    # procedure rejects 4 hypotheses, the Bonferroni type procedures 3
    assert (ab.benjamini_hochberg(P15) <= 0.05).sum() == 4
    assert (ab.holm(P15) <= 0.05).sum() == 3


def test_holm_and_bh_match_statsmodels():
    for _ in range(20):
        p = rng.uniform(0, 1, 12) ** 3
        assert np.allclose(ab.holm(p), multipletests(p, method="holm")[1], rtol=1e-12)
        assert np.allclose(ab.benjamini_hochberg(p), multipletests(p, method="fdr_bh")[1], rtol=1e-12)
    assert np.allclose(ab.holm(P15), multipletests(P15, method="holm")[1])


def test_dunnett_matches_scipy_with_big_equal_variance_groups():
    c = rng.normal(0, 1, 40000)
    ts = [rng.normal(m, 1, 40000) for m in (0.0, 0.012, 0.02)]
    d = ab.dunnett(c, ts)
    ref = stats.dunnett(*ts, control=c)
    ci = ref.confidence_interval()
    assert np.allclose(d["p_dunnett"], ref.pvalue, rtol=0.01, atol=2e-4)
    se = np.sqrt(2 / 40000)
    assert np.allclose(d["ci_low"], ci.low, atol=0.02 * se)
    assert np.allclose(d["ci_high"], ci.high, atol=0.02 * se)


def test_dunnett_with_one_arm_is_the_z_test():
    c, t = rng.normal(0, 1, 5000), rng.normal(0.05, 1, 5000)
    d = ab.dunnett(c, [t])
    r = ab.mean_test(c, t)
    assert d["p_dunnett"].iloc[0] == pytest.approx(d["p"].iloc[0], rel=1e-6)
    assert d["p"].iloc[0] == pytest.approx(r.p_value, rel=0.02)    # normal vs t tail


# -------------------------------------------------------- heterogeneity
def test_cochran_q_matches_statsmodels():
    est = np.array([0.12, 0.30, -0.05, 0.2, 0.08])
    se = np.array([0.1, 0.12, 0.09, 0.2, 0.05])
    res = combine_effects(est, se**2)
    assert ab.heterogeneity_test(est, se) == pytest.approx(stats.chi2.sf(res.q, len(est) - 1), rel=1e-10)


def test_trend_test_matches_weighted_least_squares():
    x = np.arange(10.0)
    se = rng.uniform(0.5, 2, 10)
    est = 0.3 - 0.05 * x + rng.normal(0, se)
    slope, s, p = ab.trend_test(est, se, x)
    fit = sm.WLS(est, sm.add_constant(x), weights=1 / se**2).fit(cov_type="fixed scale")
    assert slope == pytest.approx(fit.params[1], rel=1e-10)
    assert s == pytest.approx(fit.bse[1], rel=1e-10)
    assert p == pytest.approx(fit.pvalues[1], rel=1e-8)


# -------------------------------------------------------------- planning
def test_sample_size_matches_statsmodels_power():
    for sd, mde, power, ratio in ((1, 0.1, 0.8, 1), (2.5, 0.2, 0.9, 1), (1, 0.05, 0.8, 2)):
        n, _ = ab.sample_size(sd, mde, power=power, ratio=ratio)
        ref = NormalIndPower().solve_power(effect_size=mde / sd, alpha=0.05, power=power, ratio=ratio)
        assert abs(n - ref) <= 1


def test_mde_inverts_sample_size():
    n, _ = ab.sample_size(2.0, 0.1)
    assert ab.mde(2.0, n) == pytest.approx(0.1, rel=1e-3)
    assert ab.mde(2.0, n) <= 0.1
    ref = NormalIndPower().solve_power(nobs1=1000, alpha=0.05, power=0.8, ratio=1)
    assert ab.mde(1.0, 1000) == pytest.approx(ref, rel=1e-4)


def test_ratio_sd_is_the_per_user_delta_variance():
    x, n = rng.poisson(2, 3000).astype(float), rng.poisson(4, 3000).astype(float) + 1
    r = ab.ratio_test(x, n, x, n)
    assert ab.ratio_sd(x, n) ** 2 / 3000 == pytest.approx(r.se**2 / 2, rel=1e-10)


def test_run_length_by_hand():
    plan = ab.run_length(10000, 1500, traffic_share=0.5)
    assert plan.users_needed == 20000
    assert plan.days == 27            # 20000 / 750 = 26.7
    assert plan.days_recommended == 28
    assert ab.run_length(100, 1000).days_recommended == 7


# ---------------------------------------------------------------- Bayes
def test_beta_chance_to_win_matches_numerical_integration():
    a = np.r_[np.ones(120), np.zeros(880)]
    b = np.r_[np.ones(150), np.zeros(850)]
    r = ab.bayes_test(a, b)
    fc, ft = stats.beta(121, 881), stats.beta(151, 851)
    exact, _ = integrate.quad(lambda x: fc.pdf(x) * ft.sf(x), 0, 1, limit=200)
    assert r.chance_treatment_better == pytest.approx(exact, abs=0.004)


def test_prior_posterior_closed_form():
    a, b = rng.normal(0, 1, 3000), rng.normal(0.04, 1, 3000)
    s = 0.03
    r = ab.bayes_test(a, b, prior_sd=s, draws=400_000)
    est = b.mean() - a.mean()
    se2 = a.var(ddof=1) / 3000 + b.var(ddof=1) / 3000
    m, v = est * s**2 / (s**2 + se2), s**2 * se2 / (s**2 + se2)
    sd = np.sqrt(v)
    assert r.chance_treatment_better == pytest.approx(stats.norm.cdf(m / sd), abs=0.003)
    loss_ship = sd * stats.norm.pdf(m / sd) - m * stats.norm.cdf(-m / sd)
    assert r.expected_loss_ship == pytest.approx(loss_ship, rel=0.02, abs=1e-5)
    assert r.lift_interval[0] == pytest.approx(m - 1.96 * sd, abs=sd * 0.02)


# ------------------------------------------------------------ sequential
def test_msprt_one_look_by_hand():
    a, b = rng.normal(0, 1, 2000), rng.normal(0.08, 1.2, 2000)
    tau = 0.05
    r = ab.msprt(a, b, tau=tau, n_looks=1)
    d = b.mean() - a.mean()
    V = a.var(ddof=1) / 2000 + b.var(ddof=1) / 2000
    L = np.sqrt(V / (V + tau**2)) * np.exp(tau**2 * d**2 / (2 * V * (V + tau**2)))
    assert r.p_value[0] == pytest.approx(min(1, 1 / L), rel=1e-9)
    # the confidence sequence is the set of effects the test doesn't reject
    for edge in (r.ci_low[0], r.ci_high[0]):
        dd = d - edge
        L_edge = np.sqrt(V / (V + tau**2)) * np.exp(tau**2 * dd**2 / (2 * V * (V + tau**2)))
        assert L_edge == pytest.approx(1 / 0.05, rel=1e-8)


@pytest.mark.parametrize("k, pocock, obf", [(2, 2.178, 1.977), (3, 2.289, 2.004), (4, 2.361, 2.024),
                                            (5, 2.413, 2.040)])
def test_classic_boundaries_match_jennison_turnbull(k, pocock, obf):
    # Jennison and Turnbull (2000), tables 2.1 and 2.3, two sided alpha 0.05
    assert group_sequential_design(k, kind="pocock", spending=False).bounds[0] == pytest.approx(pocock, abs=6e-4)
    assert group_sequential_design(k, kind="obrien_fleming", spending=False).bounds[-1] == pytest.approx(obf, abs=6e-4)


@pytest.mark.parametrize("k, pocock, obf", [(2, 1.110, 1.008), (3, 1.166, 1.017), (4, 1.202, 1.024)])
def test_inflation_factors_match_jennison_turnbull(k, pocock, obf):
    # maximum sample size relative to a fixed test, 80% power
    assert group_sequential_design(k, kind="pocock", spending=False).inflation(0.8) == pytest.approx(pocock, abs=6e-4)
    assert group_sequential_design(k, kind="obrien_fleming", spending=False).inflation(0.8) == pytest.approx(obf, abs=6e-4)


def test_lan_demets_boundaries_match_gsdesign():
    # gsDesign / rpact, 5 equally spaced looks, two sided alpha 0.05
    obf = group_sequential_design(5, kind="obrien_fleming").bounds
    assert np.allclose(obf, [4.8769, 3.3569, 2.6803, 2.2898, 2.0310], atol=2e-4)
    poc = group_sequential_design(5, kind="pocock").bounds
    assert np.allclose(poc, [2.4380, 2.4268, 2.4101, 2.3966, 2.3859], atol=3e-4)


def test_group_sequential_spends_exactly_alpha():
    for kind in ("pocock", "obrien_fleming"):
        for spending in (True, False):
            d = group_sequential_design(4, kind=kind, spending=spending, info=[0.2, 0.5, 0.7, 1.0])
            assert d.stop_probabilities(0).sum() == pytest.approx(0.05, abs=1e-6)


def test_one_look_design_is_the_z_test():
    d = group_sequential_design(1)
    assert d.bounds[0] == pytest.approx(stats.norm.isf(0.025), abs=1e-6)
    assert d.power(2.8016) == pytest.approx(0.8, abs=1e-3)


# ------------------------------------------------------ robust and rank tools
def test_mann_whitney_matches_statsmodels_and_scipy():
    c, t = rng.lognormal(0, 1, 400), rng.lognormal(0.15, 1.3, 500)
    r = ab.mann_whitney(c, t)
    ref = rank_compare_2indep(c, t, use_t=False)
    assert r.prob_treatment_higher == pytest.approx(1 - ref.prob1, rel=1e-12)
    lo, hi = ref.conf_int()
    assert r.ci[0] == pytest.approx(1 - hi, rel=1e-10)
    assert r.p_value == pytest.approx(stats.mannwhitneyu(t, c, method="asymptotic").pvalue, rel=1e-12)
    u = stats.mannwhitneyu(t, c).statistic
    assert r.prob_treatment_higher == pytest.approx(u / (len(c) * len(t)), rel=1e-12)


def test_bootstrap_matches_scipy_within_monte_carlo_error():
    c, t = rng.lognormal(0, 1, 600), rng.lognormal(0.1, 1, 700)
    r = ab.bootstrap_test(c, t, n_boot=4000, seed=1)
    ref = stats.bootstrap((t, c), lambda x, y, axis: x.mean(axis) - y.mean(axis), method="percentile",
                          n_resamples=20000, rng=np.random.default_rng(2))
    se = ab.mean_test(c, t).se
    assert abs(r.ci[0] - ref.confidence_interval.low) < 0.12 * se
    assert abs(r.ci[1] - ref.confidence_interval.high) < 0.12 * se
    assert r.se == pytest.approx(ref.standard_error, rel=0.06)


def test_quantile_test_point_estimate():
    c, t = rng.exponential(1, 1001), rng.exponential(1.2, 1501)
    r = ab.quantile_test(c, t, 0.5, n_boot=200)
    assert r.lift == pytest.approx(np.median(t) - np.median(c), rel=1e-12)


def test_noninferiority_matches_statsmodels():
    c, t = rng.normal(10, 2, 800), rng.normal(9.9, 2.5, 900)
    r = ab.noninferiority_test(c, t, 0.3)
    assert r.p_value == pytest.approx(ttest_ind(t, c, alternative="larger", usevar="unequal", value=-0.3)[1],
                                      rel=1e-10)
    r = ab.noninferiority_test(c, t, 0.3, higher_is_better=False)
    assert r.p_value == pytest.approx(ttest_ind(t, c, alternative="smaller", usevar="unequal", value=0.3)[1],
                                      rel=1e-10)
    # relative margin: 2% of the control mean
    r = ab.noninferiority_test(c, t, 0.02, relative=True)
    m = 0.02 * c.mean()
    assert r.p_value == pytest.approx(ttest_ind(t, c, alternative="larger", usevar="unequal", value=-m)[1],
                                      rel=1e-10)


def test_winsorize_caps_at_the_pooled_quantile():
    c, t = rng.lognormal(0, 1, 500), rng.lognormal(0, 1, 500)
    cc, ct, (lo, hi) = ab.winsorize(c, t, 0.95, 0.05)
    both = np.r_[c, t]
    assert hi == pytest.approx(np.quantile(both, 0.95)) and lo == pytest.approx(np.quantile(both, 0.05))
    assert cc.max() <= hi and ct.min() >= lo


def test_experiment_screen_uses_the_segment_q_test():
    n = 6000
    df = pd.DataFrame({"v": rng.choice(["A", "B"], n), "seg": rng.choice(["x", "y", "z"], n),
                       "y": rng.normal(0, 1, n)})
    e = ab.Experiment(df, "v", "A", "B")
    _, het = e.segments("y", "seg")
    s = e.screen("y", ["seg"])
    assert s["p"].iloc[0] == pytest.approx(het, rel=1e-12)
