"""
The failure modes found in simulations/stress.py, each pinned down by a
test so it can't quietly come back.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import abkit as ab
from abkit.core import SKEW_LIMIT, _diff_skewness

rng = np.random.default_rng(99)


def test_skew_note_fires_for_uneven_skewed_groups_only():
    c, t = rng.lognormal(0, 2, 1000), rng.lognormal(0, 2, 19000)
    assert any("skewed metric with uneven groups" in n for n in ab.mean_test(c, t).notes)
    # same sizes: the skew cancels, no note
    assert _diff_skewness(rng.lognormal(0, 2, 5000), rng.lognormal(0, 2, 5000)) == 0.0
    assert not ab.mean_test(rng.lognormal(0, 2, 5000), rng.lognormal(0, 2, 5000)).notes
    # uneven but symmetric data: no note
    assert not ab.mean_test(rng.normal(0, 1, 1000), rng.normal(0, 1, 19000)).notes
    # nearly even groups on a skewed metric (the Cookie Cats case): no note
    assert abs(_diff_skewness(rng.lognormal(0, 2, 44700), rng.lognormal(0, 2, 45489))) < SKEW_LIMIT


def test_uneven_skewed_errors_land_on_one_side():
    # the reason for the note: false positives pile up on one side
    low = high = 0
    for _ in range(400):
        r = ab.mean_test(rng.lognormal(0, 2, 500), rng.lognormal(0, 2, 9500))
        low += r.ci[1] < 0
        high += r.ci[0] > 0
    assert high / 400 > 0.06 and low / 400 < 0.02


def test_msprt_handles_rare_conversions():
    # before the fix, 0 conversions so far gave a zero variance and an
    # interval of one point that missed the truth about a third of the time
    misses = 0
    eff = 0.0009
    for _ in range(150):
        c = (rng.random(20000) < 0.001).astype(float)
        t = (rng.random(20000) < 0.001 + eff).astype(float)
        r = ab.msprt(c, t, tau=eff)
        misses += ((r.ci_low > eff) | (r.ci_high < eff)).any()
        assert (r.ci_low <= r.ci_high).all()
    assert misses / 150 <= 0.07


def test_msprt_works_for_huge_means():
    a, b = rng.normal(0, 1, 3000), rng.normal(0.05, 1, 3000)
    r, s = ab.msprt(a, b, tau=0.05), ab.msprt(a + 1e9, b + 1e9, tau=0.05)
    assert np.allclose(r.p_value, s.p_value, rtol=1e-5)
    assert (s.ci_low <= s.ci_high).all()


def _clustered(n_clusters, size, icc, eff=0.0):
    def arm(shift):
        k = np.maximum(1, rng.lognormal(np.log(size) - 0.5, 1.0, n_clusters).astype(int))
        g = np.repeat(np.arange(n_clusters), k)
        return rng.normal(0, np.sqrt(icc), n_clusters)[g] + rng.normal(0, np.sqrt(1 - icc), len(g)) + shift, g
    return (*arm(0.0), *arm(eff))


def test_row_level_test_breaks_on_clusters_and_cluster_test_doesnt():
    row = clu = 0
    for _ in range(300):
        yc, gc, yt, gt = _clustered(40, 40, 0.05)
        row += ab.mean_test(yc, yt).p_value < 0.05
        clu += ab.cluster_test(yc, yt, gc, gt).p_value < 0.05
    assert row / 300 > 0.2
    assert clu / 300 < 0.11


def test_cluster_permutation_p_value_is_exact_with_few_clusters():
    hits = 0
    for i in range(300):
        yc, gc, yt, gt = _clustered(6, 200, 0.1)
        hits += ab.cluster_test(yc, yt, gc, gt, permutations=500, seed=i).p_value < 0.05
    assert hits / 300 < 0.085


def test_experiment_cluster_and_srm_by_unit():
    yc, gc, yt, gt = _clustered(60, 30, 0.05, 0.2)
    df = pd.DataFrame({"v": ["A"] * len(yc) + ["B"] * len(yt), "y": np.r_[yc, yt],
                       "city": np.r_[gc, gt + 1000]})
    e = ab.Experiment(df, "v", "A", "B")
    r = e.analyze("y", cluster="city")
    assert r.n_control == 60 and r.n_treatment == 60
    assert r.se > e.analyze("y").se * 1.3
    assert e.srm(unit="city") == pytest.approx(1.0)


def test_missing_values():
    n = 4000
    y = rng.normal(0, 1, 2 * n)
    y[n:][rng.random(n) < 0.3] = np.nan          # treatment loses 30%, control nothing
    df = pd.DataFrame({"v": ["A"] * n + ["B"] * n, "y": y})
    e = ab.Experiment(df, "v", "A", "B")
    with pytest.raises(ValueError):
        e.analyze("y")                            # still refuses by default
    r = e.analyze("y", dropna=True)
    assert any("goes missing more in one group" in s for s in r.notes)
    with pytest.warns(UserWarning, match="missing more often"):
        e.check(metrics=["y"])
    assert e.missing("y")["missing_treatment"] == pytest.approx(0.3, abs=0.03)


def test_check_still_warns_on_srm_and_says_warning_sign():
    df = pd.DataFrame({"v": ["A"] * 6000 + ["B"] * 5000, "y": rng.normal(0, 1, 11000)})
    with pytest.warns(UserWarning, match="sample ratio mismatch"):
        ab.Experiment(df, "v", "A", "B").check()
    assert "warning sign" in ab.Experiment.check.__doc__
    assert "not proof" in ab.srm_test.__doc__


def test_cupac_cross_fit_does_not_leak_but_in_sample_does():
    from sklearn.neighbors import KNeighborsRegressor
    n = 1500
    lifts, leaky = [], []
    for i in range(40):
        X = rng.normal(0, 1, (2 * n, 3))
        y = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2] + rng.normal(0, 1, 2 * n)
        y[n:] += 0.2
        ya, _, _ = ab.cupac_adjust(y, X, model=KNeighborsRegressor(20), seed=i)
        lifts.append(ya[n:].mean() - ya[:n].mean())
        pred = KNeighborsRegressor(5).fit(X, y).predict(X)
        yb, _ = ab.cuped_adjust(y, pred)
        leaky.append(yb[n:].mean() - yb[:n].mean())
    assert np.mean(lifts) == pytest.approx(0.2, abs=0.03)
    assert np.mean(leaky) < 0.17                  # shrinks the effect toward 0


def test_adjusted_test_with_varying_effects_keeps_its_level():
    hits = 0
    for _ in range(400):
        xc, xt = rng.normal(0, 1, 1000), rng.normal(0, 1, 1000)
        yc = 0.7 * xc + rng.normal(0, 0.7, 1000)
        yt = 0.7 * xt + 0.5 * xt + rng.normal(0, 0.7, 1000)      # average effect 0, varies by user
        hits += ab.adjusted_test(yc, yt, xc, xt).p_value < 0.05
    assert hits / 400 < 0.08


def test_warnings_dont_leak_from_quiet_paths():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ab.mean_test(rng.normal(0, 1, 100), rng.normal(0, 1, 100))
        ab.msprt(rng.normal(0, 1, 2000), rng.normal(0, 1, 2000), tau=0.1)
