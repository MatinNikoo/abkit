import numpy as np
import pandas as pd
import pytest
from scipy import stats

import abkit as ab

rng = np.random.default_rng(7)


def test_mean_test_matches_scipy_welch():
    a, b = rng.normal(0, 1, 500), rng.normal(0.2, 1.5, 700)
    r = ab.mean_test(a, b)
    ref = stats.ttest_ind(b, a, equal_var=False)
    assert r.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    assert r.lift == pytest.approx(b.mean() - a.mean())
    assert r.ci[0] < r.lift < r.ci[1]


def test_proportions_are_just_means():
    a = rng.binomial(1, 0.10, 20000)
    b = rng.binomial(1, 0.11, 20000)
    r = ab.mean_test(a, b)
    se = np.sqrt(a.mean() * (1 - a.mean()) / len(a) + b.mean() * (1 - b.mean()) / len(b))
    assert r.se == pytest.approx(se, rel=1e-3)


def test_bad_input_is_rejected():
    with pytest.raises(ValueError):
        ab.mean_test([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        ab.mean_test([1.0, np.nan], [1.0, 2.0])


def test_cuped_keeps_mean_and_cuts_variance():
    x = rng.normal(0, 1, 50000)
    y = 0.8 * x + rng.normal(0, 0.6, 50000)
    y_adj, theta = ab.cuped_adjust(y, x)
    assert y_adj.mean() == pytest.approx(y.mean())
    rho = np.corrcoef(x, y)[0, 1]
    assert y_adj.var() / y.var() == pytest.approx(1 - rho**2, rel=0.02)
    assert theta == pytest.approx(0.8, abs=0.02)


def test_ratio_test_delta_method_matches_bootstrap():
    # sessions per user and clicks per user, clicks depend on sessions
    def sim(n):
        s = rng.poisson(5, n) + 1
        return rng.binomial(s, 0.3), s
    nc, dc = sim(4000)
    nt, dt = sim(4000)
    r = ab.ratio_test(nc, dc, nt, dt)
    boots = []
    for _ in range(400):
        i, j = rng.integers(0, 4000, 4000), rng.integers(0, 4000, 4000)
        boots.append(nt[j].sum() / dt[j].sum() - nc[i].sum() / dc[i].sum())
    assert r.se == pytest.approx(np.std(boots), rel=0.15)


def test_srm():
    assert ab.srm_test(5000, 5000) == pytest.approx(1.0)
    assert ab.srm_test(44700, 45489) < 0.01   # the Cookie Cats split


def test_sample_size_gives_the_power_it_promises():
    n, _ = ab.sample_size(sd=1.0, mde=0.1, power=0.8)
    hits = 0
    for _ in range(400):
        a, b = rng.normal(0, 1, n), rng.normal(0.1, 1, n)
        hits += ab.mean_test(a, b).p_value < 0.05
    assert 0.74 < hits / 400 < 0.86


def test_multiple_testing():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(ab.holm(p), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(ab.benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])
    assert (ab.benjamini_hochberg(p) <= ab.holm(p)).all()


def test_experiment_end_to_end():
    n = 10000
    g = rng.integers(0, 2, n)
    x = rng.normal(10, 2, n)
    df = pd.DataFrame({"v": np.where(g, "B", "A"), "x": x,
                       "y": x + rng.normal(0, 1, n) + 0.05 * g})
    e = ab.Experiment(df, "v", "A", "B")
    assert e.check()["srm_p"] > 0.001
    plain, adjusted = e.analyze("y"), e.analyze("y", covariate="x")
    assert adjusted.se < plain.se * 0.6
    table = e.analyze_many(["y", "x"])
    assert list(table.columns)[-1] == "p_bh"


def test_experiment_flags_srm():
    df = pd.DataFrame({"v": ["A"] * 6000 + ["B"] * 5000, "y": rng.normal(0, 1, 11000)})
    with pytest.warns(UserWarning, match="sample ratio mismatch"):
        ab.Experiment(df, "v", "A", "B").check()


def test_tiny_p_values_print_as_less_than():
    from abkit.core import fmt_p
    assert fmt_p(0.01234) == "p = 0.0123"
    assert fmt_p(3e-9) == "p < 0.0001"
    assert fmt_p(0.02, "adjusted p") == "adjusted p = 0.0200"
    big = ab.mean_test(rng.normal(0, 1, 2000), rng.normal(1, 1, 2000))
    assert "p < 0.0001" in str(big)
