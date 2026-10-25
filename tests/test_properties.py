"""
Property based tests: invariances that have to hold for any data, checked
on data hypothesis makes up (and shrinks to the smallest failing case).
"""

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import abkit as ab

SET = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
vals = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False, width=64)
arr = arrays(np.float64, st.integers(5, 60), elements=vals)
pos = arrays(np.float64, st.integers(5, 60), elements=st.floats(0.1, 1e3, width=64))
bits = arrays(np.float64, st.integers(10, 200), elements=st.sampled_from([0.0, 1.0]))


def spread(a):
    return np.std(a) > 1e-6 * max(1.0, np.abs(a).max())


@SET
@given(arr, arr)
def test_swapping_arms_flips_the_sign(a, b):
    assume(spread(a) and spread(b))
    r, s = ab.mean_test(a, b), ab.mean_test(b, a)
    assert s.lift == pytest.approx(-r.lift, abs=1e-9 * (1 + abs(r.lift)))
    assert s.p_value == pytest.approx(r.p_value, rel=1e-9, abs=1e-12)
    assert s.ci[0] == pytest.approx(-r.ci[1], abs=1e-9 * (1 + abs(r.ci[1])))


@SET
@given(arr, arr, st.floats(-1e4, 1e4), st.floats(0.01, 100))
def test_shift_and_scale(a, b, shift, scale):
    assume(spread(a) and spread(b))
    r = ab.mean_test(a, b)
    s = ab.mean_test(a * scale + shift, b * scale + shift)
    assert s.p_value == pytest.approx(r.p_value, rel=1e-6, abs=1e-9)
    assert s.lift == pytest.approx(r.lift * scale, rel=1e-6, abs=1e-6 * scale * (1 + np.abs(a).max()))
    assert s.se == pytest.approx(r.se * scale, rel=1e-6)


@SET
@given(arr, arr)
def test_p_values_and_intervals_are_sane(a, b):
    assume(spread(a) and spread(b))
    r = ab.mean_test(a, b)
    assert 0 <= r.p_value <= 1
    assert r.ci[0] <= r.lift <= r.ci[1]
    # the interval excludes 0 exactly when p < alpha
    assert (r.p_value < 0.05) == (r.ci[0] > 0 or r.ci[1] < 0)


@SET
@given(arr, arr)
def test_wider_intervals_for_smaller_alpha(a, b):
    assume(spread(a) and spread(b))
    w = [np.diff(ab.mean_test(a, b, alpha=al).ci)[0] for al in (0.2, 0.1, 0.05, 0.01)]
    assert all(x <= y + 1e-12 for x, y in zip(w, w[1:]))


@SET
@given(arr, st.floats(0.01, 50))
def test_bigger_true_shift_means_smaller_p(a, d):
    assume(spread(a))
    b = a[::-1].copy()
    p1 = ab.mean_test(a, b + d).p_value
    p2 = ab.mean_test(a, b + 2 * d).p_value
    assert p2 <= p1 + 1e-12


@SET
@given(arrays(np.float64, st.integers(1, 40), elements=st.floats(0, 1)))
def test_holm_and_bh(p):
    h, b = ab.holm(p), ab.benjamini_hochberg(p)
    assert (h >= p - 1e-15).all() and (b >= p - 1e-15).all()
    assert (h <= 1).all() and (b <= 1).all()
    assert (b <= h + 1e-12).all()
    order = np.argsort(p, kind="stable")
    assert (np.diff(h[order]) >= -1e-12).all()
    assert (np.diff(b[order]) >= -1e-12).all()


@SET
@given(st.integers(10, 10**6), st.integers(10, 10**6))
def test_srm_is_symmetric_and_a_probability(a, b):
    p = ab.srm_test(a, b)
    assert 0 <= p <= 1
    assert p == pytest.approx(ab.srm_test(b, a), rel=1e-9, abs=1e-300)


@SET
@given(arr, st.floats(-100, 100))
def test_cuped_keeps_the_mean_and_ignores_covariate_shifts(y, shift):
    x = np.sin(np.arange(len(y))) * 3 + y * 0.1
    assume(np.std(x) > 1e-3)
    y1, th = ab.cuped_adjust(y, x)
    y2, th2 = ab.cuped_adjust(y, x + shift)
    assert y1.mean() == pytest.approx(y.mean(), abs=1e-7 * (1 + np.abs(y).max()))
    assert np.allclose(y1, y2, atol=1e-6 * (1 + np.abs(y).max()))
    assert th == pytest.approx(th2, rel=1e-6, abs=1e-9)


@SET
@given(pos, pos)
def test_ratio_test_scales_and_flips(num, den):
    n = min(len(num), len(den))
    num, den = num[:n], den[:n]
    assume(n >= 5 and spread(num) and spread(den) and spread(num / den))
    num2, den2 = num[::-1] * 1.3, den[::-1]
    r = ab.ratio_test(num, den, num2, den2)
    s = ab.ratio_test(num2, den2, num, den)
    assert s.lift == pytest.approx(-r.lift, rel=1e-9, abs=1e-12)
    assert 0 <= r.p_value <= 1
    k = ab.ratio_test(num * 7, den, num2 * 7, den2)
    assert k.lift == pytest.approx(7 * r.lift, rel=1e-9, abs=1e-9)
    assert k.p_value == pytest.approx(r.p_value, rel=1e-6, abs=1e-12)


@SET
@given(arrays(np.float64, st.integers(2, 10), elements=st.floats(-5, 5)),
       arrays(np.float64, st.integers(2, 10), elements=st.floats(0.05, 3)))
def test_heterogeneity_is_order_free(est, se):
    k = min(len(est), len(se))
    est, se = est[:k], se[:k]
    assume(k >= 2)
    p = ab.heterogeneity_test(est, se)
    assert 0 <= p <= 1
    perm = np.arange(k)[::-1]
    assert ab.heterogeneity_test(est[perm], se[perm]) == pytest.approx(p, rel=1e-9, abs=1e-15)
    assert ab.heterogeneity_test(est + 3.0, se) == pytest.approx(p, rel=1e-6, abs=1e-12)


@SET
@given(st.floats(0.1, 10), st.floats(0.01, 1), st.floats(0.55, 0.95))
def test_sample_size_monotone(sd, mde, power):
    n, _ = ab.sample_size(sd, mde, power=power)
    assert ab.sample_size(sd, mde * 1.5, power=power)[0] <= n
    assert ab.sample_size(sd, mde, power=min(0.99, power + 0.03))[0] >= n
    assert ab.sample_size(sd * 1.5, mde, power=power)[0] >= n


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 10**6), st.floats(-1e9, 1e9))
def test_msprt_shift_invariant_and_p_never_rises(seed, shift):
    r_ = np.random.default_rng(seed)
    a, b = r_.normal(0, 1, 2000), r_.normal(0.05, 1, 2000)
    r = ab.msprt(a, b, tau=0.05)
    s = ab.msprt(a + shift, b + shift, tau=0.05)
    assert np.allclose(r.p_value, s.p_value, rtol=1e-4, atol=1e-6)
    assert (np.diff(r.p_value) <= 1e-15).all()
    assert ((0 <= r.p_value) & (r.p_value <= 1)).all()
    assert (r.ci_low <= r.ci_high).all()


@settings(max_examples=25, deadline=None)
@given(bits, bits)
def test_binary_metrics_never_produce_a_collapsed_sequence(a, b):
    assume(a.sum() > 0 and b.sum() > 0)
    r = ab.msprt(a, b, tau=0.05, n_looks=10, min_per_arm=2)
    assert ((0 <= r.p_value) & (r.p_value <= 1)).all()
    assert (r.ci_high - r.ci_low > 0).all()


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 10**6))
def test_bayes_swap_and_bounds(seed):
    r_ = np.random.default_rng(seed)
    a, b = r_.normal(0, 1, 300), r_.normal(r_.normal(0, 0.2), 1, 300)
    x, y = ab.bayes_test(a, b, draws=50_000), ab.bayes_test(b, a, draws=50_000)
    assert x.chance_treatment_better + y.chance_treatment_better == pytest.approx(1, abs=0.01)
    assert x.lift_interval[0] <= x.lift_interval[1]
    assert x.expected_loss_ship >= 0 and x.expected_loss_keep >= 0


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 10**6), st.floats(0.05, 3))
def test_mann_whitney_swap_and_scale_free(seed, scale):
    r_ = np.random.default_rng(seed)
    a, b = r_.lognormal(0, 1, 80), r_.lognormal(0.2, 1, 90)
    r, s = ab.mann_whitney(a, b), ab.mann_whitney(b, a)
    assert r.prob_treatment_higher + s.prob_treatment_higher == pytest.approx(1, abs=1e-12)
    assert r.p_value == pytest.approx(s.p_value, rel=1e-9)
    k = ab.mann_whitney(a * scale, b * scale)       # ranks don't care about units
    assert k.prob_treatment_higher == pytest.approx(r.prob_treatment_higher, abs=1e-12)


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 10**6), st.floats(0.01, 1))
def test_noninferiority_monotone_in_margin(seed, margin):
    r_ = np.random.default_rng(seed)
    a, b = r_.normal(0, 1, 200), r_.normal(-0.05, 1, 200)
    p1 = ab.noninferiority_test(a, b, margin).p_value
    p2 = ab.noninferiority_test(a, b, margin * 2).p_value
    assert p2 <= p1 + 1e-12
    assert 0 <= p1 <= 1


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 10**6))
def test_cluster_test_swaps_and_counts_clusters(seed):
    r_ = np.random.default_rng(seed)
    gc, gt = r_.integers(0, 30, 400), r_.integers(0, 25, 300)
    yc, yt = r_.normal(0, 1, 400) + r_.normal(0, 0.5, 30)[gc], r_.normal(0, 1, 300) + r_.normal(0, 0.5, 25)[gt]
    assume(len(np.unique(gc)) > 2 and len(np.unique(gt)) > 2)
    r, s = ab.cluster_test(yc, yt, gc, gt), ab.cluster_test(yt, yc, gt, gc)
    assert s.lift == pytest.approx(-r.lift, abs=1e-12)
    assert s.p_value == pytest.approx(r.p_value, rel=1e-9)
    assert r.n_control == len(np.unique(gc))
    # relabelling clusters changes nothing
    q = ab.cluster_test(yc, yt, gc + 1000, gt * 3)
    assert q.p_value == pytest.approx(r.p_value, rel=1e-12)


@settings(max_examples=20, deadline=None)
@given(st.integers(2, 8), st.sampled_from(["pocock", "obrien_fleming"]), st.booleans())
def test_group_sequential_bounds(k, kind, spending):
    d = ab.group_sequential_design(k, kind=kind, spending=spending)
    assert d.stop_probabilities(0).sum() == pytest.approx(0.05, abs=1e-5)
    assert (d.bounds > 1.5).all()
    if kind == "obrien_fleming":
        assert (np.diff(d.bounds) < 0).all()      # strict early, close to 1.96 at the end
    assert d.power(3.0) > d.power(2.0)
