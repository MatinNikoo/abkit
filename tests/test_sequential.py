import numpy as np

import abkit as ab
from abkit.sequential import naive_peeking

rng = np.random.default_rng(11)


def false_positive_rates(n_sims=300, n=4000, looks=40):
    seq = naive = 0
    for _ in range(n_sims):
        a, b = rng.normal(0, 1, n), rng.normal(0, 1, n)
        seq += ab.msprt(a, b, tau=0.05, n_looks=looks).stopped
        naive += (naive_peeking(a, b, n_looks=looks) < 0.05).any()
    return seq / n_sims, naive / n_sims


def test_msprt_controls_false_positives_when_peeking():
    seq, naive = false_positive_rates()
    assert seq <= 0.06          # stays at or under alpha even with 40 looks
    assert naive > 0.12         # peeking at a normal p value does not


def test_p_value_never_goes_up_and_ci_shrinks():
    a, b = rng.normal(0, 1, 5000), rng.normal(0.08, 1, 5000)
    r = ab.msprt(a, b, tau=0.05)
    assert (np.diff(r.p_value) <= 1e-15).all()
    assert (np.diff(r.ci_high - r.ci_low) <= 1e-12).all()


def test_msprt_finds_a_real_effect():
    found = sum(ab.msprt(rng.normal(0, 1, 6000), rng.normal(0.1, 1, 6000), tau=0.1).stopped
                for _ in range(100))
    assert found > 80


def test_confidence_sequence_covers_truth_at_every_look():
    misses = 0
    for _ in range(200):
        r = ab.msprt(rng.normal(0, 1, 3000), rng.normal(0.05, 1, 3000), tau=0.05)
        misses += ((r.ci_low > 0.05) | (r.ci_high < 0.05)).any()
    assert misses / 200 <= 0.06
