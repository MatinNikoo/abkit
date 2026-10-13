"""
Sequential testing: results you're allowed to check every day.

The problem. A normal p value assumes you look once, at a sample size you
picked in advance. If you check every day and stop the first time p < 0.05,
the real false positive rate is far higher than 5% (it climbs toward 100%
the longer you keep peeking). Everyone peeks, so this matters.

The fix used here is the mixture sequential probability ratio test (mSPRT)
from Johari, Koomen, Pekelis and Walsh (2017), the method behind
Optimizely's stats engine. Instead of one p value it builds a running
likelihood ratio that stays valid at every look. The matching confidence
sequence (Howard, Ramdas, McAuliffe and Sekhon, 2021) is a confidence
interval that holds at every look at the same time.

tau sets the scale of effects you expect. A good default is roughly the
smallest effect you'd care about. Results stay valid for any tau, it only
changes how fast you can stop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SequentialResult:
    n_control: np.ndarray          # users seen in control at each look
    n_treatment: np.ndarray
    estimate: np.ndarray           # treatment mean minus control mean
    ci_low: np.ndarray             # always valid confidence sequence
    ci_high: np.ndarray
    p_value: np.ndarray            # always valid p value, never goes up
    alpha: float

    @property
    def stopped(self) -> bool:
        return bool((self.p_value < self.alpha).any())

    @property
    def stop_index(self) -> int | None:
        hit = np.flatnonzero(self.p_value < self.alpha)
        return int(hit[0]) if len(hit) else None

    def __str__(self) -> str:
        i = self.stop_index
        if i is None:
            return (f"no decision after {self.n_control[-1] + self.n_treatment[-1]:,} users; "
                    f"effect between {self.ci_low[-1]:+.4g} and {self.ci_high[-1]:+.4g}")
        return (f"stopped at look {i + 1} with {self.n_control[i] + self.n_treatment[i]:,} users; "
                f"effect {self.estimate[i]:+.4g} ({self.ci_low[i]:+.4g} to {self.ci_high[i]:+.4g}), "
                f"always valid p = {self.p_value[i]:.4f}")


def _running_stats(x: np.ndarray, idx: np.ndarray):
    """Mean and variance of x[:k] for every k in idx, in one pass."""
    s1 = np.cumsum(x)[idx - 1]
    s2 = np.cumsum(x * x)[idx - 1]
    m = s1 / idx
    v = (s2 - idx * m * m) / np.maximum(idx - 1, 1)
    return m, np.maximum(v, 0.0)


def msprt(control, treatment, tau: float, alpha: float = 0.05, n_looks: int = 50,
          min_per_arm: int = 100) -> SequentialResult:
    """
    Run the mSPRT on data in arrival order.

    control and treatment are each arm's outcomes in the order users arrived.
    The test looks n_looks times at evenly spaced points. With a normal
    mixture prior N(0, tau^2) on the true difference, the likelihood ratio
    after seeing an estimate d with variance V has a closed form:

        L = sqrt(V / (V + tau^2)) * exp(tau^2 d^2 / (2 V (V + tau^2)))

    and p = min(1, 1 / L), kept as a running minimum. By Ville's inequality
    the chance that L ever crosses 1/alpha under the null is at most alpha.
    The variance is plugged in from the data, which makes this valid for
    large samples, the normal case for online tests.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    nc_all, nt_all = len(c), len(t)
    looks = np.linspace(0, 1, n_looks + 1)[1:]
    ic = np.maximum((looks * nc_all).astype(int), 2)
    it = np.maximum((looks * nt_all).astype(int), 2)
    keep = (ic >= min_per_arm) & (it >= min_per_arm)
    if not keep.any():
        keep[-1] = True
    ic, it = ic[keep], it[keep]

    mc, vc = _running_stats(c, ic)
    mt, vt = _running_stats(t, it)
    d = mt - mc
    V = vc / ic + vt / it
    V = np.maximum(V, 1e-300)
    t2 = tau * tau

    log_L = 0.5 * np.log(V / (V + t2)) + t2 * d * d / (2 * V * (V + t2))
    p = np.minimum.accumulate(np.minimum(1.0, np.exp(-log_L)))

    half = np.sqrt(V * (V + t2) / t2 * (2 * np.log(1 / alpha) + np.log((V + t2) / V)))
    lo, hi = d - half, d + half
    # a confidence sequence can be intersected over time and stays valid
    lo, hi = np.maximum.accumulate(lo), np.minimum.accumulate(hi)
    return SequentialResult(ic, it, d, lo, hi, p, alpha)


def sequential_test(control, treatment, mde: float, alpha: float = 0.05, n_looks: int = 50) -> SequentialResult:
    """mSPRT with tau set to the smallest effect you care about."""
    return msprt(control, treatment, tau=abs(mde), alpha=alpha, n_looks=n_looks)


def naive_peeking(control, treatment, alpha: float = 0.05, n_looks: int = 50, min_per_arm: int = 100) -> np.ndarray:
    """
    The wrong way, kept for comparison: an ordinary z test p value at each
    look. Stopping the first time this dips below alpha inflates false positives.
    """
    from scipy import stats

    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    looks = np.linspace(0, 1, n_looks + 1)[1:]
    ic = np.maximum((looks * len(c)).astype(int), 2)
    it = np.maximum((looks * len(t)).astype(int), 2)
    keep = (ic >= min_per_arm) & (it >= min_per_arm)
    ic, it = ic[keep], it[keep]
    mc, vc = _running_stats(c, ic)
    mt, vt = _running_stats(t, it)
    z = (mt - mc) / np.sqrt(np.maximum(vc / ic + vt / it, 1e-300))
    return 2 * stats.norm.sf(np.abs(z))
