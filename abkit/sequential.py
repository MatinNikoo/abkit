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
    # shift first: running sums of squares lose all precision when the mean is
    # huge next to the spread (a metric around 1e8 with sd 1 used to give p = 0)
    x0 = x[0]
    x = x - x0
    s1 = np.cumsum(x)[idx - 1]
    s2 = np.cumsum(x * x)[idx - 1]
    m = s1 / idx
    v = (s2 - idx * m * m) / np.maximum(idx - 1, 1)
    return m + x0, np.maximum(v, 0.0)


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
    # a group with no spread yet (0 conversions so far on a rare metric) has a
    # plug in variance of 0, and the interval collapses to a point that then
    # poisons every later look. skip those looks.
    keep &= (_running_stats(c, ic)[1] > 0) & (_running_stats(t, it)[1] > 0)
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


# ----------------------------------------------------- group sequential
# The other classic answer to peeking: decide in advance how many times
# you'll look (say 5), then use a stricter cutoff at each look so the total
# false positive rate stays at alpha. O'Brien Fleming spends almost nothing
# early and keeps the last look close to a normal test. Pocock uses about
# the same cutoff at every look, so it stops early more often but pays for
# it at the end. The Lan DeMets spending versions let the looks happen at
# any time, as long as the decision to look doesn't depend on the data.

from functools import lru_cache

from scipy import optimize, stats


def spend(t, alpha: float = 0.05, kind: str = "obrien_fleming"):
    """Share of alpha used up by information fraction t (Lan and DeMets, 1983)."""
    t = np.clip(np.asarray(t, dtype=float), 0, 1)
    if kind == "obrien_fleming":
        # alpha / 2 spent on each side with 2 - 2 Phi(z_{alpha/4} / sqrt(t)), the
        # convention in gsDesign, rpact and ldbounds, so the boundaries match theirs
        with np.errstate(divide="ignore"):
            return 4 * stats.norm.sf(stats.norm.isf(alpha / 4) / np.sqrt(t))
    if kind == "pocock":
        return alpha * np.log(1 + (np.e - 1) * t)
    raise ValueError("kind must be 'obrien_fleming' or 'pocock'")


class _Walk:
    """
    Distribution of the score S_k = Z_k sqrt(t_k), a Brownian motion in
    information time, kept on a grid over the region where the test hasn't
    stopped yet. Crossing probabilities come from integrating the normal
    increment over that grid (Armitage, McPherson and Rowe, 1969).
    """

    def __init__(self, drift: float = 0.0, points: int = 301):
        self.drift, self.points = drift, points
        self.grid = self.dens = None
        self.t = 0.0

    def _step(self, t):
        dt = t - self.t
        mu, sd = self.drift * dt, np.sqrt(dt)
        return mu, sd

    def cross(self, t, b):
        """P(not stopped before, and |Z| >= b at information t)."""
        mu, sd = self._step(t)
        edge = b * np.sqrt(t)
        if self.grid is None:
            return float(stats.norm.sf((edge - mu) / sd) + stats.norm.cdf((-edge - mu) / sd))
        s = self.grid + mu
        up = stats.norm.sf((edge - s) / sd)
        down = stats.norm.cdf((-edge - s) / sd)
        return float(np.dot(self.w * self.dens, up + down))

    def advance(self, t, b):
        mu, sd = self._step(t)
        edge = b * np.sqrt(t)
        grid = np.linspace(-edge, edge, self.points)
        w = np.full(self.points, 2.0)                     # Simpson weights
        w[1::2] = 4.0
        w[0] = w[-1] = 1.0
        w *= (grid[1] - grid[0]) / 3
        if self.grid is None:
            dens = stats.norm.pdf((grid - mu) / sd) / sd
        else:
            u = (grid[:, None] - self.grid[None, :] - mu) / sd
            dens = (np.exp(-0.5 * u * u) / (sd * np.sqrt(2 * np.pi))) @ (self.w * self.dens)
        self.grid, self.w, self.dens, self.t = grid, w, dens, t


def _bounds_spending(info, alpha, kind):
    cum = spend(info, alpha, kind)
    walk, bounds = _Walk(), []
    prev = 0.0
    for t, a in zip(info, cum):
        target = a - prev
        if target <= 1e-15:
            b = 40.0
        else:
            b = optimize.brentq(lambda b: walk.cross(t, b) - target, 0.01, 40, xtol=1e-10)
        bounds.append(b)
        prev = a
        walk.advance(t, b)
    return np.array(bounds)


def _crossing(info, bounds, drift=0.0):
    """Probability of stopping at each look."""
    walk, out = _Walk(drift), []
    for t, b in zip(info, bounds):
        out.append(walk.cross(t, b))
        walk.advance(t, b)
    return np.array(out)


def _bounds_classic(info, alpha, kind):
    shape = np.ones(len(info)) if kind == "pocock" else 1 / np.sqrt(info)
    c = optimize.brentq(lambda c: _crossing(info, c * shape).sum() - alpha, 0.5, 20, xtol=1e-10)
    return c * shape


@lru_cache(maxsize=64)
def _bounds(info: tuple, alpha: float, kind: str, spending: bool) -> tuple:
    f = _bounds_spending if spending else _bounds_classic
    return tuple(f(np.array(info), alpha, kind))


@dataclass
class GroupSequentialDesign:
    info: np.ndarray        # information fraction at each look, ending at 1
    bounds: np.ndarray      # stop if |z| at a look is at least this
    alpha: float
    kind: str
    spending: bool

    def stop_probabilities(self, drift: float = 0.0) -> np.ndarray:
        """
        Chance of stopping at each look when the z score at the final look
        would have mean `drift` (0 means no effect).
        """
        return _crossing(self.info, self.bounds, drift)

    def power(self, drift: float) -> float:
        return float(self.stop_probabilities(drift).sum())

    def inflation(self, power: float = 0.8) -> float:
        """
        How many more users the design needs at its last look to keep the
        same power as a single look test. Multiply a fixed sample size by this.
        """
        fixed = stats.norm.isf(self.alpha / 2) + stats.norm.ppf(power)
        drift = optimize.brentq(lambda d: self.power(d) - power, 0.1, 20, xtol=1e-10)
        return float((drift / fixed) ** 2)

    def expected_sample(self, drift: float) -> float:
        """Average share of the maximum sample used before stopping."""
        p = self.stop_probabilities(drift)
        return float((p * self.info).sum() + (1 - p.sum()) * 1.0)

    def __str__(self) -> str:
        name = ("Lan DeMets " if self.spending else "") + self.kind.replace("_", " ").title()
        rows = [f"{name} design, {len(self.info)} looks, alpha {self.alpha}"]
        rows += [f"  look {i + 1} at {t:.0%} of users: stop if |z| >= {b:.3f}"
                 for i, (t, b) in enumerate(zip(self.info, self.bounds))]
        return "\n".join(rows)


def group_sequential_design(n_looks: int = 5, alpha: float = 0.05, kind: str = "obrien_fleming",
                            spending: bool = True, info=None) -> GroupSequentialDesign:
    """
    Two sided boundaries for a test you'll check n_looks times.

    kind is "obrien_fleming" or "pocock". spending=True uses the Lan DeMets
    spending function versions (the usual choice now). spending=False gives
    the original constant shape boundaries, which assume equally spaced
    looks. info lets you set the share of users at each look yourself.
    """
    if kind not in ("obrien_fleming", "pocock"):
        raise ValueError("kind must be 'obrien_fleming' or 'pocock'")
    if info is None:
        info = np.arange(1, n_looks + 1) / n_looks
    info = np.asarray(info, dtype=float)
    if (np.diff(info) <= 0).any() or info[0] <= 0 or abs(info[-1] - 1) > 1e-12:
        raise ValueError("info must increase and end at 1")
    b = np.array(_bounds(tuple(np.round(info, 12)), float(alpha), kind, bool(spending)))
    return GroupSequentialDesign(info, b, alpha, kind, spending)


@dataclass
class GroupSequentialResult:
    design: GroupSequentialDesign
    n_control: np.ndarray
    n_treatment: np.ndarray
    estimate: np.ndarray
    z: np.ndarray
    ci_low: np.ndarray      # repeated confidence intervals, valid at every planned look
    ci_high: np.ndarray

    @property
    def stopped(self) -> bool:
        return bool((np.abs(self.z) >= self.design.bounds).any())

    @property
    def stop_index(self) -> int | None:
        hit = np.flatnonzero(np.abs(self.z) >= self.design.bounds)
        return int(hit[0]) if len(hit) else None

    def __str__(self) -> str:
        i = self.stop_index
        if i is None:
            return (f"no decision at any of {len(self.z)} looks; final z = {self.z[-1]:+.3f} "
                    f"(needed {self.design.bounds[-1]:.3f})")
        return (f"stopped at look {i + 1} with {self.n_control[i] + self.n_treatment[i]:,} users; "
                f"effect {self.estimate[i]:+.4g} ({self.ci_low[i]:+.4g} to {self.ci_high[i]:+.4g}), "
                f"z = {self.z[i]:+.3f} vs bound {self.design.bounds[i]:.3f}")


def group_sequential_test(control, treatment, n_looks: int = 5, alpha: float = 0.05,
                          kind: str = "obrien_fleming", spending: bool = True) -> GroupSequentialResult:
    """
    Replay a group sequential test on data in arrival order, looking at
    n_looks evenly spaced points. The full arrays are the planned maximum
    sample. The z score at each look is the usual Welch statistic, and the
    interval at look k is estimate +- bound_k * se_k (a repeated
    confidence interval, valid at every planned look at once).
    """
    design = group_sequential_design(n_looks, alpha, kind, spending)
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    ic = np.maximum(np.ceil(design.info * len(c)).astype(int), 2)
    it = np.maximum(np.ceil(design.info * len(t)).astype(int), 2)
    mc, vc = _running_stats(c, ic)
    mt, vt = _running_stats(t, it)
    d = mt - mc
    se = np.sqrt(vc / ic + vt / it)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, d / np.where(se > 0, se, 1), np.where(d == 0, 0.0, np.sign(d) * np.inf))
    return GroupSequentialResult(design, ic, it, d, z, d - design.bounds * se, d + design.bounds * se)
