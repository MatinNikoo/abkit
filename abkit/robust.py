"""
Tools for skewed metrics like revenue, where a few big spenders move the
average around and the t test's normal approximation can be off.

Each one answers a slightly different question, so read the docstring
before swapping one for another:

* bootstrap_test: the same question as the t test (difference in means),
  with an interval that doesn't assume normality.
* winsorize: caps extreme values. Much less noise, but the answer is now
  about the capped mean, which is biased for the real mean.
* quantile_test: did the median (or the 90th percentile) move?
* mann_whitney: is a random treatment user likely to beat a random control
  user? Not a test of means.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .core import TestResult, _as_float


def winsorize(control, treatment, upper: float | None = 0.99, lower: float | None = None):
    """
    Cap values above the `upper` quantile (and below `lower`) of both groups
    pooled. Returns (control, treatment, (low_cap, high_cap)).

    Capping cuts the variance a lot on heavy tailed metrics, but the result
    estimates the change in the capped mean. If the treatment works by
    making big spenders spend more, capping hides part of that. In the
    stress tests a 99th percentile cap on lognormal revenue lost 10% of
    the true lift at sd 1 and 37% at sd 2 (see docs/validation.md).
    """
    c, t = _as_float(control), _as_float(treatment)
    both = np.concatenate([c, t])
    lo = float(np.quantile(both, lower)) if lower is not None else -np.inf
    hi = float(np.quantile(both, upper)) if upper is not None else np.inf
    return np.clip(c, lo, hi), np.clip(t, lo, hi), (lo, hi)


def _stat_fn(stat):
    if stat == "mean":
        return lambda a, axis=-1: a.mean(axis=axis)
    if stat == "median":
        return lambda a, axis=-1: np.median(a, axis=axis)
    if isinstance(stat, float):
        return lambda a, axis=-1: np.quantile(a, stat, axis=axis)
    if callable(stat):
        return stat
    raise ValueError("stat must be 'mean', 'median', a quantile like 0.9, or a function")


def _resample(x, f, n_boot, rng):
    # chunked so a 1M row metric doesn't need a 1M x 2000 matrix
    out = np.empty(n_boot)
    step = max(1, int(2_000_000 // len(x)))
    for i in range(0, n_boot, step):
        k = min(step, n_boot - i)
        out[i:i + k] = f(x[rng.integers(0, len(x), (k, len(x)))], axis=1)
    return out


def _resample_mean_var(x, n_boot, rng):
    m, v = np.empty(n_boot), np.empty(n_boot)
    step = max(1, int(2_000_000 // len(x)))
    for i in range(0, n_boot, step):
        k = min(step, n_boot - i)
        r = x[rng.integers(0, len(x), (k, len(x)))]
        m[i:i + k], v[i:i + k] = r.mean(axis=1), r.var(axis=1, ddof=1)
    return m, v


def bootstrap_test(control, treatment, stat="mean", n_boot: int = 2000, alpha: float = 0.05,
                   seed: int = 0, metric: str = "metric", method: str = "percentile") -> TestResult:
    """
    Percentile bootstrap for the difference in any statistic (mean by
    default). Each group is resampled on its own, which matches how the
    users were split. The p value is the bootstrap one: twice the smaller
    share of resampled differences on either side of zero, so it can't go
    below about 2 / n_boot.

    method="studentized" (means only) bootstraps the t statistic instead
    (bootstrap-t). It corrects for skew, which the percentile version and
    the t test both miss, so it's the one to use for a skewed metric with
    uneven groups.

    The bootstrap is not a fix for everything. With heavy tails it can
    undercover too, because a sample of a few thousand users rarely
    contains the whales that drive the real mean.
    """
    c, t = _as_float(control), _as_float(treatment)
    rng = np.random.default_rng(seed)
    if method == "studentized":
        if stat != "mean":
            raise ValueError("the studentized bootstrap here is for means only")
        return _bootstrap_t(c, t, n_boot, alpha, rng, metric)
    if method != "percentile":
        raise ValueError("method must be 'percentile' or 'studentized'")
    f = _stat_fn(stat)
    bc, bt = _resample(c, f, n_boot, rng), _resample(t, f, n_boot, rng)
    sc, st = float(f(c, axis=0)), float(f(t, axis=0))
    diff = st - sc
    d = bt - bc
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1 / n_boot)
    if sc != 0:
        rel = bt / bc - 1
        rel_ci = tuple(float(v) for v in np.quantile(rel[np.isfinite(rel)], [alpha / 2, 1 - alpha / 2]))
        rel_lift = st / sc - 1
    else:
        rel_lift, rel_ci = float("nan"), (float("nan"), float("nan"))
    name = stat if isinstance(stat, str) else (f"q{stat:g}" if isinstance(stat, float) else "statistic")
    return TestResult(metric, sc, st, diff, float(d.std(ddof=1)), (float(lo), float(hi)), float(p),
                      float(rel_lift), rel_ci, len(c), len(t), alpha,
                      [f"bootstrap of the {name}, {n_boot:,} resamples"])


def _bootstrap_t(c, t, n_boot, alpha, rng, metric):
    nc, nt = len(c), len(t)
    diff = t.mean() - c.mean()
    se = np.sqrt(c.var(ddof=1) / nc + t.var(ddof=1) / nt)
    mc, vc = _resample_mean_var(c, n_boot, rng)
    mt, vt = _resample_mean_var(t, n_boot, rng)
    se_b = np.sqrt(vc / nc + vt / nt)
    ok = se_b > 0
    tb = ((mt - mc) - diff)[ok] / se_b[ok]
    q_lo, q_hi = np.quantile(tb, [alpha / 2, 1 - alpha / 2])
    ci = (float(diff - q_hi * se), float(diff - q_lo * se))
    t_obs = diff / se if se > 0 else 0.0
    p = min(1.0, 2 * min((tb >= t_obs).mean(), (tb <= t_obs).mean()) + 1 / n_boot)
    mean_c = c.mean()
    rel = diff / mean_c if mean_c else float("nan")
    rel_ci = (ci[0] / mean_c, ci[1] / mean_c) if mean_c else (float("nan"), float("nan"))
    return TestResult(metric, float(mean_c), float(t.mean()), float(diff), float(se), ci, float(p),
                      float(rel), rel_ci, nc, nt, alpha, [f"bootstrap-t of the mean, {n_boot:,} resamples"])


def quantile_test(control, treatment, q: float = 0.5, n_boot: int = 2000, alpha: float = 0.05,
                  seed: int = 0, metric: str = "metric") -> TestResult:
    """
    Quantile treatment effect: did the q-th quantile move? q = 0.5 is the
    median, q = 0.9 asks whether heavy users changed. Bootstrap interval.
    On metrics with lots of ties (most users at 0) a quantile can sit on
    the same value in every resample, and then the test can't see anything.
    """
    if not 0 < q < 1:
        raise ValueError("q must be between 0 and 1")
    return bootstrap_test(control, treatment, float(q), n_boot, alpha, seed, metric)


@dataclass
class RankResult:
    metric: str
    prob_treatment_higher: float      # P(T > C) + half of P(T = C)
    ci: tuple[float, float]
    p_value: float                    # Mann-Whitney, two sided
    alpha: float = 0.05

    def __str__(self) -> str:
        return (f"{self.metric}: a random treatment user beats a random control user "
                f"{self.prob_treatment_higher:.1%} of the time "
                f"({1 - self.alpha:.0%} CI {self.ci[0]:.1%} to {self.ci[1]:.1%}); Mann-Whitney p = {self.p_value:.4f}")


def mann_whitney(control, treatment, alpha: float = 0.05, metric: str = "metric") -> RankResult:
    """
    Mann-Whitney U test plus the number it actually estimates: the chance
    a random treatment user has a higher value than a random control user
    (0.5 means no difference). The interval uses the Brunner-Munzel
    variance, which stays right when the two groups have different shapes.

    This is not a test of means. Revenue can go up in total while most
    users spend a bit less, and then this says the opposite of the t test.
    And two groups with the same mean but different shapes get flagged.
    Use it for "do users generally do better", not for "is revenue up".
    """
    c, t = _as_float(control), _as_float(treatment)
    nc, nt = len(c), len(t)
    ranks = stats.rankdata(np.concatenate([c, t]))
    rc, rt = ranks[:nc], ranks[nc:]
    wc, wt = stats.rankdata(c), stats.rankdata(t)
    prob = (rt.mean() - (nt + 1) / 2) / nc
    vc = np.var(rc - wc, ddof=1) / nt**2
    vt = np.var(rt - wt, ddof=1) / nc**2
    se = np.sqrt(vc / nc + vt / nt)
    z = stats.norm.isf(alpha / 2)
    p = stats.mannwhitneyu(t, c, alternative="two-sided", method="asymptotic").pvalue
    return RankResult(metric, float(prob), (float(prob - z * se), float(prob + z * se)), float(p), alpha)
