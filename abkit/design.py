"""
Planning a test before it runs: how many users it needs, the smallest
effect it can find, how many days that takes, and an A/A simulator that
checks whether a metric behaves on your own data before you trust any
test on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .core import _as_float, _ratio_and_var, cluster_test, mean_test, sample_size


def _z(alpha, power):
    return stats.norm.isf(alpha / 2) + stats.norm.ppf(power)


def mde(sd: float, n_control: int, n_treatment: int | None = None, alpha: float = 0.05,
        power: float = 0.8) -> float:
    """
    Smallest absolute change in a mean that a test with these group sizes
    finds with the given power. The inverse of sample_size.
    """
    nt = n_control if n_treatment is None else n_treatment
    if sd <= 0 or n_control < 2 or nt < 2:
        raise ValueError("need sd > 0 and at least 2 users per group")
    return float(_z(alpha, power) * sd * np.sqrt(1 / n_control + 1 / nt))


def proportion_mde(p: float, n_control: int, n_treatment: int | None = None, alpha: float = 0.05,
                   power: float = 0.8) -> float:
    """Same as mde for a conversion rate p."""
    if not 0 < p < 1:
        raise ValueError("p must be between 0 and 1")
    return mde(np.sqrt(p * (1 - p)), n_control, n_treatment, alpha, power)


def ratio_sd(numerator, denominator) -> float:
    """
    Per user standard deviation of a ratio metric like clicks per session,
    from pilot data with one row per user. Plug it into sample_size or mde.

    This is the delta method variance times the number of users, so it
    counts the correlation between sessions of the same user. Using the
    per session sd instead (sqrt(p (1 - p)) for a click rate) is the usual
    mistake and it promises more power than you get.
    """
    _, var, n = _ratio_and_var(numerator, denominator)
    return float(np.sqrt(var * n))


def ratio_sample_size(numerator, denominator, mde: float, alpha: float = 0.05, power: float = 0.8,
                      ratio: float = 1.0) -> tuple[int, int]:
    """Users per group to detect an absolute change mde in a ratio metric, from pilot data."""
    return sample_size(ratio_sd(numerator, denominator), mde, alpha, power, ratio)


def ratio_mde(numerator, denominator, n_control: int, n_treatment: int | None = None,
              alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute change in a ratio metric these group sizes can find, from pilot data."""
    return mde(ratio_sd(numerator, denominator), n_control, n_treatment, alpha, power)


@dataclass
class RunPlan:
    users_needed: int
    users_per_day: float
    days: int                 # days to reach the sample
    days_recommended: int     # rounded up to whole weeks

    def __str__(self) -> str:
        return (f"{self.users_needed:,} users at {self.users_per_day:,.0f} a day: {self.days} days, "
                f"run {self.days_recommended} to cover whole weeks")


def run_length(n_per_group: int, users_per_day: float, groups: int = 2, traffic_share: float = 1.0,
               whole_weeks: bool = True) -> RunPlan:
    """
    How long a test has to run. users_per_day counts new users who enter
    the test each day, traffic_share is the part of traffic you give the
    test. With whole_weeks the answer rounds up to full weeks (at least
    one), so every day of the week shows up equally often. Weekend users
    often behave differently, and a test that ends on a Tuesday weights them wrong.
    """
    if users_per_day <= 0 or not 0 < traffic_share <= 1:
        raise ValueError("users_per_day must be positive and traffic_share in (0, 1]")
    need = int(n_per_group * groups)
    days = int(np.ceil(need / (users_per_day * traffic_share)))
    rec = int(max(7, np.ceil(days / 7) * 7)) if whole_weeks else days
    return RunPlan(need, users_per_day * traffic_share, days, rec)


@dataclass
class AAResult:
    sims: int
    false_positive_rate: float
    mc_se: float
    uniform_p: float          # KS test that the p values are uniform
    p_values: np.ndarray

    def __str__(self) -> str:
        flag = "" if abs(self.false_positive_rate - 0.05) < 3 * self.mc_se + 0.005 else "   check this metric"
        return (f"A/A on your data, {self.sims} splits: {self.false_positive_rate:.1%} came out "
                f"significant at 5% (+- {self.mc_se:.1%}); p values uniform? p = {self.uniform_p:.3f}{flag}")


def aa_test(values, sims: int = 1000, share: float = 0.5, clusters=None, test=None,
            seed: int = 0) -> AAResult:
    """
    Split real data at random many times, with no treatment at all, and
    see how often the test calls a difference. It should be about 5%,
    with p values spread evenly between 0 and 1. If it isn't, the metric
    breaks the test's assumptions (heavy tails, too few users, rows that
    aren't independent) and real results on it can't be read at face value.

    With clusters, whole clusters are assigned together, like the real test
    (all sessions of a user land in the same group), and the default test
    becomes cluster_test. test can be any function (control, treatment) ->
    result with a p_value. Passing clusters with test=mean_test shows what
    a row level analysis of a user randomized test does to false positives.
    """
    y = _as_float(values)
    rng = np.random.default_rng(seed)
    p = np.empty(sims)
    if clusters is not None:
        g = np.asarray(clusters)
        ids, inv = np.unique(g, return_inverse=True)
        for i in range(sims):
            t_rows = (rng.random(len(ids)) < share)[inv]
            if test is None:
                p[i] = cluster_test(y[~t_rows], y[t_rows], g[~t_rows], g[t_rows]).p_value
            else:
                p[i] = test(y[~t_rows], y[t_rows]).p_value
    else:
        test = test or mean_test
        for i in range(sims):
            t_rows = rng.random(len(y)) < share
            p[i] = test(y[~t_rows], y[t_rows]).p_value
    fp = float((p < 0.05).mean())
    return AAResult(sims, fp, float(np.sqrt(0.05 * 0.95 / sims)), float(stats.kstest(p, "uniform").pvalue), p)
