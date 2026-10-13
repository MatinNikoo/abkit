"""
Fixed horizon tests: the usual "run it for N users, then look once" analysis.

Everything works on plain numpy arrays so it's easy to test and reuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    """What a test returns. Lift is treatment minus control."""

    metric: str
    control_mean: float
    treatment_mean: float
    lift: float
    se: float
    ci: tuple[float, float]
    p_value: float
    rel_lift: float
    rel_ci: tuple[float, float]
    n_control: int
    n_treatment: int
    alpha: float = 0.05
    notes: list[str] = field(default_factory=list)

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def __str__(self) -> str:
        lo, hi = self.rel_ci
        lines = [
            f"{self.metric}: control {self.control_mean:.4g}, treatment {self.treatment_mean:.4g}",
            f"  lift {self.lift:+.4g} ({self.rel_lift:+.2%}), {1 - self.alpha:.0%} CI {lo:+.2%} to {hi:+.2%}",
            f"  p = {self.p_value:.4f}{'  (significant)' if self.significant else ''}",
        ]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def _as_float(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim != 1 or len(a) < 2:
        raise ValueError("need a 1D array with at least 2 values")
    if not np.isfinite(a).all():
        raise ValueError("values must be finite (drop or fill missing values first)")
    return a


def _relative(mc, mt, var_mc, var_mt, z):
    """
    Relative lift (mt / mc - 1) and its CI by the delta method.

    The two group means are independent, so
    Var(mt/mc) ~ var_mt/mc^2 + mt^2 var_mc/mc^4.
    """
    if mc == 0:
        return float("nan"), (float("nan"), float("nan"))
    r = mt / mc
    se = np.sqrt(var_mt / mc**2 + mt**2 * var_mc / mc**4)
    return r - 1, (r - 1 - z * se, r - 1 + z * se)


def mean_test(control, treatment, alpha: float = 0.05, metric: str = "metric") -> TestResult:
    """
    Welch's t test on the difference in means.

    Works for averages (revenue per user) and for 0/1 outcomes (conversion),
    since a proportion is just the mean of a 0/1 column. With the sample
    sizes in online tests the t and normal answers are basically identical.
    """
    c, t = _as_float(control), _as_float(treatment)
    nc, nt = len(c), len(t)
    mc, mt = c.mean(), t.mean()
    vc, vt = c.var(ddof=1) / nc, t.var(ddof=1) / nt
    se = np.sqrt(vc + vt)
    diff = mt - mc
    # Welch-Satterthwaite degrees of freedom
    dof = (vc + vt) ** 2 / (vc**2 / (nc - 1) + vt**2 / (nt - 1)) if se > 0 else np.inf
    tcrit = stats.t.ppf(1 - alpha / 2, dof)
    p = 2 * stats.t.sf(abs(diff) / se, dof) if se > 0 else 1.0
    rel, rel_ci = _relative(mc, mt, vc, vt, stats.norm.ppf(1 - alpha / 2))
    return TestResult(metric, mc, mt, diff, se, (diff - tcrit * se, diff + tcrit * se),
                      float(p), rel, rel_ci, nc, nt, alpha)


def ratio_test(num_c, den_c, num_t, den_t, alpha: float = 0.05, metric: str = "ratio") -> TestResult:
    """
    Test a ratio metric like clicks per session, where the unit that was
    randomized (the user) isn't the unit in the denominator (the session).

    Treating each session as independent understates the variance, because
    sessions from the same user are correlated. The delta method gets the
    variance of sum(num) / sum(den) right at the user level
    (Deng, Knoblich and Lu, 2018).
    """
    def ratio_and_var(num, den):
        num, den = _as_float(num), _as_float(den)
        if len(num) != len(den):
            raise ValueError("numerator and denominator need one value per user")
        n = len(num)
        mx, my = num.mean(), den.mean()
        if my == 0:
            raise ValueError("denominator averages to zero")
        sxx, syy = num.var(ddof=1), den.var(ddof=1)
        sxy = np.cov(num, den, ddof=1)[0, 1]
        r = mx / my
        var = (sxx / my**2 - 2 * mx * sxy / my**3 + mx**2 * syy / my**4) / n
        return r, var, n

    rc, vc, nc = ratio_and_var(num_c, den_c)
    rt, vt, nt = ratio_and_var(num_t, den_t)
    se = np.sqrt(vc + vt)
    z = stats.norm.ppf(1 - alpha / 2)
    diff = rt - rc
    p = 2 * stats.norm.sf(abs(diff) / se) if se > 0 else 1.0
    rel, rel_ci = _relative(rc, rt, vc, vt, z)
    return TestResult(metric, rc, rt, diff, se, (diff - z * se, diff + z * se),
                      float(p), rel, rel_ci, nc, nt, alpha)


def cuped_adjust(y, x, theta=None):
    """
    CUPED (Deng, Xu, Kohavi and Walker, 2013).

    x is something measured before the experiment started, like each user's
    activity in the previous month. Subtract the part of y that x predicts:

        y_adj = y - theta * (x - mean(x)),   theta = cov(y, x) / var(x)

    The mean doesn't change, but the variance drops by a factor of about
    (1 - corr(x, y)^2). That's the same answer with fewer users.

    x can also be a 2D array with one column per covariate. Then theta is
    the least squares fit of y on all of them, and the variance drops by
    about (1 - R^2) of that fit.

    x must not be affected by the treatment, or the estimate gets biased.
    Pass theta if you want to reuse one estimated on pooled data.
    Returns (y_adj, theta).
    """
    y = _as_float(y)
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] != len(y):
        raise ValueError("y and x need the same number of rows")
    if not np.isfinite(X).all():
        raise ValueError("covariates must be finite")
    Xc = X - X.mean(axis=0)
    if theta is None:
        theta, *_ = np.linalg.lstsq(Xc, y - y.mean(), rcond=None)
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    y_adj = y - Xc @ theta
    return y_adj, (float(theta[0]) if theta.size == 1 else theta)


def heterogeneity_test(estimates, ses) -> float:
    """
    Cochran's Q test: do the segment effects differ by more than noise?

    Each segment's effect is weighted by 1 / se^2. Under "same effect
    everywhere", Q follows a chi square with (segments - 1) degrees of
    freedom. A small p value means the treatment really works differently
    in some segments.
    """
    d = np.asarray(estimates, dtype=float)
    w = 1 / np.asarray(ses, dtype=float) ** 2
    if len(d) < 2:
        raise ValueError("need at least two segments")
    pooled = (w * d).sum() / w.sum()
    q = (w * (d - pooled) ** 2).sum()
    return float(stats.chi2.sf(q, len(d) - 1))


def srm_test(n_control: int, n_treatment: int, expected_share: float = 0.5) -> float:
    """
    Sample ratio mismatch check. If you meant to split 50/50 and got 52/48,
    something in the assignment or logging is broken, and the results can't
    be trusted no matter how significant they look. Returns a chi square
    p value. Common practice is to stop and investigate below 0.001 to 0.01.
    """
    n = n_control + n_treatment
    exp = np.array([n * (1 - expected_share), n * expected_share])
    obs = np.array([n_control, n_treatment], dtype=float)
    return float(stats.chisquare(obs, exp).pvalue)


def sample_size(sd: float, mde: float, alpha: float = 0.05, power: float = 0.8,
                ratio: float = 1.0) -> tuple[int, int]:
    """
    Users needed per group to detect an absolute difference `mde` in a mean
    with standard deviation `sd`. `ratio` is treatment size over control size.
    Returns (n_control, n_treatment).
    """
    if mde <= 0 or sd <= 0:
        raise ValueError("sd and mde must be positive")
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    nc = (1 + 1 / ratio) * (z * sd / mde) ** 2
    return int(np.ceil(nc)), int(np.ceil(nc * ratio))


def proportion_sample_size(p: float, mde: float, alpha: float = 0.05,
                           power: float = 0.8, ratio: float = 1.0) -> tuple[int, int]:
    """Same as sample_size for a conversion rate p and an absolute lift mde."""
    if not 0 < p < 1:
        raise ValueError("p must be between 0 and 1")
    return sample_size(np.sqrt(p * (1 - p)), mde, alpha, power, ratio)
