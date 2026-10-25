"""
Fixed horizon tests: the usual "run it for N users, then look once" analysis.

Everything works on plain numpy arrays so it's easy to test and reuse.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

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
    ci_level: float | None = None      # set when the interval isn't the usual 1 - alpha

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def to_dict(self) -> dict:
        """Every field as plain numbers and lists, ready for json.dumps."""
        from .data import plain
        return plain({**asdict(self), "significant": self.significant})

    def __str__(self) -> str:
        lo, hi = self.rel_ci
        lines = [
            f"{self.metric}: control {self.control_mean:.4g}, treatment {self.treatment_mean:.4g}",
            f"  lift {self.lift:+.4g} ({self.rel_lift:+.2%}), "
            f"{self.ci_level or 1 - self.alpha:.0%} CI {lo:+.2%} to {hi:+.2%}",
            f"  {fmt_p(self.p_value)}{'  (significant)' if self.significant else ''}",
        ]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def fmt_p(p: float, label: str = "p") -> str:
    """p = 0.0123, or p < 0.0001 when it is smaller than the four digits show."""
    return f"{label} < 0.0001" if p < 1e-4 else f"{label} = {p:.4f}"


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
    notes = []
    g = _diff_skewness(c, t)
    if abs(g) > SKEW_LIMIT:
        notes.append(f"skewed metric with uneven groups ({nc:,} vs {nt:,}): the difference has skewness "
                     f"{g:+.2f}, so one side of the interval is too short and false positives pile up "
                     "on that side. More users in the smaller group is the real fix; capping or "
                     "bootstrap_test(method='studentized') help partly")
    return TestResult(metric, mc, mt, diff, se, (diff - tcrit * se, diff + tcrit * se),
                      float(p), rel, rel_ci, nc, nt, alpha, notes)


# Kohavi et al. (2014) want n > 355 s^2 users for a mean with skewness s, which is
# the same as the mean's own skewness s / sqrt(n) staying under 1 / sqrt(355)
SKEW_LIMIT = 1 / np.sqrt(355)


def _diff_skewness(c, t) -> float:
    """
    Skewness of mean(t) - mean(c) if both groups had the same shape. The
    skew cancels when the groups are the same size and doesn't when one is
    much smaller, which is where the normal approximation goes wrong.
    """
    nc, nt = len(c), len(t)
    if nc == nt:
        return 0.0
    rc, rt = c - c.mean(), t - t.mean()
    m2 = (np.dot(rc, rc) + np.dot(rt, rt)) / (nc + nt)
    if m2 <= 0:
        return 0.0
    m3 = ((rc**3).sum() + (rt**3).sum()) / (nc + nt)
    skew = m3 / m2**1.5
    return float(skew * (1 / nt**2 - 1 / nc**2) / (1 / nc + 1 / nt) ** 1.5)


def _ratio_and_var(num, den):
    """sum(num) / sum(den) and its delta method variance, one row per user."""
    num, den = _as_float(num), _as_float(den)
    if len(num) != len(den):
        raise ValueError("numerator and denominator need one value per user")
    n = len(num)
    mx, my = num.mean(), den.mean()
    if my == 0:
        raise ValueError("denominator averages to zero")
    r = mx / my
    # same as (sxx / my^2 - 2 mx sxy / my^3 + mx^2 syy / my^4) / n, written as
    # the variance of the linearized values so it doesn't cancel badly when
    # every user has nearly the same ratio
    z = num - r * den
    var = np.dot(z, z) / (n - 1) / n / my**2
    return r, var, n


def ratio_test(num_c, den_c, num_t, den_t, alpha: float = 0.05, metric: str = "ratio") -> TestResult:
    """
    Test a ratio metric like clicks per session, where the unit that was
    randomized (the user) isn't the unit in the denominator (the session).

    Treating each session as independent understates the variance, because
    sessions from the same user are correlated. The delta method gets the
    variance of sum(num) / sum(den) right at the user level
    (Deng, Knoblich and Lu, 2018).
    """
    rc, vc, nc = _ratio_and_var(num_c, den_c)
    rt, vt, nt = _ratio_and_var(num_t, den_t)
    se = np.sqrt(vc + vt)
    z = stats.norm.ppf(1 - alpha / 2)
    diff = rt - rc
    p = 2 * stats.norm.sf(abs(diff) / se) if se > 0 else 1.0
    rel, rel_ci = _relative(rc, rt, vc, vt, z)
    return TestResult(metric, rc, rt, diff, se, (diff - z * se, diff + z * se),
                      float(p), rel, rel_ci, nc, nt, alpha)


def noninferiority_test(control, treatment, margin: float, higher_is_better: bool = True,
                        relative: bool = False, alpha: float = 0.05, metric: str = "metric") -> TestResult:
    """
    Guardrail check: is the treatment no worse than control by more than
    `margin`? Use it for metrics you don't expect to improve but can't
    afford to hurt, like page load time or crash rate.

    A plain test that "didn't find a drop" is not evidence there's no drop;
    it may just have too few users. This flips the question: the null is
    "it's worse by at least margin", and a small p value is evidence that
    it isn't. One sided Welch test. relative=True reads margin as a share of
    the control mean (0.01 is 1%), treating that mean as known.

    The result's p_value is one sided, and ci is the (1 - 2 alpha) interval,
    so "passes" is the same as the interval staying clear of the margin.
    """
    if margin <= 0:
        raise ValueError("margin must be positive")
    c, t = _as_float(control), _as_float(treatment)
    r = mean_test(c, t, 2 * alpha, metric)
    m = margin * abs(r.control_mean) if relative else margin
    if r.se > 0:
        vc, vt = c.var(ddof=1) / len(c), t.var(ddof=1) / len(t)
        dof = (vc + vt) ** 2 / (vc**2 / (len(c) - 1) + vt**2 / (len(t) - 1))
        # distance from the margin, in the direction that counts as "not worse"
        stat = (r.lift + m) / r.se if higher_is_better else (m - r.lift) / r.se
        p = float(stats.t.sf(stat, dof))
    else:
        p = 0.0 if (r.lift > -m if higher_is_better else r.lift < m) else 1.0
    worst = r.ci[0] if higher_is_better else r.ci[1]
    ok = p < alpha
    r.p_value, r.alpha, r.ci_level = p, alpha, 1 - 2 * alpha
    r.notes = [f"non-inferiority margin {'-' if higher_is_better else '+'}{m:.4g}: "
               + ("passes" if ok else "not shown") + f" (worst case in the interval {worst:+.4g})"] + r.notes
    return r


def _cluster_sums(y, clusters):
    y = _as_float(y)
    g = np.asarray(clusters)
    if g.shape != y.shape:
        raise ValueError("need one cluster id per row")
    _, inv = np.unique(g, return_inverse=True)
    return np.bincount(inv, weights=y), np.bincount(inv).astype(float)


def _cluster_t(S, K, A):
    """Studentized ratio difference for each row of the 0/1 assignment matrix A."""
    def arm(a):
        n = a.sum(axis=1)
        s, k = a @ S, a @ K
        r = s / k
        ss = a @ (S * S) - 2 * r * (a @ (S * K)) + r * r * (a @ (K * K))
        return r, ss / ((n - 1) * n * (k / n) ** 2)
    rt, vt = arm(A)
    rc, vc = arm(1 - A)
    return (rt - rc) / np.sqrt(vc + vt)


def cluster_test(control, treatment, clusters_c, clusters_t, alpha: float = 0.05,
                 metric: str = "metric", permutations: int = 0, seed: int = 0) -> TestResult:
    """
    Difference in means when rows come in groups that aren't independent:
    several sessions per user, or whole cities or classrooms randomized
    together. A row level t test treats every row as new information and
    its intervals come out too narrow.

    Rows are summed per cluster and the mean is analyzed as the ratio
    sum(y) / rows with the delta method, which is the same thing as a
    cluster robust standard error. With few clusters the normal
    approximation is too optimistic, so the p value and interval use a t
    distribution with Welch degrees of freedom counted in clusters.
    n_control and n_treatment in the result count clusters.

    With only a handful of clusters, or clusters of very different sizes,
    even that runs a bit liberal. permutations=2000 replaces the p value
    with a randomization test: reshuffle which clusters got treatment,
    recompute the same statistic, and see how extreme the real one is. That
    p value is exact for the sharp null (treatment changes no cluster's
    outcome) and only approximate for the null of no average effect, though
    the studentized statistic kept it near 5% in the stress tests. The
    interval stays the t interval.
    """
    sc, kc = _cluster_sums(control, clusters_c)
    st, kt = _cluster_sums(treatment, clusters_t)
    if min(len(sc), len(st)) < 2:
        raise ValueError("need at least 2 clusters per group")
    rc, vc, gc = _ratio_and_var(sc, kc)
    rt, vt, gt = _ratio_and_var(st, kt)
    se = np.sqrt(vc + vt)
    diff = rt - rc
    dof = (vc + vt) ** 2 / (vc**2 / (gc - 1) + vt**2 / (gt - 1)) if se > 0 else np.inf
    tcrit = stats.t.ppf(1 - alpha / 2, dof)
    p = 2 * stats.t.sf(abs(diff) / se, dof) if se > 0 else 1.0
    rel, rel_ci = _relative(rc, rt, vc, vt, tcrit)
    notes = [f"{gc:,} and {gt:,} clusters ({int(kc.sum()):,} and {int(kt.sum()):,} rows)"]
    if permutations and se > 0:
        S, K = np.concatenate([sc, st]), np.concatenate([kc, kt])
        rng = np.random.default_rng(seed)
        A = np.zeros((permutations, gc + gt))
        idx = np.argsort(rng.random((permutations, gc + gt)), axis=1)[:, :gt]
        np.put_along_axis(A, idx, 1.0, axis=1)
        z_perm = _cluster_t(S, K, A)
        p = float((1 + (np.abs(z_perm) >= abs(diff / se) - 1e-12).sum()) / (permutations + 1))
        notes.append(f"p value from {permutations:,} reshuffles of the clusters")
    elif min(gc, gt) < 20:
        notes.append("fewer than 20 clusters in a group, so the interval leans on the t "
                     "approximation; treat a borderline p value as borderline")
    return TestResult(metric, rc, rt, diff, se, (diff - tcrit * se, diff + tcrit * se),
                      float(p), rel, rel_ci, gc, gt, alpha, notes)


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


def adjusted_test(control, treatment, x_control, x_treatment, alpha: float = 0.05,
                  metric: str = "metric") -> TestResult:
    """
    Regression adjustment with a separate slope in each group (Lin, 2013).
    Like CUPED, but the covariate is allowed to predict the outcome
    differently under treatment, which happens when the effect is bigger
    for some users than others. With a 50/50 split it gives about the same
    answer as CUPED. With an uneven split and an effect that varies by user
    it can be noticeably more precise (see docs/validation.md).

    x_control and x_treatment are pre period covariates, one column each or
    a 2D array. The standard error includes the extra term for the two
    slopes being different, which the usual robust standard error leaves
    out when users are a sample from a bigger population.
    """
    yc, yt = _as_float(control), _as_float(treatment)
    Xc, Xt = np.asarray(x_control, dtype=float), np.asarray(x_treatment, dtype=float)
    Xc, Xt = (Xc[:, None] if Xc.ndim == 1 else Xc), (Xt[:, None] if Xt.ndim == 1 else Xt)
    if len(Xc) != len(yc) or len(Xt) != len(yt):
        raise ValueError("covariates need one row per user")
    X = np.vstack([Xc, Xt])
    xbar = X.mean(axis=0)
    k = X.shape[1]
    est, var, slopes = [], [], []
    for y, Z in ((yc, Xc), (yt, Xt)):
        D = np.column_stack([np.ones(len(y)), Z - xbar])
        beta, *_ = np.linalg.lstsq(D, y, rcond=None)
        e = y - D @ beta
        est.append(beta[0])
        var.append(e @ e / (len(y) - k - 1) / len(y))
        slopes.append(beta[1:])
    dslope = slopes[1] - slopes[0]
    sx = np.atleast_2d(np.cov(X, rowvar=False))
    se = float(np.sqrt(var[0] + var[1] + dslope @ sx @ dslope / len(X)))
    diff = float(est[1] - est[0])
    z = stats.norm.ppf(1 - alpha / 2)
    p = float(2 * stats.norm.sf(abs(diff) / se)) if se > 0 else 1.0
    mc, mt = yc.mean(), yt.mean()
    rel = diff / mc if mc else float("nan")
    rel_ci = ((diff - z * se) / mc, (diff + z * se) / mc) if mc else (float("nan"), float("nan"))
    raw = np.sqrt(yc.var(ddof=1) / len(yc) + yt.var(ddof=1) / len(yt))
    cut = 1 - (se / raw) ** 2 if raw > 0 else 0.0
    return TestResult(metric, mc, mt, diff, se, (diff - z * se, diff + z * se), p, rel, rel_ci,
                      len(yc), len(yt), alpha, [f"regression adjustment by group cut variance by {cut:.0%}"])


def cupac_adjust(y, features, model=None, folds: int = 2, seed: int = 0):
    """
    CUPAC: CUPED where the covariate is a model's prediction of y from
    pre period features (the idea DoorDash's experimentation team wrote up
    in 2020). A model can use many features and nonlinear patterns, so it
    usually removes more noise than a few linear covariates.

    The catch is leakage. A model trained on the same rows it predicts
    partly memorizes each user's y, and then subtracting the prediction
    eats into the treatment effect. So the prediction for every row comes
    from a model fit on the other folds only (cross fitting). The model
    is fit on both groups pooled, never on the assignment.

    model: anything with fit(X, y) and predict(X), like a scikit-learn
    regressor. Default is scikit-learn's HistGradientBoostingRegressor.
    Returns (y_adj, theta, prediction).
    """
    import copy

    y = _as_float(y)
    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] != len(y):
        raise ValueError("y and features need the same number of rows")
    if folds < 2:
        raise ValueError("cross fitting needs at least 2 folds")
    if model is None:
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
        except ImportError as e:   # pragma: no cover
            raise ImportError("cupac_adjust needs scikit-learn, or pass your own model") from e
        model = HistGradientBoostingRegressor(max_iter=100, random_state=seed)
    fold = np.random.default_rng(seed).permutation(len(y)) % folds
    pred = np.empty(len(y))
    for k in range(folds):
        m = copy.deepcopy(model).fit(X[fold != k], y[fold != k])
        pred[fold == k] = m.predict(X[fold == k])
    y_adj, theta = cuped_adjust(y, pred)
    return y_adj, theta, pred


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
    Sample ratio mismatch check. If you meant to split 50/50 and got 52/48
    on a big test, something in assignment, logging or data cleaning may be
    dropping users from one group, and that can bias every metric. Returns a
    chi square p value.

    A small p value is a warning sign worth chasing down, not proof that the
    experiment is broken: by chance alone 1 healthy test in 100 lands below
    0.01, which is why teams that run many tests often alert only below
    0.001. Count the unit that was randomized (users, or whole clusters),
    not rows. Counting sessions or users in clustered tests flags almost
    every test even when nothing is wrong.
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
