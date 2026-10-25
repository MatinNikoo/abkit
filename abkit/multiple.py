"""
Corrections for testing many metrics at once.

Check 20 metrics at alpha 0.05 and you expect one false "win" even when
nothing changed. Holm controls the chance of any false positive.
Benjamini-Hochberg controls the share of false positives among the wins,
which is usually the better trade for a dashboard of metrics.
"""

from __future__ import annotations

import numpy as np


def holm(p_values) -> np.ndarray:
    """Holm adjusted p values (family wise error rate)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def benjamini_hochberg(p_values) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values (false discovery rate)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    running = 1.0
    for k, i in enumerate(order):
        rank = m - k
        running = min(running, p[i] * m / rank)
        adj[i] = running
    return adj


def dunnett(control, treatments, alpha: float = 0.05, names=None):
    """
    Compare several treatments to one control (an A/B/C/D test), with
    p values and intervals adjusted for the number of comparisons.

    Every comparison shares the same control group, so the z scores are
    correlated. Dunnett's method uses that correlation and is less strict
    than Bonferroni while keeping the chance of any false win at alpha.
    This version allows each group its own variance (Welch style) and uses
    the normal approximation, which is fine at online test sizes. The
    intervals hold for all comparisons at once.

    Returns a DataFrame with one row per treatment.
    """
    import pandas as pd
    from scipy import optimize, stats

    c = np.asarray(control, dtype=float)
    ts = [np.asarray(t, dtype=float) for t in treatments]
    k = len(ts)
    if k < 1:
        raise ValueError("need at least one treatment")
    names = list(names) if names is not None else [f"treatment_{i + 1}" for i in range(k)]
    vc = c.var(ddof=1) / len(c)
    vt = np.array([t.var(ddof=1) / len(t) for t in ts])
    diff = np.array([t.mean() - c.mean() for t in ts])
    se = np.sqrt(vc + vt)
    z = diff / se
    # corr(z_i, z_j) comes only from the shared control mean
    corr = vc / np.sqrt(np.outer(vc + vt, vc + vt))
    np.fill_diagonal(corr, 1.0)

    def inside(x):
        if k == 1:
            return 1 - 2 * stats.norm.sf(x)
        mvn = stats.multivariate_normal(np.zeros(k), corr, allow_singular=True, seed=0)
        return mvn.cdf(np.full(k, x), lower_limit=np.full(k, -x))

    p_adj = np.array([min(1.0, max(0.0, 1 - inside(abs(zi)))) for zi in z])
    crit = optimize.brentq(lambda x: inside(x) - (1 - alpha), 0.5, 10, xtol=1e-6)
    return pd.DataFrame({
        "arm": names, "control_mean": c.mean(), "treatment_mean": [t.mean() for t in ts],
        "lift": diff, "ci_low": diff - crit * se, "ci_high": diff + crit * se,
        "p": 2 * stats.norm.sf(np.abs(z)), "p_dunnett": p_adj,
    })
