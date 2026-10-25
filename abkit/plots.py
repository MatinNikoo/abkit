"""
Small charts for effects and intervals. Each function returns the
matplotlib Figure, so `fig.savefig("effects.png", dpi=150)` is all it
takes to keep one. Needs matplotlib (pip install "abkit[plots]").
"""

from __future__ import annotations

import numpy as np

BLUE, GREY, RED = "#2F4BC4", "#8B93A3", "#B8322D"


def _pyplot():
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError('plots need matplotlib: pip install "abkit[plots]"') from e
    return matplotlib, plt


def _finish(ax, title=None, xlabel=None):
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.figure.tight_layout()
    return ax.figure


def _rows(results):
    """Name, estimate, low and high for TestResults, an analyze_many table, or readout rows."""
    out = []
    if hasattr(results, "to_dict") and hasattr(results, "columns"):
        results = results.to_dict("records")
    elif not isinstance(results, (list, tuple)):
        results = [results]
    for r in results:
        if hasattr(r, "rel_ci"):
            out.append((r.metric, r.rel_lift, *r.rel_ci))
        elif "rel_ci" in r:
            out.append((r["metric"], r["rel_lift"], *r["rel_ci"]))
        else:
            out.append((r["metric"], r["rel_lift"], r["ci_low"], r["ci_high"]))
    return out


def forest(labels, estimates, low, high, significant=None, title=None, xlabel="change vs control",
           percent=True, ax=None):
    """Points with intervals, one row per label, zero marked. Blue when the interval excludes zero."""
    _, plt = _pyplot()
    n = len(labels)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.55 * n + 1.3))
    est, lo, hi = (np.asarray(x, dtype=float) for x in (estimates, low, high))
    sig = (lo > 0) | (hi < 0) if significant is None else np.asarray(significant, dtype=bool)
    y = np.arange(n)[::-1]
    for i in range(n):
        color = BLUE if sig[i] else GREY
        ax.plot([lo[i], hi[i]], [y[i], y[i]], color=color, lw=2, solid_capstyle="round")
        ax.plot(est[i], y[i], "o", color=color, ms=6)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y, [str(x) for x in labels])
    ax.set_ylim(-0.7, n - 0.3)
    if percent:
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.1%}" if v else "0")
    return _finish(ax, title, xlabel)


def effects(results, title=None, ax=None):
    """Relative change and its interval for each metric. Takes TestResults, an analyze_many table or readout rows."""
    rows = _rows(results)
    return forest([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
                  title=title, ax=ax)


def segments(table, title=None, ax=None):
    """Lift by segment from Experiment.segments (absolute change, interval from the table)."""
    return forest(table["segment"].astype(str), table["lift"], table["ci_low"], table["ci_high"],
                  title=title, xlabel="change vs control", percent=False, ax=ax)


def confidence_sequence(result, title=None, ax=None):
    """
    The estimate and its interval at each look, from Experiment.sequential
    or Experiment.group_sequential. A band that clears zero is a stop.
    """
    _, plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.6))
    n = result.n_control + result.n_treatment
    ax.fill_between(n, result.ci_low, result.ci_high, color=BLUE, alpha=0.18, lw=0, label="interval")
    ax.plot(n, result.estimate, color=BLUE, lw=1.6, label="estimate")
    ax.axhline(0, color="k", lw=0.8)
    i = result.stop_index
    if i is not None:
        ax.axvline(n[i], color=RED, lw=1, ls="--")
        ax.annotate(f"stopped at {n[i]:,} users", (n[i], ax.get_ylim()[1]), xytext=(-6, -14),
                    textcoords="offset points", ha="right", fontsize=8, color=RED)
    ax.set_ylabel("treatment minus control")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _finish(ax, title, "users so far")


def daily_lifts(checks, title=None, ax=None):
    """Lift per day with its interval and the fitted trend, from Experiment.over_time."""
    _, plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.6))
    d = checks.daily
    ax.errorbar(d["day"], d["lift"], yerr=[d["lift"] - d["ci_low"], d["ci_high"] - d["lift"]],
                fmt="o", color=BLUE, ms=4, lw=1.2, capsize=0)
    x = (d["day"] - d["day"].min()).dt.days.to_numpy(float)
    w = 1 / d["se"].to_numpy() ** 2
    xm, ym = (w * x).sum() / w.sum(), (w * d["lift"]).sum() / w.sum()
    ax.plot(d["day"], ym + checks.trend_slope * (x - xm), color=RED, lw=1.2,
            label=f"trend {checks.trend_slope:+.3g} per day, p = {checks.trend_p:.3f}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("treatment minus control")
    ax.legend(frameon=False, fontsize=8)
    ax.figure.autofmt_xdate()
    return _finish(ax, title, None)
