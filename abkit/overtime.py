"""
Does the effect change over time? Two things to rule out before
trusting a lift measured over a couple of weeks:

* Novelty (or primacy). Users click a new button because it's new, and
  the lift fades. Or they need time to learn it, and the lift grows. If
  the daily lift has a clear trend, the average over the test isn't what
  you'll get after launch.
* Day of week. Weekend users can respond differently. A test that ran
  for 10 days counts some weekdays twice, which tilts the average.

Both checks work on the daily lifts. They treat those as independent,
which holds when each day brings different users (new user tests) and is
only approximate when the same users come back every day.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .core import heterogeneity_test, mean_test


def trend_test(estimates, ses, x=None) -> tuple[float, float, float]:
    """
    Weighted least squares slope of the estimates on x (default 0, 1, 2, ...),
    with weights 1 / se^2. Returns (slope, se of slope, p value). A small p
    value means the effect is drifting.
    """
    d = np.asarray(estimates, dtype=float)
    w = 1 / np.asarray(ses, dtype=float) ** 2
    x = np.arange(len(d), dtype=float) if x is None else np.asarray(x, dtype=float)
    if len(d) < 3:
        raise ValueError("need at least three periods")
    xm = (w * x).sum() / w.sum()
    sxx = (w * (x - xm) ** 2).sum()
    slope = (w * (x - xm) * d).sum() / sxx
    se = np.sqrt(1 / sxx)
    return float(slope), float(se), float(2 * stats.norm.sf(abs(slope / se)))


@dataclass
class TimeChecks:
    daily: pd.DataFrame
    trend_slope: float          # change in lift per day
    trend_p: float
    weekday: pd.DataFrame | None
    weekday_p: float | None     # do weekdays differ? (Cochran's Q)

    def __str__(self) -> str:
        lines = [f"lift over {len(self.daily)} days: trend {self.trend_slope:+.4g} per day, p = {self.trend_p:.4f}"
                 + ("   the effect is drifting" if self.trend_p < 0.05 else "")]
        if self.weekday_p is not None:
            lines.append(f"day of week differences: p = {self.weekday_p:.4f}"
                         + ("   weekdays respond differently" if self.weekday_p < 0.05 else ""))
        return "\n".join(lines)


def time_checks(control, treatment, time_c, time_t, min_per_day: int = 30) -> TimeChecks:
    """
    Lift for each calendar day, a test for a trend across days, and a test
    for differences between days of the week (needs at least two weeks of data
    for the weekday split to mean much). time_c and time_t are timestamps
    (anything pandas can parse). Days with fewer than min_per_day users in
    either group are skipped.
    """
    def days(x):
        return pd.to_datetime(pd.Series(np.asarray(x))).dt.normalize().to_numpy()
    c = pd.DataFrame({"y": np.asarray(control, dtype=float), "day": days(time_c)})
    t = pd.DataFrame({"y": np.asarray(treatment, dtype=float), "day": days(time_t)})
    rows = []
    for day in sorted(set(c["day"]) | set(t["day"])):
        yc, yt = c.loc[c["day"] == day, "y"].to_numpy(), t.loc[t["day"] == day, "y"].to_numpy()
        if min(len(yc), len(yt)) < max(min_per_day, 2):
            continue
        r = mean_test(yc, yt)
        if r.se == 0:
            continue
        rows.append({"day": day, "n": len(yc) + len(yt), "lift": r.lift, "se": r.se,
                     "ci_low": r.ci[0], "ci_high": r.ci[1]})
    daily = pd.DataFrame(rows)
    if len(daily) < 3:
        raise ValueError("need at least three days with enough users")
    x = (daily["day"] - daily["day"].min()).dt.days.to_numpy(float)
    slope, _, p_trend = trend_test(daily["lift"], daily["se"], x)
    weekday, p_week = None, None
    wd = daily["day"].dt.day_of_week
    if wd.nunique() >= 2:
        # pool days by weekday with inverse variance weights, then ask if they differ
        w = 1 / daily["se"] ** 2
        g = pd.DataFrame({"weekday": wd, "wl": w * daily["lift"], "w": w}).groupby("weekday").sum()
        weekday = pd.DataFrame({"weekday": g.index, "lift": g["wl"] / g["w"], "se": 1 / np.sqrt(g["w"])})
        weekday = weekday.reset_index(drop=True)
        p_week = heterogeneity_test(weekday["lift"], weekday["se"])
    return TimeChecks(daily, slope, p_trend, weekday, p_week)
