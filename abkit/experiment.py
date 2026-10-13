"""
A small front end so a whole analysis is a few lines:

    exp = Experiment(df, variant="group", control="A", treatment="B")
    exp.check()                                   # sample ratio, balance
    exp.analyze("revenue", covariate="revenue_before")
    exp.analyze_many(["clicks", "orders", "revenue"])
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .core import TestResult, cuped_adjust, heterogeneity_test, mean_test, ratio_test, srm_test
from .multiple import benjamini_hochberg, holm
from .bayes import BayesResult, bayes_test
from .sequential import SequentialResult, msprt


class Experiment:
    def __init__(self, df: pd.DataFrame, variant: str, control, treatment,
                 expected_share: float = 0.5, alpha: float = 0.05):
        if variant not in df:
            raise KeyError(f"no column called {variant!r}")
        groups = set(df[variant].unique())
        missing = {control, treatment} - groups
        if missing:
            raise ValueError(f"variant column has no rows for {missing}")
        self.df = df
        self.variant = variant
        self.control = control
        self.treatment = treatment
        self.expected_share = expected_share
        self.alpha = alpha
        self._c = df[df[variant] == control]
        self._t = df[df[variant] == treatment]

    # ------------------------------------------------------------------ checks
    def srm(self) -> float:
        return srm_test(len(self._c), len(self._t), self.expected_share)

    def check(self, pre_period: list[str] | None = None, srm_threshold: float = 0.01) -> dict:
        """
        Health checks before trusting any result.

        Sample ratio mismatch: did the split come out like it should?
        Balance: do columns measured before the test look the same in both
        groups? If they don't, randomization or logging is broken.
        """
        out = {"n_control": len(self._c), "n_treatment": len(self._t), "srm_p": self.srm()}
        if out["srm_p"] < srm_threshold:
            warnings.warn(
                f"sample ratio mismatch (p = {out['srm_p']:.2g}). The split isn't what it "
                "should be, so treat every result with suspicion until you find out why.",
                stacklevel=2,
            )
        if pre_period:
            out["balance"] = {col: mean_test(self._c[col], self._t[col], metric=col).p_value
                              for col in pre_period}
        return out

    # ---------------------------------------------------------------- analysis
    def analyze(self, metric: str, covariate: str | list[str] | None = None) -> TestResult:
        """
        Test one metric. With covariates measured before the experiment,
        uses CUPED to cut the variance. Pass one column name or a list.
        The adjustment is fit on both groups pooled so it can't leak the
        treatment effect.
        """
        yc, yt = self._c[metric].to_numpy(float), self._t[metric].to_numpy(float)
        if covariate is None:
            return mean_test(yc, yt, self.alpha, metric)

        covs = [covariate] if isinstance(covariate, str) else list(covariate)
        notes = []
        for col in covs:
            bal = mean_test(self._c[col].to_numpy(float), self._t[col].to_numpy(float)).p_value
            if bal < 0.001:
                notes.append(f"{col} differs between groups (p = {bal:.2g}); "
                             "it may be affected by the treatment, so CUPED could be biased")
        y = np.concatenate([yc, yt])
        X = np.vstack([self._c[covs].to_numpy(float), self._t[covs].to_numpy(float)])
        y_adj, _ = cuped_adjust(y, X)
        raw = mean_test(yc, yt, self.alpha, metric)
        res = mean_test(y_adj[: len(yc)], y_adj[len(yc):], self.alpha, metric)
        # report the real group means, with the adjusted lift and interval
        res.control_mean, res.treatment_mean = raw.control_mean, raw.treatment_mean
        mc = raw.control_mean
        if mc:
            res.rel_lift = res.lift / mc
            res.rel_ci = (res.ci[0] / mc, res.ci[1] / mc)
        cut = 1 - (res.se / raw.se) ** 2 if raw.se > 0 else 0.0
        res.notes = notes + [f"CUPED on {', '.join(covs)} cut variance by {cut:.0%}"]
        return res

    def segments(self, metric: str, by: str, correction: str = "bh",
                 covariate: str | list[str] | None = None) -> tuple[pd.DataFrame, float]:
        """
        The effect inside each segment (new vs returning users, country,
        platform), with p values adjusted for looking at several segments,
        plus one test of whether the segments really differ.

        Returns (table, heterogeneity p value). Slicing into segments is
        where most fake findings come from, so read the heterogeneity test
        before believing any single segment.
        """
        rows, ests, ses = [], [], []
        for seg, part in self.df.groupby(by, sort=True):
            sub = Experiment(part, self.variant, self.control, self.treatment,
                             self.expected_share, self.alpha)
            r = sub.analyze(metric, covariate)
            rows.append({"segment": seg, "n": r.n_control + r.n_treatment, "lift": r.lift,
                         "rel_lift": r.rel_lift, "ci_low": r.ci[0], "ci_high": r.ci[1], "p": r.p_value})
            ests.append(r.lift); ses.append(r.se)
        table = pd.DataFrame(rows)
        adj = {"bh": benjamini_hochberg, "holm": holm, "none": lambda x: x}[correction]
        table[f"p_{correction}"] = adj(table["p"].to_numpy())
        return table, heterogeneity_test(ests, ses)

    def bayes(self, metric: str, prior_sd: float | None = None) -> BayesResult:
        """Chance the treatment is better, and the expected loss of shipping it."""
        return bayes_test(self._c[metric].to_numpy(float), self._t[metric].to_numpy(float), metric,
                          prior_sd=prior_sd)

    def analyze_ratio(self, numerator: str, denominator: str, name: str | None = None) -> TestResult:
        return ratio_test(self._c[numerator], self._c[denominator],
                          self._t[numerator], self._t[denominator],
                          self.alpha, name or f"{numerator} per {denominator}")

    def analyze_many(self, metrics: list[str], correction: str = "bh") -> pd.DataFrame:
        """Test several metrics and adjust the p values for testing them together."""
        results = [self.analyze(m) for m in metrics]
        p = np.array([r.p_value for r in results])
        adj = {"bh": benjamini_hochberg, "holm": holm, "none": lambda x: x}[correction](p)
        return pd.DataFrame({
            "metric": metrics,
            "control": [r.control_mean for r in results],
            "treatment": [r.treatment_mean for r in results],
            "rel_lift": [r.rel_lift for r in results],
            "ci_low": [r.rel_ci[0] for r in results],
            "ci_high": [r.rel_ci[1] for r in results],
            "p": p,
            f"p_{correction}": adj,
        })

    def sequential(self, metric: str, mde: float, order: str | None = None, n_looks: int = 50) -> SequentialResult:
        """
        Replay the test as if you checked it n_looks times while it ran.
        `order` is a column giving arrival order (a timestamp, say).
        """
        c, t = self._c, self._t
        if order:
            c, t = c.sort_values(order), t.sort_values(order)
        return msprt(c[metric].to_numpy(float), t[metric].to_numpy(float),
                     tau=abs(mde), alpha=self.alpha, n_looks=n_looks)
