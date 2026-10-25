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

from .data import as_frame, need_columns, need_groups
from .core import (TestResult, cluster_test, cupac_adjust, cuped_adjust, heterogeneity_test, mean_test,
                   noninferiority_test, ratio_test, srm_test)
from .multiple import benjamini_hochberg, dunnett, holm
from .bayes import BayesResult, bayes_test
from .sequential import GroupSequentialResult, SequentialResult, group_sequential_test, msprt


class Experiment:
    """
    One experiment in a table: a column naming each row's group, plus
    metric columns. The table can be a pandas or polars DataFrame, a dict
    of columns or a path to a CSV, Parquet, Excel or JSON file. Usually one
    row per user; with several rows per user (or per city), pass cluster=
    to the methods that take it.

    expected_share is the share you meant to put in treatment, used by the
    split check. alpha applies to every test and interval.
    """

    def __init__(self, df, variant: str, control, treatment,
                 expected_share: float = 0.5, alpha: float = 0.05):
        df = as_frame(df)
        need_groups(df, variant, control, treatment)
        if not 0 < expected_share < 1 or not 0 < alpha < 1:
            raise ValueError("expected_share and alpha must be between 0 and 1")
        self.df = df
        self.variant = variant
        self.control = control
        self.treatment = treatment
        self.expected_share = expected_share
        self.alpha = alpha
        self._c = df[df[variant] == control]
        self._t = df[df[variant] == treatment]

    def _need(self, *cols, numeric: bool = True) -> None:
        flat = [c for col in cols if col is not None for c in ([col] if isinstance(col, str) else col)]
        need_columns(self.df, flat, numeric=numeric)

    # ------------------------------------------------------------------ checks
    def srm(self, unit: str | None = None) -> float:
        """
        Sample ratio p value. If whole groups of rows were randomized
        together (sessions of one user, users in one city), pass the column
        that was randomized as unit, so the check counts units, not rows.
        Counting rows flags almost every clustered test even when nothing is wrong.
        """
        self._need(unit, numeric=False)
        if unit is None:
            return srm_test(len(self._c), len(self._t), self.expected_share)
        return srm_test(self._c[unit].nunique(), self._t[unit].nunique(), self.expected_share)

    def missing(self, metric: str) -> dict:
        """Share of missing values in each group, and a p value for "one group loses more"."""
        self._need(metric)
        mc = self._c[metric].isna().to_numpy(float)
        mt = self._t[metric].isna().to_numpy(float)
        p = mean_test(mc, mt).p_value if (mc.any() or mt.any()) else 1.0
        return {"missing_control": float(mc.mean()), "missing_treatment": float(mt.mean()), "p": float(p)}

    def check(self, pre_period: list[str] | None = None, srm_threshold: float = 0.01,
              unit: str | None = None, metrics: list[str] | None = None) -> dict:
        """
        Health checks before trusting any result.

        Sample ratio mismatch: did the split come out like it should? A
        small p value is a warning sign worth chasing down (logging,
        bots, redirects, users dropped from one group). It doesn't tell you
        which, and the experiment may be fine. By chance alone, 1 test in 100 that's fine
        lands below 0.01.
        Balance: do columns measured before the test look the same in both
        groups? If they don't, randomization or logging may be off.
        Missing values: does one group lose more of a metric than the other?
        """
        self._need(pre_period, metrics)
        out = {"n_control": len(self._c), "n_treatment": len(self._t), "srm_p": self.srm(unit)}
        if out["srm_p"] < srm_threshold:
            warnings.warn(
                f"sample ratio mismatch (p = {out['srm_p']:.2g}). The split isn't what it "
                "should be, so treat every result with suspicion until you find out why.",
                stacklevel=2,
            )
        if pre_period:
            out["balance"] = {col: mean_test(self._c[col], self._t[col], metric=col).p_value
                              for col in pre_period}
        if metrics:
            out["missing"] = {m: self.missing(m) for m in metrics}
            for m, r in out["missing"].items():
                if r["p"] < srm_threshold:
                    warnings.warn(f"{m} is missing more often in one group ({r['missing_control']:.1%} vs "
                                  f"{r['missing_treatment']:.1%}, p = {r['p']:.2g}); a complete case "
                                  "comparison can be biased", stacklevel=2)
        return out

    # ---------------------------------------------------------------- analysis
    def analyze(self, metric: str, covariate: str | list[str] | None = None, cluster: str | None = None,
                dropna: bool = False) -> TestResult:
        """
        Test one metric. With covariates measured before the experiment,
        uses CUPED to cut the variance. Pass one column name or a list.
        The adjustment is fit on both groups pooled so it can't leak the
        treatment effect.

        cluster: the column that was randomized when rows aren't independent
        (user id for session rows, city for city level tests). Then the
        test runs at that level with cluster_test.
        dropna: drop rows where the metric or a covariate is missing,
        instead of raising. The result notes how much was dropped and
        whether one group lost more.
        """
        covs = [] if covariate is None else ([covariate] if isinstance(covariate, str) else list(covariate))
        self._need(metric, covs)
        self._need(cluster, numeric=False)
        c, t = self._c, self._t
        notes = []
        if dropna:
            miss = self.missing(metric)
            cols = [metric, *covs]
            c, t = c.dropna(subset=cols), t.dropna(subset=cols)
            if len(c) < len(self._c) or len(t) < len(self._t):
                notes.append(f"dropped {len(self._c) - len(c):,} control and {len(self._t) - len(t):,} "
                             f"treatment rows with missing values")
                if miss["p"] < 0.01:
                    notes.append(f"{metric} goes missing more in one group ({miss['missing_control']:.1%} vs "
                                 f"{miss['missing_treatment']:.1%}, p = {miss['p']:.2g}), so this comparison "
                                 "can be biased")
        yc, yt = c[metric].to_numpy(float), t[metric].to_numpy(float)

        def test(a, b):
            if cluster is None:
                return mean_test(a, b, self.alpha, metric)
            return cluster_test(a, b, c[cluster].to_numpy(), t[cluster].to_numpy(), self.alpha, metric)

        if not covs:
            res = test(yc, yt)
            res.notes = notes + res.notes
            return res

        for col in covs:
            bal = mean_test(c[col].to_numpy(float), t[col].to_numpy(float)).p_value
            if bal < 0.001:
                notes.append(f"{col} differs between groups (p = {bal:.2g}); "
                             "it may be affected by the treatment, so CUPED could be biased")
        y = np.concatenate([yc, yt])
        X = np.vstack([c[covs].to_numpy(float), t[covs].to_numpy(float)])
        y_adj, _ = cuped_adjust(y, X)
        raw = test(yc, yt)
        res = test(y_adj[: len(yc)], y_adj[len(yc):])
        # report the real group means, with the adjusted lift and interval
        res.control_mean, res.treatment_mean = raw.control_mean, raw.treatment_mean
        mc = raw.control_mean
        if mc:
            res.rel_lift = res.lift / mc
            res.rel_ci = (res.ci[0] / mc, res.ci[1] / mc)
        cut = 1 - (res.se / raw.se) ** 2 if raw.se > 0 else 0.0
        res.notes = notes + [f"CUPED on {', '.join(covs)} cut variance by {cut:.0%}"] + \
            [n for n in res.notes if "skew" in n or "cluster" in n]
        return res

    def readout(self, metrics: list[str], fmt: str = "markdown", **kw) -> str:
        """The full readout as markdown, HTML or JSON (see abkit.report.readout for the options)."""
        from .report import readout
        return readout(self, metrics, fmt=fmt, **kw)

    def analyze_cupac(self, metric: str, features: list[str], model=None, folds: int = 2,
                      seed: int = 0) -> TestResult:
        """
        CUPED with a model's prediction as the covariate (see cupac_adjust).
        features are columns measured before the test. Cross fitted, so the
        prediction for a user never comes from a model that saw their outcome.
        """
        self._need(metric, features)
        yc, yt = self._c[metric].to_numpy(float), self._t[metric].to_numpy(float)
        X = np.vstack([self._c[features].to_numpy(float), self._t[features].to_numpy(float)])
        y_adj, _, _ = cupac_adjust(np.concatenate([yc, yt]), X, model, folds, seed)
        raw = mean_test(yc, yt, self.alpha, metric)
        res = mean_test(y_adj[: len(yc)], y_adj[len(yc):], self.alpha, metric)
        res.control_mean, res.treatment_mean = raw.control_mean, raw.treatment_mean
        if raw.control_mean:
            res.rel_lift = res.lift / raw.control_mean
            res.rel_ci = (res.ci[0] / raw.control_mean, res.ci[1] / raw.control_mean)
        cut = 1 - (res.se / raw.se) ** 2 if raw.se > 0 else 0.0
        res.notes = [f"CUPAC on {len(features)} features, {folds} fold cross fit, cut variance by {cut:.0%}"]
        return res

    def analyze_skewed(self, metric: str, quantiles=(0.5, 0.9), cap: float = 0.99,
                       n_boot: int = 2000) -> pd.DataFrame:
        """
        Several views of a skewed metric side by side: the plain mean test,
        a bootstrap interval for the mean, the mean after capping at the
        `cap` quantile (less noise, biased), quantile effects, and the
        Mann-Whitney chance that a treatment user beats a control user.
        If they disagree, the effect is coming from the tail.
        """
        from .robust import bootstrap_test, mann_whitney, quantile_test, winsorize

        self._need(metric)
        c, t = self._c[metric].to_numpy(float), self._t[metric].to_numpy(float)
        rows = []
        def add(name, r, question):
            rows.append({"view": name, "question": question, "control": r.control_mean,
                         "treatment": r.treatment_mean, "lift": r.lift, "ci_low": r.ci[0],
                         "ci_high": r.ci[1], "p": r.p_value})
        add("mean (t test)", mean_test(c, t, self.alpha, metric), "did the mean move")
        add("mean (bootstrap)", bootstrap_test(c, t, "mean", n_boot, self.alpha), "did the mean move")
        cc, ct, (_, hi) = winsorize(c, t, cap)
        add(f"mean capped at {hi:.4g}", mean_test(cc, ct, self.alpha), "did the capped mean move (biased)")
        for q in quantiles:
            add(f"quantile {q:g}", quantile_test(c, t, q, n_boot, self.alpha), f"did the {q:g} quantile move")
        mw = mann_whitney(c, t, self.alpha, metric)
        rows.append({"view": "Mann-Whitney", "question": "does a random treatment user beat a control user",
                     "control": 0.5, "treatment": mw.prob_treatment_higher,
                     "lift": mw.prob_treatment_higher - 0.5, "ci_low": mw.ci[0] - 0.5,
                     "ci_high": mw.ci[1] - 0.5, "p": mw.p_value})
        return pd.DataFrame(rows)

    def guardrails(self, margins: dict, higher_is_better: dict | None = None,
                   relative: bool = True) -> pd.DataFrame:
        """
        Non-inferiority check for each guardrail metric. margins maps a
        metric to the worst change you'd accept (relative by default, so
        0.01 means 1% worse), or to the name of a preset such as "latency"
        (see abkit.presets). Returns one row per metric with a pass column.
        """
        from .presets import resolve_guardrail

        higher_is_better = higher_is_better or {}
        rows = []
        for m, margin in margins.items():
            self._need(m)
            margin, preset_hib = resolve_guardrail(margin)
            hib = higher_is_better.get(m, True if preset_hib is None else preset_hib)
            r = noninferiority_test(self._c[m].to_numpy(float), self._t[m].to_numpy(float), margin, hib,
                                    relative, self.alpha, m)
            rows.append({"metric": m, "control": r.control_mean, "treatment": r.treatment_mean,
                         "rel_lift": r.rel_lift, "margin": margin, "higher_is_better": hib,
                         "p": r.p_value, "passes": r.p_value < self.alpha})
        return pd.DataFrame(rows)

    def compare_arms(self, metric: str, arms: list | None = None) -> pd.DataFrame:
        """
        For tests with more than one treatment: every arm against control,
        with Dunnett adjusted p values and simultaneous intervals. arms
        defaults to every value in the variant column except control.
        """
        self._need(metric)
        arms = arms if arms is not None else [a for a in sorted(self.df[self.variant].unique(), key=str)
                                              if a != self.control]
        ts = [self.df.loc[self.df[self.variant] == a, metric].to_numpy(float) for a in arms]
        return dunnett(self._c[metric].to_numpy(float), ts, self.alpha, names=arms)

    def over_time(self, metric: str, time: str, min_per_day: int = 30):
        """Daily lifts, a novelty trend test and a day of week test (see overtime.time_checks)."""
        from .overtime import time_checks
        self._need(metric)
        self._need(time, numeric=False)
        return time_checks(self._c[metric], self._t[metric], self._c[time], self._t[time], min_per_day)

    def screen(self, metric: str, features: list[str], bins: int = 4, correction: str = "bh") -> pd.DataFrame:
        """
        Which user traits change the effect? For each feature, split users
        into its categories (numbers are cut into `bins` equal sized groups)
        and run the heterogeneity test. The p values are then adjusted for
        screening several features. Treat a hit as a lead to confirm in a
        new test.
        """
        self._need(metric)
        self._need(features, numeric=False)
        rows = []
        for f in features:
            col = self.df[f]
            if pd.api.types.is_numeric_dtype(col) and col.nunique() > bins:
                key = pd.qcut(col.rank(method="first"), bins, labels=False)
            else:
                key = col
            sub = Experiment(self.df.assign(_seg=key), self.variant, self.control, self.treatment,
                             self.expected_share, self.alpha)
            table, het = sub.segments(metric, "_seg", correction="none")
            spread = table["lift"].max() - table["lift"].min()
            rows.append({"feature": f, "groups": len(table), "lift_spread": spread, "p": het})
        out = pd.DataFrame(rows)
        adj = {"bh": benjamini_hochberg, "holm": holm, "none": lambda x: x}[correction]
        out[f"p_{correction}"] = adj(out["p"].to_numpy())
        return out.sort_values("p").reset_index(drop=True)

    def aa(self, metric: str, sims: int = 1000, cluster: str | None = None, seed: int = 0):
        """
        A/A simulation on the control group: split it at random many times
        and check the test calls a difference about 5% of the time.
        """
        from .design import aa_test
        self._need(metric)
        self._need(cluster, numeric=False)
        y = self._c[metric].to_numpy(float)
        g = self._c[cluster].to_numpy() if cluster else None
        return aa_test(y, sims, 0.5, g, seed=seed)

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
        self._need(metric, covariate)
        self._need(by, numeric=False)
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
        self._need(metric)
        return bayes_test(self._c[metric].to_numpy(float), self._t[metric].to_numpy(float), metric,
                          prior_sd=prior_sd)

    def analyze_ratio(self, numerator: str, denominator: str, name: str | None = None) -> TestResult:
        """A ratio metric like clicks per session, one row per user, by the delta method (see ratio_test)."""
        self._need(numerator, denominator)
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
        self._need(metric)
        self._need(order, numeric=False)
        c, t = self._c, self._t
        if order:
            c, t = c.sort_values(order), t.sort_values(order)
        return msprt(c[metric].to_numpy(float), t[metric].to_numpy(float),
                     tau=abs(mde), alpha=self.alpha, n_looks=n_looks)

    def group_sequential(self, metric: str, n_looks: int = 5, kind: str = "obrien_fleming",
                         order: str | None = None) -> GroupSequentialResult:
        """
        Replay a group sequential test with n_looks planned looks (see
        sequential.group_sequential_test). The data you have is taken as
        the planned maximum sample.
        """
        self._need(metric)
        self._need(order, numeric=False)
        c, t = self._c, self._t
        if order:
            c, t = c.sort_values(order), t.sort_values(order)
        return group_sequential_test(c[metric].to_numpy(float), t[metric].to_numpy(float),
                                     n_looks, self.alpha, kind)
