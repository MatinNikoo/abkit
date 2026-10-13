"""
Bayesian readout: "how likely is it that B is better, and what do I lose
if I ship it and I'm wrong?"

Product managers usually want that instead of a p value. It answers a
different question, so abkit reports it next to the regular test rather
than instead of it.

For 0/1 metrics (conversion, retention) each group gets a Beta posterior
from a flat Beta(1, 1) prior. For averages, the posterior for each mean is
approximately normal (the central limit theorem does the work at these
sample sizes). The chance to win and the expected loss come from draws
from those posteriors.

One catch worth knowing. With a flat prior, the chance to win is honest
when real effects are big compared to the noise, but it's overconfident
when most real effects are tiny, which is the usual case in product tests.
The fix is a prior on the lift itself, ideally the spread of lifts seen in
past experiments (prior_sd). That pulls small, noisy wins back toward zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BayesResult:
    metric: str
    chance_treatment_better: float
    expected_loss_ship: float      # average shortfall if you ship treatment and it's worse
    expected_loss_keep: float      # average shortfall if you keep control and treatment was better
    lift_interval: tuple[float, float]   # 95% credible interval for treatment minus control

    def __str__(self) -> str:
        lo, hi = self.lift_interval
        return (f"{self.metric}: {self.chance_treatment_better:.1%} chance treatment is better; "
                f"95% credible interval {lo:+.4g} to {hi:+.4g}; "
                f"expected loss if you ship it {self.expected_loss_ship:.4g}")


def _is_binary(a: np.ndarray) -> bool:
    return bool(np.isin(a, (0, 1)).all())


def bayes_test(control, treatment, metric: str = "metric", draws: int = 200_000,
               seed: int = 0, prior_sd: float | None = None) -> BayesResult:
    """
    prior_sd: if given, a normal prior centered at zero on the lift
    (treatment minus control) with this standard deviation. Set it to how
    much lifts usually vary across your past experiments.
    """
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    rng = np.random.default_rng(seed)
    if prior_sd is not None:
        if prior_sd <= 0:
            raise ValueError("prior_sd must be positive")
        est = t.mean() - c.mean()
        se2 = c.var(ddof=1) / len(c) + t.var(ddof=1) / len(t)
        p2 = prior_sd ** 2
        d = rng.normal(est * p2 / (p2 + se2), np.sqrt(p2 * se2 / (p2 + se2)), draws)
        return BayesResult(metric, float((d > 0).mean()),
                           float(np.maximum(-d, 0).mean()), float(np.maximum(d, 0).mean()),
                           (float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))))
    if _is_binary(c) and _is_binary(t):
        pc = rng.beta(1 + c.sum(), 1 + len(c) - c.sum(), draws)
        pt = rng.beta(1 + t.sum(), 1 + len(t) - t.sum(), draws)
    else:
        pc = rng.normal(c.mean(), c.std(ddof=1) / np.sqrt(len(c)), draws)
        pt = rng.normal(t.mean(), t.std(ddof=1) / np.sqrt(len(t)), draws)
    d = pt - pc
    return BayesResult(metric, float((d > 0).mean()),
                       float(np.maximum(-d, 0).mean()), float(np.maximum(d, 0).mean()),
                       (float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))))
