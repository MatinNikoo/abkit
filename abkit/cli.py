"""
Command line interface.

    abkit report data.csv --variant version --control gate_30 --treatment gate_40 \
        --metrics retention_1 retention_7 [--covariate col ...] [--segment col]
    abkit size --baseline 0.19 --mde 0.01            # conversion rate
    abkit size --sd 12.5 --mde 0.5                   # an average
"""

from __future__ import annotations

import argparse
import sys
import warnings

import pandas as pd

from . import __version__
from .core import proportion_sample_size, sample_size
from .experiment import Experiment


def _report(a) -> int:
    df = pd.read_csv(a.csv)
    for col in [a.variant, *a.metrics, *(a.covariate or []), *([a.segment] if a.segment else [])]:
        if col not in df:
            print(f"error: no column called {col!r} in {a.csv}", file=sys.stderr)
            return 2
    for m in a.metrics:
        if df[m].dtype == bool:
            df[m] = df[m].astype(int)
    exp = Experiment(df, a.variant, a.control, a.treatment, a.split, a.alpha)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        h = exp.check(pre_period=a.covariate)
    print(f"{a.control}: {h['n_control']:,} users   {a.treatment}: {h['n_treatment']:,} users")
    print(f"sample ratio check: p = {h['srm_p']:.4f}" + ("   MISMATCH, investigate before trusting results" if caught else "   ok"))
    for col, p in (h.get("balance") or {}).items():
        print(f"balance on {col}: p = {p:.4f}" + ("   unbalanced" if p < 0.001 else "   ok"))
    print()

    for m in a.metrics:
        print(exp.analyze(m, a.covariate))
        b = exp.bayes(m, a.prior_sd)
        print(f"  {b.chance_treatment_better:.1%} chance {a.treatment} is better; "
              f"expected loss if you ship it {b.expected_loss_ship:.4g}")
        print()
    if len(a.metrics) > 1:
        t = exp.analyze_many(a.metrics, correction=a.correction)
        print(f"adjusted for testing {len(a.metrics)} metrics ({a.correction}):")
        for _, r in t.iterrows():
            print(f"  {r['metric']:<24} p = {r['p']:.4f}   adjusted p = {r[f'p_{a.correction}']:.4f}")
        print()
    if a.segment:
        for m in a.metrics:
            table, het = exp.segments(m, a.segment, a.correction, a.covariate)
            print(f"{m} by {a.segment} (do segments differ? p = {het:.4f})")
            for _, r in table.iterrows():
                print(f"  {str(r['segment']):<16} n = {int(r['n']):>9,}   lift {r['rel_lift']:+.2%}   "
                      f"adjusted p = {r[f'p_{a.correction}']:.4f}")
            print()
    return 0


def _size(a) -> int:
    if (a.baseline is None) == (a.sd is None):
        print("error: give either --baseline (a conversion rate) or --sd (for an average)", file=sys.stderr)
        return 2
    if a.baseline is not None:
        n, _ = proportion_sample_size(a.baseline, a.mde, a.alpha, a.power)
    else:
        n, _ = sample_size(a.sd, a.mde, a.alpha, a.power)
    print(f"{n:,} users per group ({2 * n:,} total) to detect a change of {a.mde} "
          f"with {a.power:.0%} power at alpha {a.alpha}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="abkit", description="Analyze and plan A/B tests.")
    p.add_argument("--version", action="version", version=f"abkit {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("report", help="analyze an experiment from a CSV with one row per user")
    r.add_argument("csv")
    r.add_argument("--variant", required=True, help="column with the group name")
    r.add_argument("--control", required=True)
    r.add_argument("--treatment", required=True)
    r.add_argument("--metrics", nargs="+", required=True)
    r.add_argument("--covariate", nargs="+", help="columns measured before the test, for CUPED")
    r.add_argument("--segment", help="column to break results down by")
    r.add_argument("--correction", choices=["bh", "holm", "none"], default="bh")
    r.add_argument("--split", type=float, default=0.5, help="intended share in treatment")
    r.add_argument("--alpha", type=float, default=0.05)
    r.add_argument("--prior-sd", type=float, default=None,
                   help="spread of lifts in past tests, for the Bayesian readout")
    r.set_defaults(func=_report)

    s = sub.add_parser("size", help="how many users a test needs")
    s.add_argument("--baseline", type=float, help="current conversion rate, like 0.19")
    s.add_argument("--sd", type=float, help="standard deviation, for an average metric")
    s.add_argument("--mde", type=float, required=True, help="smallest absolute change you care about")
    s.add_argument("--power", type=float, default=0.8)
    s.add_argument("--alpha", type=float, default=0.05)
    s.set_defaults(func=_size)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
