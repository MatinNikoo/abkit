"""
Reproduces the rare event problem in the mSPRT. On a 0.1% conversion rate the
first looks often have no conversions at all in a group, the plug in variance is
0, and the confidence sequence collapses to a point. The sequence is intersected
over time, so every later look inherits the collapse.

msprt_old below is the mSPRT before the fix (it keeps looks where a group has no
spread yet). abkit.msprt now skips those looks.

Run:  python simulations/msprt_rare_events.py     (under a minute)
Writes results/msprt_rare_events.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import abkit as ab  # noqa: E402
from abkit.sequential import SequentialResult, _running_stats  # noqa: E402

SIMS, N, LOOKS = 600, 20000, 50
P_C, P_T = 0.001, 0.00189


def msprt_old(control, treatment, tau, alpha=0.05, n_looks=50, min_per_arm=100):
    c, t = np.asarray(control, float), np.asarray(treatment, float)
    looks = np.linspace(0, 1, n_looks + 1)[1:]
    ic = np.maximum((looks * len(c)).astype(int), 2)
    it = np.maximum((looks * len(t)).astype(int), 2)
    keep = (ic >= min_per_arm) & (it >= min_per_arm)
    ic, it = ic[keep], it[keep]
    mc, vc = _running_stats(c, ic)
    mt, vt = _running_stats(t, it)
    d = mt - mc
    V = np.maximum(vc / ic + vt / it, 1e-300)
    t2 = tau * tau
    log_L = 0.5 * np.log(V / (V + t2)) + t2 * d * d / (2 * V * (V + t2))
    p = np.minimum.accumulate(np.minimum(1.0, np.exp(-log_L)))
    half = np.sqrt(V * (V + t2) / t2 * (2 * np.log(1 / alpha) + np.log((V + t2) / V)))
    lo, hi = np.maximum.accumulate(d - half), np.minimum.accumulate(d + half)
    return SequentialResult(ic, it, d, lo, hi, p, alpha)


def rate(k, n):
    return {"rate": k / n, "mc_se": max(k / n * (1 - k / n), 1 / n) ** 0.5 / n ** 0.5}


def main():
    rng = np.random.default_rng(2026)
    true = P_T - P_C
    miss = {"before_fix": 0, "after_fix": 0}
    empty = {"before_fix": 0, "after_fix": 0}
    for _ in range(SIMS):
        c = rng.binomial(1, P_C, N).astype(float)
        t = rng.binomial(1, P_T, N).astype(float)
        for name, fn in (("before_fix", msprt_old), ("after_fix", ab.msprt)):
            r = fn(c, t, tau=true, n_looks=LOOKS)
            miss[name] += bool(((r.ci_low > true) | (r.ci_high < true)).any())
            empty[name] += bool((r.ci_low > r.ci_high).any())
    out = {"sims": SIMS, "users_per_group": N, "control_rate": P_C, "treatment_rate": P_T, "looks": LOOKS,
           "interval_misses_truth_at_some_look": {k: rate(v, SIMS) for k, v in miss.items()},
           "interval_empty_at_some_look": {k: rate(v, SIMS) for k, v in empty.items()}}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "msprt_rare_events.json").write_text(json.dumps(out, indent=2))
    for k in miss:
        print(f"{k}: misses the truth {miss[k] / SIMS:.1%}, empty interval {empty[k] / SIMS:.1%}")


if __name__ == "__main__":
    main()
