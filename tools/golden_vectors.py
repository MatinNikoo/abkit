"""
Reference inputs and outputs for checking another implementation of the
same math (the Experiment Lab on my site runs it in JavaScript) against
this library. Each case stores its exact inputs and what abkit returns,
including edge cases: tiny groups, uneven groups, rare conversions, a
group with no spread, huge means, negative values.

Run:  python tools/golden_vectors.py      writes results/golden_vectors.json
tests/test_golden.py recomputes every case, so the file can't drift from the code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import abkit as ab  # noqa: E402

rng = np.random.default_rng(7)


def r6(a):
    # round inputs so the stored numbers are exactly what was used
    return [float(f"{v:.6g}") for v in np.asarray(a, dtype=float)]


def tr(res):
    return {"lift": res.lift, "se": res.se, "ci": list(res.ci), "p_value": res.p_value,
            "rel_lift": res.rel_lift, "rel_ci": list(res.rel_ci), "control_mean": res.control_mean,
            "treatment_mean": res.treatment_mean}


def sample(kind, n):
    if kind == "normal":
        return rng.normal(rng.uniform(-5, 50), rng.uniform(0.5, 10), n)
    if kind == "lognormal":
        return rng.lognormal(0, rng.uniform(0.5, 2), n)
    if kind == "binary":
        return (rng.random(n) < rng.choice([0.002, 0.01, 0.05, 0.2, 0.5])).astype(float)
    if kind == "counts":
        return rng.poisson(rng.uniform(0.5, 6), n).astype(float)
    if kind == "huge":
        return 1e8 + rng.normal(0, 1, n)
    if kind == "negative":
        return -rng.exponential(3, n)
    raise ValueError(kind)


def build():
    out = {"abkit_version": ab.__version__, "note": "inputs are exact; outputs are float64 from abkit"}
    cases = []
    kinds = ["normal", "lognormal", "binary", "counts", "huge", "negative"]
    for i in range(72):
        kind = kinds[i % len(kinds)]
        nc, nt = int(rng.integers(2, 200)), int(rng.integers(2, 200))
        if i % 9 == 0:
            nt = nc
        c, t = r6(sample(kind, nc)), r6(sample(kind, nt))
        if i == 5:
            c = [1.0] * nc                       # a group with no spread
        alpha = [0.05, 0.1, 0.01][i % 3]
        res = ab.mean_test(c, t, alpha=alpha)
        cases.append({"kind": kind, "alpha": alpha, "control": c, "treatment": t, "expected": tr(res)})
    out["mean_test"] = cases

    out["srm_test"] = []
    for _ in range(30):
        n = int(rng.integers(100, 2_000_000))
        share = float(rng.choice([0.5, 0.5, 0.1, 0.2, 0.9]))
        nt = int(rng.binomial(n, share * rng.uniform(0.98, 1.02)))
        out["srm_test"].append({"n_control": n - nt, "n_treatment": nt, "expected_share": share,
                                "expected": ab.srm_test(n - nt, nt, share)})

    out["ratio_test"] = []
    for _ in range(20):
        nc, nt = int(rng.integers(5, 150)), int(rng.integers(5, 150))
        dc, dt = r6(rng.poisson(4, nc) + 1), r6(rng.poisson(4, nt) + 1)
        xc = r6(rng.binomial(np.array(dc, int), rng.uniform(0.1, 0.5)))
        xt = r6(rng.binomial(np.array(dt, int), rng.uniform(0.1, 0.5)))
        out["ratio_test"].append({"num_c": xc, "den_c": dc, "num_t": xt, "den_t": dt,
                                  "expected": tr(ab.ratio_test(xc, dc, xt, dt))})

    out["cuped"] = []
    for _ in range(15):
        n = int(rng.integers(20, 200))
        x = rng.normal(0, 1, 2 * n)
        y = r6(rng.uniform(0, 1) * x + rng.normal(0, 1, 2 * n) + np.r_[np.zeros(n), np.full(n, 0.2)])
        x = r6(x)
        y_adj, theta = ab.cuped_adjust(y, x)
        out["cuped"].append({"y": y, "x": x, "n_control": n, "expected": {
            "theta": theta, "test_on_adjusted": tr(ab.mean_test(y_adj[:n], y_adj[n:]))}})

    out["msprt"] = []
    for i in range(15):
        kind = ["normal", "binary", "lognormal"][i % 3]
        n = int(rng.integers(150, 500))
        c, t = r6(sample(kind, n)), r6(sample(kind, n))
        tau = float(rng.choice([0.01, 0.05, 0.2]))
        r = ab.msprt(c, t, tau=tau, n_looks=10, min_per_arm=20)
        out["msprt"].append({"kind": kind, "tau": tau, "n_looks": 10, "min_per_arm": 20, "control": c,
                             "treatment": t, "expected": {
                                 "n_control": r.n_control.tolist(), "n_treatment": r.n_treatment.tolist(),
                                 "estimate": r.estimate.tolist(), "p_value": r.p_value.tolist(),
                                 "ci_low": r.ci_low.tolist(), "ci_high": r.ci_high.tolist()}})

    out["multiple"] = []
    for _ in range(15):
        p = r6(rng.uniform(0, 1, int(rng.integers(1, 25))) ** 3)
        out["multiple"].append({"p": p, "holm": ab.holm(p).tolist(),
                                "benjamini_hochberg": ab.benjamini_hochberg(p).tolist()})

    out["heterogeneity_test"] = []
    for _ in range(10):
        k = int(rng.integers(2, 10))
        est, se = r6(rng.normal(0, 0.1, k)), r6(rng.uniform(0.02, 0.1, k))
        out["heterogeneity_test"].append({"estimates": est, "ses": se, "expected": ab.heterogeneity_test(est, se)})

    out["sample_size"] = []
    for _ in range(20):
        p = float(rng.choice([0.01, 0.05, 0.19, 0.45]))
        mde = float(round(p * rng.uniform(0.02, 0.3), 5))
        power = float(rng.choice([0.8, 0.9]))
        out["sample_size"].append({"baseline": p, "mde": mde, "power": power, "alpha": 0.05,
                                   "expected": list(ab.proportion_sample_size(p, mde, 0.05, power))})
    return out


def strict(x):
    # plain JSON has no NaN; JavaScript's JSON.parse would choke on it, so use null
    if isinstance(x, dict):
        return {k: strict(v) for k, v in x.items()}
    if isinstance(x, list):
        return [strict(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


if __name__ == "__main__":
    data = strict(build())
    path = ROOT / "results" / "golden_vectors.json"
    path.write_text(json.dumps(data, allow_nan=False))
    n = sum(len(v) for v in data.values() if isinstance(v, list))
    print(f"wrote {n} cases to {path} ({path.stat().st_size / 1e3:.0f} KB)")
