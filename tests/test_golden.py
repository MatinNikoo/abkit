"""
Recompute every case in results/golden_vectors.json. If this fails, the
library's numbers changed: either fix the code or regenerate the file on
purpose (python tools/golden_vectors.py) and say why in the CHANGELOG,
because the browser Lab is checked against this file.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import abkit as ab

PATH = Path(__file__).resolve().parents[1] / "results" / "golden_vectors.json"
G = json.loads(PATH.read_text()) if PATH.exists() else None
pytestmark = pytest.mark.skipif(G is None, reason="run python tools/golden_vectors.py first")


def same(a, b):
    # null in the file stands for NaN
    a = np.asarray(a, dtype=float)
    b = np.asarray([np.nan if v is None else v for v in b] if isinstance(b, list) else
                   (np.nan if b is None else b), dtype=float)
    return np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)


def check(res, exp):
    for k in ("lift", "se", "ci", "p_value", "rel_lift", "rel_ci", "control_mean", "treatment_mean"):
        assert same(getattr(res, k), exp[k]), k


def test_mean_test_cases():
    for c in G["mean_test"]:
        check(ab.mean_test(c["control"], c["treatment"], alpha=c["alpha"]), c["expected"])


def test_srm_cases():
    for c in G["srm_test"]:
        assert same(ab.srm_test(c["n_control"], c["n_treatment"], c["expected_share"]), c["expected"])


def test_ratio_cases():
    for c in G["ratio_test"]:
        check(ab.ratio_test(c["num_c"], c["den_c"], c["num_t"], c["den_t"]), c["expected"])


def test_cuped_cases():
    for c in G["cuped"]:
        y_adj, theta = ab.cuped_adjust(c["y"], c["x"])
        n = c["n_control"]
        assert same(theta, c["expected"]["theta"])
        check(ab.mean_test(y_adj[:n], y_adj[n:]), c["expected"]["test_on_adjusted"])


def test_msprt_cases():
    for c in G["msprt"]:
        r = ab.msprt(c["control"], c["treatment"], tau=c["tau"], n_looks=c["n_looks"], min_per_arm=c["min_per_arm"])
        for k, v in c["expected"].items():
            assert same(getattr(r, k), v), k


def test_multiple_and_heterogeneity_and_size_cases():
    for c in G["multiple"]:
        assert same(ab.holm(c["p"]), c["holm"]) and same(ab.benjamini_hochberg(c["p"]), c["benjamini_hochberg"])
    for c in G["heterogeneity_test"]:
        assert same(ab.heterogeneity_test(c["estimates"], c["ses"]), c["expected"])
    for c in G["sample_size"]:
        assert list(ab.proportion_sample_size(c["baseline"], c["mde"], c["alpha"], c["power"])) == c["expected"]
