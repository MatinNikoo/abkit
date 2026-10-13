import numpy as np
import pandas as pd
import pytest

import abkit as ab
from abkit.cli import main

rng = np.random.default_rng(23)


def test_cuped_with_two_covariates_removes_r_squared():
    n = 60000
    X = rng.normal(0, 1, (n, 2))
    y = 0.6 * X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.7, n)
    y_adj, theta = ab.cuped_adjust(y, X)
    assert np.allclose(theta, [0.6, 0.5], atol=0.02)
    r2 = 1 - np.var(y_adj) / np.var(y)
    assert r2 == pytest.approx((0.36 + 0.25) / (0.36 + 0.25 + 0.49), abs=0.01)


def test_more_covariates_beat_one():
    n = 20000
    g = rng.integers(0, 2, n)
    a, b = rng.normal(0, 1, n), rng.normal(0, 1, n)
    df = pd.DataFrame({"v": np.where(g, "B", "A"), "a": a, "b": b,
                       "y": a + b + rng.normal(0, 1, n) + 0.03 * g})
    e = ab.Experiment(df, "v", "A", "B")
    assert e.analyze("y", ["a", "b"]).se < e.analyze("y", "a").se < e.analyze("y").se


def test_heterogeneity_test_false_positive_rate():
    hits = 0
    for _ in range(1000):
        se = rng.uniform(0.5, 2, 5)
        hits += ab.heterogeneity_test(rng.normal(1.0, se), se) < 0.05
    assert 0.03 < hits / 1000 < 0.07


def test_segments_find_a_real_difference():
    n = 30000
    seg = rng.choice(["mobile", "desktop"], n)
    g = rng.integers(0, 2, n)
    effect = np.where(seg == "mobile", 0.15, 0.0)
    df = pd.DataFrame({"v": np.where(g, "B", "A"), "seg": seg,
                       "y": rng.normal(0, 1, n) + effect * g})
    table, het = ab.Experiment(df, "v", "A", "B").segments("y", "seg")
    assert het < 0.01
    assert set(table["segment"]) == {"mobile", "desktop"}
    assert table.set_index("segment").loc["mobile", "p_bh"] < 0.05


def test_cli_report_and_size(tmp_path, capsys):
    n = 4000
    g = rng.integers(0, 2, n)
    df = pd.DataFrame({"group": np.where(g, "new", "old"),
                       "converted": rng.random(n) < 0.1 + 0.02 * g,
                       "before": rng.normal(0, 1, n),
                       "country": rng.choice(["US", "CA"], n)})
    path = tmp_path / "exp.csv"
    df.to_csv(path, index=False)
    assert main(["report", str(path), "--variant", "group", "--control", "old", "--treatment", "new",
                 "--metrics", "converted", "--covariate", "before", "--segment", "country"]) == 0
    out = capsys.readouterr().out
    assert "sample ratio check" in out and "converted by country" in out
    assert main(["size", "--baseline", "0.1", "--mde", "0.01"]) == 0
    assert "users per group" in capsys.readouterr().out
    assert main(["report", str(path), "--variant", "nope", "--control", "a", "--treatment", "b",
                 "--metrics", "converted"]) == 2


def test_bayes_binary_matches_known_case():
    a = np.r_[np.ones(100), np.zeros(900)]
    b = np.r_[np.ones(130), np.zeros(870)]
    r = ab.bayes_test(a, b)
    assert 0.97 < r.chance_treatment_better < 0.995
    assert r.expected_loss_ship < r.expected_loss_keep
    assert r.lift_interval[0] < 0.03 < r.lift_interval[1]


def _calibration(truth_sd, prior_sd, n_sims=500):
    r_ = np.random.default_rng(5)
    said, right = [], []
    for _ in range(n_sims):
        true = r_.normal(0, truth_sd)
        a, b = r_.normal(0, 1, 2000), r_.normal(true, 1, 2000)
        r = ab.bayes_test(a, b, draws=20000, prior_sd=prior_sd)
        said.append(r.chance_treatment_better); right.append(true > 0)
    said, right = np.array(said), np.array(right)
    band = (said > 0.8) & (said < 0.95)
    return said[band].mean(), right[band].mean()


def test_flat_prior_is_calibrated_when_effects_are_big():
    said, right = _calibration(truth_sd=0.1, prior_sd=None)
    assert abs(said - right) < 0.07


def test_flat_prior_is_overconfident_when_effects_are_tiny_and_prior_fixes_it():
    said, right = _calibration(truth_sd=0.03, prior_sd=None)
    assert said - right > 0.05            # says ~88%, right less often
    said, right = _calibration(truth_sd=0.03, prior_sd=0.03)
    assert abs(said - right) < 0.07       # with the right prior, honest again


def test_experiment_bayes():
    r_ = np.random.default_rng(1)
    df = pd.DataFrame({"v": ["A"] * 3000 + ["B"] * 3000,
                       "y": np.r_[r_.binomial(1, 0.1, 3000), r_.binomial(1, 0.12, 3000)]})
    r = ab.Experiment(df, "v", "A", "B").bayes("y")
    assert r.chance_treatment_better > 0.95
