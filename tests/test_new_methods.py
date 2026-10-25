import numpy as np
import pandas as pd
import pytest

import abkit as ab
from abkit.report import build, readout

rng = np.random.default_rng(5)


def _df(n=8000, lift=0.0, guard_drop=0.0, arms=("A", "B")):
    v = rng.choice(list(arms), n)
    t = (v != arms[0]).astype(float)
    return pd.DataFrame({
        "v": v,
        "y": rng.normal(10, 2, n) + lift * t,
        "conv": (rng.random(n) < 0.1 + 0.0 * t).astype(int),
        "speed": rng.normal(100, 10, n) * (1 - guard_drop * t),
        "x": rng.normal(0, 1, n),
        "country": rng.choice(["US", "CA", "UK"], n),
    })


# ----------------------------------------------------------- planning
def test_aa_test_is_calibrated_on_clean_data_and_catches_clusters():
    y = rng.normal(0, 1, 5000)
    r = ab.aa_test(y, sims=600, seed=1)
    assert abs(r.false_positive_rate - 0.05) < 3 * r.mc_se + 0.01
    k = rng.poisson(6, 800) + 1
    g = np.repeat(np.arange(800), k)
    yy = rng.normal(0, 1, 800)[g] + rng.normal(0, 1, len(g))
    rows = ab.aa_test(yy, sims=200, clusters=g, test=ab.mean_test, seed=2)   # users split, rows tested
    users = ab.aa_test(yy, sims=200, clusters=g, seed=3)
    assert rows.false_positive_rate > 0.15
    assert users.false_positive_rate < 0.1
    assert "A/A on your data" in str(users)


def test_proportion_mde_and_ratio_mde():
    n, _ = ab.proportion_sample_size(0.2, 0.01)
    assert ab.proportion_mde(0.2, n) == pytest.approx(0.01, rel=1e-3)
    x, d = rng.poisson(2, 4000).astype(float), rng.poisson(5, 4000).astype(float) + 1
    n, _ = ab.ratio_sample_size(x, d, 0.01)
    assert ab.ratio_mde(x, d, n) == pytest.approx(0.01, rel=1e-3)
    with pytest.raises(ValueError):
        ab.run_length(100, 0)


# --------------------------------------------------------- group sequential
def test_group_sequential_controls_false_positives_and_finds_effects():
    hits_null = hits_alt = 0
    for _ in range(400):
        a = rng.normal(0, 1, 3000)
        hits_null += ab.group_sequential_test(a, rng.normal(0, 1, 3000)).stopped
        hits_alt += ab.group_sequential_test(a, rng.normal(0.1, 1, 3000), kind="pocock").stopped
    assert hits_null / 400 < 0.075
    assert hits_alt / 400 > 0.8


def test_group_sequential_result_and_experiment_wrapper():
    df = _df(lift=0.3)
    r = ab.Experiment(df, "v", "A", "B").group_sequential("y", n_looks=4)
    assert r.stopped and r.stop_index is not None
    i = r.stop_index
    assert r.ci_low[i] <= r.estimate[i] <= r.ci_high[i]
    assert "stopped at look" in str(r)
    assert "look 1" in str(r.design)
    with pytest.raises(ValueError):
        ab.group_sequential_design(3, info=[0.5, 0.4, 1.0])


# -------------------------------------------------------------- analysis
def test_compare_arms():
    df = _df(arms=("A", "B", "C", "D"))
    df.loc[df.v == "D", "y"] += 0.5
    t = ab.Experiment(df, "v", "A", "B").compare_arms("y")
    assert list(t["arm"]) == ["B", "C", "D"]
    assert t.set_index("arm").loc["D", "p_dunnett"] < 0.001
    assert (t["p_dunnett"] >= t["p"] - 1e-12).all()


def test_guardrails_table():
    df = _df(guard_drop=0.0)
    g = ab.Experiment(df, "v", "A", "B").guardrails({"speed": 0.02, "y": 0.02}, {"speed": False})
    assert set(g.columns) >= {"metric", "p", "passes"}
    assert g["passes"].all()
    df = _df(guard_drop=0.05)       # 5% faster is good when lower is better...
    g = ab.Experiment(df, "v", "A", "B").guardrails({"speed": 0.01})   # ...but here higher is better
    assert not g["passes"].iloc[0]


def test_analyze_skewed_table():
    n = 6000
    df = pd.DataFrame({"v": rng.choice(["A", "B"], n), "rev": rng.lognormal(0, 1.2, n)})
    t = ab.Experiment(df, "v", "A", "B").analyze_skewed("rev", n_boot=300)
    assert list(t["view"][:3]) == ["mean (t test)", "mean (bootstrap)", t["view"][2]]
    assert t["view"][2].startswith("mean capped at")
    assert "Mann-Whitney" in set(t["view"])
    assert ((t["p"] >= 0) & (t["p"] <= 1)).all()


def test_analyze_cupac_cuts_variance():
    n = 4000
    X = rng.normal(0, 1, (n, 3))
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["v"] = rng.choice(["A", "B"], n)
    df["y"] = np.sin(2 * df.a) * 2 + df.b * df.c + rng.normal(0, 0.5, n)
    e = ab.Experiment(df, "v", "A", "B")
    r = e.analyze_cupac("y", ["a", "b", "c"])
    assert r.se < 0.6 * e.analyze("y").se
    assert "cross fit" in r.notes[0]


def test_time_checks_find_novelty_and_weekends():
    days = pd.date_range("2026-03-02", periods=14, freq="D")
    def make(lift):
        yc, yt, tc, tt = [], [], [], []
        for i, d in enumerate(days):
            yc.append(rng.normal(0, 1, 1500)); tc += [d] * 1500
            yt.append(rng.normal(lift(i, d), 1, 1500)); tt += [d] * 1500
        return np.concatenate(yc), np.concatenate(yt), tc, tt
    r = ab.time_checks(*make(lambda i, d: 0.3 * np.exp(-i / 3)))
    assert r.trend_p < 0.01 and r.trend_slope < 0
    r = ab.time_checks(*make(lambda i, d: 0.25 if d.day_of_week >= 5 else 0.0))
    assert r.weekday_p < 0.01
    assert "lift over 14 days" in str(r)
    df = pd.DataFrame({"v": ["A"] * 3000 + ["B"] * 3000, "y": rng.normal(0, 1, 6000),
                       "ts": list(pd.date_range("2026-03-02", periods=6, freq="D").repeat(500)) * 2})
    assert len(ab.Experiment(df, "v", "A", "B").over_time("y", "ts", min_per_day=100).daily) == 6


def test_screen_flags_the_real_modifier():
    n = 12000
    df = _df(n)
    df["f"] = rng.normal(0, 1, n)
    df["y"] += 0.6 * (df.v == "B") * (df.f > 0)
    s = ab.Experiment(df, "v", "A", "B").screen("y", ["f", "x", "country"])
    assert s.iloc[0]["feature"] == "f" and s.iloc[0]["p_bh"] < 0.001
    assert (s["p_bh"] >= s["p"] - 1e-12).all()


def test_bootstrap_studentized_and_errors():
    c, t = rng.lognormal(0, 1, 500), rng.lognormal(0, 1, 3000)
    r = ab.bootstrap_test(c, t, method="studentized", n_boot=500)
    assert r.ci[0] < r.lift < r.ci[1]
    with pytest.raises(ValueError):
        ab.bootstrap_test(c, t, stat="median", method="studentized")
    with pytest.raises(ValueError):
        ab.quantile_test(c, t, 1.5)


# ---------------------------------------------------------------- report
@pytest.mark.parametrize("lift, guard_drop, decision", [
    (0.5, 0.0, "Ship"), (-0.5, 0.0, "Don't ship"), (0.0, 0.0, "No clear effect"), (0.5, 0.2, "Look closer")])
def test_report_decisions(lift, guard_drop, decision):
    df = _df(lift=lift, guard_drop=guard_drop)
    e = ab.Experiment(df, "v", "A", "B")
    rep = build(e, ["y", "conv"], guardrails={"speed": 0.01})
    assert rep["decision"] == decision
    if decision == "No clear effect":
        assert "80% power" in rep["why"]


def test_report_holds_on_srm_and_calls_it_a_warning():
    df = _df()
    df = pd.concat([df, df[df.v == "B"].iloc[:600]])
    e = ab.Experiment(df, "v", "A", "B")
    rep = build(e, ["y"])
    assert rep["decision"] == "Hold" and "doesn't show the test is broken" in rep["why"]
    md = readout(e, ["y"], segment="country")
    assert "y by country" in md
    h = readout(e, ["y"], fmt="html")
    assert "<table>" in h and "Hold" in h
    with pytest.raises(ValueError):
        readout(e, ["y"], fmt="pdf")


def test_report_lower_is_better():
    df = _df(guard_drop=0.05)
    e = ab.Experiment(df, "v", "A", "B")
    rep = build(e, ["speed"], higher_is_better={"speed": False})
    assert rep["decision"] == "Ship"


def test_report_shows_chance_to_win_only_with_a_prior():
    e = ab.Experiment(_df(lift=0.5), "v", "A", "B")
    assert "Chance better" not in readout(e, ["y"])
    assert "Chance better" in readout(e, ["y"], prior_sd=0.5)
