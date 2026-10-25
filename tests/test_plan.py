"""The plan writer, checked against independent power calculations."""

import numpy as np
import pytest

import abkit as ab
from abkit import plan
from abkit.cli import main

sm_power = pytest.importorskip("statsmodels.stats.power")


def test_fixed_sample_matches_statsmodels():
    p = plan.build("t", "orders", baseline=0.1, mde=0.01, power=0.8, date="2026-01-01")
    sd = np.sqrt(0.1 * 0.9)
    n = sm_power.NormalIndPower().solve_power(effect_size=0.01 / sd, alpha=0.05, power=0.8, ratio=1)
    assert p["n_control"] == pytest.approx(n, abs=1.01)
    assert p["n_control"] == ab.proportion_sample_size(0.1, 0.01)[0]


def test_unequal_split_matches_statsmodels():
    p = plan.build("t", "orders", sd=2.0, mde=0.1, split=0.2)
    n = sm_power.NormalIndPower().solve_power(effect_size=0.1 / 2.0, alpha=0.05, power=0.8, ratio=0.25)
    assert p["n_control"] == pytest.approx(n, abs=1.01)
    assert p["n_treatment"] == pytest.approx(n * 0.25, abs=1.01)


def test_relative_mde_and_cuped_shrink_the_sample():
    a = plan.build("t", "x", baseline=0.2, relative_mde=0.05)
    assert a["mde"] == pytest.approx(0.01)
    b = plan.build("t", "x", baseline=0.2, relative_mde=0.05, cuped_rho=0.6)
    assert b["n_control"] / a["n_control"] == pytest.approx(1 - 0.36, abs=1e-3)


def test_cuped_planning_matches_a_simulation():
    # the sample the plan asks for should reach about 80% power when the covariate has that correlation
    rho, n_sims = 0.6, 500
    p = plan.build("t", "x", sd=1.0, mde=0.2, cuped_rho=rho)
    n = p["n_control"]
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(n_sims):
        x = rng.normal(size=2 * n)
        g = np.repeat([0, 1], n)
        y = rho * x + np.sqrt(1 - rho**2) * rng.normal(size=2 * n) + 0.2 * g
        adj, _ = ab.cuped_adjust(y, x[:, None])
        hits += ab.mean_test(adj[:n], adj[n:]).p_value < 0.05
    assert abs(hits / n_sims - 0.8) < 0.07


def test_group_sequential_section_uses_the_design():
    p = plan.build("t", "x", baseline=0.1, mde=0.01, sequential="group", looks=4)
    d = ab.group_sequential_design(4)
    assert p["looks"]["inflation"] == pytest.approx(d.inflation(0.8))
    assert p["looks"]["bounds"] == pytest.approx(d.bounds.tolist())
    assert p["looks"]["n_control"] >= p["n_control"]
    text = plan.to_markdown(p)
    assert "Planned looks: 4" in text and f"{d.bounds[0]:.3f}" in text


def test_markdown_covers_every_choice():
    p = plan.build("Checkout", "orders", hypothesis="A shorter form lifts orders.", baseline=0.1,
                   relative_mde=0.05, daily_users=3000, secondary=["revenue"], guardrails={"load_ms": "latency",
                   "refunds": 0.2}, higher_is_better={"refunds": False}, covariate="orders_before", cuped_rho=0.5,
                   cluster="city", segments=["country"], owner="Matin", date="2026-10-24")
    text = plan.to_markdown(p)
    for piece in ["# Experiment plan: Checkout", "Written 2026-10-24 by Matin", "A shorter form lifts orders.",
                  "| Secondary | revenue |", "| Guardrail | load_ms | down |", "| Guardrail | refunds | down |",
                  "CUPED on orders_before", "clusters of city", "Segments: country", "days", "## Decision rule"]:
        assert piece in text
    assert chr(0x2014) not in text and chr(0x2013) not in text


def test_bad_plans_raise():
    with pytest.raises(ValueError):
        plan.build("t", "x", mde=0.1)
    with pytest.raises(ValueError):
        plan.build("t", "x", baseline=0.1)
    with pytest.raises(ValueError):
        plan.build("t", "x", sd=1, relative_mde=0.1)
    with pytest.raises(ValueError):
        plan.build("t", "x", baseline=0.1, mde=0.01, cuped_rho=1.2)
    with pytest.raises(ValueError):
        plan.build("t", "x", baseline=0.1, mde=0.01, sequential="weekly")


def test_plan_command(tmp_path, capsys):
    out = tmp_path / "plan.md"
    assert main(["plan", "--name", "Gate 40", "--primary", "retention_7", "--baseline", "0.19",
                 "--relative-mde", "0.05", "--guardrail", "retention_1:retention", "--sequential", "group",
                 "--daily-users", "4000", "--date", "2026-10-24", "--out", str(out)]) == 0
    text = out.read_text()
    assert "# Experiment plan: Gate 40" in text and "Planned looks: 5" in text
    capsys.readouterr()
    assert main(["plan", "--name", "x", "--primary", "y", "--sd", "2", "--mde", "0.1", "--json", "-"]) == 0
    import json
    data = json.loads(capsys.readouterr().out)
    assert data["n_control"] == ab.sample_size(2, 0.1)[0]
    assert main(["plan", "--name", "x", "--primary", "y"]) == 2
