import matplotlib
import numpy as np
import pandas as pd
import pytest

import abkit as ab
from abkit import plots

matplotlib.use("Agg")
rng = np.random.default_rng(5)


@pytest.fixture
def exp():
    n = 6000
    g = rng.integers(0, 2, n)
    day = pd.Timestamp("2026-03-01") + pd.to_timedelta(rng.integers(0, 14, n), unit="D")
    df = pd.DataFrame({"group": np.where(g, "B", "A"), "y": rng.normal(10, 2, n) + 0.3 * g,
                       "z": rng.binomial(1, 0.2 + 0.03 * g), "seg": rng.choice(["u", "v", "w"], n), "day": day})
    return ab.Experiment(df, "group", "A", "B")


def test_effects_from_results_tables_and_rows(exp):
    r = [exp.analyze("y"), exp.analyze("z")]
    fig = plots.effects(r, title="two metrics")
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["y", "z"]
    assert len(ax.lines) >= 4
    plots.effects(exp.analyze_many(["y", "z"]))
    from abkit.report import build
    plots.effects(build(exp, ["y", "z"])["rows"])


def test_other_charts_draw(exp, tmp_path):
    table, _ = exp.segments("y", "seg")
    plots.segments(table)
    plots.confidence_sequence(exp.sequential("y", mde=0.3, n_looks=20)).savefig(tmp_path / "a.png")
    plots.confidence_sequence(exp.group_sequential("y", n_looks=4))
    plots.daily_lifts(exp.over_time("y", "day")).savefig(tmp_path / "b.png")
    assert (tmp_path / "a.png").stat().st_size > 2000


def test_interval_is_drawn_where_the_result_says(exp):
    r = exp.analyze("y")
    ax = plots.effects([r]).axes[0]
    xs = ax.lines[0].get_xdata()
    assert xs[0] == pytest.approx(r.rel_ci[0]) and xs[1] == pytest.approx(r.rel_ci[1])
