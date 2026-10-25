import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import abkit as ab
from abkit.cli import main

rng = np.random.default_rng(41)


@pytest.fixture
def csv(tmp_path):
    n = 6000
    g = rng.integers(0, 2, n)
    df = pd.DataFrame({"group": np.where(g, "new", "old"),
                       "converted": rng.random(n) < 0.1 + 0.02 * g,
                       "revenue": rng.lognormal(0, 1, n) * (1 + 0.05 * g),
                       "load_ms": rng.normal(800, 100, n),
                       "before": rng.normal(0, 1, n),
                       "user": np.arange(n),
                       "country": rng.choice(["US", "CA"], n)})
    path = tmp_path / "exp.csv"
    df.to_csv(path, index=False)
    return path


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out.strip() == f"abkit {ab.__version__}"


def test_size_variants(capsys, tmp_path):
    assert main(["size", "--baseline", "0.1", "--mde", "0.01"]) == 0
    n = ab.proportion_sample_size(0.1, 0.01)[0]
    assert f"{n:,} users per group" in capsys.readouterr().out
    assert main(["size", "--sd", "2", "--mde", "0.1"]) == 0
    assert f"{ab.sample_size(2, 0.1)[0]:,} users" in capsys.readouterr().out
    assert main(["size", "--baseline", "0.2", "--users", "10000"]) == 0
    assert "smallest change" in capsys.readouterr().out
    assert main(["size", "--baseline", "0.2", "--mde", "0.01", "--daily-users", "2000", "--traffic-share", "0.5"]) == 0
    assert "cover whole weeks" in capsys.readouterr().out
    pilot = pd.DataFrame({"clicks": rng.poisson(2, 3000), "sessions": rng.poisson(5, 3000) + 1})
    path = tmp_path / "pilot.csv"
    pilot.to_csv(path, index=False)
    assert main(["size", "--pilot", str(path), "--numerator", "clicks", "--denominator", "sessions",
                 "--mde", "0.01"]) == 0
    out = capsys.readouterr().out
    n = ab.ratio_sample_size(pilot["clicks"], pilot["sessions"], 0.01)[0]
    assert "delta method" in out and f"{n:,} users per group" in out


def test_size_errors(capsys):
    assert main(["size", "--mde", "0.01"]) == 2
    assert main(["size", "--baseline", "0.1", "--sd", "1", "--mde", "0.01"]) == 2
    assert main(["size", "--baseline", "0.1"]) == 2
    assert main(["size", "--pilot", "x.csv", "--mde", "0.1"]) == 2
    assert "error" in capsys.readouterr().err


def test_report_prints_checks_first(csv, capsys):
    assert main(["report", str(csv), "--variant", "group", "--control", "old", "--treatment", "new",
                 "--metrics", "converted", "revenue"]) == 0
    out = capsys.readouterr().out
    assert out.index("sample ratio check") < out.index("converted:")
    assert "adjusted for testing 2 metrics" in out


def test_report_errors(csv, capsys):
    base = ["report", str(csv), "--variant", "group", "--control", "old", "--treatment", "new"]
    assert main(base + ["--metrics", "nope"]) == 2
    assert main(base + ["--metrics", "converted", "--guardrail", "load_ms"]) == 2
    assert main(["report", str(csv), "--variant", "group", "--control", "old", "--treatment", "zzz",
                 "--metrics", "converted"]) == 2
    assert "error" in capsys.readouterr().err


def test_report_writes_markdown_and_html(csv, tmp_path):
    md, page = tmp_path / "r.md", tmp_path / "r.html"
    assert main(["report", str(csv), "--variant", "group", "--control", "old", "--treatment", "new",
                 "--metrics", "converted", "revenue", "--primary", "converted", "--guardrail", "load_ms:0.02",
                 "--lower-is-better", "load_ms", "--segment", "country", "--markdown", str(md),
                 "--html", str(page)]) == 0
    text = md.read_text()
    assert text.startswith("# Experiment readout")
    assert "**Decision:" in text and "## Guardrails" in text and "converted by country" in text
    h = page.read_text()
    assert h.startswith("<!doctype html>") and "<table>" in h and "Decision:" in h
    assert "load_ms" in h


def test_aa_command(csv, capsys):
    assert main(["aa", str(csv), "--metric", "revenue", "--sims", "200"]) == 0
    assert "A/A on your data, 200 splits" in capsys.readouterr().out
    assert main(["aa", str(csv), "--metric", "nope"]) == 2


def test_installed_entry_point(csv):
    # the console script, as a user would run it
    r = subprocess.run([sys.executable, "-m", "abkit.cli", "size", "--baseline", "0.19", "--mde", "0.01"],
                       capture_output=True, text=True, check=True)
    assert "users per group" in r.stdout


def test_report_json_matches_the_python_numbers(csv, capsys):
    import json
    base = ["report", str(csv), "--variant", "group", "--control", "old", "--treatment", "new"]
    assert main(base + ["--metrics", "converted", "--guardrail", "load_ms:latency", "--json", "-"]) == 0
    data = json.loads(capsys.readouterr().out)
    df = pd.read_csv(csv)
    r = ab.Experiment(df, "group", "old", "new").analyze("converted")
    row = data["rows"][0]
    assert row["p"] == pytest.approx(r.p_value) and row["rel_lift"] == pytest.approx(r.rel_lift)
    assert data["guardrails"][0]["higher_is_better"] is False
    assert data["decision"] in {"Ship", "Hold", "Look closer", "Don't ship", "No clear effect"}


def test_report_plot_and_file_outputs(csv, tmp_path, capsys):
    png, js = tmp_path / "e.png", tmp_path / "r.json"
    assert main(["report", str(csv), "--variant", "group", "--control", "old", "--treatment", "new",
                 "--metrics", "converted", "revenue", "--plot", str(png), "--json", str(js)]) == 0
    assert png.read_bytes()[:4] == b"\x89PNG" and js.read_text().startswith("{")
    assert "wrote" in capsys.readouterr().out


def test_html_readout_embeds_the_chart(csv, tmp_path):
    page = tmp_path / "r.html"
    main(["report", str(csv), "--variant", "group", "--control", "old", "--treatment", "new",
          "--metrics", "converted", "revenue", "--html", str(page)])
    assert "data:image/png;base64," in page.read_text()


def test_report_text_ends_with_guardrails_and_decision(csv, capsys):
    assert main(["report", str(csv), "--variant", "group", "--control", "old", "--treatment", "new",
                 "--metrics", "converted", "--guardrail", "load_ms:latency", "--cluster", "user"]) == 0
    out = capsys.readouterr().out
    assert "guardrails (non-inferiority)" in out and out.rstrip().splitlines()[-2].startswith("decision:")
    assert "chance" not in out.split("decision:")[0]


def test_where_and_sheet_filters(csv, capsys):
    base = ["report", str(csv), "--variant", "group", "--control", "old", "--treatment", "new",
            "--metrics", "converted"]
    assert main(base + ["--where", "country == 'US'"]) == 0
    us = pd.read_csv(csv).query("country == 'US'")
    assert f"{(us.group == 'old').sum():,} users" in capsys.readouterr().out
    assert main(base + ["--where", "country == 'XX'"]) == 2
    assert main(base + ["--where", "nonsense ==="]) == 2


def test_config_file_and_overrides(csv, tmp_path, capsys):
    import json
    toml = tmp_path / "exp.toml"
    toml.write_text(f'''data = "{csv.as_posix()}"
variant = "group"
control = "old"
treatment = "new"
metrics = ["converted", "revenue"]
primary = "converted"
alpha = 0.1
[guardrail]
load_ms = "latency"
''')
    assert main(["report", "--config", str(toml), "--json", "-"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["alpha"] == 0.1 and data["primary"] == "converted" and data["guardrails"][0]["metric"] == "load_ms"
    assert main(["report", "--config", str(toml), "--alpha", "0.01", "--json", "-"]) == 0
    assert json.loads(capsys.readouterr().out)["alpha"] == 0.01
    js = tmp_path / "exp.json"
    js.write_text(json.dumps({"data": str(csv), "variant": "group", "control": "old", "treatment": "new",
                              "metrics": "converted", "guardrail": ["load_ms:0.05"], "lower-is-better": ["load_ms"]}))
    assert main(["report", "--config", str(js), "--json", "-"]) == 0
    assert json.loads(capsys.readouterr().out)["guardrails"][0]["margin"] == 0.05
    bad = tmp_path / "bad.toml"
    bad.write_text('varient = "group"\n')
    assert main(["report", "--config", str(bad)]) == 2
    assert "unknown option 'varient'" in capsys.readouterr().err
    assert main(["report", "--config", str(tmp_path / "missing.toml")]) == 2


def test_one_config_serves_report_and_plan(csv, tmp_path, capsys):
    cfg = tmp_path / "shared.toml"
    cfg.write_text(f'data = "{csv.as_posix()}"\nvariant = "group"\ncontrol = "old"\ntreatment = "new"\n'
                   'metrics = ["converted"]\nprimary = "converted"\nname = "Shared"\nbaseline = 0.1\n'
                   'relative_mde = 0.1\n')
    assert main(["plan", "--config", str(cfg)]) == 0
    assert "# Experiment plan: Shared" in capsys.readouterr().out
    assert main(["report", "--config", str(cfg), "--markdown", "-"]) == 0


def test_missing_required_options_are_named(csv, capsys):
    assert main(["report", str(csv), "--variant", "group"]) == 2
    assert "--control is required" in capsys.readouterr().err
    assert main(["report"]) == 2
    assert "data file is required" in capsys.readouterr().err


def test_input_errors_are_one_line(csv, tmp_path, capsys):
    base = ["--variant", "group", "--control", "old", "--treatment", "new"]
    assert main(["report", str(tmp_path / "nope.csv"), *base, "--metrics", "converted"]) == 2
    assert capsys.readouterr().err.startswith("error: no file at")
    assert main(["report", str(csv), *base, "--metrics", "country"]) == 2
    assert "isn't numeric" in capsys.readouterr().err
    assert main(["report", str(csv), "--variant", "grup", "--control", "old", "--treatment", "new",
                 "--metrics", "converted"]) == 2
    assert "Did you mean 'group'" in capsys.readouterr().err
    assert main(["report", str(csv), *base, "--metrics", "converted", "--guardrail", "load_ms:huh"]) == 2


def test_size_json_relative_and_cuped(capsys):
    import json
    assert main(["size", "--baseline", "0.2", "--relative-mde", "0.05", "--cuped-rho", "0.5", "--json", "-"]) == 0
    d = json.loads(capsys.readouterr().out)
    sd = (0.2 * 0.8) ** 0.5 * (1 - 0.25) ** 0.5
    assert d["users_per_group"] == ab.sample_size(sd, 0.01)[0]
    assert main(["size", "--sd", "1", "--relative-mde", "0.05"]) == 2


def test_aa_json_and_presets(csv, capsys):
    import json
    assert main(["aa", str(csv), "--metric", "revenue", "--sims", "100", "--json", "-"]) == 0
    assert json.loads(capsys.readouterr().out)["sims"] == 100
    assert main(["presets"]) == 0
    assert "latency" in capsys.readouterr().out


def test_plot_without_matplotlib_is_a_clear_error(csv, monkeypatch, capsys):
    import builtins
    real = builtins.__import__

    def no_mpl(name, *a, **k):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_mpl)
    assert main(["report", str(csv), "--variant", "group", "--control", "old", "--treatment", "new",
                 "--metrics", "converted", "--plot", "x.png"]) == 2
    assert 'pip install "abkit[plots]"' in capsys.readouterr().err
