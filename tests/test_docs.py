"""Every python example in docs/ and the README has to run."""

import re
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md"]
BLOCKS = [(f, i, code) for f in FILES
          for i, code in enumerate(re.findall(r"```python\n(.*?)```", f.read_text(), re.S))]


# stand in for the user's csv in the README example
FIXTURE = """
import numpy as np, pandas as pd
rng = np.random.default_rng(1)
n = 4000
df = pd.DataFrame({
    "group": rng.choice(["A", "B"], n),
    "user_id": np.arange(n),
    "country": rng.choice(["US", "DE", "BR"], n),
    "visits_last_month": rng.poisson(5, n),
    "revenue_last_month": rng.gamma(2, 5, n),
})
df["revenue"] = 0.8 * df["revenue_last_month"] + rng.gamma(2, 3, n)
df["sessions"] = rng.poisson(4, n) + 1
df["clicks"] = rng.binomial(df["sessions"], 0.2)
df["orders"] = rng.binomial(1, 0.19, n)
df["load_time"] = rng.normal(2, 0.3, n)
"""


@pytest.mark.parametrize("path, i, code", BLOCKS, ids=[f"{f.relative_to(ROOT)}#{i}" for f, i, _ in BLOCKS])
def test_example_runs(path, i, code, monkeypatch, tmp_path, capsys):
    if path.name == "README.md" and "ab.Experiment(df" in code:
        code = FIXTURE + code
    elif "pd.read_csv(" in code or "df" in code and "df =" not in code and "df=" not in code:
        pytest.skip("needs your own data")
    monkeypatch.chdir(tmp_path)
    exec(compile(code, f"{path.name}#{i}", "exec"), {"__name__": "__docs__"})
