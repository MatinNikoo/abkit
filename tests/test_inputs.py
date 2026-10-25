"""Reading data from files and frames, and the errors for bad input."""

import numpy as np
import pandas as pd
import pytest

import abkit as ab
from abkit.data import InputError, as_frame, need_columns, plain, read_table

rng = np.random.default_rng(7)


@pytest.fixture
def df():
    n = 3000
    g = rng.integers(0, 2, n)
    return pd.DataFrame({"group": np.where(g, "B", "A"), "orders": rng.binomial(1, 0.1 + 0.02 * g),
                         "revenue": rng.gamma(2, 3, n), "label": rng.choice(["x", "y"], n)})


def reference(df):
    return ab.Experiment(df, "group", "A", "B").analyze("orders")


@pytest.mark.parametrize("ext", ["csv", "tsv", "json", "jsonl", "parquet", "xlsx", "csv.gz"])
def test_every_file_type_gives_the_same_answer(df, tmp_path, ext):
    pytest.importorskip("pyarrow") if ext == "parquet" else None
    pytest.importorskip("openpyxl") if ext == "xlsx" else None
    path = tmp_path / f"exp.{ext}"
    {"csv": lambda: df.to_csv(path, index=False), "csv.gz": lambda: df.to_csv(path, index=False),
     "tsv": lambda: df.to_csv(path, sep="\t", index=False), "json": lambda: df.to_json(path),
     "jsonl": lambda: df.to_json(path, orient="records", lines=True),
     "parquet": lambda: df.to_parquet(path), "xlsx": lambda: df.to_excel(path, index=False)}[ext]()
    got = ab.Experiment(path, "group", "A", "B").analyze("orders")
    assert got.p_value == pytest.approx(reference(df).p_value, rel=1e-12)
    assert got.lift == pytest.approx(reference(df).lift, rel=1e-12)


def test_polars_dict_and_records(df):
    pl = pytest.importorskip("polars")
    want = reference(df).p_value
    assert ab.Experiment(pl.from_pandas(df), "group", "A", "B").analyze("orders").p_value == pytest.approx(want)
    assert ab.Experiment(df.to_dict("list"), "group", "A", "B").analyze("orders").p_value == pytest.approx(want)
    assert ab.Experiment(df.to_dict("records"), "group", "A", "B").analyze("orders").p_value == pytest.approx(want)


def test_file_errors(tmp_path):
    with pytest.raises(InputError, match="no file at"):
        read_table(tmp_path / "nope.csv")
    odd = tmp_path / "x.docx"
    odd.write_text("a")
    with pytest.raises(InputError, match="don't know how to read"):
        read_table(odd)
    empty = tmp_path / "empty.csv"
    empty.write_text("a,b\n")
    with pytest.raises(InputError, match="no rows"):
        read_table(empty)
    with pytest.raises(InputError, match="can't turn"):
        as_frame(3)


def test_column_errors_name_the_fix(df):
    with pytest.raises(InputError, match="Did you mean 'group'"):
        ab.Experiment(df, "grup", "A", "B")
    with pytest.raises(InputError, match="treatment group 'C'.*'A', 'B'"):
        ab.Experiment(df, "group", "A", "C")
    with pytest.raises(InputError, match="same group"):
        ab.Experiment(df, "group", "A", "A")
    exp = ab.Experiment(df, "group", "A", "B")
    with pytest.raises(InputError, match="Did you mean 'revenue'"):
        exp.analyze("revenu")
    with pytest.raises(InputError, match="isn't numeric.*'x'"):
        exp.analyze("label")
    with pytest.raises(InputError, match="no column called 'zzz'|no column"):
        exp.segments("orders", by="zzz")
    with pytest.raises(InputError):
        exp.check(pre_period=["nope"])
    with pytest.raises(ValueError):
        ab.Experiment(df, "group", "A", "B", alpha=2)
    with pytest.raises(InputError, match="Columns are"):
        need_columns(df, ["something_far_away"])


def test_input_error_is_a_value_error(df):
    with pytest.raises(ValueError):
        ab.Experiment(df, "nope", "A", "B")


def test_plain_makes_json_safe_data():
    out = plain({"a": np.float64(1.5), "b": (np.int64(2), float("nan")), "c": np.array([1.0, np.inf]),
                 "d": np.bool_(True), "e": pd.Timestamp("2026-01-02")})
    assert out == {"a": 1.5, "b": [2, None], "c": [1.0, None], "d": True, "e": "2026-01-02 00:00:00"}


def test_presets_resolve_and_feed_guardrails(df):
    from abkit.presets import GUARDRAILS, resolve_guardrail
    assert resolve_guardrail("latency") == (0.05, False)
    assert resolve_guardrail(0.03) == (0.03, None)
    assert resolve_guardrail("0.03") == (0.03, None)
    with pytest.raises(ValueError, match="presets are"):
        resolve_guardrail("nonsense")
    df = df.assign(load=rng.normal(100, 10, len(df)))
    exp = ab.Experiment(df, "group", "A", "B")
    by_preset = exp.guardrails({"load": "latency"})
    by_hand = exp.guardrails({"load": 0.05}, higher_is_better={"load": False})
    assert by_preset["p"].iloc[0] == by_hand["p"].iloc[0]
    assert not by_preset["higher_is_better"].iloc[0]
    assert set(GUARDRAILS) >= {"latency", "retention", "revenue"}


def test_result_to_dict_and_experiment_readout(df):
    import json
    exp = ab.Experiment(df, "group", "A", "B")
    r = exp.analyze("orders")
    d = json.loads(json.dumps(r.to_dict()))
    assert d["p_value"] == r.p_value and d["rel_ci"] == list(r.rel_ci) and d["significant"] == r.significant
    assert exp.readout(["orders"]).startswith("# Experiment readout")
    assert json.loads(exp.readout(["orders"], fmt="json"))["primary"] == "orders"
