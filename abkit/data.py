"""
Getting data in. Files (CSV, TSV, Parquet, Excel, JSON), pandas frames,
polars frames and plain dicts all end up as one pandas DataFrame, and
bad column names or non numeric metrics fail with a message that says
what to fix.
"""

from __future__ import annotations

import difflib
from pathlib import Path

import numpy as np
import pandas as pd


class InputError(ValueError):
    """The data or the options don't fit together. The message says what to change."""


_READERS = {
    ".csv": pd.read_csv, ".tsv": lambda p: pd.read_csv(p, sep="\t"), ".txt": pd.read_csv,
    ".parquet": pd.read_parquet, ".pq": pd.read_parquet,
    ".xlsx": pd.read_excel, ".xlsm": pd.read_excel, ".xls": pd.read_excel,
    ".json": pd.read_json, ".jsonl": lambda p: pd.read_json(p, lines=True),
    ".ndjson": lambda p: pd.read_json(p, lines=True),
}
_NEEDS = {".parquet": "pyarrow", ".pq": "pyarrow", ".xlsx": "openpyxl", ".xlsm": "openpyxl", ".xls": "xlrd"}


def read_table(path) -> pd.DataFrame:
    """Read a data file. The extension picks the reader, and .gz or .zip is fine for text formats."""
    p = Path(path)
    if not p.exists():
        raise InputError(f"no file at {p}")
    suffixes = [s.lower() for s in p.suffixes]
    ext = suffixes[-1] if suffixes else ""
    if ext in (".gz", ".zip", ".bz2", ".xz") and len(suffixes) > 1:
        ext = suffixes[-2]
    if ext not in _READERS:
        raise InputError(f"don't know how to read {p.name}; use one of {', '.join(sorted(_READERS))}")
    try:
        df = _READERS[ext](p)
    except ImportError as e:
        raise InputError(f"reading {ext} files needs {_NEEDS.get(ext, 'another package')}: "
                         f"pip install {_NEEDS.get(ext, '')}".strip()) from e
    if df.empty:
        raise InputError(f"{p.name} has no rows")
    return df


def as_frame(data) -> pd.DataFrame:
    """A path, pandas DataFrame, polars DataFrame (or anything with to_pandas), dict or list of records."""
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, (str, Path)):
        return read_table(data)
    if hasattr(data, "to_pandas"):
        return data.to_pandas()
    if isinstance(data, (dict, list)):
        return pd.DataFrame(data)
    raise InputError(f"can't turn a {type(data).__name__} into a table; pass a DataFrame or a file path")


def need_columns(df: pd.DataFrame, cols, numeric: bool = False, what: str = "column") -> None:
    """Fail clearly when a column is missing (with a close match if there is one) or isn't numbers."""
    for col in cols:
        if col not in df:
            near = difflib.get_close_matches(str(col), [str(c) for c in df.columns], n=1, cutoff=0.6)
            hint = f" Did you mean {near[0]!r}?" if near else f" Columns are {', '.join(map(str, df.columns))}."
            raise InputError(f"no {what} called {col!r}.{hint}")
        if numeric and not (pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])):
            sample = ", ".join(map(repr, df[col].dropna().unique()[:3]))
            raise InputError(f"{col!r} isn't numeric (it holds values like {sample}). A metric needs numbers "
                             "or True/False; recode the column first.")


def need_groups(df: pd.DataFrame, variant: str, control, treatment) -> None:
    """Both groups must exist in the variant column. The error lists the groups that do."""
    need_columns(df, [variant], what="variant column")
    found = df[variant].dropna().unique()
    for role, g in (("control", control), ("treatment", treatment)):
        if g not in set(found):
            shown = ", ".join(sorted(map(repr, found), key=str)[:8])
            raise InputError(f"the {role} group {g!r} isn't in column {variant!r}, which has {shown}")
    if control == treatment:
        raise InputError("control and treatment are the same group")


def plain(x):
    """Turn nested results into JSON safe data, with numpy numbers turned into numbers and nan into null."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [plain(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    return x if x is None or isinstance(x, str) else str(x)
