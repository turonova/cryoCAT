"""Pure DataFrame operations for the table editor (W1).

Dash-free and directly testable.  Every function returns a **new** DataFrame;
inputs are never modified.

eval limitations (W7)
---------------------
``pandas.DataFrame.eval`` supports arithmetic, comparison, and boolean
expressions over column names.  It cannot express string methods, custom UDFs,
or multi-step pipelines — those belong in the console instead.
"""
from __future__ import annotations

import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────

CAST_DTYPES: list[str] = ["int32", "int64", "float32", "float64", "str", "bool"]
MERGE_HOW:   list[str] = ["inner", "left", "right", "outer"]


# ── Operations ─────────────────────────────────────────────────────────────────

def derive_column(df: pd.DataFrame, name: str, expression: str) -> pd.DataFrame:
    """Return *df* with a new (or replaced) column *name* = ``df.eval(expression)``.

    Raises
    ------
    ValueError
        When *expression* is invalid.
    """
    if not name or not name.strip():
        raise ValueError("Column name must not be empty.")
    try:
        values = df.eval(expression)
    except Exception as exc:
        raise ValueError(f"Cannot evaluate expression {expression!r}: {exc}") from None
    return df.assign(**{name: values})


def rename_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Return *df* with columns renamed according to *mapping*.

    Keys absent from *df.columns* are silently ignored.
    """
    return df.rename(columns=mapping)


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return *df* with *columns* removed.

    Columns absent from *df* are silently ignored.
    """
    to_drop = [c for c in columns if c in df.columns]
    return df.drop(columns=to_drop)


def reorder_columns(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Return *df* with columns in *order*, remaining columns appended.

    Entries in *order* absent from *df.columns* are silently ignored.
    """
    present = [c for c in order if c in df.columns]
    rest = [c for c in df.columns if c not in set(present)]
    return df[present + rest]


def cast_column(df: pd.DataFrame, column: str, dtype: str) -> pd.DataFrame:
    """Return *df* with *column* cast to *dtype*.

    Raises
    ------
    ValueError
        When *column* is absent or the cast fails.
    """
    if column not in df.columns:
        raise ValueError(f"Column {column!r} is not in the DataFrame.")
    try:
        new_col = df[column].astype(dtype)
    except Exception as exc:
        raise ValueError(f"Cannot cast column {column!r} to {dtype!r}: {exc}") from None
    return df.assign(**{column: new_col})


def merge_tables(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: list[str],
    how: str,
) -> pd.DataFrame:
    """Return the pandas merge of *left* and *right* on key columns *on*."""
    return pd.merge(left, right, on=on, how=how)


def concat_tables(
    frames: list[pd.DataFrame],
    labels: list[str],
    label_column: str = "source",
) -> pd.DataFrame:
    """Stack *frames*, tagging each row with its entry in *labels*.

    *label_column* carries the source label.  Columns absent in some frames
    become NaN (W3).
    """
    tagged = [df.assign(**{label_column: lbl}) for df, lbl in zip(frames, labels)]
    return pd.concat(tagged, ignore_index=True)


# ── Schema check ───────────────────────────────────────────────────────────────

def satisfies_motl_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Return ``(ok, missing_columns)`` against the 20-column motl schema.

    Lazily imports ``Motl.motl_columns`` so this module stays light at import.
    """
    from cryocat.core.cryomotl import Motl
    required: list[str] = list(Motl.motl_columns)
    missing = [c for c in required if c not in df.columns]
    return not missing, missing


# ── Reporting helpers ──────────────────────────────────────────────────────────

def merge_pre_report(
    left: pd.DataFrame, right: pd.DataFrame, on: list[str]
) -> dict:
    """Stats before a merge — for user preview (W2).

    Returns a dict: ``left_n``, ``right_n``, ``matching_keys``,
    ``left_unique_keys``, ``right_unique_keys``.
    """
    valid_on = [c for c in on if c in left.columns and c in right.columns]
    left_n, right_n = len(left), len(right)
    if not valid_on:
        return {
            "left_n": left_n, "right_n": right_n,
            "matching_keys": 0, "left_unique_keys": 0, "right_unique_keys": 0,
        }
    if len(valid_on) == 1:
        c = valid_on[0]
        lk = set(left[c].dropna().tolist())
        rk = set(right[c].dropna().tolist())
    else:
        lk = set(map(tuple, left[valid_on].dropna().values.tolist()))
        rk = set(map(tuple, right[valid_on].dropna().values.tolist()))
    return {
        "left_n": left_n,
        "right_n": right_n,
        "matching_keys": len(lk & rk),
        "left_unique_keys": left.drop_duplicates(subset=valid_on).shape[0],
        "right_unique_keys": right.drop_duplicates(subset=valid_on).shape[0],
    }


def merge_post_report(left_n: int, right_n: int, result_n: int, how: str) -> str:
    """Human-readable summary of row-count change after a merge (W2)."""
    msg = (
        f"Left: {left_n:,} rows · right: {right_n:,} rows → result: {result_n:,} rows."
    )
    if how == "inner" and result_n < min(left_n, right_n):
        dropped = min(left_n, right_n) - result_n
        msg += f"  {dropped:,} row(s) had no match and were dropped."
    elif result_n > max(left_n, right_n):
        extra = result_n - max(left_n, right_n)
        msg += f"  {extra:,} extra row(s) — check for non-unique keys."
    return msg


def concat_nan_report(result: pd.DataFrame, n_frames: int) -> str:
    """Report NaN values introduced by column-set mismatch after concat (W3)."""
    total_nan = int(result.isnull().sum().sum())
    if total_nan == 0:
        return f"Concatenated {n_frames} table(s); no missing values introduced."
    return (
        f"Concatenated {n_frames} table(s); "
        f"{total_nan:,} NaN value(s) from mismatched column sets."
    )


def parse_rename_pairs(text: str) -> dict[str, str]:
    """Parse "old=new\\n..." text into a mapping dict.  Pure."""
    result: dict[str, str] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if "=" in line:
            old, _, new = line.partition("=")
            old, new = old.strip(), new.strip()
            if old and new:
                result[old] = new
    return result


def suggested_label(source_label: str, operation: str) -> str:
    """Derive a default result label from the source label and operation name."""
    suffix = {
        "derive": "derived", "rename": "renamed", "drop": "dropped",
        "reorder": "reordered", "cast": "cast",
        "merge": "merged", "concat": "concat",
    }.get(operation, operation)
    return f"{source_label}_{suffix}"
