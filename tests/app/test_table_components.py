"""Tests for table-related components and page utilities.

Covers:
- Pure-function tests for _tableops.py: derive, rename, drop, reorder, cast,
  merge, concat, report helpers, parse_rename_pairs, satisfies_motl_schema,
  suggested_label.
- Pool-integration tests: schema routing, motl-pool correctness.
- Save round-trips and _create_from_selected_data (tablesave).
- Working copy lifecycle and guards (tableeditor).
- Layout / ID assertions for the tables tab redesign: picker, replace default,
  edit panel as offcanvas, load in utilities, ops in sidebar, picker both pools.
"""
from __future__ import annotations

import ast
import inspect
import pathlib
import tempfile
import textwrap

import numpy as np
import pandas as pd
import pytest
from dash import html, dcc, no_update
import dash_bootstrap_components as dbc

from cryocat.app.components._tableops import (
    cast_column,
    concat_nan_report,
    concat_tables,
    derive_column,
    drop_columns,
    merge_post_report,
    merge_pre_report,
    merge_tables,
    parse_rename_pairs,
    reorder_columns,
    rename_columns,
    satisfies_motl_schema,
    suggested_label,
)
from cryocat.app.components.tablesave import _create_from_selected_data


# ===========================================================================
# Shared fixtures
# ===========================================================================

@pytest.fixture()
def simple_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})


@pytest.fixture()
def motl_df() -> pd.DataFrame:
    """Minimal DataFrame that satisfies the 20-column motl schema."""
    from cryocat.core.cryomotl import Motl
    rng = np.random.default_rng(0)
    n = 10
    return pd.DataFrame(
        {col: rng.random(n).astype("float32") for col in Motl.motl_columns}
    )


# ===========================================================================
# derive_column
# ===========================================================================

class TestDeriveColumn:
    def test_purity(self, simple_df):
        original_cols = list(simple_df.columns)
        _ = derive_column(simple_df, "d", "a + b")
        assert list(simple_df.columns) == original_cols

    def test_adds_column(self, simple_df):
        result = derive_column(simple_df, "d", "a + b")
        assert "d" in result.columns

    def test_expression_evaluated(self, simple_df):
        result = derive_column(simple_df, "d", "a + b")
        expected = simple_df["a"] + simple_df["b"]
        pd.testing.assert_series_equal(result["d"], expected, check_names=False)

    def test_replaces_existing_column(self, simple_df):
        result = derive_column(simple_df, "a", "a * 2")
        assert list(result["a"]) == [2, 4, 6]

    def test_invalid_expression_raises(self, simple_df):
        with pytest.raises(ValueError, match="expression"):
            derive_column(simple_df, "d", "no_such_col * 2")

    def test_empty_name_raises(self, simple_df):
        with pytest.raises(ValueError):
            derive_column(simple_df, "", "a + b")


# ===========================================================================
# rename_columns
# ===========================================================================

class TestRenameColumns:
    def test_purity(self, simple_df):
        original = list(simple_df.columns)
        _ = rename_columns(simple_df, {"a": "alpha"})
        assert list(simple_df.columns) == original

    def test_renames(self, simple_df):
        result = rename_columns(simple_df, {"a": "alpha", "b": "beta"})
        assert "alpha" in result.columns and "beta" in result.columns
        assert "a" not in result.columns

    def test_absent_key_ignored(self, simple_df):
        result = rename_columns(simple_df, {"z": "zeta"})
        assert list(result.columns) == list(simple_df.columns)

    def test_row_count_unchanged(self, simple_df):
        result = rename_columns(simple_df, {"a": "alpha"})
        assert len(result) == len(simple_df)


# ===========================================================================
# drop_columns
# ===========================================================================

class TestDropColumns:
    def test_purity(self, simple_df):
        original = list(simple_df.columns)
        _ = drop_columns(simple_df, ["a"])
        assert list(simple_df.columns) == original

    def test_drops(self, simple_df):
        result = drop_columns(simple_df, ["a", "b"])
        assert "a" not in result.columns and "b" not in result.columns
        assert "c" in result.columns

    def test_absent_column_ignored(self, simple_df):
        result = drop_columns(simple_df, ["z"])
        assert list(result.columns) == list(simple_df.columns)

    def test_row_count_unchanged(self, simple_df):
        result = drop_columns(simple_df, ["a"])
        assert len(result) == len(simple_df)


# ===========================================================================
# reorder_columns
# ===========================================================================

class TestReorderColumns:
    def test_purity(self, simple_df):
        original = list(simple_df.columns)
        _ = reorder_columns(simple_df, ["c", "a"])
        assert list(simple_df.columns) == original

    def test_order_respected(self, simple_df):
        result = reorder_columns(simple_df, ["c", "a"])
        assert result.columns[0] == "c" and result.columns[1] == "a"

    def test_remaining_appended(self, simple_df):
        result = reorder_columns(simple_df, ["c"])
        assert list(result.columns[:2]) in [["c", "a"], ["c", "b"]]
        assert "c" in result.columns

    def test_absent_order_entry_ignored(self, simple_df):
        result = reorder_columns(simple_df, ["z", "a"])
        assert result.columns[0] == "a"

    def test_row_count_unchanged(self, simple_df):
        result = reorder_columns(simple_df, ["c", "b", "a"])
        assert len(result) == len(simple_df)


# ===========================================================================
# cast_column
# ===========================================================================

class TestCastColumn:
    def test_purity(self, simple_df):
        original_dtype = simple_df["a"].dtype
        _ = cast_column(simple_df, "a", "float64")
        assert simple_df["a"].dtype == original_dtype

    def test_cast_int_to_float(self, simple_df):
        result = cast_column(simple_df, "a", "float64")
        assert result["a"].dtype == np.float64

    def test_absent_column_raises(self, simple_df):
        with pytest.raises(ValueError, match="not in"):
            cast_column(simple_df, "z", "float64")

    def test_incompatible_cast_raises(self, simple_df):
        with pytest.raises(ValueError, match="Cannot cast"):
            cast_column(simple_df, "c", "float64")

    def test_row_count_unchanged(self, simple_df):
        result = cast_column(simple_df, "a", "float64")
        assert len(result) == len(simple_df)


# ===========================================================================
# merge_tables
# ===========================================================================

class TestMergeTables:
    @pytest.fixture()
    def left(self):
        return pd.DataFrame({"id": [1, 2, 3], "val_l": [10, 20, 30]})

    @pytest.fixture()
    def right(self):
        return pd.DataFrame({"id": [2, 3, 4], "val_r": [200, 300, 400]})

    def test_purity(self, left, right):
        left_copy = left.copy()
        _ = merge_tables(left, right, on=["id"], how="inner")
        pd.testing.assert_frame_equal(left, left_copy)

    def test_inner_row_count(self, left, right):
        result = merge_tables(left, right, on=["id"], how="inner")
        assert len(result) == 2

    def test_left_row_count(self, left, right):
        result = merge_tables(left, right, on=["id"], how="left")
        assert len(result) == len(left)

    def test_outer_row_count(self, left, right):
        result = merge_tables(left, right, on=["id"], how="outer")
        assert len(result) == 4


# ===========================================================================
# concat_tables
# ===========================================================================

class TestConcatTables:
    @pytest.fixture()
    def frames(self):
        df1 = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        df2 = pd.DataFrame({"x": [5, 6], "z": [7, 8]})
        return [df1, df2]

    def test_purity(self, frames):
        original_cols = [list(df.columns) for df in frames]
        _ = concat_tables(frames, ["A", "B"])
        assert [list(df.columns) for df in frames] == original_cols

    def test_row_count(self, frames):
        result = concat_tables(frames, ["A", "B"])
        assert len(result) == sum(len(f) for f in frames)

    def test_label_column_added(self, frames):
        result = concat_tables(frames, ["A", "B"], label_column="src")
        assert "src" in result.columns
        assert set(result["src"].unique()) == {"A", "B"}

    def test_nan_from_mismatch(self, frames):
        result = concat_tables(frames, ["A", "B"])
        assert result["y"].isnull().any()
        assert result["z"].isnull().any()


# ===========================================================================
# Reporting helpers
# ===========================================================================

class TestReportHelpers:
    def test_merge_post_report_inner_drop(self):
        msg = merge_post_report(10, 10, 5, "inner")
        assert "5" in msg and "no match" in msg.lower()

    def test_merge_post_report_outer_extra(self):
        msg = merge_post_report(5, 5, 12, "outer")
        assert "extra" in msg.lower()

    def test_merge_post_report_clean(self):
        msg = merge_post_report(10, 10, 10, "left")
        assert "10" in msg
        assert "extra" not in msg.lower() and "no match" not in msg.lower()

    def test_concat_nan_report_none(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        msg = concat_nan_report(df, 2)
        assert "no missing" in msg.lower()

    def test_concat_nan_report_some(self):
        df = pd.DataFrame({"a": [1, None], "b": [None, 4]})
        msg = concat_nan_report(df, 2)
        assert "NaN" in msg or "nan" in msg.lower()

    def test_merge_pre_report_matching_keys(self):
        left  = pd.DataFrame({"id": [1, 2, 3], "v": [10, 20, 30]})
        right = pd.DataFrame({"id": [2, 3, 4], "w": [200, 300, 400]})
        report = merge_pre_report(left, right, ["id"])
        assert report["matching_keys"] == 2

    def test_merge_pre_report_no_shared_cols(self):
        left  = pd.DataFrame({"a": [1, 2]})
        right = pd.DataFrame({"b": [3, 4]})
        report = merge_pre_report(left, right, ["a"])
        assert report["matching_keys"] == 0


class TestParseRenamePairs:
    def test_basic(self):
        assert parse_rename_pairs("a=b\nc=d") == {"a": "b", "c": "d"}

    def test_no_equals_ignored(self):
        assert parse_rename_pairs("no_eq\na=b") == {"a": "b"}

    def test_empty_string(self):
        assert parse_rename_pairs("") == {}

    def test_strips_whitespace(self):
        assert parse_rename_pairs("  a = b  ") == {"a": "b"}


# ===========================================================================
# satisfies_motl_schema
# ===========================================================================

class TestSatisfiesMotlSchema:
    def test_full_schema_passes(self, motl_df):
        ok, missing = satisfies_motl_schema(motl_df)
        assert ok
        assert missing == []

    def test_missing_column_fails(self, motl_df):
        partial = motl_df.drop(columns=["score"])
        ok, missing = satisfies_motl_schema(partial)
        assert not ok
        assert "score" in missing

    def test_extra_columns_ok(self, motl_df):
        df = motl_df.copy()
        df["extra_col"] = 0
        ok, missing = satisfies_motl_schema(df)
        assert ok


def test_suggested_label_known_ops():
    assert suggested_label("my_table", "derive") == "my_table_derived"
    assert suggested_label("my_table", "merge")  == "my_table_merged"


def test_suggested_label_unknown_op():
    result = suggested_label("x", "frobnicate")
    assert "x" in result and "frobnicate" in result


# ===========================================================================
# Pool integration
# ===========================================================================

class TestPoolIntegration:
    def test_motl_to_motl_schema_preserved(self, motl_df):
        result = derive_column(motl_df, "extra", "score * 2")
        ok, _ = satisfies_motl_schema(result)
        assert ok

    def test_drop_required_column_schema_fails(self, motl_df):
        result = drop_columns(motl_df, ["score"])
        ok, missing = satisfies_motl_schema(result)
        assert not ok
        assert "score" in missing

    def test_id_column_survives_rename(self, motl_df):
        result = rename_columns(motl_df, {"score": "my_score"})
        assert "subtomo_id" in result.columns

    def test_insert_motl_stores_correct_rows(self, motl_df):
        from cryocat.app.pool import PoolState, insert_motl, get_rows, clear_payloads
        clear_payloads()
        state = PoolState.from_stores({}, {}, 0)
        state, motl_id = insert_motl(
            state, motl_df,
            label="test", motl_type="emmotl",
            extra=None, meta={}, source_path="", has_tab=True,
        )
        retrieved = get_rows(motl_id)
        assert isinstance(retrieved, pd.DataFrame)
        assert len(retrieved) == len(motl_df)
        assert set(retrieved.columns) >= set(motl_df.columns)

    def test_motl_pool_contains_only_dataframes(self, motl_df):
        from cryocat.app.pool import PoolState, insert_motl, get_rows, clear_payloads
        clear_payloads()
        state = PoolState.from_stores({}, {}, 0)
        state, motl_id = insert_motl(
            state, motl_df,
            label="test", motl_type="emmotl",
            extra=None, meta={}, source_path="", has_tab=True,
        )
        assert isinstance(get_rows(motl_id), pd.DataFrame)

    def test_revision_bumps_on_insert(self, motl_df):
        from cryocat.app.pool import PoolState, insert_motl, clear_payloads
        clear_payloads()
        state = PoolState.from_stores({}, {}, 0)
        state, id1 = insert_motl(
            state, motl_df, label="A", motl_type="emmotl",
            extra=None, meta={}, source_path="", has_tab=True,
        )
        state, id2 = insert_motl(
            state, motl_df, label="B", motl_type="emmotl",
            extra=None, meta={}, source_path="", has_tab=True,
        )
        assert id1 != id2
        assert isinstance(state.registry[id1]["revision"], int)
        assert isinstance(state.registry[id2]["revision"], int)

    def test_add_as_motl_schema_satisfied(self, motl_df):
        from cryocat.app.pool import PoolState, insert_motl, get_rows, clear_payloads
        clear_payloads()
        result = rename_columns(motl_df, {"score": "score"})
        ok, missing = satisfies_motl_schema(result)
        assert ok, f"Expected schema ok; missing: {missing}"
        state = PoolState.from_stores({}, {}, 0)
        state, motl_id = insert_motl(
            state, result, label="renamed", motl_type="emmotl",
            extra=None, meta={}, source_path="", has_tab=True,
        )
        retrieved = get_rows(motl_id)
        assert satisfies_motl_schema(retrieved)[0]

    def test_add_as_motl_schema_not_satisfied(self, motl_df):
        result = drop_columns(motl_df, ["score", "tomo_id"])
        ok, missing = satisfies_motl_schema(result)
        assert not ok
        assert "score" in missing

    def test_script_line_non_df_kwargs_parseable(self):
        name = "new_col"
        expression = "score * 2"
        fragment = f"ops.derive_column(name={name!r}, expression={expression!r}, df=None)"
        try:
            compile(fragment, "<string>", "eval")
        except SyntaxError as exc:
            pytest.fail(f"Script fragment is not valid Python: {exc}\n  {fragment}")


# ===========================================================================
# tablesave: wiring checks
# ===========================================================================

def test_tablesave_csv_no_direct_pool_get_rows():
    from cryocat.app.components.tablesave import register_tablesave_csv_callbacks
    src = inspect.getsource(register_tablesave_csv_callbacks)
    assert "get_rows" not in src, (
        "register_tablesave_csv_callbacks still calls pool.get_rows directly; pass resolve_df instead"
    )


def test_tableplot_no_direct_pool_get_rows():
    from cryocat.app.components import tableplot
    src = inspect.getsource(tableplot)
    assert "get_rows" not in src, (
        "tableplot.py still calls pool.get_rows directly; pass resolve_df instead"
    )


def test_register_tablesave_csv_no_resolve_df():
    from cryocat.app.components.tablesave import register_tablesave_csv_callbacks
    sig = inspect.signature(register_tablesave_csv_callbacks)
    assert "resolve_df" not in sig.parameters


def test_register_table_callbacks_does_not_forward_resolve_df_to_tablesave():
    from cryocat.app.components.tableview import register_table_callbacks
    src = textwrap.dedent(inspect.getsource(register_table_callbacks))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            fn_name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            if fn_name == "register_tablesave_csv_callbacks":
                kwarg_names = {kw.arg for kw in node.keywords}
                assert "resolve_df" not in kwarg_names


def test_csv_save_callback_registers_without_resolve_df():
    from dash import Dash
    from cryocat.app.components.tablesave import register_tablesave_csv_callbacks

    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div()
    register_tablesave_csv_callbacks(app, "t")

    outputs = [str(c.get("output", "")) for c in app._callback_list]
    assert any("t-csv-status-label" in o for o in outputs)


def test_csv_save_uses_pool_get_table_df():
    from cryocat.app.components.tablesave import register_tablesave_csv_callbacks
    src = inspect.getsource(register_tablesave_csv_callbacks)
    assert "get_table_df" in src
    assert "run_operation" in src


def test_csv_save_no_motl_reference():
    from cryocat.app.components.tablesave import register_tablesave_csv_callbacks
    src = inspect.getsource(register_tablesave_csv_callbacks)
    assert "motl_id" not in src
    assert "subtomo_id" not in src


def test_csv_save_table_pool_roundtrip(tmp_path):
    from cryocat.app import datapool, pool as _pool
    datapool._table_payloads.clear()
    datapool._table_counter[0] = 0

    df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
    ref = datapool.insert(df, label="test", id_column="id")

    assert ref["table_id"].startswith("table_")
    resolved = _pool.get_table_df(ref)
    assert resolved is not None
    assert list(resolved.columns) == ["id", "val"]
    assert len(resolved) == 3


def test_tableplot_registers_with_resolve_df():
    from dash import Dash
    from cryocat.app.components.tableplot import register_table_plot_callbacks

    df = pd.DataFrame({"col1": [1.0, 2.0], "col2": [3.0, 4.0]})
    ref = {"desc_key": "dk1"}

    def resolver(r):
        return df if r and r.get("desc_key") else None

    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div(id="tp-store")
    register_table_plot_callbacks(app, "tp", "tp-store", resolve_df=resolver)

    outputs = [str(c.get("output", "")) for c in app._callback_list]
    assert any("tp-graph-options-dropdown" in o for o in outputs)


# ===========================================================================
# _create_from_selected_data
# ===========================================================================

def test_create_from_selected_data_basic():
    rows = [{"x": 1.0}, {"x": 2.0}]
    data_out, type_out, registry_out, active_tab = _create_from_selected_data(
        rows, "emmotl", {}, slot_idx=0, max_motls=3
    )
    assert data_out is not None
    non_null = [d for d in data_out if d is not no_update]
    assert len(non_null) == 1
    assert non_null[0] == rows
    assert active_tab.startswith("me-tab-")


def test_create_from_selected_data_no_free_slot():
    rows = [{"x": 1.0}]
    full_registry = {str(i): {"active": True} for i in range(3)}
    result = _create_from_selected_data(rows, "emmotl", full_registry, slot_idx=0, max_motls=3)
    assert result == (None, None, None, None)


def test_create_from_selected_data_skips_active_slots():
    rows = [{"x": 1.0}]
    registry = {"0": {"active": True}, "1": {"active": False}}
    data_out, type_out, registry_out, active_tab = _create_from_selected_data(
        rows, "emmotl", registry, slot_idx=0, max_motls=3
    )
    assert data_out is not None
    assert active_tab == "me-tab-1"


def test_create_from_selected_data_registry_label():
    rows = [{"x": 1.0}, {"x": 2.0}]
    registry = {"0": {"label": "My source", "active": True}, "1": {"active": True}}
    data_out, type_out, registry_out, active_tab = _create_from_selected_data(
        rows, "emmotl", registry, slot_idx=0, max_motls=4
    )
    assert registry_out is not None
    new_entry = registry_out[active_tab.replace("me-tab-", "")]
    assert "My source" in new_entry["label"]
    assert "2" in new_entry["label"]


def test_create_from_selected_data_long_label_truncated():
    rows = [{"x": 1.0}]
    registry = {"0": {"label": "A" * 50, "active": True}}
    data_out, type_out, registry_out, active_tab = _create_from_selected_data(
        rows, "emmotl", registry, slot_idx=0, max_motls=3
    )
    slot = active_tab.replace("me-tab-", "")
    label = registry_out[slot]["label"]
    assert "..." in label or "…" in label


# ===========================================================================
# Save round-trips
# ===========================================================================

_MOTL_COLUMNS = [
    "score", "geom1", "geom2", "subtomo_id", "tomo_id", "object_id",
    "subtomo_mean", "x", "y", "z", "shift_x", "shift_y", "shift_z",
    "geom3", "geom4", "geom5", "phi", "psi", "theta", "class",
]


@pytest.fixture()
def _bypass_logger(monkeypatch):
    def _direct(fn, kwargs):
        return fn(**kwargs)
    monkeypatch.setattr("cryocat.app.apputils.run_operation", _direct)


@pytest.fixture()
def au1_rows():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.uniform(0, 100, size=(3, len(_MOTL_COLUMNS))), columns=_MOTL_COLUMNS)
    df["tomo_id"] = 1.0
    df["subtomo_id"] = [1.0, 2.0, 3.0]
    df["class"] = 1.0
    return df.to_dict("records")


def test_emmotl_round_trip(_bypass_logger, au1_rows):
    from cryocat.app.apputils import save_motl
    from cryocat.core.cryomotl import Motl

    with tempfile.NamedTemporaryFile(suffix=".em", delete=False) as f:
        out = pathlib.Path(f.name)
    try:
        save_motl(str(out), au1_rows, "emmotl")
        loaded = Motl.load(str(out), "emmotl")
        assert len(loaded.df) == len(au1_rows)
    finally:
        out.unlink(missing_ok=True)


def test_stopgap_round_trip(_bypass_logger, au1_rows):
    from cryocat.app.apputils import save_motl
    from cryocat.core.cryomotl import Motl

    with tempfile.NamedTemporaryFile(suffix=".star", delete=False) as f:
        out = pathlib.Path(f.name)
    try:
        save_motl(str(out), au1_rows, "stopgap")
        loaded = Motl.load(str(out), "stopgap")
        assert len(loaded.df) == len(au1_rows)
    finally:
        out.unlink(missing_ok=True)


def test_relion31_round_trip(_bypass_logger, au1_rows):
    from cryocat.app.apputils import save_motl
    from cryocat.core.cryomotl import Motl

    with tempfile.NamedTemporaryFile(suffix=".star", delete=False) as f:
        out = pathlib.Path(f.name)
    try:
        save_motl(str(out), au1_rows, "relion", rln_version=3.1)
        loaded = Motl.load(str(out), "relion", version=3.1)
        assert len(loaded.df) == len(au1_rows)
    finally:
        out.unlink(missing_ok=True)


def test_dynamo_round_trip(_bypass_logger, au1_rows):
    from cryocat.app.apputils import save_motl
    from cryocat.core.cryomotl import Motl

    with tempfile.NamedTemporaryFile(suffix=".tbl", delete=False) as f:
        out = pathlib.Path(f.name)
    try:
        save_motl(str(out), au1_rows, "dynamo")
        loaded = Motl.load(str(out), "dynamo")
        assert len(loaded.df) == len(au1_rows)
    finally:
        out.unlink(missing_ok=True)


def test_relion5_no_tomos_returns_message_before_writing(_bypass_logger, au1_rows):
    from cryocat.app.apputils import save_motl

    with tempfile.NamedTemporaryFile(suffix=".star", delete=False) as f:
        out = pathlib.Path(f.name)
    original_size = out.stat().st_size
    try:
        result = save_motl(str(out), au1_rows, "relion", rln_version=5.0, rln_tomos=None)
        assert isinstance(result, str)
        assert "tomogram" in result.lower()
        assert out.stat().st_size == original_size
    finally:
        out.unlink(missing_ok=True)


# ===========================================================================
# Working copy lifecycle — pure working-copy functions
# ===========================================================================

def _make_motl_df() -> pd.DataFrame:
    from cryocat.app.components._tableops import satisfies_motl_schema
    from cryocat.core.cryomotl import Motl
    cols = Motl.motl_columns
    return pd.DataFrame({c: [float(i) for i in range(20)] for c in cols})


class TestWorkingCopyLifecycle:

    def setup_method(self):
        from cryocat.app.suite.pages import _wcopy
        _wcopy.clear_all()

    def test_op_changes_wc_not_source(self):
        from cryocat.app.suite.pages._wcopy import init_copy, apply_op, get_copy

        source_df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        init_copy(
            "data_1", source_df,
            source_n_rows=3, source_kind="data", source_reader="plain",
        )
        result_df = source_df.copy()
        result_df["c"] = result_df["a"] * 2
        apply_op("data_1", result_df)

        wc = get_copy("data_1")
        assert "c" in wc.columns
        assert "c" not in source_df.columns

    def test_apply_changes_source_bumps_revision(self, monkeypatch):
        from cryocat.app.suite.pages._wcopy import init_copy, apply_op, get_copy, clear
        from cryocat.app.pool import insert_motl, PoolState, get_rows

        src_df = _make_motl_df()
        state = PoolState.from_stores({}, {}, 0)
        state, motl_id = insert_motl(
            state, src_df, label="test", motl_type="emmotl",
            extra=None, meta={}, source_path="", has_tab=True,
        )
        pr, pm, pn = state.to_stores()
        old_rev = pr[motl_id]["revision"]

        init_copy(
            motl_id, src_df,
            source_n_rows=len(src_df), source_kind="motl", source_reader="",
        )
        wc_df = src_df.copy()
        wc_df = wc_df.iloc[::-1].reset_index(drop=True)
        apply_op(motl_id, wc_df)

        from cryocat.app.pool import replace_motl_rows
        new_state = replace_motl_rows(state, motl_id, wc_df)
        clear(motl_id)

        new_pr, new_pm, new_pn = new_state.to_stores()
        assert new_pr[motl_id]["revision"] == old_rev + 1
        assert get_rows(motl_id) is not None

    def test_save_as_new_leaves_source_unchanged(self):
        from cryocat.app.suite.pages._wcopy import init_copy, apply_op, clear
        from cryocat.app.datapool import DataPoolState, insert_entry, get_payload

        source_df = pd.DataFrame({"x": [10, 20, 30]})
        state = DataPoolState.from_stores({}, 0)
        state, data_id = insert_entry(state, source_df, label="original", reader="plain", source_path="")

        init_copy(
            data_id, source_df,
            source_n_rows=3, source_kind="data", source_reader="plain",
        )
        wc_df = source_df.assign(y=source_df["x"] ** 2)
        apply_op(data_id, wc_df)

        state, new_id = insert_entry(state, wc_df, label="working copy", reader="table_op", source_path="")
        clear(data_id)

        original = get_payload(data_id)
        assert "y" not in original.columns
        assert "y" in get_payload(new_id).columns
        assert new_id != data_id

    def test_discard_clears_working_copy(self):
        from cryocat.app.suite.pages._wcopy import init_copy, apply_op, clear, get_copy, has_copy

        source_df = pd.DataFrame({"a": [1]})
        init_copy("data_2", source_df, source_n_rows=1, source_kind="data", source_reader="plain")
        apply_op("data_2", source_df.assign(b=2))

        assert has_copy("data_2")
        clear("data_2")
        assert not has_copy("data_2")
        assert get_copy("data_2") is None

    def test_lazy_materialisation(self):
        from cryocat.app.suite.pages._wcopy import has_copy, init_copy

        source_id = "data_lazy_test"
        assert not has_copy(source_id)

        source_df = pd.DataFrame({"v": [1, 2]})
        init_copy(source_id, source_df, source_n_rows=2, source_kind="data", source_reader="plain")
        assert has_copy(source_id)


# ===========================================================================
# Guards — validation and guard logic
# ===========================================================================

class TestGuards:

    def setup_method(self):
        from cryocat.app.suite.pages import _wcopy
        _wcopy.clear_all()

    def test_apply_disabled_when_motl_column_dropped(self):
        from cryocat.app.suite.pages._wcopy import validate_for_apply
        from cryocat.core.cryomotl import Motl
        cols = [c for c in Motl.motl_columns if c != "subtomo_id"]
        df = pd.DataFrame({c: [1.0] for c in cols})

        ok, reason = validate_for_apply(df, "motl")
        assert not ok
        assert reason
        assert "subtomo_id" in reason or "motl" in reason.lower()

    def test_apply_disabled_nn_missing_column(self):
        from cryocat.app.suite.pages._wcopy import validate_for_apply

        df = pd.DataFrame({"other_col": [1, 2, 3]})
        ok, reason = validate_for_apply(df, "data", source_reader="nn")
        assert not ok
        assert reason

    def test_apply_valid_for_plain_table_always(self):
        from cryocat.app.suite.pages._wcopy import validate_for_apply

        df = pd.DataFrame({"whatever": [1, 2, 3]})
        ok, reason = validate_for_apply(df, "data", source_reader="plain")
        assert ok
        assert reason == ""

    def test_switching_source_preserves_working_copy(self):
        from cryocat.app.suite.pages._wcopy import init_copy, apply_op, has_copy, get_ops_count

        source_df = pd.DataFrame({"a": [1, 2, 3]})
        init_copy("data_1", source_df, source_n_rows=3, source_kind="data", source_reader="plain")
        apply_op("data_1", source_df.assign(b=2))

        assert get_ops_count("data_1") == 1
        assert has_copy("data_1")

    def test_revision_change_detected(self):
        from cryocat.app.suite.pages._wcopy import init_copy, source_changed

        source_df = pd.DataFrame({"a": [1, 2, 3]})
        init_copy(
            "data_rev", source_df,
            source_n_rows=3, source_kind="data", source_reader="plain",
        )
        assert source_changed("data_rev", current_n_rows=4)
        assert not source_changed("data_rev", current_n_rows=3)

    def test_modal_layout_has_no_wc_store(self):
        from cryocat.app.components.tableeditor import get_table_editor

        modal_layout = get_table_editor("te-modal")

        def _collect_ids(node, acc: set) -> None:
            cid = getattr(node, "id", None)
            if isinstance(cid, str):
                acc.add(cid)
            children = getattr(node, "children", None)
            if children is None:
                return
            if not isinstance(children, list):
                children = [children]
            for ch in children:
                if hasattr(ch, "id") or hasattr(ch, "children"):
                    _collect_ids(ch, acc)

        found_ids: set[str] = set()
        _collect_ids(modal_layout, found_ids)
        assert "te-modal-wc-changed" not in found_ids

    def test_wc_layout_has_wc_store(self):
        from cryocat.app.components.tableeditor import get_table_editor

        wc_layout = get_table_editor("te-wc", working_copy_mode=True)

        def _collect_ids(node, acc: set) -> None:
            cid = getattr(node, "id", None)
            if isinstance(cid, str):
                acc.add(cid)
            children = getattr(node, "children", None)
            if children is None:
                return
            if not isinstance(children, list):
                children = [children]
            for ch in children:
                if hasattr(ch, "id") or hasattr(ch, "children"):
                    _collect_ids(ch, acc)

        found_ids: set[str] = set()
        _collect_ids(wc_layout, found_ids)
        assert "te-wc-wc-changed" in found_ids


# ===========================================================================
# Tables tab layout verification helpers
# ===========================================================================

def _collect_ids(component, found=None) -> set:
    if found is None:
        found = set()
    if hasattr(component, "id") and isinstance(component.id, str):
        found.add(component.id)
    for attr in ("children", "layout"):
        children = getattr(component, attr, None)
        if children is None:
            continue
        if isinstance(children, (list, tuple)):
            for child in children:
                _collect_ids(child, found)
        elif hasattr(children, "id") or hasattr(children, "children"):
            _collect_ids(children, found)
    return found


def _collect_types(component, types=None) -> list[type]:
    if types is None:
        types = []
    types.append(type(component))
    for attr in ("children", "layout"):
        children = getattr(component, attr, None)
        if children is None:
            continue
        if isinstance(children, (list, tuple)):
            for child in children:
                _collect_types(child, types)
        elif hasattr(children, "id") or hasattr(children, "children"):
            _collect_types(children, types)
    return types


# ===========================================================================
# Picker diagnosis
# ===========================================================================

class TestTablesPickerDiagnosis:
    def test_motls_appear_in_options(self):
        from cryocat.app.components.entrypicker import _build_options
        pool_reg = {"motl_1": {"label": "run1", "n_rows": 1000}}
        opts = _build_options(pool_reg, {})
        values = [o["value"] for o in opts]
        assert "motl:motl_1" in values

    def test_separator_present_for_motls(self):
        from cryocat.app.components.entrypicker import _build_options
        pool_reg = {"motl_1": {"label": "run1", "n_rows": 500}}
        opts = _build_options(pool_reg, {})
        disabled = [o for o in opts if o.get("disabled")]
        labels = [o["label"] for o in disabled]
        assert any("Motl" in lbl for lbl in labels)

    def test_table_entries_also_appear(self):
        from cryocat.app.components.entrypicker import _build_options
        dp_reg = {"data_1": {"label": "my_table", "kind": "dataframe", "n_rows": 200}}
        opts = _build_options({}, dp_reg)
        values = [o["value"] for o in opts]
        assert "data:data_1" in values

    def test_volumes_excluded(self):
        from cryocat.app.components.entrypicker import _build_options
        dp_reg = {"data_2": {"label": "myvol", "kind": "volume"}}
        opts = _build_options({}, dp_reg)
        values = [o["value"] for o in opts]
        assert "data:data_2" not in values


# ===========================================================================
# Replace is the default
# ===========================================================================

class TestReplaceDefault:
    def test_create_new_checkbox_exists(self):
        from cryocat.app.components.tableeditor import get_table_editor
        layout = get_table_editor("te-test")
        ids = _collect_ids(layout)
        assert "te-test-create-new" in ids
        assert "te-test-replace-source" not in ids

    def test_create_new_default_off(self):
        from cryocat.app.components.tableeditor import get_table_editor

        def _find_checklist(component):
            if isinstance(component, dbc.Checklist):
                cid = getattr(component, "id", None)
                if isinstance(cid, str) and cid.endswith("-create-new"):
                    return component
            for attr in ("children",):
                children = getattr(component, attr, None)
                if isinstance(children, list):
                    for child in children:
                        result = _find_checklist(child)
                        if result is not None:
                            return result
                elif children is not None:
                    result = _find_checklist(children)
                    if result is not None:
                        return result
            return None

        layout = get_table_editor("te-test2")
        checklist = _find_checklist(layout)
        assert checklist is not None
        assert checklist.value == []

    def test_apply_logic_inverted(self):
        def _compute_replace(create_new_val):
            return "yes" not in (create_new_val or [])

        assert _compute_replace([]) is True
        assert _compute_replace(["yes"]) is False
        assert _compute_replace(None) is True


# ===========================================================================
# Edit panel is an Offcanvas
# ===========================================================================

class TestEditPanelOffcanvas:
    def test_edit_panel_is_offcanvas(self):
        from cryocat.app.components.tableview import get_table_component

        layout = get_table_component("tv-test", show_editor=True)
        types = _collect_types(layout)
        assert dbc.Offcanvas in types
        modal_ids = _collect_ids(layout)
        assert "tv-test-edit-modal" not in modal_ids

    def test_edit_offcanvas_id(self):
        from cryocat.app.components.tableview import get_table_component

        layout = get_table_component("tv-test2", show_editor=True)
        ids = _collect_ids(layout)
        assert "tv-test2-edit-offcanvas" in ids

    def test_edit_offcanvas_placement_end(self):
        from cryocat.app.components.tableview import get_table_component

        def _find_offcanvas(component, target_id):
            if isinstance(component, dbc.Offcanvas):
                if getattr(component, "id", None) == target_id:
                    return component
            for attr in ("children",):
                children = getattr(component, attr, None)
                if isinstance(children, list):
                    for child in children:
                        result = _find_offcanvas(child, target_id)
                        if result is not None:
                            return result
                elif children is not None:
                    result = _find_offcanvas(children, target_id)
                    if result is not None:
                        return result
            return None

        layout = get_table_component("tv-test3", show_editor=True)
        oc = _find_offcanvas(layout, "tv-test3-edit-offcanvas")
        assert oc is not None
        assert oc.placement == "end"


# ===========================================================================
# Load lives in Utilities
# ===========================================================================

class TestLoadInUtilities:
    def test_pdatapool_has_no_load_button(self):
        from cryocat.app.suite.pages import pdatapool
        ids = _collect_ids(pdatapool.layout)
        assert "dp-load-btn" not in ids

    def test_pdatapool_has_no_reader_dropdown(self):
        from cryocat.app.suite.pages import pdatapool
        ids = _collect_ids(pdatapool.layout)
        assert "dp-reader-dd" not in ids

    def test_putilities_has_load_button(self):
        from cryocat.app.suite.pages import putilities
        ids = _collect_ids(putilities.layout)
        assert "util-load-btn" in ids

    def test_putilities_has_reader_dropdown(self):
        from cryocat.app.suite.pages import putilities
        ids = _collect_ids(putilities.layout)
        assert "util-load-reader-dd" in ids


# ===========================================================================
# Operations in Tables sidebar
# ===========================================================================

class TestOpsInSidebar:
    def test_tableeditor_op_dropdown_in_layout(self):
        from cryocat.app.suite.pages import pdatapool
        ids = _collect_ids(pdatapool.layout)
        assert "dp-edit-op-dd" in ids

    def test_edit_source_picker_in_layout(self):
        from cryocat.app.suite.pages import pdatapool
        ids = _collect_ids(pdatapool.layout)
        assert "dp-edit-src-dd" in ids

    def test_no_dp_edit_offcanvas(self):
        from cryocat.app.suite.pages import pdatapool
        ids = _collect_ids(pdatapool.layout)
        assert "dp-edit-offcanvas" not in ids

    def test_no_edit_open_button(self):
        from cryocat.app.suite.pages import pdatapool
        ids = _collect_ids(pdatapool.layout)
        assert "dp-edit-open-btn" not in ids


# ===========================================================================
# Picker lists both pools
# ===========================================================================

class TestPickerBothPools:
    def test_both_groups_present(self):
        from cryocat.app.components.entrypicker import _build_options
        pool_reg = {"motl_1": {"label": "run1", "n_rows": 100}}
        dp_reg   = {"data_1": {"label": "t1",   "kind": "dataframe", "n_rows": 50}}
        opts = _build_options(pool_reg, dp_reg)
        disabled = [o for o in opts if o.get("disabled")]
        labels = [o["label"] for o in disabled]
        assert any("Motl" in lbl for lbl in labels)
        assert any("Table" in lbl for lbl in labels)

    def test_motl_precedes_table_in_options(self):
        from cryocat.app.components.entrypicker import _build_options
        pool_reg = {"motl_1": {"label": "run1", "n_rows": 100}}
        dp_reg   = {"data_1": {"label": "t1",   "kind": "dataframe", "n_rows": 50}}
        opts = _build_options(pool_reg, dp_reg)
        values = [o["value"] for o in opts if not o.get("disabled")]
        assert values.index("motl:motl_1") < values.index("data:data_1")

    def test_decode_value_round_trips(self):
        from cryocat.app.components.entrypicker import decode_value
        assert decode_value("motl:motl_1") == {"motl_id": "motl_1"}
        assert decode_value("data:data_1") == {"data_id": "data_1"}
        assert decode_value(None) is None
        assert decode_value("_sep_motl") is None
