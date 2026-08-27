"""Tests for tablegrid pure functions and infinite-row-model invariants.

T1 — apply_sort_model, apply_filter_model, slice_block are pure and correct
T2 — no grid response exceeds one block; _MAX_GRID_ROWS absent
T3 — select-all resolves all filtered rows by identity (GRID_SERVER_SIDE_ROWS.md)
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

# T1 + T3: these imports are RED until tablegrid exposes them
from cryocat.app.components.tablegrid import (
    apply_filter_model,
    apply_sort_model,
    resolve_filtered_ids,
    resolve_select_all_ids,
    slice_block,
    subset_motl_rows,
)

# T2: block_to_records lives in pool (pool serialisation boundary)
from cryocat.app.pool import block_to_records


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "subtomo_id": np.arange(1, n + 1, dtype=float),
        "score":      rng.random(n).astype("float32"),
        "tomo_id":    rng.integers(1, 6, n).astype(float),
        "x":          rng.uniform(-50, 50, n).astype("float32"),
        "label":      [f"item_{i % 5}" for i in range(n)],
    })


@pytest.fixture
def large_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60_000
    from cryocat.core.cryomotl import Motl
    cols = Motl.motl_columns
    df = pd.DataFrame({c: rng.random(n).astype("float32") for c in cols})
    df["subtomo_id"] = np.arange(1, n + 1, dtype=float)
    return df


# ── T1: apply_sort_model ──────────────────────────────────────────────────────

def test_apply_sort_single_asc(sample_df):
    """Single-column ascending sort produces monotonically increasing values."""
    result = apply_sort_model(sample_df, [{"colId": "score", "sort": "asc"}])
    assert result["score"].is_monotonic_increasing


def test_apply_sort_single_desc(sample_df):
    """Single-column descending sort produces monotonically decreasing values."""
    result = apply_sort_model(sample_df, [{"colId": "score", "sort": "desc"}])
    assert result["score"].is_monotonic_decreasing


def test_apply_sort_multi_column(sample_df):
    """Multi-column sort: primary column sorted first, secondary within ties."""
    result = apply_sort_model(
        sample_df,
        [{"colId": "tomo_id", "sort": "asc"}, {"colId": "score", "sort": "desc"}],
    )
    assert result["tomo_id"].is_monotonic_increasing
    # Within each tomo_id group, score should be descending
    for _, grp in result.groupby("tomo_id", sort=False):
        assert grp["score"].is_monotonic_decreasing or len(grp) == 1


def test_apply_sort_empty_sort_model_is_passthrough(sample_df):
    """Empty sort_model returns the DataFrame unchanged (same index order)."""
    result = apply_sort_model(sample_df, [])
    pd.testing.assert_frame_equal(result, sample_df)


def test_sort_applied_before_slice(sample_df):
    """Block 0 of a descending sort is not block 0 of the unsorted frame."""
    sorted_df = apply_sort_model(sample_df, [{"colId": "score", "sort": "desc"}])
    block = slice_block(sorted_df, 0, 20)
    # Top row must be the global maximum, not the first unsorted row
    assert block.iloc[0]["score"] == pytest.approx(sample_df["score"].max(), rel=1e-4)
    # Block 0 of unsorted frame has a different top row
    unsorted_block0 = slice_block(sample_df, 0, 20)
    assert unsorted_block0.iloc[0]["score"] != pytest.approx(sample_df["score"].max(), rel=1e-4)


# ── T1: apply_filter_model ────────────────────────────────────────────────────

def test_filter_number_equals(sample_df):
    """filterType='number', type='equals' keeps only exact matches."""
    target = float(sample_df["tomo_id"].iloc[10])
    result = apply_filter_model(
        sample_df,
        {"tomo_id": {"filterType": "number", "type": "equals", "filter": target}},
        {},
    )
    assert (result["tomo_id"] == target).all()
    assert len(result) > 0


def test_filter_number_greater_than(sample_df):
    """filterType='number', type='greaterThan' keeps rows above threshold."""
    threshold = 0.5
    result = apply_filter_model(
        sample_df,
        {"score": {"filterType": "number", "type": "greaterThan", "filter": threshold}},
        {},
    )
    assert (result["score"] > threshold).all()


def test_filter_number_less_than(sample_df):
    """filterType='number', type='lessThan' keeps rows below threshold."""
    threshold = 0.5
    result = apply_filter_model(
        sample_df,
        {"score": {"filterType": "number", "type": "lessThan", "filter": threshold}},
        {},
    )
    assert (result["score"] < threshold).all()


def test_filter_number_in_range(sample_df):
    """filterType='number', type='inRange' keeps rows inside [filter, filterTo]."""
    result = apply_filter_model(
        sample_df,
        {"score": {"filterType": "number", "type": "inRange", "filter": 0.3, "filterTo": 0.7}},
        {},
    )
    assert ((result["score"] >= 0.3) & (result["score"] <= 0.7)).all()
    assert len(result) > 0


def test_filter_text_contains(sample_df):
    """filterType='text', type='contains' keeps rows whose string contains the value."""
    result = apply_filter_model(
        sample_df,
        {"label": {"filterType": "text", "type": "contains", "filter": "item_3"}},
        {},
    )
    assert (result["label"].str.contains("item_3")).all()
    assert len(result) > 0


def test_filter_slider_and_grid_combine(sample_df):
    """Slider filters (dict) and grid filterModel combine with AND."""
    grid_filter = {"score": {"filterType": "number", "type": "greaterThan", "filter": 0.3}}
    slider_filter = {"x": (0.0, 25.0)}
    result = apply_filter_model(sample_df, grid_filter, slider_filter)
    assert (result["score"] > 0.3).all()
    assert ((result["x"] >= 0.0) & (result["x"] <= 25.0)).all()


def test_filter_slider_only(sample_df):
    """Slider filters alone (empty grid filterModel) work correctly."""
    result = apply_filter_model(sample_df, {}, {"score": (0.4, 0.6)})
    assert ((result["score"] >= 0.4) & (result["score"] <= 0.6)).all()


def test_filter_empty_is_passthrough(sample_df):
    """Empty filter_model and empty slider_filters return all rows."""
    result = apply_filter_model(sample_df, {}, {})
    assert len(result) == len(sample_df)


def test_filter_then_sort_equals_sort_then_filter(sample_df):
    """apply_filter_model then apply_sort_model equals apply_sort_model then apply_filter_model."""
    filter_model = {"score": {"filterType": "number", "type": "greaterThan", "filter": 0.4}}
    sort_model = [{"colId": "score", "sort": "asc"}]

    fts = apply_sort_model(apply_filter_model(sample_df, filter_model, {}), sort_model)
    stf = apply_filter_model(apply_sort_model(sample_df, sort_model), filter_model, {})
    pd.testing.assert_frame_equal(fts.reset_index(drop=True), stf.reset_index(drop=True))


# ── T1: slice_block ───────────────────────────────────────────────────────────

def test_slice_block_basic(sample_df):
    """slice_block(df, 0, 100) returns exactly 100 rows."""
    block = slice_block(sample_df, 0, 100)
    assert len(block) == 100


def test_slice_block_middle(sample_df):
    """slice_block(df, 50, 100) returns rows 50..99."""
    block = slice_block(sample_df, 50, 100)
    assert len(block) == 50
    pd.testing.assert_frame_equal(block.reset_index(drop=True),
                                   sample_df.iloc[50:100].reset_index(drop=True))


def test_slice_block_beyond_end(sample_df):
    """slice_block beyond the end returns fewer rows without raising."""
    block = slice_block(sample_df, 180, 250)
    assert len(block) == 20  # 200 - 180
    assert len(block) < 70  # definitely fewer than requested


def test_slice_block_empty_range(sample_df):
    """slice_block with start == end returns empty frame."""
    block = slice_block(sample_df, 10, 10)
    assert len(block) == 0


# ── T2: _MAX_GRID_ROWS and truncation notice absent ───────────────────────────

def test_no_max_grid_rows_in_app():
    """T2a — _MAX_GRID_ROWS must not exist anywhere under cryocat/app/."""
    app_dir = pathlib.Path(__file__).parent.parent.parent / "cryocat" / "app"
    found = []
    for py_file in sorted(app_dir.rglob("*.py")):
        if "_MAX_GRID_ROWS" in py_file.read_text(encoding="utf-8", errors="ignore"):
            found.append(str(py_file.relative_to(app_dir)))
    assert not found, f"_MAX_GRID_ROWS found in: {found}"


def test_no_truncation_notice_in_app():
    """T2b — get_grid_row_count_notice must not exist under cryocat/app/."""
    app_dir = pathlib.Path(__file__).parent.parent.parent / "cryocat" / "app"
    found = []
    for py_file in sorted(app_dir.rglob("*.py")):
        if "get_grid_row_count_notice" in py_file.read_text(encoding="utf-8", errors="ignore"):
            found.append(str(py_file.relative_to(app_dir)))
    assert not found, f"get_grid_row_count_notice found in: {found}"


def test_block_response_under_budget(large_df):
    """T2c — one cache block serialised via block_to_records is under 512 KB."""
    block = slice_block(large_df, 0, 100)
    payload = {"rowData": block_to_records(block, max_rows=100), "rowCount": len(large_df)}
    size_bytes = len(json.dumps(payload))
    assert size_bytes < 512 * 1024, (
        f"Block payload {size_bytes / 1024:.1f} KB exceeds 512 KB budget"
    )


def test_block_exceeds_budget_raises(large_df):
    """T2d — block_to_records raises ValueError when given more than max_rows."""
    with pytest.raises(ValueError):
        block_to_records(large_df.head(200), max_rows=100)


# ── T3: select-all resolves all filtered rows ─────────────────────────────────

def test_select_all_resolves_full_filtered_set(large_df):
    """T3 — resolve_select_all_ids returns all rows matching the filter, not just cached ones.

    With a score threshold that selects ~5 000 of 60 000 rows, we must get
    ~5 000 identities — not the 100 in the grid's current cache block.
    """
    threshold = large_df["score"].quantile(0.9167)  # top ~8.3% → ~5000 rows
    filter_model = {
        "score": {"filterType": "number", "type": "greaterThan", "filter": float(threshold)}
    }
    ids = resolve_select_all_ids(large_df, filter_model, {})
    expected = int((large_df["score"] > threshold).sum())
    assert len(ids) == expected
    assert len(ids) > 100, "Must return more than one cache block's worth of rows"


def test_select_all_with_slider_filter(sample_df):
    """T3 — resolve_select_all_ids respects slider (non-filterModel) filters."""
    ids = resolve_select_all_ids(sample_df, {}, {"score": (0.6, 1.0)})
    expected_ids = sample_df.loc[sample_df["score"] >= 0.6, "subtomo_id"].tolist()
    assert sorted(ids) == sorted(expected_ids)


def test_select_all_empty_filters_returns_all(sample_df):
    """T3 — no filters → all row identities returned."""
    ids = resolve_select_all_ids(sample_df, {}, {})
    assert len(ids) == len(sample_df)


def test_select_all_uses_subtomo_id_not_index(sample_df):
    """T3 — identity values are subtomo_ids, not positional indices."""
    # Shuffle the frame so row positions differ from subtomo_id values
    shuffled = sample_df.sample(frac=1, random_state=7).reset_index(drop=True)
    ids = resolve_select_all_ids(shuffled, {}, {})
    assert set(ids) == set(sample_df["subtomo_id"].tolist())


# ── GRID_INITIAL_LOAD_AND_LEFTOVERS T1 ───────────────────────────────────────
# rows_response and initial_grid_options are new pure helpers added in W1.
# All tests below are RED until W1 lands (ImportError → test fails).

def test_rows_response_no_request_returns_empty():
    """Handler with None request returns empty response — never raises PreventUpdate."""
    from cryocat.app.components.tablegrid import rows_response
    result = rows_response(None, None)
    assert result == {"rowData": [], "rowCount": 0}


def test_rows_response_missing_df_returns_empty():
    """Handler with None df (payload missing) returns empty — never raises PreventUpdate."""
    from cryocat.app.components.tablegrid import rows_response
    request = {"startRow": 0, "endRow": 100, "sortModel": [], "filterModel": {}}
    result = rows_response(request, None)
    assert result == {"rowData": [], "rowCount": 0}


def test_rows_response_first_block_natural_order(sample_df):
    """startRow=0, endRow=100, empty sort+filter → first 100 rows in natural (unmodified) order."""
    from cryocat.app.components.tablegrid import rows_response
    request = {"startRow": 0, "endRow": 100, "sortModel": [], "filterModel": {}}
    result = rows_response(request, sample_df)
    assert len(result["rowData"]) == 100
    assert result["rowCount"] == len(sample_df)
    # Natural order: rowData must match the first 100 rows of the frame as-is
    expected = sample_df.iloc[:100].to_dict("records")
    assert result["rowData"] == expected


def test_rows_response_row_count_is_filtered_total(large_df):
    """rowCount equals the full filtered set size, not the cache block size."""
    from cryocat.app.components.tablegrid import rows_response
    threshold = float(large_df["score"].quantile(0.5))
    filter_model = {
        "score": {"filterType": "number", "type": "greaterThan", "filter": threshold}
    }
    request = {"startRow": 0, "endRow": 100, "sortModel": [], "filterModel": filter_model}
    result = rows_response(request, large_df)
    expected_total = int((large_df["score"] > threshold).sum())
    assert result["rowCount"] == expected_total
    assert len(result["rowData"]) <= 100


def test_get_grid_starts_with_empty_col_defs():
    """GRID_ROWS_FIX_3: get_grid() must start with columnDefs=[] so no request fires before load."""
    from cryocat.app.components.tablegrid import get_grid
    grid = get_grid("test-pfx")
    assert grid.columnDefs == [], f"Expected [], got {grid.columnDefs!r}"


# ── GRID_EMPTY_TABLE_FIX T1 ───────────────────────────────────────────────────
# T1: rows_response must not poison the grid when data is temporarily absent.
# Tests below are RED until W2 adds the n_rows_hint parameter.

def test_rows_response_no_data_does_not_poison_grid():
    """GRID_EMPTY_TABLE_FIX T1a: absent payload + non-zero hint -> rowCount != 0."""
    from cryocat.app.components.tablegrid import rows_response
    request = {"startRow": 0, "endRow": 100, "sortModel": [], "filterModel": {}}
    result = rows_response(request, None, n_rows_hint=60_000)
    assert result["rowData"] == []
    assert result["rowCount"] == 60_000  # must NOT be 0 — 0 poisons the grid


def test_rows_response_with_data_ignores_hint(large_df):
    """GRID_EMPTY_TABLE_FIX T1b: when df present, rowCount is filtered total, not the hint."""
    from cryocat.app.components.tablegrid import rows_response
    request = {"startRow": 0, "endRow": 100, "sortModel": [], "filterModel": {}}
    result = rows_response(request, large_df, n_rows_hint=999)
    assert len(result["rowData"]) == 100
    assert result["rowCount"] == len(large_df)  # real total, not hint


# ── GRID_EMPTY_TABLE_FIX T2 ───────────────────────────────────────────────────
# T2: col_defs_from_df produces the correct column list for the static grid.

def test_col_defs_fields_match_df_columns(sample_df):
    """GRID_EMPTY_TABLE_FIX T2b: col_defs_from_df produces fields matching df columns."""
    from cryocat.app.components.tablegrid import col_defs_from_df
    defs = col_defs_from_df(sample_df)
    col_fields = [c["field"] for c in defs]
    assert col_fields == list(sample_df.columns)


# ── GRID_ROWS_FIX ─────────────────────────────────────────────────────────────

def test_rows_response_null_request_returns_empty():
    """GRID_ROWS_FIX: rows_response(None, None, 0) returns empty dict, never raises."""
    from cryocat.app.components.tablegrid import rows_response
    result = rows_response(None, None, 0)
    assert result == {"rowData": [], "rowCount": 0}


def test_rows_callback_not_prevent_initial_call():
    """GRID_ROWS_FIX: _rows must not use prevent_initial_call.

    prevent_initial_call suppresses the first getRowsRequest, which fires when
    the grid mounts.  AG Grid does not retry, so the grid stays empty forever.
    """
    from dash import Dash, html
    from cryocat.app.components.tablegrid import register_tablegrid_callbacks

    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div(id="ric-grid-container")
    register_tablegrid_callbacks(
        app, "ric",
        resolve_df=lambda ref: None,
        resolve_n_rows=lambda ref: 0,
    )

    rows_output = "ric-grid.getRowsResponse"
    cb_entry = next(
        (c for c in app._callback_list if c.get("output") == rows_output),
        None,
    )
    assert cb_entry is not None, f"_rows callback not found (output={rows_output!r})"
    assert not cb_entry.get("prevent_initial_call"), (
        "_rows has prevent_initial_call=True — this swallows the first getRowsRequest; "
        "remove it; AG Grid does not retry, so the grid stays empty"
    )


def test_cols_callback_registered():
    """GRID_BUILD_ONCE: _cols must output to {prefix}-grid.columnDefs.

    The grid is built once in the layout.  _cols updates columnDefs in-place
    when data loads — this is the signal that triggers purgeInfiniteCache and
    a fresh getRowsRequest.
    """
    from dash import Dash, html
    from cryocat.app.components.tablegrid import register_tablegrid_callbacks

    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div(id="ric-grid-container")
    register_tablegrid_callbacks(
        app, "ric",
        resolve_df=lambda ref: None,
        resolve_n_rows=lambda ref: 0,
    )

    outputs = [str(c.get("output", "")) for c in app._callback_list]
    assert any("ric-grid.columnDefs" in o for o in outputs), (
        "_cols callback not found — nothing outputs to ric-grid.columnDefs"
    )


def test_row_keys_match_column_fields(sample_df):
    """GRID_BUILD_ONCE S3: to_dict('records') keys must cover all col_defs_from_df fields.

    If a field name in columnDefs doesn't match a key in the row dicts, AG Grid
    renders blank cells even though the data loaded correctly.
    """
    from cryocat.app.components.tablegrid import col_defs_from_df
    fields = {d["field"] for d in col_defs_from_df(sample_df)}
    row = sample_df.to_dict("records")[0]
    assert set(row) >= fields, (
        f"Row dict keys {set(row)} do not cover all col_def fields {fields}"
    )


# ── GRID_VISIBILITY_AND_METADATA T1/T2 ───────────────────────────────────────
# T1 tab-guard tests removed: the tab-activation machinery (_tab_active,
# tabs_id/tab_value) was itself a workaround for prevent_initial_call=True on
# _rows.  Removing prevent_initial_call makes it unnecessary.

def test_rows_response_returns_real_rows_with_no_sort_or_filter(sample_df):
    """T1 — _rows still returns real rows for startRow=0, endRow=100 with no sort/filter."""
    from cryocat.app.components.tablegrid import rows_response
    request = {"startRow": 0, "endRow": 100, "sortModel": [], "filterModel": {}}
    result = rows_response(request, sample_df)
    assert len(result["rowData"]) == 100
    assert result["rowCount"] == len(sample_df)
    assert result["rowData"][0]["subtomo_id"] == sample_df.iloc[0]["subtomo_id"]


def test_get_grid_has_no_static_column_size():
    """GRID_ZERO_WIDTH: get_grid() must NOT set columnSize statically.

    Setting columnSize at construction calls sizeColumnsToFit() while the grid is
    inside a hidden Bootstrap tab pane or page-wrapper (display:none), producing
    warning #29 and leaving columns at zero width. columnSize is set by
    _update_col_defs when the motl loads, at which point the container is visible.
    """
    from cryocat.app.components.tablegrid import get_grid
    grid = get_grid("test-pfx")
    val = getattr(grid, "columnSize", None)
    assert val is None, (
        f"get_grid sets columnSize={val!r} statically — move it to _update_col_defs output"
    )


def test_col_defs_no_fixed_width(sample_df):
    """T2 (column sizing) — col_defs_from_df produces no fixed 'width' that defeats fitting."""
    from cryocat.app.components.tablegrid import col_defs_from_df
    defs = col_defs_from_df(sample_df)
    for col_def in defs:
        assert "width" not in col_def, (
            f"Column '{col_def['field']}' has fixed width {col_def['width']!r} — remove it"
        )


# ── GRID_ROWS_FIX_2 / GRID_ROWS_FIX_3 ───────────────────────────────────────


def test_col_defs_no_none_min_width(sample_df):
    """GRID_ROWS_FIX_2: col_defs_from_df must not set minWidth to None."""
    from cryocat.app.components.tablegrid import col_defs_from_df
    defs = col_defs_from_df(sample_df)
    for col_def in defs:
        assert col_def.get("minWidth") is not None or "minWidth" not in col_def, (
            f"Column '{col_def['field']}' has minWidth=None — omit the key instead"
        )


def test_col_defs_no_checkbox_selection(sample_df):
    """GRID_ROWS_FIX_3: col_defs_from_df must not set checkboxSelection.

    AG Grid 35 warning #129: headerCheckbox is only available for clientSide /
    serverSide row models; using it with infinite row model prevents the
    datasource from initialising.
    """
    from cryocat.app.components.tablegrid import col_defs_from_df
    defs = col_defs_from_df(sample_df)
    for col_def in defs:
        assert "checkboxSelection" not in col_def, (
            f"Column '{col_def['field']}' has checkboxSelection — "
            "incompatible with infinite row model (warning #129)"
        )


def test_get_grid_no_column_size_options():
    """GRID_ROWS_FIX_3: get_grid() must not pass columnSizeOptions.

    AG Grid 35 reports 'invalid gridOptions property columnSizeOptions'.
    """
    from cryocat.app.components.tablegrid import get_grid
    grid = get_grid("test-pfx")
    val = getattr(grid, "columnSizeOptions", None)
    assert val is None, (
        f"get_grid has columnSizeOptions={val!r} — remove it"
    )


# ── TABLE_SELECTION_TO_MOTL T1: resolve_filtered_ids ─────────────────────────


def test_resolve_filtered_ids_returns_subtomo_ids(sample_df):
    """T1 — resolve_filtered_ids returns subtomo_id values for matching rows."""
    filter_model = {"score": {"filterType": "number", "type": "greaterThan", "filter": 0.7}}
    ids = resolve_filtered_ids(sample_df, filter_model, {})
    expected = sample_df.loc[sample_df["score"] > 0.7, "subtomo_id"].tolist()
    assert ids == expected


def test_resolve_filtered_ids_no_subtomo_id_raises():
    """T1 — resolve_filtered_ids raises ValueError when subtomo_id column is absent."""
    df = pd.DataFrame({"score": [0.1, 0.5, 0.9], "x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="subtomo_id"):
        resolve_filtered_ids(df, {}, {})


def test_resolve_filtered_ids_agrees_with_select_all(sample_df):
    """T1 — resolve_filtered_ids and resolve_select_all_ids return the same ids."""
    filter_model = {"tomo_id": {"filterType": "number", "type": "equals", "filter": 2.0}}
    ids_filtered = resolve_filtered_ids(sample_df, filter_model, {})
    ids_select_all = resolve_select_all_ids(sample_df, filter_model, {})
    assert ids_filtered == ids_select_all


def test_resolve_filtered_ids_empty_filters_returns_all(sample_df):
    """T1 — empty filter_model and slider_filters return all subtomo_ids."""
    ids = resolve_filtered_ids(sample_df, {}, {})
    assert ids == sample_df["subtomo_id"].tolist()


def test_resolve_filtered_ids_slider_and_grid_combine(sample_df):
    """T1 — grid filterModel and slider_filters combine with AND."""
    filter_model = {"score": {"filterType": "number", "type": "greaterThan", "filter": 0.4}}
    slider_filters = {"x": (-25.0, 25.0)}
    ids = resolve_filtered_ids(sample_df, filter_model, slider_filters)
    mask = (sample_df["score"] > 0.4) & (sample_df["x"] >= -25.0) & (sample_df["x"] <= 25.0)
    expected = sample_df.loc[mask, "subtomo_id"].tolist()
    assert ids == expected


# ── TABLE_SELECTION_TO_MOTL T2: subset_motl_rows ─────────────────────────────


def test_subset_motl_rows_basic(sample_df):
    """T2 — subset_motl_rows returns rows matching the given ids in original order."""
    target_ids = sample_df["subtomo_id"].iloc[5:10].tolist()
    result = subset_motl_rows(sample_df, target_ids)
    assert list(result["subtomo_id"]) == target_ids


def test_subset_motl_rows_missing_ids_ignored(sample_df):
    """T2 — ids not present in the DataFrame are silently ignored."""
    target_ids = [1.0, 2.0, 9999.0]  # 9999 does not exist
    result = subset_motl_rows(sample_df, target_ids)
    assert set(result["subtomo_id"]) == {1.0, 2.0}


def test_subset_motl_rows_duplicates_no_duplicate_rows(sample_df):
    """T2 — passing duplicate ids does not produce duplicate rows."""
    target_ids = [1.0, 1.0, 2.0, 2.0]
    result = subset_motl_rows(sample_df, target_ids)
    assert len(result) == 2
    assert list(result["subtomo_id"]) == list(result["subtomo_id"].unique())


def test_subset_motl_rows_preserves_all_columns(sample_df):
    """T2 — all source DataFrame columns are present in the result."""
    target_ids = sample_df["subtomo_id"].iloc[:10].tolist()
    result = subset_motl_rows(sample_df, target_ids)
    assert list(result.columns) == list(sample_df.columns)


def test_subset_motl_rows_store_column(sample_df):
    """T2 — store_column writes values mapped from subtomo_id into a new column."""
    target_ids = [1.0, 2.0, 3.0]
    val_map = {1.0: "A", 2.0: "B", 3.0: "C"}
    result = subset_motl_rows(sample_df, target_ids, store_column="label2", values=val_map)
    assert "label2" in result.columns
    for row in result.itertuples():
        assert row.label2 == val_map[row.subtomo_id]


def test_subset_motl_rows_no_subtomo_id_raises():
    """T2 — subset_motl_rows raises ValueError when subtomo_id column is absent."""
    df = pd.DataFrame({"score": [0.1, 0.5, 0.9]})
    with pytest.raises(ValueError, match="subtomo_id"):
        subset_motl_rows(df, [0, 1])


# ── POST_RESET_CHANGES T1 — Sizing ───────────────────────────────────────────


def test_no_size_to_fit_in_source():
    """POST_RESET_CHANGES T1: no responsiveSizeToFit, sizeToFit, or columnSizeOptions
    anywhere under cryocat/app/.

    These attributes call sizeColumnsToFit() which produces AG Grid warning #29
    (zero-width columns) when the grid container is hidden.
    """
    app_dir = pathlib.Path(__file__).parent.parent.parent / "cryocat" / "app"
    found = []
    for py_file in sorted(app_dir.rglob("*.py")):
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        rel = str(py_file.relative_to(app_dir))
        if "responsiveSizeToFit" in text:
            found.append(f"{rel}: responsiveSizeToFit")
        if "columnSizeOptions" in text:
            found.append(f"{rel}: columnSizeOptions")
    assert not found, (
        "Found fit-to-window attributes that cause AG Grid warning #29: " + str(found)
    )


def test_every_col_def_has_min_width(sample_df):
    """POST_RESET_CHANGES T1: col_defs_from_df gives every column a positive minWidth.

    minWidth prevents zero-width columns in hidden containers and ensures readable
    column headers even on narrow viewports.
    """
    from cryocat.app.components.tablegrid import col_defs_from_df
    defs = col_defs_from_df(sample_df)
    for col_def in defs:
        assert "minWidth" in col_def, (
            f"Column '{col_def['field']}' is missing minWidth"
        )
        assert col_def["minWidth"] > 0, (
            f"Column '{col_def['field']}' has non-positive minWidth={col_def['minWidth']!r}"
        )


def test_col_defs_no_fixed_width(sample_df):
    """POST_RESET_CHANGES T1: col_defs_from_df must not set a fixed 'width'.

    A fixed width prevents columns from growing when there is room and defeats
    the purpose of the horizontal-scroll approach.
    """
    from cryocat.app.components.tablegrid import col_defs_from_df
    defs = col_defs_from_df(sample_df)
    for col_def in defs:
        assert "width" not in col_def, (
            f"Column '{col_def['field']}' has fixed width {col_def.get('width')!r} — remove it"
        )


# ── POST_RESET_CHANGES T2 — The indicator ────────────────────────────────────


def test_no_dcc_loading_in_app_and_tablegrid():
    """POST_RESET_CHANGES T2: dcc.Loading must not appear in app.py or tablegrid.py.

    dcc.Loading hides its children while any callback runs into that subtree, and
    unmounts stateful components like AG Grid on each loading toggle.
    """
    files = {
        "suite/app.py": pathlib.Path(__file__).parent.parent.parent / "cryocat" / "app" / "suite" / "app.py",
        "components/tablegrid.py": pathlib.Path(__file__).parent.parent.parent / "cryocat" / "app" / "components" / "tablegrid.py",
    }
    for rel, path in files.items():
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "dcc.Loading" not in text, (
            f"dcc.Loading found in {rel} — it must not wrap page containers or grids"
        )


def test_startup_indicator_in_layout(suite_app):
    """POST_RESET_CHANGES T2: suite-startup-indicator must be in the static layout."""
    from tests.app.conftest import collect_ids
    layout_ids = collect_ids(suite_app.layout)
    assert "suite-startup-indicator" in layout_ids, (
        "suite-startup-indicator not found in layout — W1 startup indicator is missing"
    )


def test_router_hides_startup_indicator(suite_app):
    """POST_RESET_CHANGES T2: the router callback must output suite-startup-indicator.style.

    This ensures the indicator is replaced (hidden) by the first callback fire —
    it does not wrap the content; content and indicator are never both visible.
    """
    found = False
    for item in suite_app._callback_list:
        out = item.get("output", "")
        targets = []
        if isinstance(out, str) and out.startswith(".."):
            parts = out.split("...")
            parts[0] = parts[0][2:]
            if parts[-1].endswith(".."):
                parts[-1] = parts[-1][:-2]
            targets = [p for p in parts if p]
        elif isinstance(out, list):
            targets = out
        else:
            targets = [out]
        for o in targets:
            if isinstance(o, str) and "." in o:
                cid, prop = o.rsplit(".", 1)
                prop = prop.split("@", 1)[0]
                if cid == "suite-startup-indicator" and prop == "style":
                    found = True
    assert found, (
        "No callback outputs to suite-startup-indicator.style — "
        "the indicator will never be hidden when the app is ready"
    )
