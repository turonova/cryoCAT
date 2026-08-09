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
    resolve_select_all_ids,
    slice_block,
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
        "tomo_id":    rng.integers(1, 6, n, dtype=float),
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
