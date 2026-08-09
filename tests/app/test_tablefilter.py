"""Tests for cryocat.app.components.tablefilter pure functions."""
from __future__ import annotations

import pandas as pd
import pytest
from dash import no_update

from cryocat.app.components.tablefilter import slider_specs, sync_bounds


# ── slider_specs ──────────────────────────────────────────────────────────────
# slider_specs converts a column_ranges dict {col: [min, max, step]} to
# [{column, min, max, step}]. Range filtering/step logic lives in pool._compute_entry_metadata.

def test_slider_specs_basic():
    specs = slider_specs({"x": [1.0, 3.0, 0.02]})
    assert len(specs) == 1
    assert specs[0] == {"column": "x", "min": 1.0, "max": 3.0, "step": 0.02}


def test_slider_specs_multiple_columns():
    specs = slider_specs({"x": [0.0, 1.0, 0.01], "y": [10.0, 20.0, 1.0]})
    assert len(specs) == 2
    assert specs[0]["column"] == "x"
    assert specs[1]["column"] == "y"


def test_slider_specs_empty():
    assert slider_specs({}) == []


def test_slider_specs_preserves_step():
    specs = slider_specs({"n": [0.0, 9.0, 1.0]})
    assert specs[0]["step"] == pytest.approx(1.0)


def test_slider_specs_float_step():
    specs = slider_specs({"x": [0.0, 1.0, 0.01]})
    assert specs[0]["step"] == pytest.approx(0.01)


# ── sync_bounds ───────────────────────────────────────────────────────────────

def test_sync_bounds_slider_triggered():
    result = sync_bounds([2.0, 8.0], 1.0, 9.0, 0.0, 10.0, "prefix-filter-slider")
    slider_out, min_out, max_out = result
    assert slider_out is no_update
    assert min_out == pytest.approx(2.0)
    assert max_out == pytest.approx(8.0)


def test_sync_bounds_min_input_triggered():
    result = sync_bounds([1.0, 8.0], 3.0, 9.0, 0.0, 10.0, "prefix-filter-min")
    slider_out, min_out, max_out = result
    assert slider_out == [pytest.approx(3.0), pytest.approx(8.0)]
    assert min_out == pytest.approx(3.0)
    assert max_out is no_update


def test_sync_bounds_min_clamped_to_slider_max():
    result = sync_bounds([1.0, 5.0], 9.0, 9.0, 0.0, 10.0, "prefix-filter-min")
    slider_out, min_out, max_out = result
    assert min_out == pytest.approx(5.0)
    assert slider_out == [pytest.approx(5.0), pytest.approx(5.0)]


def test_sync_bounds_max_input_triggered():
    result = sync_bounds([1.0, 8.0], 1.0, 6.0, 0.0, 10.0, "prefix-filter-max")
    slider_out, min_out, max_out = result
    assert slider_out == [pytest.approx(1.0), pytest.approx(6.0)]
    assert min_out is no_update
    assert max_out == pytest.approx(6.0)


def test_sync_bounds_max_clamped_to_slider_min():
    result = sync_bounds([5.0, 9.0], 1.0, 2.0, 0.0, 10.0, "prefix-filter-max")
    slider_out, min_out, max_out = result
    assert max_out == pytest.approx(5.0)
    assert slider_out == [pytest.approx(5.0), pytest.approx(5.0)]


def test_sync_bounds_none_min_input():
    result = sync_bounds([1.0, 9.0], None, 9.0, 0.0, 10.0, "prefix-filter-min")
    assert result == (no_update, no_update, no_update)


def test_sync_bounds_none_max_input():
    result = sync_bounds([1.0, 9.0], 1.0, None, 0.0, 10.0, "prefix-filter-max")
    assert result == (no_update, no_update, no_update)


def test_sync_bounds_unknown_trigger():
    result = sync_bounds([1.0, 9.0], 1.0, 9.0, 0.0, 10.0, "some-other-id")
    assert result == (no_update, no_update, no_update)
