"""Tests for cryocat.app.apputils pure functions and run_operation_to_pool.

Covers:
- format_columns, get_relevant_features, _scalar, _format_relion_params,
  get_class_by_name, make_axis_trace.
- run_operation_to_pool: success path (event, pool entry, provenance),
  error path (event, pool unchanged, no provenance),
  replaces= semantics, receiver resolution.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

import cryocat.app.provenance as prov
from cryocat.app import session
from cryocat.app.apputils import (
    format_columns,
    get_relevant_features,
    _scalar,
    _format_relion_params,
    get_class_by_name,
    make_axis_trace,
    run_operation_to_pool,
)
from cryocat.app.pool import PoolState


# ===========================================================================
# format_columns
# ===========================================================================

def test_format_columns_float():
    df = pd.DataFrame({"x": [1.5, 2.5], "n": [1, 2]})
    cols = format_columns(df)
    assert len(cols) == 2
    float_col = next(c for c in cols if c["name"] == "x")
    assert float_col["type"] == "numeric"
    int_col = next(c for c in cols if c["name"] == "n")
    assert "type" not in int_col


def test_format_columns_not_editable():
    df = pd.DataFrame({"a": [1]})
    cols = format_columns(df)
    assert all(not c["editable"] for c in cols)


def test_format_columns_empty():
    df = pd.DataFrame()
    assert format_columns(df) == []


# ===========================================================================
# get_relevant_features
# ===========================================================================

def test_get_relevant_features_basic():
    features = ["feat_a", "feat_b", "feat_c_desc"]
    result = get_relevant_features("desc", features)
    assert "feat_c_desc" not in result
    assert "feat_a" in result
    assert "feat_b" in result


def test_get_relevant_features_none_removed():
    features = ["a", "b_desc", "c_desc"]
    result = get_relevant_features("desc", features)
    assert result == ["a"]


# ===========================================================================
# _scalar
# ===========================================================================

class TestScalar:
    def test_none(self):
        assert _scalar(None) is None

    def test_float(self):
        assert _scalar(3.14) == pytest.approx(3.14)

    def test_list(self):
        assert _scalar([5.0]) == pytest.approx(5.0)

    def test_empty_list(self):
        assert _scalar([]) is None

    def test_non_numeric_string(self):
        assert _scalar("abc") is None

    def test_numeric_string(self):
        result = _scalar("2.5")
        assert result == pytest.approx(2.0)


# ===========================================================================
# _format_relion_params
# ===========================================================================

def test_format_relion_params_empty():
    assert _format_relion_params({}) == ""
    assert _format_relion_params(None) == ""


def test_format_relion_params_version():
    result = _format_relion_params({"version": 3.1, "pixel_size": 2.0})
    assert "Relion 3.1" in result
    assert "pixel size" in result
    assert "2" in result


def test_format_relion_params_binning():
    result = _format_relion_params({"version": 5.0, "binning": 4})
    assert "binning: 4" in result


def test_format_relion_params_formats():
    result = _format_relion_params({
        "version": 3.1,
        "tomo_format": ".mrc",
        "subtomo_format": ".hdf",
    })
    assert ".mrc" in result
    assert ".hdf" in result


# ===========================================================================
# get_class_by_name
# ===========================================================================

def test_get_class_by_name_found():
    from cryocat.analysis.tango import TwistDescriptor
    cls = get_class_by_name("TwistDescriptor")
    assert cls is TwistDescriptor


def test_get_class_by_name_not_found():
    with pytest.raises(ValueError, match="not found"):
        get_class_by_name("NonExistentClass")


# ===========================================================================
# make_axis_trace
# ===========================================================================

def test_make_axis_trace_adds_three_traces():
    import plotly.graph_objects as go
    fig = go.Figure()
    initial = len(fig.data)
    make_axis_trace(fig)
    assert len(fig.data) == initial + 3


def test_make_axis_trace_custom_colors():
    import plotly.graph_objects as go
    fig = go.Figure()
    colors = ["red", "green", "blue"]
    make_axis_trace(fig, colors=colors)
    for trace, color in zip(fig.data, colors):
        assert trace.line.color == color


# ===========================================================================
# run_operation_to_pool — shared fixtures
# ===========================================================================

@pytest.fixture(autouse=True)
def _clean(tmp_path):
    session.close_session()
    prov.clear()
    session.start_session(log_dir=tmp_path)
    yield
    session.close_session()
    prov.clear()


@pytest.fixture
def empty_state():
    return PoolState.from_stores({}, {}, {}, {}, 0)


class FakeMotl:
    def __init__(self, n: int = 5):
        self.df = pd.DataFrame({"x": range(n)})

    def get_unique_values(self, col: str):
        return self.df[col].unique()

    def shrink(self, *, by: int = 1) -> "FakeMotl":
        return FakeMotl(max(0, len(self.df) - by))


def _make_motl(n: int = 5) -> FakeMotl:
    return FakeMotl(n)


def _load(*, n: int = 5) -> FakeMotl:
    return FakeMotl(n)


def _fail(**_kw):
    raise ValueError("simulated failure")


def _call_events():
    return [e for e in session.events() if e.get("kind") == "call"]


# ===========================================================================
# Success path
# ===========================================================================

class TestSuccess:
    def test_writes_exactly_one_call_event(self, empty_state):
        before = len(_call_events())
        run_operation_to_pool(_load, {"n": 5}, empty_state)
        assert len(_call_events()) - before == 1

    def test_call_event_status_ok(self, empty_state):
        run_operation_to_pool(_load, {"n": 5}, empty_state)
        ev = _call_events()[-1]
        assert ev["status"] == "ok"

    def test_creates_one_pool_entry(self, empty_state):
        new_state, motl_id, _ = run_operation_to_pool(_load, {"n": 5}, empty_state)
        assert motl_id in new_state.registry
        assert len(new_state.registry) == 1

    def test_pool_entry_has_correct_n_rows(self, empty_state):
        new_state, motl_id, _ = run_operation_to_pool(_load, {"n": 7}, empty_state)
        assert new_state.registry[motl_id]["n_rows"] == 7

    def test_assign_to_in_event(self, empty_state):
        run_operation_to_pool(_load, {"n": 5}, empty_state)
        ev = _call_events()[-1]
        assert ev.get("assign_to") == "motl_1"

    def test_motl_id_is_motl_zero(self, empty_state):
        _, motl_id, _ = run_operation_to_pool(_load, {"n": 5}, empty_state)
        assert motl_id == "motl_1"

    def test_result_stamped_with_pool_motl_id(self, empty_state):
        _, motl_id, result = run_operation_to_pool(_load, {"n": 5}, empty_state)
        assert getattr(result, "_pool_motl_id", None) == motl_id

    def test_provenance_recorded(self, empty_state):
        _, motl_id, _ = run_operation_to_pool(_load, {"n": 5}, empty_state)
        assert prov.var_for(motl_id) == prov.bind(motl_id)

    def test_provenance_seq_matches_last_event(self, empty_state):
        run_operation_to_pool(_load, {"n": 5}, empty_state)
        ev = _call_events()[-1]
        assert prov.producer("motl_1") == ev["seq"]

    def test_pool_rows_match_result(self, empty_state):
        from cryocat.app.pool import get_rows
        new_state, motl_id, result = run_operation_to_pool(_load, {"n": 4}, empty_state)
        stored_df = get_rows(motl_id)
        assert len(stored_df) == len(result.df)

    def test_next_id_incremented(self, empty_state):
        new_state, _, _ = run_operation_to_pool(_load, {"n": 3}, empty_state)
        assert new_state.next_id == 1

    def test_custom_label(self, empty_state):
        new_state, motl_id, _ = run_operation_to_pool(_load, {"n": 3}, empty_state, label="My Motl")
        assert new_state.registry[motl_id]["label"] == "My Motl"


# ===========================================================================
# Error path
# ===========================================================================

class TestError:
    def test_writes_exactly_one_call_event(self, empty_state):
        before = len(_call_events())
        with pytest.raises(ValueError):
            run_operation_to_pool(_fail, {}, empty_state)
        assert len(_call_events()) - before == 1

    def test_call_event_status_error(self, empty_state):
        with pytest.raises(ValueError):
            run_operation_to_pool(_fail, {}, empty_state)
        ev = _call_events()[-1]
        assert ev["status"] == "error"

    def test_pool_unchanged_on_error(self, empty_state):
        with pytest.raises(ValueError):
            run_operation_to_pool(_fail, {}, empty_state)
        assert len(empty_state.registry) == 0
        assert empty_state.next_id == 0

    def test_no_new_pool_entry_on_error(self, empty_state):
        try:
            run_operation_to_pool(_fail, {}, empty_state)
        except ValueError:
            pass
        assert len(empty_state.registry) == 0

    def test_provenance_not_recorded_on_error(self, empty_state):
        with pytest.raises(ValueError):
            run_operation_to_pool(_fail, {}, empty_state)
        assert prov.producer("motl_1") is None

    def test_error_event_has_assign_to(self, empty_state):
        with pytest.raises(ValueError):
            run_operation_to_pool(_fail, {}, empty_state)
        ev = _call_events()[-1]
        assert ev.get("assign_to") == "motl_1"

    def test_error_event_has_error_dict(self, empty_state):
        with pytest.raises(ValueError):
            run_operation_to_pool(_fail, {}, empty_state)
        ev = _call_events()[-1]
        assert "error" in ev
        assert ev["error"]["type"] == "ValueError"

    def test_exception_is_reraised(self, empty_state):
        with pytest.raises(ValueError, match="simulated failure"):
            run_operation_to_pool(_fail, {}, empty_state)


# ===========================================================================
# replaces= semantics
# ===========================================================================

class TestReplaces:
    def _setup(self, empty_state):
        new_state, motl_id, _ = run_operation_to_pool(_load, {"n": 5}, empty_state)
        return new_state, motl_id

    def test_replaces_keeps_same_motl_id(self, empty_state):
        state1, mid = self._setup(empty_state)
        motl = _make_motl(5)
        motl._pool_motl_id = mid
        prov.record(mid, 0)

        new_state2, returned_id, _ = run_operation_to_pool(
            motl.shrink, {"by": 2}, state1, replaces=mid
        )
        assert returned_id == mid

    def test_replaces_does_not_create_new_entry(self, empty_state):
        state1, mid = self._setup(empty_state)
        motl = _make_motl(5)
        motl._pool_motl_id = mid

        new_state2, _, _ = run_operation_to_pool(
            motl.shrink, {"by": 2}, state1, replaces=mid
        )
        assert len(new_state2.registry) == 1
        assert mid in new_state2.registry

    def test_replaces_updates_n_rows(self, empty_state):
        state1, mid = self._setup(empty_state)
        motl = _make_motl(5)
        motl._pool_motl_id = mid

        new_state2, _, _ = run_operation_to_pool(
            motl.shrink, {"by": 2}, state1, replaces=mid
        )
        assert new_state2.registry[mid]["n_rows"] == 3

    def test_replaces_next_id_unchanged(self, empty_state):
        state1, mid = self._setup(empty_state)
        motl = _make_motl(5)

        new_state2, _, _ = run_operation_to_pool(
            motl.shrink, {"by": 1}, state1, replaces=mid
        )
        assert new_state2.next_id == state1.next_id

    def test_replaces_assign_to_uses_same_var(self, empty_state):
        state1, mid = self._setup(empty_state)
        motl = _make_motl(5)

        run_operation_to_pool(motl.shrink, {"by": 1}, state1, replaces=mid)
        ev = _call_events()[-1]
        assert ev.get("assign_to") == prov.bind(mid)

    def test_replaces_updates_provenance_seq(self, empty_state):
        state1, mid = self._setup(empty_state)
        seq_after_first = prov.producer(mid)
        motl = _make_motl(5)

        run_operation_to_pool(motl.shrink, {"by": 1}, state1, replaces=mid)
        seq_after_second = prov.producer(mid)
        assert seq_after_second > seq_after_first


# ===========================================================================
# Receiver resolution
# ===========================================================================

class TestReceiverResolution:
    def test_receiver_field_set_when_pool_motl_id_stamped(self, empty_state):
        state1, mid, _ = run_operation_to_pool(_load, {"n": 5}, empty_state)
        motl = _make_motl(5)
        motl._pool_motl_id = mid

        run_operation_to_pool(motl.shrink, {"by": 1}, state1, replaces=mid)
        ev = _call_events()[-1]
        assert ev.get("receiver") == prov.bind(mid)

    def test_receiver_none_for_free_function(self, empty_state):
        run_operation_to_pool(_load, {"n": 5}, empty_state)
        ev = _call_events()[-1]
        assert ev.get("receiver") is None
