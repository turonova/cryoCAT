"""Tests for cryocat.app.datapool — data pool state reducers and readers.

Covers DataPoolState construction, insert_entry, remove_entry, get_payload, and
the pool-viewer bridge (set_view_df / clear_view_df), reader discovery
integration (_reader_options list and round-trip reading via get_payload).

Design constraints
------------------
* No import from pdatapool (avoids Dash component import at test time).
* All helpers are imported from datapool and discovery only.
* Uses monkeypatch to restore module-level _payloads / pool._payloads between
  tests so each test is independent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryocat.app import datapool
from cryocat.app.datapool import (
    DataEntry,
    DataPayloadMissing,
    DataPoolState,
    clear_payloads,
    clear_view_df,
    get_payload,
    insert_entry,
    remove_entry,
    set_view_df,
    _detect_kind,
    _make_entry,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_payloads():
    """Wipe server-side stores before and after every test."""
    clear_payloads()
    # also clear pool bridge if pool is importable
    try:
        from cryocat.app.pool import _payloads as _pp
        _pp.pop("dp-view", None)
    except Exception:
        pass
    yield
    clear_payloads()
    try:
        from cryocat.app.pool import _payloads as _pp
        _pp.pop("dp-view", None)
    except Exception:
        pass


def _empty_state() -> DataPoolState:
    return DataPoolState.from_stores(None, None)


def _state_with_df() -> tuple[DataPoolState, str]:
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
    state = _empty_state()
    return insert_entry(state, df, label="my_df", reader="r1", source_path="/tmp/a.csv")


def _state_with_arr() -> tuple[DataPoolState, str]:
    arr = np.arange(12, dtype=float).reshape(3, 4)
    state = _empty_state()
    return insert_entry(state, arr, label="my_arr", reader="r2", source_path="/tmp/b.npy")


def _state_with_vol() -> tuple[DataPoolState, str]:
    vol = np.zeros((8, 8, 8), dtype=np.float32)
    state = _empty_state()
    return insert_entry(state, vol, label="my_vol", reader="r3", source_path="/tmp/c.mrc")


# ═══════════════════════════════════════════════════════════════════════════════
# Pool state and handles
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataPoolState:
    # ── T1a: empty state construction ─────────────────────────────────────────

    def test_empty_state_from_none(self):
        state = DataPoolState.from_stores(None, None)
        assert state.registry == {}
        assert state.next_id == 0

    def test_empty_state_to_stores_roundtrip(self):
        state = DataPoolState.from_stores({}, 0)
        reg, nid = state.to_stores()
        assert reg == {}
        assert nid == {"n": 0, "kc": {}}

    # ── T1b: detect_kind ──────────────────────────────────────────────────────

    def test_detect_kind_dataframe(self):
        assert _detect_kind(pd.DataFrame({"a": [1]})) == "dataframe"

    def test_detect_kind_1d_array(self):
        assert _detect_kind(np.array([1, 2, 3])) == "array"

    def test_detect_kind_2d_array(self):
        assert _detect_kind(np.zeros((4, 4))) == "array"

    def test_detect_kind_3d_array_is_volume(self):
        assert _detect_kind(np.zeros((4, 4, 4))) == "volume"

    def test_detect_kind_dict(self):
        assert _detect_kind({"key": "value"}) == "dict"

    def test_detect_kind_unknown_fallback(self):
        assert _detect_kind(object()) == "dataframe"

    # ── T1c: insert_entry ─────────────────────────────────────────────────────

    def test_insert_entry_increments_next_id(self):
        state, data_id = _state_with_df()
        assert state.next_id == 1
        assert data_id == "data_1"

    def test_insert_entry_two_items(self):
        state1, id1 = _state_with_df()
        arr = np.zeros((3, 3))
        state2, id2 = insert_entry(state1, arr, label="arr", reader="r2", source_path="/b")
        assert state2.next_id == 2
        assert id1 in state2.registry
        assert id2 in state2.registry

    def test_insert_entry_stores_payload_server_side(self):
        state, data_id = _state_with_df()
        assert data_id in datapool._payloads

    def test_insert_entry_registry_has_all_fields(self):
        state, data_id = _state_with_df()
        rec = state.registry[data_id]
        assert rec["kind"] == "dataframe"
        assert rec["label"] == "my_df"
        assert rec["source_path"] == "/tmp/a.csv"

    def test_insert_dataframe_entry_kind_and_columns(self):
        state, data_id = _state_with_df()
        rec = state.registry[data_id]
        assert rec["kind"] == "dataframe"
        assert rec["columns"] == ["x", "y"]
        assert rec["n_rows"] == 3

    def test_insert_array_entry_shape(self):
        state, data_id = _state_with_arr()
        rec = state.registry[data_id]
        assert rec["kind"] == "array"
        assert tuple(rec["shape"]) == (3, 4)

    def test_insert_volume_entry_kind(self):
        state, data_id = _state_with_vol()
        rec = state.registry[data_id]
        assert rec["kind"] == "volume"

    # ── T1d: remove_entry ────────────────────────────────────────────────────

    def test_remove_entry_leaves_state_clean(self):
        state, data_id = _state_with_df()
        new_state = remove_entry(state, data_id)
        assert data_id not in new_state.registry
        assert new_state.next_id == state.next_id

    def test_remove_entry_drops_payload(self):
        state, data_id = _state_with_df()
        remove_entry(state, data_id)
        assert data_id not in datapool._payloads

    def test_remove_entry_noop_for_unknown_id(self):
        state = _empty_state()
        same = remove_entry(state, "data-999")
        assert same.registry == {}
        assert same.next_id == 0

    # ── T1e: get_payload ─────────────────────────────────────────────────────

    def test_get_payload_returns_original_object(self):
        df = pd.DataFrame({"z": [7, 8, 9]})
        state = _empty_state()
        state, data_id = insert_entry(state, df, label="L", reader="r", source_path="/p")
        result = get_payload(data_id, state)
        assert result is df

    def test_get_payload_missing_raises(self):
        with pytest.raises(DataPayloadMissing):
            get_payload("data-999")

    def test_get_payload_missing_message_contains_data_id(self):
        try:
            get_payload("data-42")
        except DataPayloadMissing as exc:
            assert "data-42" in str(exc)

    # ── T1f: pool viewer bridge ───────────────────────────────────────────────

    def test_set_view_df_writes_dataframe_to_pool(self):
        pytest.importorskip("cryocat.app.pool")
        from cryocat.app.pool import _payloads as _pp
        state, data_id = _state_with_df()
        set_view_df(data_id, state)
        assert "dp-view" in _pp
        stored = _pp["dp-view"].rows
        assert isinstance(stored, pd.DataFrame)
        assert list(stored.columns) == ["x", "y"]

    def test_set_view_df_2d_array_converts_columns(self):
        pytest.importorskip("cryocat.app.pool")
        from cryocat.app.pool import _payloads as _pp
        state, data_id = _state_with_arr()
        set_view_df(data_id, state)
        assert "dp-view" in _pp
        df = _pp["dp-view"].rows
        assert list(df.columns) == ["col_0", "col_1", "col_2", "col_3"]

    def test_set_view_df_volume_is_noop(self):
        pytest.importorskip("cryocat.app.pool")
        from cryocat.app.pool import _payloads as _pp
        state, data_id = _state_with_vol()
        set_view_df(data_id, state)
        assert "dp-view" not in _pp

    def test_clear_view_df_removes_bridge(self):
        pytest.importorskip("cryocat.app.pool")
        from cryocat.app.pool import _payloads as _pp
        state, data_id = _state_with_df()
        set_view_df(data_id, state)
        assert "dp-view" in _pp
        clear_view_df()
        assert "dp-view" not in _pp

    def test_clear_view_df_safe_when_not_set(self):
        pytest.importorskip("cryocat.app.pool")
        from cryocat.app.pool import _payloads as _pp
        _pp.pop("dp-view", None)
        clear_view_df()  # must not raise

    # ── T1g: state immutability ───────────────────────────────────────────────

    def test_state_is_frozen(self):
        state = _empty_state()
        with pytest.raises((AttributeError, TypeError)):
            state.next_id = 99  # type: ignore[misc]

    def test_insert_entry_returns_new_state(self):
        state0 = _empty_state()
        state1, _ = insert_entry(
            state0, pd.DataFrame(), label="x", reader="r", source_path="/x"
        )
        assert state1 is not state0
        assert state0.next_id == 0

    # ── T1h: from_stores / to_stores roundtrip ────────────────────────────────

    def test_from_to_stores_roundtrip(self):
        original = DataPoolState(
            registry={"data-1": {"data_id": "data-1", "label": "L"}}, next_id=7
        )
        reg, nid = original.to_stores()
        restored = DataPoolState.from_stores(reg, nid)
        assert restored.registry == original.registry
        assert restored.next_id == 7

    # ── T1i: _make_entry field completeness ───────────────────────────────────

    def test_make_entry_dataframe_populates_all_optional_fields(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        entry = _make_entry(df, "data-1", "lab", "rk", "/src")
        assert isinstance(entry, DataEntry)
        assert entry.n_rows == 2
        assert entry.columns == ["a", "b"]
        assert entry.shape == (2, 2)
        assert entry.dtype is None

    def test_make_entry_array_populates_shape_and_dtype(self):
        arr = np.ones((5, 3), dtype=np.float64)
        entry = _make_entry(arr, "data-2", "arr", "r2", "/a.npy")
        assert entry.shape == (5, 3)
        assert entry.dtype == "float64"
        assert entry.n_rows == 5

    def test_make_entry_dict_populates_n_rows(self):
        d = {"k1": 1, "k2": 2, "k3": 3}
        entry = _make_entry(d, "data-3", "dd", "r3", "/d.json")
        assert entry.n_rows == 3
        assert entry.kind == "dict"
        assert entry.columns is None


# ═══════════════════════════════════════════════════════════════════════════════
# Dropdown inclusion / exclusion
# ═══════════════════════════════════════════════════════════════════════════════

# Per-format sub-readers that must NOT appear in the dropdown.
_EXCLUDED_FNS = {
    "gctf_read",
    "ctffind4_read",
    "warp_ctf_read",
    "relion_ctffind4_read",
    "imod_com_read",
    "get_data_from_warp_xml",
    "one_value_per_line_read",
}

# Category loaders that MUST appear (W1 keep table).
_EXPECTED_LABELS = {
    "Defocus / CTF",
    "Tilt angles",
    "Total dose",
    "Tomogram dimensions",
    "Z shifts",
    "Euler angles",
    "Indices",
    "FSC curve",
    "CMM markers",
    "Marker coordinates",
    "Generic table",
    "Dictionary / JSON",
    "Volume / map",
}


class TestDropdownCategories:
    """Per-format readers excluded; category loaders present."""

    @pytest.fixture(autouse=True)
    def _load(self):
        from cryocat.app import discovery
        discovery.load_registry()
        self._entries = discovery.readers()

    def test_excluded_readers_absent(self):
        """None of the 7 per-format readers appears in discovery.readers()."""
        present = {e.fn.__name__ for e in self._entries}
        for fn_name in _EXCLUDED_FNS:
            assert fn_name not in present, (
                f"{fn_name!r} is in discovery.readers() but should have been removed "
                "(per-format readers must not appear in the dropdown)."
            )

    def test_expected_labels_present(self):
        """Every expected category loader appears in the dropdown."""
        labels = {e.label for e in self._entries}
        for expected in _EXPECTED_LABELS:
            assert expected in labels, (
                f"Label {expected!r} is missing from discovery.readers(); "
                "add it or verify the @gui_exposed decorator is present."
            )

    def test_no_unexpected_entries(self):
        """No label outside the expected set appears (catches accidental additions)."""
        labels = {e.label for e in self._entries}
        unexpected = labels - _EXPECTED_LABELS
        assert not unexpected, (
            f"Unexpected reader label(s) found: {unexpected}. "
            "Add to _EXPECTED_LABELS or remove the decorator."
        )

    def test_sorted_by_label(self):
        labels = [e.label for e in self._entries]
        assert labels == sorted(labels)


# ═══════════════════════════════════════════════════════════════════════════════
# Format parameter reaches the call
# ═══════════════════════════════════════════════════════════════════════════════

_TEST_DATA = (
    __file__[:__file__.rindex("tests") + len("tests")]
    + "/test_data/TS_017"
)


class TestFormatParameter:
    """Format selector is a parameter, not a separate entry."""

    def test_defocus_file_type_param_is_exposed(self):
        """file_type of defocus_load is not hidden — it appears in the form."""
        from cryocat.app import discovery
        discovery.load_registry()
        entry = next(e for e in discovery.readers() if e.label == "Defocus / CTF")
        assert "file_type" not in entry.hide, (
            "file_type must be exposed (not hidden) so the user can select the format."
        )

    def test_defocus_load_ctffind4_returns_dataframe(self):
        """defocus_load called with file_type='ctffind4' loads the ctffind4 file."""
        import pathlib
        from cryocat.utils.ioutils import defocus_load
        ctffind4_path = pathlib.Path(_TEST_DATA) / "017_ctffind4.txt"
        result = defocus_load(str(ctffind4_path), file_type="ctffind4")
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) >= {"defocus1", "defocus2", "defocus_mean"}

    def test_tlt_load_tlt_file_no_format_param(self):
        """tlt_load accepts a .tlt file without any format parameter."""
        import pathlib
        from cryocat.utils.ioutils import tlt_load
        tlt_path = pathlib.Path(_TEST_DATA) / "017.tlt"
        result = tlt_load(str(tlt_path))
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_tlt_load_xml_via_internal_sniffing(self):
        """tlt_load accepts a Warp XML without a format parameter (internal sniffing)."""
        import pathlib
        from cryocat.utils.ioutils import tlt_load
        xml_path = pathlib.Path(_TEST_DATA) / "017.xml"
        result = tlt_load(str(xml_path))
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_insert_and_get_dataframe(self):
        df = pd.DataFrame({"col": range(5)})
        state = DataPoolState.from_stores(None, None)
        state2, data_id = insert_entry(state, df, label="t2df", reader="r", source_path="/p")
        result = get_payload(data_id, state2)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_insert_and_get_array(self):
        arr = np.random.default_rng(0).random((4, 6))
        state = DataPoolState.from_stores(None, None)
        state2, data_id = insert_entry(state, arr, label="t2arr", reader="r", source_path="/p")
        result = get_payload(data_id, state2)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 6)

    def test_insert_and_get_dict(self):
        d = {"alpha": 1, "beta": 2}
        state = DataPoolState.from_stores(None, None)
        state2, data_id = insert_entry(state, d, label="t2d", reader="r", source_path="/p")
        result = get_payload(data_id, state2)
        assert result == d
