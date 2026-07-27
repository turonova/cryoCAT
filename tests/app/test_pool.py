"""Tests for cryocat.app.pool — T6 acceptance suite."""

import json
import pytest

from cryocat.app.pool import (
    PoolEntry,
    PoolState,
    default_label,
    insert_motl,
    remove_motl,
    set_active,
    get_rows,
    active_ids,
)


_EMPTY = PoolState.from_stores(None, None, None, None, None)
_ROWS = [{"x": 1}, {"x": 2}]


class TestInsertMotl:
    def test_first_insert_yields_motl_0(self):
        state, mid = insert_motl(_EMPTY, _ROWS)
        assert mid == "motl-0"

    def test_next_id_increments(self):
        state, _ = insert_motl(_EMPTY, _ROWS)
        assert state.next_id == 1

    def test_entry_in_all_four_stores(self):
        state, mid = insert_motl(_EMPTY, _ROWS)
        assert mid in state.registry
        assert mid in state.motls
        assert mid in state.extra
        assert mid in state.meta

    def test_insert_without_extra_stores_explicit_none(self):
        """motlsink regression: missing extra/meta must NOT be absent — store None."""
        state, mid = insert_motl(_EMPTY, _ROWS)
        assert mid in state.extra
        assert state.extra[mid] is None

    def test_insert_without_meta_stores_explicit_none(self):
        state, mid = insert_motl(_EMPTY, _ROWS)
        assert mid in state.meta
        assert state.meta[mid] is None

    def test_custom_label(self):
        state, mid = insert_motl(_EMPTY, _ROWS, label="my label")
        assert state.registry[mid]["label"] == "my label"

    def test_default_label(self):
        state, mid = insert_motl(_EMPTY, _ROWS)
        assert state.registry[mid]["label"] == default_label(0)

    def test_motl_type_stored(self):
        state, mid = insert_motl(_EMPTY, _ROWS, motl_type="stopgap")
        assert state.registry[mid]["type"] == "stopgap"

    def test_n_rows_stored(self):
        state, mid = insert_motl(_EMPTY, _ROWS)
        assert state.registry[mid]["n_rows"] == 2

    def test_active_true_by_default(self):
        state, mid = insert_motl(_EMPTY, _ROWS)
        assert state.registry[mid]["active"] is True

    def test_rows_stored(self):
        state, mid = insert_motl(_EMPTY, _ROWS)
        assert state.motls[mid] == _ROWS

    def test_extra_stored(self):
        extra = [{"e": 1}]
        state, mid = insert_motl(_EMPTY, _ROWS, extra=extra)
        assert state.extra[mid] == extra

    def test_meta_stored(self):
        meta = {"script_expr": "Motl.load(...)"}
        state, mid = insert_motl(_EMPTY, _ROWS, meta=meta)
        assert state.meta[mid] == meta


class TestIdsNeverReused:
    def test_ids_sequential(self):
        s0 = _EMPTY
        s1, m0 = insert_motl(s0, _ROWS)
        s2, m1 = insert_motl(s1, _ROWS)
        s3, m2 = insert_motl(s2, _ROWS)
        assert m0 == "motl-0"
        assert m1 == "motl-1"
        assert m2 == "motl-2"

    def test_remove_then_insert_yields_next_id(self):
        """Ids are never reused: insert ×3, remove motl-1, insert → motl-3."""
        s = _EMPTY
        s, _ = insert_motl(s, _ROWS)   # motl-0
        s, _ = insert_motl(s, _ROWS)   # motl-1
        s, _ = insert_motl(s, _ROWS)   # motl-2
        s = remove_motl(s, "motl-1")
        s, m3 = insert_motl(s, _ROWS)
        assert m3 == "motl-3"


class TestRemoveMotl:
    def test_remove_clears_all_stores(self):
        s, mid = insert_motl(_EMPTY, _ROWS)
        s = remove_motl(s, mid)
        assert mid not in s.registry
        assert mid not in s.motls
        assert mid not in s.extra
        assert mid not in s.meta

    def test_remove_unknown_id_is_noop(self):
        s = remove_motl(_EMPTY, "motl-99")
        assert s == _EMPTY

    def test_remove_does_not_decrement_next_id(self):
        s, mid = insert_motl(_EMPTY, _ROWS)
        n = s.next_id
        s = remove_motl(s, mid)
        assert s.next_id == n


class TestSetActive:
    def test_set_inactive(self):
        s, mid = insert_motl(_EMPTY, _ROWS)
        s = set_active(s, mid, False)
        assert s.registry[mid]["active"] is False

    def test_set_active_back(self):
        s, mid = insert_motl(_EMPTY, _ROWS)
        s = set_active(s, mid, False)
        s = set_active(s, mid, True)
        assert s.registry[mid]["active"] is True

    def test_set_active_unknown_id_is_noop(self):
        original = _EMPTY
        result = set_active(original, "motl-99", False)
        assert result == original


class TestGetRows:
    def test_returns_rows(self):
        s, mid = insert_motl(_EMPTY, _ROWS)
        assert get_rows(s, mid) == _ROWS

    def test_missing_id_returns_empty_list(self):
        assert get_rows(_EMPTY, "motl-0") == []


class TestActiveIds:
    def test_preserves_insertion_order(self):
        s = _EMPTY
        s, _ = insert_motl(s, _ROWS)   # motl-0
        s, _ = insert_motl(s, _ROWS)   # motl-1
        s, _ = insert_motl(s, _ROWS)   # motl-2
        assert active_ids(s) == ["motl-0", "motl-1", "motl-2"]

    def test_inactive_excluded(self):
        s = _EMPTY
        s, _ = insert_motl(s, _ROWS)   # motl-0
        s, _ = insert_motl(s, _ROWS)   # motl-1
        s = set_active(s, "motl-0", False)
        assert active_ids(s) == ["motl-1"]


class TestJsonRoundTrip:
    def test_to_stores_survives_json(self):
        s, _ = insert_motl(_EMPTY, _ROWS, meta={"k": 1})
        stores = s.to_stores()
        recovered = json.loads(json.dumps(stores, default=list))
        reg, motls, extra, meta, nid = recovered
        s2 = PoolState.from_stores(reg, motls, extra, meta, nid)
        assert s2 == s

    def test_round_trip_identity(self):
        s, _ = insert_motl(_EMPTY, _ROWS)
        assert PoolState.from_stores(*s.to_stores()) == s


class TestDefaultLabel:
    def test_label_numbering_matches_id(self):
        assert default_label(0) == "Motl 1"
        assert default_label(1) == "Motl 2"
        assert default_label(9) == "Motl 10"
