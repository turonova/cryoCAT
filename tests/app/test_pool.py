"""Tests for cryocat.app.pool — server-side payload redesign."""

import json

import pandas as pd
import pytest

from cryocat.app.pool import (
    PoolEntry,
    PoolPayload,
    PoolPayloadMissing,
    PoolState,
    active_ids,
    clear_payloads,
    default_label,
    get_extra,
    get_rows,
    insert_motl,
    remove_motl,
    replace_motl_rows,
    set_active,
)


_ROWS = pd.DataFrame({"x": [1, 2]})
_EXTRA = pd.DataFrame({"e": [10, 20]})


@pytest.fixture(autouse=True)
def _clean():
    clear_payloads()
    yield
    clear_payloads()


def _empty() -> PoolState:
    return PoolState.from_stores(None)


# ── PoolState.from_stores ────────────────────────────────────────────────────────

class TestFromStores:
    def test_zero_extra_args(self):
        s = PoolState.from_stores(None)
        assert s.registry == {}
        assert s.meta == {}
        assert s.next_id == 0

    def test_two_extra_args(self):
        s = PoolState.from_stores({"motl-1": {}}, {"motl-1": None}, 1)
        assert s.next_id == 1

    def test_legacy_five_arg_discards_motls_extra(self):
        # 5-arg form (registry, motls, extra, meta, next_id) — motls/extra silently dropped
        s = PoolState.from_stores({"motl-1": {}}, {"motl-1": []}, {"motl-1": None}, {}, 1)
        assert s.next_id == 1
        assert s.registry == {"motl-1": {}}

    def test_invalid_arg_count_raises(self):
        with pytest.raises(TypeError):
            PoolState.from_stores({}, {})  # 3 args total = 2 extra = invalid


# ── to_stores ────────────────────────────────────────────────────────────────────

class TestToStores:
    def test_returns_three_tuple(self):
        s = _empty()
        result = s.to_stores()
        assert len(result) == 3

    def test_json_roundtrip_no_row_data(self):
        s, mid = insert_motl(_empty(), _ROWS, meta={"k": 1})
        registry, meta, nid = json.loads(json.dumps(s.to_stores()))
        s2 = PoolState.from_stores(registry, meta, nid)
        assert s2.next_id == s.next_id
        assert mid in s2.registry
        # actual row values must NOT appear in stores (column names in handle are fine)
        serialised = json.dumps(s.to_stores())
        assert '"x": 1' not in serialised and '"x": 2' not in serialised

    def test_round_trip_identity(self):
        s, _ = insert_motl(_empty(), _ROWS)
        s2 = PoolState.from_stores(*s.to_stores())
        assert s2 == s


# ── insert_motl ──────────────────────────────────────────────────────────────────

class TestInsertMotl:
    def test_first_id_is_motl_1(self):
        _, mid = insert_motl(_empty(), _ROWS)
        assert mid == "motl-1"

    def test_next_id_increments(self):
        s, _ = insert_motl(_empty(), _ROWS)
        assert s.next_id == 1

    def test_entry_in_registry_and_meta(self):
        s, mid = insert_motl(_empty(), _ROWS)
        assert mid in s.registry
        assert mid in s.meta

    def test_meta_stored(self):
        meta = {"script_expr": "Motl.load(...)"}
        s, mid = insert_motl(_empty(), _ROWS, meta=meta)
        assert s.meta[mid] == meta

    def test_custom_label(self):
        s, mid = insert_motl(_empty(), _ROWS, label="my label")
        assert s.registry[mid]["label"] == "my label"

    def test_default_label(self):
        s, mid = insert_motl(_empty(), _ROWS)
        assert s.registry[mid]["label"] == default_label(0)

    def test_motl_type_stored(self):
        s, mid = insert_motl(_empty(), _ROWS, motl_type="stopgap")
        assert s.registry[mid]["type"] == "stopgap"

    def test_n_rows_stored(self):
        s, mid = insert_motl(_empty(), _ROWS)
        assert s.registry[mid]["n_rows"] == 2

    def test_active_true_by_default(self):
        s, mid = insert_motl(_empty(), _ROWS)
        assert s.registry[mid]["active"] is True

    def test_revision_starts_at_zero(self):
        s, mid = insert_motl(_empty(), _ROWS)
        assert s.registry[mid]["revision"] == 0

    def test_payload_stored_server_side(self):
        _, mid = insert_motl(_empty(), _ROWS)
        df = get_rows(mid)
        assert list(df.columns) == ["x"]
        assert len(df) == 2

    def test_extra_stored_server_side(self):
        _, mid = insert_motl(_empty(), _ROWS, extra=_EXTRA)
        df = get_extra(mid)
        assert df is not None
        assert list(df.columns) == ["e"]

    def test_no_extra_gives_none(self):
        _, mid = insert_motl(_empty(), _ROWS)
        assert get_extra(mid) is None

    def test_accepts_list_of_dicts(self):
        _, mid = insert_motl(_empty(), [{"x": 1}, {"x": 2}])
        df = get_rows(mid)
        assert len(df) == 2

    def test_source_path_stored(self):
        s, mid = insert_motl(_empty(), _ROWS, source_path="/data/motl.em")
        assert s.registry[mid]["source_path"] == "/data/motl.em"


# ── IDs never reused ─────────────────────────────────────────────────────────────

class TestIdsNeverReused:
    def test_sequential(self):
        s = _empty()
        s, m1 = insert_motl(s, _ROWS)
        s, m2 = insert_motl(s, _ROWS)
        s, m3 = insert_motl(s, _ROWS)
        assert [m1, m2, m3] == ["motl-1", "motl-2", "motl-3"]

    def test_remove_then_insert_yields_next_id(self):
        s = _empty()
        s, _ = insert_motl(s, _ROWS)   # motl-1
        s, _ = insert_motl(s, _ROWS)   # motl-2
        s, _ = insert_motl(s, _ROWS)   # motl-3
        s = remove_motl(s, "motl-2")
        s, m4 = insert_motl(s, _ROWS)
        assert m4 == "motl-4"


# ── remove_motl ──────────────────────────────────────────────────────────────────

class TestRemoveMotl:
    def test_removes_from_registry_and_meta(self):
        s, mid = insert_motl(_empty(), _ROWS)
        s = remove_motl(s, mid)
        assert mid not in s.registry
        assert mid not in s.meta

    def test_evicts_server_side_payload(self):
        _, mid = insert_motl(_empty(), _ROWS)
        s, _ = insert_motl(_empty(), _ROWS)
        s = remove_motl(s, mid)
        with pytest.raises(PoolPayloadMissing):
            get_rows(mid)

    def test_unknown_id_is_noop(self):
        s = _empty()
        assert remove_motl(s, "motl-99") == s

    def test_next_id_unchanged(self):
        s, mid = insert_motl(_empty(), _ROWS)
        n = s.next_id
        s = remove_motl(s, mid)
        assert s.next_id == n


# ── set_active ───────────────────────────────────────────────────────────────────

class TestSetActive:
    def test_set_inactive(self):
        s, mid = insert_motl(_empty(), _ROWS)
        s = set_active(s, mid, False)
        assert s.registry[mid]["active"] is False

    def test_set_active_back(self):
        s, mid = insert_motl(_empty(), _ROWS)
        s = set_active(s, mid, False)
        s = set_active(s, mid, True)
        assert s.registry[mid]["active"] is True

    def test_bumps_revision(self):
        s, mid = insert_motl(_empty(), _ROWS)
        rev0 = s.registry[mid]["revision"]
        s = set_active(s, mid, False)
        assert s.registry[mid]["revision"] == rev0 + 1

    def test_unknown_id_is_noop(self):
        s = _empty()
        assert set_active(s, "motl-99", False) == s


# ── replace_motl_rows ────────────────────────────────────────────────────────────

class TestReplaceMotlRows:
    def test_updates_server_side_payload(self):
        s, mid = insert_motl(_empty(), _ROWS)
        new_rows = pd.DataFrame({"x": [9, 8, 7]})
        replace_motl_rows(s, mid, new_rows)
        assert len(get_rows(mid)) == 3

    def test_bumps_revision(self):
        s, mid = insert_motl(_empty(), _ROWS)
        rev0 = s.registry[mid]["revision"]
        s2 = replace_motl_rows(s, mid, pd.DataFrame({"x": [9]}))
        assert s2.registry[mid]["revision"] == rev0 + 1

    def test_preserves_extra(self):
        s, mid = insert_motl(_empty(), _ROWS, extra=_EXTRA)
        replace_motl_rows(s, mid, pd.DataFrame({"x": [9]}))
        extra = get_extra(mid)
        assert extra is not None and list(extra.columns) == ["e"]

    def test_unknown_id_is_noop(self):
        s = _empty()
        assert replace_motl_rows(s, "motl-99", _ROWS) == s

    def test_updates_n_rows_in_registry(self):
        s, mid = insert_motl(_empty(), _ROWS)
        s2 = replace_motl_rows(s, mid, pd.DataFrame({"x": [1, 2, 3]}))
        assert s2.registry[mid]["n_rows"] == 3


# ── get_rows ─────────────────────────────────────────────────────────────────────

class TestGetRows:
    def test_returns_dataframe(self):
        _, mid = insert_motl(_empty(), _ROWS)
        df = get_rows(mid)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_raises_when_missing(self):
        with pytest.raises(PoolPayloadMissing):
            get_rows("motl-99")

    def test_error_includes_source_path_when_state_given(self):
        s, mid = insert_motl(_empty(), _ROWS, source_path="/data/motl.em")
        clear_payloads()
        with pytest.raises(PoolPayloadMissing, match="/data/motl.em"):
            get_rows(mid, state=s)


# ── get_extra ────────────────────────────────────────────────────────────────────

class TestGetExtra:
    def test_returns_none_when_no_extra(self):
        _, mid = insert_motl(_empty(), _ROWS)
        assert get_extra(mid) is None

    def test_returns_dataframe_when_extra_present(self):
        _, mid = insert_motl(_empty(), _ROWS, extra=_EXTRA)
        df = get_extra(mid)
        assert df is not None
        assert len(df) == 2

    def test_returns_none_for_unknown_id(self):
        assert get_extra("motl-99") is None


# ── active_ids ───────────────────────────────────────────────────────────────────

class TestActiveIds:
    def test_preserves_insertion_order(self):
        s = _empty()
        s, _ = insert_motl(s, _ROWS)
        s, _ = insert_motl(s, _ROWS)
        s, _ = insert_motl(s, _ROWS)
        assert active_ids(s) == ["motl-1", "motl-2", "motl-3"]

    def test_inactive_excluded(self):
        s = _empty()
        s, _ = insert_motl(s, _ROWS)
        s, _ = insert_motl(s, _ROWS)
        s = set_active(s, "motl-1", False)
        assert active_ids(s) == ["motl-2"]


# ── clear_payloads ───────────────────────────────────────────────────────────────

class TestClearPayloads:
    def test_all_payloads_gone_after_clear(self):
        _, mid = insert_motl(_empty(), _ROWS)
        clear_payloads()
        with pytest.raises(PoolPayloadMissing):
            get_rows(mid)

    def test_clears_idempotent(self):
        clear_payloads()
        clear_payloads()


# ── default_label ────────────────────────────────────────────────────────────────

class TestDefaultLabel:
    def test_numbering(self):
        assert default_label(0) == "Motl 1"
        assert default_label(1) == "Motl 2"
        assert default_label(9) == "Motl 10"
