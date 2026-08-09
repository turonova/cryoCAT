"""Tests for cryocat.app.pool — server-side payload redesign."""

import json

import pandas as pd
import pytest

from cryocat.app.pool import (
    PoolEntry,
    PoolPayload,
    PoolPayloadMissing,
    PoolState,
    _compute_entry_metadata,
    active_ids,
    clear_payloads,
    default_label,
    get_extra,
    get_rows,
    insert_motl,
    remove_motl,
    replace_motl_rows,
    set_active,
    set_has_tab,
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


# ── _compute_entry_metadata ───────────────────────────────────────────────────────

_META_DF = pd.DataFrame({
    "tomo_id": [1, 1, 2, 2, 3],
    "x": [1.0, 2.0, 3.0, 4.0, 5.0],
    "y": [10.0, 20.0, 30.0, 40.0, 50.0],
    "label": ["a", "b", "c", "d", "e"],  # non-numeric
    "const": [7.0, 7.0, 7.0, 7.0, 7.0],  # constant — excluded from ranges
})


class TestComputeEntryMetadata:
    def test_numeric_columns_detected(self):
        num_cols, _, _ = _compute_entry_metadata(_META_DF)
        assert set(num_cols) == {"tomo_id", "x", "y", "const"}
        assert "label" not in num_cols

    def test_ranges_computed(self):
        _, ranges, _ = _compute_entry_metadata(_META_DF)
        assert "x" in ranges
        assert ranges["x"][0] == 1.0 and ranges["x"][1] == 5.0   # [min, max, step]
        assert "y" in ranges
        assert ranges["y"][0] == 10.0 and ranges["y"][1] == 50.0

    def test_constant_column_excluded_from_ranges(self):
        _, ranges, _ = _compute_entry_metadata(_META_DF)
        assert "const" not in ranges

    def test_tomo_ids_sorted(self):
        _, _, tids = _compute_entry_metadata(_META_DF)
        assert tids == [1, 2, 3]

    def test_no_tomo_id_column_gives_empty(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        _, _, tids = _compute_entry_metadata(df)
        assert tids == []

    def test_empty_df_gives_empty(self):
        num_cols, ranges, tids = _compute_entry_metadata(pd.DataFrame())
        assert num_cols == []
        assert ranges == {}
        assert tids == []

    def test_all_values_json_serialisable(self):
        num_cols, ranges, tids = _compute_entry_metadata(_META_DF)
        json.dumps({"num_cols": num_cols, "ranges": ranges, "tids": tids})


# ── PoolEntry metadata fields ─────────────────────────────────────────────────────

class TestPoolEntryMetadata:
    def test_insert_populates_numeric_columns(self):
        s, mid = insert_motl(_empty(), _META_DF)
        assert "x" in s.registry[mid]["numeric_columns"]
        assert "label" not in s.registry[mid]["numeric_columns"]

    def test_insert_populates_column_ranges(self):
        s, mid = insert_motl(_empty(), _META_DF)
        ranges = s.registry[mid]["column_ranges"]
        assert ranges["x"][0] == 1.0 and ranges["x"][1] == 5.0  # [min, max, step]

    def test_insert_populates_tomo_ids(self):
        s, mid = insert_motl(_empty(), _META_DF)
        assert s.registry[mid]["tomo_ids"] == [1, 2, 3]

    def test_replace_updates_metadata(self):
        s, mid = insert_motl(_empty(), _META_DF)
        new_df = pd.DataFrame({"tomo_id": [5, 5], "x": [100.0, 200.0]})
        s2 = replace_motl_rows(s, mid, new_df)
        assert s2.registry[mid]["tomo_ids"] == [5]
        r = s2.registry[mid]["column_ranges"]["x"]
        assert r[0] == 100.0 and r[1] == 200.0  # [min, max, step]

    def test_metadata_survives_json_roundtrip(self):
        s, mid = insert_motl(_empty(), _META_DF)
        registry, meta, nid = json.loads(json.dumps(s.to_stores()))
        s2 = PoolState.from_stores(registry, meta, nid)
        assert s2.registry[mid]["tomo_ids"] == [1, 2, 3]
        r = s2.registry[mid]["column_ranges"]["x"]
        assert r[0] == 1.0 and r[1] == 5.0  # [min, max, step]

    def test_old_entry_without_fields_defaults_to_empty(self):
        # Simulate a registry dict from before P2 (no new fields)
        old_entry = {
            "label": "Motl 1", "type": "emmotl", "n_rows": 2, "n_columns": 1,
            "columns": ["x"], "active": True, "source_path": None, "revision": 0,
            "has_tab": True,
        }
        # These must not raise — consumers fall back to empty containers
        assert old_entry.get("numeric_columns", []) == []
        assert old_entry.get("column_ranges", {}) == {}
        assert old_entry.get("tomo_ids", []) == []


# ── set_has_tab ───────────────────────────────────────────────────────────────────

class TestSetHasTab:
    def test_set_false(self):
        s, mid = insert_motl(_empty(), _ROWS)
        s2 = set_has_tab(s, mid, False)
        assert s2.registry[mid]["has_tab"] is False

    def test_set_true_after_false(self):
        s, mid = insert_motl(_empty(), _ROWS)
        s = set_has_tab(s, mid, False)
        s = set_has_tab(s, mid, True)
        assert s.registry[mid]["has_tab"] is True

    def test_bumps_revision(self):
        s, mid = insert_motl(_empty(), _ROWS)
        rev0 = s.registry[mid]["revision"]
        s2 = set_has_tab(s, mid, False)
        assert s2.registry[mid]["revision"] == rev0 + 1

    def test_unknown_id_is_noop(self):
        s = _empty()
        assert set_has_tab(s, "motl-99", False) == s

    def test_other_fields_unchanged(self):
        s, mid = insert_motl(_empty(), _ROWS, label="keep-me")
        s2 = set_has_tab(s, mid, False)
        assert s2.registry[mid]["label"] == "keep-me"
        assert s2.registry[mid]["n_rows"] == s.registry[mid]["n_rows"]


# ── Revision policy: every mutation bumps revision ────────────────────────────────
#
# These are the ONLY functions that mutate an existing PoolEntry in the registry.
# Each must bump `revision` so pool-registry watchers re-fire.  If a new mutation
# function is added to pool.py, add it here.

_MUTATION_CASES = [
    (
        "replace_motl_rows",
        lambda s, mid: replace_motl_rows(s, mid, pd.DataFrame({"x": [99]})),
    ),
    (
        "set_active/false",
        lambda s, mid: set_active(s, mid, False),
    ),
    (
        "set_active/true",
        lambda s, mid: set_active(s, mid, True),
    ),
    (
        "set_has_tab/false",
        lambda s, mid: set_has_tab(s, mid, False),
    ),
    (
        "set_has_tab/true",
        lambda s, mid: set_has_tab(s, mid, True),
    ),
]


class TestRevisionPolicy:
    @pytest.mark.parametrize("name,mutate", _MUTATION_CASES, ids=[c[0] for c in _MUTATION_CASES])
    def test_revision_bumps(self, name, mutate):
        s, mid = insert_motl(_empty(), _ROWS)
        rev_before = s.registry[mid]["revision"]
        s2 = mutate(s, mid)
        assert s2.registry[mid]["revision"] == rev_before + 1, (
            f"{name}: expected revision {rev_before + 1}, got {s2.registry[mid]['revision']}"
        )

    @pytest.mark.parametrize("name,mutate", _MUTATION_CASES, ids=[c[0] for c in _MUTATION_CASES])
    def test_revision_monotone_across_chained_mutations(self, name, mutate):
        s, mid = insert_motl(_empty(), _ROWS)
        for _ in range(5):
            rev_before = s.registry[mid]["revision"]
            s = mutate(s, mid)
            assert s.registry[mid]["revision"] == rev_before + 1

    @pytest.mark.parametrize("name,mutate", _MUTATION_CASES, ids=[c[0] for c in _MUTATION_CASES])
    def test_other_entries_revision_unchanged(self, name, mutate):
        s, mid1 = insert_motl(_empty(), _ROWS)
        s, mid2 = insert_motl(s, _ROWS)
        rev2_before = s.registry[mid2]["revision"]
        s2 = mutate(s, mid1)
        assert s2.registry[mid2]["revision"] == rev2_before
