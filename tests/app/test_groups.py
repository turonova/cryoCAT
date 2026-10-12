"""Tests for pool group reducers."""

import pytest

from cryocat.app.pool import (
    GroupState,
    PoolState,
    clear_payloads,
    create_group,
    delete_group,
    add_to_group,
    remove_from_group,
    reorder_group,
    rename_group,
    group_members,
    purge_motl_from_groups,
    insert_motl,
    natural_sort_key,
)

import pandas as pd

_ROWS = pd.DataFrame({"x": [1, 2]})


@pytest.fixture(autouse=True)
def _clean():
    clear_payloads()
    yield
    clear_payloads()


def _empty_g() -> GroupState:
    return GroupState.from_store(None)


def _empty_p() -> PoolState:
    return PoolState.from_stores(None)


# ── GroupState serialisation ──────────────────────────────────────────────────

class TestGroupStateRoundTrip:
    def test_empty_from_store(self):
        g = GroupState.from_store(None)
        assert g.groups == {}
        assert g.next_id == 0

    def test_to_store_round_trip(self):
        g, _ = create_group(_empty_g(), ["motl-1", "motl-2"], label="test")
        data = g.to_store()
        g2 = GroupState.from_store(data)
        assert g2.groups == g.groups
        assert g2.next_id == g.next_id


# ── create_group ─────────────────────────────────────────────────────────────

class TestCreateGroup:
    def test_first_group_id_is_group_1(self):
        _, gid = create_group(_empty_g(), [])
        assert gid == "group-1"

    def test_next_id_increments(self):
        g, _ = create_group(_empty_g(), [])
        assert g.next_id == 1

    def test_custom_label(self):
        g, gid = create_group(_empty_g(), [], label="my_run")
        assert g.groups[gid]["label"] == "my_run"

    def test_default_label(self):
        g, gid = create_group(_empty_g(), [])
        assert g.groups[gid]["label"] == "Group 1"

    def test_member_order_preserved(self):
        members = ["motl-3", "motl-1", "motl-2"]
        g, gid = create_group(_empty_g(), members)
        assert g.groups[gid]["members"] == members

    def test_ids_sequential_never_reused(self):
        g = _empty_g()
        g, g1 = create_group(g, [])
        g, g2 = create_group(g, [])
        g = delete_group(g, g1)
        g, g3 = create_group(g, [])
        assert g1 == "group-1"
        assert g2 == "group-2"
        assert g3 == "group-3"


# ── rename_group ─────────────────────────────────────────────────────────────

class TestRenameGroup:
    def test_rename_updates_label(self):
        g, gid = create_group(_empty_g(), [])
        g = rename_group(g, gid, "new_name")
        assert g.groups[gid]["label"] == "new_name"

    def test_rename_unknown_is_noop(self):
        g = _empty_g()
        assert rename_group(g, "group-99", "x") == g


# ── add_to_group / remove_from_group ─────────────────────────────────────────

class TestAddRemoveMember:
    def test_add_appends_to_end(self):
        g, gid = create_group(_empty_g(), ["motl-1"])
        g = add_to_group(g, gid, "motl-2")
        assert g.groups[gid]["members"] == ["motl-1", "motl-2"]

    def test_add_duplicate_is_noop(self):
        g, gid = create_group(_empty_g(), ["motl-1"])
        g2 = add_to_group(g, gid, "motl-1")
        assert g2 == g

    def test_add_to_unknown_group_is_noop(self):
        g = _empty_g()
        assert add_to_group(g, "group-99", "motl-1") == g

    def test_remove_from_group(self):
        g, gid = create_group(_empty_g(), ["motl-1", "motl-2", "motl-3"])
        g = remove_from_group(g, gid, "motl-2")
        assert g.groups[gid]["members"] == ["motl-1", "motl-3"]

    def test_remove_absent_is_noop(self):
        g, gid = create_group(_empty_g(), ["motl-1"])
        g2 = remove_from_group(g, gid, "motl-99")
        assert g2 == g

    def test_remove_from_unknown_group_is_noop(self):
        g = _empty_g()
        assert remove_from_group(g, "group-99", "motl-1") == g


# ── reorder_group ─────────────────────────────────────────────────────────────

class TestReorderGroup:
    def test_reorder(self):
        g, gid = create_group(_empty_g(), ["motl-1", "motl-2", "motl-3"])
        g = reorder_group(g, gid, ["motl-3", "motl-1", "motl-2"])
        assert g.groups[gid]["members"] == ["motl-3", "motl-1", "motl-2"]

    def test_reorder_unknown_is_noop(self):
        g = _empty_g()
        assert reorder_group(g, "group-99", []) == g


# ── delete_group ─────────────────────────────────────────────────────────────

class TestDeleteGroup:
    def test_delete_removes_group(self):
        g, gid = create_group(_empty_g(), ["motl-1"])
        g = delete_group(g, gid)
        assert gid not in g.groups

    def test_delete_does_not_delete_motls(self):
        p = _empty_p()
        p, mid = insert_motl(p, _ROWS)
        g, gid = create_group(_empty_g(), [mid])
        g = delete_group(g, gid)
        assert mid in p.registry

    def test_delete_unknown_is_noop(self):
        g = _empty_g()
        assert delete_group(g, "group-99") == g

    def test_motl_moved_from_first_group_when_created_in_second(self):
        """Creating a group that includes motl-1 removes it from any previous group."""
        g = _empty_g()
        g, g1 = create_group(g, ["motl-1", "motl-2"])
        g, g2 = create_group(g, ["motl-1"])  # motl-1 leaves g1
        assert "motl-1" not in g.groups[g1]["members"]
        assert "motl-2" in g.groups[g1]["members"]
        assert "motl-1" in g.groups[g2]["members"]

    def test_motl_moved_from_first_group_when_added_to_second(self):
        """add_to_group moves the motl out of its previous group."""
        g = _empty_g()
        g, g1 = create_group(g, ["motl-1"])
        g, g2 = create_group(g, [])
        g = add_to_group(g, g2, "motl-1")
        assert "motl-1" not in g.groups[g1]["members"]
        assert "motl-1" in g.groups[g2]["members"]


# ── group_members ─────────────────────────────────────────────────────────────

class TestGroupMembers:
    def test_returns_ordered_members(self):
        g, gid = create_group(_empty_g(), ["motl-3", "motl-1", "motl-2"])
        assert group_members(g, gid) == ["motl-3", "motl-1", "motl-2"]

    def test_returns_empty_for_unknown(self):
        assert group_members(_empty_g(), "group-99") == []


# ── purge_motl_from_groups ────────────────────────────────────────────────────

class TestPurgeMotlFromGroups:
    def test_removes_from_all_groups(self):
        g = _empty_g()
        g, g1 = create_group(g, ["motl-1", "motl-2"])
        g, g2 = create_group(g, ["motl-2", "motl-3"])
        g = purge_motl_from_groups(g, "motl-2")
        assert "motl-2" not in g.groups[g1]["members"]
        assert "motl-2" not in g.groups[g2]["members"]
        assert "motl-1" in g.groups[g1]["members"]
        assert "motl-3" in g.groups[g2]["members"]

    def test_purge_absent_motl_is_noop(self):
        g, gid = create_group(_empty_g(), ["motl-1"])
        g2 = purge_motl_from_groups(g, "motl-99")
        assert g2.groups == g.groups


# ── natural_sort_key ──────────────────────────────────────────────────────────

class TestNaturalSortKey:
    def test_run_2_before_run_10(self):
        names = ["run_10", "run_2", "run_1"]
        assert sorted(names, key=natural_sort_key) == ["run_1", "run_2", "run_10"]

    def test_plain_lexicographic_fails(self):
        names = ["run_10", "run_2", "run_1"]
        assert sorted(names) != ["run_1", "run_2", "run_10"]

    def test_mixed_stems(self):
        names = ["tomo_003", "tomo_10", "tomo_002", "tomo_1"]
        assert sorted(names, key=natural_sort_key) == ["tomo_1", "tomo_002", "tomo_003", "tomo_10"]
