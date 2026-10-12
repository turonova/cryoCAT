"""Tests for cryocat.app.provenance — pool-id-keyed provenance table.

Covers:
- bind is deterministic and injective.
- producer on an unknown id returns None.
- Forgotten ids stay forgotten.
- Lineage of a recorded chain has the producers in ascending seq order.
- Per-kind namespace: twist and descriptor bindings.
- Pool stamp: get_motl stamps _pool_motl_id.
- Cross-kind registry: variable names never collide across motl/twist/desc.
- Pool purity: twist and desc entries must not appear in the motl pool.
- get_motl stamps pool identity correctly.
"""
from __future__ import annotations

import pandas as pd
import pytest

import cryocat.app.provenance as prov
from cryocat.app import pool
from cryocat.core.cryomotl import Motl


@pytest.fixture(autouse=True)
def _reset():
    prov.clear()
    pool.clear_payloads()
    yield
    prov.clear()
    pool.clear_payloads()


# ===========================================================================
# bind — deterministic and injective
# ===========================================================================

def test_bind_replaces_hyphen():
    assert prov.bind("motl-0") == "motl_0"
    assert prov.bind("motl-3") == "motl_3"
    assert prov.bind("motl-12") == "motl_12"


def test_bind_deterministic():
    assert prov.bind("motl-5") == prov.bind("motl-5")


def test_bind_injective():
    ids = ["motl-0", "motl-1", "motl-2", "motl-10", "motl-100"]
    names = [prov.bind(i) for i in ids]
    assert len(names) == len(set(names))


def test_bind_no_side_effects():
    prov.bind("motl-99")
    assert prov.producer("motl-99") is None


# ===========================================================================
# record / producer
# ===========================================================================

def test_producer_unknown_id_is_none():
    assert prov.producer("motl-0") is None


def test_record_then_producer():
    prov.record("motl-0", 1)
    assert prov.producer("motl-0") == 1


def test_record_overwrites():
    prov.record("motl-0", 1)
    prov.record("motl-0", 7)
    assert prov.producer("motl-0") == 7


def test_record_multiple_entries():
    prov.record("motl-0", 1)
    prov.record("motl-1", 3)
    prov.record("motl-2", 5)
    assert prov.producer("motl-0") == 1
    assert prov.producer("motl-1") == 3
    assert prov.producer("motl-2") == 5


# ===========================================================================
# var_for
# ===========================================================================

def test_var_for_unrecorded_is_none():
    assert prov.var_for("motl-0") is None


def test_var_for_recorded():
    prov.record("motl-3", 7)
    assert prov.var_for("motl-3") == "motl_3"


def test_var_for_agrees_with_bind():
    prov.record("motl-5", 2)
    assert prov.var_for("motl-5") == prov.bind("motl-5")


# ===========================================================================
# forget
# ===========================================================================

def test_forget_removes_entry():
    prov.record("motl-0", 1)
    prov.forget("motl-0")
    assert prov.producer("motl-0") is None
    assert prov.var_for("motl-0") is None


def test_forget_unknown_id_is_noop():
    prov.forget("motl-99")


def test_forget_leaves_others_intact():
    prov.record("motl-0", 1)
    prov.record("motl-1", 3)
    prov.forget("motl-0")
    assert prov.producer("motl-1") == 3


def test_forgotten_ids_stay_forgotten():
    prov.record("motl-0", 1)
    prov.forget("motl-0")
    prov.record("motl-1", 5)
    assert prov.producer("motl-0") is None


# ===========================================================================
# Lineage of a chain — producers in ascending seq order
# ===========================================================================

def test_chain_producers_in_order():
    prov.record("motl-0", 1)
    prov.record("motl-1", 3)
    prov.record("motl-2", 5)

    chain = ["motl-0", "motl-1", "motl-2"]
    seqs = [prov.producer(mid) for mid in chain]
    assert seqs == sorted(seqs)
    assert seqs == [1, 3, 5]


def test_chain_var_names_in_order():
    prov.record("motl-0", 1)
    prov.record("motl-1", 3)
    prov.record("motl-2", 5)

    vars_ = [prov.var_for(mid) for mid in ["motl-0", "motl-1", "motl-2"]]
    assert vars_ == ["motl_0", "motl_1", "motl_2"]


# ===========================================================================
# clear
# ===========================================================================

def test_clear_resets_all():
    prov.record("motl-0", 1)
    prov.record("motl-1", 3)
    prov.clear()
    assert prov.producer("motl-0") is None
    assert prov.producer("motl-1") is None


# ===========================================================================
# Per-kind namespace: twist and descriptor bindings
# ===========================================================================

def test_bind_twist_kind():
    assert prov.bind("twist-1") == "twist_1"
    assert prov.bind("twist-5") == "twist_5"


def test_bind_desc_kind():
    assert prov.bind("desc-1") == "desc_1"
    assert prov.bind("desc-3") == "desc_3"


def test_kinds_do_not_collide():
    assert prov.bind("motl-1") != prov.bind("twist-1")
    assert prov.bind("motl-1") != prov.bind("desc-1")
    assert prov.bind("twist-1") != prov.bind("desc-1")


def test_twist_var_for_after_record():
    prov.record("twist-1", 10)
    assert prov.var_for("twist-1") == "twist_1"


def test_desc_var_for_after_record():
    prov.record("desc-2", 15)
    assert prov.var_for("desc-2") == "desc_2"


def test_consecutive_ids_within_kind_are_unique():
    twist_vars = [prov.bind(f"twist-{i}") for i in range(1, 6)]
    assert len(twist_vars) == len(set(twist_vars))

    desc_vars = [prov.bind(f"desc-{i}") for i in range(1, 6)]
    assert len(desc_vars) == len(set(desc_vars))


# ===========================================================================
# Pool stamp: get_motl stamps _pool_motl_id
# ===========================================================================

def test_pool_stamp_enables_var_resolution():
    """A Motl returned by pool.get_motl carries _pool_motl_id so
    invoke_operation can resolve it to its variable name via var_for."""
    cols = Motl.motl_columns
    df = pd.DataFrame([{c: 0.0 for c in cols}])
    pool._payloads["test-motl-1"] = pool.PoolPayload(rows=df, extra=None)

    try:
        motl = pool.get_motl("test-motl-1")
        assert hasattr(motl, "_pool_motl_id")
        assert motl._pool_motl_id == "test-motl-1"

        prov.record("test-motl-1", 99)
        assert prov.var_for("test-motl-1") == "test_motl_1"
    finally:
        pool._payloads.pop("test-motl-1", None)
        pool.clear_payloads()


# ===========================================================================
# Cross-kind variable names never collide
# ===========================================================================

def test_motl_twist_desc_names_never_collide():
    for n in range(1, 6):
        motl_var = prov.bind(f"motl-{n}")
        twist_var = prov.bind(f"twist-{n}")
        desc_var = prov.bind(f"desc-{n}")
        assert motl_var != twist_var
        assert motl_var != desc_var
        assert twist_var != desc_var


def test_all_kind_prefixes_produce_distinct_names():
    ids = [
        "motl-1", "motl-2",
        "twist-1", "twist-2",
        "desc-1", "desc-2",
        "data-1",
        "complex-1",
    ]
    names = [prov.bind(i) for i in ids]
    assert len(names) == len(set(names))


def test_twist_consecutive_ids_unique():
    names = [prov.bind(f"twist-{i}") for i in range(1, 11)]
    assert len(names) == len(set(names))


def test_desc_consecutive_ids_unique():
    names = [prov.bind(f"desc-{i}") for i in range(1, 11)]
    assert len(names) == len(set(names))


# ===========================================================================
# Pool purity: twist and desc entries must not appear in motl pool
# ===========================================================================

def _motl_df(n_rows: int = 1) -> pd.DataFrame:
    return pd.DataFrame([{c: 0.0 for c in Motl.motl_columns} for _ in range(n_rows)])


def test_twist_not_inserted_into_pool():
    prov.record("twist-1", 10)
    assert pool._payloads.get("twist-1") is None


def test_desc_not_inserted_into_pool():
    prov.record("desc-1", 20)
    assert pool._payloads.get("desc-1") is None


def test_pool_contains_only_motl_ids_after_mixed_session():
    from cryocat.app.pool import PoolState, insert_motl

    df = _motl_df(3)
    state = PoolState(registry={}, meta={}, next_id=0)
    new_state, motl_id = insert_motl(state, df, label="test motl")

    prov.record("twist-1", 5)
    prov.record("desc-1", 10)

    payload_keys = set(pool._payloads.keys())
    non_motl = {k for k in payload_keys if not k.startswith("motl_")}
    assert not non_motl
    assert motl_id in payload_keys


# ===========================================================================
# get_motl stamps pool identity correctly
# ===========================================================================

def test_get_motl_stamps_pool_id():
    from cryocat.app.pool import get_motl

    df = _motl_df(2)
    pool._payloads["motl-42"] = pool.PoolPayload(rows=df, extra=None)

    motl = get_motl("motl-42")
    assert hasattr(motl, "_pool_motl_id")
    assert motl._pool_motl_id == "motl-42"


def test_get_motl_tagged_resolves_via_var_for():
    from cryocat.app.pool import get_motl

    df = _motl_df(2)
    pool._payloads["motl-7"] = pool.PoolPayload(rows=df, extra=None)
    motl = get_motl("motl-7")

    prov.record("motl-7", 99)

    var = prov.var_for(motl._pool_motl_id)
    assert var == "motl_7"
