"""Tests for Part F — @-variable resolution in generate_kwargs.

Covers:
- @x resolves to a bound pool motl
- @x resolves to a bound console local
- @x is rejected with a reason when x is not bound
- type mismatch is reported in plain language (not a traceback)
- generate_kwargs passes through non-@ values unchanged
"""
from __future__ import annotations

import pytest
import pandas as pd

from cryocat.core.cryomotl import Motl


# ── Fixtures ───────────────────────────────────────────────────────────────────

class _FakePool:
    """Minimal PoolState stand-in for tests that don't exercise pool I/O."""
    def __init__(self, registry=None):
        self.registry = registry or {}
        self.next_id = len(self.registry)


@pytest.fixture
def pool_with_motl():
    """Pytest fixture yielding (pool_state, motl) with provenance; cleans up after.

    Patches both pool.get_rows and execute._pool_get_rows so build_namespace
    works correctly regardless of whether execute.py was imported earlier.
    """
    from cryocat.app import provenance as _prov
    from cryocat.app import session as _session
    import cryocat.app.pool as _pool_mod
    import cryocat.app.console.execute as _exec_mod

    _prov.clear()
    _session.start_session()

    pool = _FakePool({"motl-1": {"active": True, "label": "test_motl"}})
    pool.next_id = 1
    _prov.record("motl-1", 1)

    df = pd.DataFrame({"subtomo_id": [1.0, 2.0], "tomo_id": [1.0, 1.0]})
    for col in Motl.motl_columns:
        if col not in df.columns:
            df[col] = 0.0

    _pool_orig = _pool_mod.get_rows
    _exec_orig = _exec_mod._pool_get_rows

    def _fake_get_rows(mid):
        if mid == "motl-1":
            return df
        raise _pool_mod.PoolPayloadMissing(mid)

    _pool_mod.get_rows = _fake_get_rows
    _exec_mod._pool_get_rows = _fake_get_rows
    motl_obj = Motl(df)
    motl_obj._pool_motl_id = "motl-1"

    yield pool, motl_obj

    _pool_mod.get_rows = _pool_orig
    _exec_mod._pool_get_rows = _exec_orig
    _prov.clear()


# ── Tests for _bound_session_vars ─────────────────────────────────────────────

def test_bound_vars_includes_pool_motl_with_provenance(pool_with_motl):
    pool, _ = pool_with_motl

    from cryocat.app.apputils import _bound_session_vars
    bound = _bound_session_vars(pool)
    assert "motl_1" in bound
    assert isinstance(bound["motl_1"], Motl)


def test_bound_vars_excludes_pool_motl_without_provenance():
    from cryocat.app import provenance as _prov
    _prov.clear()
    pool = _FakePool({"motl-99": {"active": True}})
    # No provenance recorded for motl-99.

    import cryocat.app.pool as _pool_mod
    _orig = _pool_mod.get_rows
    _pool_mod.get_rows = lambda mid: (_ for _ in ()).throw(_pool_mod.PoolPayloadMissing(mid))
    try:
        from cryocat.app.apputils import _bound_session_vars
        bound = _bound_session_vars(pool)
        assert "motl_99" not in bound
    finally:
        _pool_mod.get_rows = _orig
        _prov.clear()


def test_bound_vars_includes_console_locals():
    from cryocat.app.console.execute import _CONSOLE_LOCALS
    from cryocat.app import provenance as _prov
    _prov.clear()
    _CONSOLE_LOCALS.clear()
    _CONSOLE_LOCALS["my_result"] = [1, 2, 3]

    pool = _FakePool()
    try:
        from cryocat.app.apputils import _bound_session_vars
        bound = _bound_session_vars(pool)
        assert "my_result" in bound
        assert bound["my_result"] == [1, 2, 3]
    finally:
        _CONSOLE_LOCALS.clear()
        _prov.clear()


# ── Tests for _resolve_at_var ─────────────────────────────────────────────────

def test_resolve_at_var_bound_pool_motl(pool_with_motl):
    pool, motl_obj = pool_with_motl

    from cryocat.app.apputils import _resolve_at_var
    result = _resolve_at_var("motl_1", "text", pool)
    assert isinstance(result, Motl)
    assert len(result.df) == 2


def test_resolve_at_var_console_local():
    from cryocat.app.console.execute import _CONSOLE_LOCALS
    from cryocat.app import provenance as _prov
    _prov.clear()
    _CONSOLE_LOCALS.clear()
    _CONSOLE_LOCALS["my_list"] = [1, 2, 3]

    pool = _FakePool()
    try:
        from cryocat.app.apputils import _resolve_at_var
        result = _resolve_at_var("my_list", "text", pool)
        assert result == [1, 2, 3]
    finally:
        _CONSOLE_LOCALS.clear()
        _prov.clear()


def test_resolve_at_var_unbound_raises_with_reason():
    from cryocat.app import provenance as _prov
    _prov.clear()
    pool = _FakePool()

    from cryocat.app.apputils import _resolve_at_var
    with pytest.raises(ValueError, match="not a bound session variable"):
        _resolve_at_var("nonexistent", "text", pool)


def test_resolve_at_var_unbound_lists_available():
    from cryocat.app.console.execute import _CONSOLE_LOCALS
    from cryocat.app import provenance as _prov
    _prov.clear()
    _CONSOLE_LOCALS.clear()
    _CONSOLE_LOCALS["alpha"] = 42
    _CONSOLE_LOCALS["beta"] = "path"

    pool = _FakePool()
    try:
        from cryocat.app.apputils import _resolve_at_var
        with pytest.raises(ValueError) as exc_info:
            _resolve_at_var("gamma", "text", pool)
        msg = str(exc_info.value)
        # Message must mention at least one available name.
        assert "alpha" in msg or "beta" in msg
    finally:
        _CONSOLE_LOCALS.clear()
        _prov.clear()


def test_resolve_at_var_type_mismatch_path_field():
    from cryocat.app.console.execute import _CONSOLE_LOCALS
    from cryocat.app import provenance as _prov
    _prov.clear()
    _CONSOLE_LOCALS.clear()
    _CONSOLE_LOCALS["my_df"] = pd.DataFrame({"x": [1, 2]})

    pool = _FakePool()
    try:
        from cryocat.app.apputils import _resolve_at_var
        with pytest.raises(ValueError, match="expects a file path"):
            _resolve_at_var("my_df", "MapSource", pool)
    finally:
        _CONSOLE_LOCALS.clear()
        _prov.clear()


def test_resolve_at_var_type_mismatch_number_field():
    from cryocat.app.console.execute import _CONSOLE_LOCALS
    from cryocat.app import provenance as _prov
    _prov.clear()
    _CONSOLE_LOCALS.clear()
    _CONSOLE_LOCALS["my_str"] = "hello"

    pool = _FakePool()
    try:
        from cryocat.app.apputils import _resolve_at_var
        with pytest.raises(ValueError, match="expects a number"):
            _resolve_at_var("my_str", "float", pool)
    finally:
        _CONSOLE_LOCALS.clear()
        _prov.clear()


def test_resolve_at_var_plain_language_error_no_traceback():
    """Error message must be a plain string — no Python traceback prefixes."""
    from cryocat.app import provenance as _prov
    _prov.clear()
    pool = _FakePool()

    from cryocat.app.apputils import _resolve_at_var
    with pytest.raises(ValueError) as exc_info:
        _resolve_at_var("x", "text", pool)
    msg = str(exc_info.value)
    assert "Traceback" not in msg
    assert "File " not in msg
    assert "Error" not in msg or msg.startswith("@")  # no library-style prefixes


# ── Tests for generate_kwargs with @-sigil ────────────────────────────────────

def test_generate_kwargs_at_sigil_resolved():
    from cryocat.app.console.execute import _CONSOLE_LOCALS
    from cryocat.app import provenance as _prov
    _prov.clear()
    _CONSOLE_LOCALS.clear()
    _CONSOLE_LOCALS["my_val"] = 999

    pool = _FakePool()
    # "str" is the valid TYPE_HANDLERS tag for string parameters.
    ids_ = [{"type": "op-param", "owner": "", "param": "count", "tag": "str"}]
    values = ["@my_val"]

    try:
        from cryocat.app.apputils import generate_kwargs
        result = generate_kwargs(ids_, values, pool_state=pool)
        assert result["count"] == 999
    finally:
        _CONSOLE_LOCALS.clear()
        _prov.clear()


def test_generate_kwargs_without_at_unchanged():
    """Non-@ values are parsed normally regardless of pool_state."""
    pool = _FakePool()
    ids_ = [{"type": "op-param", "owner": "", "param": "label", "tag": "str"}]
    values = ["hello"]

    from cryocat.app.apputils import generate_kwargs
    result = generate_kwargs(ids_, values, pool_state=pool)
    assert result["label"] == "hello"


def test_generate_kwargs_no_pool_state_at_passthrough():
    """Without pool_state, @-values are passed to the tag's parse function as-is."""
    ids_ = [{"type": "op-param", "owner": "", "param": "label", "tag": "str"}]
    values = ["@not_resolved"]

    from cryocat.app.apputils import generate_kwargs
    result = generate_kwargs(ids_, values)
    assert result["label"] == "@not_resolved"
