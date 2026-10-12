"""Tests for P2 — event schema construction and validation.

Covers:
- validate_result accepts each canonical form and rejects constraint violations.
- Builder functions produce correctly-shaped dicts.
- invoke_operation emits call events to the stream (success and error paths).
- Events with unexpected extra keys and events missing every optional key are
  tolerated by the builders (unknown-key forward compatibility).
"""
from __future__ import annotations

import pytest

import cryocat.app.session as session
from cryocat.app.logger import dash_logger
from cryocat.app.event import (
    RESULT_BUDGET,
    call_event,
    describe,
    message_event,
    pool_event,
    summarise_result,
    validate_result,
)
from cryocat.app.logger import invoke_operation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset(tmp_path):
    session.close_session()
    dash_logger.buffer.clear()
    session.start_session(tmp_path)
    yield
    session.close_session()
    dash_logger.buffer.clear()


# ---------------------------------------------------------------------------
# validate_result — accepts canonical forms
# ---------------------------------------------------------------------------

def test_validate_result_none():
    validate_result(None)  # must not raise


def test_validate_result_minimal():
    validate_result({"type": "Motl"})


def test_validate_result_full_canonical():
    validate_result({
        "type": "Motl",
        "n_rows": 1204,
        "dropped": 96,
    })


def test_validate_result_accepts_list_value():
    validate_result({"type": "Motl", "columns": ["x", "y", "z"]})  # list is valid JSON


def test_validate_result_accepts_dict_value():
    validate_result({"type": "Motl", "delta": {"n_rows_before": 100, "n_rows_after": 90}})


def test_validate_result_within_budget():
    # A dict whose JSON is well within 4096 bytes must not raise.
    d = {"type": "Motl", "columns": [str(i) for i in range(50)]}
    validate_result(d)


def test_validate_result_none_value_ok():
    # None is a valid JSON scalar
    validate_result({"type": "NoneType", "value": None})


# ---------------------------------------------------------------------------
# validate_result — rejects constraint violations
# ---------------------------------------------------------------------------

def test_validate_result_rejects_over_budget():
    # Build a dict whose JSON exceeds RESULT_BUDGET bytes.
    big_text = "x" * (RESULT_BUDGET + 1)
    with pytest.raises(ValueError, match="budget"):
        validate_result({"type": "Motl", "text": big_text})


def test_validate_result_rejects_ndarray():
    try:
        import numpy as np
        arr = np.array([1, 2, 3])
    except ImportError:
        pytest.skip("numpy not available")
    with pytest.raises(ValueError, match="non-JSON-serializable"):
        validate_result({"arr": arr})


# ---------------------------------------------------------------------------
# summarise_result
# ---------------------------------------------------------------------------

def test_summarise_result_none():
    assert summarise_result(None) is None


def test_summarise_result_string():
    r = summarise_result("/data/out.em")
    assert r["type"] == "str"
    assert r["value"] == "/data/out.em"
    validate_result(r)


def test_summarise_result_int():
    r = summarise_result(42)
    assert r["type"] == "int"
    assert r["value"] == 42
    validate_result(r)


class _FakeMotl:
    def __init__(self, n):
        import pandas as pd
        self.df = pd.DataFrame({"x": range(n)})


def test_summarise_result_motl():
    m = _FakeMotl(500)
    r = summarise_result(m)
    assert r["type"] == "_FakeMotl"
    assert r["n_rows"] == 500
    validate_result(r)


def test_summarise_result_always_passes_validate():
    for val in [None, 1, 1.5, True, "path", _FakeMotl(10)]:
        r = summarise_result(val)
        validate_result(r)  # must not raise


# ---------------------------------------------------------------------------
# Builder functions — canonical form and required keys
# ---------------------------------------------------------------------------

def test_call_event_ok_canonical():
    ev = call_event(
        "cryocat.core.cryomotl.Motl.clean_by_distance",
        {"distance": "20"},
        status="ok",
        imports=[["cryomotl", "from cryocat.core import cryomotl"]],
        receiver="motl_0",
        assign_to="motl_0",
        result={"type": "Motl", "n_rows": 1204},
        duration_s=0.41,
        source="gui",
        tool="editor",
    )
    assert ev["kind"] == "call"
    assert ev["status"] == "ok"
    assert ev["fn"] == "cryocat.core.cryomotl.Motl.clean_by_distance"
    assert ev["kwargs_src"] == {"distance": "20"}
    assert ev["result"]["n_rows"] == 1204
    assert ev["duration_s"] == pytest.approx(0.41, abs=0.001)
    assert "error" not in ev


def test_call_event_error_canonical():
    ev = call_event(
        "cryocat.core.cryomotl.Motl.split_by_feature",
        {"column_names": "'tomo_nr'"},
        status="error",
        error={"type": "KeyError", "msg": "'tomo_nr'", "traceback": "..."},
        duration_s=0.01,
    )
    assert ev["kind"] == "call"
    assert ev["status"] == "error"
    assert ev["error"]["type"] == "KeyError"
    assert "result" not in ev


def test_call_event_minimal_no_optionals():
    ev = call_event("some.fn", {}, status="ok")
    assert ev["kind"] == "call"
    assert ev["fn"] == "some.fn"
    assert ev["receiver"] is None
    assert ev["assign_to"] is None
    assert ev["imports"] == []
    assert "result" not in ev
    assert "error" not in ev
    assert "duration_s" not in ev
    assert "tool" not in ev


def test_pool_event_canonical():
    ev = pool_event("rename", "motl-3", detail={"label": "cleaned"})
    assert ev["kind"] == "pool"
    assert ev["action"] == "rename"
    assert ev["motl_id"] == "motl-3"
    assert ev["detail"] == {"label": "cleaned"}


def test_pool_event_no_detail():
    ev = pool_event("remove", "motl-0")
    assert ev["detail"] == {}


def test_message_event_info():
    ev = message_event("Loading complete.")
    assert ev["kind"] == "message"
    assert ev["text"] == "Loading complete."
    assert ev["level"] == "info"


def test_message_event_error():
    ev = message_event("File not found", level="error")
    assert ev["level"] == "error"


# ---------------------------------------------------------------------------
# Unknown extra keys and missing optional keys — forward compat
# ---------------------------------------------------------------------------

def test_call_event_tolerates_extra_key():
    ev = call_event("fn", {}, status="ok")
    ev["future_field"] = "v2_extension"  # should not cause issues
    assert ev["kind"] == "call"


def test_call_event_missing_optionals_is_valid():
    ev = call_event("fn", {}, status="ok")
    # All optional keys must be absent without error
    for k in ("result", "error", "duration_s", "tool", "receiver", "assign_to"):
        assert ev.get(k) is None or k not in ev


# ---------------------------------------------------------------------------
# invoke_operation — emits call events to the stream
# ---------------------------------------------------------------------------

def _add(a, b):
    return a + b

_add.__module__ = "tests.stub"
_add.__qualname__ = "_add"


def test_invoke_operation_emits_ok_event():
    invoke_operation(_add, {"a": 1, "b": 2})

    evs = session.events()
    call_evs = [e for e in evs if e.get("kind") == "call"]
    assert len(call_evs) == 1
    ev = call_evs[0]
    assert ev["status"] == "ok"
    assert ev["fn"] == "tests.stub._add"
    assert ev["kwargs_src"] == {"a": "1", "b": "2"}
    assert "duration_s" in ev
    assert ev["result"] == {"type": "int", "value": 3}


def test_invoke_operation_emits_error_event():
    def _fail(x):
        raise ValueError("boom")
    _fail.__module__ = "tests.stub"
    _fail.__qualname__ = "_fail"

    with pytest.raises(ValueError, match="boom"):
        invoke_operation(_fail, {"x": "hello"})

    evs = session.events()
    call_evs = [e for e in evs if e.get("kind") == "call"]
    assert len(call_evs) == 1
    ev = call_evs[0]
    assert ev["status"] == "error"
    assert ev["error"]["type"] == "ValueError"
    assert ev["error"]["msg"] == "boom"
    assert "traceback" in ev["error"]
    assert "duration_s" in ev


def test_invoke_operation_event_has_seq_and_t():
    invoke_operation(_add, {"a": 0, "b": 0})
    evs = session.events()
    call_evs = [e for e in evs if e.get("kind") == "call"]
    ev = call_evs[0]
    assert isinstance(ev["seq"], int)
    assert isinstance(ev["t"], str)


def test_invoke_operation_result_passes_validate():
    invoke_operation(_add, {"a": 1, "b": 2})
    evs = session.events()
    call_evs = [e for e in evs if e.get("kind") == "call"]
    result = call_evs[0].get("result")
    validate_result(result)  # must not raise


# ---------------------------------------------------------------------------
# describe — structured result summary
# ---------------------------------------------------------------------------

class _FakeMotlFull:
    def __init__(self, n, ncols=5):
        import pandas as pd
        cols = [f"col_{i}" for i in range(ncols)]
        self.df = pd.DataFrame({c: range(n) for c in cols})

    def get_unique_values(self, col):  # duck-typing marker
        return []


def test_describe_none_returns_minimal():
    d = describe(None)
    assert d == {"type": "None"}


def test_describe_none_with_pool_id():
    d = describe(None, pool_id="motl-1")
    assert d["pool_id"] == "motl-1"


def test_describe_motl_returns_structured_fields():
    m = _FakeMotlFull(26, ncols=20)
    d = describe(m, pool_id="motl-1", label="run1")
    assert d["type"] == "_FakeMotlFull"
    assert d["n_rows"] == 26
    assert d["n_columns"] == 20
    assert len(d["columns"]) == 15  # capped at 15
    assert d["pool_id"] == "motl-1"
    assert d["label"] == "run1"
    validate_result(d)


def test_describe_motl_delta():
    m = _FakeMotlFull(20)
    before = {"n_rows": 26}
    d = describe(m, before=before)
    assert d["delta"] == {"n_rows_before": 26, "n_rows_after": 20}


def test_describe_str():
    d = describe("hello")
    assert d["type"] == "str"
    assert d["value"] == "hello"
    validate_result(d)


def test_describe_int():
    d = describe(42)
    assert d["type"] == "int"
    assert d["value"] == 42
    validate_result(d)


def test_describe_always_within_budget():
    # Wide columns + long str() must not exceed budget.
    m = _FakeMotlFull(10000, ncols=200)
    d = describe(m, pool_id="motl-1", label="x" * 200)
    validate_result(d)  # must not raise


def test_describe_source_field():
    d = describe("ok", source="/data/myfile.em")
    assert d["source"] == "/data/myfile.em"
