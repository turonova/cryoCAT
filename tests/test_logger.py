"""Tests for cryocat.app.logger rendering helpers."""

import pytest
from cryocat.app.logger import _render_call, _render_python_line, _render_value, dash_logger


# ── Fixture callables ──────────────────────────────────────────────────────────

def plain_function(x, y):
    return x + y


class _Dummy:
    @classmethod
    def class_method(cls, a, b):
        return a

    def instance_method(self, a):
        return a


_dummy_instance = _Dummy()


# ── _render_call ───────────────────────────────────────────────────────────────

def test_render_call_plain_function():
    label = _render_call(plain_function, {"x": 1, "y": 2})
    assert label == "plain_function(x=1, y=2)"


def test_render_call_classmethod():
    fn = _Dummy.class_method
    label = _render_call(fn, {"a": 10, "b": 20})
    # Must NOT produce "type.class_method" — __qualname__ gives the right class name
    assert label == "_Dummy.class_method(a=10, b=20)"
    assert not label.startswith("type.")


def test_render_call_instance_method():
    fn = _dummy_instance.instance_method
    label = _render_call(fn, {"a": 5})
    assert label == "_Dummy.instance_method(a=5)"


# ── _render_python_line ────────────────────────────────────────────────────────

def test_render_python_line_plain_function():
    line, imports = _render_python_line(plain_function, {"x": 1, "y": 2})
    # module is "tests.test_logger" → short is "test_logger"
    assert "plain_function" in line
    assert "x=1" in line and "y=2" in line


def test_render_python_line_classmethod():
    fn = _Dummy.class_method
    line, imports = _render_python_line(fn, {"a": 10, "b": 20})
    # Must produce something like "test_logger._Dummy.class_method(a=10, b=20)"
    assert "_Dummy.class_method" in line
    assert "a=10" in line
    assert "b=20" in line


def test_render_python_line_instance_method():
    fn = _dummy_instance.instance_method
    line, imports = _render_python_line(fn, {"a": 5})
    # Instance method renders as <receiver_expr>.instance_method(a=5)
    assert "instance_method" in line
    assert "a=5" in line


# ── _render_value — primitives ─────────────────────────────────────────────────

def test_render_value_string():
    expr, imps = _render_value("hello")
    assert expr == "'hello'"
    assert imps == []


def test_render_value_int():
    expr, imps = _render_value(42)
    assert expr == "42"


def test_render_value_none():
    expr, imps = _render_value(None)
    assert expr == "None"


def test_render_value_small_list():
    expr, imps = _render_value([1, 2, 3])
    assert expr == "[1, 2, 3]"
    assert imps == []


# ── _render_value — motl side-table ────────────────────────────────────────────

class _FakeMotl:
    """Minimal duck-type for a Motl object (just enough attributes for detection)."""
    def get_unique_values(self, col):  # noqa: D401
        return []

    df = None  # just needs to exist; _render_value checks hasattr only


def test_render_value_motl_without_source_fallback():
    m = _FakeMotl()
    dash_logger._motl_sources.pop(id(m), None)  # ensure no stale entry
    expr, imps = _render_value(m)
    assert "None" in expr  # placeholder because no path, no source
    assert any("cryomotl" in stmt for _, stmt in imps)


def test_render_value_motl_with_source():
    m = _FakeMotl()
    expected = "cryomotl.Motl.load('test.em', 'emmotl')"
    dash_logger.record_motl_source(m, expected)
    expr, imps = _render_value(m)
    assert expr == expected
    assert any("cryomotl" in stmt for _, stmt in imps)
    dash_logger._motl_sources.pop(id(m), None)  # clean up


def test_render_value_motl_depth_exceeded_skips_source():
    m = _FakeMotl()
    dash_logger.record_motl_source(m, "cryomotl.Motl.load('deep.em')")
    from cryocat.app.logger import _MAX_MOTL_DEPTH
    expr, _ = _render_value(m, depth=_MAX_MOTL_DEPTH)  # at or beyond limit
    assert "deep.em" not in expr  # source is suppressed at max depth
    dash_logger._motl_sources.pop(id(m), None)


def test_dash_logger_record_and_get_motl_source():
    m = _FakeMotl()
    dash_logger.record_motl_source(m, "some_expr")
    assert dash_logger.get_motl_source(m) == "some_expr"
    other = _FakeMotl()
    assert dash_logger.get_motl_source(other) is None
    dash_logger._motl_sources.pop(id(m), None)
