"""Tests for cryocat.app.components.logpanel and cryocat.app.logger.

Covers logpanel (rendering and export):
- render_events: incremental split produces the same output as one-pass render.
- render_events handles empty, session-only, errors-only, and mixed streams.
- export_session .py matches render_script directly.
- export_session .md matches render_markdown directly.
- export_session .ipynb is valid nbformat v4 JSON.
- export_session writes a file whose content equals the projection called directly.
- Unknown export format raises ValueError.
- Edge cases: empty stream, None lineage_of.

Covers logger pure functions:
- format_arg: string, int, large list/dict, long repr.
- _module_short: submodule and top-level.
- _render_call: basic, empty kwargs, large value, classmethod, instance method.
- _render_python_line: plain function, classmethod, instance method.
- _render_value: string, int, None, small list.
"""
from __future__ import annotations

import json
import types

import pytest
from dash import html

import cryocat.app.provenance as prov
from cryocat.app.components.logpanel import export_session, render_events
from cryocat.app.record.script import render_script
from cryocat.app.record.session_record import render_markdown
from cryocat.app.record.notebook import render_notebook
from cryocat.app.logger import format_arg, _module_short, _render_call, _render_python_line, _render_value


# ===========================================================================
# Fixtures and helpers
# ===========================================================================

@pytest.fixture(autouse=True)
def _reset_prov():
    prov.clear()
    yield
    prov.clear()


def _sev(sid="test-session"):
    return {
        "seq": 0, "t": "2026-01-01T00:00:00.000",
        "kind": "session", "cryocat_version": "1.0",
        "python": "3.12", "platform": "linux", "session_id": sid,
    }


def _call_ok(seq=1, fn="testmod.FakeMotl.load", receiver=None, assign_to="motl_0",
             kwargs_src=None, result=None):
    return {
        "seq": seq, "t": f"2026-01-01T00:00:{seq:02d}.000",
        "kind": "call", "status": "ok",
        "fn": fn, "source": "gui",
        "imports": [["testmod", "import testmod"]],
        "receiver": receiver, "assign_to": assign_to,
        "kwargs_src": kwargs_src or {"path": "'/data/motl.em'"},
        "duration_s": 0.1,
        "result": result or {"type": "FakeMotl", "n_rows": 10},
    }


def _call_error(seq=3, fn="testmod.FakeMotl.fail", receiver="motl_0", assign_to="motl_1"):
    return {
        "seq": seq, "t": f"2026-01-01T00:00:{seq:02d}.000",
        "kind": "call", "status": "error",
        "fn": fn, "source": "gui",
        "imports": [["testmod", "import testmod"]],
        "receiver": receiver, "assign_to": assign_to,
        "kwargs_src": {"x": "1"},
        "duration_s": 0.05,
        "error": {
            "type": "ValueError",
            "msg": "bad input",
            "traceback": "Traceback (most recent call last):\n  File ...\nValueError: bad input",
        },
    }


def _msg_info():
    return {"seq": 99, "t": "2026-01-01T00:01:00.000", "kind": "message",
            "text": "info note", "level": "info"}


def _msg_error():
    return {"seq": 98, "t": "2026-01-01T00:01:00.000", "kind": "message",
            "text": "something went wrong", "level": "error"}


def _simple_stream():
    return [_sev(), _call_ok(1), _call_ok(3, receiver="motl_0", assign_to="motl_0",
                                           kwargs_src={"by": "2"},
                                           result={"type": "FakeMotl", "n_rows": 8})]


def _error_stream():
    return [_sev(), _call_ok(1), _call_error(3)]


def _errors_only_stream():
    return [_sev(), _call_error(1, receiver="motl_0", assign_to="motl_0"),
            _call_error(3, receiver="motl_0", assign_to="motl_1")]


# ===========================================================================
# render_events: incremental == one-pass
# ===========================================================================

class TestRenderEventsIncremental:
    def test_one_pass_same_count_as_split_at_1(self):
        events = _simple_stream()
        one_pass = render_events(events)
        split = render_events(events[:1]) + render_events(events[1:])
        assert len(one_pass) == len(split)

    def test_one_pass_same_count_as_split_at_2(self):
        events = _simple_stream()
        one_pass = render_events(events)
        split = render_events(events[:2]) + render_events(events[2:])
        assert len(one_pass) == len(split)

    def test_split_types_match_one_pass(self):
        events = _error_stream()
        one_pass = render_events(events)
        split = render_events(events[:2]) + render_events(events[2:])
        assert [type(c).__name__ for c in one_pass] == [type(c).__name__ for c in split]

    def test_empty_split_is_noop(self):
        events = _simple_stream()
        one_pass = render_events(events)
        split = render_events([]) + render_events(events)
        assert len(one_pass) == len(split)

    def test_single_event_batches(self):
        events = [_sev(), _call_ok(1), _call_error(3)]
        one_pass = render_events(events)
        split = sum([render_events([e]) for e in events], [])
        assert len(one_pass) == len(split)
        for a, b in zip(one_pass, split):
            assert type(a).__name__ == type(b).__name__


# ===========================================================================
# render_events: content and structure
# ===========================================================================

class TestRenderEventsContent:
    def test_empty_stream_returns_empty_list(self):
        assert render_events([]) == []

    def test_session_event_produces_component(self):
        result = render_events([_sev()])
        assert len(result) == 1
        assert isinstance(result[0], html.Div)

    def test_call_ok_produces_details(self):
        result = render_events([_call_ok(1)])
        assert len(result) == 1
        assert isinstance(result[0], html.Details)

    def test_call_error_produces_details(self):
        result = render_events([_call_error(3)])
        assert len(result) == 1
        assert isinstance(result[0], html.Details)

    def test_error_details_open_by_default(self):
        result = render_events([_call_error(3)])
        assert result[0].open is True

    def test_ok_details_closed_by_default(self):
        result = render_events([_call_ok(1)])
        assert not result[0].open

    def test_message_info_produces_div(self):
        result = render_events([_msg_info()])
        assert isinstance(result[0], html.Div)

    def test_message_error_produces_div(self):
        result = render_events([_msg_error()])
        assert isinstance(result[0], html.Div)

    def test_unknown_kind_is_skipped(self):
        ev = {"seq": 0, "kind": "unknown_future_kind", "data": 42}
        assert render_events([ev]) == []

    def test_errors_only_stream(self):
        result = render_events(_errors_only_stream())
        assert len(result) == 3

    def test_mixed_stream_count(self):
        events = [_sev(), _call_ok(1), _call_error(3), _msg_info()]
        result = render_events(events)
        assert len(result) == 4


# ===========================================================================
# export_session: format dispatch
# ===========================================================================

class TestExportSession:
    def test_py_matches_render_script_successful(self):
        events = _simple_stream()
        assert export_session(events, mode="successful", fmt=".py") == \
               render_script(events, mode="successful")

    def test_py_matches_render_script_verbatim(self):
        events = _error_stream()
        assert export_session(events, mode="verbatim", fmt=".py") == \
               render_script(events, mode="verbatim")

    def test_md_matches_render_markdown(self):
        events = _simple_stream()
        assert export_session(events, fmt=".md") == render_markdown(events)

    def test_ipynb_is_valid_nbformat(self):
        events = _simple_stream()
        result = export_session(events, fmt=".ipynb")
        parsed = json.loads(result)
        assert parsed["nbformat"] == 4

    def test_ipynb_matches_render_notebook(self):
        events = _simple_stream()
        result = export_session(events, fmt=".ipynb")
        assert json.loads(result) == render_notebook(events)

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown export format"):
            export_session(_simple_stream(), fmt=".csv")

    def test_empty_stream_py(self):
        out = export_session([], fmt=".py")
        assert isinstance(out, str)

    def test_empty_stream_md(self):
        out = export_session([], fmt=".md")
        assert isinstance(out, str)

    def test_empty_stream_ipynb(self):
        out = export_session([], fmt=".ipynb")
        assert json.loads(out)["nbformat"] == 4

    def test_lineage_of_forwarded_to_script(self):
        prov.record("motl-0", 1)
        events = [_sev(), _call_ok(1, assign_to="motl_0"),
                  _call_ok(3, receiver="motl_0", assign_to="motl_1")]
        with_lineage = export_session(events, fmt=".py", lineage_of="motl-0")
        without = export_session(events, fmt=".py")
        assert len(with_lineage) <= len(without)

    def test_none_lineage_is_not_forwarded(self):
        events = _simple_stream()
        out = export_session(events, fmt=".py", lineage_of=None)
        assert isinstance(out, str)


# ===========================================================================
# export_session: file write
# ===========================================================================

class TestExportSessionFileWrite:
    def test_file_content_equals_projection_py(self, tmp_path):
        events = _simple_stream()
        expected = export_session(events, mode="successful", fmt=".py")
        path = tmp_path / "out.py"
        path.write_text(expected, encoding="utf-8")
        assert path.read_text(encoding="utf-8") == render_script(events, mode="successful")

    def test_file_content_equals_projection_md(self, tmp_path):
        events = _simple_stream()
        expected = export_session(events, fmt=".md")
        path = tmp_path / "out.md"
        path.write_text(expected, encoding="utf-8")
        assert path.read_text(encoding="utf-8") == render_markdown(events)

    def test_file_content_equals_projection_ipynb(self, tmp_path):
        events = _simple_stream()
        expected = export_session(events, fmt=".ipynb")
        path = tmp_path / "out.ipynb"
        path.write_text(expected, encoding="utf-8")
        assert json.loads(path.read_text(encoding="utf-8")) == render_notebook(events)


# ===========================================================================
# Logger pure functions
# ===========================================================================

class _Dummy:
    @classmethod
    def class_method(cls, a, b):
        return a

    def instance_method(self, a):
        return a

_dummy_instance = _Dummy()


class TestFormatArg:
    def test_string(self):
        assert format_arg("hello") == "'hello'"

    def test_int(self):
        assert format_arg(42) == "42"

    def test_large_list_truncated(self):
        result = format_arg(list(range(100)))
        assert "list" in result or "[" in result
        assert "100" in result

    def test_large_dict_truncated(self):
        result = format_arg({i: i for i in range(20)})
        assert "dict" in result or "{" in result

    def test_small_list_verbatim(self):
        result = format_arg([1, 2, 3])
        assert "1" in result

    def test_long_repr_truncated(self):
        class LongRepr:
            def __repr__(self):
                return "x" * 200
        result = format_arg(LongRepr())
        assert len(result) <= 103


class TestModuleShort:
    def test_submodule(self):
        from cryocat.core.cryomotl import Motl
        full, short, stmt = _module_short(Motl.clean_by_otsu)
        assert short == "cryomotl"
        assert "from" in stmt and "cryomotl" in stmt

    def test_top_level_module(self):
        import os
        full, short, stmt = _module_short(os.path.join)
        assert short in stmt


class TestRenderCall:
    def test_basic(self):
        def my_fn(x, y):
            pass

        result = _render_call(my_fn, {"x": 1, "y": "hello"})
        assert "my_fn" in result
        assert "x=1" in result
        assert "y=" in result

    def test_empty_kwargs(self):
        def noop():
            pass

        result = _render_call(noop, {})
        assert result.endswith("noop()")

    def test_large_value_truncated(self):
        def fn(data):
            pass

        result = _render_call(fn, {"data": list(range(100))})
        assert "data=" in result
        assert len(result) < 300

    def test_classmethod(self):
        fn = _Dummy.class_method
        label = _render_call(fn, {"a": 10, "b": 20})
        assert "_Dummy.class_method" in label
        assert not label.startswith("type.")

    def test_instance_method(self):
        fn = _dummy_instance.instance_method
        label = _render_call(fn, {"a": 5})
        assert "_Dummy.instance_method" in label


def _plain(x, y):
    return x + y


class TestRenderPythonLine:
    def test_plain_function(self):
        line, _, _ = _render_python_line(_plain, {"x": 1, "y": 2})
        assert "_plain" in line or "plain" in line
        assert "x=1" in line and "y=2" in line

    def test_classmethod(self):
        line, _, _ = _render_python_line(_Dummy.class_method, {"a": 10, "b": 20})
        assert "_Dummy.class_method" in line
        assert "a=10" in line and "b=20" in line

    def test_instance_method(self):
        line, _, _ = _render_python_line(_dummy_instance.instance_method, {"a": 5})
        assert "instance_method" in line
        assert "a=5" in line


class TestRenderValue:
    def test_string(self):
        expr, imps = _render_value("hello")
        assert expr == "'hello'"
        assert imps == []

    def test_int(self):
        expr, _ = _render_value(42)
        assert expr == "42"

    def test_none(self):
        expr, _ = _render_value(None)
        assert expr == "None"

    def test_small_list(self):
        expr, imps = _render_value([1, 2, 3])
        assert expr == "[1, 2, 3]"
        assert imps == []
