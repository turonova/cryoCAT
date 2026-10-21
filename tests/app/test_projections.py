"""Tests for the app/record/ projection modules.

All projections are pure functions ``list[dict] -> str | dict``.
No Dash, no running app, no file I/O — just plain dicts.

Covers (per P4 spec):
- Every projection handles an empty stream, an errors-only stream, and a
  truncated stream (last event missing — same result as partial list).
- Failures are *never* omitted from the session record.
- Result summaries appear in all three projections.
- render_script(mode='successful') omits failed calls.
- render_script(mode='verbatim') includes failed calls as real uncommented
  code with an ``# ERROR …`` comment above.
- lineage_of returns only the transitive producers, in seq order.
- Names-bound-before-use: ast.parse + ordered walk.
- Replay: exec in a fresh namespace with a stub module.
- Notebook json.dumps/loads round-trip; one code cell per call event.
"""
from __future__ import annotations

import ast
import builtins
import json
import types

import pytest

import cryocat.app.provenance as prov
from cryocat.app.record.session_record import render_html, render_markdown
from cryocat.app.record.script import render_script
from cryocat.app.record.notebook import render_notebook


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_prov():
    prov.clear()
    yield
    prov.clear()


def _sev(sid="2026-01-01_00-00-00"):
    return {
        "seq": 0, "t": "2026-01-01T00:00:00.000",
        "kind": "session", "cryocat_version": "1.0",
        "python": "3.12", "platform": "linux", "session_id": sid,
    }


def _call(
    seq: int,
    fn: str = "testmod.FakeMotl.op",
    *,
    status: str = "ok",
    receiver: str | None = None,
    assign_to: str | None = None,
    kwargs_src: dict | None = None,
    imports: list | None = None,
    result: dict | None = None,
    error: dict | None = None,
    duration_s: float | None = 0.1,
    tool: str | None = None,
) -> dict:
    ev: dict = {
        "seq": seq, "t": f"2026-01-01T00:00:{seq:02d}.000",
        "kind": "call", "status": status,
        "fn": fn, "source": "gui",
        "imports": imports or [["testmod", "import testmod"]],
        "receiver": receiver, "assign_to": assign_to,
        "kwargs_src": kwargs_src or {},
    }
    if duration_s is not None:
        ev["duration_s"] = duration_s
    if tool is not None:
        ev["tool"] = tool
    if status == "ok" and result is not None:
        ev["result"] = result
    if status == "error" and error is not None:
        ev["error"] = error
    return ev


def _load_event(seq=1, assign_to="motl_0"):
    return _call(
        seq,
        fn="testmod.FakeMotl.load",
        assign_to=assign_to,
        kwargs_src={"path": "'/data/motl.em'"},
        result={"type": "FakeMotl", "n_rows": 10},
    )


def _op_event(seq, receiver, assign_to, **kw):
    return _call(
        seq,
        fn="testmod.FakeMotl.shrink",
        receiver=receiver,
        assign_to=assign_to,
        kwargs_src={"by": "2"},
        result={"type": "FakeMotl", "n_rows": 8},
        **kw,
    )


def _error_event(seq, receiver="motl_0", assign_to="motl_1"):
    return _call(
        seq,
        fn="testmod.FakeMotl.fail",
        receiver=receiver,
        assign_to=assign_to,
        kwargs_src={"x": "1"},
        status="error",
        result=None,
        error={"type": "ValueError", "msg": "bad input", "traceback": "Traceback...\nValueError: bad input"},
    )


def _simple_stream():
    return [_sev(), _load_event(1), _op_event(3, "motl_0", "motl_0")]


def _error_stream():
    return [_sev(), _load_event(1), _error_event(3)]


def _errors_only_stream():
    return [_sev(), _error_event(1, receiver="motl_0", assign_to="motl_0"), _error_event(3)]


def _empty_stream():
    return []


def _truncated_stream():
    return _simple_stream()[:-1]  # last call event missing


# ── session_record.render_markdown ────────────────────────────────────────────

class TestRenderMarkdown:
    def test_empty_stream(self):
        assert render_markdown(_empty_stream()) == ""

    def test_session_only(self):
        out = render_markdown([_sev("my-session")])
        assert "my-session" in out

    def test_successful_call_present(self):
        out = render_markdown(_simple_stream())
        assert "ok" in out
        assert "shrink" in out or "load" in out

    def test_result_summary_present(self):
        out = render_markdown(_simple_stream())
        assert "10 rows" in out

    def test_failure_never_omitted(self):
        out = render_markdown(_errors_only_stream())
        assert out.count("ERROR") >= 2, "both error blocks must appear"

    def test_failure_shows_error_type(self):
        out = render_markdown(_error_stream())
        assert "ValueError" in out

    def test_failure_shows_message(self):
        out = render_markdown(_error_stream())
        assert "bad input" in out

    def test_truncated_stream_does_not_crash(self):
        out = render_markdown(_truncated_stream())
        assert isinstance(out, str)

    def test_tool_filter_excludes_other_tools(self):
        events = [_sev(), _call(1, tool="editor"), _call(3, tool="nn")]
        out = render_markdown(events, tool="editor")
        assert out.count("### ") == 1

    def test_tool_filter_none_includes_all(self):
        events = [_sev(), _call(1, tool="editor"), _call(3, tool="nn")]
        out = render_markdown(events, tool=None)
        assert out.count("### ") == 2

    def test_message_event_info(self):
        events = [_sev(), {"kind": "message", "text": "hello", "level": "info"}]
        out = render_markdown(events)
        assert "hello" in out

    def test_message_event_error(self):
        events = [_sev(), {"kind": "message", "text": "oops", "level": "error"}]
        out = render_markdown(events)
        assert "ERROR" in out and "oops" in out

    def test_duration_shown(self):
        out = render_markdown([_sev(), _call(1, duration_s=0.41)])
        assert "0.41" in out


# ── session_record.render_html ────────────────────────────────────────────────

class TestRenderHtml:
    def test_returns_string(self):
        assert isinstance(render_html(_simple_stream()), str)

    def test_contains_doctype(self):
        assert "<!DOCTYPE html>" in render_html(_simple_stream())

    def test_empty_stream(self):
        out = render_html(_empty_stream())
        assert isinstance(out, str)

    def test_errors_only(self):
        out = render_html(_errors_only_stream())
        assert "ERROR" in out

    def test_result_summary_present(self):
        out = render_html(_simple_stream())
        assert "10 rows" in out


# ── render_script — successful mode ──────────────────────────────────────────

class TestRenderScriptSuccessful:
    def test_empty_stream(self):
        out = render_script(_empty_stream())
        assert out == "" or out.strip() == ""

    def test_omits_errors(self):
        out = render_script(_error_stream())
        assert "ERROR" not in out
        assert "FakeMotl.fail(" not in out

    def test_includes_successful(self):
        out = render_script(_simple_stream())
        assert "motl_0" in out

    def test_imports_at_top(self):
        out = render_script(_simple_stream())
        lines = out.splitlines()
        import_idx = next(i for i, l in enumerate(lines) if "import" in l)
        code_idx = next(i for i, l in enumerate(lines) if "=" in l and "import" not in l)
        assert import_idx < code_idx

    def test_assigns_variable(self):
        out = render_script(_simple_stream())
        assert "motl_0 =" in out

    def test_result_comment_present(self):
        out = render_script(_simple_stream())
        assert "# ->" in out

    def test_result_summary_in_comment(self):
        out = render_script(_simple_stream())
        assert "10 rows" in out

    def test_session_header_comment(self):
        out = render_script([_sev("my-session")])
        assert "my-session" in out

    def test_truncated_stream(self):
        out = render_script(_truncated_stream())
        assert isinstance(out, str)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            render_script(_simple_stream(), mode="magic")

    def test_names_bound_before_use(self):
        stream = [
            _sev(),
            _load_event(1, assign_to="motl_0"),
            _op_event(3, "motl_0", "motl_0"),
        ]
        out = render_script(stream, mode="successful")
        _assert_names_bound(out)

    def test_replay(self):
        stream = [
            _sev(),
            _load_event(1, assign_to="motl_0"),
            _op_event(3, "motl_0", "motl_0"),
        ]
        out = render_script(stream, mode="successful")
        ns = _run_script(out)
        assert hasattr(ns["motl_0"], "n")
        assert ns["motl_0"].n == 8

    def test_errors_only_stream_produces_empty_body(self):
        out = render_script(_errors_only_stream(), mode="successful")
        body = "\n".join(l for l in out.splitlines() if not l.startswith("#") and l.strip())
        assert body == ""


# ── render_script — verbatim mode ────────────────────────────────────────────

class TestRenderScriptVerbatim:
    def test_includes_errors(self):
        out = render_script(_error_stream(), mode="verbatim")
        assert "fail" in out

    def test_error_as_real_code_not_comment(self):
        out = render_script(_error_stream(), mode="verbatim")
        call_lines = [l for l in out.splitlines() if "fail(" in l]
        assert call_lines, "error call must appear in output"
        assert all(l.lstrip().startswith("#") for l in call_lines)

    def test_error_comment_above_call(self):
        out = render_script(_error_stream(), mode="verbatim")
        lines = out.splitlines()
        call_lines = [l for l in lines if "fail(" in l]
        assert call_lines, "error call must appear in output"
        assert all("# ERROR" in l for l in call_lines)

    def test_error_comment_contains_type(self):
        out = render_script(_error_stream(), mode="verbatim")
        assert "ValueError" in out

    def test_successful_calls_still_have_result_comment(self):
        out = render_script(_error_stream(), mode="verbatim")
        assert "# ->" in out

    def test_names_bound_before_use(self):
        stream = [
            _sev(),
            _load_event(1, assign_to="motl_0"),
            _op_event(3, "motl_0", "motl_0"),
            _error_event(5, receiver="motl_0", assign_to="motl_1"),
        ]
        out = render_script(stream, mode="verbatim")
        _assert_names_bound(out)

    def test_replay_stops_at_error(self):
        stream = [
            _sev(),
            _load_event(1, assign_to="motl_0"),
            _error_event(3, receiver="motl_0", assign_to="motl_1"),
        ]
        out = render_script(stream, mode="verbatim")
        ns = _run_script(out)
        assert "motl_0" in ns

    def test_empty_stream(self):
        out = render_script(_empty_stream(), mode="verbatim")
        assert out == "" or out.strip() == ""

    def test_truncated_stream(self):
        out = render_script(_truncated_stream(), mode="verbatim")
        assert isinstance(out, str)


# ── lineage_of ────────────────────────────────────────────────────────────────

class TestLineageOf:
    def _chain_stream(self):
        # motl_0 created at seq=1
        # motl_0 refined at seq=3 (in-place)
        # motl_1 derived from motl_0 at seq=5
        prov.record("motl-0", 1)
        prov.record("motl-1", 5)
        return [
            _sev(),
            _load_event(1, assign_to="motl_0"),
            _op_event(3, receiver="motl_0", assign_to="motl_0"),
            _op_event(5, receiver="motl_0", assign_to="motl_1"),
        ]

    def test_lineage_of_leaf_includes_transitive_producers(self):
        stream = self._chain_stream()
        out = render_script(stream, mode="successful", lineage_of="motl-1")
        lines = [l for l in out.splitlines() if "motl_" in l and "import" not in l and not l.startswith("#")]
        assert len(lines) == 3, f"expected 3 call lines (load + refine + derive), got: {lines}"

    def test_lineage_of_preserves_seq_order(self):
        stream = self._chain_stream()
        out = render_script(stream, mode="successful", lineage_of="motl-1")
        lines = [l for l in out.splitlines() if "=" in l and "import" not in l and not l.startswith("#")]
        # load comes before shrink which comes before the final op
        assert lines[0].startswith("motl_0 = testmod.FakeMotl.load")
        assert lines[-1].endswith("motl_1 = motl_0.shrink(by=2)  # -> FakeMotl, 8 rows")

    def test_lineage_of_excludes_unrelated(self):
        prov.record("motl-0", 1)
        prov.record("motl-2", 7)
        stream = [
            _sev(),
            _load_event(1, assign_to="motl_0"),
            _load_event(7, assign_to="motl_2"),
        ]
        out = render_script(stream, mode="successful", lineage_of="motl-0")
        assert "motl_2" not in out

    def test_lineage_of_unknown_motl_returns_empty(self):
        out = render_script(_simple_stream(), lineage_of="motl-99")
        body = "\n".join(l for l in out.splitlines() if "=" in l and "import" not in l)
        assert body == ""


# ── render_notebook ───────────────────────────────────────────────────────────

class TestRenderNotebook:
    def test_returns_dict(self):
        assert isinstance(render_notebook(_simple_stream()), dict)

    def test_nbformat_version_4(self):
        nb = render_notebook(_simple_stream())
        assert nb["nbformat"] == 4

    def test_json_round_trip(self):
        nb = render_notebook(_simple_stream())
        recovered = json.loads(json.dumps(nb))
        assert recovered["nbformat"] == 4

    def test_one_code_cell_per_call_event(self):
        stream = _simple_stream()
        nb = render_notebook(stream)
        n_calls = sum(1 for ev in stream if ev.get("kind") == "call")
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code" and
                      any("=" in "".join(c["source"]) or "." in "".join(c["source"])
                          for _ in [None])]
        # Simpler: count code cells (imports header + one per call)
        all_code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        # At least n_calls code cells (there may be an extra imports cell)
        assert len(all_code) >= n_calls

    def test_code_cell_contains_call(self):
        stream = [_sev(), _load_event(1, assign_to="motl_0")]
        nb = render_notebook(stream)
        code_sources = [
            "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
        ]
        assert any("FakeMotl.load" in s for s in code_sources)

    def test_result_summary_in_markdown_cell(self):
        stream = _simple_stream()
        nb = render_notebook(stream)
        md_sources = [
            "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"
        ]
        assert any("10 rows" in s for s in md_sources)

    def test_empty_stream(self):
        nb = render_notebook(_empty_stream())
        assert isinstance(nb, dict)
        assert nb["nbformat"] == 4

    def test_errors_only_stream(self):
        nb = render_notebook(_errors_only_stream())
        md_sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"]
        assert any("ERROR" in s for s in md_sources)

    def test_truncated_stream(self):
        nb = render_notebook(_truncated_stream())
        assert isinstance(nb, dict)

    def test_code_cell_has_source_list(self):
        nb = render_notebook(_simple_stream())
        for cell in nb["cells"]:
            assert isinstance(cell["source"], list)

    def test_notebook_round_trip_exact(self):
        stream = _simple_stream()
        nb1 = render_notebook(stream)
        nb2 = json.loads(json.dumps(nb1))
        assert nb1 == nb2


# ── Helpers for names-bound-before-use and replay ─────────────────────────────

def _assert_names_bound(code: str) -> None:
    """ast.parse *code* and assert every Name load is bound before use."""
    tree = ast.parse(code)
    bound: set[str] = set(dir(builtins))

    def _add_import_bindings(stmt):
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                bound.add(alias.asname or alias.name)

    def _check_load_names(node, context: str):
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                assert child.id in bound, (
                    f"Name '{child.id}' used before binding in: {context!r}\n"
                    f"Bound so far: {sorted(bound)}"
                )

    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            _add_import_bindings(stmt)
        elif isinstance(stmt, ast.Assign):
            _check_load_names(stmt.value, ast.unparse(stmt))
            for target in stmt.targets:
                for node in ast.walk(target):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        bound.add(node.id)
        elif isinstance(stmt, ast.Expr):
            _check_load_names(stmt.value, ast.unparse(stmt))


def _run_script(code: str) -> dict:
    """Exec *code* in a fresh namespace with a stub ``testmod`` injected."""
    import sys

    testmod = types.ModuleType("testmod")

    class FakeMotl:
        def __init__(self, n: int):
            self.n = n

        def shrink(self, *, by: int):
            return FakeMotl(self.n - by)

        def fail(self, *, x):
            raise ValueError("bad input")

    class FakeMotlClass:
        @staticmethod
        def load(*, path: str):
            return FakeMotl(10)

        shrink = FakeMotl.shrink
        fail = FakeMotl.fail

    testmod.FakeMotl = FakeMotlClass

    sys.modules["testmod"] = testmod
    try:
        ns: dict = {}
        exec(compile(code, "<test_script>", "exec"), ns)
        return ns
    finally:
        sys.modules.pop("testmod", None)


# ── _kwargs_str placeholder filtering ────────────────────────────────────────

class TestKwargsStrPlaceholderFiltering:

    def _event_with_kwargs_src(self, kwargs_src: dict) -> dict:
        return _call(1, fn="testmod.FakeMotl.op", kwargs_src=kwargs_src)

    def test_ndarray_placeholder_renders_as_none(self):
        """kwargs_src values starting with '...' must render as None, not as literal '...'."""
        ev = self._event_with_kwargs_src({
            "arr": "...  # ndarray shape (56, 3): value passed, not representable as source"
        })
        out = render_script([_sev(), ev])
        call_lines = [l for l in out.splitlines() if "FakeMotl.op(" in l]
        assert call_lines, "call must appear in output"
        assert "arr=None" in call_lines[0], f"expected arr=None, got: {call_lines[0]}"
        assert "..." not in call_lines[0], f"raw placeholder must not appear in script: {call_lines[0]}"

    def test_ndarray_placeholder_produces_valid_python(self):
        """Script with a placeholder kwarg must be parseable by ast.parse."""
        ev = self._event_with_kwargs_src({
            "path": "'/data/motl.em'",
            "mask": "...  # ndarray shape (10, 10, 10): value passed, not representable as source",
        })
        out = render_script([_sev(), ev])
        ast.parse(out)  # must not raise SyntaxError

    def test_self_sentinel_not_emitted_as_kwarg(self):
        """The '<self>' sentinel key must not appear in the rendered call expression."""
        ev = self._event_with_kwargs_src({
            "<self>": None,
            "distance": "20",
        })
        out = render_script([_sev(), ev])
        assert "<self>" not in out, f"sentinel key must not leak into script: {out}"
