"""Tests for the console module — parser, executor, and provenance integration.

Covers:
- parse(): directives, expression commands, pool assigns, local assigns, desugaring,
  rejected constructs, syntax errors, and Command properties.
- execute(): event emission with source=="console", error path, pool assign,
  local assign, underscore binding, build_namespace, and SAFE_BUILTINS.
- Provenance integration: command_src carried in events, script projection uses
  command_src verbatim, console and GUI produce identical script lines for the
  same operation, round-trip re-execution reproduces pool state, and history()
  returns command_src strings.
"""
from __future__ import annotations

import pytest

from cryocat.app.console.parse import (
    Command,
    ConsoleSyntaxError,
    ConsoleRejected,
    parse,
    validate,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_state(n_rows: int = 5, label: str = "Test"):
    """Return a PoolState with one motl entry (motl-0) using proper 20-col rows."""
    from cryocat.app.pool import PoolState, insert_motl
    from tests.app.conftest import make_motl_rows
    state0 = PoolState.from_stores({}, {}, {}, {}, 0)
    state1, _ = insert_motl(state0, make_motl_rows(n_rows), label=label)
    return state1


def _run(text: str, state=None, n_rows: int = 5):
    """Parse + execute *text* in *state* (defaults to one-motl pool)."""
    from cryocat.app.console.parse import parse
    from cryocat.app.console.execute import execute
    if state is None:
        state = _make_state(n_rows)
    return execute(parse(text), state)


# ===========================================================================
# Parser tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ok(text: str, kind: str, **kwargs) -> Command:
    cmd = parse(text)
    assert cmd.kind == kind, f"expected kind={kind!r}, got {cmd.kind!r}"
    for k, v in kwargs.items():
        assert getattr(cmd, k) == v, f"{k}: expected {v!r}, got {getattr(cmd, k)!r}"
    return cmd


# ---------------------------------------------------------------------------
# Directives
# ---------------------------------------------------------------------------

def test_parse_help_bare():
    cmd = _parse_ok("help", "directive")
    assert cmd.directive == "help"
    assert cmd.target is None


def test_parse_help_with_arg():
    cmd = _parse_ok("help Motl.clean_by_distance", "directive")
    assert cmd.directive == "help"
    assert cmd.target == "Motl.clean_by_distance"


def test_parse_vars():
    cmd = _parse_ok("vars", "directive")
    assert cmd.directive == "vars"


def test_parse_history():
    cmd = _parse_ok("history", "directive")
    assert cmd.directive == "history"


def test_parse_clear():
    cmd = _parse_ok("clear", "directive")
    assert cmd.directive == "clear"


# ---------------------------------------------------------------------------
# Expression commands (desugared #n)
# ---------------------------------------------------------------------------

def test_parse_bare_pool_ref_expr():
    # #3 alone: desugars to motl_3 (a bare Name expression)
    cmd = _parse_ok("#3", "expr")
    assert cmd.src == "motl_3"
    assert cmd.raw == "#3"


def test_parse_method_call_expr():
    cmd = _parse_ok("#3.clean_by_distance(distance=20)", "expr")
    assert "motl_3" in cmd.src
    assert cmd.target is None


def test_parse_plain_python_expr():
    # Plain Python (no #) also works as expr
    cmd = _parse_ok("len([1, 2, 3])", "expr")
    assert cmd.src == "len([1, 2, 3])"


def test_parse_arithmetic_expr():
    _parse_ok("2 + 2", "expr")


# ---------------------------------------------------------------------------
# Pool assign (#N = expr  or  motl_N = expr)
# ---------------------------------------------------------------------------

def test_parse_pool_assign_hash_syntax():
    cmd = _parse_ok("#3 = #3.clean_by_distance(distance=20)", "pool_assign")
    assert cmd.target == "3"
    assert cmd.src == "motl_3 = motl_3.clean_by_distance(distance=20)"


def test_parse_pool_assign_motl_syntax():
    # GUI script lines use motl_N directly — should also parse as pool_assign.
    cmd = _parse_ok("motl_0 = motl_0.split_by_feature(feature='tomo_id')", "pool_assign")
    assert cmd.target == "0"


def test_parse_pool_assign_preserves_rhs():
    cmd = _parse_ok("#0 = #0", "pool_assign")
    assert cmd.src == "motl_0 = motl_0"
    assert cmd.target == "0"


# ---------------------------------------------------------------------------
# Local assign
# ---------------------------------------------------------------------------

def test_parse_local_assign():
    cmd = _parse_ok("x = 42", "assign")
    assert cmd.target == "x"
    assert cmd.src == "x = 42"


def test_parse_local_assign_with_pool_ref():
    cmd = _parse_ok("n = len(#2.df)", "assign")
    assert cmd.target == "n"
    assert "motl_2" in cmd.src


# ---------------------------------------------------------------------------
# Desugaring edge cases
# ---------------------------------------------------------------------------

def test_hash_inside_string_preserved():
    # The string "#3" should NOT be desugared to "motl_3".
    cmd = parse('"#3 is not a pool ref"')
    assert '"#3 is not a pool ref"' in cmd.src
    assert "motl_3" not in cmd.src


def test_hash_without_digits_is_comment():
    # "#" without digits is a Python comment; the expression before it is valid.
    cmd = parse("x = 1  # only a comment on this line")
    assert cmd.kind == "assign"
    assert cmd.target == "x"


def test_multi_pool_refs_in_one_command():
    # Multiple #n references in one expression
    cmd = parse("#0.merge_with(#1)")
    assert "motl_0" in cmd.src
    assert "motl_1" in cmd.src
    assert cmd.kind == "expr"


def test_desugar_does_not_partially_match():
    # #12 must become motl_12, not motl_1 + leftover 2
    cmd = parse("#12.df")
    assert "motl_12" in cmd.src
    assert "motl_1" not in cmd.src or "motl_12" in cmd.src


# ---------------------------------------------------------------------------
# Rejected constructs
# ---------------------------------------------------------------------------

def test_rejects_import():
    with pytest.raises(ConsoleRejected, match="[Ii]mport"):
        parse("import os")


def test_rejects_from_import():
    with pytest.raises(ConsoleRejected, match="[Ii]mport"):
        parse("from os import path")


def test_rejects_def():
    with pytest.raises(ConsoleRejected):
        parse("def f(): pass")


def test_rejects_class():
    with pytest.raises(ConsoleRejected):
        parse("class Foo: pass")


def test_rejects_lambda():
    with pytest.raises(ConsoleRejected):
        parse("f = lambda x: x")


def test_rejects_while():
    with pytest.raises(ConsoleRejected):
        parse("while True: pass")


def test_rejects_for():
    with pytest.raises(ConsoleRejected):
        parse("for x in []: pass")


def test_rejects_try():
    with pytest.raises(ConsoleRejected):
        parse("try:\n pass\nexcept:\n pass")


def test_rejects_with():
    with pytest.raises(ConsoleRejected):
        parse("with open('f') as fh: pass")


def test_rejects_dunder_attr():
    with pytest.raises(ConsoleRejected, match="private|dunder"):
        parse("x.__class__")


def test_rejects_private_attr():
    with pytest.raises(ConsoleRejected, match="private|dunder"):
        parse("x._internal")


def test_rejects_builtins_access():
    with pytest.raises(ConsoleRejected):
        parse("__builtins__")


def test_rejects_multiple_statements():
    with pytest.raises(ConsoleRejected, match="single"):
        parse("x = 1; y = 2")


# ---------------------------------------------------------------------------
# Syntax errors
# ---------------------------------------------------------------------------

def test_syntax_error_has_col():
    with pytest.raises(ConsoleSyntaxError) as exc_info:
        parse("def (")
    assert exc_info.value.col is not None or exc_info.value.line is not None


def test_empty_string_raises():
    with pytest.raises(ConsoleSyntaxError, match="empty"):
        parse("")


def test_whitespace_only_raises():
    with pytest.raises(ConsoleSyntaxError, match="empty"):
        parse("   ")


# ---------------------------------------------------------------------------
# Command properties
# ---------------------------------------------------------------------------

def test_command_is_frozen():
    cmd = parse("x = 42")
    with pytest.raises((AttributeError, TypeError)):
        cmd.kind = "expr"  # type: ignore[misc]


def test_raw_preserved_verbatim():
    raw = "  #3.clean_by_distance(distance=20)  "
    cmd = parse(raw)
    assert cmd.raw == raw.strip()


def test_src_is_desugared():
    cmd = parse("#7.df.head()")
    assert cmd.src == "motl_7.df.head()"
    assert "#" not in cmd.src


# ===========================================================================
# Executor tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Event properties
# ---------------------------------------------------------------------------

def test_bare_expr_emits_console_event():
    result = _run("1 + 1")
    ev = result.event
    assert ev["kind"] == "call"
    assert ev["source"] == "console"
    assert ev["command_src"] == "1 + 1"


def test_error_emits_error_event():
    result = _run("1 / 0")
    assert not result.ok
    assert result.event["status"] == "error"
    assert result.event["source"] == "console"
    assert result.event.get("command_src") == "1 / 0"
    assert result.event.get("error") is not None


def test_pool_assign_event_has_assign_to():
    result = _run("#1 = #1")
    ev = result.event
    assert ev["source"] == "console"
    assert ev["assign_to"] == "motl_1"


def test_directive_emits_event():
    state = _make_state()
    from cryocat.app.console.parse import parse
    from cryocat.app.console.execute import execute
    cmd = parse("vars")
    result = execute(cmd, state)
    assert result.event["source"] == "console"
    assert result.ok


# ---------------------------------------------------------------------------
# Return value / _ binding
# ---------------------------------------------------------------------------

def test_bare_expr_binds_underscore():
    from cryocat.app.console.execute import _CONSOLE_LOCALS
    _CONSOLE_LOCALS.clear()
    result = _run("2 + 3")
    assert result.ok
    assert _CONSOLE_LOCALS.get("_") == 5


def test_error_does_not_update_underscore():
    from cryocat.app.console.execute import _CONSOLE_LOCALS
    _CONSOLE_LOCALS.clear()
    _CONSOLE_LOCALS["_"] = 99
    result = _run("1 / 0")
    assert not result.ok
    # _ must not change on error
    assert _CONSOLE_LOCALS.get("_") == 99


def test_local_assign_binds_to_console_locals():
    from cryocat.app.console.execute import _CONSOLE_LOCALS
    _CONSOLE_LOCALS.clear()
    result = _run("myvar = 42")
    assert result.ok
    assert _CONSOLE_LOCALS.get("myvar") == 42


def test_local_assign_does_not_touch_pool():
    state = _make_state()
    result = _run("myvar = 99", state)
    assert result.new_state is state


# ---------------------------------------------------------------------------
# Pool assign
# ---------------------------------------------------------------------------

def test_pool_assign_replaces_entry():
    state = _make_state(5)
    # Identity assignment: #1 = #1 (same motl, same rows)
    result = _run("#1 = #1", state)
    assert result.ok, result.error
    assert result.new_state is not state
    assert result.new_state.registry["motl_1"]["n_rows"] == 5


def test_pool_assign_non_motl_raises():
    state = _make_state()
    result = _run("#1 = 42", state)
    assert not result.ok
    assert "motl" in result.error.lower() or "Motl" in result.error


def test_pool_assign_unknown_id_raises():
    # motl-99 does not exist in the pool
    from cryocat.app.pool import PoolState
    state = PoolState.from_stores({}, {}, {}, {}, 0)
    result = _run("#99 = #0", state)
    assert not result.ok


# ---------------------------------------------------------------------------
# Error path leaves pool unchanged
# ---------------------------------------------------------------------------

def test_error_leaves_pool_unchanged():
    state = _make_state()
    result = _run("undefined_variable_xyz", state)
    assert not result.ok
    assert result.new_state is state


# ---------------------------------------------------------------------------
# Namespace
# ---------------------------------------------------------------------------

def test_build_namespace_contains_pool_motl():
    from cryocat.app.console.execute import build_namespace
    state = _make_state()
    ns = build_namespace(state)
    assert "motl_1" in ns
    assert hasattr(ns["motl_1"], "df")


def test_build_namespace_pool_motl_has_id_stamp():
    from cryocat.app.console.execute import build_namespace
    state = _make_state()
    ns = build_namespace(state)
    assert getattr(ns["motl_1"], "_pool_motl_id", None) == "motl_1"


def test_build_namespace_restricts_builtins():
    from cryocat.app.console.execute import build_namespace, SAFE_BUILTINS
    state = _make_state()
    ns = build_namespace(state)
    assert ns["__builtins__"] is SAFE_BUILTINS
    # open / eval / exec must NOT be available
    assert "open" not in SAFE_BUILTINS
    assert "eval" not in SAFE_BUILTINS
    assert "exec" not in SAFE_BUILTINS


def test_eval_in_namespace_works():
    # Verify that safe builtins are actually callable from inside eval()
    state = _make_state()
    result = _run("len([1, 2, 3])")
    assert result.ok
    assert result.value == 3


# ---------------------------------------------------------------------------
# SAFE_BUILTINS contract
# ---------------------------------------------------------------------------

def test_safe_builtins_contains_expected():
    from cryocat.app.console.execute import SAFE_BUILTINS
    for name in ("len", "range", "list", "tuple", "dict", "set",
                  "min", "max", "sum", "sorted", "abs", "round",
                  "enumerate", "zip", "str", "int", "float", "bool"):
        assert name in SAFE_BUILTINS, f"{name!r} missing from SAFE_BUILTINS"


# ===========================================================================
# Console provenance integration tests
# ===========================================================================

# ---------------------------------------------------------------------------
# command_src present in console events
# ---------------------------------------------------------------------------

def test_console_event_has_command_src():
    result = _run("#1 = #1")
    ev = result.event
    assert "command_src" in ev
    assert ev["command_src"] == "motl_1 = motl_1"


def test_console_event_source_is_console():
    result = _run("len(range(10))")
    assert result.event["source"] == "console"


def test_error_event_has_command_src():
    result = _run("1 / 0")
    assert not result.ok
    assert result.event.get("command_src") == "1 / 0"


# ---------------------------------------------------------------------------
# Script projection uses command_src verbatim
# ---------------------------------------------------------------------------

def _make_console_event(command_src: str, status: str = "ok", result_d=None) -> dict:
    """Build a minimal console call event."""
    from cryocat.app.event import call_event
    return call_event(
        "console.eval",
        {"_expr_": repr(command_src)},
        status=status,
        source="console",
        command_src=command_src,
        result=result_d,
    )


def _render_events(events: list[dict], mode: str = "successful") -> str:
    from cryocat.app.record.script import render_script
    return render_script(events, mode=mode)


def test_script_uses_command_src_verbatim():
    ev = _make_console_event("motl_0 = motl_0.clean_by_distance(distance=20)")
    ev["seq"] = 0
    script = _render_events([ev])
    assert "motl_0 = motl_0.clean_by_distance(distance=20)" in script


def test_script_skips_failed_console_event_in_successful_mode():
    ev = _make_console_event("1 / 0", status="error")
    ev["seq"] = 0
    script = _render_events([ev], mode="successful")
    # Successful mode only includes ok events
    assert "1 / 0" not in script


def test_script_includes_failed_console_event_in_verbatim_mode():
    ev = _make_console_event("1 / 0", status="error")
    ev["seq"] = 0
    ev["error"] = {"type": "ZeroDivisionError", "msg": "division by zero", "traceback": ""}
    script = _render_events([ev], mode="verbatim")
    assert "1 / 0" in script
    assert "ERROR" in script


def test_console_events_do_not_add_imports():
    ev = _make_console_event("motl_0.clean()")
    ev["seq"] = 0
    ev["imports"] = [["cryomotl", "from cryocat.core import cryomotl"]]
    script = _render_events([ev])
    # Console events must not contribute imports
    assert "import" not in script


# ---------------------------------------------------------------------------
# Console and GUI produce identical script lines for the same operation
# ---------------------------------------------------------------------------

def test_console_and_gui_identical_script_lines():
    """Pool assign on console produces the same script line as a GUI invoke_operation."""
    from cryocat.app.event import call_event
    from cryocat.app.record.script import render_script

    # Simulate a GUI call event for the same operation.
    gui_ev = call_event(
        "cryocat.core.cryomotl.Motl.clean_by_distance",
        {"distance": "20"},
        status="ok",
        source="gui",
        receiver="motl_0",
        assign_to="motl_0",
    )
    gui_ev["seq"] = 0

    # Simulate a console call event for the same operation.
    console_ev = call_event(
        "console.eval",
        {"_expr_": repr("motl_0 = motl_0.clean_by_distance(distance=20)")},
        status="ok",
        source="console",
        command_src="motl_0 = motl_0.clean_by_distance(distance=20)",
        assign_to="motl_0",
    )
    console_ev["seq"] = 1

    gui_script = render_script([gui_ev]).strip()
    console_script = render_script([console_ev]).strip()

    # Both should render as the same Python statement.
    assert gui_script == console_script, (
        f"GUI script:     {gui_script!r}\n"
        f"Console script: {console_script!r}"
    )


# ---------------------------------------------------------------------------
# Round-trip: script re-execution produces same pool state
# ---------------------------------------------------------------------------

def test_round_trip_pool_assign():
    """Console command -> script line -> re-execute -> same pool row count."""
    from cryocat.app.record.script import render_script
    from cryocat.app.console.parse import parse
    from cryocat.app.console.execute import execute

    state = _make_state(n_rows=5, label="Test")

    # Step 1: run a console command that updates the pool
    text = "#1 = #1"   # identity — keeps same rows
    result = _run(text, state)
    assert result.ok, result.error

    # Step 2: generate a script from the event
    ev = result.event
    ev["seq"] = 0
    script = render_script([ev]).strip()

    # Step 3: the script line starts with command_src (may have trailing comment).
    script_code = script.split("  # ->")[0].strip()
    assert script_code == ev["command_src"], (
        f"Script line {script_code!r} != command_src {ev['command_src']!r}"
    )

    # Step 4: re-execute the script line as a console command in a fresh pool
    state2 = _make_state(n_rows=5, label="Test")
    result2 = execute(parse(script), state2)
    assert result2.ok, result2.error
    assert result2.new_state.registry["motl_1"]["n_rows"] == 5


# ---------------------------------------------------------------------------
# history() reads successful console commands from the event stream
# ---------------------------------------------------------------------------

def test_history_returns_command_src_strings(tmp_path):
    """history() reads successful console commands from the event stream."""
    from cryocat.app import session
    from cryocat.app.console.help import history

    session.start_session(log_dir=str(tmp_path))
    try:
        _run("1 + 1")   # emits a console event
        _run("2 + 2")
        cmds = history()
        assert "1 + 1" in cmds
        assert "2 + 2" in cmds
    finally:
        session.close_session()
