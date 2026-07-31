"""Console command executor — the console's dispatch chokepoint.

Every command that reaches :func:`execute` emits exactly one
``kind == "call"`` event with ``source == "console"``.  No command is
silently discarded; failure events carry the traceback.

Pool state is passed in and returned immutably: the caller (the Dash
callback) applies :attr:`ConsoleResult.new_state` to the pool stores.

Module-level state
------------------
``_CONSOLE_LOCALS`` — dict of names bound by ``assign`` commands.
  Cleared by the ``_clean_server_state`` autouse fixture between tests.
``_add_pending`` — accumulates :func:`add` calls made during ``eval``.
  Flushed (with pool insertion) after the primary eval completes.

Public API
----------
* :data:`SAFE_BUILTINS`
* :data:`_CONSOLE_LOCALS`
* :class:`ConsoleResult`
* :class:`ConsoleExecuteError`
* :func:`build_namespace`
* :func:`execute`
"""
from __future__ import annotations

import ast
import contextlib
import io
import time
import traceback as _tb
from dataclasses import dataclass
from typing import Any

from cryocat.app.console.parse import Command, ConsoleRejected
from cryocat.app.pool import PoolState

# ---------------------------------------------------------------------------
# Module-level mutable state
# ---------------------------------------------------------------------------

_CONSOLE_LOCALS: dict = {}
_add_pending: list = []   # list of (value, label | None) tuples


# ---------------------------------------------------------------------------
# Safe builtins
# ---------------------------------------------------------------------------

SAFE_BUILTINS: dict = {
    "len": len,
    "range": range,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
    "min": min,
    "max": max,
    "sum": sum,
    "sorted": sorted,
    "abs": abs,
    "round": round,
    "enumerate": enumerate,
    "zip": zip,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "print": print,
    "type": type,
    "isinstance": isinstance,
    "None": None,
    "True": True,
    "False": False,
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConsoleExecuteError(RuntimeError):
    """A console command failed for a logical reason (bad pool state, etc.)."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ConsoleResult:
    """Return value of :func:`execute`.

    Attributes
    ----------
    value : Any
        Return value of the evaluated expression (``None`` on error or void).
    summary : str
        Short human-readable text for the output pane.
    new_state : PoolState
        Updated pool state.  Identical to the input state when the pool was
        not changed by this command.
    event : dict
        The ``call`` event dict that was emitted to the session stream.
    ok : bool
        ``True`` on success, ``False`` on any exception.
    error : str | None
        Exception message on failure; ``None`` on success.
    """
    value: Any
    summary: str
    new_state: PoolState
    event: dict
    ok: bool
    error: str | None


# ---------------------------------------------------------------------------
# Namespace builder
# ---------------------------------------------------------------------------

def build_namespace(state: PoolState) -> dict:
    """Build the eval namespace for *state*.

    The namespace contains, in priority order (last wins on collision):

    1. Registry callables by short name (``clean_by_distance``, …).
    2. Pool motls as ``motl_N`` Motl objects stamped with ``_pool_motl_id``.
    3. Console locals (names bound by prior ``assign`` commands).
    4. Special: ``_`` (last result), ``add`` (pool-insertion helper).
    5. Restricted ``__builtins__``.
    """
    import pandas as pd
    from cryocat.core.cryomotl import Motl
    from cryocat.app import discovery, provenance

    ns: dict = {}

    # 1. Registry callables (lowest priority — pool vars shadow them)
    try:
        for entry in discovery.entries():
            short = entry.fn.__name__
            if short not in ns:
                ns[short] = entry.fn
    except Exception:
        pass

    # 2. Pool entries as Motl objects
    for motl_id, meta in state.registry.items():
        if not meta.get("active", True):
            continue
        rows = state.motls.get(motl_id) or []
        var_name = provenance.bind(motl_id)   # "motl-3" → "motl_3"
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        try:
            motl = Motl(df)
            motl._pool_motl_id = motl_id
            ns[var_name] = motl
        except Exception:
            pass

    # 3. Console locals (shadow pool vars — user can rebind anything except motl_N)
    ns.update(_CONSOLE_LOCALS)

    # 4. Special names
    ns["add"] = _console_add

    # 5. Restricted builtins
    ns["__builtins__"] = SAFE_BUILTINS

    return ns


# ---------------------------------------------------------------------------
# add() helper (fills _add_pending during eval)
# ---------------------------------------------------------------------------

def _console_add(value: Any, *, label: str | None = None) -> str:
    """Schedule *value* for insertion into the pool after eval completes.

    Returns a placeholder string; the real motl_id is assigned after the
    call event is emitted.
    """
    _add_pending.append((value, label))
    return f"<pending #{len(_add_pending)}>"


# ---------------------------------------------------------------------------
# Pool-update helpers
# ---------------------------------------------------------------------------

def _update_pool_entry(state: PoolState, motl_key: str, value: Any) -> PoolState:
    from cryocat.app.pool import replace_motl_rows

    if not hasattr(value, "df"):
        raise ConsoleExecuteError(
            f"Cannot assign to pool entry {motl_key!r}: "
            f"value must be a Motl object, got {type(value).__name__}"
        )
    if motl_key not in state.registry:
        raise ConsoleExecuteError(
            f"Pool entry {motl_key!r} not found — use add() to insert a new entry"
        )
    rows = value.df.to_dict("records")
    old_entry = state.registry[motl_key]
    return replace_motl_rows(
        state, motl_key, rows,
        label=old_entry.get("label"),
        motl_type=type(value).__name__,
    )


def _flush_add_pending(state: PoolState, last_seq_fn) -> tuple[PoolState, list[str]]:
    """Insert all pending add() values into the pool.

    *last_seq_fn* is a callable returning the seq of the most-recently-emitted
    event (used to record provenance for each inserted motl).
    """
    from cryocat.app import provenance
    from cryocat.app.pool import insert_motl, default_label

    new_ids: list[str] = []
    for value, label in _add_pending:
        rows = value.df.to_dict("records") if hasattr(value, "df") else []
        lbl = label or default_label(state.next_id)
        motl_type = type(value).__name__ if hasattr(value, "df") else "unknown"
        state, motl_id = insert_motl(state, rows, label=lbl, motl_type=motl_type)
        provenance.record(motl_id, last_seq_fn())
        new_ids.append(motl_id)
    _add_pending.clear()
    return state, new_ids


# ---------------------------------------------------------------------------
# Directive handler
# ---------------------------------------------------------------------------

def _handle_directive(cmd: Command, state: PoolState) -> ConsoleResult:
    from cryocat.app.console.help import history as _history, help_text, suggest

    d = cmd.directive
    arg = cmd.target

    if d == "clear":
        _CONSOLE_LOCALS.clear()
        summary = "Console locals cleared."
    elif d == "vars":
        pool_vars = [
            f"motl_{mid.split('-')[1]} ({meta.get('label', mid)})"
            for mid, meta in state.registry.items()
            if meta.get("active", True) and "-" in mid and mid.split("-")[1].isdigit()
        ]
        local_names = list(_CONSOLE_LOCALS.keys())
        lines = []
        if pool_vars:
            lines.append("pool: " + ", ".join(pool_vars))
        if local_names:
            lines.append("locals: " + ", ".join(local_names))
        summary = "\n".join(lines) if lines else "(empty)"
    elif d == "history":
        cmds = _history()
        summary = "\n".join(cmds[-20:]) if cmds else "(no history)"
    elif d == "help":
        summary = help_text(arg) if arg else "Directives: help <key>, vars, history, clear"
    else:
        summary = f"Unknown directive {d!r}"

    # Directives emit a lightweight call event.
    from cryocat.app import session as _session
    from cryocat.app.event import call_event as _call_event
    ev = _call_event(
        "console.directive",
        {"directive": repr(cmd.raw)},
        status="ok",
        source="console",
        command_src=cmd.src,
    )
    _session.emit(ev)
    return ConsoleResult(
        value=None, summary=summary,
        new_state=state, event=ev,
        ok=True, error=None,
    )


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def execute(cmd: Command, state: PoolState) -> ConsoleResult:
    """Execute *cmd* against *state* and return a :class:`ConsoleResult`.

    Exactly one ``kind == "call"`` event with ``source == "console"`` is
    emitted to the session stream — on both success and failure paths.

    Parameters
    ----------
    cmd : Command
        Parsed command (from :func:`~cryocat.app.console.parse.parse`).
    state : PoolState
        Current pool state snapshot.

    Returns
    -------
    ConsoleResult
        Always returned, never raised.  Check :attr:`~ConsoleResult.ok`.
    """
    from cryocat.app import session as _session, provenance
    from cryocat.app.event import call_event as _call_event, describe

    if cmd.kind == "directive":
        return _handle_directive(cmd, state)

    ns = build_namespace(state)
    _add_pending.clear()

    t0 = time.monotonic()

    # -- Derive the sub-expression to evaluate --------------------------------
    # For both Expr and Assign nodes, .value is the RHS expression.
    stmt = cmd.node.body[0]
    rhs_node = stmt.value    # ast.Expr.value  or  ast.Assign.value

    try:
        ast.fix_missing_locations(rhs_node)
        rhs_compiled = compile(
            ast.Expression(body=rhs_node), "<console>", "eval"
        )
        _stdout_buf = io.StringIO()
        with contextlib.redirect_stdout(_stdout_buf):
            value = eval(rhs_compiled, ns)  # noqa: S307 — namespace is restricted

        # -- Post-eval pool updates -------------------------------------------
        if cmd.kind == "pool_assign":
            motl_key = f"motl-{cmd.target}"
            new_state = _update_pool_entry(state, motl_key, value)
            _CONSOLE_LOCALS["_"] = value
            assign_var = provenance.bind(motl_key)
        elif cmd.kind == "assign":
            _CONSOLE_LOCALS[cmd.target] = value
            _CONSOLE_LOCALS["_"] = value
            new_state = state
            assign_var = None
        else:  # expr
            _CONSOLE_LOCALS["_"] = value
            new_state = state
            assign_var = None

        duration = round(time.monotonic() - t0, 3)
        pool_id_for_result = motl_key if cmd.kind == "pool_assign" else None
        result_summary = describe(value, pool_id=pool_id_for_result)
        ev = _call_event(
            "console.eval",
            {"_expr_": repr(cmd.raw)},
            status="ok",
            source="console",
            command_src=cmd.src,
            assign_to=assign_var,
            result=result_summary,
            duration_s=duration,
        )
        _session.emit(ev)

        # Record provenance for pool_assign after emit (needs last_seq).
        if cmd.kind == "pool_assign":
            provenance.record(f"motl-{cmd.target}", _session.last_seq())

        # Flush any add() calls that happened during eval.
        if _add_pending:
            new_state, _ = _flush_add_pending(new_state, _session.last_seq)

        # Summary: prefer captured stdout (print output), else str(value).
        printed = _stdout_buf.getvalue().strip()
        summary = printed if printed else (str(value) if value is not None else "")
        return ConsoleResult(
            value=value, summary=summary,
            new_state=new_state, event=ev,
            ok=True, error=None,
        )

    except Exception as exc:
        duration = round(time.monotonic() - t0, 3)
        tb_str = _tb.format_exc()
        ev = _call_event(
            "console.eval",
            {"_expr_": repr(cmd.raw)},
            status="error",
            source="console",
            command_src=cmd.src,
            error={"type": type(exc).__name__, "msg": str(exc), "traceback": tb_str},
            duration_s=duration,
        )
        _session.emit(ev)
        _add_pending.clear()
        return ConsoleResult(
            value=None,
            summary=f"{type(exc).__name__}: {exc}",
            new_state=state,
            event=ev,
            ok=False,
            error=str(exc),
        )
