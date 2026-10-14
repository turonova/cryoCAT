"""Console completion, help text, and command history.

Public API
----------
* :class:`Suggestion`
* :func:`suggest`
* :func:`help_text`
* :func:`history`
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Suggestion:
    """One auto-completion candidate."""

    text: str    # the full completion text
    kind: str    # "pool" | "registry" | "local" | "builtin"
    detail: str  # short description (label, first docstring line, …)


def suggest(prefix: str, state) -> list[Suggestion]:
    """Return completions that start with *prefix* (desugared).

    *state* is a :class:`~cryocat.app.pool.PoolState` used to enumerate pool
    ids.  Returns at most 10 suggestions, pool ids first.
    """
    from cryocat.app import provenance
    from cryocat.app.console.execute import SAFE_BUILTINS, _CONSOLE_LOCALS
    from cryocat.app.console.sugar import desugar

    pfx = desugar(prefix.rstrip()).rsplit(" ", 1)[-1]  # last token

    results: list[Suggestion] = []

    # Pool ids: motl_N
    for mid, meta in (state.registry or {}).items():
        if not meta.get("active", True):
            continue
        var = provenance.bind(mid)
        if var.startswith(pfx):
            label = meta.get("label", mid)
            results.append(Suggestion(text=var, kind="pool", detail=label))

    # Registry short names
    try:
        from cryocat.app import discovery
        for entry in discovery.entries():
            name = entry.fn.__name__
            if name.startswith(pfx):
                results.append(
                    Suggestion(text=name, kind="registry", detail=entry.label)
                )
    except Exception:
        pass

    # Console locals
    for name in _CONSOLE_LOCALS:
        if name.startswith(pfx) and name != "_":
            results.append(Suggestion(text=name, kind="local", detail="local"))

    # Safe builtins
    for name in SAFE_BUILTINS:
        if isinstance(name, str) and name.startswith(pfx):
            results.append(Suggestion(text=name, kind="builtin", detail="builtin"))

    return results[:10]


def help_text(key: str | None) -> str:
    """Return a one-paragraph help string for *key*.

    *key* may be a registry key (``"Motl.clean_by_distance"``), a plain
    function name, or ``None``.  Falls back gracefully when not found.
    """
    if not key:
        return (
            "Directives: help <key>, vars, history, clear\n"
            "Pool refs: #n  (e.g. #0.clean_by_distance(distance=20))\n"
            "Assign to pool: #n = expr  |  local: x = expr\n"
            "Insert into pool: add(expr, label='my label')"
        )

    import inspect

    # Try registry first
    try:
        from cryocat.app import discovery
        entry = discovery.get(key)
        fn = entry.fn
        try:
            sig = str(inspect.signature(fn))
        except Exception:
            sig = "(...)"
        doc = inspect.getdoc(fn) or ""
        first_para = doc.split("\n\n")[0].replace("\n", " ") if doc else ""
        return f"{entry.label}{sig}\n{first_para}" if first_para else f"{entry.label}{sig}"
    except KeyError:
        pass

    # Try by function name in registry
    try:
        from cryocat.app import discovery
        for entry in discovery.entries():
            if entry.fn.__name__ == key:
                fn = entry.fn
                sig = str(inspect.signature(fn))
                doc = inspect.getdoc(fn) or ""
                first_para = doc.split("\n\n")[0].replace("\n", " ") if doc else ""
                return f"{entry.label}{sig}\n{first_para}" if first_para else f"{entry.label}{sig}"
    except Exception:
        pass

    return f"No help available for {key!r}"


def history() -> list[str]:
    """Return desugared source of successful console commands (most recent last).

    Reads from the currently open session stream.
    """
    from cryocat.app import session
    return [
        ev["command_src"]
        for ev in session.events()
        if ev.get("kind") == "call"
        and ev.get("source") == "console"
        and ev.get("status") == "ok"
        and ev.get("command_src")
    ]
