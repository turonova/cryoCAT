"""Script projection — renders a session stream as a runnable Python script.

Public API
----------
* :func:`render_script` — ``list[dict] -> str``; two modes + optional lineage filter.
"""
from __future__ import annotations

from cryocat.app.record._common import call_expr, format_result


def render_script(
    events: list[dict],
    *,
    mode: str = "successful",
    lineage_of: str | None = None,
) -> str:
    """Render *events* as a Python script.

    Parameters
    ----------
    events:
        Flat list of event dicts as returned by ``session.events()``.
    mode:
        ``"successful"`` (default) — only successful calls, the clean
        replayable recipe.  ``"verbatim"`` — every call in order; failed
        calls appear as real uncommented code with the error comment above.
    lineage_of:
        Pool id (e.g. ``"motl_3"``). When given, only the transitive
        producers of that entry are included.
    """
    if mode not in ("successful", "verbatim"):
        raise ValueError(f"mode must be 'successful' or 'verbatim', got {mode!r}")

    call_events = [ev for ev in events if ev.get("kind") == "call"]

    if lineage_of is not None:
        seqs = _lineage_seqs(events, lineage_of)
        call_events = [ev for ev in call_events if ev.get("seq") in seqs]

    output_events = (
        [ev for ev in call_events if ev.get("status") == "ok"]
        if mode == "successful"
        else call_events
    )

    # Collect imports from non-console events that appear in the output.
    # Console events reference variables already in scope — no extra imports.
    seen_imports: dict[str, str] = {}
    for ev in output_events:
        if ev.get("source") == "console":
            continue
        for imp in ev.get("imports") or []:
            short, stmt = imp[0], imp[1]
            seen_imports[short] = stmt

    parts: list[str] = []

    session_ev = next((e for e in events if e.get("kind") == "session"), None)
    if session_ev:
        sid = session_ev.get("session_id", "")
        parts.append(f"# cryocat session {sid}")
        parts.append("")

    if seen_imports:
        parts.extend(seen_imports.values())
        parts.append("")

    for ev in output_events:
        status = ev.get("status", "ok")
        res_str = format_result(ev.get("result")) if status == "ok" else ""
        comment = f"  # -> {res_str}" if res_str else ""

        # Console events: use command_src verbatim (it is already valid Python).
        if ev.get("source") == "console" and ev.get("command_src"):
            command_src = ev["command_src"]
            if mode == "verbatim" and status != "ok":
                err = ev.get("error") or {}
                parts.append(f"# ERROR {err.get('type', 'Error')}: {err.get('msg', '')}")
                parts.append(command_src)
            else:
                parts.append(f"{command_src}{comment}")
            continue

        # GUI / tool events: reconstruct from fn / receiver / kwargs_src.
        expr = call_expr(ev)
        assign = ev.get("assign_to")
        if mode == "verbatim" and status != "ok":
            err = ev.get("error") or {}
            err_type = err.get("type", "Error")
            err_msg = err.get("msg", "")
            parts.append(f"# ERROR {err_type}: {err_msg}")
            parts.append(f"{assign} = {expr}" if assign else expr)
        else:
            parts.append(f"{assign} = {expr}{comment}" if assign else f"{expr}{comment}")

    text = "\n".join(parts)
    return text.rstrip("\n") + "\n" if text.strip() else ""


# ── Lineage walk ──────────────────────────────────────────────────────────────

def _lineage_seqs(events: list[dict], motl_id: str) -> set[int]:
    """Return the set of seq numbers that transitively produced *motl_id*.

    Follows the ``receiver`` chain backwards through ``assign_to`` edges.
    Cycles (in-place reassignment) are handled by the ``explored_vars`` guard.
    """
    from cryocat.app import provenance as prov

    target_var = prov.bind(motl_id)
    result_seqs: set[int] = set()
    vars_to_explore: set[str] = {target_var}
    explored_vars: set[str] = set()

    while vars_to_explore:
        var = vars_to_explore.pop()
        if var in explored_vars:
            continue
        explored_vars.add(var)

        for ev in events:
            if ev.get("kind") != "call":
                continue
            if ev.get("assign_to") != var:
                continue
            seq = ev.get("seq")
            if seq is not None:
                result_seqs.add(seq)
            recv = ev.get("receiver")
            if recv and recv not in explored_vars:
                vars_to_explore.add(recv)

    return result_seqs
