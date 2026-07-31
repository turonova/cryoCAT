"""Notebook projection — renders a session stream as an nbformat v4 notebook.

Public API
----------
* :func:`render_notebook` — ``list[dict] -> dict``; one code cell per call.
"""
from __future__ import annotations

from cryocat.app.record._common import call_expr, format_result, ts_from
from cryocat.app.suite.pages._codegen_base import code_cell, markdown_cell, notebook


def render_notebook(events: list[dict], **kw) -> dict:
    """Render *events* as an nbformat v4 notebook dict.

    One markdown cell per call event (timestamp + status + result summary) and
    one code cell per call event (the call statement).  Session and message
    events contribute a markdown cell each.

    The returned dict round-trips through ``json.dumps`` / ``json.loads``.
    """
    cells: list[dict] = []

    session_ev = next((e for e in events if e.get("kind") == "session"), None)
    if session_ev:
        sid = session_ev.get("session_id", "")
        ver = session_ev.get("cryocat_version", "?")
        cells.append(markdown_cell(f"# cryoCAT Session {sid} — v{ver}"))

    # Collect unique imports across all call events for a header code cell.
    seen_imports: dict[str, str] = {}
    for ev in events:
        if ev.get("kind") != "call":
            continue
        for imp in ev.get("imports") or []:
            short, stmt = imp[0], imp[1]
            seen_imports[short] = stmt
    if seen_imports:
        cells.append(code_cell("\n".join(seen_imports.values())))

    for ev in events:
        kind = ev.get("kind")
        if kind == "call":
            cells.extend(_call_cells(ev))
        elif kind == "message":
            level = ev.get("level", "info")
            text = ev.get("text", "")
            prefix = "**[ERROR]** " if level == "error" else ""
            cells.append(markdown_cell(f"{prefix}{text}"))

    return notebook(cells)


def _call_cells(ev: dict) -> list[dict]:
    fn = ev.get("fn", "unknown")
    name = fn.rsplit(".", 1)[-1]
    status = ev.get("status", "?")
    t = ts_from(ev)

    if status == "ok":
        dur = ev.get("duration_s")
        dur_str = f" · {dur} s" if dur is not None else ""
        res = format_result(ev.get("result"))
        res_str = f" → {res}" if res else ""
        md_src = f"**{t}** · `{name}` · ok{dur_str}{res_str}"
    else:
        err = ev.get("error") or {}
        err_type = err.get("type", "Error")
        err_msg = err.get("msg", "")
        md_src = f"**{t}** · `{name}` · **ERROR** · {err_type}: {err_msg}"

    assign = ev.get("assign_to")
    expr = call_expr(ev)
    code_src = f"{assign} = {expr}" if assign else expr

    return [markdown_cell(md_src), code_cell(code_src)]
