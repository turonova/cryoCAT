"""Session record projection — chronological, human-readable markdown.

Public API
----------
* :func:`render_markdown` — ``list[dict] -> str``; all events in order.
* :func:`render_html`     — thin HTML wrapper around :func:`render_markdown`.
"""
from __future__ import annotations

import html as _html

from cryocat.app.record._common import call_expr, format_result, ts_from


def render_markdown(events: list[dict], *, tool: str | None = None) -> str:
    """Render *events* as a Markdown session record.

    Parameters
    ----------
    events:
        Flat list of event dicts as returned by ``session.events()``.
    tool:
        When given, only ``call`` events whose ``tool`` field matches are
        included.  ``None`` means all calls.
    """
    blocks: list[str] = []
    for ev in events:
        kind = ev.get("kind")
        if kind == "session":
            sid = ev.get("session_id", ev.get("t", ""))
            ver = ev.get("cryocat_version", "?")
            blocks.append(f"# cryoCAT Session {sid} — v{ver}")
        elif kind == "call":
            if tool is not None and ev.get("tool") != tool:
                continue
            blocks.append(_call_block(ev))
        elif kind == "message":
            level = ev.get("level", "info")
            text = ev.get("text", "")
            if level == "error":
                blocks.append(f"> **[ERROR]** {text}")
            else:
                blocks.append(f"> {text}")
    return "\n\n".join(blocks) + "\n" if blocks else ""


def _call_block(ev: dict) -> str:
    fn = ev.get("fn", "unknown")
    name = fn.rsplit(".", 1)[-1]
    status = ev.get("status", "?")
    t = ts_from(ev)
    lines: list[str] = []

    if status == "ok":
        dur = ev.get("duration_s")
        dur_str = f" · {dur} s" if dur is not None else ""
        lines.append(f"### {t} · {name} · ok{dur_str}")
        lines.append(call_expr(ev))
        res = format_result(ev.get("result"))
        if res:
            lines.append(f"→ {res}")
    else:
        err = ev.get("error") or {}
        err_type = err.get("type", "Error")
        lines.append(f"### {t} · {name} · ERROR · {err_type}")
        lines.append(call_expr(ev))
        msg = err.get("msg", "")
        if msg:
            lines.append(f"  {err_type}: {msg}")
        hint = err.get("hint", "")
        if hint:
            lines.append(f"  {hint}")
        tb = err.get("traceback", "")
        for tb_line in tb.splitlines():
            lines.append(f"  {tb_line}")

    return "\n".join(lines)


# ── HTML ───────────────────────────────────────────────────────────────────────

def render_html(events: list[dict], **kw) -> str:
    """Render *events* as a standalone HTML document.

    All keyword arguments are forwarded to :func:`render_markdown`.
    """
    md = render_markdown(events, **kw)
    body = _md_to_html(md)
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>"
        "<meta charset=\"utf-8\"><title>cryoCAT Session</title></head>\n"
        f"<body>\n{body}\n</body></html>\n"
    )


def _md_to_html(md: str) -> str:
    """Minimal Markdown-to-HTML converter for the session record format."""
    html_parts: list[str] = []
    in_pre = False
    for line in md.splitlines():
        if line.startswith("# "):
            html_parts.append(f"<h1>{_html.escape(line[2:])}</h1>")
        elif line.startswith("### "):
            html_parts.append(f"<h3>{_html.escape(line[4:])}</h3>")
        elif line.startswith("> **[ERROR]**"):
            html_parts.append(
                f"<blockquote><strong>[ERROR]</strong>"
                f"{_html.escape(line[13:])}</blockquote>"
            )
        elif line.startswith("> "):
            html_parts.append(f"<blockquote>{_html.escape(line[2:])}</blockquote>")
        elif line.startswith("→ "):
            html_parts.append(f"<p><em>{_html.escape(line)}</em></p>")
        elif line.startswith("  "):
            if not in_pre:
                html_parts.append("<pre>")
                in_pre = True
            html_parts.append(_html.escape(line))
        else:
            if in_pre:
                html_parts.append("</pre>")
                in_pre = False
            if line.strip():
                html_parts.append(f"<p><code>{_html.escape(line)}</code></p>")
    if in_pre:
        html_parts.append("</pre>")
    return "\n".join(html_parts)
