"""Log panel component — offcanvas pane that renders from the event stream.

P5 changes:
- Display renders from ``session.events()`` (the JSONL stream) instead of the
  flat ``(msg, source)`` buffer.  The buffer is still queried for the cheap
  "any new error?" auto-open check.
- Each ``call`` event renders as one collapsible ``html.Details`` row so the
  pane stays compact even after many operations.
- Polling stops doing heavy work while the offcanvas is closed; it does the
  minimum needed to auto-open on error.
- "Export" button opens a dialog: choose mode (successful/verbatim), format
  (.py / .md / .ipynb), optional lineage motl-id, then save to a path.

Public API
----------
* :func:`render_events`    — pure function ``list[dict] -> list[component]``
* :func:`export_session`   — pure function: project events to a string
* :func:`get_log_panel`    — layout factory
* :func:`register_log_panel_callbacks` — register Dash callbacks
"""
from __future__ import annotations

import json as _json

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, ctx, dcc, html, no_update

from cryocat.app.logger import dash_logger

# ── Styling ────────────────────────────────────────────────────────────────────

_STATUS_STYLE = {
    "ok":    {"color": "var(--color9)", "fontWeight": "bold", "marginRight": "4px"},
    "error": {"color": "#e05252",       "fontWeight": "bold", "marginRight": "4px"},
}
_ROW_STYLE = {
    "fontFamily": "monospace",
    "marginBottom": "4px",
    "borderLeft": "3px solid #ccc",
    "paddingLeft": "6px",
}
_ERR_ROW_STYLE = {**_ROW_STYLE, "borderLeftColor": "#e05252"}
_SUMMARY_STYLE = {"cursor": "pointer", "listStyle": "none", "padding": "2px 0"}
_PRE_STYLE = {
    "marginTop": "4px",
    "padding": "4px",
    "background": "#1e1e1e",
    "color": "#f8f8f2",
    "borderRadius": "3px",
    "overflowX": "auto",
    "whiteSpace": "pre-wrap",
}
_MSG_STYLE_INFO = {
    "color": "#888888",
    "fontFamily": "monospace",
    "whiteSpace": "pre-wrap",
    "marginBottom": "2px",
}
_MSG_STYLE_ERR = {**_MSG_STYLE_INFO, "color": "#e05252", "fontWeight": "bold"}


# ── Pure rendering ─────────────────────────────────────────────────────────────

def render_events(events: list[dict]) -> list:
    """Convert a list of session events to a list of Dash components.

    This is the pure rendering kernel.  It is stateless and deterministic:
    ``render_events(A) + render_events(B) == render_events(A + B)``
    in terms of count and type for any disjoint prefix/suffix split.

    Parameters
    ----------
    events:
        Events as returned by ``session.events()``.  Unknown kinds are
        silently ignored.
    """
    items: list = []
    for ev in events:
        comp = _render_event(ev)
        if comp is not None:
            items.append(comp)
    return items


def _render_event(ev: dict):
    kind = ev.get("kind")
    if kind == "call":
        return _render_call_row(ev)
    if kind == "message":
        return _render_message_row(ev)
    if kind == "session":
        return _render_session_row(ev)
    return None


def _render_call_row(ev: dict):
    from cryocat.app.record._common import call_expr, format_result
    status = ev.get("status", "ok")
    dur = ev.get("duration_s")
    assign_to = ev.get("assign_to")

    # Summary line: always visible — icon + script line + duration
    icon = "✓" if status == "ok" else "✗"
    icon_style = _STATUS_STYLE.get(status, _STATUS_STYLE["ok"])
    script_line = f"{assign_to} = {call_expr(ev)}" if assign_to else call_expr(ev)
    summary_children = [
        html.Span(f"{icon} ", style=icon_style),
        html.Span(script_line, style={"fontFamily": "monospace"}),
    ]
    if dur is not None:
        summary_children.append(html.Span(f"  ({dur} s)", style={"color": "#888", "marginLeft": "4px"}))

    details_children: list = [html.Summary(summary_children, style=_SUMMARY_STYLE)]

    if status == "ok":
        res = ev.get("result")
        # Pool identity + structured summary
        res_label = format_result(res)
        if res_label:
            details_children.append(
                html.Div(f"→ {res_label}", style={"color": "var(--color9)", "marginTop": "4px"})
            )
        # Rich text from str(result) — only when present
        if res and res.get("text"):
            details_children.append(html.Pre(res["text"], style=_PRE_STYLE))
    else:
        err = ev.get("error") or {}
        err_type = err.get("type", "Error")
        err_msg = err.get("msg", "")
        hint = err.get("hint", "")
        tb = err.get("traceback", "")
        pre_lines = [f"{err_type}: {err_msg}"]
        if hint:
            pre_lines.append(hint)
        if tb:
            pre_lines.extend(["", tb])
        details_children.append(html.Pre("\n".join(pre_lines), style=_PRE_STYLE))

    return html.Details(
        details_children,
        style=_ERR_ROW_STYLE if status != "ok" else _ROW_STYLE,
        open=(status != "ok"),
    )


def _render_message_row(ev: dict):
    level = ev.get("level", "info")
    text = ev.get("text", "")
    style = _MSG_STYLE_ERR if level == "error" else _MSG_STYLE_INFO
    prefix = "⚠ " if level == "error" else ""
    return html.Div(f"{prefix}{text}", style=style)


def _render_session_row(ev: dict):
    sid = ev.get("session_id", ev.get("t", ""))
    ver = ev.get("cryocat_version", "?")
    return html.Div(
        [html.Hr(style={"margin": "4px 0"}),
         html.Span(f"Session {sid} — v{ver}", style={"color": "#888"})],
    )


# ── Export projection ──────────────────────────────────────────────────────────

def export_session(
    events: list[dict],
    *,
    mode: str = "successful",
    fmt: str = ".py",
    lineage_of: str | None = None,
) -> str:
    """Project *events* to a string in the requested format.

    Parameters
    ----------
    events:
        Flat list of event dicts as returned by ``session.events()``.
    mode:
        ``"successful"`` or ``"verbatim"`` — forwarded to ``render_script``.
    fmt:
        ``".py"``, ``".md"``, or ``".ipynb"``.
    lineage_of:
        Pool id (e.g. ``"motl_3"``).  When given, only the transitive
        producers of that entry are included.  Only used for ``.py`` exports.
    """
    if fmt == ".py":
        from cryocat.app.record.script import render_script
        return render_script(events, mode=mode, lineage_of=lineage_of)
    if fmt == ".md":
        from cryocat.app.record.session_record import render_markdown
        return render_markdown(events)
    if fmt == ".ipynb":
        from cryocat.app.record.notebook import render_notebook
        return _json.dumps(render_notebook(events), indent=2)
    raise ValueError(f"Unknown export format {fmt!r}; expected '.py', '.md', or '.ipynb'")


# ── Layout ─────────────────────────────────────────────────────────────────────

def get_log_panel(prefix: str):
    """Return the log offcanvas + stores + poll interval as a list of components."""
    return [
        dcc.Store(id=f"{prefix}-index", data=0),        # buffer index for error detection
        dcc.Store(id=f"{prefix}-last-seq", data=-1),    # last rendered event seq
        dcc.Store(id=f"{prefix}-save-path-store"),
        dcc.Interval(id=f"{prefix}-poll", interval=3000, n_intervals=0),
        dbc.Offcanvas(
            [
                # ── Toolbar ───────────────────────────────────────────────────
                html.Div(
                    [
                        dbc.Button("Save",    id=f"{prefix}-save-btn",    color="secondary", size="sm", className="me-1"),
                        dbc.Button("Save As", id=f"{prefix}-save-as-btn", color="primary",   size="sm", className="me-1"),
                        dbc.Button("Export",  id=f"{prefix}-export-btn",  color="primary",   size="sm"),
                        html.Span(id=f"{prefix}-save-status",
                                  style={"marginLeft": "0.75rem", "color": "grey"}),
                    ],
                    style={**{"display": "flex", "alignItems": "center"}, "marginBottom": "0.5rem"},
                ),
                html.Hr(style={"margin": "0.5rem 0"}),
                # ── Event display ─────────────────────────────────────────────
                html.Div(id=f"{prefix}-output", style={"overflowY": "auto"}),

                # ── Save As modal ─────────────────────────────────────────────
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Save Log As")),
                        dbc.ModalBody(
                            dbc.Input(id=f"{prefix}-save-path-input", type="text",
                                      placeholder="Full path including extension (e.g. /path/log.txt)"),
                        ),
                        dbc.ModalFooter([
                            html.Span(id=f"{prefix}-saveas-status",
                                      style={"marginRight": "auto", "color": "grey"}),
                            dbc.Button("Save", id=f"{prefix}-saveas-confirm-btn", color="primary"),
                        ]),
                    ],
                    id=f"{prefix}-save-as-modal",
                    is_open=False,
                    centered=True,
                ),

                # ── Export modal ──────────────────────────────────────────────
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Export Session")),
                        dbc.ModalBody([
                            dbc.Label("Mode", className="fw-bold"),
                            dbc.RadioItems(
                                id=f"{prefix}-export-mode",
                                options=[
                                    {"label": "Successful only (clean recipe)", "value": "successful"},
                                    {"label": "Verbatim (includes failures)",   "value": "verbatim"},
                                ],
                                value="successful",
                                className="mb-3",
                            ),
                            dbc.Label("Format", className="fw-bold"),
                            dbc.RadioItems(
                                id=f"{prefix}-export-format",
                                options=[
                                    {"label": "Python script (.py)",      "value": ".py"},
                                    {"label": "Session record (.md)",     "value": ".md"},
                                    {"label": "Jupyter notebook (.ipynb)", "value": ".ipynb"},
                                ],
                                value=".py",
                                className="mb-3",
                            ),
                            dbc.Label("Lineage of (optional, e.g. motl_3)", className="fw-bold"),
                            dbc.Input(id=f"{prefix}-export-lineage", type="text",
                                      placeholder="motl_3", className="mb-3"),
                            dbc.Label("Save to (full path)", className="fw-bold"),
                            dbc.Input(id=f"{prefix}-export-path", type="text",
                                      placeholder="/path/to/session_export.py"),
                        ]),
                        dbc.ModalFooter([
                            html.Span(id=f"{prefix}-export-status",
                                      style={"marginRight": "auto", "color": "grey"}),
                            dbc.Button("Cancel", id=f"{prefix}-export-cancel-btn",
                                       color="secondary", className="me-1"),
                            dbc.Button("Export", id=f"{prefix}-export-confirm-btn", color="primary"),
                        ]),
                    ],
                    id=f"{prefix}-export-modal",
                    is_open=False,
                    centered=True,
                    size="lg",
                ),
            ],
            id=f"{prefix}-panel",
            title="Log Output",
            placement="end",
            scrollable=True,
            style={"width": "500px"},
            is_open=False,
        ),
    ]


# ── Pure helpers (extracted from callbacks for testability) ───────────────────

def _poll_log_impl(open_btn_id, triggered, existing_children, last_seq, last_index, is_open, session):
    """Core logic of the poll/open callback with all arguments explicit."""
    _, new_index, _, new_error = dash_logger.get_logs(last_index)
    if open_btn_id is not None and triggered == open_btn_id:
        all_events = session.events()
        return render_events(all_events), _max_seq(all_events, last_seq), new_index, True
    if not is_open:
        if new_error:
            all_events = session.events()
            return render_events(all_events), _max_seq(all_events, last_seq), new_index, True
        return no_update, no_update, new_index, no_update
    all_events = session.events()
    new_events = [e for e in all_events if e.get("seq", -1) > last_seq]
    if not new_events:
        return no_update, no_update, new_index, no_update
    combined = (existing_children or []) + render_events(new_events)
    return combined, _max_seq(all_events, last_seq), new_index, no_update


def _safe_write(path: str, content: str) -> str | None:
    """Write *content* to *path*. Returns an OSError message on failure, None on success."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return None
    except OSError as e:
        return str(e)


# ── Callbacks ──────────────────────────────────────────────────────────────────

def register_log_panel_callbacks(app, prefix: str, open_btn_id: str | None = None):
    """Register all log panel callbacks.

    Parameters
    ----------
    open_btn_id:
        ID of the button that opens the panel.  ``None`` means the panel can
        only be opened automatically (on error) or programmatically.
    """
    from cryocat.app import session as _session

    # ── Main poll / open callback ─────────────────────────────────────────────
    inputs = [Input(f"{prefix}-poll", "n_intervals")]
    if open_btn_id is not None:
        inputs.append(Input(open_btn_id, "n_clicks"))

    @app.callback(
        Output(f"{prefix}-output", "children"),
        Output(f"{prefix}-last-seq", "data"),
        Output(f"{prefix}-index", "data"),
        Output(f"{prefix}-panel", "is_open"),
        *inputs,
        State(f"{prefix}-output", "children"),
        State(f"{prefix}-last-seq", "data"),
        State(f"{prefix}-index", "data"),
        State(f"{prefix}-panel", "is_open"),
        prevent_initial_call=True,
    )
    def update_log(*cb_args):
        # States are always the last 4 args regardless of how many inputs are present.
        existing_children, last_seq, last_index, is_open = cb_args[-4], cb_args[-3], cb_args[-2], cb_args[-1]
        triggered = ctx.triggered_id
        return _poll_log_impl(open_btn_id, triggered, existing_children, last_seq, last_index, is_open, _session)

    # ── Save As modal open ────────────────────────────────────────────────────
    @app.callback(
        Output(f"{prefix}-save-as-modal", "is_open"),
        Input(f"{prefix}-save-as-btn", "n_clicks"),
        Input(f"{prefix}-save-btn", "n_clicks"),
        State(f"{prefix}-save-path-store", "data"),
        prevent_initial_call=True,
    )
    def open_log_save_as(_, _2, saved_path):
        if ctx.triggered_id == f"{prefix}-save-as-btn":
            return True
        if ctx.triggered_id == f"{prefix}-save-btn" and not saved_path:
            return True
        return no_update

    # ── Save As confirm ───────────────────────────────────────────────────────
    @app.callback(
        Output(f"{prefix}-save-as-modal", "is_open", allow_duplicate=True),
        Output(f"{prefix}-saveas-status", "children"),
        Output(f"{prefix}-save-path-store", "data"),
        Output(f"{prefix}-save-status", "children"),
        Input(f"{prefix}-saveas-confirm-btn", "n_clicks"),
        State(f"{prefix}-save-path-input", "value"),
        prevent_initial_call=True,
    )
    def confirm_log_save_as(_, path):
        if not path:
            return no_update, "Specify a filename.", no_update, no_update
        err = _safe_write(path, dash_logger.get_all_logs())
        if err:
            return no_update, err, no_update, no_update
        return False, "Saved.", path, f"Saved to {path}"

    # ── Save (quick) ──────────────────────────────────────────────────────────
    @app.callback(
        Output(f"{prefix}-save-status", "children", allow_duplicate=True),
        Input(f"{prefix}-save-btn", "n_clicks"),
        State(f"{prefix}-save-path-store", "data"),
        prevent_initial_call=True,
    )
    def save_log(_, path):
        if not path:
            return no_update
        err = _safe_write(path, dash_logger.get_all_logs())
        return err if err else f"Saved to {path}"

    # ── Export modal open ─────────────────────────────────────────────────────
    @app.callback(
        Output(f"{prefix}-export-modal", "is_open"),
        Input(f"{prefix}-export-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_export(_):
        return True

    # ── Export confirm / cancel ───────────────────────────────────────────────
    @app.callback(
        Output(f"{prefix}-export-modal", "is_open", allow_duplicate=True),
        Output(f"{prefix}-export-status", "children"),
        Output(f"{prefix}-save-status", "children", allow_duplicate=True),
        Input(f"{prefix}-export-confirm-btn", "n_clicks"),
        Input(f"{prefix}-export-cancel-btn", "n_clicks"),
        State(f"{prefix}-export-mode", "value"),
        State(f"{prefix}-export-format", "value"),
        State(f"{prefix}-export-lineage", "value"),
        State(f"{prefix}-export-path", "value"),
        prevent_initial_call=True,
    )
    def do_export(_, _2, mode, fmt, lineage, path):
        if ctx.triggered_id == f"{prefix}-export-cancel-btn":
            return False, no_update, no_update
        if not path:
            return no_update, "Specify a file path.", no_update
        try:
            content = export_session(
                _session.events(),
                mode=mode or "successful",
                fmt=fmt or ".py",
                lineage_of=lineage or None,
            )
        except (ValueError, KeyError) as e:
            return no_update, str(e), no_update
        err = _safe_write(path, content)
        if err:
            return no_update, err, no_update
        return False, f"Exported to {path}", f"Exported to {path}"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _max_seq(events: list[dict], default: int) -> int:
    """Return the highest seq in *events*, or *default* if empty."""
    return max((e.get("seq", -1) for e in events), default=default)
