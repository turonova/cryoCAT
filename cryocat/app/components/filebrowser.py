"""Single app-level file-browser modal (Phase 10, D1).

Exposes two public symbols:

    get_file_browser() -> Component
        The one modal.  Mount once at app level in suite/app.py and
        tango/layout.py (before page layouts).

    register_file_browser_callbacks(app) -> None
        Wire all modal callbacks.  Call once, at app level.

Security: this component exposes the filesystem the server process can read.
The app must be bound to 127.0.0.1 — see GUI_CONVENTIONS.md §12.

ID scheme
---------
All browser-internal ids are plain strings prefixed ``browser-``.
They are app-global; mount the modal exactly once per app.
"""
from __future__ import annotations

import os
from pathlib import Path

import dash
from dash import html, dcc, Input, Output, State, ALL, no_update, ctx
import dash_bootstrap_components as dbc

from cryocat.app import ids
from cryocat.app.components.filesystem import (
    Entry,
    list_dir,
    breadcrumbs,
    resolve_input,
    validate,
)


# ── Layout helpers ────────────────────────────────────────────────────────────

def _breadcrumb_bar(crumbs: list[tuple[str, str]]) -> html.Div:
    """Render a clickable breadcrumb row."""
    items = []
    for i, (name, path) in enumerate(crumbs):
        is_last = i == len(crumbs) - 1
        items.append(
            dbc.Button(
                name,
                id={"type": "browser-crumb", "i": i},
                color="link",
                size="sm",
                disabled=is_last,
                style={
                    "padding": "0 0.2rem",
                    "textDecoration": "none" if is_last else "underline",
                    "color": "var(--bs-secondary)" if is_last else "var(--bs-primary)",
                },
            )
        )
        if not is_last:
            items.append(html.Span("/", style={"color": "var(--bs-secondary)"}))
    return html.Div(items, style={"display": "flex", "flexWrap": "wrap", "alignItems": "center", "gap": "0"})


def _entry_icon(is_dir: bool) -> str:
    return "📁 " if is_dir else "📄 "


def _render_listing(
    entries: list[Entry],
    error: str | None,
) -> list:
    """Render directory listing as clickable list items."""
    if error:
        return [
            dbc.Alert(error, color="warning", className="mb-0 py-2")
        ]

    if not entries:
        return [
            html.Div(
                "Empty directory",
                style={"color": "var(--bs-secondary)", "padding": "0.5rem"},
            )
        ]

    items = []
    for i, entry in enumerate(entries):
        id_ = {"type": "browser-dir-entry", "i": i} if entry.is_dir else {"type": "browser-file-entry", "i": i}
        size_str = ""
        if not entry.is_dir and entry.size is not None:
            if entry.size >= 1_048_576:
                size_str = f"  {entry.size / 1_048_576:.1f} MB"
            elif entry.size >= 1024:
                size_str = f"  {entry.size / 1024:.0f} KB"
            else:
                size_str = f"  {entry.size} B"

        items.append(
            dbc.ListGroupItem(
                [
                    html.Span(_entry_icon(entry.is_dir)),
                    html.Span(
                        entry.name + ("/" if entry.is_dir else ""),
                        style={"fontWeight": "600" if entry.is_dir else "normal"},
                    ),
                    html.Span(
                        size_str,
                        style={"marginLeft": "auto", "color": "var(--bs-secondary)"},
                    ),
                ],
                id=id_,
                n_clicks=0,
                action=True,
                style={
                    "cursor": "pointer",
                    "padding": "0.3rem 0.6rem",
                    "display": "flex",
                    "alignItems": "center",
                },
            )
        )
    return items


_MODE_TITLES: dict[str, str] = {
    "open": "Open file",
    "directory": "Select directory",
    "save": "Save as…",
}


def _listing_outputs(cwd: str, mode: str, exts: tuple, show_hidden: bool) -> tuple:
    entries, error = list_dir(cwd, extensions=exts, show_hidden=show_hidden)
    paths = [e.path for e in entries]
    crumb_bar = _breadcrumb_bar(breadcrumbs(cwd))
    ext_hint = f"Filter: {', '.join(exts)}" if exts else ""
    filename_row_style = {"display": "block"} if mode == "save" else {"display": "none"}
    return _render_listing(entries, error), crumb_bar, paths, cwd, ext_hint, _MODE_TITLES.get(mode, "Browse"), filename_row_style


def _resolve_confirm_path(nav_value: str | None, filename_value: str | None, mode: str) -> str:
    path = str(Path(nav_value or "") / (filename_value or "")) if mode == "save" else (nav_value or "")
    resolved, err = resolve_input(path)
    return resolved if (resolved and not err) else path


def _update_last_dirs(last_dirs, final_path: str, kind: str) -> dict:
    dirs = dict(last_dirs or {})
    if final_path and kind:
        p = Path(final_path)
        dirs[kind] = str(p) if p.is_dir() else str(p.parent)
    return dirs


# ── Public: layout ────────────────────────────────────────────────────────────

def get_file_browser() -> html.Div:
    """Return the app-level browser stores + modal.  Mount once per app."""
    return html.Div(
        [
            # ── App-level stores ──────────────────────────────────────────────
            dcc.Store(id=ids.BROWSER_REQUEST, data={}),
            dcc.Store(id=ids.BROWSER_CWD, data=""),
            dcc.Store(id=ids.BROWSER_LAST_DIR, data={}),
            dcc.Store(id=ids.BROWSER_RESULT, data={}),
            # Internal: snapshot of the listed entry paths (by index)
            dcc.Store(id="browser-entry-paths", data=[]),

            # ── Modal ─────────────────────────────────────────────────────────
            dbc.Modal(
                [
                    dbc.ModalHeader(
                        [
                            dbc.ModalTitle(id="browser-title", children="Browse"),
                            dbc.Button(
                                "↑ Up",
                                id="browser-up-btn",
                                color="secondary",
                                size="sm",
                                style={"marginLeft": "0.5rem"},
                            ),
                        ],
                        close_button=True,
                    ),
                    dbc.ModalBody(
                        [
                            # Breadcrumb bar
                            html.Div(id="browser-breadcrumbs", style={"marginBottom": "0.5rem"}),

                            # Path text input (type-in / shows current selection)
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("Path"),
                                    dbc.Input(
                                        id="browser-nav-input",
                                        type="text",
                                        debounce=True,
                                        placeholder="Type or paste a path…",
                                        ),
                                ],
                                size="sm",
                                style={"marginBottom": "0.5rem"},
                            ),

                            # Show-hidden toggle + extension hint
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Checklist(
                                            options=[{"label": "Show hidden", "value": "show"}],
                                            value=[],
                                            id="browser-show-hidden",
                                            inline=True,
                                            ),
                                        width="auto",
                                    ),
                                    dbc.Col(
                                        html.Span(
                                            id="browser-ext-hint",
                                            style={
                                                "color": "var(--bs-secondary)",
                                            },
                                        ),
                                        width="auto",
                                        style={"marginLeft": "auto"},
                                    ),
                                ],
                                className="g-0",
                                style={"marginBottom": "0.4rem"},
                            ),

                            # Directory listing
                            html.Div(
                                dbc.ListGroup(
                                    id="browser-listing",
                                    flush=True,
                                    style={"maxHeight": "320px", "overflowY": "auto"},
                                ),
                                style={
                                    "border": "1px solid var(--bs-border-color)",
                                    "borderRadius": "0.25rem",
                                    "marginBottom": "0.5rem",
                                },
                            ),

                            # Save-mode filename row (hidden in open/directory modes)
                            html.Div(
                                dbc.InputGroup(
                                    [
                                        dbc.InputGroupText("Filename"),
                                        dbc.Input(
                                            id="browser-filename-input",
                                            type="text",
                                            placeholder="filename.ext",
                                            ),
                                    ],
                                    size="sm",
                                ),
                                id="browser-filename-row",
                                style={"display": "none"},
                            ),
                        ]
                    ),
                    dbc.ModalFooter(
                        [
                            html.Span(
                                id="browser-validation-msg",
                                style={"color": "var(--bs-danger)", "marginRight": "auto"},
                            ),
                            dbc.Button(
                                "Confirm",
                                id="browser-confirm-btn",
                                color="primary",
                                size="sm",
                            ),
                            dbc.Button(
                                "Cancel",
                                id="browser-cancel-btn",
                                color="secondary",
                                size="sm",
                                style={"marginLeft": "0.4rem"},
                            ),
                        ]
                    ),
                ],
                id="browser-modal",
                size="lg",
                is_open=False,
                scrollable=False,
                centered=True,
            ),
        ]
    )


# ── Public: callbacks ─────────────────────────────────────────────────────────

def register_file_browser_callbacks(app) -> None:
    """Register all file-browser callbacks.  Call once at app level."""

    # ── 1. Browse button click → open modal, populate BROWSER_REQUEST + CWD ──

    @app.callback(
        Output(ids.BROWSER_REQUEST, "data"),
        Output(ids.BROWSER_CWD, "data", allow_duplicate=True),
        Output("browser-modal", "is_open", allow_duplicate=True),
        Input({"type": "path-browse-btn", "owner": ALL}, "n_clicks"),
        State({"type": "path-browse-meta", "owner": ALL}, "data"),
        State(ids.BROWSER_LAST_DIR, "data"),
        prevent_initial_call=True,
    )
    def open_browser(btn_clicks, meta_list, last_dirs):
        if not any(c for c in (btn_clicks or []) if c):
            raise dash.exceptions.PreventUpdate
        owner = ctx.triggered_id["owner"]
        meta = next((s["value"] or {} for s in ctx.states_list[0] if s["id"]["owner"] == owner), {})
        mode, kind, exts = meta.get("mode", "open"), meta.get("kind", ""), meta.get("extensions", [])
        start_dir, _ = resolve_input((last_dirs or {}).get(kind) or str(Path.home()))
        return {"owner": owner, "mode": mode, "kind": kind, "extensions": exts}, start_dir, True

    # ── 2. CWD change → update listing, breadcrumbs, nav-input, ext-hint ─────

    @app.callback(
        Output("browser-listing", "children"),
        Output("browser-breadcrumbs", "children"),
        Output("browser-entry-paths", "data"),
        Output("browser-nav-input", "value", allow_duplicate=True),
        Output("browser-ext-hint", "children"),
        Output("browser-title", "children"),
        Output("browser-filename-row", "style"),
        Input(ids.BROWSER_CWD, "data"),
        State(ids.BROWSER_REQUEST, "data"),
        State("browser-show-hidden", "value"),
        prevent_initial_call=True,
    )
    def update_listing(cwd, request, show_hidden_val):
        if not cwd:
            raise dash.exceptions.PreventUpdate
        req = request or {}
        return _listing_outputs(cwd, req.get("mode", "open"), tuple(req.get("extensions") or []), bool(show_hidden_val))

    # ── 3. Show-hidden toggle → re-render ────────────────────────────────────

    @app.callback(
        Output("browser-listing", "children", allow_duplicate=True),
        Output("browser-entry-paths", "data", allow_duplicate=True),
        Input("browser-show-hidden", "value"),
        State(ids.BROWSER_CWD, "data"),
        State(ids.BROWSER_REQUEST, "data"),
        prevent_initial_call=True,
    )
    def toggle_hidden(show_hidden_val, cwd, request):
        if not cwd:
            raise dash.exceptions.PreventUpdate
        request = request or {}
        exts = tuple(request.get("extensions") or [])
        show_hidden = bool(show_hidden_val)
        entries, error = list_dir(cwd, extensions=exts, show_hidden=show_hidden)
        paths = [e.path for e in entries]
        return _render_listing(entries, error), paths

    # ── 4. Directory entry click → navigate ──────────────────────────────────

    @app.callback(
        Output(ids.BROWSER_CWD, "data", allow_duplicate=True),
        Input({"type": "browser-dir-entry", "i": ALL}, "n_clicks"),
        State("browser-entry-paths", "data"),
        prevent_initial_call=True,
    )
    def navigate_dir(clicks, paths):
        if not any(c for c in (clicks or []) if c):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        i = triggered["i"]
        if not paths or i >= len(paths):
            raise dash.exceptions.PreventUpdate
        return paths[i]

    # ── 5. File entry click → populate nav input ──────────────────────────────

    @app.callback(
        Output("browser-nav-input", "value", allow_duplicate=True),
        Input({"type": "browser-file-entry", "i": ALL}, "n_clicks"),
        State("browser-entry-paths", "data"),
        prevent_initial_call=True,
    )
    def select_file(clicks, paths):
        if not any(c for c in (clicks or []) if c):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        i = triggered["i"]
        if not paths or i >= len(paths):
            raise dash.exceptions.PreventUpdate
        return paths[i]

    # ── 6. Breadcrumb click → navigate ───────────────────────────────────────

    @app.callback(
        Output(ids.BROWSER_CWD, "data", allow_duplicate=True),
        Input({"type": "browser-crumb", "i": ALL}, "n_clicks"),
        State("browser-breadcrumbs", "children"),
        State(ids.BROWSER_CWD, "data"),
        prevent_initial_call=True,
    )
    def navigate_breadcrumb(clicks, crumb_children, cwd):
        if not any(c for c in (clicks or []) if c):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        i = triggered["i"]
        # Reconstruct the path from breadcrumbs
        crumbs = breadcrumbs(cwd or "")
        if i < len(crumbs):
            return crumbs[i][1]
        raise dash.exceptions.PreventUpdate

    # ── 7. Up button → parent directory ──────────────────────────────────────

    @app.callback(
        Output(ids.BROWSER_CWD, "data", allow_duplicate=True),
        Input("browser-up-btn", "n_clicks"),
        State(ids.BROWSER_CWD, "data"),
        prevent_initial_call=True,
    )
    def navigate_up(n_clicks, cwd):
        if not n_clicks or not cwd:
            raise dash.exceptions.PreventUpdate
        p = Path(cwd)
        parent = p.parent
        if parent == p:
            raise dash.exceptions.PreventUpdate
        return str(parent)

    # ── 8. Nav-input debounce → navigate if directory ─────────────────────────

    @app.callback(
        Output(ids.BROWSER_CWD, "data", allow_duplicate=True),
        Input("browser-nav-input", "value"),
        State(ids.BROWSER_CWD, "data"),
        prevent_initial_call=True,
    )
    def nav_input_changed(text, cwd):
        if not text:
            raise dash.exceptions.PreventUpdate
        resolved, err = resolve_input(text)
        if err:
            raise dash.exceptions.PreventUpdate
        p = Path(resolved)
        if p.is_dir():
            return resolved
        # It's a file path or doesn't exist: keep CWD at its parent if valid
        parent = p.parent
        if parent.is_dir() and str(parent) != cwd:
            return str(parent)
        raise dash.exceptions.PreventUpdate

    # ── 9. Validate on nav-input change ──────────────────────────────────────

    @app.callback(
        Output("browser-validation-msg", "children"),
        Input("browser-nav-input", "value"),
        Input("browser-filename-input", "value"),
        State(ids.BROWSER_REQUEST, "data"),
        prevent_initial_call=True,
    )
    def validate_selection(nav_value, filename_value, request):
        req = request or {}
        mode, exts = req.get("mode", "open"), tuple(req.get("extensions") or [])
        path = str(Path(nav_value or "") / (filename_value or "")) if mode == "save" else (nav_value or "")
        if not path:
            return ""
        resolved, err = resolve_input(path)
        if err:
            return err
        return validate(resolved, mode=mode, extensions=exts) or ""

    # ── 10. Confirm → write-back, close modal ────────────────────────────────

    @app.callback(
        Output("browser-modal", "is_open", allow_duplicate=True),
        Output({"type": "path-input", "owner": ALL}, "value"),
        Output(ids.BROWSER_RESULT, "data"),
        Output(ids.BROWSER_LAST_DIR, "data", allow_duplicate=True),
        Input("browser-confirm-btn", "n_clicks"),
        State("browser-nav-input", "value"),
        State("browser-filename-input", "value"),
        State(ids.BROWSER_REQUEST, "data"),
        State(ids.BROWSER_LAST_DIR, "data"),
        prevent_initial_call=True,
    )
    def on_confirm(n_clicks, nav_value, filename_value, request, last_dirs):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        req = request or {}
        owner, mode, kind = req.get("owner", ""), req.get("mode", "open"), req.get("kind", "")
        final_path = _resolve_confirm_path(nav_value, filename_value, mode)
        new_dirs = _update_last_dirs(last_dirs, final_path, kind)
        updates = [final_path if e["id"]["owner"] == owner else no_update for e in ctx.outputs_list[1]]
        return False, updates, {"owner": owner, "value": final_path}, new_dirs

    # ── 11. Cancel → close modal ─────────────────────────────────────────────

    @app.callback(
        Output("browser-modal", "is_open", allow_duplicate=True),
        Input("browser-cancel-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def on_cancel(n_clicks):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        return False
