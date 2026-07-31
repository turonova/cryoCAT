"""Dash console UI — a bottom offcanvas with an interactive command panel.

The console gives users a restricted-Python REPL that shares the suite pool.
Every successful command is recorded in the event stream and appears in the
generated script via its ``command_src`` field.

Public API
----------
* :func:`get_console_offcanvas` — returns the `dbc.Offcanvas` layout fragment.
* :func:`get_console_toggle_btn` — returns the "Console" nav button.
* :func:`register_console_callbacks` — wires all callbacks onto *app*.
"""
from __future__ import annotations

from dash import html, dcc, Input, Output, State, no_update, clientside_callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from cryocat.app import ids


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _entry_div(entry: dict) -> html.Div:
    """Render one history entry (command + result/error)."""
    cmd = entry.get("cmd", "")
    summary = entry.get("summary", "")
    ok = entry.get("ok", True)

    cmd_color = "var(--bs-info)" if ok else "var(--bs-danger)"
    return html.Div(
        [
            html.Span(
                f">>> {cmd}",
                style={
                    "fontFamily": "monospace",
                    "fontSize": "0.82rem",
                    "color": cmd_color,
                    "display": "block",
                },
            ),
            html.Span(
                summary,
                style={
                    "fontFamily": "monospace",
                    "fontSize": "0.78rem",
                    "color": "var(--bs-secondary-color)",
                    "display": "block",
                    "paddingLeft": "1.5rem",
                    "whiteSpace": "pre-wrap",
                },
            ) if summary else None,
        ],
        style={"marginBottom": "0.25rem"},
    )


def get_console_offcanvas(prefix: str) -> list:
    """Return the console offcanvas and its backing store.

    Place the result in the app layout at the top level (alongside
    ``dcc.Location``, pool stores, etc.) so it is always mounted.
    """
    return [
        dcc.Store(id=f"{prefix}-history", data=[]),
        dbc.Offcanvas(
            html.Div(
                [
                    # Output pane
                    html.Div(
                        id=f"{prefix}-output",
                        style={
                            "height": "160px",
                            "overflowY": "auto",
                            "background": "var(--bs-dark-bg-subtle)",
                            "borderRadius": "4px",
                            "padding": "0.4rem 0.6rem",
                            "marginBottom": "0.4rem",
                            "fontFamily": "monospace",
                        },
                    ),
                    # Suggestions
                    html.Div(
                        id=f"{prefix}-suggestions",
                        style={
                            "fontSize": "0.78rem",
                            "color": "var(--bs-secondary-color)",
                            "minHeight": "1.2rem",
                            "marginBottom": "0.3rem",
                        },
                    ),
                    # Input row
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText(
                                ">>>",
                                style={
                                    "fontFamily": "monospace",
                                    "fontSize": "0.85rem",
                                    "padding": "0.2rem 0.5rem",
                                },
                            ),
                            dbc.Input(
                                id=f"{prefix}-input",
                                type="text",
                                placeholder="type a command or 'help'",
                                debounce=False,
                                n_submit=0,
                                style={"fontFamily": "monospace", "fontSize": "0.85rem"},
                                autoComplete="off",
                                autoFocus=True,
                            ),
                        ],
                        size="sm",
                    ),
                ],
                style={"display": "flex", "flexDirection": "column", "height": "100%"},
            ),
            id=f"{prefix}-offcanvas",
            title="Console",
            placement="bottom",
            is_open=False,
            style={"height": "280px"},
            backdrop=False,
            scrollable=False,
        ),
    ]


def get_console_toggle_btn(prefix: str) -> dbc.Button:
    """Return the toggle button to place in the suite nav bar."""
    return dbc.Button(
        "Console",
        id=f"{prefix}-open-btn",
        color="secondary",
        size="sm",
        style={"alignSelf": "center", "marginRight": "0.5rem"},
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def register_console_callbacks(app, prefix: str) -> None:
    """Register all console callbacks on *app*.

    Parameters
    ----------
    prefix : str
        The same prefix passed to :func:`get_console_offcanvas` and
        :func:`get_console_toggle_btn`.
    """

    # -- Toggle offcanvas -------------------------------------------------------
    @app.callback(
        Output(f"{prefix}-offcanvas", "is_open"),
        Input(f"{prefix}-open-btn", "n_clicks"),
        State(f"{prefix}-offcanvas", "is_open"),
        prevent_initial_call=True,
    )
    def _toggle(n, is_open):
        return not is_open

    # -- Render history into output pane ----------------------------------------
    @app.callback(
        Output(f"{prefix}-output", "children"),
        Input(f"{prefix}-history", "data"),
    )
    def _render_output(history):
        history = history or []
        return [_entry_div(e) for e in history[-30:]]

    # -- Suggestions on input change --------------------------------------------
    @app.callback(
        Output(f"{prefix}-suggestions", "children"),
        Input(f"{prefix}-input", "value"),
        State(ids.POOL_REGISTRY, "data"),
        prevent_initial_call=True,
    )
    def _suggest(text, registry):
        if not text or len(text) < 2:
            return []
        try:
            from cryocat.app.console.help import suggest
            from cryocat.app.pool import PoolState
            state = PoolState.from_stores(registry or {}, {}, {}, {}, 0)
            suggs = suggest(text, state)
            if not suggs:
                return []
            spans = []
            for s in suggs[:8]:
                spans.append(
                    html.Span(
                        s.text,
                        style={
                            "marginRight": "0.6rem",
                            "padding": "0 0.2rem",
                            "background": "var(--bs-secondary-bg)",
                            "borderRadius": "2px",
                            "cursor": "default",
                        },
                        title=f"{s.kind}: {s.detail}",
                    )
                )
            return spans
        except Exception:
            return []

    # -- Submit command ---------------------------------------------------------
    @app.callback(
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_MOTLS, "data", allow_duplicate=True),
        Output(ids.POOL_EXTRA, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output(f"{prefix}-history", "data"),
        Output(f"{prefix}-input", "value"),
        Input(f"{prefix}-input", "n_submit"),
        State(f"{prefix}-input", "value"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_MOTLS, "data"),
        State(ids.POOL_EXTRA, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        State(f"{prefix}-history", "data"),
        prevent_initial_call=True,
    )
    def _on_submit(
        n_submit, text,
        registry, motls, extra, meta, next_id,
        history,
    ):
        if not text or not text.strip():
            raise PreventUpdate

        raw = text.strip()
        history = list(history or [])

        from cryocat.app.console import parse as _parse, execute as _exec
        from cryocat.app.pool import PoolState

        state = PoolState.from_stores(registry, motls, extra, meta, next_id)

        # -- Parse -------------------------------------------------------------
        try:
            cmd = _parse.parse(raw)
        except (_parse.ConsoleSyntaxError, _parse.ConsoleRejected) as exc:
            history.append({"cmd": raw, "summary": str(exc), "ok": False})
            return (
                no_update, no_update, no_update, no_update, no_update,
                history, "",
            )

        # -- Execute -----------------------------------------------------------
        result = _exec.execute(cmd, state)
        history.append({"cmd": raw, "summary": result.summary, "ok": result.ok})

        # -- Pool updates ------------------------------------------------------
        if result.new_state is not state:
            reg, motls_d, extra_d, meta_d, nid = result.new_state.to_stores()
        else:
            reg = motls_d = extra_d = meta_d = nid = no_update

        return reg, motls_d, extra_d, meta_d, nid, history, ""
