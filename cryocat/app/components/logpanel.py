from dash import html, dcc, Input, Output, State, no_update, ctx
import dash
import dash_bootstrap_components as dbc
from cryocat.app.logger import dash_logger

_SOURCE_STYLE = {
    "error":  {"color": "#e05252", "fontWeight": "bold"},
    "cryocat": {"color": "var(--color9)"},
    "dash":   {"color": "#888888"},
}


def _entry_spans(entries):
    """Convert [(msg, source), ...] to a list of html.Div elements."""
    spans = []
    for msg, source in entries:
        style = dict(_SOURCE_STYLE.get(source, {}))
        style["whiteSpace"] = "pre-wrap"
        style["wordBreak"] = "break-word"
        style["marginBottom"] = "2px"
        style["fontFamily"] = "monospace"
        style["fontSize"] = "0.82rem"
        spans.append(html.Div(msg, style=style))
    return spans


def get_log_panel(prefix: str):
    """Returns the log offcanvas + stores + poll interval as a list of components."""
    return [
        dcc.Store(id=f"{prefix}-index", data=0),
        dcc.Store(id=f"{prefix}-save-path-store"),
        dcc.Interval(id=f"{prefix}-poll", interval=3000, n_intervals=0),
        dbc.Offcanvas(
            [
                html.Div(
                    [
                        dbc.Button("Save", id=f"{prefix}-save-btn", color="secondary", size="sm", className="me-1"),
                        dbc.Button("Save As", id=f"{prefix}-save-as-btn", color="primary", size="sm"),
                        html.Span(id=f"{prefix}-save-status",
                                  style={"marginLeft": "0.75rem", "fontSize": "0.8rem", "color": "grey"}),
                    ],
                    style={"display": "flex", "alignItems": "center", "marginBottom": "0.5rem"},
                ),
                html.Hr(style={"margin": "0.5rem 0"}),
                html.Div(id=f"{prefix}-output", style={"overflowY": "auto"}),
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Save Log As")),
                        dbc.ModalBody(
                            dbc.Input(id=f"{prefix}-save-path-input", type="text",
                                      placeholder="Full path including extension (e.g. /path/log.txt)"),
                        ),
                        dbc.ModalFooter([
                            html.Span(id=f"{prefix}-saveas-status",
                                      style={"marginRight": "auto", "fontSize": "0.8rem", "color": "grey"}),
                            dbc.Button("Save", id=f"{prefix}-saveas-confirm-btn", color="primary"),
                        ]),
                    ],
                    id=f"{prefix}-save-as-modal",
                    is_open=False,
                    centered=True,
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


def register_log_panel_callbacks(app, prefix: str, open_btn_id: str | None = None):
    """Register log panel callbacks.

    Parameters
    ----------
    open_btn_id : str or None
        ID of the button that opens the panel.  When None the panel can only be
        opened automatically (on error) or via the poll interval.
    """
    inputs = [Input(f"{prefix}-poll", "n_intervals")]
    if open_btn_id is not None:
        inputs.append(Input(open_btn_id, "n_clicks"))

    @app.callback(
        Output(f"{prefix}-output", "children"),
        Output(f"{prefix}-index", "data"),
        Output(f"{prefix}-panel", "is_open"),
        *inputs,
        State(f"{prefix}-index", "data"),
        State(f"{prefix}-panel", "is_open"),
        prevent_initial_call=True,
    )
    def update_log(*cb_args):
        # Unpack positional args: (n_intervals [, n_clicks], last_index, is_open)
        last_index = cb_args[-2]
        is_open = cb_args[-1]

        triggered = ctx.triggered_id
        entries, new_index, new_dash, new_error = dash_logger.get_logs(last_index)

        # Button click → show full log, open panel
        if open_btn_id is not None and triggered == open_btn_id:
            all_entries, all_index, _, _ = dash_logger.get_logs(0)
            return _entry_spans(all_entries), all_index, True

        # Auto-open on error; update display when there are new entries
        if new_error:
            all_entries, all_index, _, _ = dash_logger.get_logs(0)
            return _entry_spans(all_entries), all_index, True

        if new_dash or entries:
            all_entries, all_index, _, _ = dash_logger.get_logs(0)
            return _entry_spans(all_entries), all_index, is_open

        raise dash.exceptions.PreventUpdate

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
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(dash_logger.get_all_logs())
            return False, "Saved.", path, f"Saved to {path}"
        except Exception as e:
            return no_update, str(e), no_update, no_update

    @app.callback(
        Output(f"{prefix}-save-status", "children", allow_duplicate=True),
        Input(f"{prefix}-save-btn", "n_clicks"),
        State(f"{prefix}-save-path-store", "data"),
        prevent_initial_call=True,
    )
    def save_log(_, path):
        if not path:
            return no_update
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(dash_logger.get_all_logs())
            return f"Saved to {path}"
        except Exception as e:
            return str(e)
