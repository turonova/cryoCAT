from dash import html, dcc, Input, Output, State, no_update, ctx
import dash
import dash_bootstrap_components as dbc
from cryocat.app.logger import dash_logger


def get_log_panel(prefix: str):
    """Returns the log offcanvas + open button store components.

    prefix should be 'log' for Tango or 'me-log' for Suite.
    """
    return [
        dcc.Store(id=f"{prefix}-index", data=0),
        dcc.Store(id=f"{prefix}-save-path-store"),
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
                html.Pre(id=f"{prefix}-output"),
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


def register_log_panel_callbacks(app, prefix: str):
    """Register the 4 log callbacks with the given app and prefix.

    For Tango: prefix='log', open button id='open-log-btn'
    For Suite: prefix='me-log', open button id='me-open-log-btn'
    """
    # Determine the open button id based on convention
    if prefix == "log":
        open_btn_id = "open-log-btn"
    else:
        open_btn_id = f"{prefix.replace('log', 'open-log')}-btn"

    @app.callback(
        Output(f"{prefix}-output", "children"),
        Output(f"{prefix}-index", "data"),
        Output(f"{prefix}-panel", "is_open"),
        Input(open_btn_id, "n_clicks"),
        State(f"{prefix}-index", "data"),
        State(f"{prefix}-panel", "is_open"),
        prevent_initial_call=True,
    )
    def update_log(open_clicks, last_index, is_open):
        triggered = ctx.triggered_id
        new_logs, new_index, has_dash_logs = dash_logger.get_logs(last_index)
        full_log = dash_logger.get_all_logs()

        if triggered == open_btn_id:
            return full_log, new_index, True

        if has_dash_logs:
            return new_logs, new_index, True

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
