from cryocat.app.logger import dash_logger

import dash
from dash import html, dcc
import numpy as np
import sys
from cryocat.app.layout.sidebar import get_column_sidebar
from cryocat.app.layout.table import get_main_content
import cryocat.app.callbacks
import dash_bootstrap_components as dbc
from dash import callback, Input, Output, State, html, dcc, ALL, ctx, no_update
from cryocat.app.layout.tomoview import get_viewer_component, register_viewer_callbacks


dash.register_page(__name__, path="/", name="Tango")

layout = html.Div(
    [
        dcc.Store(id="main-motl-data-store"),
        dcc.Store(id="nn-motl-data-store"),
        dcc.Store(id="main-motl-extra-data-store"),  # storing stopgap|relion|dynamo_df for the main motl
        dcc.Store(id="nn-motl-extra-data-store"),  # storing stopgap|relion|dynamo_df for the nn motl
        dcc.Store(id="main-motl-data-type"),  # stores motl type of the main motl
        dcc.Store(id="nn-motl-data-type"),  # stores motl type of the nn motl
        dcc.Store(id="tviewer-motl-data"),
        dcc.Store(id="tviewer-motl-index", data=0),
        dcc.Store(id="tviewer-motl-nn-data"),
        dcc.Store(id="tviewer-motl-nn-index", data=0),
        dcc.Store(id="tviewer-twist-data"),
        dcc.Store(id="tviewer-twist-index", data=0),
        dcc.Store(id="tviewer-kmeans-data"),
        dcc.Store(id="tviewer-kmeans-index", data=0),
        dcc.Store(id="tviewer-proximity-data"),
        dcc.Store(id="tviewer-proximity-index", data=0),
        dcc.Store(id="merged-motl-twist-data-store"),
        dcc.Store(id="merged-motl-proximity-data-store"),  # storing proximity clustering info in class of motl
        dcc.Store(id="merged-motl-kmeans-data-store"),  # storing k means clustering info in class of motl
        dcc.Store(id="tviewer-desc-data"),
        dcc.Store(id="tviewer-desc-index", data=0),
        dcc.Store(id="twist-global-radius"),  # to store radius
        dcc.Store(id="main-relion-optics-store"),  # storing optics data for relion main motl
        dcc.Store(id="nn-relion-optics-store"),  # storing optics data for relion nn motl
        dcc.Store(id="main-relion5-tomos-store"),  # storing tomogram file content for relion5 main motl
        dcc.Store(id="nn-relion5-tomos-store"),  # storing tomogram file content for relion5 nn motl
        dcc.Store(id="main-relion5-tomos-filename"),  # storing tomogram file name for relion5 main motl
        dcc.Store(id="nn-relion5-tomos-filename"),  # storing tomogram file name for relion5 nn motl
        dcc.Store(id="main-relion-params-store"),  # relion loading params (pixel_size, binning, formats)
        dcc.Store(id="nn-relion-params-store"),  # relion loading params for nn motl
        dcc.Store(id="save-main-relion5-tomos-store"),  # storing tomogram file content for saving output
        dcc.Store(id="save-main-relion5-tomos-filename"),  # storing tomogram file name for relion5 for saving output
        dcc.Store(id=f"kmeans-global-data-store"),  # storing k means data
        dcc.Store(id=f"tabv-motl-global-data-store"),  # main motl table
        dcc.Store(id=f"tabv-motl-nn-global-data-store"),  # nn motl table if available
        dcc.Store(id=f"tabv-nn-global-data-store"),  # nearest neighbor table
        dcc.Store(id=f"tabv-twist-global-data-store"),  # twis descriptor table
        dcc.Store(id=f"tabv-desc-global-data-store"),  # additional descriptor table
        dbc.Row(
            [
                # Sidebar
                get_column_sidebar(),
                # Main content
                get_main_content(),
            ],
            className="g-0",
            style={"margin": "0", "padding": "0"},
        ),
        # dcc.Interval(id="log-check", interval=1000, n_intervals=0),
        dcc.Store(id="log-index", data=0),
        dcc.Store(id="log-content", data=""),
        dcc.Store(id="log-save-path-store"),
        dbc.Offcanvas(
            [
                html.Div(
                    [
                        dbc.Button("Save", id="log-save-btn", color="secondary", size="sm", className="me-1"),
                        dbc.Button("Save As", id="log-save-as-btn", color="primary", size="sm"),
                        html.Span(id="log-save-status",
                                  style={"marginLeft": "0.75rem", "fontSize": "0.8rem", "color": "grey"}),
                    ],
                    style={"display": "flex", "alignItems": "center", "marginBottom": "0.5rem"},
                ),
                html.Hr(style={"margin": "0.5rem 0"}),
                html.Pre(id="log-output"),
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Save Log As")),
                        dbc.ModalBody(
                            dbc.Input(id="log-save-path-input", type="text",
                                      placeholder="Full path including extension (e.g. /path/log.txt)"),
                        ),
                        dbc.ModalFooter([
                            html.Span(id="log-saveas-status",
                                      style={"marginRight": "auto", "fontSize": "0.8rem", "color": "grey"}),
                            dbc.Button("Save", id="log-saveas-confirm-btn", color="primary"),
                        ]),
                    ],
                    id="log-save-as-modal",
                    is_open=False,
                    centered=True,
                ),
            ],
            id="log-panel",
            title="Log Output",
            placement="end",
            scrollable=True,
            style={"width": "500px"},
            is_open=False,
        ),
    ],
    style={"margin": "0", "padding": "0"},
)


@callback(
    Output("log-output", "children"),
    Output("log-index", "data"),
    Output("log-panel", "is_open"),
    Input("open-log-btn", "n_clicks"),
    State("log-index", "data"),
    State("log-panel", "is_open"),
    prevent_initial_call=True,
)
def update_log(open_clicks, last_index, is_open):
    triggered = ctx.triggered_id
    new_logs, new_index, has_dash_logs = dash_logger.get_logs(last_index)
    full_log = dash_logger.get_all_logs()

    if triggered == "open-log-btn":
        return full_log, new_index, True

    if has_dash_logs:
        return new_logs, new_index, True

    raise dash.exceptions.PreventUpdate


@callback(
    Output("log-save-as-modal", "is_open"),
    Input("log-save-as-btn", "n_clicks"),
    Input("log-save-btn", "n_clicks"),
    State("log-save-path-store", "data"),
    prevent_initial_call=True,
)
def open_log_save_as(_, _2, saved_path):
    if ctx.triggered_id == "log-save-as-btn":
        return True
    if ctx.triggered_id == "log-save-btn" and not saved_path:
        return True
    return no_update


@callback(
    Output("log-save-as-modal", "is_open", allow_duplicate=True),
    Output("log-saveas-status", "children"),
    Output("log-save-path-store", "data"),
    Output("log-save-status", "children"),
    Input("log-saveas-confirm-btn", "n_clicks"),
    State("log-save-path-input", "value"),
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


@callback(
    Output("log-save-status", "children", allow_duplicate=True),
    Input("log-save-btn", "n_clicks"),
    State("log-save-path-store", "data"),
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
