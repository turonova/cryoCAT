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
        # dcc.Store(id=f"kmeans-cluster-data-store"),  # storing k means clustering info
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
        dcc.Interval(id="log-check", interval=1000, n_intervals=0),
        dcc.Store(id="log-index", data=0),
        dcc.Store(id="log-content", data=""),
        dbc.Offcanvas(
            html.Pre(id="log-output"),
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
    # Input("log-check", "n_intervals"),
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
