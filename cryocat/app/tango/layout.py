from cryocat.app.logger import dash_logger

from dash import html, dcc
import dash_bootstrap_components as dbc

from cryocat.app.tango.sidebar import get_column_sidebar
from cryocat.app.tango.table import get_main_content
from cryocat.app.components.logpanel import get_log_panel, register_log_panel_callbacks


layout = html.Div(
    [
        dcc.Store(id="main-motl-data-store"),
        dcc.Store(id="nn-motl-data-store"),
        dcc.Store(id="main-motl-extra-data-store"),
        dcc.Store(id="nn-motl-extra-data-store"),
        dcc.Store(id="main-motl-data-type"),
        dcc.Store(id="nn-motl-data-type"),
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
        dcc.Store(id="merged-motl-proximity-data-store"),
        dcc.Store(id="merged-motl-kmeans-data-store"),
        dcc.Store(id="tviewer-desc-data"),
        dcc.Store(id="tviewer-desc-index", data=0),
        dcc.Store(id="twist-global-radius"),
        dcc.Store(id="main-relion-optics-store"),
        dcc.Store(id="nn-relion-optics-store"),
        dcc.Store(id="main-relion5-tomos-store"),
        dcc.Store(id="nn-relion5-tomos-store"),
        dcc.Store(id="main-relion5-tomos-filename"),
        dcc.Store(id="nn-relion5-tomos-filename"),
        dcc.Store(id="main-relion-params-store"),
        dcc.Store(id="nn-relion-params-store"),
        dcc.Store(id="save-main-relion5-tomos-store"),
        dcc.Store(id="save-main-relion5-tomos-filename"),
        dcc.Store(id="kmeans-global-data-store"),
        dcc.Store(id="tabv-motl-global-data-store"),
        dcc.Store(id="tabv-motl-nn-global-data-store"),
        dcc.Store(id="tabv-nn-global-data-store"),
        dcc.Store(id="tabv-twist-global-data-store"),
        dcc.Store(id="tabv-desc-global-data-store"),
        dbc.Row(
            [
                get_column_sidebar(),
                get_main_content(),
            ],
            className="g-0",
            style={"margin": "0", "padding": "0"},
        ),
        *get_log_panel("log"),
    ],
    style={"margin": "0", "padding": "0"},
)


def register_tango_log_callbacks(app):
    register_log_panel_callbacks(app, "log")
