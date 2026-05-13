import sys
from cryocat.app.logger import dash_logger, patch_class, patch_function

sys.stdout = dash_logger

import dash

from dash import html, dcc
import dash_bootstrap_components as dbc

from cryocat.app.layout.graphsettings import get_graph_settings_components, register_graph_settings_callbacks
from cryocat.core.cryomotl import Motl
from cryocat.app import apputils

# Auto-log all public Motl methods (load is excluded: manually logged with the
# original filename instead of the temp path).
patch_class(Motl, exclude=('load',))
patch_function(apputils, 'save_motl')

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.MINTY],
    suppress_callback_exceptions=True,
)


app.layout = dbc.Container(
    [dcc.Location(id="url"), *get_graph_settings_components(), dash.page_container],
    fluid=True,
    className="p-0",
)

register_graph_settings_callbacks()

server = app.server


def tango_app():
    app.run()


if __name__ == "__main__":
    app.run(debug=True)
