import dash
import dash_bootstrap_components as dbc

from cryocat.app.suite.pages.pmotl import layout, register_callbacks
from cryocat.app.components.graphsettings import register_graph_settings_callbacks

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    requests_pathname_prefix="/",
    routes_pathname_prefix="/",
    suppress_callback_exceptions=True,
)

app.layout = layout

register_callbacks(app)
register_graph_settings_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)
