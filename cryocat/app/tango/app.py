import dash
import dash_bootstrap_components as dbc

from cryocat.app.tango.layout import layout, register_tango_log_callbacks
from cryocat.app.tango.sidebar import register_tango_sidebar_callbacks
from cryocat.app.tango.table import register_tango_table_callbacks
from cryocat.app.tango.callbacks import register_tango_callbacks
from cryocat.app.components.graphsettings import register_graph_settings_callbacks

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.MINTY],
    requests_pathname_prefix="/tango/",
    routes_pathname_prefix="/tango/",
    suppress_callback_exceptions=True,
)

app.layout = layout

register_tango_sidebar_callbacks(app)
register_tango_table_callbacks(app)
register_tango_callbacks(app)
register_tango_log_callbacks(app)
register_graph_settings_callbacks(app)
