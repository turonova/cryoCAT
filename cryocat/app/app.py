import sys
from cryocat.app.logger import dash_logger

sys.stdout = dash_logger

import dash

from dash import html, dcc
import dash_bootstrap_components as dbc


app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.MINTY],
    suppress_callback_exceptions=True,
)

app.layout = dbc.Container(
    [dcc.Location(id="url"), dash.page_container],
    fluid=True,
    className="p-0",
)

server = app.server


def tango_app():
    app.run()


if __name__ == "__main__":
    app.run(debug=True)
