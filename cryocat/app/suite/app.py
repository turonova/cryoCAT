"""Suite Dash app — multi-tool workspace built around a shared motl pool.

This module owns:
  * the suite-global **motl pool** stores (``pool-*``) — declared at app level
    so they survive tool/route changes and are readable by every tool;
  * the **router** — swaps page content based on the URL;
  * the **tool selector** — top-nav pills rendered from the ``TOOLS`` registry.

Each tool is a page module exposing ``layout`` (attribute) and
``register_callbacks(app)``. See :mod:`cryocat.app.suite.tools`.
"""

import importlib

import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

from cryocat.app.suite.tools import TOOLS, DEFAULT_PATH
from cryocat.app.components.graphsettings import (
    get_graph_settings_components,
    register_graph_settings_callbacks,
)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    requests_pathname_prefix="/",
    routes_pathname_prefix="/",
    suppress_callback_exceptions=True,
)


# ── Suite-global motl pool ──────────────────────────────────────────────────────
# The pool is the shared spine across all tools: a small, soft-sized set of
# loaded motls. Declared here (app level) so the data persists across route
# changes and every tool can read/write it. ``motl_id`` is a stable string key
# (``motl-<n>`` using the ``pool-next-id`` counter) — there is no fixed slot cap.
POOL_STORES = [
    dcc.Store(id="pool-registry", data={}),  # { motl_id: {label, type, n_rows, active} }
    dcc.Store(id="pool-motls", data={}),     # { motl_id: <serialized motl rows> }
    dcc.Store(id="pool-extra", data={}),     # { motl_id: <stopgap/relion/dynamo extra df> }
    dcc.Store(id="pool-meta", data={}),      # { motl_id: <relion params, data_type, ...> }
    dcc.Store(id="pool-next-id", data=0),    # incrementing counter for stable motl_id
]


# ── Tool pages ──────────────────────────────────────────────────────────────────
_PAGES = {t["path"]: importlib.import_module(t["module"]) for t in TOOLS}


def _normalize(pathname):
    """Strip a trailing slash; map empty / root / unknown to the default path."""
    if not pathname:
        return DEFAULT_PATH
    path = pathname.rstrip("/")
    return path or DEFAULT_PATH


def _tool_selector(active_path):
    """Top-nav pills, rendered from the TOOLS registry."""
    return dbc.Nav(
        [
            dbc.NavLink(t["label"], href=t["path"], active=(t["path"] == active_path))
            for t in TOOLS
        ],
        pills=True,
        className="p-2",
    )


# ── Layout ──────────────────────────────────────────────────────────────────────
app.layout = dbc.Container(
    [
        dcc.Location(id="suite-url"),
        *POOL_STORES,
        *get_graph_settings_components(),
        html.Div(id="suite-tool-selector"),
        html.Div(id="suite-page-content"),
    ],
    fluid=True,
    className="p-0",
)


# ── Router ──────────────────────────────────────────────────────────────────────
@app.callback(
    Output("suite-page-content", "children"),
    Output("suite-tool-selector", "children"),
    Input("suite-url", "pathname"),
)
def _route(pathname):
    path = _normalize(pathname)
    page = _PAGES.get(path) or _PAGES[DEFAULT_PATH]
    return page.layout, _tool_selector(path)


# ── Callback registration ───────────────────────────────────────────────────────
# Registered once, up front. Inert until the relevant tool's IDs render
# (suppress_callback_exceptions=True covers callbacks for not-yet-rendered pages).
register_graph_settings_callbacks(app)
for _t in TOOLS:
    _PAGES[_t["path"]].register_callbacks(app)


if __name__ == "__main__":
    app.run(debug=True)
