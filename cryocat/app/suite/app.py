"""Suite Dash app — multi-tool workspace built around a shared motl pool.

This module owns:
  * the suite-global **motl pool** stores (``pool-*``) — declared at app level
    so they survive tool/route changes and are readable by every tool;
  * the **router** — CSS-toggles which pre-mounted page is visible (no DOM
    destruction on navigation);
  * the **tool selector** — top-nav pills rendered from the ``TOOLS`` registry.

Each tool is a page module exposing ``layout`` (attribute) and
``register_callbacks(app)``. See :mod:`cryocat.app.suite.tools`.
"""

import importlib
import os

import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

from cryocat.app.suite.tools import TOOLS, DEFAULT_PATH
from cryocat.app.components.graphsettings import (
    get_graph_settings_components,
    register_graph_settings_callbacks,
)
from cryocat.app.components.logpanel import get_log_panel, register_log_panel_callbacks
from cryocat.app.logger import dash_logger

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
_PAGES = {t["id"]: importlib.import_module(t["module"]) for t in TOOLS}
_DEFAULT_ID = next(t["id"] for t in TOOLS if t["path"] == DEFAULT_PATH)


def _resolve_active_tool(pathname: str) -> str:
    """Map a URL pathname to a tool id; unknown paths fall back to the default."""
    if not pathname:
        return _DEFAULT_ID
    path = pathname.rstrip("/") or DEFAULT_PATH
    for t in TOOLS:
        if t["path"] == path:
            return t["id"]
    return _DEFAULT_ID


def _tool_selector(active_id: str):
    """Top-nav pills, rendered from the TOOLS registry."""
    return dbc.Nav(
        [
            dbc.NavLink(t["label"], href=t["path"], active=(t["id"] == active_id))
            for t in TOOLS
        ],
        pills=True,
        className="suite-nav",
    )


def _page_wrappers():
    """Build all page layouts, each in a stable Div; all hidden initially."""
    return [
        html.Div(
            _PAGES[t["id"]].layout,
            id=f"page-wrap-{t['id']}",
            style={"display": "none"},
        )
        for t in TOOLS
    ]


# ── Layout ──────────────────────────────────────────────────────────────────────
app.layout = dbc.Container(
    [
        dcc.Location(id="suite-url"),
        *POOL_STORES,
        *get_graph_settings_components(),
        *get_log_panel("suite-log"),
        html.Div(
            [
                html.Div(id="suite-tool-selector", style={"flex": "1"}),
                dbc.Button(
                    "Show log",
                    id="suite-open-log-btn",
                    color="secondary",
                    size="sm",
                    style={"alignSelf": "center", "marginRight": "0.5rem"},
                ),
            ],
            className="suite-nav-bar",
            style={"display": "flex", "alignItems": "center"},
        ),
        html.Div(_page_wrappers(), id="suite-page-content"),
    ],
    fluid=True,
    className="p-0",
)


# ── Router ──────────────────────────────────────────────────────────────────────
# All pages are mounted once at startup. Navigation only toggles display style —
# no React tree is destroyed, so in-page state is fully preserved across routes.
_route_outputs = [Output(f"page-wrap-{t['id']}", "style") for t in TOOLS] + [
    Output("suite-tool-selector", "children")
]


@app.callback(*_route_outputs, Input("suite-url", "pathname"))
def _route(pathname):
    active_id = _resolve_active_tool(pathname)
    styles = [
        {"display": "block"} if t["id"] == active_id else {"display": "none"}
        for t in TOOLS
    ]
    return *styles, _tool_selector(active_id)


# ── Callback registration ───────────────────────────────────────────────────────
# Registered once, up front. All page IDs are live from the start because every
# page is mounted in the layout — suppress_callback_exceptions is still set as a
# safety net but is no longer strictly required.
dash_logger.start_session(os.path.expanduser("~/.cryocat/sessions"))

register_graph_settings_callbacks(app)
register_log_panel_callbacks(app, "suite-log", open_btn_id="suite-open-log-btn")
for _t in TOOLS:
    _PAGES[_t["id"]].register_callbacks(app)


if __name__ == "__main__":
    app.run(debug=True)
