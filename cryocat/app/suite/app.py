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

# Start the session stream now so events are captured when running standalone
# (server.py calls this before importing us; start_session() is idempotent).
from cryocat.app import session as _session
_session.start_session()

import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

from cryocat.app import ids
from cryocat.app.suite.tools import TOOLS, DEFAULT_PATH
from cryocat.app.components.graphsettings import (
    get_graph_settings_components,
    register_graph_settings_callbacks,
)
from cryocat.app.components.logpanel import get_log_panel, register_log_panel_callbacks
from cryocat.app.components.consoleui import (
    get_console_offcanvas,
    get_console_toggle_btn,
    register_console_callbacks,
)
from cryocat.app.components.filebrowser import (
    get_file_browser,
    register_file_browser_callbacks,
)
from cryocat.app.components.rotationmodal import (
    get_rotation_modal,
    register_rotation_modal_callbacks,
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
    dcc.Store(id=ids.POOL_REGISTRY, data={}),  # { motl_id: {label, type, n_rows, active} }
    dcc.Store(id=ids.POOL_MOTLS, data={}),     # { motl_id: <serialized motl rows> }
    dcc.Store(id=ids.POOL_EXTRA, data={}),     # { motl_id: <stopgap/relion/dynamo extra df> }
    dcc.Store(id=ids.POOL_META, data={}),      # { motl_id: <relion params, data_type, ...> }
    dcc.Store(id=ids.POOL_NEXT_ID, data=0),    # incrementing counter for stable motl_id
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
            id=ids.page_wrap_id(t["id"]),
            style={"display": "none"},
        )
        for t in TOOLS
    ]


# ── Layout ──────────────────────────────────────────────────────────────────────
app.layout = dbc.Container(
    [
        dcc.Location(id=ids.SUITE_URL),
        *POOL_STORES,
        get_file_browser(),
        get_rotation_modal(),
        *get_graph_settings_components(),
        *get_log_panel("suite-log"),
        *get_console_offcanvas("suite-console"),
        html.Div(
            [
                html.Div(id=ids.SUITE_TOOL_SELECTOR, style={"flex": "1"}),
                get_console_toggle_btn("suite-console"),
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
        html.Div(_page_wrappers(), id=ids.SUITE_PAGE_CONTENT),
    ],
    fluid=True,
    className="p-0",
)


# ── Router ──────────────────────────────────────────────────────────────────────
# All pages are mounted once at startup. Navigation only toggles display style —
# no React tree is destroyed, so in-page state is fully preserved across routes.
_route_outputs = [Output(ids.page_wrap_id(t["id"]), "style") for t in TOOLS] + [
    Output(ids.SUITE_TOOL_SELECTOR, "children")
]


@app.callback(*_route_outputs, Input(ids.SUITE_URL, "pathname"))
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
register_file_browser_callbacks(app)
register_rotation_modal_callbacks(app)
register_graph_settings_callbacks(app)
register_log_panel_callbacks(app, "suite-log", open_btn_id="suite-open-log-btn")
register_console_callbacks(app, "suite-console")
for _t in TOOLS:
    _PAGES[_t["id"]].register_callbacks(app)


if __name__ == "__main__":
    app.run(debug=True)
