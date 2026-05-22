import dash
from dash import html, dcc, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc

from cryocat.app.suite.motlsidebar import get_motl_editor_sidebar, register_motl_editor_sidebar_callbacks, MAX_MOTLS
from cryocat.app.components.tomoview import get_viewer_component, register_viewer_callbacks
from cryocat.app.components.tableview import get_table_component, register_table_callbacks
from cryocat.app.components.tableplot import register_table_plot_callbacks
from cryocat.app.components.motlio import get_motl_simple_save_component, register_motl_simple_save_callbacks
from cryocat.app.components.logpanel import get_log_panel, register_log_panel_callbacks
from cryocat.app.apputils import _scalar, _format_relion_params
from cryocat.app.logger import dash_logger


# ── Stores ────────────────────────────────────────────────────────────────────

def _make_stores():
    stores = [
        dcc.Store(id="motls-registry", data={}),
        dcc.Store(id="me-results-store"),
        dcc.Store(id="me-results-label-store"),
        dcc.Store(id="me-load-motl-data-store"),
        dcc.Store(id="me-load-motl-extra-data-store"),
        dcc.Store(id="me-load-motl-data-type"),
        dcc.Store(id="me-load-relion-optics-store"),
        dcc.Store(id="me-load-relion5-tomos-store"),
        dcc.Store(id="me-load-relion5-tomos-filename"),
        dcc.Store(id="me-load-relion-params-store"),
        dcc.Store(id="me-res-tv-data"),
        dcc.Store(id="me-res-tv-index", data=0),
        dcc.Store(id="me-res-tabv-global-data-store"),
        dcc.Store(id="me-res-motl-data-type"),
        dcc.Store(id="me-res-motl-extra-data-store"),
        dcc.Store(id="me-res-relion-optics-store"),
        dcc.Store(id="me-res-relion5-tomos-store"),
        dcc.Store(id="me-res-relion5-tomos-filename"),
    ]
    for i in range(MAX_MOTLS):
        stores += [
            dcc.Store(id=f"me-{i}-motl-data-store"),
            dcc.Store(id=f"me-{i}-motl-extra-data-store"),
            dcc.Store(id=f"me-{i}-motl-data-type"),
            dcc.Store(id=f"me-{i}-relion-optics-store"),
            dcc.Store(id=f"me-{i}-relion5-tomos-store"),
            dcc.Store(id=f"me-{i}-relion5-tomos-filename"),
            dcc.Store(id=f"me-{i}-relion-params-store"),
            dcc.Store(id=f"me-{i}-tv-data"),
            dcc.Store(id=f"me-{i}-tv-index", data=0),
            dcc.Store(id=f"me-{i}-tabv-global-data-store"),
            dcc.Store(id=f"me-{i}-undo-store"),
        ]
    return stores


# ── Per-slot tab content ───────────────────────────────────────────────────────

def _slot_tab_content(i):
    return dbc.Tab(
        id=f"me-tab-slot-{i}",
        tab_id=f"me-tab-{i}",
        label=f"Slot {i + 1}",
        disabled=True,
        children=html.Div(
            [
                get_table_component(f"me-{i}-tabv", connected_motl_prefix=f"me-{i}"),
                html.Hr(style={"margin": "0.5rem 0"}),
                get_viewer_component(f"me-{i}-tv"),
                html.Hr(style={"margin": "0.5rem 0"}),
                get_motl_simple_save_component(f"me-{i}-save"),
            ],
            style={"padding": "0.5rem"},
        ),
    )


def _results_tab_content():
    return dbc.Tab(
        id="me-tab-results",
        tab_id="me-tab-results",
        label="Results",
        disabled=True,
        children=html.Div(
            [
                get_table_component("me-res-tabv", connected_motl_prefix="me-res", show_create_from_selected=False),
                html.Hr(style={"margin": "0.5rem 0"}),
                get_viewer_component("me-res-tv"),
            ],
            style={"padding": "0.5rem"},
        ),
    )


def _get_main_content():
    tabs = [_slot_tab_content(i) for i in range(MAX_MOTLS)]
    tabs.append(_results_tab_content())
    return dbc.Col(
        dbc.Tabs(
            id="me-tabs",
            active_tab=f"me-tab-0",
            children=tabs,
            style={"padding": "0.5rem"},
        ),
        width=9,
        style={"margin": "0", "padding": "0"},
    )


# ── Page layout ────────────────────────────────────────────────────────────────

layout = html.Div(
    [
        *_make_stores(),
        dbc.Row(
            [
                get_motl_editor_sidebar(),
                _get_main_content(),
            ],
            className="g-0",
            style={"margin": "0", "padding": "0"},
        ),
        *get_log_panel("me-log"),
    ],
    style={"margin": "0", "padding": "0"},
)


# ── Callback registration ──────────────────────────────────────────────────────

def register_callbacks(app):

    register_motl_editor_sidebar_callbacks(app, MAX_MOTLS)

    # Register viewer, table, plot, and save callbacks for each slot.
    for _i in range(MAX_MOTLS):
        register_viewer_callbacks(app, f"me-{_i}-tv", tabs_id=None)
        register_table_callbacks(
            app,
            f"me-{_i}-tabv",
            csv_only=False,
            connected_motl_prefix=f"me-{_i}",
            slot_idx=_i,
            max_motls=MAX_MOTLS,
        )
        register_table_plot_callbacks(app, f"me-{_i}-tabv-table-plot", f"me-{_i}-tabv-global-data-store", table_grid_id=f"me-{_i}-tabv-grid")
        register_motl_simple_save_callbacks(app, f"me-{_i}-save", f"me-{_i}-tabv-global-data-store", f"me-{_i}")

    register_viewer_callbacks(app, "me-res-tv", tabs_id=None)
    register_table_callbacks(app, "me-res-tabv", csv_only=False, connected_motl_prefix="me-res")
    register_table_plot_callbacks(app, "me-res-tabv-table-plot", "me-res-tabv-global-data-store")

    _register_slot_connectors_all(app)
    _register_relion_params_connectors_all(app)

    @app.callback(
        *[Output(f"me-tab-slot-{i}", "label") for i in range(MAX_MOTLS)],
        *[Output(f"me-tab-slot-{i}", "disabled") for i in range(MAX_MOTLS)],
        Input("motls-registry", "data"),
        prevent_initial_call=True,
    )
    def update_tab_labels(registry):
        registry = registry or {}
        labels, disableds = [], []
        for i in range(MAX_MOTLS):
            meta = registry.get(str(i), {})
            if meta.get("active"):
                raw = meta.get("label", f"Motl {i + 1}")
                labels.append(raw[:22] + "…" if len(raw) > 22 else raw)
                disableds.append(False)
            else:
                labels.append(f"Slot {i + 1}")
                disableds.append(True)
        return (*labels, *disableds)

    @app.callback(
        Output("me-tab-results", "disabled"),
        Output("me-res-tv-data", "data", allow_duplicate=True),
        Output("me-res-tabv-global-data-store", "data", allow_duplicate=True),
        Output("me-tab-results", "label"),
        Output("me-tabs", "active_tab", allow_duplicate=True),
        Input("me-results-store", "data"),
        State("me-results-label-store", "data"),
        prevent_initial_call=True,
    )
    def connect_results(data, label):
        if not data:
            raise dash.exceptions.PreventUpdate
        display_label = label or "Results"
        return False, data, data, display_label, "me-tab-results"

    register_log_panel_callbacks(app, "me-log")


# ── Per-slot connecting callbacks ──────────────────────────────────────────────

def _register_slot_connectors_all(app):
    for slot_idx in range(MAX_MOTLS):
        _register_slot_connectors(app, slot_idx)


def _register_slot_connectors(app, slot_idx):
    """Wire the raw motl store → viewer data + table global store for one slot."""

    @app.callback(
        Output(f"me-{slot_idx}-tv-data", "data", allow_duplicate=True),
        Output(f"me-{slot_idx}-tabv-global-data-store", "data", allow_duplicate=True),
        Input(f"me-{slot_idx}-motl-data-store", "data"),
        prevent_initial_call=True,
    )
    def _connect_motl(data, _s=slot_idx):
        return data, data

    @app.callback(
        Output(f"me-{slot_idx}-tv-data", "data", allow_duplicate=True),
        Input(f"me-{slot_idx}-tabv-global-data-store", "data"),
        Input(f"me-{slot_idx}-tabv-grid", "rowData"),
        prevent_initial_call=True,
    )
    def _connect_table_to_viewer(global_data, table_rows, _s=slot_idx):
        if ctx.triggered_id == f"me-{_s}-tabv-grid":
            return table_rows
        return global_data


# ── Per-slot relion params inline display ──────────────────────────────────────

def _register_relion_params_connectors_all(app):
    for slot_idx in range(MAX_MOTLS):
        _register_relion_params_connector(app, slot_idx)


def _register_relion_params_connector(app, slot_idx):
    @app.callback(
        Output(f"me-{slot_idx}-relion-params-inline", "children"),
        Input(f"me-{slot_idx}-relion-params-store", "data"),
        prevent_initial_call=True,
    )
    def _update_inline(params, _s=slot_idx):
        return _format_relion_params(params)
