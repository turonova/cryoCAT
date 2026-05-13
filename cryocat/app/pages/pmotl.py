import dash
from dash import html, dcc, Input, Output, State, callback, no_update, ctx
import dash_bootstrap_components as dbc

from cryocat.app.layout.motlsidebar import get_motl_editor_sidebar, register_motl_editor_sidebar_callbacks, MAX_MOTLS
from cryocat.app.layout.tomoview import get_viewer_component, register_viewer_callbacks
from cryocat.app.layout.tableview import get_table_component, register_table_callbacks
from cryocat.app.layout.tableplot import register_table_plot_callbacks
from cryocat.app.layout.motlio import get_motl_simple_save_component, register_motl_simple_save_callbacks
from cryocat.app.logger import dash_logger

dash.register_page(__name__, path="/motl", name="Motl Editor")


# ── Stores ────────────────────────────────────────────────────────────────────

def _make_stores():
    stores = [
        # Registry: {str(slot_idx): {label, active}}
        dcc.Store(id="motls-registry", data={}),
        # Multi-motl operation results
        dcc.Store(id="me-results-store"),
        dcc.Store(id="me-results-label-store"),
        # Shared load-form stores (prefix "me-load")
        dcc.Store(id="me-load-motl-data-store"),
        dcc.Store(id="me-load-motl-extra-data-store"),
        dcc.Store(id="me-load-motl-data-type"),
        dcc.Store(id="me-load-relion-optics-store"),
        dcc.Store(id="me-load-relion5-tomos-store"),
        dcc.Store(id="me-load-relion5-tomos-filename"),
        dcc.Store(id="me-load-relion-params-store"),
        # Results viewer / table stores
        dcc.Store(id="me-res-tv-data"),
        dcc.Store(id="me-res-tv-index", data=0),
        dcc.Store(id="me-res-tabv-global-data-store"),
        # Dummy motl-mode stores for the results table save dialog
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
            # Viewer stores for slot i (prefix "me-{i}-tv")
            dcc.Store(id=f"me-{i}-tv-data"),
            dcc.Store(id=f"me-{i}-tv-index", data=0),
            # Table stores for slot i (prefix "me-{i}-tabv")
            dcc.Store(id=f"me-{i}-tabv-global-data-store"),
            # One-level undo store for single-motl operations
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
        dcc.Store(id="me-log-index", data=0),
        dcc.Store(id="me-log-save-path-store"),
        dbc.Offcanvas(
            [
                html.Div(
                    [
                        dbc.Button("Save", id="me-log-save-btn", color="secondary", size="sm", className="me-1"),
                        dbc.Button("Save As", id="me-log-save-as-btn", color="primary", size="sm"),
                        html.Span(id="me-log-save-status",
                                  style={"marginLeft": "0.75rem", "fontSize": "0.8rem", "color": "grey"}),
                    ],
                    style={"display": "flex", "alignItems": "center", "marginBottom": "0.5rem"},
                ),
                html.Hr(style={"margin": "0.5rem 0"}),
                html.Pre(id="me-log-output"),
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Save Log As")),
                        dbc.ModalBody(
                            dbc.Input(id="me-log-save-path-input", type="text",
                                      placeholder="Full path including extension (e.g. /path/log.txt)"),
                        ),
                        dbc.ModalFooter([
                            html.Span(id="me-log-saveas-status",
                                      style={"marginRight": "auto", "fontSize": "0.8rem", "color": "grey"}),
                            dbc.Button("Save", id="me-log-saveas-confirm-btn", color="primary"),
                        ]),
                    ],
                    id="me-log-save-as-modal",
                    is_open=False,
                    centered=True,
                ),
            ],
            id="me-log-panel",
            title="Log Output",
            placement="end",
            scrollable=True,
            style={"width": "500px"},
            is_open=False,
        ),
    ],
    style={"margin": "0", "padding": "0"},
)


# ── Callback registration ──────────────────────────────────────────────────────

register_motl_editor_sidebar_callbacks(MAX_MOTLS)

# Register viewer, table, plot, and save callbacks for each slot.
for _i in range(MAX_MOTLS):
    register_viewer_callbacks(f"me-{_i}-tv", tabs_id=None)
    register_table_callbacks(
        f"me-{_i}-tabv",
        csv_only=False,
        connected_motl_prefix=f"me-{_i}",
        slot_idx=_i,
        max_motls=MAX_MOTLS,
    )
    register_table_plot_callbacks(f"me-{_i}-tabv-table-plot", f"me-{_i}-tabv-global-data-store", table_grid_id=f"me-{_i}-tabv-grid")
    register_motl_simple_save_callbacks(f"me-{_i}-save", f"me-{_i}-tabv-global-data-store", f"me-{_i}")

register_viewer_callbacks("me-res-tv", tabs_id=None)
register_table_callbacks("me-res-tabv", csv_only=False, connected_motl_prefix="me-res")
register_table_plot_callbacks("me-res-tabv-table-plot", "me-res-tabv-global-data-store")


# ── Per-slot connecting callbacks ──────────────────────────────────────────────

def _register_slot_connectors(slot_idx):
    """Wire the raw motl store → viewer data + table global store for one slot."""

    @callback(
        Output(f"me-{slot_idx}-tv-data", "data", allow_duplicate=True),
        Output(f"me-{slot_idx}-tabv-global-data-store", "data", allow_duplicate=True),
        Input(f"me-{slot_idx}-motl-data-store", "data"),
        prevent_initial_call=True,
    )
    def _connect_motl(data, _s=slot_idx):
        return data, data

    @callback(
        Output(f"me-{slot_idx}-tv-data", "data", allow_duplicate=True),
        Input(f"me-{slot_idx}-tabv-global-data-store", "data"),
        Input(f"me-{slot_idx}-tabv-grid", "rowData"),
        prevent_initial_call=True,
    )
    def _connect_table_to_viewer(global_data, table_rows, _s=slot_idx):
        if ctx.triggered_id == f"me-{_s}-tabv-grid":
            return table_rows
        return global_data


for _i in range(MAX_MOTLS):
    _register_slot_connectors(_i)


# ── Per-slot relion params inline display ──────────────────────────────────────

def _scalar(v):
    if v is None:
        return None
    try:
        if hasattr(v, "__len__"):
            v = v[0] if len(v) > 0 else None
        return float(v) if v is not None else None
    except (TypeError, ValueError, IndexError):
        return None


def _format_relion_params(params):
    if not params:
        return ""
    parts = [f"Relion {params.get('version', '')}"]
    ps = _scalar(params.get("pixel_size"))
    bn = _scalar(params.get("binning"))
    if ps is not None:
        parts.append(f"pixel size: {ps:.4g} Å")
    if bn is not None:
        parts.append(f"binning: {bn:.4g}")
    if params.get("tomo_format"):
        parts.append(f"tomo format: {params['tomo_format']}")
    if params.get("subtomo_format"):
        parts.append(f"subtomo format: {params['subtomo_format']}")
    return "  |  ".join(parts)


def _register_relion_params_connector(slot_idx):
    @callback(
        Output(f"me-{slot_idx}-relion-params-inline", "children"),
        Input(f"me-{slot_idx}-relion-params-store", "data"),
        prevent_initial_call=True,
    )
    def _update_inline(params, _s=slot_idx):
        return _format_relion_params(params)


for _i in range(MAX_MOTLS):
    _register_relion_params_connector(_i)


# ── Tab management: enable tab + update label when slot receives data ──────────

@callback(
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


# Enable the Results tab and route data to its viewer and table.
@callback(
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


@callback(
    Output("me-log-output", "children"),
    Output("me-log-index", "data"),
    Output("me-log-panel", "is_open"),
    Input("me-open-log-btn", "n_clicks"),
    State("me-log-index", "data"),
    State("me-log-panel", "is_open"),
    prevent_initial_call=True,
)
def update_log(open_clicks, last_index, is_open):
    return dash_logger.get_all_logs(), len(dash_logger.buffer), True


@callback(
    Output("me-log-save-as-modal", "is_open"),
    Input("me-log-save-as-btn", "n_clicks"),
    Input("me-log-save-btn", "n_clicks"),
    State("me-log-save-path-store", "data"),
    prevent_initial_call=True,
)
def open_log_save_as(_, _2, saved_path):
    if ctx.triggered_id == "me-log-save-as-btn":
        return True
    if ctx.triggered_id == "me-log-save-btn" and not saved_path:
        return True
    return no_update


@callback(
    Output("me-log-save-as-modal", "is_open", allow_duplicate=True),
    Output("me-log-saveas-status", "children"),
    Output("me-log-save-path-store", "data"),
    Output("me-log-save-status", "children"),
    Input("me-log-saveas-confirm-btn", "n_clicks"),
    State("me-log-save-path-input", "value"),
    prevent_initial_call=True,
)
def confirm_log_save_as(_, path):
    if not path:
        return no_update, "Specify a filename.", no_update, no_update
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(dash_logger.get_all_logs())
        return False, "Saved.", path, f"Saved to {path}"
    except Exception as e:
        return no_update, str(e), no_update, no_update


@callback(
    Output("me-log-save-status", "children", allow_duplicate=True),
    Input("me-log-save-btn", "n_clicks"),
    State("me-log-save-path-store", "data"),
    prevent_initial_call=True,
)
def save_log(_, path):
    if not path:
        return no_update
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(dash_logger.get_all_logs())
        return f"Saved to {path}"
    except Exception as e:
        return str(e)
