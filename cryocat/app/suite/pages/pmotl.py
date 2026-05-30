"""Motl editor page — pool-backed, view-slot model.

The suite-global motl **pool** (``pool-*`` stores in
:mod:`cryocat.app.suite.app`) is the source of truth for motl data. The editor
renders into ``N_SLOTS`` fixed *view slots* whose table/viewer/save callbacks
are registered once up front with literal ``me-{i}`` prefixes. ``me-slot-map``
maps each slot to a pool ``motl_id``; two sync callbacks keep slot stores and
the pool in agreement:

* **pool -> slots** — fires when ``me-slot-map`` changes (a slot is (re)assigned
  or a load happens); pushes ``pool-motls[mid]`` (+ extra/meta) into the slot
  stores.
* **slots -> pool** — fires when any slot's table data changes (edits,
  operations); writes the slot data back into ``pool-motls`` so it persists and
  is visible to other tools.
"""

import dash
from dash import html, dcc, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc

from cryocat.app.suite.motlsidebar import (
    get_motl_editor_sidebar,
    register_motl_editor_sidebar_callbacks,
    N_SLOTS,
)
from cryocat.app.components.tomoview import get_viewer_component, register_viewer_callbacks
from cryocat.app.components.tableview import get_table_component, register_table_callbacks
from cryocat.app.components.tableplot import register_table_plot_callbacks
from cryocat.app.components.tablecluster import register_table_cluster_callbacks
from cryocat.app.components.motlio import get_motl_simple_save_component, register_motl_simple_save_callbacks
from cryocat.app.apputils import _format_relion_params


# ── Stores ──────────────────────────────────────────────────────────────────────
# Pool stores (pool-*) live in suite/app.py. Declared here: the editor's load
# staging stores, the Results-tab stores, the per-view-slot stores, and the
# slot<->motl_id map.

def _make_stores():
    stores = [
        dcc.Store(id="me-slot-map", data=[None] * N_SLOTS, storage_type="session"),
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
    for i in range(N_SLOTS):
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


# ── Per-slot tab content ────────────────────────────────────────────────────────

def _slot_tab_content(i):
    return dbc.Tab(
        id=f"me-tab-slot-{i}",
        tab_id=f"me-tab-{i}",
        label=f"Slot {i + 1}",
        disabled=True,
        children=html.Div(
            [
                get_table_component(
                    f"me-{i}-tabv",
                    connected_motl_prefix=f"me-{i}",
                    show_create_from_selected=True,
                ),
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
    tabs = [_slot_tab_content(i) for i in range(N_SLOTS)]
    tabs.append(_results_tab_content())
    return dbc.Col(
        dbc.Tabs(
            id="me-tabs",
            active_tab="me-tab-0",
            children=tabs,
            style={"padding": "0.5rem"},
        ),
        width=9,
        style={"margin": "0", "padding": "0"},
    )


# ── Page layout ─────────────────────────────────────────────────────────────────

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
    ],
    style={"margin": "0", "padding": "0"},
)


# ── Callback registration ───────────────────────────────────────────────────────

def register_callbacks(app):

    register_motl_editor_sidebar_callbacks(app)

    # Per-slot viewer / table / plot / save callbacks — registered once up front
    # with literal prefixes (shared components are untouched).
    for _i in range(N_SLOTS):
        register_viewer_callbacks(app, f"me-{_i}-tv", tabs_id=None)
        register_table_callbacks(
            app,
            f"me-{_i}-tabv",
            csv_only=False,
            connected_motl_prefix=f"me-{_i}",
        )
        register_table_plot_callbacks(
            app,
            f"me-{_i}-tabv-table-plot",
            f"me-{_i}-tabv-global-data-store",
            table_grid_id=f"me-{_i}-tabv-grid",
        )
        register_table_cluster_callbacks(
            app,
            f"me-{_i}-tabv-table-cluster",
            f"me-{_i}-tabv-global-data-store",
            table_grid_id=f"me-{_i}-tabv-grid",
        )
        register_motl_simple_save_callbacks(app, f"me-{_i}-save", f"me-{_i}-tabv-global-data-store", f"me-{_i}")

    # Results tab
    register_viewer_callbacks(app, "me-res-tv", tabs_id=None)
    register_table_callbacks(app, "me-res-tabv", csv_only=False, connected_motl_prefix="me-res")
    register_table_plot_callbacks(app, "me-res-tabv-table-plot", "me-res-tabv-global-data-store")
    register_table_cluster_callbacks(app, "me-res-tabv-table-cluster", "me-res-tabv-global-data-store")

    _register_pool_sync(app)
    _register_create_from_selected(app)
    _register_slot_connectors_all(app)
    _register_relion_params_connectors_all(app)

    # Tab labels / enabled state, driven by the slot map + pool registry.
    @app.callback(
        *[Output(f"me-tab-slot-{i}", "label") for i in range(N_SLOTS)],
        *[Output(f"me-tab-slot-{i}", "disabled") for i in range(N_SLOTS)],
        Input("me-slot-map", "data"),
        Input("pool-registry", "data"),
        prevent_initial_call=True,
    )
    def update_tab_labels(slot_map, registry):
        slot_map = slot_map or [None] * N_SLOTS
        registry = registry or {}
        labels, disabled = [], []
        for i in range(N_SLOTS):
            mid = slot_map[i] if i < len(slot_map) else None
            if mid and mid in registry:
                raw = registry[mid].get("label", f"Motl {i + 1}")
                labels.append(raw[:22] + "…" if len(raw) > 22 else raw)
                disabled.append(False)
            else:
                labels.append(f"Slot {i + 1}")
                disabled.append(True)
        return (*labels, *disabled)

    # Results tab connection.
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



# ── Pool <-> slot synchronisation ────────────────────────────────────────────────

def _register_pool_sync(app):
    """Two callbacks keep the view-slot stores and the pool in agreement."""

    # pool -> slots: fires when the slot map changes (load / reassignment).
    @app.callback(
        *[Output(f"me-{i}-motl-data-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        *[Output(f"me-{i}-motl-extra-data-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        *[Output(f"me-{i}-motl-data-type", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        *[Output(f"me-{i}-relion-optics-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        *[Output(f"me-{i}-relion5-tomos-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        *[Output(f"me-{i}-relion5-tomos-filename", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        *[Output(f"me-{i}-relion-params-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        *[Output(f"me-{i}-undo-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        Input("me-slot-map", "data"),
        State("pool-motls", "data"),
        State("pool-extra", "data"),
        State("pool-meta", "data"),
        prevent_initial_call=True,
    )
    def sync_pool_to_slots(slot_map, pool_motls, pool_extra, pool_meta):
        slot_map = slot_map or [None] * N_SLOTS
        pool_motls = pool_motls or {}
        pool_extra = pool_extra or {}
        pool_meta = pool_meta or {}

        data, extra, dtype, optics, r5t, r5tn, rparams, undo = ([] for _ in range(8))
        for i in range(N_SLOTS):
            mid = slot_map[i] if i < len(slot_map) else None
            if mid and mid in pool_motls:
                meta = pool_meta.get(mid) or {}
                data.append(pool_motls.get(mid))
                extra.append(pool_extra.get(mid))
                dtype.append(meta.get("data_type"))
                optics.append(meta.get("relion_optics"))
                r5t.append(meta.get("relion5_tomos"))
                r5tn.append(meta.get("relion5_tomos_filename"))
                rparams.append(meta.get("relion_params"))
                undo.append(None)
            else:
                for lst in (data, extra, dtype, optics, r5t, r5tn, rparams, undo):
                    lst.append(None)

        return (*data, *extra, *dtype, *optics, *r5t, *r5tn, *rparams, *undo)

    # slots -> pool: fires when any slot's table data changes (edits/operations).
    @app.callback(
        Output("pool-motls", "data", allow_duplicate=True),
        Output("pool-registry", "data", allow_duplicate=True),
        *[Input(f"me-{i}-tabv-global-data-store", "data") for i in range(N_SLOTS)],
        State("me-slot-map", "data"),
        State("pool-motls", "data"),
        State("pool-registry", "data"),
        prevent_initial_call=True,
    )
    def sync_slots_to_pool(*args):
        slot_globals = args[:N_SLOTS]
        slot_map, pool_motls, registry = args[N_SLOTS:]
        slot_map = slot_map or [None] * N_SLOTS
        pool_motls = dict(pool_motls or {})
        registry = dict(registry or {})

        changed = False
        for i in range(N_SLOTS):
            mid = slot_map[i] if i < len(slot_map) else None
            if not mid:
                continue
            rows = slot_globals[i]
            if rows is None:
                continue
            if pool_motls.get(mid) != rows:
                pool_motls[mid] = rows
                if mid in registry:
                    registry[mid] = {**registry[mid], "n_rows": len(rows)}
                changed = True

        if not changed:
            raise dash.exceptions.PreventUpdate
        return pool_motls, registry


# ── "Create new from selected" → new pool motl ───────────────────────────────────

def _register_create_from_selected(app):
    """Each slot table's "Create new from selected" button spawns a new pool motl
    from that grid's selected rows, and surfaces it in the first free slot."""

    @app.callback(
        Output("pool-registry", "data", allow_duplicate=True),
        Output("pool-motls", "data", allow_duplicate=True),
        Output("pool-extra", "data", allow_duplicate=True),
        Output("pool-meta", "data", allow_duplicate=True),
        Output("pool-next-id", "data", allow_duplicate=True),
        Output("me-slot-map", "data", allow_duplicate=True),
        Output("me-tabs", "active_tab", allow_duplicate=True),
        *[Input(f"me-{i}-tabv-create-from-selected-btn", "n_clicks") for i in range(N_SLOTS)],
        *[State(f"me-{i}-tabv-grid", "selectedRows") for i in range(N_SLOTS)],
        State("me-slot-map", "data"),
        State("pool-registry", "data"),
        State("pool-motls", "data"),
        State("pool-extra", "data"),
        State("pool-meta", "data"),
        State("pool-next-id", "data"),
        prevent_initial_call=True,
    )
    def create_from_selected(*args):
        n_clicks = args[:N_SLOTS]
        selected_all = args[N_SLOTS : 2 * N_SLOTS]
        slot_map, registry, pool_motls, pool_extra, pool_meta, next_id = args[2 * N_SLOTS :]

        if not any(n_clicks):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        if not (isinstance(triggered, str) and triggered.endswith("-tabv-create-from-selected-btn")):
            raise dash.exceptions.PreventUpdate
        try:
            src = int(triggered[len("me-") : triggered.index("-tabv-")])
        except (ValueError, IndexError):
            raise dash.exceptions.PreventUpdate

        selected_rows = selected_all[src]
        if not selected_rows:
            raise dash.exceptions.PreventUpdate

        registry = dict(registry or {})
        pool_motls = dict(pool_motls or {})
        pool_extra = dict(pool_extra or {})
        pool_meta = dict(pool_meta or {})
        next_id = next_id or 0
        slot_map = list(slot_map or [None] * N_SLOTS)
        while len(slot_map) < N_SLOTS:
            slot_map.append(None)

        src_mid = slot_map[src] if src < len(slot_map) else None
        src_meta = registry.get(src_mid, {}) if src_mid else {}
        src_label = src_meta.get("label", f"Slot {src + 1}")
        src_type = src_meta.get("type", "emmotl")
        short = src_label[:15] + "…" if len(src_label) > 15 else src_label

        mid = f"motl-{next_id}"
        registry[mid] = {
            "label": f"Sel from {short} ({len(selected_rows)})",
            "type": src_type,
            "n_rows": len(selected_rows),
            "active": True,
        }
        pool_motls[mid] = selected_rows
        pool_extra[mid] = None
        pool_meta[mid] = {
            "data_type": src_type,
            "relion_optics": None,
            "relion5_tomos": None,
            "relion5_tomos_filename": None,
            "relion_params": None,
        }

        free = next((i for i in range(N_SLOTS) if not slot_map[i]), None)
        active_tab = no_update
        if free is not None:
            slot_map[free] = mid
            active_tab = f"me-tab-{free}"

        return registry, pool_motls, pool_extra, pool_meta, next_id + 1, slot_map, active_tab


# ── Per-slot connecting callbacks ────────────────────────────────────────────────

def _register_slot_connectors_all(app):
    for slot_idx in range(N_SLOTS):
        _register_slot_connectors(app, slot_idx)


def _register_slot_connectors(app, slot_idx):
    """Wire the raw slot motl store -> viewer data + table global store."""

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


# ── Per-slot relion params inline display ────────────────────────────────────────

def _register_relion_params_connectors_all(app):
    for slot_idx in range(N_SLOTS):
        _register_relion_params_connector(app, slot_idx)


def _register_relion_params_connector(app, slot_idx):
    @app.callback(
        Output(f"me-{slot_idx}-relion-params-inline", "children"),
        Input(f"me-{slot_idx}-relion-params-store", "data"),
        prevent_initial_call=True,
    )
    def _update_inline(params, _s=slot_idx):
        return _format_relion_params(params)
