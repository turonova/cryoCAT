"""Motl editor page — pool-backed, view-slot model.

The suite-global motl **pool** (``pool-*`` stores in
:mod:`cryocat.app.suite.app`) is the source of truth for motl data. The editor
renders into ``N_SLOTS`` fixed *view slots* whose table/viewer/save callbacks
are registered once up front with literal ``me-{i}`` prefixes. ``me-slot-map``
maps each slot to a pool ``motl_id``.

**pool -> slots** — fires when ``me-slot-map`` changes; pushes row data into
slot stores and writes a ``{"motl_id": …, "rev": …}`` reference into each
slot's ``tabv-global-data-store`` so pool-aware table components read rows
directly from the pool server side.
"""

import dash
from dash import html, dcc, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc

from cryocat.app import ids
from cryocat.app.pool import PoolState, insert_motl, get_rows, get_extra, PoolPayloadMissing
from cryocat.app.suite.motlsidebar import (
    get_motl_editor_sidebar,
    register_motl_editor_sidebar_callbacks,
    N_SLOTS,
)
from cryocat.app.components.tomoview import get_viewer_component, register_viewer_callbacks
from cryocat.app.components.tableview import get_table_component, register_table_callbacks
from cryocat.app.components.tablesave import register_table_save_callbacks
from cryocat.app.components.tableplot import register_table_plot_callbacks
from cryocat.app.components.tablecluster import register_table_cluster_callbacks
from cryocat.app.apputils import _format_relion_params


# ── Stores ──────────────────────────────────────────────────────────────────────
# Pool stores (pool-*) live in suite/app.py. Declared here: the editor's load
# staging stores, the Results-tab stores, the per-view-slot stores, and the
# slot<->motl_id map.

def _make_stores():
    stores = [
        dcc.Store(id="me-slot-map", data=[None] * N_SLOTS),
        dcc.Store(id="me-group-expand", data={}),  # { group_id: bool } — collapsed by default
        dcc.Store(id="me-active-target", data=None),  # { type: "motl"|"group", id: str } | None
        dcc.Store(id="me-results-store"),
        dcc.Store(id="me-results-label-store"),
        dcc.Store(id="me-col-merge-draft", data={}),
        dcc.Store(id="me-col-merge-config", data={}),
        dcc.Store(id="me-col-merge-motls", data=[]),
        dcc.Store(id="me-load-motl-data-store"),
        dcc.Store(id="me-load-motl-extra-data-store"),
        dcc.Store(id="me-load-motl-data-type"),
        dcc.Store(id="me-load-relion-optics-store"),
        dcc.Store(id="me-load-relion-params-store"),
        dcc.Store(id="me-res-tv-data"),
        dcc.Store(id="me-res-tv-index", data=0),
        dcc.Store(id="me-res-tabv-global-data-store"),
        dcc.Store(id="me-res-motl-data-type"),
        dcc.Store(id="me-res-motl-extra-data-store"),
        dcc.Store(id="me-res-relion-optics-store"),
        dcc.Store(id="me-res-rln-tomos-store"),
        dcc.Store(id="me-res-rln-tomos-filename"),
    ]
    for i in range(N_SLOTS):
        stores += [
            dcc.Store(id=f"me-{i}-motl-data-store"),
            dcc.Store(id=f"me-{i}-motl-extra-data-store"),
            dcc.Store(id=f"me-{i}-motl-data-type"),
            dcc.Store(id=f"me-{i}-relion-optics-store"),
            dcc.Store(id=f"me-{i}-rln-tomos-store"),
            dcc.Store(id=f"me-{i}-rln-tomos-filename"),
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
                    save_dialog_prefix=f"me-{i}-save",
                ),
                html.Hr(style={"margin": "0.5rem 0"}),
                get_viewer_component(f"me-{i}-tv"),
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
        register_table_callbacks(app, f"me-{_i}-tabv")
        register_table_save_callbacks(
            app, f"me-{_i}-tabv",
            connected_motl_prefix=f"me-{_i}",
            save_dialog_prefix=f"me-{_i}-save",
        )
        register_table_plot_callbacks(
            app,
            f"me-{_i}-tabv-table-plot",
            f"me-{_i}-tabv-global-data-store",
            table_grid_id=f"me-{_i}-tabv-grid",
            pool_aware=True,
        )
        register_table_cluster_callbacks(
            app,
            f"me-{_i}-tabv-table-cluster",
            f"me-{_i}-tabv-global-data-store",
            table_grid_id=f"me-{_i}-tabv-grid",
            pool_aware=True,
        )

    # Results tab
    register_viewer_callbacks(app, "me-res-tv", tabs_id=None)
    register_table_callbacks(app, "me-res-tabv")
    register_table_save_callbacks(app, "me-res-tabv", connected_motl_prefix="me-res")
    register_table_plot_callbacks(app, "me-res-tabv-table-plot", "me-res-tabv-global-data-store", pool_aware=True)
    register_table_cluster_callbacks(app, "me-res-tabv-table-cluster", "me-res-tabv-global-data-store", pool_aware=True)

    _register_pool_sync(app)
    _register_create_from_selected(app)
    _register_slot_connectors_all(app)
    _register_relion_params_connectors_all(app)
    _register_save_connectors(app)

    # Tab labels / enabled state, driven by the slot map + pool registry.
    @app.callback(
        *[Output(f"me-tab-slot-{i}", "label") for i in range(N_SLOTS)],
        *[Output(f"me-tab-slot-{i}", "disabled") for i in range(N_SLOTS)],
        Input("me-slot-map", "data"),
        Input(ids.POOL_REGISTRY, "data"),
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
    """pool -> slots: fires when the slot map changes (load / reassignment)."""

    @app.callback(
        *[Output(f"me-{i}-motl-data-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        *[Output(f"me-{i}-motl-extra-data-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        *[Output(f"me-{i}-motl-data-type", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        *[Output(f"me-{i}-relion-optics-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        *[Output(f"me-{i}-rln-tomos-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        *[Output(f"me-{i}-rln-tomos-filename", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        *[Output(f"me-{i}-relion-params-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        *[Output(f"me-{i}-undo-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        *[Output(f"me-{i}-tabv-global-data-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        Input("me-slot-map", "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_REGISTRY, "data"),
        prevent_initial_call=True,
    )
    def sync_pool_to_slots(slot_map, pool_meta, registry):
        slot_map = slot_map or [None] * N_SLOTS
        pool_meta = pool_meta or {}
        registry = registry or {}

        data, extra, dtype, optics, r5t, r5tn, rparams, undo, table_refs = ([] for _ in range(9))
        for i in range(N_SLOTS):
            mid = slot_map[i] if i < len(slot_map) else None
            has_payload = False
            if mid:
                try:
                    get_rows(mid)  # existence check only; rows stay server-side
                    has_payload = True
                except PoolPayloadMissing:
                    pass
            if has_payload:
                meta = pool_meta.get(mid) or {}
                extra_df = get_extra(mid)
                rev = registry.get(mid, {}).get("revision", 0)
                data.append(None)  # motl-data-store: no rows serialized; use pool ref via tabv-global-data-store
                extra.append(mid if extra_df is not None else None)  # mid string — callers use get_extra(mid)
                dtype.append(meta.get("data_type"))
                optics.append(meta.get("relion_optics"))
                r5t.append(meta.get("relion5_tomos"))
                r5tn.append(meta.get("relion5_tomos_filename"))
                rparams.append(meta.get("relion_params"))
                undo.append(None)
                table_refs.append({"motl_id": mid, "rev": rev})
            else:
                for lst in (data, extra, dtype, optics, r5t, r5tn, rparams, undo, table_refs):
                    lst.append(None)

        return (*data, *extra, *dtype, *optics, *r5t, *r5tn, *rparams, *undo, *table_refs)

    @app.callback(
        *[Output(f"me-{i}-tabv-global-data-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        Input(ids.POOL_REGISTRY, "data"),
        State("me-slot-map", "data"),
        *[State(f"me-{i}-tabv-global-data-store", "data") for i in range(N_SLOTS)],
        prevent_initial_call=True,
    )
    def _sync_revisions(registry, slot_map, *current_refs):
        registry = registry or {}
        slot_map = slot_map or [None] * N_SLOTS
        outs = []
        changed = False
        for i in range(N_SLOTS):
            mid = slot_map[i] if i < len(slot_map) else None
            ref = current_refs[i]
            if mid and mid in registry:
                new_rev = registry[mid].get("revision", 0)
                if isinstance(ref, dict) and ref.get("rev") == new_rev:
                    outs.append(no_update)
                else:
                    outs.append({"motl_id": mid, "rev": new_rev})
                    changed = True
            else:
                outs.append(no_update)
        if not changed:
            raise dash.exceptions.PreventUpdate
        return tuple(outs)


# ── "Create new from selected" → new pool motl ───────────────────────────────────

def _register_create_from_selected(app):
    """Each slot table's "Create new from selected" button spawns a new pool motl
    from that grid's selected rows, and surfaces it in the first free slot."""

    @app.callback(
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output("me-slot-map", "data", allow_duplicate=True),
        Output("me-tabs", "active_tab", allow_duplicate=True),
        *[Input(f"me-{i}-tabv-create-from-selected-btn", "n_clicks") for i in range(N_SLOTS)],
        *[State(f"me-{i}-tabv-grid", "selectedRows") for i in range(N_SLOTS)],
        State("me-slot-map", "data"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        prevent_initial_call=True,
    )
    def create_from_selected(*args):
        n_clicks = args[:N_SLOTS]
        selected_all = args[N_SLOTS : 2 * N_SLOTS]
        slot_map, registry, pool_meta, next_id = args[2 * N_SLOTS :]

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

        state = PoolState.from_stores(registry, pool_meta, next_id)
        slot_map = list(slot_map or [None] * N_SLOTS)
        while len(slot_map) < N_SLOTS:
            slot_map.append(None)

        src_mid = slot_map[src] if src < len(slot_map) else None
        src_meta = state.registry.get(src_mid, {}) if src_mid else {}
        src_label = src_meta.get("label", f"Slot {src + 1}")
        src_type = src_meta.get("type", "emmotl")
        short = src_label[:15] + "…" if len(src_label) > 15 else src_label

        # TODO(P9): route through run_operation_to_pool once selection is tracked.
        state, mid = insert_motl(
            state, selected_rows,
            label=f"Sel from {short} ({len(selected_rows)})",
            motl_type=src_type,
            meta={
                "data_type": src_type,
                "relion_optics": None,
                "relion5_tomos": None,
                "relion5_tomos_filename": None,
                "relion_params": None,
            },
        )

        free = next((i for i in range(N_SLOTS) if not slot_map[i]), None)
        active_tab = no_update
        if free is not None:
            slot_map[free] = mid
            active_tab = f"me-tab-{free}"

        return (*state.to_stores(), slot_map, active_tab)


# ── Save dialog connectors — keep save dialog motl-id and prefill in sync ────────

def _register_save_connectors(app):
    """Update each slot's save dialog stores when the slot map or pool meta changes."""
    @app.callback(
        *[Output(f"me-{i}-save-motl-id", "data") for i in range(N_SLOTS)],
        *[Output(f"me-{i}-save-prefill", "data") for i in range(N_SLOTS)],
        Input("me-slot-map", "data"),
        State(ids.POOL_META, "data"),
        prevent_initial_call=True,
    )
    def _sync_save_stores(slot_map, pool_meta):
        slot_map = slot_map or [None] * N_SLOTS
        pool_meta = pool_meta or {}
        motl_ids, prefills = [], []
        for i in range(N_SLOTS):
            mid = slot_map[i] if i < len(slot_map) else None
            motl_ids.append(mid)
            prefills.append(pool_meta.get(mid) if mid else None)
        return (*motl_ids, *prefills)


# ── Per-slot connecting callbacks ────────────────────────────────────────────────

def _register_slot_connectors_all(app):
    for slot_idx in range(N_SLOTS):
        _register_slot_connectors(app, slot_idx)


def _register_slot_connectors(app, slot_idx):
    """Wire the slot table store -> viewer data."""

    @app.callback(
        Output(f"me-{slot_idx}-tv-data", "data", allow_duplicate=True),
        Input(f"me-{slot_idx}-tabv-global-data-store", "data"),
        prevent_initial_call=True,
    )
    def _connect_table_to_viewer(global_data, _s=slot_idx):
        # Only fires on pool-reference changes (not on grid rowData changes).
        # This collapses the 2× viewer-callback firing that occurred because
        # load_data_to_grid also wrote rowData which re-triggered the chain. (W3)
        if not global_data or not isinstance(global_data, dict) or "motl_id" not in global_data:
            raise dash.exceptions.PreventUpdate
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
