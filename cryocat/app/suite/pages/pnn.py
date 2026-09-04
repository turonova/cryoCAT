"""Nearest-neighbor analysis tool.

Computation is decoupled from clustering:

* **Compute NN analysis** -- builds the per-pair table from the suite-pool
  motl(s) using ``NearestNeighbors`` with the auto-generated parameter form,
  optionally adds angular-distance columns, and renders the xyz panels.
* **Post-processing** -- enriches the existing NN table without recomputing:
  pulls extra columns from source motls via ``add_motl_columns``, or adds
  angular / Euclidean distances after the fact.
* **Clustering** -- K-means or proximity clustering on the NN result columns.
  Select the clustering type in the sidebar accordion; parameters and results
  appear in the main area below the scatter panels.

Contract: exposes ``layout`` and ``register_callbacks(app)``.
"""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import dash
from dash import html, dcc, ctx, Input, Output, State, ALL, no_update
import dash_bootstrap_components as dbc

from cryocat.core.cryomotl import Motl
from cryocat.analysis.nnana import NearestNeighbors
from cryocat.analysis import visplot
from cryocat.app.apputils import generate_kwargs, run_operation
from cryocat.app import ids, formgen, styles
from cryocat.app.formgen import make_dropdown
from cryocat.app.components.graphsettings import style_figure
from cryocat.app.pool import (
    get_rows as _get_rows,
    get_motl as _get_motl, PoolPayloadMissing as _PoolPayloadMissing,
    PoolState,
)
from cryocat.app.components.poolpicker import get_pool_picker, register_pool_picker_callbacks
from cryocat.app.components.poolslotlist import (
    get_pool_slot_list,
    register_pool_slot_list_callbacks,
    register_slot_focus_callback,
    _first_free_slot,
)
from cryocat.app.components.customel import customel_graph
from cryocat.app.components.tableview import get_table_component, register_table_callbacks
from cryocat.app.components.tableplot import register_table_plot_callbacks
from cryocat.app.components.tablecluster import register_table_cluster_callbacks
from cryocat.app.pageshell import page_shell, sidebar_accordion
from cryocat.app.components.tablesource import get_table_source, register_table_source_callbacks
from cryocat.app.components.tabletomotl import get_table_to_motl, register_table_to_motl_callbacks

DYNAMIC_IDS: list[tuple[str, str]] = [
    ("nn-out-tabv-grid-container", "nn-out-tabv-grid"),
]


_MOTL_COL_OPTIONS = [{"label": c, "value": c} for c in Motl.motl_columns]

_NN_SLOTS = 5
_nn_table_refs: dict[str, dict] = {}   # data_id → full table-pool ref (for slot display)
_nn_xyz_figs: dict[str, Any] = {}      # data_id → styled figure dict (for slot display)

from cryocat.app import datapool as _datapool


def _nn_row_extra(data_id: str, entry: dict) -> list:
    return [
        dbc.Button(
            "✕",
            id={"type": "dp-remove-btn", "data_id": data_id},
            size="sm",
            color=styles.BTN_NEUTRAL,
            n_clicks=0,
            style={"flexShrink": 0, "padding": "0 4px"},
        )
    ]


def _nn_csv_save(path, grid_data, used_motls):
    """CSV saver for the NN table: puts motl_id first and column_name second."""
    df = pd.DataFrame(grid_data)
    column_name = (used_motls or {}).get("column_name")
    if column_name and column_name in df.columns:
        lead = [c for c in ("motl_id", column_name) if c in df.columns]
        cols = lead + [c for c in df.columns if c not in lead]
        df = df[cols]
    run_operation(df.to_csv, {"path_or_buf": path, "index": False})
    return False, f"Saved to {path}"


# ── Layout ──────────────────────────────────────────────────────────────────────

def _postprocess_sidebar_content():
    """Sidebar content for the 'Post-processing' accordion item."""
    hint = {"color": "var(--color9)", "marginBottom": "0.4rem"}
    lbl = {"fontWeight": "bold", "marginBottom": "0.2rem"}
    return html.Div(
        [
            html.Label("Add columns from source motls:", style=lbl),
            html.P(
                "Enrich the NN table with extra columns pulled from the original motls. "
                "Columns are added as qp_<col> and/or nn_<col>.",
                style=hint,
            ),
            make_dropdown("nn-pp-add-cols", _MOTL_COL_OPTIONS, None, multi=True,
                          placeholder="Select motl columns to add…",
                          style={"marginBottom": "0.3rem"}),
            html.Label("Add for sides:", style={"marginBottom": "0.2rem"}),
            dbc.Checklist(
                id="nn-pp-sides",
                options=[
                    {"label": "Query particle (qp)", "value": "qp"},
                    {"label": "Neighbor (nn)", "value": "nn"},
                ],
                value=["qp", "nn"],
                inline=True,
                labelStyle={},
                style={"marginBottom": "0.5rem"},
            ),
            html.Hr(style={"margin": "0.4rem 0"}),
            dbc.Checklist(
                id="nn-pp-angular-toggle",
                options=[{"label": "Add angular distances", "value": "on"}],
                value=[],
                inline=True,
                style={"marginBottom": "0.3rem"},
            ),
            html.Div(
                formgen.build_form(
                    NearestNeighbors.get_angular_distances,
                    id_type="nn-forms-params",
                    id_extra={"cls_name": "nn-pp-angular"},
                ),
                id="nn-pp-angular-form-wrap",
                style={"display": "none"},
            ),
            html.Hr(style={"margin": "0.4rem 0"}),
            dbc.Checklist(
                id="nn-pp-dist-toggle",
                options=[{"label": "Add NN distances", "value": "on"}],
                value=[],
                inline=True,
                style={"marginBottom": "0.5rem"},
            ),
            dbc.Button(
                "Apply",
                id="nn-pp-apply-btn",
                color="primary",
                size="sm",
                style={"width": "100%"},
            ),
            html.Div(
                id="nn-pp-status",
                style={
                    "color": "var(--color9)",
                    "marginTop": "0.4rem",
                    "wordBreak": "break-word",
                },
            ),
        ]
    )


def _nn_load_fn(path, motl_selection=None):
    """Load an NN table from a CSV file; returns dict with df and column_name."""
    nn = NearestNeighbors.load(file_path=path)
    from cryocat.app.suite.pages._motl_link import ordered_selection_to_motl_links, check_motl_overlap
    motl_links = ordered_selection_to_motl_links(motl_selection)
    result = {"df": nn.df, "column_name": nn.column_name, "motl_links": motl_links}
    if motl_selection:
        query_id = motl_selection[0] if isinstance(motl_selection, list) else motl_selection
        _, _, msg = check_motl_overlap(nn.df, "qp_subtomo_id", query_id)
        result["status_extra"] = msg
    return result


def _sidebar() -> list:
    return [
        sidebar_accordion(
            [
                dbc.AccordionItem(
                    get_table_source(
                        "nn-src",
                        extra_file_children=[
                            formgen.form_row(
                                "motl_selection",
                                formgen.make_dropdown(
                                    {"type": "nn-src-ts-extra", "param": "motl_selection"},
                                    [],
                                    None,
                                    multi=True,
                                    clearable=True,
                                    placeholder="None — set after loading",
                                ),
                                "Order matters: first selection is the query motl, rest are neighbours. "
                                "Optional: set here or in Source motl link after loading.",
                                label_id="nn-ts-motl-sel-lbl",
                                label_text="Source motl(s)",
                                truly_optional=True,
                            ),
                        ],
                        compute_children=[
                            get_pool_picker("nn"),
                            # nn_type: user-friendly labels; build widget manually, wrap with form_row
                            formgen.form_row(
                                "nn_type",
                                formgen.make_dropdown(
                                    formgen._mk_id("nn-forms-params", "nn_type", "Literal", {"cls_name": "nn-params"}),
                                    [
                                        {"label": "Closest distance", "value": "closest_dist"},
                                        {"label": "Radius", "value": "radius"},
                                    ],
                                    "closest_dist",
                                    clearable=False,
                                ),
                                "How nearest-neighbor candidates are selected: by closest k distances, or all within a radius.",
                            ),
                            html.Div(
                                formgen.build_form(
                                    NearestNeighbors,
                                    id_type="nn-forms-params",
                                    id_extra={"cls_name": "nn-params"},
                                    exclude=["input_data", "nn_type", "exclude_column_name"],
                                ),
                            ),
                            # exclude_column_name: clearable motl-column dropdown (None by default)
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label(
                                                "Exclude column name (opt.)",
                                                id="nn-excl-col-lbl",
                                                style={"margin": 0},
                                            ),
                                            dbc.Tooltip(
                                                "When set, NN candidates sharing the query "
                                                "particle's value in this column are excluded. "
                                                "For closest-distance (k-NN) mode the row count "
                                                "is unchanged (k neighbors are still returned, "
                                                "just from different objects) and distances "
                                                "typically increase. For radius mode the row "
                                                "count can decrease because excluded candidates "
                                                "are simply dropped.",
                                                target="nn-excl-col-lbl",
                                                placement="right",
                                            ) if styles.TOOLTIPS_ENABLED else None,
                                        ],
                                        style={
                                            "width": "45%", "display": "flex",
                                            "alignItems": "center", "boxSizing": "border-box",
                                            "paddingRight": "4px",
                                        },
                                    ),
                                    html.Div(
                                        make_dropdown(
                                            {
                                                "type": "nn-forms-params",
                                                "owner": "",
                                                "param": "exclude_column_name",
                                                "tag": "Literal",
                                                "cls_name": "nn-params",
                                            },
                                            _MOTL_COL_OPTIONS, None,
                                            clearable=True,
                                            placeholder="None (optional)",
                                        ),
                                        style={"width": "55%"},
                                    ),
                                ],
                                style={**{"display": "flex", "flexDirection": "row", "width": "100%", "alignItems": "center"}, "marginBottom": "0.25rem"},
                            ),
                            html.Div(
                                dbc.Checklist(
                                    id="nn-dist-toggle",
                                    options=[{"label": "Compute euclidean distances", "value": "on"}],
                                    value=[],
                                    inline=True,
                                ),
                                id="nn-dist-toggle-wrap",
                                style={"display": "none", "marginTop": "0.3rem"},
                            ),
                            dbc.Checklist(
                                id="nn-angular-toggle",
                                options=[{"label": "Compute angular distances", "value": "on"}],
                                value=[],
                                inline=True,
                                style={"marginBottom": "0.4rem"},
                            ),
                            html.Div(
                                formgen.build_form(
                                    NearestNeighbors.get_angular_distances,
                                    id_type="nn-forms-params",
                                    id_extra={"cls_name": "nn-angular"},
                                ),
                                id="nn-angular-form-wrap",
                                style={"display": "none"},
                            ),
                            dbc.Button(
                                "Compute NN analysis",
                                id="nn-compute-btn",
                                color="primary",
                                size="sm",
                                style={"width": "100%", "marginTop": "0.5rem"},
                            ),
                            html.Div(
                                id="nn-stats-text",
                                style={
                                    "color": "var(--color9)",
                                    "marginTop": "0.5rem",
                                    "wordBreak": "break-word",
                                },
                            ),
                        ],
                        file_extensions=(".csv",),
                        label="Source",
                    ),
                    title="NN table",
                    item_id="nn-acc-params",
                ),
                dbc.AccordionItem(
                    _postprocess_sidebar_content(),
                    title="Post-processing",
                    item_id="nn-acc-postprocess",
                ),
                dbc.AccordionItem(
                    get_table_to_motl("nn-ttm"),
                    title="Create motl from NN table",
                    item_id="nn-acc-create",
                ),
                dbc.AccordionItem(
                    get_pool_slot_list("nn-pool"),
                    title="NN results in pool",
                    item_id="nn-acc-pool",
                ),
            ],
            active_item=["nn-acc-params"],
        ),
    ]


def _main() -> list:
    slot_tabs = [
        dbc.Tab(
            label=f"Slot {i + 1}",
            tab_id=f"nn-slot-{i}",
            id=f"nn-slot-tab-{i}",
            disabled=True,
        )
        for i in range(_NN_SLOTS)
    ]
    return [
        dcc.Store(id="nn-out-tabv-global-data-store"),
        dbc.Tabs(
            slot_tabs,
            id="nn-slot-tabs",
            active_tab="nn-slot-0",
            style={"marginBottom": "0.5rem"},
        ),
        get_table_component("nn-out-tabv", show_editor=True),
        html.Hr(style={"margin": "0.5rem 0"}),
        html.Div(id="nn-xyz-graph-area"),
    ]


layout = html.Div(
    [
        dcc.Store(id="nn-result"),
        # Ordered list of pool motl-ids used in the last NN run, plus is_multi flag.
        dcc.Store(id="nn-used-motls-store"),
        dcc.Store(id="nn-cluster-cols-store", data=[]),
        dcc.Store(id="nn-pool-registry", data={}),
        dcc.Store(id="nn-pool-slot-map", data=[None] * _NN_SLOTS),
        dcc.Store(id="nn-pool-active-id"),
        page_shell(_sidebar(), _main()),
    ],
    style={"margin": "0", "padding": "0"},
)


# ── Callback helpers ───────────────────────────────────────────────────────────

def _kwargs_by_cls(param_ids, param_values, target_cls, pool_state=None):
    """Demux flat ALL-state to kwargs for a single ``cls_name``."""
    ids, vals = [], []
    for pid, val in zip(param_ids, param_values):
        if pid.get("cls_name") == target_cls:
            ids.append(pid)
            vals.append(val)
    return generate_kwargs(ids, vals, pool_state) if ids else {}


# ── Callbacks ──────────────────────────────────────────────────────────────────

def register_callbacks(app):
    from dash import ALL as _ALL
    formgen.register_form_callbacks(app, "nn-forms-params", {"cls_name": _ALL})
    register_pool_picker_callbacks(app, "nn")
    register_pool_slot_list_callbacks(
        app, "nn-pool", "nn-pool-registry", "nn-pool-slot-map", _NN_SLOTS,
        row_extra_fn=_nn_row_extra,
        active_id_store_id="nn-pool-active-id",
    )
    register_slot_focus_callback(
        app, "nn-pool-slot-map", "nn-slot-tabs", "nn-slot-", _NN_SLOTS,
    )
    register_table_to_motl_callbacks(
        app, "nn-ttm",
        source_table_id="nn-out-tabv-grid",
        id_column="qp_subtomo_id",
    )
    register_table_source_callbacks(
        app, "nn-src",
        check_fn=NearestNeighbors.check_nn_columns,
        load_fn=_nn_load_fn,
    )
    register_table_callbacks(
        app, "nn-out-tabv",
        resolve_df=_datapool.resolve_df, resolve_n_rows=_datapool.resolve_n_rows,
        extra_csv_states=[State("nn-used-motls-store", "data")],
        custom_csv_save_fn=_nn_csv_save,
        show_editor=True,
    )
    register_table_plot_callbacks(
        app, "nn-out-tabv-table-plot", "nn-out-tabv-global-data-store",
        special_graphs=["Orientational distribution", "Polar NN distances"],
        table_grid_id="nn-out-tabv-grid",
        resolve_df=_datapool.resolve_df,
    )
    register_table_cluster_callbacks(
        app, "nn-out-tabv-table-cluster", "nn-out-tabv-global-data-store",
        table_grid_id="nn-out-tabv-grid",
        cluster_cols_store_id="nn-cluster-cols-store",
        resolve_df=_datapool.resolve_df,
    )

    @app.callback(
        Output("nn-dist-toggle-wrap", "style"),
        Input({"type": "nn-forms-params", "owner": "", "param": "nn_type", "tag": "Literal", "cls_name": "nn-params"}, "value"),
    )
    def _toggle_dist_form(nn_type):
        return {"display": "block"} if nn_type == "radius" else {"display": "none"}

    @app.callback(
        Output("nn-angular-form-wrap", "style"),
        Input("nn-angular-toggle", "value"),
    )
    def _toggle_angular_form(angular_on):
        return {"display": "block"} if angular_on else {"display": "none"}

    @app.callback(
        Output("nn-pp-angular-form-wrap", "style"),
        Input("nn-pp-angular-toggle", "value"),
    )
    def _toggle_pp_angular_form(angular_on):
        return {"display": "block"} if angular_on else {"display": "none"}

    # ── Compute NN: fills the table and the xyz panel. ────────────────────────
    @app.callback(
        Output("nn-xyz-graph-area", "children"),
        Output("nn-out-tabv-global-data-store", "data"),
        Output("nn-stats-text", "children"),
        Output("nn-result", "data"),
        Output("nn-used-motls-store", "data"),
        # allow_duplicate: tableeditor._on_apply also writes DATA_POOL_REGISTRY
        Output(ids.DATA_POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.DATA_POOL_NEXT_ID,  "data", allow_duplicate=True),
        Output("nn-pool-slot-map", "data", allow_duplicate=True),
        Input("nn-compute-btn", "n_clicks"),
        State("nn-value", "data"),
        State({"type": "nn-forms-params", "owner": ALL, "cls_name": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": "nn-forms-params", "owner": ALL, "cls_name": ALL, "param": ALL, "tag": ALL}, "id"),
        State("nn-angular-toggle", "value"),
        State("nn-dist-toggle", "value"),
        State(ids.DATA_POOL_REGISTRY, "data"),
        State(ids.DATA_POOL_NEXT_ID,  "data"),
        State(ids.GRAPH_SETTINGS_STORE, "data"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        State("nn-pool-slot-map", "data"),
        prevent_initial_call=True,
    )
    def compute_nn(n_clicks, selected, param_values, param_ids, angular_on, compute_dist,
                   dp_registry, dp_next_id, gs_settings, registry, pool_meta, pool_next_id,
                   slot_map):
        _no = no_update
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        if not selected:
            return (no_update, no_update, "Select at least one motl from the pool.",
                    no_update, no_update, no_update, no_update, no_update)
        if isinstance(selected, str):
            selected = [selected]

        motls = []
        for m in selected:
            try:
                motl_obj = _get_motl(m)
                motls.append(motl_obj)
            except _PoolPayloadMissing:
                pass
        if not motls:
            return (no_update, no_update, "The selected motls have no data.",
                    no_update, no_update, no_update, no_update, no_update)

        pool_state = PoolState.from_stores(registry, pool_meta, pool_next_id)
        nn_kwargs = _kwargs_by_cls(param_ids, param_values, "nn-params", pool_state)
        angular_kwargs = _kwargs_by_cls(param_ids, param_values, "nn-angular", pool_state)

        from cryocat.app.datapool import DataPoolState as _DPState
        from cryocat.app import provenance as _prov
        from cryocat.app.logger import invoke_operation as _invoke_op
        _ds = _DPState.from_stores(dp_registry, dp_next_id)
        _kind_count = _ds.kind_counters.get("nn", 0) + 1
        nn_key = f"nn-{_kind_count}"
        var = _prov.bind(nn_key)
        try:
            nn_input = motls[0] if len(motls) == 1 else motls
            nn_stats = _invoke_op(NearestNeighbors, {"input_data": nn_input, **nn_kwargs}, assign_to=var)
            normalized = nn_stats.get_normalized_coord(add_to_df=True)
            nn_stats.get_rotated_coord(add_to_df=True)
        except Exception as exc:
            return (no_update, no_update, f"Error: {exc}",
                    no_update, no_update, no_update, no_update, no_update)

        status_bits = []
        if nn_kwargs.get("nn_type") == "closest_dist" and "nn_dist" in nn_stats.df:
            status_bits.append(
                f"{len(nn_stats.df)} rows — "
                f"Mean distance: {nn_stats.df['nn_dist'].mean():.3f}; "
                f"Median distance: {nn_stats.df['nn_dist'].median():.3f}"
            )
        else:
            status_bits.append(f"NN analysis complete — {len(nn_stats.df)} neighbor rows.")

        if compute_dist and nn_kwargs.get("nn_type") == "radius":
            try:
                qp_coords = nn_stats.df[["qp_coord_x", "qp_coord_y", "qp_coord_z"]].to_numpy()
                nn_coords = nn_stats.df[["nn_coord_x", "nn_coord_y", "nn_coord_z"]].to_numpy()
                nn_stats.df["nn_dist"] = np.linalg.norm(nn_coords - qp_coords, axis=1)
                status_bits.append(
                    f"Mean distance: {nn_stats.df['nn_dist'].mean():.3f}; "
                    f"Median distance: {nn_stats.df['nn_dist'].median():.3f}"
                )
            except Exception as exc:
                status_bits.append(f"Euclidean distances skipped: {exc}")

        if angular_on:
            try:
                rot_type = angular_kwargs.get("rotation_type", "angular_distance")
                ang = run_operation(nn_stats.get_angular_distances, {"rotation_type": rot_type})
                if rot_type == "all":
                    nn_stats.df["angular_distance"] = ang[0]
                    nn_stats.df["cone_distance"] = ang[1]
                    nn_stats.df["in_plane_distance"] = ang[2]
                    status_bits.append("Angular distances: all 3 metrics added.")
                else:
                    nn_stats.df[rot_type] = ang
                    status_bits.append(f"Angular distance ({rot_type}) added.")
            except Exception as exc:
                status_bits.append(f"Angular distances skipped: {exc}")

        table_data = nn_stats.df.to_dict("records")
        from cryocat.app.suite.pages._motl_link import ordered_selection_to_motl_links
        nn_motl_links = ordered_selection_to_motl_links(selected)
        nn_ref = _datapool.insert(
            nn_stats.df, label=var, id_column="qp_subtomo_id",
            motl_links=nn_motl_links,
        )
        from cryocat.app import session as _session
        _prov.record(nn_key, _session.last_seq())

        _ds, _dp_id = _datapool.insert_entry(
            _ds, nn_stats.df,
            label=var,
            reader="nn",
            source_path="",
            motl_links=nn_motl_links,
            entry_kind="nn",
        )
        nn_ref = {**nn_ref, "data_id": _dp_id}
        new_dp_reg, new_dp_next = _ds.to_stores()

        # BK1: assign to lowest free slot; report if all full.
        # Pattern: cryocat/app/suite/motlsidebar.py lines 712–722.
        sm = list(slot_map or [None] * _NN_SLOTS)
        while len(sm) < _NN_SLOTS:
            sm.append(None)
        free = _first_free_slot(sm, _NN_SLOTS)
        if free is not None:
            sm[free] = _dp_id
            new_slot_map = sm
            status_bits.append(f"→ slot {free + 1}")
        else:
            new_slot_map = no_update
            status_bits.append(f"→ pool (all {_NN_SLOTS} slots in use)")

        nn_df = pd.DataFrame(
            np.column_stack((normalized, nn_stats.df["nn_subtomo_id"].values)),
            columns=["x", "y", "z", "nn_subtomo_id"],
        )
        _fig = visplot.plot_scatter_xyz_panels(
            nn_df, coord_columns=["x", "y", "z"], hover_column_name="nn_subtomo_id"
        )
        _fig_d = style_figure(_fig, gs_settings or {})
        xyz_graph = customel_graph("nn", "xyz", dcc.Graph(id={"type": "styled-graph", "owner": "nn", "name": "xyz"}, figure=go.Figure(_fig_d)))
        _nn_table_refs[_dp_id] = nn_ref
        _nn_xyz_figs[_dp_id] = _fig_d
        from cryocat.app.console.vars import register_console_var
        register_console_var(var, nn_stats.df)

        used_motls_store = {
            "names": selected,
            "is_multi": len(selected) > 1,
            "column_name": nn_stats.column_name,
        }

        return (
            xyz_graph, nn_ref, " | ".join(status_bits), table_data,
            used_motls_store, new_dp_reg, new_dp_next, new_slot_map,
        )

    # ── Post-processing: enrich existing NN table without recomputing. ────────
    @app.callback(
        Output("nn-result", "data", allow_duplicate=True),
        Output("nn-out-tabv-global-data-store", "data", allow_duplicate=True),
        Output("nn-pp-status", "children"),
        Output(ids.DATA_POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.DATA_POOL_NEXT_ID,  "data", allow_duplicate=True),
        Input("nn-pp-apply-btn", "n_clicks"),
        State("nn-result", "data"),
        State("nn-used-motls-store", "data"),
        State("nn-pp-add-cols", "value"),
        State("nn-pp-sides", "value"),
        State("nn-pp-angular-toggle", "value"),
        State({"type": "nn-forms-params", "owner": ALL, "cls_name": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": "nn-forms-params", "owner": ALL, "cls_name": ALL, "param": ALL, "tag": ALL}, "id"),
        State("nn-pp-dist-toggle", "value"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        State("nn-out-tabv-global-data-store", "data"),
        State(ids.DATA_POOL_REGISTRY, "data"),
        State(ids.DATA_POOL_NEXT_ID,  "data"),
        prevent_initial_call=True,
    )
    def _apply_postprocessing(
        n_clicks, nn_result, used_motls,
        add_cols, sides, angular_on, param_values, param_ids, dist_on,
        registry, pool_meta, pool_next_id,
        current_ref, dp_registry, dp_next_id,
    ):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not nn_result:
            return no_update, no_update, "Run NN analysis first.", no_update, no_update

        df = pd.DataFrame(nn_result)
        status_bits = []

        # Add columns from source motls via add_motl_columns
        if add_cols:
            if not used_motls:
                status_bits.append("Cannot add motl columns: no NN run found.")
            else:
                all_names = used_motls.get("names", [])
                is_multi = used_motls.get("is_multi", False)
                motl_list = []
                for name in all_names:
                    try:
                        motl_list.append(Motl(_get_rows(name)))
                    except _PoolPayloadMissing:
                        pass
                # Single-motl NearestNeighbors stores motls=[motl, motl] so that
                # motls[motl_id=1] is valid; mirror that here.
                if not is_multi and len(motl_list) == 1:
                    motl_list = [motl_list[0], motl_list[0]]
                if not motl_list:
                    status_bits.append("Cannot add motl columns: source motls not in pool.")
                else:
                    try:
                        nn_obj = NearestNeighbors(input_data=None)
                        nn_obj.df = df
                        nn_obj.motls = motl_list
                        active_sides = sides or ["qp", "nn"]
                        run_operation(nn_obj.add_motl_columns, {
                            "column_names": add_cols,
                            "sides": active_sides,
                            "add_to_df": True,
                        })
                        df = nn_obj.df
                        status_bits.append(f"Added {len(add_cols)} column(s) from source motls.")
                    except Exception as exc:
                        status_bits.append(f"add_motl_columns failed: {exc}")

        # Add angular distances
        pool_state = PoolState.from_stores(registry, pool_meta, pool_next_id)
        if angular_on:
            try:
                angular_kwargs = _kwargs_by_cls(param_ids, param_values, "nn-pp-angular", pool_state)
                rot_type = angular_kwargs.get("rotation_type", "angular_distance")
                nn_obj = NearestNeighbors(input_data=None)
                nn_obj.df = df
                ang = run_operation(nn_obj.get_angular_distances, {"rotation_type": rot_type})
                if rot_type == "all":
                    df["angular_distance"] = ang[0]
                    df["cone_distance"] = ang[1]
                    df["in_plane_distance"] = ang[2]
                    status_bits.append("Angular distances: all 3 metrics added.")
                else:
                    df[rot_type] = ang
                    status_bits.append(f"Angular distance ({rot_type}) added.")
            except Exception as exc:
                status_bits.append(f"Angular distances failed: {exc}")

        # Add Euclidean NN distances from stored coordinates
        if dist_on:
            try:
                qp = df[["qp_coord_x", "qp_coord_y", "qp_coord_z"]].to_numpy()
                nn = df[["nn_coord_x", "nn_coord_y", "nn_coord_z"]].to_numpy()
                df["nn_dist"] = np.linalg.norm(nn - qp, axis=1)
                status_bits.append(
                    f"NN distances added — mean: {df['nn_dist'].mean():.3f}, "
                    f"median: {df['nn_dist'].median():.3f}."
                )
            except Exception as exc:
                status_bits.append(f"NN distances failed: {exc}")

        if not status_bits:
            return no_update, no_update, "Nothing selected.", no_update, no_update

        new_data = df.to_dict("records")
        existing_links = (current_ref or {}).get("motl_links")
        nn_ref = _datapool.insert(
            df, label="NN analysis", id_column="qp_subtomo_id",
            motl_links=existing_links,
        )
        data_id = (current_ref or {}).get("data_id")
        nn_ref = {**nn_ref, "data_id": data_id} if data_id else nn_ref

        from cryocat.app.datapool import DataPoolState as _DPState
        ds = _DPState.from_stores(dp_registry, dp_next_id)
        if data_id:
            ds = _datapool.replace_entry(ds, data_id, df)
        new_dp_reg, new_dp_next = ds.to_stores()

        return new_data, nn_ref, " | ".join(status_bits), new_dp_reg, new_dp_next

    # ── Handle NN table loaded from file (via tablesource) ────────────────────
    @app.callback(
        Output("nn-xyz-graph-area", "children", allow_duplicate=True),
        Output("nn-out-tabv-global-data-store", "data", allow_duplicate=True),
        Output("nn-result", "data", allow_duplicate=True),
        Output("nn-used-motls-store", "data", allow_duplicate=True),
        Output(ids.DATA_POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.DATA_POOL_NEXT_ID,  "data", allow_duplicate=True),
        Output("nn-pool-slot-map", "data", allow_duplicate=True),
        Input("nn-src-ts-loaded", "data"),
        State("nn-value", "data"),
        State(ids.GRAPH_SETTINGS_STORE, "data"),
        State(ids.DATA_POOL_REGISTRY, "data"),
        State(ids.DATA_POOL_NEXT_ID,  "data"),
        State("nn-pool-slot-map", "data"),
        prevent_initial_call=True,
    )
    def _handle_nn_loaded(loaded, selected, gs_settings, dp_registry, dp_next_id, slot_map):
        if not loaded:
            raise dash.exceptions.PreventUpdate

        from io import StringIO
        from cryocat.app.datapool import DataPoolState as _DPState
        from cryocat.app import provenance as _prov

        df = pd.read_json(StringIO(loaded["df"]), orient="split")
        column_name = loaded.get("column_name", "tomo_id")

        _ds = _DPState.from_stores(dp_registry, dp_next_id)
        _kind_count = _ds.kind_counters.get("nn", 0) + 1
        nn_key = f"nn-{_kind_count}"
        var = _prov.bind(nn_key)

        selected_names = []
        motl_list = None
        is_multi = False

        if selected:
            if isinstance(selected, str):
                selected = [selected]
            motl_objs = []
            for m in selected:
                try:
                    motl_objs.append(Motl(_get_rows(m)))
                    selected_names.append(m)
                except _PoolPayloadMissing:
                    pass
            if motl_objs:
                is_multi = len(selected_names) > 1
                motl_list = motl_objs if is_multi else [motl_objs[0], motl_objs[0]]

        nn_stats = NearestNeighbors(input_data=None)
        nn_stats.df = df
        nn_stats.column_name = column_name
        nn_stats.motls = motl_list
        nn_stats.paired = False

        xyz_graph = no_update
        _fig_d = None
        try:
            normalized = nn_stats.get_normalized_coord(add_to_df=True)
            nn_stats.get_rotated_coord(add_to_df=True)
            nn_df = pd.DataFrame(
                np.column_stack((normalized, nn_stats.df["nn_subtomo_id"].values)),
                columns=["x", "y", "z", "nn_subtomo_id"],
            )
            _fig = visplot.plot_scatter_xyz_panels(
                nn_df, coord_columns=["x", "y", "z"], hover_column_name="nn_subtomo_id"
            )
            _fig_d = style_figure(_fig, gs_settings or {})
            xyz_graph = customel_graph("nn", "xyz", dcc.Graph(id={"type": "styled-graph", "owner": "nn", "name": "xyz"}, figure=go.Figure(_fig_d)))
        except Exception:
            pass

        table_data = nn_stats.df.to_dict("records")
        loaded_links = loaded.get("motl_links")
        nn_ref = _datapool.insert(
            nn_stats.df,
            label=var,
            id_column="qp_subtomo_id",
            motl_links=loaded_links,
        )
        used_motls_store = {
            "names": selected_names,
            "is_multi": is_multi,
            "column_name": column_name,
        }

        _ds, _dp_id = _datapool.insert_entry(
            _ds, nn_stats.df,
            label=var,
            reader="nn",
            source_path="",
            motl_links=loaded_links,
            entry_kind="nn",
        )
        nn_ref = {**nn_ref, "data_id": _dp_id}
        _nn_table_refs[_dp_id] = nn_ref
        if _fig_d is not None:
            _nn_xyz_figs[_dp_id] = _fig_d
        from cryocat.app.console.vars import register_console_var
        register_console_var(var, nn_stats.df)
        new_dp_reg, new_dp_next = _ds.to_stores()

        # BK1: assign to lowest free slot; report if all full.
        # Pattern: cryocat/app/suite/motlsidebar.py lines 712–722.
        sm = list(slot_map or [None] * _NN_SLOTS)
        while len(sm) < _NN_SLOTS:
            sm.append(None)
        free = _first_free_slot(sm, _NN_SLOTS)
        if free is not None:
            sm[free] = _dp_id
            new_slot_map = sm
        else:
            new_slot_map = no_update

        return (
            xyz_graph, nn_ref, table_data, used_motls_store,
            new_dp_reg, new_dp_next, new_slot_map,
        )

    # ── W1/W3: Populate source-motl dropdown options from pool registry ───────

    @app.callback(
        Output({"type": "nn-src-ts-extra", "param": "motl_selection"}, "options"),
        Input(ids.POOL_REGISTRY, "data"),
    )
    def _populate_nn_source_motl_options(registry):
        opts = [
            {"label": v.get("label", k), "value": k}
            for k, v in (registry or {}).items()
        ]
        return opts

    # ── W4: Gate table-to-motl buttons on motl_links["query"] ────────────────

    @app.callback(
        Output("nn-ttm-ttm-write-btn", "disabled"),
        Output("nn-ttm-ttm-write-btn", "title"),
        Output("nn-ttm-ttm-create-btn", "disabled"),
        Output("nn-ttm-ttm-create-btn", "title"),
        Input("nn-out-tabv-global-data-store", "data"),
    )
    def _gate_nn_ttm(ref):
        from cryocat.app.suite.pages._motl_link import get_motl_role_id
        query_mid = get_motl_role_id((ref or {}).get("motl_links"), "query")
        if query_mid:
            return False, f"Targets query motl: {query_mid}", False, f"Targets query motl: {query_mid}"
        msg = "Load data with a source motl selected to enable motl operations."
        return True, msg, True, msg

    # ── Sync filtered NN registry from the global data pool ──────────────────

    @app.callback(
        Output("nn-pool-registry", "data"),
        Input(ids.DATA_POOL_REGISTRY, "data"),
    )
    def _sync_nn_pool_registry(dp_registry):
        return {
            k: v for k, v in (dp_registry or {}).items()
            if v.get("reader") == "nn"
        }

    # ── Slot tab labels follow slot map + pool registry ───────────────────────

    @app.callback(
        *[Output(f"nn-slot-tab-{i}", "label") for i in range(_NN_SLOTS)],
        *[Output(f"nn-slot-tab-{i}", "disabled") for i in range(_NN_SLOTS)],
        Input("nn-pool-slot-map", "data"),
        State("nn-pool-registry", "data"),
    )
    def _update_nn_slot_tabs(slot_map, pool_registry):
        reg = pool_registry or {}
        sm = list(slot_map or [None] * _NN_SLOTS)
        labels, disableds = [], []
        for i, data_id in enumerate(sm[:_NN_SLOTS]):
            if data_id and data_id in reg:
                labels.append(reg[data_id].get("label", data_id))
                disableds.append(False)
            else:
                labels.append(f"Slot {i + 1}")
                disableds.append(True)
        return *labels, *disableds

    # ── Tab click → active-id (also fires when slot focus shifts) ─────────────

    @app.callback(
        Output("nn-pool-active-id", "data", allow_duplicate=True),
        Input("nn-slot-tabs", "active_tab"),
        State("nn-pool-slot-map", "data"),
        prevent_initial_call=True,
    )
    def _on_nn_slot_tab_clicked(active_tab, slot_map):
        if not active_tab:
            raise dash.exceptions.PreventUpdate
        try:
            i = int(active_tab.split("-")[-1])
        except (IndexError, ValueError):
            raise dash.exceptions.PreventUpdate
        data_id = (slot_map or [])[i] if slot_map and i < len(slot_map) else None
        if not data_id:
            raise dash.exceptions.PreventUpdate
        return data_id

    # ── Active-id → load table + xyz from server-side refs ───────────────────

    @app.callback(
        Output("nn-out-tabv-global-data-store", "data", allow_duplicate=True),
        Output("nn-xyz-graph-area", "children", allow_duplicate=True),
        Input("nn-pool-active-id", "data"),
        prevent_initial_call=True,
    )
    def _on_nn_active_id_change(data_id):
        if not data_id:
            raise dash.exceptions.PreventUpdate
        table_ref = _nn_table_refs.get(data_id)
        if table_ref is None:
            raise dash.exceptions.PreventUpdate
        xyz_fig_dict = _nn_xyz_figs.get(data_id)
        xyz_child = customel_graph("nn", "xyz", dcc.Graph(id={"type": "styled-graph", "owner": "nn", "name": "xyz"}, figure=go.Figure(xyz_fig_dict))) if xyz_fig_dict else no_update
        return table_ref, xyz_child
