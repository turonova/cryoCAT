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

import numpy as np
import pandas as pd

import dash
from dash import html, dcc, ctx, Input, Output, State, ALL, no_update
import dash_bootstrap_components as dbc

from cryocat.core.cryomotl import Motl
from cryocat.analysis.nnana import NearestNeighbors
from cryocat.analysis import visplot
from cryocat.app.apputils import generate_kwargs, run_operation
from cryocat.app import ids, formgen
from cryocat.app.formgen import make_dropdown
from cryocat.app.pool import (
    get_rows as _get_rows,
    get_motl as _get_motl, PoolPayloadMissing as _PoolPayloadMissing,
)
from cryocat.app.components.poolpicker import get_pool_picker, register_pool_picker_callbacks
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

_nn_counter: list[int] = [0]

from cryocat.app import datapool as _datapool


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


def _nn_load_fn(path):
    """Load an NN table from a CSV file; returns dict with df and column_name."""
    nn = NearestNeighbors.load(file_path=path)
    return {"df": nn.df, "column_name": nn.column_name}


def _sidebar() -> list:
    return [
        sidebar_accordion(
            [
                dbc.AccordionItem(
                    get_table_source(
                        "nn-src",
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
                                            ),
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
            ],
            active_item=["nn-acc-params"],
        ),
    ]


def _main() -> list:
    return [
        dcc.Store(id="nn-out-tabv-global-data-store"),
        get_table_component("nn-out-tabv"),
        html.Hr(style={"margin": "0.5rem 0"}),
        html.Div(id="nn-xyz-graph-area"),
    ]


layout = html.Div(
    [
        dcc.Store(id="nn-result"),
        # Ordered list of pool motl-ids used in the last NN run, plus is_multi flag.
        dcc.Store(id="nn-used-motls-store"),
        dcc.Store(id="nn-cluster-cols-store", data=[]),
        page_shell(_sidebar(), _main()),
    ],
    style={"margin": "0", "padding": "0"},
)


# ── Callback helpers ───────────────────────────────────────────────────────────

def _kwargs_by_cls(param_ids, param_values, target_cls):
    """Demux flat ALL-state to kwargs for a single ``cls_name``."""
    ids, vals = [], []
    for pid, val in zip(param_ids, param_values):
        if pid.get("cls_name") == target_cls:
            ids.append(pid)
            vals.append(val)
    return generate_kwargs(ids, vals) if ids else {}


# ── Callbacks ──────────────────────────────────────────────────────────────────

def register_callbacks(app):
    register_pool_picker_callbacks(app, "nn")
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
    )
    register_table_plot_callbacks(
        app, "nn-out-tabv-table-plot", "nn-out-tabv-global-data-store",
        special_graphs=["Orientational distribution", "Polar NN distances"],
        table_grid_id="nn-out-tabv-grid",
    )
    register_table_cluster_callbacks(
        app, "nn-out-tabv-table-cluster", "nn-out-tabv-global-data-store",
        table_grid_id="nn-out-tabv-grid",
        cluster_cols_store_id="nn-cluster-cols-store",
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
        Input("nn-compute-btn", "n_clicks"),
        State("nn-value", "data"),
        State({"type": "nn-forms-params", "owner": ALL, "cls_name": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": "nn-forms-params", "owner": ALL, "cls_name": ALL, "param": ALL, "tag": ALL}, "id"),
        State("nn-angular-toggle", "value"),
        State("nn-dist-toggle", "value"),
        prevent_initial_call=True,
    )
    def compute_nn(n_clicks, selected, param_values, param_ids, angular_on, compute_dist):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        if not selected:
            return (no_update, no_update, "Select at least one motl from the pool.",
                    no_update, no_update)
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
                    no_update, no_update)

        nn_kwargs = _kwargs_by_cls(param_ids, param_values, "nn-params")
        angular_kwargs = _kwargs_by_cls(param_ids, param_values, "nn-angular")

        _nn_counter[0] += 1
        nn_key = f"nn-{_nn_counter[0]}"
        from cryocat.app import provenance as _prov
        from cryocat.app.logger import invoke_operation as _invoke_op
        var = _prov.bind(nn_key)
        try:
            nn_input = motls[0] if len(motls) == 1 else motls
            nn_stats = _invoke_op(NearestNeighbors, {"input_data": nn_input, **nn_kwargs}, assign_to=var)
            normalized = nn_stats.get_normalized_coord(add_to_df=True)
            nn_stats.get_rotated_coord(add_to_df=True)
        except Exception as exc:
            return (no_update, no_update, f"Error: {exc}",
                    no_update, no_update)

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
        nn_ref = _datapool.insert(nn_stats.df, label="NN analysis", id_column="qp_subtomo_id")
        from cryocat.app import session as _session
        _prov.record(nn_key, _session.last_seq())

        nn_df = pd.DataFrame(
            np.column_stack((normalized, nn_stats.df["nn_subtomo_id"].values)),
            columns=["x", "y", "z", "nn_subtomo_id"],
        )
        xyz_graph = dcc.Graph(
            figure=visplot.plot_scatter_xyz_panels(
                nn_df, coord_columns=["x", "y", "z"], hover_column_name="nn_subtomo_id"
            )
        )

        used_motls_store = {
            "names": selected,
            "is_multi": len(selected) > 1,
            "column_name": nn_stats.column_name,
        }

        return (
            xyz_graph, nn_ref, " | ".join(status_bits), table_data,
            used_motls_store,
        )

    # ── Post-processing: enrich existing NN table without recomputing. ────────
    @app.callback(
        Output("nn-result", "data", allow_duplicate=True),
        Output("nn-out-tabv-global-data-store", "data", allow_duplicate=True),
        Output("nn-pp-status", "children"),
        Input("nn-pp-apply-btn", "n_clicks"),
        State("nn-result", "data"),
        State("nn-used-motls-store", "data"),
        State("nn-pp-add-cols", "value"),
        State("nn-pp-sides", "value"),
        State("nn-pp-angular-toggle", "value"),
        State({"type": "nn-forms-params", "owner": ALL, "cls_name": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": "nn-forms-params", "owner": ALL, "cls_name": ALL, "param": ALL, "tag": ALL}, "id"),
        State("nn-pp-dist-toggle", "value"),
        prevent_initial_call=True,
    )
    def _apply_postprocessing(
        n_clicks, nn_result, used_motls,
        add_cols, sides, angular_on, param_values, param_ids, dist_on,
    ):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not nn_result:
            return no_update, no_update, "Run NN analysis first."

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
        if angular_on:
            try:
                angular_kwargs = _kwargs_by_cls(param_ids, param_values, "nn-pp-angular")
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
            return no_update, no_update, "Nothing selected."

        new_data = df.to_dict("records")
        nn_ref = _datapool.insert(df, label="NN analysis", id_column="qp_subtomo_id")
        return new_data, nn_ref, " | ".join(status_bits)

    # ── Handle NN table loaded from file (via tablesource) ────────────────────
    @app.callback(
        Output("nn-xyz-graph-area", "children", allow_duplicate=True),
        Output("nn-out-tabv-global-data-store", "data", allow_duplicate=True),
        Output("nn-result", "data", allow_duplicate=True),
        Output("nn-used-motls-store", "data", allow_duplicate=True),
        Input("nn-src-ts-loaded", "data"),
        State("nn-value", "data"),
        prevent_initial_call=True,
    )
    def _handle_nn_loaded(loaded, selected):
        if not loaded:
            raise dash.exceptions.PreventUpdate

        from io import StringIO
        df = pd.read_json(StringIO(loaded["df"]), orient="split")
        column_name = loaded.get("column_name", "tomo_id")

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
        try:
            normalized = nn_stats.get_normalized_coord(add_to_df=True)
            nn_stats.get_rotated_coord(add_to_df=True)
            nn_df = pd.DataFrame(
                np.column_stack((normalized, nn_stats.df["nn_subtomo_id"].values)),
                columns=["x", "y", "z", "nn_subtomo_id"],
            )
            xyz_graph = dcc.Graph(
                figure=visplot.plot_scatter_xyz_panels(
                    nn_df, coord_columns=["x", "y", "z"], hover_column_name="nn_subtomo_id"
                )
            )
        except Exception:
            pass

        table_data = nn_stats.df.to_dict("records")
        nn_ref = _datapool.insert(nn_stats.df, label="NN analysis", id_column="qp_subtomo_id")
        used_motls_store = {
            "names": selected_names,
            "is_multi": is_multi,
            "column_name": column_name,
        }

        return (
            xyz_graph, nn_ref, table_data, used_motls_store,
        )
