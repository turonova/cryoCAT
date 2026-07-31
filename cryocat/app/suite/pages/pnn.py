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
from cryocat.app.pool import PoolState, insert_motl as _insert_motl
from cryocat.app.components.motlsource import get_motl_source, register_motl_source_callbacks
from cryocat.app.components.tableview import get_table_component, register_table_callbacks
from cryocat.app.components.tableplot import register_table_plot_callbacks
from cryocat.app.components.tablecluster import register_table_cluster_callbacks
from cryocat.app.pageshell import page_shell, sidebar_accordion


_MOTL_COL_OPTIONS = [{"label": c, "value": c} for c in Motl.motl_columns]
_NN_TRANSFER_EXCLUDE = {"motl_id", "qp_id", "nn_id"}


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

def _create_motl_sidebar_content():
    """Sidebar content for the 'Create motl from NN table' accordion item."""
    hint = {"fontSize": "0.8rem", "color": "var(--color9)", "marginBottom": "0.4rem"}
    lbl = {"fontWeight": "bold", "fontSize": "0.85rem", "marginBottom": "0.2rem"}
    return html.Div(
        [
            html.Div(id="nn-sel-motl-info", style=hint),
            html.Label("Motls to include:", style=lbl),
            html.P(
                "Order matters — first = query-particle motl, subsequent = neighbor motls.",
                style=hint,
            ),
            dcc.Checklist(
                id="nn-sel-motl-checklist",
                options=[],
                value=[],
                labelStyle={"display": "block", "marginBottom": "0.2rem", "fontSize": "0.85rem"},
            ),
            html.Hr(style={"margin": "0.4rem 0"}),
            html.Label("Rows to include:", style=lbl),
            dbc.RadioItems(
                id="nn-sel-rows-mode",
                options=[
                    {"label": "All", "value": "all"},
                    {"label": "Selected rows only", "value": "selected"},
                ],
                value="all",
                inline=False,
                className="sidebar-checklist",
                labelStyle={"fontSize": "0.85rem"},
            ),
            html.Div(
                [
                    html.Hr(style={"margin": "0.4rem 0"}),
                    html.Label("Use particle IDs from:", style=lbl),
                    dbc.RadioItems(
                        id="nn-sel-motl-id-type",
                        options=[
                            {"label": "Query particle (qp_subtomo_id)", "value": "qp"},
                            {"label": "Neighbor (nn_subtomo_id)", "value": "nn"},
                        ],
                        value="qp",
                        inline=False,
                        className="sidebar-checklist",
                        labelStyle={"fontSize": "0.85rem"},
                    ),
                ],
                id="nn-sel-motl-id-type-wrap",
                style={"display": "none"},
            ),
            html.Div(
                [
                    html.Hr(style={"margin": "0.4rem 0"}),
                    html.Label("Source-motl index column:", style=lbl),
                    dcc.Dropdown(
                        id="nn-sel-motl-id-col",
                        options=_MOTL_COL_OPTIONS,
                        placeholder="Select column…",
                        style={"marginBottom": "0.4rem"},
                    ),
                ],
                id="nn-sel-motl-id-col-wrap",
                style={"display": "none"},
            ),
            html.Div(
                [
                    html.Hr(style={"margin": "0.4rem 0"}),
                    html.Label("Transfer NN columns to motl:", style=lbl),
                    html.P(
                        "Select columns from the NN table to carry over. "
                        "Map each to a target motl column.",
                        style=hint,
                    ),
                    dcc.Dropdown(
                        id="nn-sel-nn-cols",
                        options=[],
                        value=[],
                        multi=True,
                        placeholder="Select NN columns to transfer…",
                        style={"marginBottom": "0.4rem"},
                    ),
                    html.Div(id="nn-sel-nn-col-target-rows"),
                ],
                id="nn-cluster-transfer-wrap",
                style={"display": "none"},
            ),
            html.Hr(style={"margin": "0.4rem 0"}),
            html.Label("Save to file:", style=lbl),
            dbc.Input(
                id="nn-sel-motl-save-path",
                placeholder="Output file path (.em, .csv, …)",
                size="sm",
                style={"marginBottom": "0.3rem"},
            ),
            dbc.Button(
                "Save",
                id="nn-sel-motl-save-btn",
                color="secondary",
                size="sm",
                style={"width": "100%", "marginBottom": "0.5rem"},
            ),
            html.Label("Send to editor:", style=lbl),
            dbc.Input(
                id="nn-sel-motl-editor-label",
                placeholder="Label (optional)",
                size="sm",
                style={"marginBottom": "0.3rem"},
            ),
            dbc.Button(
                "Send to editor",
                id="nn-sel-motl-send-btn",
                color="primary",
                size="sm",
                style={"width": "100%"},
            ),
            html.Div(
                id="nn-sel-motl-status",
                style={"fontSize": "0.85rem", "color": "var(--color9)",
                       "marginTop": "0.4rem", "wordBreak": "break-word"},
            ),
        ]
    )


def _postprocess_sidebar_content():
    """Sidebar content for the 'Post-processing' accordion item."""
    hint = {"fontSize": "0.8rem", "color": "var(--color9)", "marginBottom": "0.4rem"}
    lbl = {"fontWeight": "bold", "fontSize": "0.85rem", "marginBottom": "0.2rem"}
    return html.Div(
        [
            html.Label("Add columns from source motls:", style=lbl),
            html.P(
                "Enrich the NN table with extra columns pulled from the original motls. "
                "Columns are added as qp_<col> and/or nn_<col>.",
                style=hint,
            ),
            dcc.Dropdown(
                id="nn-pp-add-cols",
                options=_MOTL_COL_OPTIONS,
                multi=True,
                placeholder="Select motl columns to add…",
                style={"marginBottom": "0.3rem"},
            ),
            html.Label("Add for sides:", style={"fontSize": "0.85rem", "marginBottom": "0.2rem"}),
            dbc.Checklist(
                id="nn-pp-sides",
                options=[
                    {"label": "Query particle (qp)", "value": "qp"},
                    {"label": "Neighbor (nn)", "value": "nn"},
                ],
                value=["qp", "nn"],
                inline=True,
                labelStyle={"fontSize": "0.85rem"},
                style={"marginBottom": "0.5rem"},
            ),
            html.Hr(style={"margin": "0.4rem 0"}),
            dbc.Checkbox(
                id="nn-pp-angular-toggle",
                label="Add angular distances",
                value=False,
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
            dbc.Checkbox(
                id="nn-pp-dist-toggle",
                label="Add NN distances",
                value=False,
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
                    "fontSize": "0.85rem",
                    "color": "var(--color9)",
                    "marginTop": "0.4rem",
                    "wordBreak": "break-word",
                },
            ),
        ]
    )


def _load_csv_sidebar_content():
    """Sidebar content for the 'Load from CSV' accordion item."""
    hint = {"fontSize": "0.8rem", "color": "var(--color9)", "marginBottom": "0.4rem"}
    lbl = {"fontWeight": "bold", "fontSize": "0.85rem", "marginBottom": "0.2rem"}
    return html.Div(
        [
            html.P(
                "Load a previously saved NN table. Motls selected in "
                "'Input motls' will be reattached; leave none selected for a "
                "data-only load (post-processing will still work, but 'Create "
                "motl from NN table' requires attached motls).",
                style=hint,
            ),
            html.Label("CSV file path:", style=lbl),
            dbc.Input(
                id="nn-load-csv-path",
                placeholder="Path to .csv file…",
                size="sm",
                style={"marginBottom": "0.5rem"},
            ),
            dbc.Button(
                "Load",
                id="nn-load-csv-btn",
                color="primary",
                size="sm",
                style={"width": "100%"},
            ),
            html.Div(
                id="nn-load-csv-status",
                style={
                    "fontSize": "0.85rem",
                    "color": "var(--color9)",
                    "marginTop": "0.4rem",
                    "wordBreak": "break-word",
                },
            ),
        ]
    )


def _sidebar() -> list:
    return [
        sidebar_accordion(
            [
                dbc.AccordionItem(
                    get_motl_source("nn", multi=True),
                    title="Input motls",
                    item_id="nn-acc-input",
                ),
                dbc.AccordionItem(
                    [
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
                            "",
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
                                            style={"fontSize": "0.85rem", "margin": 0},
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
                                    dcc.Dropdown(
                                        id={
                                            "type": "nn-forms-params",
                                            "owner": "",
                                            "param": "exclude_column_name",
                                            "tag": "Literal",
                                            "cls_name": "nn-params",
                                        },
                                        options=_MOTL_COL_OPTIONS,
                                        value=None,
                                        clearable=True,
                                        searchable=True,
                                        placeholder="None (optional)",
                                        style={"width": "100%"},
                                    ),
                                    style={"width": "55%"},
                                ),
                            ],
                            style={
                                "display": "flex", "flexDirection": "row",
                                "marginBottom": "0.25rem", "width": "100%",
                                "alignItems": "center",
                            },
                        ),
                        html.Div(
                            dbc.Checkbox(
                                id="nn-dist-toggle",
                                label="Compute euclidean distances",
                                value=False,
                            ),
                            id="nn-dist-toggle-wrap",
                            style={"display": "none", "marginTop": "0.3rem"},
                        ),
                        dbc.Checkbox(
                            id="nn-angular-toggle",
                            label="Compute angular distances",
                            value=False,
                            style={"marginTop": "0.5rem", "marginBottom": "0.4rem"},
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
                                "fontSize": "0.85rem",
                                "color": "var(--color9)",
                                "marginTop": "0.5rem",
                                "wordBreak": "break-word",
                            },
                        ),
                    ],
                    title="Compute NN table",
                    item_id="nn-acc-params",
                ),
                dbc.AccordionItem(
                    _load_csv_sidebar_content(),
                    title="Load existing NN table",
                    item_id="nn-acc-load",
                ),
                dbc.AccordionItem(
                    _postprocess_sidebar_content(),
                    title="Post-processing",
                    item_id="nn-acc-postprocess",
                ),
                dbc.AccordionItem(
                    _create_motl_sidebar_content(),
                    title="Create motl from NN table",
                    item_id="nn-acc-create",
                ),
            ],
            active_item=["nn-acc-input", "nn-acc-params"],
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
    register_motl_source_callbacks(app, "nn", multi=True)
    register_table_callbacks(
        app, "nn-out-tabv",
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
        Output("nn-sel-nn-cols", "options"),
        Output("nn-cluster-transfer-wrap", "style"),
        Input("nn-out-tabv-global-data-store", "data"),
    )
    def _update_nn_col_transfer_options(table_data):
        if not table_data:
            return [], {"display": "none"}
        df_cols = pd.DataFrame(table_data).columns.tolist()
        opts = [
            {"label": c, "value": c}
            for c in df_cols
            if c not in _NN_TRANSFER_EXCLUDE
        ]
        if not opts:
            return [], {"display": "none"}
        return opts, {"display": "block"}

    @app.callback(
        Output("nn-sel-nn-col-target-rows", "children"),
        Input("nn-sel-nn-cols", "value"),
    )
    def _build_nn_col_target_rows(selected_cols):
        if not selected_cols:
            return []
        rows = []
        for col in selected_cols:
            rows.append(
                html.Div(
                    [
                        html.Span(
                            f"{col} →",
                            style={
                                "fontSize": "0.8rem",
                                "whiteSpace": "nowrap",
                                "marginRight": "6px",
                                "color": "var(--color9)",
                            },
                        ),
                        dcc.Dropdown(
                            id={"type": "nn-sel-nn-col-target", "col": col},
                            options=_MOTL_COL_OPTIONS,
                            placeholder="Motl column…",
                            searchable=True,
                            clearable=True,
                            style={"flex": 1, "fontSize": "0.8rem"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "marginBottom": "0.3rem",
                        "width": "100%",
                    },
                )
            )
        return rows

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
        State("nn-motl-select", "value"),
        State(ids.POOL_MOTLS, "data"),
        State({"type": "nn-forms-params", "owner": ALL, "cls_name": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": "nn-forms-params", "owner": ALL, "cls_name": ALL, "param": ALL, "tag": ALL}, "id"),
        State("nn-angular-toggle", "value"),
        State("nn-dist-toggle", "value"),
        prevent_initial_call=True,
    )
    def compute_nn(n_clicks, selected, pool_motls, param_values, param_ids, angular_on, compute_dist):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        pool_motls = pool_motls or {}
        if not selected:
            return (no_update, no_update, "Select at least one motl from the pool.",
                    no_update, no_update)
        if isinstance(selected, str):
            selected = [selected]

        motls = [Motl(pd.DataFrame(pool_motls[m])) for m in selected if pool_motls.get(m)]
        if not motls:
            return (no_update, no_update, "The selected motls have no data.",
                    no_update, no_update)

        nn_kwargs = _kwargs_by_cls(param_ids, param_values, "nn-params")
        angular_kwargs = _kwargs_by_cls(param_ids, param_values, "nn-angular")

        try:
            nn_input = motls[0] if len(motls) == 1 else motls
            nn_stats = NearestNeighbors(nn_input, **nn_kwargs)
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
                ang = nn_stats.get_angular_distances(rotation_type=rot_type)
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
            xyz_graph, table_data, " | ".join(status_bits), table_data,
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
        State(ids.POOL_MOTLS, "data"),
        State("nn-pp-add-cols", "value"),
        State("nn-pp-sides", "value"),
        State("nn-pp-angular-toggle", "value"),
        State({"type": "nn-forms-params", "owner": ALL, "cls_name": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": "nn-forms-params", "owner": ALL, "cls_name": ALL, "param": ALL, "tag": ALL}, "id"),
        State("nn-pp-dist-toggle", "value"),
        prevent_initial_call=True,
    )
    def _apply_postprocessing(
        n_clicks, nn_result, used_motls, pool_motls,
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
                pool_motls = pool_motls or {}
                motl_list = [
                    Motl(pd.DataFrame(pool_motls[name]))
                    for name in all_names
                    if pool_motls.get(name)
                ]
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
                        nn_obj.add_motl_columns(add_cols, sides=active_sides, add_to_df=True)
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
                ang = nn_obj.get_angular_distances(rotation_type=rot_type)
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
        return new_data, new_data, " | ".join(status_bits)

    # ── Create motl from selection (sidebar) ────────────────────────────────────

    @app.callback(
        Output("nn-sel-motl-checklist", "options"),
        Output("nn-sel-motl-checklist", "value"),
        Output("nn-sel-motl-id-type-wrap", "style"),
        Output("nn-sel-motl-info", "children"),
        Input("nn-used-motls-store", "data"),
        State(ids.POOL_REGISTRY, "data"),
        prevent_initial_call=True,
    )
    def _populate_sel_panel(used_motls, registry):
        if not used_motls:
            return [], [], {"display": "none"}, "Run NN analysis first."

        names = used_motls.get("names", [])
        is_multi = used_motls.get("is_multi", False)
        registry = registry or {}

        options = []
        for i, name in enumerate(names):
            lbl = registry.get(name, {}).get("label", name)
            role = "qp — query particle" if i == 0 else f"nn #{i} — neighbor"
            options.append({"label": f"{i + 1}.  {lbl}  ({role})", "value": name})

        info = (
            "Single-motl analysis — choose query-particle or neighbor IDs."
            if not is_multi
            else (
                f"Multi-motl analysis — {len(names)} motl(s). "
                "First uses qp_subtomo_id; subsequent use nn_subtomo_id."
            )
        )
        id_type_style = {"display": "block"} if not is_multi else {"display": "none"}
        return options, list(names), id_type_style, info

    @app.callback(
        Output("nn-sel-motl-id-col-wrap", "style"),
        Input("nn-sel-motl-checklist", "value"),
    )
    def _toggle_id_col_wrap(checked):
        return {"display": "block"} if checked and len(checked) > 1 else {"display": "none"}

    @app.callback(
        Output("nn-sel-motl-status", "children"),
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_MOTLS, "data", allow_duplicate=True),
        Output(ids.POOL_EXTRA, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Input("nn-sel-motl-save-btn", "n_clicks"),
        Input("nn-sel-motl-send-btn", "n_clicks"),
        State("nn-out-tabv-grid", "selectedRows"),
        State("nn-out-tabv-grid", "rowData"),
        State("nn-sel-rows-mode", "value"),
        State("nn-sel-motl-checklist", "value"),
        State("nn-used-motls-store", "data"),
        State("nn-sel-motl-id-type", "value"),
        State("nn-sel-motl-id-col", "value"),
        State("nn-sel-motl-save-path", "value"),
        State("nn-sel-motl-editor-label", "value"),
        State(ids.POOL_MOTLS, "data"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_EXTRA, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        State("nn-sel-nn-cols", "value"),
        State({"type": "nn-sel-nn-col-target", "col": ALL}, "value"),
        State({"type": "nn-sel-nn-col-target", "col": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _build_and_act(
        _save_click, _send_click,
        selected_rows, all_rows, rows_mode,
        checked_motls, used_motls, id_type, id_col,
        save_path, editor_label, pool_motls, registry, pool_extra, pool_meta, next_id,
        nn_cols, nn_col_target_vals, nn_col_target_ids,
    ):
        trigger = ctx.triggered_id
        rows_mode = rows_mode or "all"
        if rows_mode == "all":
            active_rows = all_rows or []
        else:
            active_rows = selected_rows or []
        _nu5 = (no_update,) * 5
        if not active_rows:
            msg = "No rows in the table." if rows_mode == "all" else "No rows selected in the table."
            return msg, *_nu5
        if not checked_motls:
            return "No motls checked.", *_nu5
        if not used_motls:
            return "Run NN analysis first.", *_nu5

        all_names = used_motls.get("names", [])
        is_multi = used_motls.get("is_multi", False)
        pool_motls = pool_motls or {}

        # Build NN-column → motl-column mapping
        col_mapping = {}
        if nn_cols and nn_col_target_vals:
            for tid, tval in zip(nn_col_target_ids, nn_col_target_vals):
                src = tid.get("col")
                if src and src in nn_cols and tval:
                    col_mapping[src] = tval
            used_targets = list(col_mapping.values())
            if len(used_targets) != len(set(used_targets)):
                return (
                    "Two NN columns cannot map to the same motl column.",
                    no_update, no_update, no_update,
                )

        sel_df = pd.DataFrame(active_rows)
        parts = []
        for i, motl_name in enumerate(all_names):
            if motl_name not in checked_motls:
                continue
            pool_data = pool_motls.get(motl_name)
            if not pool_data:
                continue

            motl_df = pd.DataFrame(pool_data)

            if is_multi:
                if i == 0:
                    ids = set(sel_df["qp_subtomo_id"].dropna().astype(float).values)
                else:
                    mask = sel_df["motl_id"].astype(float) == float(i)
                    ids = set(sel_df.loc[mask, "nn_subtomo_id"].dropna().astype(float).values)
            else:
                col = "qp_subtomo_id" if (id_type or "qp") == "qp" else "nn_subtomo_id"
                ids = set(sel_df[col].dropna().astype(float).values)

            subset = motl_df[motl_df["subtomo_id"].isin(ids)].copy()
            if col_mapping and len(subset) > 0:
                if is_multi:
                    if i == 0:
                        id_col_nn = "qp_subtomo_id"
                        slice_df = sel_df
                    else:
                        id_col_nn = "nn_subtomo_id"
                        slice_df = sel_df[sel_df["motl_id"].astype(float) == float(i)]
                else:
                    id_col_nn = "qp_subtomo_id" if (id_type or "qp") == "qp" else "nn_subtomo_id"
                    slice_df = sel_df
                for src_col, dst_col in col_mapping.items():
                    if src_col in slice_df.columns:
                        id_to_val = (
                            slice_df.drop_duplicates(subset=[id_col_nn])
                            .set_index(id_col_nn)[src_col]
                            .dropna()
                            .to_dict()
                        )
                        subset[dst_col] = subset["subtomo_id"].map(id_to_val)
            if len(subset) == 0:
                continue
            if id_col and len(checked_motls) > 1:
                subset[id_col] = float(i)
            parts.append(subset)

        if not parts:
            return (
                "No particles matched the selection. "
                "Make sure rows are selected and the motl IDs align.",
                *_nu5,
            )

        merged_df = pd.concat(parts, ignore_index=True)

        if trigger == "nn-sel-motl-save-btn":
            if not save_path:
                return "Specify an output file path.", *_nu5
            try:
                m = Motl(merged_df)
                run_operation(m.save_to, {"output_path": save_path})
            except Exception as exc:
                return f"Save failed: {exc}", *_nu5
            return f"Saved {len(merged_df)} particles to {save_path}.", *_nu5

        if trigger == "nn-sel-motl-send-btn":
            pool_state = PoolState.from_stores(registry, pool_motls, pool_extra, pool_meta, next_id)
            # TODO(P9): route through run_operation_to_pool once NN merge is tracked.
            pool_state, new_id = _insert_motl(
                pool_state, merged_df.to_dict("records"), label=editor_label,
            )
            display_label = pool_state.registry[new_id]["label"]
            return (
                f"Sent '{display_label}' ({len(merged_df)} particles) to the editor.",
                *pool_state.to_stores(),
            )

        return no_update, *_nu5

    # ── Load NN table from CSV ────────────────────────────────────────────────
    @app.callback(
        Output("nn-xyz-graph-area", "children", allow_duplicate=True),
        Output("nn-out-tabv-global-data-store", "data", allow_duplicate=True),
        Output("nn-result", "data", allow_duplicate=True),
        Output("nn-used-motls-store", "data", allow_duplicate=True),
        Output("nn-load-csv-status", "children"),
        Input("nn-load-csv-btn", "n_clicks"),
        State("nn-load-csv-path", "value"),
        State("nn-motl-select", "value"),
        State(ids.POOL_MOTLS, "data"),
        prevent_initial_call=True,
    )
    def _load_nn_from_csv(n_clicks, csv_path, selected, pool_motls):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not csv_path:
            return no_update, no_update, no_update, no_update, "Specify a CSV file path."

        pool_motls = pool_motls or {}
        motl_list = None
        selected_names = []
        is_multi = False

        if selected:
            if isinstance(selected, str):
                selected = [selected]
            motl_objs = [Motl(pd.DataFrame(pool_motls[m])) for m in selected if pool_motls.get(m)]
            if motl_objs:
                selected_names = [m for m in selected if pool_motls.get(m)]
                is_multi = len(selected_names) > 1
                # Single-motl NearestNeighbors stores motls=[motl, motl]
                if not is_multi:
                    motl_list = [motl_objs[0], motl_objs[0]]
                else:
                    motl_list = motl_objs

        try:
            nn_stats = NearestNeighbors.load(csv_path, motls=motl_list)
        except Exception as exc:
            return no_update, no_update, no_update, no_update, f"Load failed: {exc}"

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
        used_motls_store = {
            "names": selected_names,
            "is_multi": is_multi,
            "column_name": nn_stats.column_name,
        }

        n_rows = len(nn_stats.df)
        motl_note = f" ({len(selected_names)} motl(s) attached)" if selected_names else " (no motls attached)"
        status = f"Loaded {n_rows} rows{motl_note}."
        return xyz_graph, table_data, table_data, used_motls_store, status
