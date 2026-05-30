"""Nearest-neighbor analysis tool.

Computation is decoupled from clustering:

* **Compute NN analysis** -- builds the per-pair table from the suite-pool
  motl(s) using ``NearestNeighbors`` with the auto-generated parameter form,
  optionally adds angular-distance columns, and renders the xyz panels.
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
from cryocat.app.apputils import generate_kwargs
from cryocat.app.formgen import build_form
from cryocat.app.components.motlsource import get_motl_source, register_motl_source_callbacks
from cryocat.app.components.tableview import get_table_component, register_table_callbacks
from cryocat.app.components.tableplot import register_table_plot_callbacks
from cryocat.app.components.tablecluster import register_table_cluster_callbacks



_MOTL_COL_OPTIONS = [{"label": c, "value": c} for c in Motl.motl_columns]


# ── Layout ──────────────────────────────────────────────────────────────────────

def _create_motl_sidebar_content():
    """Sidebar content for the 'Create motl from selection' accordion item."""
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


def _sidebar():
    return dbc.Col(
        html.Div(
            [
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            get_motl_source("nn", multi=True),
                            title="Input motls",
                            item_id="nn-acc-input",
                        ),
                        dbc.AccordionItem(
                            [
                                html.Div(
                                    build_form(
                                        NearestNeighbors,
                                        id_type="nn-forms-params",
                                        id_extra={"cls_name": "nn-params"},
                                        exclude=["input_data"],
                                    ),
                                ),
                                dbc.Checkbox(
                                    id="nn-angular-toggle",
                                    label="Compute angular distances",
                                    value=False,
                                    style={"marginTop": "0.5rem", "marginBottom": "0.4rem"},
                                ),
                                html.Div(
                                    build_form(
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
                            title="NN parameters",
                            item_id="nn-acc-params",
                        ),
                        dbc.AccordionItem(
                            _create_motl_sidebar_content(),
                            title="Create motl from selection",
                            item_id="nn-acc-create",
                        ),
                    ],
                    always_open=True,
                    active_item=["nn-acc-input", "nn-acc-params"],
                ),
            ],
            className="sidebar",
            style={
                "padding": "0.5rem",
                "overflowY": "auto",
                "height": "100vh",
                "display": "flex",
                "flexDirection": "column",
            },
        ),
        width=3,
        style={"margin": "0", "padding": "0", "height": "100vh", "position": "sticky", "top": "0px"},
    )


def _main():
    return dbc.Col(
        html.Div(
            [
                dcc.Store(id="nn-out-tabv-global-data-store"),
                get_table_component("nn-out-tabv"),
                html.Hr(style={"margin": "0.5rem 0"}),
                html.Div(id="nn-xyz-graph-area"),
            ],
            style={"padding": "0.5rem"},
        ),
        width=9,
        style={"margin": "0", "padding": "0"},
    )


layout = html.Div(
    [
        dcc.Store(id="nn-result"),
        # Ordered list of pool motl-ids used in the last NN run, plus is_multi flag.
        dcc.Store(id="nn-used-motls-store"),
        dbc.Row([_sidebar(), _main()], className="g-0", style={"margin": "0", "padding": "0"}),
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
    register_table_callbacks(app, "nn-out-tabv", csv_only=True)
    register_table_plot_callbacks(
        app, "nn-out-tabv-table-plot", "nn-out-tabv-global-data-store",
        special_graphs=["Orientational distribution", "Polar NN distances"],
        table_grid_id="nn-out-tabv-grid",
    )
    register_table_cluster_callbacks(
        app, "nn-out-tabv-table-cluster", "nn-out-tabv-global-data-store",
        table_grid_id="nn-out-tabv-grid",
    )
    @app.callback(
        Output("nn-angular-form-wrap", "style"),
        Input("nn-angular-toggle", "value"),
    )
    def _toggle_angular_form(angular_on):
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
        State("pool-motls", "data"),
        State({"type": "nn-forms-params", "cls_name": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": "nn-forms-params", "cls_name": ALL, "param": ALL, "tag": ALL}, "id"),
        State("nn-angular-toggle", "value"),
        prevent_initial_call=True,
    )
    def compute_nn(n_clicks, selected, pool_motls, param_values, param_ids, angular_on):
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
                f"Mean distance: {nn_stats.df['nn_dist'].mean():.3f}; "
                f"Median distance: {nn_stats.df['nn_dist'].median():.3f}"
            )
        else:
            status_bits.append(f"NN analysis complete - {len(nn_stats.df)} neighbor rows.")

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

        used_motls_store = {"names": selected, "is_multi": len(selected) > 1}

        return (
            xyz_graph, table_data, " | ".join(status_bits), table_data,
            used_motls_store,
        )

    # ── Create motl from selection (sidebar) ────────────────────────────────────

    @app.callback(
        Output("nn-sel-motl-checklist", "options"),
        Output("nn-sel-motl-checklist", "value"),
        Output("nn-sel-motl-id-type-wrap", "style"),
        Output("nn-sel-motl-info", "children"),
        Input("nn-used-motls-store", "data"),
        State("pool-registry", "data"),
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
        Output("pool-registry", "data", allow_duplicate=True),
        Output("pool-motls", "data", allow_duplicate=True),
        Output("pool-next-id", "data", allow_duplicate=True),
        Input("nn-sel-motl-save-btn", "n_clicks"),
        Input("nn-sel-motl-send-btn", "n_clicks"),
        State("nn-out-tabv-grid", "selectedRows"),
        State("nn-sel-motl-checklist", "value"),
        State("nn-used-motls-store", "data"),
        State("nn-sel-motl-id-type", "value"),
        State("nn-sel-motl-id-col", "value"),
        State("nn-sel-motl-save-path", "value"),
        State("nn-sel-motl-editor-label", "value"),
        State("pool-motls", "data"),
        State("pool-registry", "data"),
        State("pool-next-id", "data"),
        prevent_initial_call=True,
    )
    def _build_and_act(
        _save_click, _send_click,
        selected_rows, checked_motls, used_motls, id_type, id_col,
        save_path, editor_label, pool_motls, registry, next_id,
    ):
        trigger = ctx.triggered_id
        if not selected_rows:
            return "No rows selected in the table.", no_update, no_update, no_update
        if not checked_motls:
            return "No motls checked.", no_update, no_update, no_update
        if not used_motls:
            return "Run NN analysis first.", no_update, no_update, no_update

        all_names = used_motls.get("names", [])
        is_multi = used_motls.get("is_multi", False)
        pool_motls = pool_motls or {}
        sel_df = pd.DataFrame(selected_rows)

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
            if len(subset) == 0:
                continue
            if id_col and len(checked_motls) > 1:
                subset[id_col] = float(i)
            parts.append(subset)

        if not parts:
            return (
                "No particles matched the selection. "
                "Make sure rows are selected and the motl IDs align.",
                no_update, no_update, no_update,
            )

        merged_df = pd.concat(parts, ignore_index=True)

        if trigger == "nn-sel-motl-save-btn":
            if not save_path:
                return "Specify an output file path.", no_update, no_update, no_update
            try:
                Motl(merged_df).write_out(save_path)
            except Exception as exc:
                return f"Save failed: {exc}", no_update, no_update, no_update
            return (
                f"Saved {len(merged_df)} particles to {save_path}.",
                no_update, no_update, no_update,
            )

        if trigger == "nn-sel-motl-send-btn":
            registry = dict(registry or {})
            pool_out = dict(pool_motls)
            next_id = next_id or 0
            new_id = f"motl-{next_id}"
            display_label = editor_label or f"Motl {next_id + 1}"
            registry[new_id] = {
                "label": display_label, "type": "emmotl",
                "n_rows": len(merged_df), "active": True,
            }
            pool_out[new_id] = merged_df.to_dict("records")
            return (
                f"Sent '{display_label}' ({len(merged_df)} particles) to the editor.",
                registry, pool_out, next_id + 1,
            )

        return no_update, no_update, no_update, no_update
