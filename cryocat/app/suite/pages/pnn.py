"""Nearest-neighbor analysis tool.

Computation is decoupled from plotting:

* **Compute NN analysis** -- builds the per-pair table from the suite-pool
  motl(s) using ``NearestNeighbors`` with the auto-generated parameter form,
  optionally adds angular-distance columns, and renders the xyz panels.
* **Plot orientational distribution** / **Plot polar NN distances** -- each
  reads the already-computed coordinates from ``nn-coords-store`` and renders
  into its own graph area, so the user can tweak plot params and re-render
  without recomputing NN.

Each plot block has a checkbox that toggles its parameter form's visibility;
the plot button next to the checkbox triggers the render.

Contract: exposes ``layout`` and ``register_callbacks(app)``.
"""

import numpy as np
import pandas as pd

import dash
from dash import html, dcc, Input, Output, State, ALL, no_update
import dash_bootstrap_components as dbc

from cryocat.core.cryomotl import Motl
from cryocat.analysis.nnana import NearestNeighbors
from cryocat.analysis import visplot
from cryocat.app.apputils import generate_kwargs
from cryocat.app.formgen import build_form
from cryocat.app.components.motlsource import get_motl_source, register_motl_source_callbacks
from cryocat.app.components.motlsink import get_send_to_editor_button, register_send_to_editor_callbacks
from cryocat.app.components.tableview import get_table_component, register_table_callbacks
from cryocat.app.components.tableplot import register_table_plot_callbacks
from cryocat.app.components.logpanel import get_log_panel, register_log_panel_callbacks


# Params each plot helper computes itself (coordinates/distances come from
# the stored compute result; titles/output_path stay GUI-managed) -- exclude
# them from the auto-form.
_ORIENT_EXCLUDE = ["coordinates", "graph_title", "output_path"]
_POLAR_EXCLUDE = ["coordinates", "distances", "graph_title", "output_path"]


# ── Layout ──────────────────────────────────────────────────────────────────────

def _plot_block(toggle_id, toggle_label, wrap_id, plot_btn_id, form_children):
    """Checkbox + (hidden) form + plot button row."""
    return [
        dbc.Checkbox(
            id=toggle_id, label=toggle_label, value=False,
            style={"marginBottom": "0.4rem"},
        ),
        html.Div(
            [
                html.Div(form_children),
                dbc.Button(
                    f"Plot",
                    id=plot_btn_id, color="primary", size="sm",
                    style={"width": "100%", "marginTop": "0.4rem"},
                ),
            ],
            id=wrap_id,
            style={"display": "none"},
        ),
    ]


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
                            [
                                *_plot_block(
                                    toggle_id="nn-orient-toggle",
                                    toggle_label="Plot orientational distribution",
                                    wrap_id="nn-orient-form-wrap",
                                    plot_btn_id="nn-orient-plot-btn",
                                    form_children=build_form(
                                        visplot.plot_orientational_distribution,
                                        id_type="nn-forms-params",
                                        id_extra={"cls_name": "nn-orient"},
                                        exclude=_ORIENT_EXCLUDE,
                                    ),
                                ),
                                html.Hr(style={"margin": "0.5rem 0"}),
                                *_plot_block(
                                    toggle_id="nn-polar-toggle",
                                    toggle_label="Plot polar NN distances",
                                    wrap_id="nn-polar-form-wrap",
                                    plot_btn_id="nn-polar-plot-btn",
                                    form_children=build_form(
                                        visplot.plot_polar_nn_distances,
                                        id_type="nn-forms-params",
                                        id_extra={"cls_name": "nn-polar"},
                                        exclude=_POLAR_EXCLUDE,
                                    ),
                                ),
                                html.Div(
                                    id="nn-plot-status",
                                    style={
                                        "fontSize": "0.85rem",
                                        "color": "var(--color9)",
                                        "marginTop": "0.5rem",
                                        "wordBreak": "break-word",
                                    },
                                ),
                            ],
                            title="Plots",
                            item_id="nn-acc-plots",
                        ),
                        dbc.AccordionItem(
                            get_send_to_editor_button("nn"),
                            title="Send result to editor",
                            item_id="nn-acc-output",
                        ),
                    ],
                    always_open=True,
                    active_item=["nn-acc-input", "nn-acc-params", "nn-acc-plots"],
                ),
                html.Div(
                    dbc.Button(
                        "Show log",
                        id="nn-open-log-btn",
                        className="custom-radius-button",
                        style={"width": "100%"},
                    ),
                    style={"padding": "0.5rem", "marginTop": "auto"},
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
                html.H4("Nearest Neighbor Analysis", style={"marginBottom": "1rem"}),
                dcc.Store(id="nn-out-tabv-global-data-store"),
                get_table_component("nn-out-tabv"),
                html.Hr(style={"margin": "0.5rem 0"}),
                html.Div(id="nn-xyz-graph-area"),
                html.Div(id="nn-orient-graph-area"),
                html.Div(id="nn-polar-graph-area"),
            ],
            style={"padding": "0.5rem"},
        ),
        width=9,
        style={"margin": "0", "padding": "0"},
    )


layout = html.Div(
    [
        dcc.Store(id="nn-result"),
        # Cached compute result: normalized NN coords + per-pair distances.
        # Plot callbacks consume this without recomputing NN.
        dcc.Store(id="nn-coords-store"),
        dbc.Row([_sidebar(), _main()], className="g-0", style={"margin": "0", "padding": "0"}),
        *get_log_panel("nn-log"),
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


def _coords_from_store(store):
    """Extract normalized coords + optional nn_dist from the compute store."""
    if not store:
        return None, None
    coords = np.asarray(store.get("coords") or [])
    nn_dist = store.get("nn_dist")
    nn_dist = np.asarray(nn_dist) if nn_dist is not None else None
    return coords, nn_dist


# ── Callbacks ──────────────────────────────────────────────────────────────────

def register_callbacks(app):
    register_motl_source_callbacks(app, "nn", multi=True)
    register_send_to_editor_callbacks(app, "nn", "nn-result")
    register_table_callbacks(app, "nn-out-tabv", csv_only=True)
    register_table_plot_callbacks(
        app, "nn-out-tabv-table-plot", "nn-out-tabv-global-data-store", table_grid_id="nn-out-tabv-grid"
    )
    register_log_panel_callbacks(app, "nn-log")

    # Visibility for the three checkbox-gated form blocks.
    @app.callback(
        Output("nn-angular-form-wrap", "style"),
        Output("nn-orient-form-wrap", "style"),
        Output("nn-polar-form-wrap", "style"),
        Input("nn-angular-toggle", "value"),
        Input("nn-orient-toggle", "value"),
        Input("nn-polar-toggle", "value"),
    )
    def _toggle_forms(angular_on, orient_on, polar_on):
        show = {"display": "block"}
        hide = {"display": "none"}
        return (
            show if angular_on else hide,
            show if orient_on else hide,
            show if polar_on else hide,
        )

    # ── Compute NN: fills the table, the xyz panel, and the coords store. ────
    @app.callback(
        Output("nn-xyz-graph-area", "children"),
        Output("nn-out-tabv-global-data-store", "data"),
        Output("nn-stats-text", "children"),
        Output("nn-result", "data"),
        Output("nn-coords-store", "data"),
        Output("nn-orient-graph-area", "children", allow_duplicate=True),
        Output("nn-polar-graph-area", "children", allow_duplicate=True),
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
                    no_update, no_update, no_update, no_update)
        if isinstance(selected, str):
            selected = [selected]

        motls = [Motl(pd.DataFrame(pool_motls[m])) for m in selected if pool_motls.get(m)]
        if not motls:
            return (no_update, no_update, "The selected motls have no data.",
                    no_update, no_update, no_update, no_update)

        nn_kwargs = _kwargs_by_cls(param_ids, param_values, "nn-params")
        angular_kwargs = _kwargs_by_cls(param_ids, param_values, "nn-angular")

        try:
            nn_input = motls[0] if len(motls) == 1 else motls
            nn_stats = NearestNeighbors(nn_input, **nn_kwargs)
            normalized = nn_stats.get_normalized_coord(add_to_df=True)
        except Exception as exc:
            return (no_update, no_update, f"Error: {exc}",
                    no_update, no_update, no_update, no_update)

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

        coords_store = {
            "coords": np.asarray(normalized).tolist(),
            "nn_dist": (
                nn_stats.df["nn_dist"].tolist()
                if "nn_dist" in nn_stats.df.columns else None
            ),
        }

        # Re-running compute clears the dependent plot panels.
        return xyz_graph, table_data, " | ".join(status_bits), table_data, coords_store, [], []

    # ── Plot orientational distribution (uses the coords store). ──────────────
    @app.callback(
        Output("nn-orient-graph-area", "children", allow_duplicate=True),
        Output("nn-plot-status", "children", allow_duplicate=True),
        Input("nn-orient-plot-btn", "n_clicks"),
        State("nn-coords-store", "data"),
        State({"type": "nn-forms-params", "cls_name": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": "nn-forms-params", "cls_name": ALL, "param": ALL, "tag": ALL}, "id"),
        prevent_initial_call=True,
    )
    def plot_orient(n_clicks, store, param_values, param_ids):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        coords, _ = _coords_from_store(store)
        if coords is None or len(coords) == 0:
            return no_update, "Run Compute NN first."
        try:
            kwargs = _kwargs_by_cls(param_ids, param_values, "nn-orient")
            fig = visplot.plot_orientational_distribution(coords, **kwargs)
        except Exception as exc:
            return no_update, f"Orientational distribution failed: {exc}"
        return dcc.Graph(figure=fig), "Orientational distribution rendered."

    # ── Plot polar NN distances (needs nn_dist from a closest_dist run). ──────
    @app.callback(
        Output("nn-polar-graph-area", "children", allow_duplicate=True),
        Output("nn-plot-status", "children", allow_duplicate=True),
        Input("nn-polar-plot-btn", "n_clicks"),
        State("nn-coords-store", "data"),
        State({"type": "nn-forms-params", "cls_name": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": "nn-forms-params", "cls_name": ALL, "param": ALL, "tag": ALL}, "id"),
        prevent_initial_call=True,
    )
    def plot_polar(n_clicks, store, param_values, param_ids):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        coords, nn_dist = _coords_from_store(store)
        if coords is None or len(coords) == 0:
            return no_update, "Run Compute NN first."
        if nn_dist is None:
            return no_update, "Polar NN distances need nn_type='closest_dist'."
        try:
            kwargs = _kwargs_by_cls(param_ids, param_values, "nn-polar")
            fig = visplot.plot_polar_nn_distances(coords, nn_dist, **kwargs)
        except Exception as exc:
            return no_update, f"Polar NN distances failed: {exc}"
        return dcc.Graph(figure=fig), "Polar NN distances rendered."
