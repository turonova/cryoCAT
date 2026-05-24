"""Nearest-neighbor analysis tool — ported from the Tango NN tab.

Reads one or more motls from the suite pool via a multi-select picker
(``NearestNeighbors`` combines several motls when given a list), runs the NN
analysis with parameters auto-derived from ``NearestNeighbors``' docstring, and
shows the per-neighbor stats table plus the normalized-coordinate scatter
panels.

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


# ── Layout ──────────────────────────────────────────────────────────────────────

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
                            get_send_to_editor_button("nn"),
                            title="Send result to editor",
                            item_id="nn-acc-output",
                        ),
                    ],
                    always_open=True,
                    active_item=["nn-acc-input", "nn-acc-params"],
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
                html.Div(id="nn-graph-area"),
                html.Hr(style={"margin": "0.5rem 0"}),
                dcc.Store(id="nn-out-tabv-global-data-store"),
                get_table_component("nn-out-tabv"),
            ],
            style={"padding": "0.5rem"},
        ),
        width=9,
        style={"margin": "0", "padding": "0"},
    )


layout = html.Div(
    [
        dcc.Store(id="nn-result"),
        dbc.Row([_sidebar(), _main()], className="g-0", style={"margin": "0", "padding": "0"}),
        *get_log_panel("nn-log"),
    ],
    style={"margin": "0", "padding": "0"},
)


# ── Callbacks ───────────────────────────────────────────────────────────────────

def register_callbacks(app):
    register_motl_source_callbacks(app, "nn", multi=True)
    register_send_to_editor_callbacks(app, "nn", "nn-result")
    register_table_callbacks(app, "nn-out-tabv", csv_only=True)
    register_table_plot_callbacks(
        app, "nn-out-tabv-table-plot", "nn-out-tabv-global-data-store", table_grid_id="nn-out-tabv-grid"
    )
    register_log_panel_callbacks(app, "nn-log")

    @app.callback(
        Output("nn-graph-area", "children"),
        Output("nn-out-tabv-global-data-store", "data"),
        Output("nn-stats-text", "children"),
        Output("nn-result", "data"),
        Input("nn-compute-btn", "n_clicks"),
        State("nn-motl-select", "value"),
        State("pool-motls", "data"),
        State({"type": "nn-forms-params", "cls_name": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": "nn-forms-params", "cls_name": ALL, "param": ALL, "tag": ALL}, "id"),
        prevent_initial_call=True,
    )
    def compute_nn(n_clicks, selected, pool_motls, param_values, param_ids):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        pool_motls = pool_motls or {}
        if not selected:
            return no_update, no_update, "Select at least one motl from the pool.", no_update
        if isinstance(selected, str):
            selected = [selected]

        motls = [Motl(pd.DataFrame(pool_motls[m])) for m in selected if pool_motls.get(m)]
        if not motls:
            return no_update, no_update, "The selected motls have no data.", no_update

        nn_kwargs = generate_kwargs(param_ids, param_values) if param_ids else {}

        try:
            # NearestNeighbors accepts a single motl or a list to be combined.
            nn_input = motls[0] if len(motls) == 1 else motls
            nn_stats = NearestNeighbors(nn_input, **nn_kwargs)
            _ = nn_stats.get_normalized_coord(add_to_df=True)
        except Exception as exc:
            return no_update, no_update, f"Error: {exc}", no_update

        if nn_kwargs.get("nn_type") == "closest_dist" and "nn_dist" in nn_stats.df:
            info = (
                f"Mean distance: {nn_stats.df['nn_dist'].mean():.3f}; "
                f"Median distance: {nn_stats.df['nn_dist'].median():.3f}"
            )
        else:
            info = f"NN analysis complete — {len(nn_stats.df)} neighbor rows."

        table_data = nn_stats.df.to_dict("records")

        norm = np.column_stack(
            (nn_stats.get_normalized_coord(), nn_stats.df["nn_subtomo_id"].values)
        )
        nn_df = pd.DataFrame(norm, columns=["x", "y", "z", "nn_subtomo_id"])
        fig = visplot.plot_scatter_xyz_panels(
            nn_df, coord_columns=["x", "y", "z"], hover_column_name="nn_subtomo_id"
        )

        return dcc.Graph(figure=fig), table_data, info, table_data
