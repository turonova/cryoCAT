from cryocat.app.logger import dash_logger

import base64
import tempfile
import os
import numpy as np

from dash import html, dcc, ctx
from dash import Input, Output, State, callback, no_update
import pandas as pd
import dash_bootstrap_components as dbc
from cryocat import visplot
from cryocat.cryomotl import Motl
from cryocat.classutils import get_class_names_by_parent
from cryocat.app.globalvars import tomo_ids
from cryocat.app.apputils import get_print_out, save_output
from uuid import uuid4

# motl_types = [{"label": name, "value": name} for name in get_class_names_by_parent("Motl", "cryocat.cryomotl")]


def get_table_plot_component(prefix: str):
    return html.Div(
        children=[
            dbc.Col(
                children=[
                    dbc.Row(
                        dcc.Dropdown(
                            id=f"{prefix}-graph-options-dropdown",
                            multi=False,
                            placeholder="Select plot type",
                            options=["Histogram", "Line plot", "Scatter plot 1D", "Scatter plot 2D"],
                            style={
                                "width": "100%",
                                "padding": "0",  # reduce padding
                                "marginBottom": "0.5rem",
                            },
                        ),
                    ),
                    dbc.Row(
                        id=f"{prefix}-graph-options",
                        children=[
                            dbc.Row(html.Div("Graph options"), style={"marginBottom": "0.5rem"}),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id=f"{prefix}-plot-column-options-x-dropdown",
                                            multi=True,
                                            placeholder="Select data to plot on x axis",
                                            style={
                                                "width": "100%",
                                                "padding": "0",  # reduce padding
                                                "marginBottom": "0.5rem",
                                            },
                                        ),
                                        width=4,
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id=f"{prefix}-plot-color-palette-dropdown",
                                            multi=False,
                                            placeholder="Color palette",
                                            options=[
                                                {"label": s, "value": s}
                                                for s in ["Monet", "Viridis", "Cividis", "Plasma", "Jet", "Hot"]
                                            ],
                                            value="Monet",
                                            style={
                                                "width": "100%",
                                                "padding": "0",  # reduce padding
                                                "marginBottom": "0.5rem",
                                            },
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        dbc.Checkbox(
                                            id=f"{prefix}-plot-separately",
                                            label="Plot separately",
                                            value=False,  # unchecked
                                            inputStyle={"marginRight": "5px"},
                                            className="sidebar-checklist",
                                            labelStyle={"color": "var(--color12)"},
                                            disabled=True,
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        dbc.Checkbox(
                                            id=f"{prefix}-same-range",
                                            label="Same range for all",
                                            inputStyle={"marginRight": "5px"},
                                            className="sidebar-checklist",
                                            labelStyle={"color": "var(--color12)"},
                                            disabled=True,
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id=f"{prefix}-plot-grid-dropdown",
                                            multi=False,
                                            disabled=True,
                                            placeholder="Grid type",
                                            options=["Auto", "Column", "Row"],
                                            style={
                                                "width": "100%",
                                                "padding": "0",  # reduce padding
                                                "marginBottom": "0.5rem",
                                            },
                                        ),
                                        width=2,
                                    ),
                                ]
                            ),
                            dbc.Row(
                                id=f"{prefix}-scatter2D-row-options",
                                style={"display": "none"},
                                children=[
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id=f"{prefix}-plot-column-options-y-dropdown",
                                            multi=True,
                                            placeholder="Select data to plot on y axis",
                                            style={
                                                "width": "100%",
                                                "padding": "0",  # reduce padding
                                                "marginBottom": "0.5rem",
                                            },
                                        ),
                                        width=4,
                                    ),
                                ],
                            ),
                            dbc.Row(
                                id=f"{prefix}-histogram-row-options",
                                style={"display": "none"},
                                children=[
                                    dbc.Col(
                                        dbc.Input(
                                            id=f"{prefix}-histogram-bins-input",
                                            type="number",
                                            placeholder="Number of bins",
                                        ),
                                        width=4,
                                    ),
                                ],
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Button(
                                                "Plot",
                                                id=f"{prefix}-plot-graph-btn",
                                                color="light",
                                                style={"width": "100%"},
                                                n_clicks=0,
                                            ),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Button(
                                                "Clear plot(s)",
                                                id=f"{prefix}-clear-graph-btn",
                                                color="light",
                                                style={"width": "100%"},
                                                n_clicks=0,
                                                disabled=True,
                                            ),
                                        ],
                                        width=6,
                                    ),
                                ]
                            ),
                        ],
                        style={"display": "none"},
                    ),
                    html.Div(id=f"{prefix}-graph-area", children=[]),
                ],
                style={"width": "100%"},
            ),
        ]
    )


def register_table_plot_callbacks(prefix: str, connected_store_id):

    @callback(
        Output(f"{prefix}-graph-options", "style"),
        Output(f"{prefix}-plot-column-options-x-dropdown", "options"),
        Output(f"{prefix}-scatter2D-row-options", "style"),
        Output(f"{prefix}-histogram-row-options", "style"),
        Output(f"{prefix}-plot-column-options-y-dropdown", "options"),
        Input(f"{prefix}-graph-options-dropdown", "value"),
        State(connected_store_id, "data"),
        prevent_initial_call=True,
    )
    def generate_data_options(graph_type, data):

        if graph_type is None:
            return no_update

        x_axis_options = pd.DataFrame(data).columns
        y_axis_options = []

        if graph_type == "Histogram":
            histogram_options = {"display": "flex"}
            scatter_2D_options = {"display": "none"}
        elif graph_type == "Scatter plot 2D":
            histogram_options = {"display": "none"}
            scatter_2D_options = {"display": "flex"}
            y_axis_options = pd.DataFrame(data).columns
        else:
            histogram_options = {"display": "none"}
            scatter_2D_options = {"display": "none"}

        return {"display": "flex"}, x_axis_options, scatter_2D_options, histogram_options, y_axis_options

    @callback(
        Output(f"{prefix}-same-range", "disabled"),
        Output(f"{prefix}-plot-grid-dropdown", "disabled"),
        Input(f"{prefix}-plot-separately", "value"),
    )
    def toggle_separate_options(plot_separately):

        if plot_separately:
            return False, False
        else:
            return True, True

    @callback(
        Output(f"{prefix}-clear-graph-btn", "disabled"),
        Input(f"{prefix}-graph-area", "children"),
    )
    def toggle_clear_button(graph_area):
        if len(graph_area) == 0:
            return True
        else:
            return False

    @callback(
        Output(f"{prefix}-plot-separately", "disabled"),
        Input(f"{prefix}-plot-column-options-x-dropdown", "value"),
    )
    def toggle_plot_separately(x_values):

        n_selected = len(x_values) if x_values else 0

        if n_selected > 1:
            return False
        else:
            return True

    @callback(
        Output(f"{prefix}-graph-area", "children"),
        Input(f"{prefix}-plot-graph-btn", "n_clicks"),
        Input(f"{prefix}-clear-graph-btn", "n_clicks"),
        State(f"{prefix}-graph-options-dropdown", "value"),
        State(f"{prefix}-graph-area", "children"),
        State(connected_store_id, "data"),
        State(f"{prefix}-plot-column-options-x-dropdown", "value"),
        State(f"{prefix}-plot-column-options-y-dropdown", "value"),
        State(f"{prefix}-histogram-bins-input", "value"),
        State(f"{prefix}-plot-separately", "value"),
        State(f"{prefix}-same-range", "value"),
        State(f"{prefix}-plot-grid-dropdown", "value"),
        State(f"{prefix}-plot-color-palette-dropdown", "value"),
        prevent_initial_call=True,
    )
    def plot_graphs(
        plot_click,
        clear_click,
        graph_type,
        graph_area,
        data,
        x_values,
        y_values,
        h_bins,
        plot_separately,
        same_range,
        grid_spec,
        colorscale,
    ):

        trigger_id = ctx.triggered_id
        input_data = pd.DataFrame(data)

        if trigger_id == f"{prefix}-clear-graph-btn":
            return []
        elif trigger_id == f"{prefix}-plot-graph-btn":
            fig = None
            if graph_type == "Histogram":
                fig = visplot.plot_histogram(
                    input_data=input_data,
                    input_data_id=x_values,
                    bins=h_bins,
                    separate_graphs=plot_separately,
                    output_mode="count",
                    same_range_for_separate=same_range,
                    colors=colorscale,
                    opacity=None,
                    grid_spec=grid_spec,
                )
            elif graph_type == "Line plot":  # , "Scatter plot 1D", "Scatter plot 2D""
                fig = visplot.plot_line(
                    input_data=input_data,
                    input_data_id=x_values,
                    separate_graphs=plot_separately,
                    same_range_for_separate=same_range,
                    colors=colorscale,
                    opacity=None,
                    grid_spec=grid_spec,
                )
            elif graph_type == "Scatter plot 1D":  # "Scatter plot 2D""
                fig = visplot.plot_scatter2D(
                    input_data=input_data,
                    input_data_id=x_values,
                    separate_graphs=plot_separately,
                    same_range_for_separate=same_range,
                    colors=colorscale,
                    opacity=None,
                    grid_spec=grid_spec,
                )
            elif graph_type == "Scatter plot 2D":
                fig = visplot.plot_scatter2D(
                    input_data=input_data,
                    input_data_id=x_values,
                    second_axis_data=input_data,
                    second_axis_id=y_values,
                    separate_graphs=plot_separately,
                    same_range_for_separate=same_range,
                    colors=colorscale,
                    opacity=None,
                    grid_spec=grid_spec,
                )

            if fig is not None:
                return graph_area + [dcc.Graph(id=str(uuid4()), figure=fig)]

        return no_update
