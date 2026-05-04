from cryocat.app.logger import dash_logger

import base64
import tempfile
import os
import numpy as np
import json

import dash
from dash import html, dcc, ctx, ALL
from dash import Input, Output, State, callback, no_update
import pandas as pd
import dash_bootstrap_components as dbc
from cryocat.analysis import visplot
from cryocat.core.cryomotl import Motl
from cryocat.utils.classutils import get_class_names_by_parent
from cryocat.app.globalvars import tomo_ids
from cryocat.app.apputils import get_print_out, save_output
from cryocat.app.layout.customel import LabeledDropdown, InlineLabeledDropdown, InlineInputForm

# motl_types = [{"label": name, "value": name} for name in get_class_names_by_parent("Motl", "cryocat.cryomotl")]

hist_norms = [
    {"label": "None", "value": ""},
    {"label": "Percent", "value": "percent"},
    {"label": "Probability", "value": "probability"},
    {"label": "Density", "value": "Density"},
    {"label": "Probability density", "value": "probability density"},
]
hist_types = ["Count", "Sum", "Avg", "Min", "Max"]


def get_table_plot_component(prefix: str):
    return html.Div(
        children=[
            dbc.Col(
                children=[
                    dbc.Row(
                        dbc.Col(
                            dcc.Dropdown(
                                id=f"{prefix}-graph-options-dropdown",
                                multi=False,
                                placeholder="Select plot type",
                                style={
                                    "width": "99%",
                                    "padding": "0",  # reduce padding
                                    "marginBottom": "0.5rem",
                                },
                            ),
                            width=12,
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
                                            placeholder="Data to plot on x axis",
                                            style={
                                                "width": "100%",
                                                "padding": "0",  # reduce padding
                                                "marginBottom": "0.5rem",
                                            },
                                        ),
                                        width=3,
                                    ),
                                    dbc.Col(
                                        InlineLabeledDropdown(
                                            id_=f"{prefix}-plot-color-palette-dropdown",
                                            label="Color scheme",
                                            multi=False,
                                            placeholder="Color palette",
                                            options=[
                                                {"label": s, "value": s}
                                                for s in ["Monet", "Viridis", "Cividis", "Plasma", "Jet", "Hot"]
                                            ],
                                            value="Monet",
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        dbc.Checkbox(
                                            id=f"{prefix}-plot-separately",
                                            label="Plot each graph separately",
                                            value=False,  # unchecked
                                            inputStyle={"marginRight": "5px"},
                                            className="sidebar-checklist",
                                            labelStyle={"color": "var(--color9)"},
                                            disabled=True,
                                        ),
                                        width=2,
                                        className="d-flex justify-content-end",
                                    ),
                                    dbc.Col(
                                        dbc.Checkbox(
                                            id=f"{prefix}-same-range",
                                            label="Same range for all graphs",
                                            inputStyle={"marginRight": "5px"},
                                            className="sidebar-checklist",
                                            labelStyle={"color": "var(--color9)"},
                                            disabled=True,
                                        ),
                                        width=2,
                                        className="d-flex justify-content-end",
                                    ),
                                    dbc.Col(
                                        dbc.Checkbox(
                                            id=f"{prefix}-histogram2D-same-scale",
                                            label="Same scale",
                                            value=False,  # unchecked
                                            inputStyle={"marginRight": "5px"},
                                            className="sidebar-checklist",
                                            labelStyle={"color": "var(--color9)"},
                                            disabled=True,
                                        ),
                                        width=1,
                                        className="d-flex justify-content-end",
                                    ),
                                    dbc.Col(
                                        InlineLabeledDropdown(
                                            id_=f"{prefix}-plot-grid-dropdown",
                                            label="Grid type",
                                            multi=False,
                                            disabled=True,
                                            placeholder="Grid type",
                                            options=["Auto", "Column", "Row"],
                                            value="Auto",
                                        ),
                                        width=2,
                                    ),
                                ],
                                align="center",
                            ),
                            dbc.Row(
                                id=f"{prefix}-scatter2D-row-options",
                                style={"display": "none"},
                                children=[
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id=f"{prefix}-plot-column-options-y-dropdown",
                                            multi=True,
                                            placeholder="Data to plot on y axis",
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
                                        InlineInputForm(
                                            id_=f"{prefix}-histogram-bins-input",
                                            label="Number of bins",
                                            type="number",
                                            placeholder="Number of bins",
                                            value=30,
                                        ),
                                        width=4,
                                    ),
                                    dbc.Col(
                                        InlineLabeledDropdown(
                                            id_=f"{prefix}-histogram-type-input",
                                            label="Type",
                                            multi=False,
                                            placeholder="Chose histogram type",
                                            options=hist_types,
                                            value="Count",
                                        ),
                                        width=4,
                                    ),
                                    dbc.Col(
                                        InlineLabeledDropdown(
                                            id_=f"{prefix}-histogram-norm-input",
                                            label="Normalization",
                                            multi=False,
                                            placeholder="Chose normalization",
                                            options=hist_norms,
                                            value="",
                                        ),
                                        width=4,
                                    ),
                                ],
                            ),
                            dbc.Row(
                                id=f"{prefix}-histogram2D-row-options",
                                style={"display": "none"},
                                align="center",
                                children=[
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id=f"{prefix}-histogram2D-column-options-y-dropdown",
                                            multi=True,
                                            placeholder="Data to plot on y axis",
                                            style={
                                                "width": "100%",
                                                "padding": "0",  # reduce padding
                                                "marginBottom": "0.5rem",
                                            },
                                        ),
                                        width=3,
                                    ),
                                    dbc.Col(
                                        InlineInputForm(
                                            id_=f"{prefix}-histogram2D-binsx-input",
                                            label="Bins x",
                                            type="number",
                                            value=30,
                                            placeholder="Number of bins in x",
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        InlineInputForm(
                                            id_=f"{prefix}-histogram2D-binsy-input",
                                            label="Bins y",
                                            value=30,
                                            type="number",
                                            placeholder="Number of bins in y",
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        InlineLabeledDropdown(
                                            id_=f"{prefix}-histogram2D-type-input",
                                            label="Type",
                                            multi=False,
                                            placeholder="Histogram type",
                                            options=hist_types,
                                            value="Count",
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        InlineLabeledDropdown(
                                            id_=f"{prefix}-histogram2D-norm-input",
                                            label="Normalization",
                                            value="",
                                            multi=False,
                                            placeholder="Normalization",
                                            options=hist_norms,
                                        ),
                                        width=3,
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
                            dbc.Row(
                                dbc.Col(
                                    dbc.RadioItems(
                                        id=f"{prefix}-selection-mode",
                                        options=[
                                            {"label": "Replace selection", "value": "replace"},
                                            {"label": "Add to selection", "value": "add"},
                                            {"label": "Subtract from selection", "value": "subtract"},
                                        ],
                                        value="replace",
                                        inline=True,
                                        className="sidebar-checklist",
                                        labelStyle={"color": "var(--color9)", "marginRight": "1rem"},
                                    ),
                                    width=12,
                                ),
                                style={"marginTop": "0.5rem"},
                            ),
                        ],
                        style={"display": "none"},
                    ),
                    html.Div(id=f"{prefix}-graph-area", children=[]),
                    dcc.Store(id=f"{prefix}-graph-meta-store", data={}),
                    dcc.Store(id=f"{prefix}-graph-counter", data=0),
                    dbc.Modal(
                        [
                            dbc.ModalHeader(dbc.ModalTitle("Wrong inputs")),
                            dbc.ModalBody(id=f"{prefix}-modal-text-area", children=[]),
                            dbc.ModalFooter(
                                dbc.Button("Close", id=f"{prefix}-modal-main-close", className="ms-auto", n_clicks=0)
                            ),
                        ],
                        id=f"{prefix}-modal-main",
                        is_open=False,
                    ),
                ],
                style={"width": "100%"},
            ),
        ]
    )


def register_table_plot_callbacks(prefix: str, connected_store_id, special_graphs=None, table_grid_id=None):

    graph_options = [
        "Line plot",
        "Scatter plot 1D",
        "Scatter plot 2D",
        "Histogram",
        "Histogram 2D",
        "Kernel density estimation",
    ]

    if special_graphs is not None:
        graph_options = graph_options + special_graphs

    @callback(
        Output(f"{prefix}-graph-options-dropdown", "options"),
        Input("url", "pathname"),
    )
    def load_graph_options(_):

        return graph_options

    @callback(
        Output(f"{prefix}-graph-options", "style"),
        Output(f"{prefix}-plot-column-options-x-dropdown", "options"),
        Output(f"{prefix}-scatter2D-row-options", "style"),
        Output(f"{prefix}-histogram-row-options", "style"),
        Output(f"{prefix}-histogram2D-row-options", "style"),
        Output(f"{prefix}-plot-column-options-y-dropdown", "options"),
        Output(f"{prefix}-histogram2D-column-options-y-dropdown", "options"),
        Input(f"{prefix}-graph-options-dropdown", "value"),
        State(connected_store_id, "data"),
        prevent_initial_call=True,
    )
    def generate_data_options(graph_type, data):

        if graph_type is None:
            return no_update

        x_axis_options = pd.DataFrame(data).columns
        y_axis_options = []

        def get_spherical_columns(prefix):
            cols = [prefix + "_x", prefix + "_y", prefix + "_z"]

            # check if all are present
            if all(col in x_axis_options for col in cols):
                return {"label": f"{prefix}_x, {prefix}_y, {prefix}_z", "value": json.dumps(cols)}
            else:
                return None

        if graph_type == "Histogram":
            histogram_options = {"display": "flex"}
            histogram2D_options = {"display": "none"}
            scatter_2D_options = {"display": "none"}
        elif graph_type in ["Histogram 2D", "Kernel density estimation", "Spherical histogram"]:
            histogram_options = {"display": "none"}
            histogram2D_options = {"display": "flex"}
            scatter_2D_options = {"display": "none"}
            y_axis_options = pd.DataFrame(data).columns
            if graph_type == "Spherical histogram":
                dropdown_options = []
                for prefix in ["twist_so", "twist", "norm_nn"]:
                    opt = get_spherical_columns(prefix)
                    if opt:
                        dropdown_options.append(opt)
                x_axis_options = dropdown_options
                y_axis_options = ["None - computed automatically"]
        elif graph_type == "Scatter plot 2D":
            histogram_options = {"display": "none"}
            histogram2D_options = {"display": "none"}
            scatter_2D_options = {"display": "flex"}
            y_axis_options = pd.DataFrame(data).columns
        else:
            histogram_options = {"display": "none"}
            histogram2D_options = {"display": "none"}
            scatter_2D_options = {"display": "none"}

        return (
            {"display": "flex"},
            x_axis_options,
            scatter_2D_options,
            histogram_options,
            histogram2D_options,
            y_axis_options,
            y_axis_options,
        )

    @callback(
        Output(f"{prefix}-same-range", "disabled"),
        Output(f"{prefix}-plot-grid-dropdown", "disabled"),
        Output(f"{prefix}-histogram2D-same-scale", "disabled"),
        Input(f"{prefix}-plot-separately", "value"),
    )
    def toggle_separate_options(plot_separately):

        if plot_separately:
            return False, False, False
        else:
            return True, True, True

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
        State(f"{prefix}-graph-options-dropdown", "value"),
    )
    def toggle_plot_separately(x_values, graph_type):

        n_selected = len(x_values) if x_values else 0

        if n_selected > 1:
            return False
        else:
            return True

    @callback(
        Output(f"{prefix}-graph-area", "children"),
        Output(f"{prefix}-modal-text-area", "children"),
        Output(f"{prefix}-graph-meta-store", "data"),
        Output(f"{prefix}-graph-counter", "data"),
        Input(f"{prefix}-plot-graph-btn", "n_clicks"),
        Input(f"{prefix}-clear-graph-btn", "n_clicks"),
        State(f"{prefix}-graph-options-dropdown", "value"),
        State(f"{prefix}-graph-area", "children"),
        State(connected_store_id, "data"),
        State(f"{prefix}-plot-column-options-x-dropdown", "value"),
        State(f"{prefix}-plot-column-options-y-dropdown", "value"),
        State(f"{prefix}-histogram-bins-input", "value"),
        State(f"{prefix}-histogram-type-input", "value"),
        State(f"{prefix}-histogram-norm-input", "value"),
        State(f"{prefix}-histogram2D-binsx-input", "value"),
        State(f"{prefix}-histogram2D-binsy-input", "value"),
        State(f"{prefix}-histogram2D-type-input", "value"),
        State(f"{prefix}-histogram2D-norm-input", "value"),
        State(f"{prefix}-histogram2D-same-scale", "value"),
        State(f"{prefix}-histogram2D-column-options-y-dropdown", "value"),
        State(f"{prefix}-plot-separately", "value"),
        State(f"{prefix}-same-range", "value"),
        State(f"{prefix}-plot-grid-dropdown", "value"),
        State(f"{prefix}-plot-color-palette-dropdown", "value"),
        State(f"{prefix}-graph-meta-store", "data"),
        State(f"{prefix}-graph-counter", "data"),
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
        h_type,
        h_norm,
        h2D_binsx,
        h2D_binsy,
        h2D_type,
        h2D_norm,
        h2D_same_scale,
        h2D_column_options,
        plot_separately,
        same_range,
        grid_spec,
        colorscale,
        graph_meta,
        graph_counter,
    ):

        trigger_id = ctx.triggered_id
        input_data = pd.DataFrame(data)

        if trigger_id == f"{prefix}-clear-graph-btn":
            return [], [], {}, 0
        elif trigger_id == f"{prefix}-plot-graph-btn":

            x_selected = len(x_values) if x_values else 0

            if x_selected == 0:
                return (
                    graph_area,
                    f"Number of columns for X axis is {x_selected}. At least 1 column needs to be selected.",
                    no_update,
                    no_update,
                )

            fig = None
            if graph_type == "Histogram":

                fig = visplot.plot_histogram(
                    input_data=input_data,
                    input_data_id=x_values,
                    bins=h_bins,
                    separate_graphs=plot_separately,
                    hist_type=h_type.lower(),
                    hist_norm=h_norm.lower(),
                    same_range_for_separate=same_range,
                    colors=colorscale,
                    opacity=None,
                    grid_spec=grid_spec,
                )
            elif graph_type == "Histogram 2D":

                fig = visplot.plot_histogram2D(
                    input_data=input_data,
                    input_data_id=x_values,
                    separate_graphs=plot_separately,
                    second_axis_data=input_data,
                    second_axis_id=h2D_column_options,
                    hist_type=h2D_type.lower(),
                    hist_norm=h2D_norm.lower(),
                    nbinsx=h2D_binsx,
                    nbinsy=h2D_binsy,
                    same_range_for_separate=same_range,
                    colors=colorscale,
                    opacity=None,
                    grid_spec=grid_spec,
                    same_scale=h2D_same_scale,
                )
            elif graph_type == "Kernel density estimation":
                fig = visplot.plot_kde(
                    input_data=input_data,
                    input_data_id=x_values,
                    second_axis_data=input_data,
                    second_axis_id=h2D_column_options,
                    nbinsx=h2D_binsx,
                    nbinsy=h2D_binsy,
                    hist_type=h2D_type.lower(),
                    hist_norm=h2D_norm.lower(),
                    colors=colorscale,
                    opacity=None,
                    grid_spec=grid_spec,
                    same_range_for_separate=same_range,
                    same_scale=h2D_same_scale,
                )
            elif graph_type == "Spherical histogram":
                all_ids = []
                for s in x_values:
                    parsed = json.loads(s)  # e.g. ["a","b"]
                    all_ids.extend(parsed)
                fig = visplot.plot_spherical_density_hist2d(
                    input_data=input_data,
                    input_data_id=all_ids,
                    nbinsx=h2D_binsx,
                    nbinsy=h2D_binsy,
                    x_range=None,
                    y_range=None,
                    hist_type=h2D_type.lower(),
                    hist_norm=h2D_norm.lower(),
                    normalize_coord=True,
                    colors=colorscale,
                    same_scale=h2D_same_scale,
                    same_range_for_separate=same_range,
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

                y_selected = len(y_values) if y_values else 0

                if y_selected != x_selected and y_selected != 1:
                    return (
                        graph_area,
                        f"Number of columns for X axis is {x_selected} "
                        f"and number of columns for Y axis is {y_selected}."
                        "\n\nThe number of selected columns for Y axis has to match the number of "
                        "columns for X axis. Alternatively, only one column can be selected to "
                        "be used as Y axis for all columns in X.",
                        no_update,
                        no_update,
                    )

                if y_selected == 0 or x_selected == 0:
                    return (
                        graph_area,
                        f"Number of columns for X axis is {x_selected} "
                        f"and number of columns for Y axis is {y_selected}."
                        "\n\nAt least one column has to be selected for both X and Y axis.",
                        no_update,
                        no_update,
                    )

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
                # Plotly only fires selectedData for traces that have markers.
                # Add tiny invisible markers to line-only traces so selection works.
                for trace in fig.data:
                    if getattr(trace, "mode", None) == "lines":
                        trace.update(mode="lines+markers", marker=dict(size=6, opacity=0.01))
                fig.update_layout(dragmode="select")
                graph_meta = graph_meta or {}
                graph_meta[str(graph_counter)] = {"type": graph_type, "x_cols": x_values}
                new_graph = dcc.Graph(
                    id={"type": f"{prefix}-graph", "index": graph_counter},
                    figure=fig,
                )
                return graph_area + [new_graph], [], graph_meta, graph_counter + 1

        return no_update, no_update, no_update, no_update

    @callback(
        Output(f"{prefix}-modal-main", "is_open"),
        Input(f"{prefix}-modal-main-close", "n_clicks"),
        Input(f"{prefix}-modal-text-area", "children"),
    )
    def close_warning_window(_, modal_content):

        trigger_id = ctx.triggered_id

        if trigger_id == f"{prefix}-modal-text-area":
            if len(modal_content) > 0:
                return True
            else:
                return False
        elif trigger_id == f"{prefix}-modal-main-close":
            return False

        return False

    if table_grid_id is not None:

        @callback(
            Output(table_grid_id, "selectedRows"),
            Input({"type": f"{prefix}-graph", "index": ALL}, "clickData"),
            Input({"type": f"{prefix}-graph", "index": ALL}, "selectedData"),
            State(f"{prefix}-graph-meta-store", "data"),
            State(table_grid_id, "rowData"),
            State(f"{prefix}-selection-mode", "value"),
            State(table_grid_id, "selectedRows"),
            prevent_initial_call=True,
        )
        def sync_graph_selection(_click_list, _sel_list, graph_meta, row_data, sel_mode, current_selected):
            triggered = ctx.triggered_id
            if not isinstance(triggered, dict) or not row_data or not graph_meta:
                raise dash.exceptions.PreventUpdate

            data_value = ctx.triggered[0]["value"] if ctx.triggered else None
            if not data_value or not data_value.get("points"):
                raise dash.exceptions.PreventUpdate

            graph_idx = triggered.get("index")
            meta_entry = graph_meta.get(str(graph_idx))
            if meta_entry is None:
                raise dash.exceptions.PreventUpdate

            n = len(row_data)
            points = data_value["points"]

            if meta_entry["type"] == "Histogram":
                indices = []
                for p in points:
                    indices.extend(p.get("pointNumbers", []))
            else:
                indices = [p["pointIndex"] for p in points if "pointIndex" in p]

            new_rows = [row_data[i] for i in indices if 0 <= i < n]
            if not new_rows and sel_mode != "subtract":
                raise dash.exceptions.PreventUpdate

            if sel_mode == "replace":
                return new_rows

            current = current_selected or []

            if sel_mode == "add":
                existing_keys = {json.dumps(r, sort_keys=True) for r in current}
                merged = list(current)
                for r in new_rows:
                    if json.dumps(r, sort_keys=True) not in existing_keys:
                        merged.append(r)
                return merged

            if sel_mode == "subtract":
                remove_keys = {json.dumps(r, sort_keys=True) for r in new_rows}
                return [r for r in current if json.dumps(r, sort_keys=True) not in remove_keys]

            return new_rows
