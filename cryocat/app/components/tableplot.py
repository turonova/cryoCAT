from cryocat.app.logger import dash_logger

import base64
import tempfile
import os
import numpy as np
import json

import dash
from dash import html, dcc, ctx, ALL
from dash import Input, Output, State, no_update
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from cryocat.analysis import visplot
from cryocat.core.cryomotl import Motl
from cryocat.utils.classutils import get_class_names_by_parent
from cryocat.app import ids
from cryocat.app.apputils import save_output
from cryocat.app.components.customel import LabeledDropdown, InlineLabeledDropdown, InlineInputForm
from cryocat.app.components.graphsettings import styled_figure, get_graph_settings_button
from cryocat.app.formgen import make_dropdown, form_row
from cryocat.app import styles
from cryocat.app.components.paletteloader import _DISCRETE_PRESETS, _CONTINUOUS_PRESETS

hist_norms = [
    {"label": "None", "value": ""},
    {"label": "Percent", "value": "percent"},
    {"label": "Probability", "value": "probability"},
    {"label": "Density", "value": "Density"},
    {"label": "Probability density", "value": "probability density"},
]
hist_types = ["Count", "Sum", "Avg", "Min", "Max"]

_PALETTE_PRESETS = sorted({*_DISCRETE_PRESETS, *_CONTINUOUS_PRESETS})
_PALETTE_HINT = {
    "label": "─ or type any Plotly palette name ─",
    "value": "__hint__",
    "disabled": True,
}
# Plot types that aggregate data; clickData gives a bin position, not row indices.
_AGGREGATED_CLICK_IGNORE = {"Histogram 2D", "Kernel density estimation", "Spherical histogram"}


def get_table_plot_component(prefix: str):
    return html.Div(
        children=[
            # One-shot store: non-None initial value causes load_graph_options to
            # fire once at mount time without depending on any URL component.
            dcc.Store(id=f"{prefix}-options-init", data=True),
            dbc.Col(
                children=[
                    dbc.Row(
                        dbc.Col(
                            make_dropdown(
                                f"{prefix}-graph-options-dropdown",
                                [],
                                None,
                                placeholder="Select plot type",
                                style={
                                    "width": "99%",
                                    "padding": "0",
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
                                        make_dropdown(
                                            f"{prefix}-plot-column-options-x-dropdown",
                                            [],
                                            None,
                                            multi=True,
                                            placeholder="Data to plot on x axis",
                                            style={
                                                "width": "100%",
                                                "padding": "0",
                                                "marginBottom": "0.5rem",
                                            },
                                        ),
                                        width=3,
                                    ),
                                    dbc.Col(
                                        html.Div([
                                            InlineLabeledDropdown(
                                                id_=f"{prefix}-plot-color-palette-dropdown",
                                                label="Color scheme",
                                                multi=False,
                                                placeholder="Color palette",
                                                options=[
                                                    {"label": p, "value": p}
                                                    for p in _PALETTE_PRESETS
                                                ] + [_PALETTE_HINT],
                                                value="StarryNight",
                                            ),
                                            html.Div(
                                                id=f"{prefix}-palette-error",
                                                style=styles.HINT_SM,
                                            ),
                                        ]),
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
                                        make_dropdown(
                                            f"{prefix}-plot-column-options-y-dropdown",
                                            [],
                                            None,
                                            multi=True,
                                            placeholder="Data to plot on y axis",
                                            style={
                                                "width": "100%",
                                                "padding": "0",
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
                                        make_dropdown(
                                            f"{prefix}-histogram2D-column-options-y-dropdown",
                                            [],
                                            None,
                                            multi=True,
                                            placeholder="Data to plot on y axis",
                                            style={
                                                "width": "100%",
                                                "padding": "0",
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
                                id=f"{prefix}-orbd-row-options",
                                style={"display": "none"},
                                align="center",
                                children=[
                                    dbc.Col(
                                        form_row(
                                            "Binning mode",
                                            make_dropdown(
                                                f"{prefix}-orbd-binmode-dropdown",
                                                [
                                                    {"label": "Angular sampling (°)", "value": "cone_sampling"},
                                                    {"label": "Number of bins", "value": "n_bins"},
                                                ],
                                                "cone_sampling",
                                                clearable=False,
                                            ),
                                            "How to specify bin resolution: by angular sampling size in degrees "
                                            "or by an explicit bin count.",
                                            label_id=f"{prefix}-lbl-binning-mode",
                                        ),
                                        width=4,
                                    ),
                                    dbc.Col(
                                        form_row(
                                            "Value",
                                            dbc.Input(
                                                id=f"{prefix}-orbd-value-input",
                                                type="number",
                                                value=5.0,
                                                min=0.0,
                                                max=20000,
                                                step=0.5,
                                            ),
                                            "Angular sampling in degrees (1–30) or number of bins (100–20 000).",
                                            label_id=f"{prefix}-lbl-orbd-value",
                                        ),
                                        width=4,
                                    ),
                                    dbc.Col(
                                        form_row(
                                            "Height scale",
                                            dbc.Input(
                                                id=f"{prefix}-orbd-height-scale-input",
                                                type="number",
                                                value=0.3,
                                                min=0.0,
                                                max=1.0,
                                                step=0.05,
                                            ),
                                            "Maximum bar height as a fraction of the sphere radius (0–1).",
                                            label_id=f"{prefix}-lbl-height-scale",
                                        ),
                                        width=4,
                                    ),
                                ],
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Button(
                                            "Plot",
                                            id=f"{prefix}-plot-graph-btn",
                                            color="light",
                                            style={"width": "100%"},
                                            n_clicks=0,
                                        ),
                                        width=4,
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            "Clear plot(s)",
                                            id=f"{prefix}-clear-graph-btn",
                                            color="light",
                                            style={"width": "100%"},
                                            n_clicks=0,
                                            disabled=True,
                                        ),
                                        width=4,
                                    ),
                                    dbc.Col(
                                        get_graph_settings_button(prefix),
                                        width=4,
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
                            dbc.Row(
                                [
                                    dbc.Col(
                                        InlineLabeledDropdown(
                                            id_=f"{prefix}-export-format-dropdown",
                                            label="Format",
                                            options=[
                                                {"label": "PNG", "value": "png"},
                                                {"label": "SVG", "value": "svg"},
                                                {"label": "JPEG", "value": "jpeg"},
                                                {"label": "WebP", "value": "webp"},
                                            ],
                                            value="png",
                                            multi=False,
                                        ),
                                        width=3,
                                    ),
                                    dbc.Col(
                                        InlineLabeledDropdown(
                                            id_=f"{prefix}-export-scale-dropdown",
                                            label="Scale",
                                            options=[
                                                {"label": "1×", "value": 1},
                                                {"label": "2×", "value": 2},
                                                {"label": "4×", "value": 4},
                                            ],
                                            value=2,
                                            multi=False,
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        dbc.Checklist(
                                            id=f"{prefix}-export-transparent",
                                            options=[{"label": "Transparent background", "value": "on"}],
                                            value=[],
                                            inline=True,
                                        ),
                                        width=3,
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            "Download (transparent)",
                                            id=f"{prefix}-transparent-download-btn",
                                            color="light",
                                            size="sm",
                                        ),
                                        width=3,
                                    ),
                                ],
                                style={"marginTop": "0.5rem"},
                                align="center",
                            ),
                            html.Div(
                                id=f"{prefix}-export-hint",
                                style=styles.HINT_SM,
                            ),
                        ],
                        style={"display": "none"},
                    ),
                    html.Div(id=f"{prefix}-selection-count", style=styles.HINT_SM),
                    html.Div(id=f"{prefix}-graph-area", children=[]),
                    dcc.Store(id=f"{prefix}-graph-meta-store", data={}),
                    dcc.Store(id=f"{prefix}-graph-counter", data=0),
                    dcc.Download(id=f"{prefix}-transparent-download"),
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


def register_table_plot_callbacks(app, prefix: str, connected_store_id, special_graphs=None, table_grid_id=None):

    graph_options = [
        "Line plot",
        "Scatter plot 1D",
        "Scatter plot 2D",
        "Histogram",
        "Histogram 2D",
        "Kernel density estimation",
        "Orientation distribution (3D)",
    ]

    if special_graphs is not None:
        graph_options = graph_options + special_graphs

    @app.callback(
        Output(f"{prefix}-graph-options-dropdown", "options"),
        Input(f"{prefix}-options-init", "data"),
    )
    def load_graph_options(_):
        return graph_options

    @app.callback(
        Output(f"{prefix}-graph-options", "style"),
        Output(f"{prefix}-plot-column-options-x-dropdown", "options"),
        Output(f"{prefix}-scatter2D-row-options", "style"),
        Output(f"{prefix}-histogram-row-options", "style"),
        Output(f"{prefix}-histogram2D-row-options", "style"),
        Output(f"{prefix}-plot-column-options-y-dropdown", "options"),
        Output(f"{prefix}-histogram2D-column-options-y-dropdown", "options"),
        Output(f"{prefix}-orbd-row-options", "style"),
        Input(f"{prefix}-graph-options-dropdown", "value"),
        State(connected_store_id, "data"),
        prevent_initial_call=True,
    )
    def generate_data_options(graph_type, data):

        if graph_type is None:
            return no_update

        x_axis_options = pd.DataFrame(data).columns
        y_axis_options = []
        orbd_options = {"display": "none"}

        def get_spherical_columns(pfx):
            cols = [pfx + "_x", pfx + "_y", pfx + "_z"]
            if all(col in x_axis_options for col in cols):
                return {"label": f"{pfx}_x, {pfx}_y, {pfx}_z", "value": json.dumps(cols)}
            return None

        def get_angle_columns(pfx):
            cols = [pfx + "phi", pfx + "theta", pfx + "psi"]
            if all(col in x_axis_options for col in cols):
                return {"label": f"{pfx}phi, {pfx}theta, {pfx}psi", "value": json.dumps(cols)}
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
                for pfx in ["twist_so", "twist", "norm_nn"]:
                    opt = get_spherical_columns(pfx)
                    if opt:
                        dropdown_options.append(opt)
                x_axis_options = dropdown_options
                y_axis_options = ["None - computed automatically"]
        elif graph_type == "Scatter plot 2D":
            histogram_options = {"display": "none"}
            histogram2D_options = {"display": "none"}
            scatter_2D_options = {"display": "flex"}
            y_axis_options = pd.DataFrame(data).columns
        elif graph_type in ("Orientational distribution", "Polar NN distances"):
            histogram_options = {"display": "none"}
            histogram2D_options = {"display": "none"}
            scatter_2D_options = {"display": "flex"} if graph_type == "Polar NN distances" else {"display": "none"}
            all_data_cols = list(x_axis_options)
            triplet_options = []
            seen_pfx = set()
            for col in all_data_cols:
                if col.endswith("_x"):
                    pfx = col[:-2]
                    if pfx not in seen_pfx:
                        seen_pfx.add(pfx)
                        opt = get_spherical_columns(pfx)
                        if opt:
                            triplet_options.append(opt)
            x_axis_options = triplet_options
            if graph_type == "Polar NN distances":
                y_axis_options = all_data_cols
        elif graph_type == "Orientation distribution (3D)":
            histogram_options = {"display": "none"}
            histogram2D_options = {"display": "none"}
            scatter_2D_options = {"display": "none"}
            orbd_options = {"display": "flex"}
            all_data_cols = list(x_axis_options)
            angle_options = []
            seen_pfx = set()
            for col in all_data_cols:
                if col.endswith("phi"):
                    pfx = col[:-3]
                    if pfx not in seen_pfx:
                        seen_pfx.add(pfx)
                        opt = get_angle_columns(pfx)
                        if opt:
                            angle_options.append(opt)
            x_axis_options = angle_options
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
            orbd_options,
        )

    @app.callback(
        Output(f"{prefix}-same-range", "disabled"),
        Output(f"{prefix}-plot-grid-dropdown", "disabled"),
        Output(f"{prefix}-histogram2D-same-scale", "disabled"),
        Input(f"{prefix}-plot-separately", "value"),
        prevent_initial_call=True,
    )
    def toggle_separate_options(plot_separately):

        if plot_separately:
            return False, False, False
        else:
            return True, True, True

    @app.callback(
        Output(f"{prefix}-clear-graph-btn", "disabled"),
        Input(f"{prefix}-graph-area", "children"),
    )
    def toggle_clear_button(graph_area):
        if len(graph_area) == 0:
            return True
        else:
            return False

    @app.callback(
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

    @app.callback(
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
        State(ids.GRAPH_SETTINGS_STORE, "data"),
        State(f"{prefix}-orbd-binmode-dropdown", "value"),
        State(f"{prefix}-orbd-value-input", "value"),
        State(f"{prefix}-orbd-height-scale-input", "value"),
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
        settings,
        orbd_mode,
        orbd_value,
        orbd_height_scale,
    ):

        trigger_id = ctx.triggered_id
        input_data = pd.DataFrame(data) if data else pd.DataFrame()

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

            effective_colorscale = (
                colorscale
                or (settings.get("discrete_palette") if settings else None)
            )

            fig = None
            _orbd_meta: dict = {}
            if graph_type == "Histogram":

                fig = visplot.plot_histogram(
                    input_data=input_data,
                    column_names_x=x_values,
                    bins=h_bins,
                    separate_graphs=plot_separately,
                    hist_type=h_type.lower(),
                    hist_norm=h_norm.lower(),
                    same_range_for_separate=same_range,
                    colors=effective_colorscale,
                    opacity=None,
                    grid_spec=grid_spec,
                )
            elif graph_type == "Histogram 2D":

                fig = visplot.plot_histogram_2d(
                    input_data=input_data,
                    column_names_x=x_values,
                    separate_graphs=plot_separately,
                    second_axis_data=input_data,
                    column_names_y=h2D_column_options,
                    hist_type=h2D_type.lower(),
                    hist_norm=h2D_norm.lower(),
                    nbinsx=h2D_binsx,
                    nbinsy=h2D_binsy,
                    same_range_for_separate=same_range,
                    colors=effective_colorscale,
                    opacity=None,
                    grid_spec=grid_spec,
                    same_scale=h2D_same_scale,
                )
            elif graph_type == "Kernel density estimation":
                fig = visplot.plot_kde(
                    input_data=input_data,
                    column_names_x=x_values,
                    second_axis_data=input_data,
                    column_names_y=h2D_column_options,
                    nbinsx=h2D_binsx,
                    nbinsy=h2D_binsy,
                    hist_type=h2D_type.lower(),
                    hist_norm=h2D_norm.lower(),
                    colors=effective_colorscale,
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
                fig = visplot.plot_spherical_density_2d(
                    input_data=input_data,
                    column_names_x=all_ids,
                    nbinsx=h2D_binsx,
                    nbinsy=h2D_binsy,
                    x_range=None,
                    y_range=None,
                    hist_type=h2D_type.lower(),
                    hist_norm=h2D_norm.lower(),
                    normalize_coord=True,
                    colors=effective_colorscale,
                    same_scale=h2D_same_scale,
                    same_range_for_separate=same_range,
                    grid_spec=grid_spec,
                )
            elif graph_type == "Orientational distribution":
                all_ids = []
                for s in x_values:
                    all_ids.extend(json.loads(s))
                coords = input_data[all_ids].to_numpy()
                fig = visplot.plot_orientational_distribution(
                    coords, projection="stereo", colormap="viridis_r", theta_bin=12, radius_bin=5
                )
            elif graph_type == "Polar NN distances":
                all_ids = []
                for s in x_values:
                    all_ids.extend(json.loads(s))
                coords = input_data[all_ids].to_numpy()
                if not y_values:
                    return graph_area, "Select a distance column for the Y axis.", no_update, no_update
                dist_col = y_values[0] if isinstance(y_values, list) else y_values
                nn_dist = input_data[dist_col].to_numpy()
                fig = visplot.plot_polar_nn_distances(coords, nn_dist, colormap="viridis_r", marker_size=7)
            elif graph_type == "Orientation distribution (3D)":
                all_ids = []
                for s in x_values:
                    all_ids.extend(json.loads(s))
                if len(all_ids) < 3:
                    return (
                        graph_area,
                        "Select Euler angle columns (phi, theta, psi) for the orientation distribution.",
                        no_update,
                        no_update,
                    )
                angles = input_data[all_ids[:3]].to_numpy()
                continuous_pal = (settings or {}).get("continuous_palette", "StarryNight")
                n_bins_val = None
                cone_s_val = None
                if orbd_mode == "n_bins":
                    n_bins_val = int(orbd_value) if orbd_value is not None else None
                else:
                    cone_s_val = float(orbd_value) if orbd_value is not None else None
                fig = visplot.plot_rotation_normals_binned(
                    angles,
                    n_bins=n_bins_val,
                    cone_sampling=cone_s_val,
                    height_scale=float(orbd_height_scale) if orbd_height_scale is not None else 0.3,
                    colors=continuous_pal,
                )
                _orbd_meta = {
                    "x_cols": all_ids[:3],
                    "orbd_params": {"n_bins": n_bins_val, "cone_sampling": cone_s_val},
                }
            elif graph_type == "Line plot":  # , "Scatter plot 1D", "Scatter plot 2D""
                fig = visplot.plot_line(
                    input_data=input_data,
                    column_names_x=x_values,
                    separate_graphs=plot_separately,
                    same_range_for_separate=same_range,
                    colors=effective_colorscale,
                    opacity=None,
                    grid_spec=grid_spec,
                )
            elif graph_type == "Scatter plot 1D":  # "Scatter plot 2D""
                fig = visplot.plot_scatter_2d(
                    input_data=input_data,
                    column_names_x=x_values,
                    separate_graphs=plot_separately,
                    same_range_for_separate=same_range,
                    colors=effective_colorscale,
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

                fig = visplot.plot_scatter_2d(
                    input_data=input_data,
                    column_names_x=x_values,
                    second_axis_data=input_data,
                    column_names_y=y_values,
                    separate_graphs=plot_separately,
                    same_range_for_separate=same_range,
                    colors=effective_colorscale,
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
                # Apply global graph settings to the new figure
                fig = styled_figure(fig, settings or {}, uirevision=f"{prefix}-graph-{graph_counter}")
                graph_meta = graph_meta or {}
                graph_meta[str(graph_counter)] = {"type": graph_type, "x_cols": x_values, **_orbd_meta}
                new_graph = dcc.Graph(
                    id={"type": "styled-graph", "owner": prefix, "name": graph_counter},
                    figure=fig,
                )
                return graph_area + [new_graph], [], graph_meta, graph_counter + 1

        return no_update, no_update, no_update, no_update

    @app.callback(
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

    # ── Section 1: palette dropdown — dynamic options + search + validation ──

    @app.callback(
        Output(f"{prefix}-plot-color-palette-dropdown", "options"),
        Input(f"{prefix}-plot-color-palette-dropdown", "search_value"),
        State(f"{prefix}-plot-color-palette-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _palette_options(typed, current):
        opts = [{"label": p, "value": p} for p in _PALETTE_PRESETS]
        if current and current not in _PALETTE_PRESETS and current != "__hint__":
            opts.append({"label": current, "value": current})
        if typed and len(typed) >= 2 and typed not in _PALETTE_PRESETS:
            opts.insert(0, {"label": f'Use "{typed}"', "value": typed})
        opts.append(_PALETTE_HINT)
        return opts

    @app.callback(
        Output(f"{prefix}-palette-error", "children"),
        Input(f"{prefix}-plot-color-palette-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _palette_guard(value):
        if not value or value == "__hint__" or value in _PALETTE_PRESETS:
            return ""
        try:
            visplot.resolve_palette(value)
            return ""
        except Exception:
            pass
        try:
            visplot.resolve_colorscale(value)
            return ""
        except Exception:
            return f"Unknown palette: {value!r}"

    # ── Section 3: export controls ────────────────────────────────────────────

    @app.callback(
        Output(f"{prefix}-export-scale-dropdown", "disabled"),
        Input(f"{prefix}-export-format-dropdown", "value"),
    )
    def _export_scale_enable(fmt):
        return fmt == "svg"

    @app.callback(
        Output(f"{prefix}-export-hint", "children"),
        Output(f"{prefix}-export-format-dropdown", "options"),
        Input(f"{prefix}-export-transparent", "value"),
        Input(f"{prefix}-export-format-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _export_jpeg_hint(transparent, fmt):
        is_transparent = bool(transparent)
        opts = [
            {"label": "PNG", "value": "png"},
            {"label": "SVG", "value": "svg"},
            {"label": "JPEG", "value": "jpeg", "disabled": is_transparent},
            {"label": "WebP", "value": "webp"},
        ]
        hint = (
            "JPEG has no alpha channel — use PNG, SVG, or WebP for transparency."
            if is_transparent and fmt == "jpeg"
            else ""
        )
        return hint, opts

    @app.callback(
        Output({"type": "styled-graph", "owner": prefix, "name": ALL}, "config"),
        Input(f"{prefix}-export-format-dropdown", "value"),
        Input(f"{prefix}-export-scale-dropdown", "value"),
        State({"type": "styled-graph", "owner": prefix, "name": ALL}, "id"),
    )
    def _update_graph_config(fmt, scale, graph_ids):
        n = len(graph_ids)
        if n == 0:
            return []
        cfg = {
            "toImageButtonOptions": {
                "format": fmt or "png",
                "scale": scale or 2,
                "filename": "graph",
            },
        }
        return [cfg] * n

    @app.callback(
        Output(f"{prefix}-transparent-download", "data"),
        Input(f"{prefix}-transparent-download-btn", "n_clicks"),
        State({"type": "styled-graph", "owner": prefix, "name": ALL}, "figure"),
        State(f"{prefix}-export-format-dropdown", "value"),
        State(f"{prefix}-export-scale-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _transparent_download(n_clicks, figures, fmt, scale):
        if not n_clicks or not figures:
            raise dash.exceptions.PreventUpdate
        fmt = (fmt or "png").lower()
        if fmt == "jpeg":
            fmt = "png"
        fig = go.Figure(figures[-1])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        try:
            img_bytes = fig.to_image(format=fmt, scale=scale or 2)
        except Exception:
            raise dash.exceptions.PreventUpdate
        return dcc.send_bytes(img_bytes, f"graph.{fmt}")

    if table_grid_id is not None:

        @app.callback(
            Output(table_grid_id, "selectedRows"),
            Output(f"{prefix}-selection-count", "children"),
            Input({"type": "styled-graph", "owner": prefix, "name": ALL}, "clickData"),
            Input({"type": "styled-graph", "owner": prefix, "name": ALL}, "selectedData"),
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

            prop_id = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
            is_click = "clickData" in prop_id

            data_value = ctx.triggered[0]["value"] if ctx.triggered else None
            if not data_value or not data_value.get("points"):
                raise dash.exceptions.PreventUpdate

            graph_idx = triggered.get("name")
            meta_entry = graph_meta.get(str(graph_idx))
            if meta_entry is None:
                raise dash.exceptions.PreventUpdate

            graph_type = meta_entry["type"]
            n = len(row_data)
            points = data_value["points"]

            if graph_type == "Orientation distribution (3D)" and is_click:
                from scipy.spatial import cKDTree as _cKDTree
                from cryocat.utils import geom as _geom

                raw_cd = points[0].get("customdata")
                if raw_cd is None:
                    raise dash.exceptions.PreventUpdate
                orig_bin_idx = int(raw_cd[0]) if isinstance(raw_cd, list) else int(raw_cd)

                x_cols = meta_entry.get("x_cols", [])
                if len(x_cols) < 3:
                    raise dash.exceptions.PreventUpdate

                orbd_params = meta_entry.get("orbd_params", {})
                n_bins = orbd_params.get("n_bins")
                cone_s = orbd_params.get("cone_sampling")
                if n_bins is not None:
                    n_dirs = int(n_bins)
                elif cone_s is not None:
                    n_dirs = int(_geom.number_of_cone_rotations(360.0, cone_s))
                else:
                    n_dirs = int(_geom.number_of_cone_rotations(360.0, 5.0))

                bin_dirs = _geom.sample_sphere(n_dirs)
                try:
                    angles = np.array([[row[c] for c in x_cols] for row in row_data], dtype=float)
                except (KeyError, ValueError):
                    raise dash.exceptions.PreventUpdate

                normals = _geom.rotations_to_z_normals(angles, radius=1.0)
                _, assignment = _cKDTree(bin_dirs).query(normals)
                indices = [i for i, b in enumerate(assignment) if b == orig_bin_idx]

            elif graph_type == "Histogram":
                indices = []
                for p in points:
                    indices.extend(p.get("pointNumbers", []))

            else:
                if is_click and graph_type in _AGGREGATED_CLICK_IGNORE:
                    raise dash.exceptions.PreventUpdate
                indices = [p["pointIndex"] for p in points if "pointIndex" in p]

            new_rows = [row_data[i] for i in indices if 0 <= i < n]
            if not new_rows and sel_mode != "subtract":
                raise dash.exceptions.PreventUpdate

            def _count(rows: list) -> str:
                k = len(rows)
                return f"{k} row{'s' if k != 1 else ''} selected"

            if sel_mode == "replace":
                return new_rows, _count(new_rows)

            current = current_selected or []

            if sel_mode == "add":
                new_keys = {json.dumps(r, sort_keys=True) for r in new_rows}
                existing_keys = {json.dumps(r, sort_keys=True) for r in current}
                if new_keys and new_keys.issubset(existing_keys):
                    result = [r for r in current if json.dumps(r, sort_keys=True) not in new_keys]
                    return result, _count(result)
                merged = list(current)
                for r in new_rows:
                    if json.dumps(r, sort_keys=True) not in existing_keys:
                        merged.append(r)
                return merged, _count(merged)

            if sel_mode == "subtract":
                remove_keys = {json.dumps(r, sort_keys=True) for r in new_rows}
                result = [r for r in current if json.dumps(r, sort_keys=True) not in remove_keys]
                return result, _count(result)

            return new_rows, _count(new_rows)
