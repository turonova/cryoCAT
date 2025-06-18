from cryocat.app.logger import dash_logger

import dash
import pandas as pd
from dash import dash_table
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash import callback, Input, Output, State, html, dcc, ALL, ctx, no_update
from cryocat.app.layout.tomoview import get_viewer_component, register_viewer_callbacks
from cryocat.app.layout.tableview import get_table_component, register_table_callbacks

from cryocat.tango import TwistDescriptor
from cryocat.app.globalvars import global_twist
from cryocat.app.apputils import generate_kwargs


def get_tabs_menu():
    return html.Div(
        [
            dbc.Tabs(
                [
                    dbc.Tab(
                        dbc.Card(dbc.CardBody([get_motl_table()]), className="mt-3"),
                        label="Particle list",
                        tab_id="motl-tab",
                        id="motl-tab",
                    ),
                    dbc.Tab(
                        dbc.Card(dbc.CardBody([get_nn_motl_table()]), className="mt-3"),
                        label="Neigbor list",
                        tab_id="nn-motl-tab",
                        disabled=True,
                        id="nn-motl-tab",
                    ),
                    dbc.Tab(
                        dbc.Card(dbc.CardBody([get_nn_table()]), className="mt-3"),
                        label="Nearest neigbor",
                        tab_id="nn-tab",
                        disabled=True,
                        id="nn-tab",
                    ),
                    dbc.Tab(
                        dbc.Card(dbc.CardBody([get_twist_table()]), className="mt-3"),
                        label="Twist vector",
                        tab_id="twist-tab",
                        disabled=True,
                        id="twist-tab",
                    ),
                    dbc.Tab(
                        dbc.Card(dbc.CardBody([get_desc_table()]), className="mt-3"),
                        label="Descriptors",
                        tab_id="desc-tab",
                        disabled=True,
                        id="desc-tab",
                    ),
                ],
                id="table-tabs",
                active_tab="motl-tab",
            ),
            html.Div(id="tab-content"),
        ]
    )


def get_main_content():
    return dbc.Col(
        get_tabs_menu(),
        width=10,
        className="p-0",
    )


# Table with main motl
def get_motl_table():
    return html.Div(
        id="motl-display-div",
        style={"display": "none"},
        children=[
            dbc.Row(
                align="center",
                className="mb-3",
                children=[
                    dbc.Col(html.H4("MOTL Table", id="motl-table-title"), width="auto"),
                    dbc.Col(html.Div(id="file-status")),
                ],
            ),
            get_table_component("tabv-motl"),
            get_viewer_component("tviewer-motl"),
        ],
    )


# Table with main motl
def get_nn_motl_table():
    return html.Div(
        id="motl-nn-display-div",
        style={"display": "none"},
        children=[
            dbc.Row(
                align="center",
                className="mb-3",
                children=[
                    dbc.Col(html.H4("MOTL Nearest Neighbor Table", id="motl-nn-table-title"), width="auto"),
                    dbc.Col(html.Div(id="motl-nn-file-status")),
                ],
            ),
            get_table_component("tabv-motl-nn"),
            get_viewer_component("tviewer-motl-nn"),
        ],
    )


def get_nn_table():
    return html.Div(
        [
            dbc.Row(
                align="center",
                className="mb-3",
                children=[
                    # dbc.Col(html.H4("Nearest Neighbor Analysis", id="nn-table-title"), width="auto"),
                    dbc.Col(html.H4(id="nn-stats-print")),
                ],
            ),
            get_table_component("tabv-nn"),
            html.Div(id="motl-visualization", style={"marginTop": "2rem"}),
        ]
    )


def get_twist_table():
    return html.Div(
        id="twist-display-div",
        style={"display": "none"},
        children=[
            dbc.Row(
                align="center",
                className="mb-3",
                children=[
                    # dbc.Col(html.H4("Nearest Neighbor Analysis", id="nn-table-title"), width="auto"),
                    # dbc.Col(html.H4(id="nn-twist-print")),
                ],
            ),
            get_table_component("tabv-twist"),
            get_viewer_component("tviewer-twist"),
        ],
    )


def get_desc_table():
    return html.Div(
        [
            dbc.Row(
                align="center",
                className="mb-3",
                children=[
                    # dbc.Col(html.H4("Descriptors", id="pca-table-title"), width="auto"),
                    # dbc.Col(html.H4(id="pca-results-print")),
                ],
            ),
            get_table_component("tabv-desc"),
            dbc.Row(
                align="center",
                className="mb-3",
                children=[
                    dbc.Col(get_viewer_component("tviewer-desc")),
                    dbc.Col(
                        children=[
                            html.Div(id="pca-visualization"),
                            dcc.Checklist(
                                id="k-means-options",
                                options=[],
                                inline=True,
                                style={
                                    "flexWrap": "wrap",
                                    "display": "flex",
                                    "gap": "10px",
                                },
                            ),
                            dcc.Slider(
                                id=f"k-means-n-slider",
                                min=1,
                                max=10,
                                step=1,
                                value=2,
                            ),
                            dbc.Button(
                                "Compute k_means",
                                id="k-means-run-btn",
                                color="primary",
                                n_clicks=0,
                                style={"width": "100%"},
                            ),
                        ]
                    ),
                ],
            ),
        ]
    )


# Load motl into its viewer
@callback(
    Output("tviewer-motl-data", "data"),
    Output("tabv-motl-global-data-store", "data", allow_duplicate=True),
    Input("main-motl-data-store", "data"),
    prevent_initial_call=True,
)
def connect_motl_data(data):
    # print(data)
    return data, data


# Load twist into its viewer
@callback(
    Output("tviewer-twist-data", "data"),
    Output("merged-motl-twist-data-store", "data", allow_duplicate=True),
    Input("tabv-twist-global-data-store", "data"),
    State("tabv-motl-global-data-store", "data"),
    prevent_initial_call=True,
)
def connect_twist_data(desc_data, motl_data):

    desc_df = pd.DataFrame(desc_data)
    motl_df = pd.DataFrame(motl_data)
    merged = motl_df.merge(desc_df, left_on="subtomo_id", right_on="qp_id", how="left")
    merged = merged.dropna()

    return motl_data, merged.to_dict("records")


# Load descriptor into its viewer
@callback(
    Output("tviewer-desc-data", "data"),
    Input("tabv-desc-global-data-store", "data"),
    State("tabv-motl-global-data-store", "data"),
    prevent_initial_call=True,
)
def connect_desc_data(desc_data, motl_data):

    desc_df = pd.DataFrame(desc_data)
    motl_df = pd.DataFrame(motl_data)
    merged = motl_df.merge(desc_df, left_on="subtomo_id", right_on="qp_id", how="left")
    merged = merged.dropna()

    return merged.to_dict("records")


# Load descriptor into its viewer
@callback(
    Output("nn-motl-tab", "disabled"),
    Output("table-tabs", "active_tab"),
    Output("tviewer-motl-nn-data", "data"),
    Output("tabv-motl-nn-global-data-store", "data", allow_duplicate=True),
    Input("nn-motl-data-store", "data"),
    State("tabv-motl-global-data-store", "data"),
    prevent_initial_call=True,
)
def connect_motl_nn_data(motl_nn_data, motl_data):

    if not motl_nn_data:
        raise dash.exceptions.PreventUpdate

    motl_nn_df = pd.DataFrame(motl_nn_data)

    motl_df = pd.DataFrame(motl_data)
    motl_df["type"] = 1
    motl_nn_df["type"] = 2
    con_motl = pd.concat([motl_df, motl_nn_df], ignore_index=True)
    con_motl.reset_index(drop=True, inplace=True)

    return False, "nn-motl-tab", con_motl.to_dict("records"), motl_nn_data


# Register callbacks for both viewers
register_viewer_callbacks("tviewer-motl")
register_viewer_callbacks("tviewer-motl-nn")
register_viewer_callbacks(
    "tviewer-twist",
    show_dual_graph=True,
    hover_info=["subtomo_id", "tomo_id"],
    detailed_table="merged-motl-twist-data-store",
)
register_viewer_callbacks("tviewer-desc", show_dual_graph=False, hover_info=["subtomo_id"])

register_table_callbacks("tabv-motl")
register_table_callbacks("tabv-motl-nn")
register_table_callbacks("tabv-nn")
register_table_callbacks("tabv-twist")
register_table_callbacks("tabv-desc")


@callback(
    Output("motl-display-div", "style"),
    Input("main-motl-data-store", "data"),
    prevent_initial_call=True,
)
def render_motl_tab_content(data):
    if data is None:
        raise dash.exceptions.PreventUpdate

    return {"display": "block"}


@callback(
    Output("motl-nn-display-div", "style"),
    Input("nn-motl-data-store", "data"),
    prevent_initial_call=True,
)
def render_motl_nn_tab_content(data):
    if data is None:
        raise dash.exceptions.PreventUpdate

    return {"display": "block"}


@callback(
    Output("twist-display-div", "style"),
    Input("tabv-twist-global-data-store", "data"),
    prevent_initial_call=True,
)
def render_twist_tab_content(data):
    if data is None:
        raise dash.exceptions.PreventUpdate

    return {"display": "block"}


@callback(
    Output("twist-tab", "disabled"),
    Input("tabv-twist-global-data-store", "data"),
    prevent_initial_call=True,
)
def enable_twist_tab(data):
    if data is None:
        raise dash.exceptions.PreventUpdate

    return False


@callback(
    Output("table-tabs", "active_tab", allow_duplicate=True),
    Output("tabv-twist-global-data-store", "data"),
    Input("run-twist-btn", "n_clicks"),
    State("main-motl-data-store", "data"),
    State("symmetry-dropdown", "value"),
    State("c-symmetry-value", "value"),
    State({"type": "twist-forms-params", "cls_name": ALL, "param": ALL, "p_type": ALL}, "value"),
    State({"type": "twist-forms-params", "cls_name": ALL, "param": ALL, "p_type": ALL}, "id"),
    prevent_initial_call=True,
)
def compute_twist_vector(n_clicks, motl_df, symm_type, symm_value, param_valus, param_ids):

    twist_kwargs = generate_kwargs(param_ids, param_valus)

    if symm_type == "C":
        symm = symm_value
    elif symm_type == "None":
        symm = None
    else:
        symm = symm_type

    global_twist["obj"] = TwistDescriptor(input_motl=pd.DataFrame(motl_df), symm=symm, **twist_kwargs)

    return "twist-tab", global_twist["obj"].df.to_dict("records")
