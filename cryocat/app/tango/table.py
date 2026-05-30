from cryocat.app.logger import dash_logger

import dash
import pandas as pd
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash import Input, Output, State, ALL, ctx, no_update
from cryocat.app.components.tomoview import get_viewer_component, register_viewer_callbacks
from cryocat.app.components.tableview import get_table_component, register_table_callbacks
from cryocat.app.components.tableplot import register_table_plot_callbacks
from cryocat.app.components.tablecluster import register_table_cluster_callbacks

from cryocat.analysis.tango import TwistDescriptor
from cryocat.core.cryomotl import Motl
from cryocat.app.apputils import generate_kwargs, _scalar, _format_relion_params


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
                        label="Neighbor list",
                        tab_id="nn-motl-tab",
                        disabled=True,
                        id="nn-motl-tab",
                    ),
                    dbc.Tab(
                        dbc.Card(dbc.CardBody([get_nn_table()]), className="mt-3"),
                        label="Nearest neighbor",
                        tab_id="nn-tab",
                        disabled=True,
                        id="nn-tab",
                    ),
                    dbc.Tab(
                        dbc.Card(dbc.CardBody([get_twist_table()]), className="mt-3"),
                        label="Twist vectors",
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
                    dbc.Tab(
                        dbc.Card(dbc.CardBody([get_cluster_table()]), className="mt-3"),
                        label="Clustering",
                        tab_id="cluster-tab",
                        disabled=True,
                        id="cluster-tab",
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
            get_table_component("tabv-motl", connected_motl_prefix="main", show_create_from_selected=False),
            get_viewer_component("tviewer-motl"),
        ],
    )


# Table with nn motl
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
            get_table_component("tabv-motl-nn", connected_motl_prefix="nn", show_create_from_selected=False),
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
                children=[],
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
                children=[],
            ),
            get_table_component("tabv-desc"),
            dbc.Row(
                align="center",
                className="mb-3",
                children=[
                    dbc.Col(get_viewer_component("tviewer-desc")),
                ],
            ),
        ]
    )


def get_cluster_table():

    return html.Div(
        children=[
            dbc.Col(
                [
                    html.Div(
                        children=[
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(id="pca-visualization"),
                                        ],
                                        width=8,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Div(
                                                id="prox-cluster-display",
                                                style={"display": "none"},
                                                children=[
                                                    html.H4(
                                                        "Proximity clustering parameters",
                                                        style={"marginBottom": "1rem"},
                                                    ),
                                                    html.H5("Number of connected components:"),
                                                    dcc.Slider(
                                                        id=f"prox-cluster-num-com-slider",
                                                        min=0,
                                                        max=50,
                                                        step=1,
                                                        value=2,
                                                        tooltip={"placement": "right"},
                                                        marks=None,
                                                    ),
                                                    html.H5("Or", style={"marginBottom": "1rem", "Top": "0rem"}),
                                                    html.H5("Minimum occupancy of each component:"),
                                                    dcc.Slider(
                                                        id=f"prox-cluster-occ-com-slider",
                                                        min=0,
                                                        max=50,
                                                        step=1,
                                                        value=2,
                                                        tooltip={"placement": "right"},
                                                        marks=None,
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                id="k-means-display",
                                                style={"display": "none"},
                                                children=[
                                                    html.H4("K-means parameters"),
                                                    html.H5(
                                                        "Features to cluster",
                                                        style={"marginBottom": "1rem", "marginTop": "1rem"},
                                                    ),
                                                    dcc.Checklist(
                                                        id="k-means-options",
                                                        options=[],
                                                        inline=True,
                                                        style={
                                                            "display": "flex",
                                                            "flexWrap": "wrap",
                                                            "flexDirection": "column",
                                                            "height": "100px",
                                                            "overflowY": "auto",
                                                        },
                                                        labelStyle={"color": "var(--color12)"},
                                                        inputStyle={"marginRight": "5px"},
                                                        className="sidebar-checklist",
                                                    ),
                                                    html.H5(
                                                        "Number of clusters",
                                                        style={"marginBottom": "1rem", "marginTop": "1rem"},
                                                    ),
                                                    dcc.Slider(
                                                        id=f"k-means-n-slider",
                                                        min=1,
                                                        max=50,
                                                        step=1,
                                                        value=2,
                                                        tooltip={"placement": "right"},
                                                        marks=None,
                                                    ),
                                                    dbc.Button(
                                                        "Compute k_means",
                                                        id="cluster-run-btn",
                                                        color="primary",
                                                        n_clicks=0,
                                                        style={
                                                            "width": "100%",
                                                            "marginTop": "0",
                                                            "marginBottom": "1rem",
                                                        },
                                                    ),
                                                ],
                                            ),
                                        ],
                                        width=4,
                                    ),
                                ],
                            ),
                            dbc.Row(
                                [
                                    html.Div(
                                        id="proximity-graph-cont",
                                        style={"display": "none"},
                                        children=[
                                            dbc.Col(
                                                get_viewer_component("tviewer-proximity"),
                                                width=12,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        id="k-means-graph-cont",
                                        style={"display": "none"},
                                        children=[
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        get_viewer_component("tviewer-kmeans"),
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        children=[
                                                            html.Div(
                                                                [
                                                                    dcc.Dropdown(
                                                                        id="kmeans-x-axis",
                                                                        placeholder="Select X axis",
                                                                        style={"minWidth": "100px", "width": "100%"},
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="kmeans-y-axis",
                                                                        placeholder="Select Y axis",
                                                                        style={"minWidth": "100px", "width": "100%"},
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="kmeans-z-axis",
                                                                        placeholder="Select Z axis (optional)",
                                                                        style={"minWidth": "100px", "width": "100%"},
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "flex",
                                                                    "gap": "10px",
                                                                    "marginBottom": "1rem",
                                                                },
                                                            ),
                                                            dcc.Graph(id="kmeans-all-graph"),
                                                        ],
                                                        width=6,
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ]
            ),
        ]
    )


def register_tango_table_callbacks(app):

    register_viewer_callbacks(app, "tviewer-motl")
    register_viewer_callbacks(app, "tviewer-motl-nn")
    register_viewer_callbacks(
        app,
        "tviewer-twist",
        show_dual_graph=True,
        hover_info=["subtomo_id", "tomo_id"],
        detailed_table="merged-motl-twist-data-store",
    )
    register_viewer_callbacks(app, "tviewer-desc", show_dual_graph=False, hover_info=["subtomo_id"])
    register_viewer_callbacks(app, "tviewer-kmeans", show_dual_graph=False, hover_info=["subtomo_id", "class"])
    register_viewer_callbacks(app, "tviewer-proximity", show_dual_graph=False, hover_info=["subtomo_id", "class"])

    register_table_callbacks(app, "tabv-motl", csv_only=False, connected_motl_prefix="main")
    register_table_callbacks(app, "tabv-motl-nn", csv_only=False, connected_motl_prefix="nn")
    register_table_callbacks(app, "tabv-nn")
    register_table_callbacks(app, "tabv-twist")
    register_table_callbacks(app, "tabv-desc")

    register_table_plot_callbacks(app, "tabv-motl-table-plot", "tabv-motl-global-data-store", table_grid_id="tabv-motl-grid")
    register_table_plot_callbacks(app, "tabv-motl-nn-table-plot", "tabv-motl-nn-global-data-store", table_grid_id="tabv-motl-nn-grid")
    register_table_plot_callbacks(app, "tabv-nn-table-plot", "tabv-nn-global-data-store", special_graphs=["Spherical histogram"], table_grid_id="tabv-nn-grid")
    register_table_plot_callbacks(
        app, "tabv-twist-table-plot", "tabv-twist-global-data-store", special_graphs=["Spherical histogram"], table_grid_id="tabv-twist-grid"
    )
    register_table_plot_callbacks(app, "tabv-desc-table-plot", "tabv-desc-global-data-store", table_grid_id="tabv-desc-grid")

    register_table_cluster_callbacks(app, "tabv-motl-table-cluster", "tabv-motl-global-data-store", table_grid_id="tabv-motl-grid")
    register_table_cluster_callbacks(app, "tabv-motl-nn-table-cluster", "tabv-motl-nn-global-data-store", table_grid_id="tabv-motl-nn-grid")
    register_table_cluster_callbacks(app, "tabv-nn-table-cluster", "tabv-nn-global-data-store", table_grid_id="tabv-nn-grid")
    register_table_cluster_callbacks(app, "tabv-twist-table-cluster", "tabv-twist-global-data-store", table_grid_id="tabv-twist-grid")
    register_table_cluster_callbacks(app, "tabv-desc-table-cluster", "tabv-desc-global-data-store", table_grid_id="tabv-desc-grid")

    @app.callback(
        Output("main-relion-params-inline", "children"),
        Input("main-relion-params-store", "data"),
    )
    def update_main_relion_params_display(params):
        return _format_relion_params(params)

    @app.callback(
        Output("nn-relion-params-inline", "children"),
        Input("nn-relion-params-store", "data"),
    )
    def update_nn_relion_params_display(params):
        return _format_relion_params(params)

    @app.callback(
        Output("tviewer-motl-data", "data", allow_duplicate=True),
        Output("tabv-motl-global-data-store", "data", allow_duplicate=True),
        Input("main-motl-data-store", "data"),
        prevent_initial_call=True,
    )
    def connect_motl_data(data):
        return data, data

    @app.callback(
        Output("tviewer-motl-data", "data", allow_duplicate=True),
        Input("tabv-motl-global-data-store", "data"),
        Input("tabv-motl-grid", "rowData"),
        prevent_initial_call=True,
    )
    def connect_motl_table_data(motl_data, table_data):
        trigger_id = ctx.triggered_id
        if trigger_id == "tabv-motl-grid":
            changed_data = table_data
        else:
            changed_data = motl_data
        return changed_data

    @app.callback(
        Output("tviewer-twist-data", "data"),
        Output("merged-motl-twist-data-store", "data", allow_duplicate=True),
        Input("tabv-twist-global-data-store", "data"),
        Input("tabv-twist-grid", "rowData"),
        State("tabv-motl-global-data-store", "data"),
        prevent_initial_call=True,
    )
    def connect_twist_data(desc_data, table_data, motl_data):
        motl_df = pd.DataFrame(motl_data)
        trigger_id = ctx.triggered_id
        if trigger_id == "tabv-twist-grid":
            changed_df = pd.DataFrame(table_data)
            motl_df = motl_df[motl_df["subtomo_id"].isin(changed_df["qp_id"])]
        else:
            changed_df = pd.DataFrame(desc_data)
        merged = motl_df.merge(changed_df, left_on="subtomo_id", right_on="qp_id", how="left")
        merged = merged.dropna()
        return motl_df.to_dict("records"), merged.to_dict("records")

    @app.callback(
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

    @app.callback(
        Output("tviewer-kmeans-data", "data", allow_duplicate=True),
        Input("tabv-motl-global-data-store", "data"),
        prevent_initial_call=True,
    )
    def create_kmeans_dummy_graph(motl_data):
        return motl_data

    @app.callback(
        Output("nn-motl-tab", "disabled", allow_duplicate=True),
        Output("table-tabs", "active_tab", allow_duplicate=True),
        Output("tviewer-motl-nn-data", "data", allow_duplicate=True),
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

    @app.callback(
        Output("tviewer-motl-nn-data", "data", allow_duplicate=True),
        Input("tabv-motl-nn-global-data-store", "data"),
        Input("tabv-motl-nn-grid", "rowData"),
        State("tabv-motl-global-data-store", "data"),
        prevent_initial_call=True,
    )
    def connect_motl_nn_table_data(motl_nn_data, table_data, motl_data):
        if not motl_nn_data:
            raise dash.exceptions.PreventUpdate
        motl_df = pd.DataFrame(motl_data)
        trigger_id = ctx.triggered_id
        if trigger_id == "tabv-motl-nn-grid":
            changed_df = pd.DataFrame(table_data)
            motl_df = motl_df[motl_df["subtomo_id"].isin(changed_df["subtomo_id"])]
        else:
            changed_df = pd.DataFrame(motl_nn_data)
        motl_df["type"] = 1
        changed_df["type"] = 2
        con_motl = pd.concat([motl_df, changed_df], ignore_index=True)
        con_motl.reset_index(drop=True, inplace=True)
        return con_motl.to_dict("records")

    @app.callback(
        Output("motl-display-div", "style"),
        Input("main-motl-data-store", "data"),
        prevent_initial_call=True,
    )
    def render_motl_tab_content(data):
        if data is None:
            raise dash.exceptions.PreventUpdate
        return {"display": "block"}

    @app.callback(
        Output("motl-nn-display-div", "style"),
        Input("nn-motl-data-store", "data"),
        prevent_initial_call=True,
    )
    def render_motl_nn_tab_content(data):
        if data is None:
            raise dash.exceptions.PreventUpdate
        return {"display": "block"}

    @app.callback(
        Output("twist-display-div", "style"),
        Input("tabv-twist-global-data-store", "data"),
        prevent_initial_call=True,
    )
    def render_twist_tab_content(data):
        if data is None:
            raise dash.exceptions.PreventUpdate
        return {"display": "block"}

    @app.callback(
        Output("twist-tab", "disabled"),
        Input("tabv-twist-global-data-store", "data"),
        prevent_initial_call=True,
    )
    def enable_twist_tab(data):
        if data is None:
            raise dash.exceptions.PreventUpdate
        return False

    @app.callback(
        Output("custom-spinner", "style", allow_duplicate=True),
        Output("spinner-trigger", "data"),
        Input("run-twist-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def show_spinner(n):
        if not n:
            raise dash.exceptions.PreventUpdate
        style = {
            "position": "fixed",
            "top": "50%",
            "left": "50%",
            "transform": "translate(-50%, -50%)",
            "zIndex": "3000",
            "textAlign": "center",
            "display": "block",
        }
        return style, {"show": True}

    @app.callback(
        Output("table-tabs", "active_tab", allow_duplicate=True),
        Output("tabv-twist-global-data-store", "data"),
        Output("twist-global-radius", "data"),
        Output("custom-spinner", "style", allow_duplicate=True),
        Output("dummy-output", "children"),
        Input("spinner-trigger", "data"),
        State("tabv-motl-global-data-store", "data"),
        State("symmetry-dropdown", "value"),
        State("c-symmetry-value", "value"),
        State({"type": "twist-forms-params", "cls_name": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": "twist-forms-params", "cls_name": ALL, "param": ALL, "tag": ALL}, "id"),
        prevent_initial_call=True,
    )
    def compute_twist_vector(trigger, motl_df, symm_type, symm_value, param_values, param_ids):
        if not trigger or "show" not in trigger:
            raise dash.exceptions.PreventUpdate
        twist_kwargs = generate_kwargs(param_ids, param_values)
        if symm_type == "C":
            symm = symm_value
        elif symm_type == "None":
            symm = None
        else:
            symm = symm_type
        radius = twist_kwargs["nn_radius"]
        from cryocat.app.tango import global_twist
        global_twist["obj"] = TwistDescriptor(input_motl=pd.DataFrame(motl_df), symm=symm, **twist_kwargs)
        hide_style = {"display": "none"}
        return "twist-tab", global_twist["obj"].df.to_dict("records"), radius, hide_style, "Computation complete!"

    @app.callback(
        Output("kmeans-x-axis", "options"),
        Output("kmeans-y-axis", "options"),
        Output("kmeans-z-axis", "options"),
        Input("cluster-run-btn", "n_clicks"),
        State("k-means-options", "value"),
    )
    def populate_axes_dropdown(n_clicks, k_means_options):
        if not k_means_options:
            return no_update
        return k_means_options, k_means_options, k_means_options

    @app.callback(
        Output("kmeans-all-graph", "figure"),
        Input("kmeans-x-axis", "value"),
        Input("kmeans-y-axis", "value"),
        Input("kmeans-z-axis", "value"),
        State("kmeans-global-data-store", "data"),
        prevent_initial_call=True,
    )
    def update_kmeans_all_graph(x, y, z, kmeans_data):
        import plotly.express as px
        if not kmeans_data:
            return no_update
        kmeans_df = pd.DataFrame(kmeans_data)
        if not x or not y or x == y or (z and (x == z or y == z)):
            return no_update
        if z:
            fig = px.scatter_3d(kmeans_df, x=x, y=y, z=z, color="cluster")
        else:
            fig = px.scatter(kmeans_df, x=x, y=y, color="cluster")
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=20, b=20))
        return fig

    @app.callback(
        Output("proximity-graph-cont", "style", allow_duplicate=True),
        Output("merged-motl-proximity-data-store", "data", allow_duplicate=True),
        Output("tviewer-proximity-data", "data"),
        Output("tviewer-proximity-color-dropdown", "value"),
        Input("prox-cluster-occ-com-slider", "value"),
        Input("prox-cluster-num-com-slider", "value"),
        State("cluster-data-dropdown", "value"),
        State("tabv-twist-global-data-store", "data"),
        State("tabv-desc-global-data-store", "data"),
        State("tabv-motl-global-data-store", "data"),
        prevent_initial_call=True,
    )
    def compute_proximity(min_occ_number, number_of_comp, cluster_data, twist_data, desc_data, motl_data):
        trigger_id = ctx.triggered_id
        twist_desc = TwistDescriptor(input_twist=pd.DataFrame(twist_data))
        if cluster_data != "Twist descriptor base":
            desc_df = pd.DataFrame(desc_data)
            twist_desc.df = twist_desc.df[twist_desc.df["qp_id"].isin(desc_df["qp_id"])]
        if trigger_id == "prox-cluster-occ-com-slider":
            prox = twist_desc.proximity_clustering(size_connected_components=min_occ_number)
        else:
            prox = twist_desc.proximity_clustering(num_connected_components=number_of_comp)
        motl_df = pd.DataFrame(motl_data)
        for i, subset in enumerate(prox, start=1):
            subtomo_indices = list(set(subset.nodes()))
            motl_df.loc[motl_df["subtomo_id"].isin(subtomo_indices), "class"] = i
        return {"display": "block"}, motl_df.to_dict("records"), motl_df.to_dict("records"), "class"
