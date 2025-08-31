from cryocat.app.logger import dash_logger
import base64
import io
import numpy as np
from dash import html, dcc, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
from dash import callback, Input, Output, State, no_update
from cryocat.app.layout.motlio import get_motl_load_component, register_motl_load_callbacks
from cryocat.app.layout.motlio import get_motl_save_component, register_motl_save_callbacks

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from cryocat.classutils import get_class_names_by_parent, get_classes_from_names
from cryocat.cryomotl import Motl
from cryocat.nnana import NearestNeighbors
from cryocat.tango import TwistDescriptor, Descriptor, CustomDescriptor
from cryocat.nnana import plot_nn_rot_coord_df_plotly

from cryocat.app.globalvars import global_twist
from cryocat.app.apputils import generate_form_from_docstring, generate_kwargs

descriptors = get_class_names_by_parent("Descriptor", "cryocat.tango")
features = get_class_names_by_parent("Feature", "cryocat.tango")
supports = get_class_names_by_parent("Support", "cryocat.tango")
feat_desc_map = Descriptor.build_feature_descriptor_map(features, descriptors)
desc_feat_map = Descriptor.build_descriptor_feature_map(descriptors, features)

stored_outputs = {
    "K-means clustering": "merged-motl-kmeans-data-store",
    "Proximity clustering": "merged-motl-proximity-data-store",
}

label_style = {
    "marginRight": "0.5rem",
    "width": "40%",
    "display": "flex",
    "alignItems": "center",
    "boxSizing": "border-box",
}
param_style = {
    "marginRight": "0.0rem",
    "width": "60%",
    "boxSizing": "border-box",
}


def get_column_sidebar():
    return dbc.Col(
        get_sidebar(),
        width=2,
        className="p-0",
        style={
            "backgroundColor": "var(--color13)",
            "height": "100vh",  # Full viewport height
            "position": "sticky",
            "top": "0px",
            "overflowY": "auto",  # scroll if sidebar overflows
            "zIndex": 1,  # stay above content if needed
        },
    )


def get_sidebar():
    return html.Div(
        [
            html.H3(
                "Twist Analysis",
                className="mb-4",
                style={"text-align": "center", "marginTop": "1rem", "marginLeft": "0rem", "marginRight": "0rem"},
            ),
            dbc.Accordion(
                id="sidebar-accordion",
                flush=True,
                active_item="sacc-motl",
                style={"width": "100%"},
                children=[
                    dbc.AccordionItem(
                        [
                            # Motl loading operations
                            get_motl_load_component("main"),
                            dcc.Checklist(
                                ["Use as a nearest neighbor motl"],
                                ["Use as a nearest neighbor motl"],
                                inline=True,
                                id="use-nn-motl-checkbox",
                                inputStyle={"marginRight": "5px"},
                                className="sidebar-checklist",
                            ),
                            get_motl_load_component("nn", display_option="none"),
                        ],
                        title="Motl selection",
                        item_id="sacc-motl",
                    ),
                    dbc.AccordionItem(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        generate_form_from_docstring(
                                            "NearestNeighbors",
                                            id_type="nn-forms-params",
                                            id_name="nn-params",
                                            exclude_params=[
                                                "input_data",
                                            ],
                                            module_path="cryocat.nnana",
                                        ),
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            "Compute NN analysis",
                                            id="compute-nn-btn",
                                            color="light",
                                        ),
                                        width=12,
                                        className="d-grid gap-1 col-6 mx-auto mt-3",
                                    ),
                                ],
                                id="compute-nn-div",
                                style={"display": "block", "marginTop": "1rem"},
                            ),
                        ],
                        title="Nearest Neighbor",
                        item_id="sacc-nn",
                    ),
                    dbc.AccordionItem(
                        [
                            dcc.Upload(
                                id="upload-twist",
                                children=dbc.Button("Upload twist", color="light", className="upload-button"),
                                multiple=False,
                            ),
                            html.Div(
                                "or",
                                className="text-center mb-2",
                                style={"fontStyle": "italic", "color": "var(--color11)"},
                            ),
                            dbc.Row(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                generate_form_from_docstring(
                                                    "TwistDescriptor",
                                                    id_type="twist-forms-params",
                                                    id_name="twist-params",
                                                    exclude_params=[
                                                        "input_twist",
                                                        "input_motl",
                                                        "build_unique_desc",
                                                        "symm",
                                                    ],
                                                ),
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Label("Symmetry type", id="symmetry-label"),
                                                            dbc.Tooltip(
                                                                "Choose symmetry type (if any).",
                                                                target="symmetry-label",
                                                                placement="top",
                                                            ),
                                                        ],
                                                        style=label_style,
                                                    ),
                                                    html.Div(
                                                        dcc.Dropdown(
                                                            id="symmetry-dropdown",
                                                            options=[
                                                                "None",
                                                                "C",
                                                                "cube",
                                                                "tetrahedron",
                                                                "octahedron",
                                                                "icosahedron",
                                                                "dodecahedron",
                                                            ],
                                                            multi=False,
                                                            value="None",
                                                        ),
                                                        style=param_style,
                                                    ),
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "flexDirection": "row",
                                                    "boxSizing": "border-box",
                                                    "width": "100%",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Label("C-symmetry", id="c-symmetry-label"),
                                                            dbc.Tooltip(
                                                                "Choose C-symmetry value (should be bigger than 1).",
                                                                target="c-symmetry-label",
                                                                placement="top",
                                                            ),
                                                        ],
                                                        style=label_style,
                                                    ),
                                                    html.Div(
                                                        dcc.Input(
                                                            type="number", value=2, id="c-symmetry-value", min=2, step=1
                                                        ),
                                                        style=param_style,
                                                    ),
                                                ],
                                                style={
                                                    "display": "none",
                                                    "flexDirection": "row",
                                                    "boxSizing": "border-box",
                                                    "width": "100%",
                                                },
                                                id="c-symmetry-disp",
                                            ),
                                        ],
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            "Compute twist",
                                            id="run-twist-btn",
                                            color="light",
                                        ),
                                        width=12,
                                        className="d-grid gap-1 col-6 mx-auto mt-3",
                                    ),
                                ],
                                # className="align-items-center g-2 mb-2",
                                # className="d-grid gap-1 col-6 mx-auto",
                            ),  # g-2 adds spacing between cols,
                            html.Div(id="twist-status", className="small", style={"marginTop": "0.5rem"}),
                        ],
                        title="Twist Descriptor Base",
                        item_id="sacc-twist",
                    ),
                    dbc.AccordionItem(
                        [
                            html.H4("Support type"),
                            dcc.Dropdown(
                                id="support-dropdown",
                                options=supports,
                                multi=False,
                                placeholder="Choose support...",
                                style={"width": "100%"},
                            ),
                            html.Div(id="gen_support", style={"marginTop": "0.5rem", "marginBottom": "0.5rem"}),
                            html.H4("Descriptors:"),
                            dcc.Dropdown(
                                id="desc-dropdown",
                                options=descriptors,
                                multi=False,
                                placeholder="Choose features...",
                                style={"width": "100%"},
                            ),
                            html.Div(id="gen_desc", style={"marginTop": "0.5rem", "marginBottom": "0.5rem"}),
                            html.Div(
                                [
                                    html.H4("Features:", id="feat-title-label"),
                                    dcc.Dropdown(
                                        id="feat-multi-dropdown",
                                        options=features,
                                        multi=True,
                                        value="CustomDescriptor",
                                        placeholder="Choose features...",
                                        style={"width": "100%"},
                                    ),
                                ],
                                id="feat-menu-disp",
                                style={"width": "100%", "display": "none"},
                            ),
                            html.Div(id="gen_feat", style={"marginTop": "0.5rem", "marginBottom": "0.5rem"}),
                            html.Br(),
                            dbc.Button(
                                "Compute descriptors",
                                id="desc-run-btn",
                                color="light",
                                n_clicks=0,
                                style={"width": "100%"},
                            ),
                        ],
                        title="Unique descriptor",
                        item_id="sacc-options",
                    ),
                    dbc.AccordionItem(
                        [
                            html.H4("Input data"),
                            dcc.Dropdown(
                                id="cluster-data-dropdown",
                                options=["Twist descriptor base", "Unique descriptor"],
                                multi=False,
                                placeholder="Choose data...",
                                style={"width": "100%"},
                            ),
                            html.H4("Clustering types"),
                            dcc.Dropdown(
                                id="cluster-type-dropdown",
                                options=["K-means", "Proximity"],
                                multi=False,
                                placeholder="Choose clustering...",
                                style={"width": "100%"},
                            ),
                        ],
                        title="Clustering",
                        item_id="sacc-clustering",
                    ),
                ],
            ),
            html.Div(
                get_motl_save_component("save-main"),
                style={"marginTop": "auto", "paddingLeft": "2rem", "paddingRight": "2rem", "width": "100%"},
            ),
            html.Div(
                dbc.Button(
                    "Show log",
                    id="open-log-btn",
                    className="custom-radius-button",
                    style={"width": "100%"},
                ),
                style={"padding": "2rem"},
            ),
        ],
        className="sidebar",
        style={
            "width": "100%",
            "padding": "0px",
            "margin": "0",
            "display": "flex",
            "flexDirection": "column",
            "height": "100vh",
        },
    )


# Register callbacks for motl loads:
register_motl_load_callbacks("main")
register_motl_load_callbacks("nn")
register_motl_save_callbacks("save-main", stored_outputs, "tabv-motl-global-data-store", "main")


@callback(
    Output("nn-motl-tab", "disabled", allow_duplicate=True),
    Output("table-tabs", "active_tab", allow_duplicate=True),
    Output("tabv-twist-global-data-store", "data", allow_duplicate=True),
    Output("twist-global-radius", "data", allow_duplicate=True),
    Input("upload-twist", "contents"),
    prevent_initial_call=True,
)
def load_twist(upload_content):

    _, content_string = upload_content.split(",")
    decoded = base64.b64decode(content_string)

    try:
        # Read CSV into a DataFrame
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        global_twist["obj"] = TwistDescriptor(input_twist=pd.DataFrame(df))

        radius = int(np.ceil(df["euclidean_distance"].max()))

        return False, "twist-tab", df.to_dict("records"), radius
    except Exception as e:
        return True, no_update, "motl-tab", None


@callback(
    Output("sidebar-accordion", "active_item", allow_duplicate=True),
    Input("sidebar-accordion", "active_item"),
    State("main-motl-data-store", "data"),
    State("tabv-twist-global-data-store", "data"),
    prevent_initial_call=True,
)
def control_accordion(requested_item, motl_data, twist_data):

    if not motl_data and requested_item in ["sacc-nn", "sacc-twist", "sacc-options", "sacc-clustering"]:
        return None  # Block tabes without having motl first
    elif not twist_data and requested_item in ["sacc-options", "sacc-clustering"]:
        return None  # Block additional descriptors without having twist first

    return requested_item  # Allow it


# sidebar related callbacks
@callback(
    Output("nn-motl-container", "style"),
    Input("use-nn-motl-checkbox", "value"),
    prevent_initial_call=True,
)
def show_nn_motl_option(value):

    if value and "Use as a nearest neighbor motl" in value:
        return {"display": "none", "marginTop": "1rem"}
    else:
        return {"display": "block", "marginTop": "1rem"}


@callback(
    Output("table-tabs", "active_tab", allow_duplicate=True),
    Output("motl-visualization", "children"),
    Output("tabv-nn-global-data-store", "data", allow_duplicate=True),
    Output("nn-stats-print", "children"),
    Input("compute-nn-btn", "n_clicks"),
    State("main-motl-data-store", "data"),
    State("nn-motl-data-store", "data"),
    State("use-nn-motl-checkbox", "value"),
    State({"type": "nn-forms-params", "cls_name": ALL, "param": ALL, "p_type": ALL}, "value"),
    State({"type": "nn-forms-params", "cls_name": ALL, "param": ALL, "p_type": ALL}, "id"),
    prevent_initial_call=True,
)
def show_nn_motl_option(n_clicks, main_motl_df, nn_motl_df, use_single_motl, nn_values, nn_ids):

    if not main_motl_df:
        return html.Div("No main motl data available.")

    main_motl = Motl(pd.DataFrame(main_motl_df))

    nn_kwargs = generate_kwargs(nn_ids, nn_values)

    if use_single_motl and "Use as a nearest neighbor motl" in use_single_motl:
        nn_stats = NearestNeighbors(main_motl, **nn_kwargs)
    else:
        if not nn_motl_df:
            return html.Div("No nn motl data available.")

        nn_motl = Motl(pd.DataFrame(nn_motl_df))
        nn_stats = NearestNeighbors([main_motl, nn_motl], **nn_kwargs)

    _ = nn_stats.get_normalized_coord(add_to_df=True)

    if nn_kwargs["nn_type"] == "closest_dist":
        info_string = f"Mean distance: {nn_stats.df['nn_dist'].mean():.3f}; Median distance: {nn_stats.df['nn_dist'].median():.3f}"
    else:
        info_string = ""

    table_data = nn_stats.df.to_dict("records")

    # # Create subplot figure
    # fig = make_subplots(
    #     rows=1,
    #     cols=2,
    #     subplot_titles=["NN Distance", "Orientational Distribution", "Distance vs Angle"],
    #     vertical_spacing=0.15,
    # )

    # # Histogram
    # fig.add_trace(go.Histogram(x=nn_stats.df["nn_dist"], name="NN Distance"), row=1, col=1)

    # # Orientation plot as image
    # # fig.add_trace(go.Image(z=img), row=1, col=2)

    # # Distance vs Angle scatter
    # if "angular_distance" in nn_stats.df.columns:
    #     fig.add_trace(
    #         go.Scatter(
    #             x=nn_stats.df["nn_dist"], y=nn_stats.df["angular_distance"], mode="markers", name="Dist vs Angle"
    #         ),
    #         row=1,
    #         col=2,
    #     )

    # fig.update_layout(height=400, showlegend=False)

    norm_data = np.column_stack((nn_stats.get_normalized_coord(), nn_stats.df["nn_subtomo_id"].values))
    nn_df = pd.DataFrame(data=norm_data, columns=["x", "y", "z", "nn_subtomo_id"])
    fig = plot_nn_rot_coord_df_plotly(nn_df, ["x", "y", "z"], "nn_subtomo_id")

    return "nn-tab", dcc.Graph(figure=fig), table_data, info_string


@callback(
    Output("c-symmetry-disp", "style"),
    Input("symmetry-dropdown", "value"),
    State("c-symmetry-disp", "style"),
    prevent_initial_call=True,
)
def show_symmetry_input(symmetry, current_style):

    if symmetry == "C":
        return {**current_style, "display": "flex"}
    else:
        return {**current_style, "display": "none"}


@callback(
    Output("gen_support", "children"),
    Input("support-dropdown", "value"),
    prevent_initial_call=True,
)
def show_support_options(class_name):

    if not class_name:
        return []

    forms = generate_form_from_docstring(
        class_name, id_type="support-params", id_name=class_name, exclude_params=["twist_desc"]
    )
    return forms


@callback(
    Output("gen_desc", "children"),
    Output("feat-title-label", "children"),
    Output("feat-multi-dropdown", "options"),
    Output("feat-multi-dropdown", "value", allow_duplicate=True),
    Output("feat-menu-disp", "style"),
    Input("desc-dropdown", "value"),
    prevent_initial_call=True,
)
def show_desc_options(class_name):

    if not class_name:
        return [], "", [], [], {"width": "100%", "display": "none"}

    if class_name == "TwistDescriptor":
        forms = generate_form_from_docstring(
            class_name,
            id_type="desc-params",
            id_name=class_name,
            exclude_params=[
                "input_twist",
                "input_motl",
                "nn_radius",
                "feature_id",
                "symm",
                "remove_qp",
                "remove_duplicates",
                "build_unique_desc",
            ],
        )
    else:
        forms = generate_form_from_docstring(
            class_name, id_type="desc-params", id_name=class_name, exclude_params=["twist_df", "build_unique_desc"]
        )

    if class_name == "CustomDescriptor":
        feature_title = "Features"
        avail_features = features
        forms = []  # remove the parameters here, not will be fetched from elsewhere
    else:
        feature_title = "Additional features"
        ending = class_name.replace("Descriptor", "")
        avail_features = []
        for f in features:
            if not f.endswith(ending):
                avail_features.append(f)

    return forms, feature_title, avail_features, [], {"width": "100%", "display": "block"}


@callback(
    Output("gen_feat", "children"),
    # Output("feat-param-data-store", "data"),
    Input("feat-multi-dropdown", "value"),
    State("desc-dropdown", "value"),
    prevent_initial_call=True,
)
def show_feat_options(class_names, desc_class_name):

    if not class_names or not desc_class_name:
        return []

    forms = []
    used_desc_list = []

    for cls_name in class_names:
        if feat_desc_map[cls_name] == "TwistDescriptor":
            continue

        if feat_desc_map[cls_name] not in used_desc_list:
            forms.extend(
                generate_form_from_docstring(
                    feat_desc_map[cls_name],
                    id_type="feat-params",
                    id_name=cls_name,
                    exclude_params=["twist_df", "build_unique_desc"],
                )
            )
            used_desc_list.append(feat_desc_map[cls_name])

    return forms


@callback(
    Output("desc-tab", "disabled"),
    Output("table-tabs", "active_tab", allow_duplicate=True),
    Output("tabv-desc-global-data-store", "data"),
    Input("desc-run-btn", "n_clicks"),
    State("desc-dropdown", "value"),
    State("support-dropdown", "value"),
    State("feat-multi-dropdown", "value"),
    State("tabv-twist-global-data-store", "data"),
    State({"type": "support-params", "cls_name": ALL, "param": ALL, "p_type": ALL}, "value"),
    State({"type": "support-params", "cls_name": ALL, "param": ALL, "p_type": ALL}, "id"),
    State({"type": "desc-params", "cls_name": ALL, "param": ALL, "p_type": ALL}, "value"),
    State({"type": "desc-params", "cls_name": ALL, "param": ALL, "p_type": ALL}, "id"),
    State({"type": "feat-params", "cls_name": ALL, "param": ALL, "p_type": ALL}, "value"),
    State({"type": "feat-params", "cls_name": ALL, "param": ALL, "p_type": ALL}, "id"),
    prevent_initial_call=True,
)
def process_selection(
    n_clicks,
    selected_desc,
    selected_support,
    selected_features,
    twist_df,
    supp_values,
    supp_ids,
    desc_values,
    desc_ids,
    feat_values,
    feat_ids,
):
    if not selected_desc:
        return "No descritpors selected."

    twist_df = pd.DataFrame(twist_df)
    if not selected_support:
        support = twist_df
    else:
        support_kwargs = generate_kwargs(supp_ids, supp_values)
        supp_class = get_classes_from_names(selected_support, "cryocat.tango")
        support = supp_class(TwistDescriptor(input_twist=twist_df), **support_kwargs).support.df

    if selected_desc != "CustomDescriptor" and not selected_features:
        desc_kwargs = generate_kwargs(desc_ids, desc_values)
        desc = get_classes_from_names(selected_desc, "cryocat.tango")
        desc = desc(support, **desc_kwargs)
        table_data = desc.desc.to_dict("records")
    else:
        if selected_desc != "CustomDescriptor":
            all_features = selected_features + desc_feat_map[selected_desc]
            all_ids = desc_ids + feat_ids
            all_values = desc_values + feat_values
        else:
            all_features = selected_features
            all_ids = feat_ids
            all_values = feat_values

        feat_kwargs = []
        for feat_v in all_features:
            match_value, match_dict = next(
                (([all_values[i]], [d]) for i, d in enumerate(all_ids) if d.get("cls_name") == feat_v), (None, None)
            )
            if not match_value:
                feat_kwargs.append({})
            else:
                feat_kwargs.append(generate_kwargs(match_dict, match_value))

        desc = CustomDescriptor(support, feature_list=all_features, feature_kwargs=feat_kwargs)
        table_data = desc.desc.to_dict("records")

    return False, "desc-tab", table_data


@callback(
    Output("cluster-data-dropdown", "options"),
    Input("sidebar-accordion", "active_item"),
    State("tabv-twist-global-data-store", "data"),
    State("tabv-desc-global-data-store", "data"),
    prevent_initial_call=True,
)
def populate_cluster_data_dropdown(active_item, twist_data, desc_data):

    if active_item == "sacc-clustering":

        if not twist_data and desc_data:
            return []
        elif not desc_data:
            return ["Twist descriptor base"]
        else:
            return ["Twist descriptor base", "Unique descriptor"]

    raise PreventUpdate


@callback(
    Output("table-tabs", "active_tab", allow_duplicate=True),
    Output("cluster-tab", "disabled", allow_duplicate=True),
    Output("pca-visualization", "children"),
    Output("k-means-options", "options"),
    Input("cluster-data-dropdown", "value"),
    State("tabv-twist-global-data-store", "data"),
    State("tabv-desc-global-data-store", "data"),
    prevent_initial_call=True,
)
def compute_pca_analysis(cluster_data, twist_data, desc_data):

    twist_desc = TwistDescriptor(input_twist=pd.DataFrame(twist_data))

    if cluster_data == "Twist descriptor base":
        twist_desc.desc = twist_desc.df
    else:  # Unique descriptor
        twist_desc.desc = pd.DataFrame(desc_data)

    _, _, fig = twist_desc.pca_analysis(
        show_fig=False,
        scatter_kwargs={"mode": "lines+markers", "line": dict(color="#865B96")},
        bar_kwargs={"marker_color": "#865B96"},
    )

    if fig is None:
        raise PreventUpdate

    fig.update_layout(
        font=dict(size=10),
        margin=dict(l=20, r=20, t=30, b=20),
        height=200,
    )

    k_means_options = [col for col in twist_desc.desc.columns if col != "qp_id"]

    return "cluster-tab", False, dcc.Graph(figure=fig), k_means_options


@callback(
    Output("table-tabs", "active_tab", allow_duplicate=True),
    Output("cluster-tab", "disabled", allow_duplicate=True),
    Output("k-means-display", "style"),
    Output("prox-cluster-display", "style"),
    Output("k-means-graph-cont", "style", allow_duplicate=True),
    Output("proximity-graph-cont", "style", allow_duplicate=True),
    Input("cluster-type-dropdown", "value"),
    State("kmeans-global-data-store", "data"),
    State("merged-motl-proximity-data-store", "data"),
    prevent_initial_call=True,
)
def show_cluster_options(cluster_type, kmeans_data, prox_data):

    if not cluster_type:
        return PreventUpdate

    if cluster_type == "K-means":
        k_means_disp = "block"
        prox_disp = "none"
        prox_graph = "none"
        if not kmeans_data:
            k_means_graph = "none"
        else:
            k_means_graph = "block"
    else:
        k_means_disp = "none"
        prox_disp = "block"
        k_means_graph = "none"
        if not prox_data:
            prox_graph = "none"
        else:
            prox_graph = "block"

    return (
        "cluster-tab",
        False,
        {"display": k_means_disp},
        {"display": prox_disp},
        {"display": k_means_graph},
        {"display": prox_graph},
    )


@callback(
    Output("k-means-graph-cont", "style", allow_duplicate=True),
    Output("kmeans-global-data-store", "data", allow_duplicate=True),
    Output("tviewer-kmeans-data", "data", allow_duplicate=True),
    Output("merged-motl-kmeans-data-store", "data", allow_duplicate=True),
    Output("tviewer-kmeans-color-dropdown", "value"),
    Input("cluster-run-btn", "n_clicks"),
    State("cluster-data-dropdown", "value"),
    State("tabv-twist-global-data-store", "data"),
    State("tabv-desc-global-data-store", "data"),
    State("tabv-motl-global-data-store", "data"),
    State("k-means-options", "value"),
    State("k-means-n-slider", "value"),
    prevent_initial_call=True,
)
def compute_k_means(_, cluster_data, twist_data, desc_data, motl_data, cluster_options, n_clusters):

    twist_desc = TwistDescriptor(input_twist=pd.DataFrame(twist_data))

    if cluster_data == "Twist descriptor base":
        twist_desc.desc = twist_desc.df  # clustering is computed on desc, but here we want to use original base
    else:  # Unique descriptor
        twist_desc.desc = pd.DataFrame(desc_data)

    km_df = twist_desc.k_means_clustering(n_clusters, feature_ids=cluster_options)

    # Replace NaNs and shift by 1 -> unassigned cluster will have 0
    km_df["cluster"] = km_df["cluster"].fillna(-1).astype(int) + 1

    # Reduction - for the twist descriptor base case, where there can be non-unique qp_id
    if cluster_data == "Twist descriptor base":
        most_frequent = (
            km_df.groupby("qp_id")["cluster"].agg(lambda x: x.value_counts().idxmax()).reset_index()  # most frequent
        )
    else:
        most_frequent = km_df[["qp_id", "cluster"]]

    # merge with motl based on subtomo_id - qp_id correspondence
    motl_df = pd.DataFrame(motl_data)
    motl_df = motl_df.merge(most_frequent, left_on="subtomo_id", right_on="qp_id", how="left")
    motl_df["class"] = motl_df["cluster"]
    motl_df["class"] = motl_df["class"].fillna(0).astype(int)
    motl_df.drop(columns=["qp_id", "cluster"], inplace=True)  # drop the columns

    return (
        {"display": "block"},
        km_df.to_dict("records"),
        motl_df.to_dict("records"),
        motl_df.to_dict("records"),
        "class",
    )
