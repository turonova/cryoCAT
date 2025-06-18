from cryocat.app.logger import dash_logger

import base64
import os
import sys
import tempfile
import inspect
import io
import dash
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from dash import callback, Input, Output, State, html, dcc, ALL, ctx, no_update
from dash.dash_table.Format import Format, Scheme

from cryocat.app.apputils import format_columns
from cryocat.cryomotl import Motl  # or wherever your Motl class lives
from cryocat.nnana import get_nn_stats, plot_nn_rot_coord_df_plotly
from cryocat.visplot import plot_orientational_distribution
from cryocat.tango import TwistDescriptor, FeatureCatalog

global_motl = {"obj": None}
global_twist = {"obj": None}
global_nn = {"obj": None}
tomo_ids = None


@callback(Output("twist-ui", "style"), Input("upload-motl", "contents"), prevent_initial_call=True)
def show_twist_ui(contents):
    if contents:
        return {"display": "block", "marginTop": "1rem"}
    return {"display": "none"}


@callback(
    Output("motl-table", "data"), Input("main-motl-data-store", "data"), allow_duplicate=True, prevent_initial_call=True
)
def update_table_from_store(data):
    return data


@callback(
    Output("motl-table-title", "children"),
    Output("main-motl-data-store", "data", allow_duplicate=True),
    Input("apply-changes-btn", "n_clicks"),
    State("motl-table", "derived_virtual_data"),
    prevent_initial_call=True,
)
def apply_changes(n_clicks, filtered_data):
    if global_motl.get("obj") and filtered_data:
        df = pd.DataFrame(filtered_data)
        global_motl["obj"].df = df
        return f"MOTL Table {len(df)} rows applied", df.to_dict("records")
    return "MOTL Table No data", no_update


@callback(
    Output("main-motl-data-store", "data", allow_duplicate=True),
    Output("motl-table", "selected_rows", allow_duplicate=True),
    Input("remove-rows-btn", "n_clicks"),
    State("main-motl-data-store", "data"),
    State("motl-table", "selected_rows"),
    prevent_initial_call=True,
)
def remove_selected_rows(n_clicks, current_data, selected_rows):
    if not selected_rows:
        return no_update, [], "No rows selected"
    updated = [row for i, row in enumerate(current_data) if i not in selected_rows]
    return updated, []


@callback(
    Output("tviewer-desc-graph", "figure", allow_duplicate=True),
    Input("k-means-run-btn", "n_clicks"),
    State("k-means-options", "options"),
    State("k-means-n-slider", "value"),
    State("tabv-desc-global-data-store", "data"),
    State("main-motl-data-store", "data"),
    State("tviewer-desc-graph", "figure"),
    prevent_initial_call=True,
)
def compute_k_means(n_clicks, cluster_options, n_clusters, desc_data, motl_data, t_fig):

    dc = DescriptorCatalogue.load(
        pd.DataFrame(desc_data),
    )
    km_df = dc.k_means_clustering(n_clusters, feature_ids=cluster_options)

    motl_df = pd.DataFrame(motl_data)
    # Merge motl_df and cluster_df based on ID
    merged = motl_df.merge(
        km_df[["qp_id", "cluster"]],
        how="left",
        left_on="subtomo_id",
        right_on="qp_id",
    )

    # Fill NaN (meaning no cluster assigned) with -1
    merged["cluster"] = merged["cluster"].fillna(-1)

    # Now assign cluster values as color
    t_fig["data"][0]["marker"]["color"] = merged["cluster"].to_numpy()

    return t_fig
