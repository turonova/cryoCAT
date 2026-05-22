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

from dash import Input, Output, State, html, dcc, ALL, ctx, no_update
from dash.dash_table.Format import Format, Scheme

from cryocat.app.apputils import format_columns
from cryocat.core.cryomotl import Motl
from cryocat.analysis.nnana import get_nn_stats
from cryocat.analysis.visplot import plot_orientational_distribution
from cryocat.analysis.tango import TwistDescriptor, FeatureCatalog, CustomDescriptor


def register_tango_callbacks(app):

    @app.callback(
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

        dc = CustomDescriptor.load(
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
