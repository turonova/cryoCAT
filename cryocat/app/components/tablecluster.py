"""Generic table-level clustering panel.

``get_table_cluster_component(prefix)`` builds the offcanvas body.
``register_table_cluster_callbacks(app, prefix, connected_store_id, table_grid_id=None)``
wires all clustering callbacks, including an optional graph-to-table selection
sync (replace / add / subtract) that mirrors the plot panel behaviour.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA as _PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import dash
from dash import html, dcc, ctx, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from cryocat.analysis import clustering as clustering_mod
from cryocat.analysis import visplot


def get_table_cluster_component(prefix: str):
    lbl = {"fontWeight": "bold", "marginBottom": "0.3rem", "fontSize": "0.85rem"}
    return html.Div(
        children=[
            dcc.Store(id=f"{prefix}-cluster-data-store"),
            html.Div(
                [
                    html.Label("Clustering type:", style=lbl),
                    dcc.Dropdown(
                        id=f"{prefix}-cluster-type-dropdown",
                        options=["K-means", "Proximity"],
                        value=None,
                        placeholder="Choose type…",
                        clearable=True,
                        searchable=False,
                        style={"marginBottom": "0.75rem"},
                    ),
                ]
            ),
            # ── K-means options ───────────────────────────────────────────────
            html.Div(
                id=f"{prefix}-cluster-kmeans-opts",
                style={"display": "none"},
                children=[
                    html.Div(id=f"{prefix}-cluster-pca-area", style={"marginBottom": "0.5rem"}),
                    html.Label("Features to cluster:", style=lbl),
                    dcc.Checklist(
                        id=f"{prefix}-cluster-features-check",
                        options=[],
                        value=[],
                        style={"columnCount": 3, "columnGap": "0.5rem"},
                        labelStyle={"display": "block", "marginBottom": "0.2rem", "breakInside": "avoid"},
                        inputStyle={"marginRight": "5px"},
                    ),
                    html.Label("Number of clusters:", style={**lbl, "marginTop": "0.5rem"}),
                    dcc.Slider(
                        id=f"{prefix}-cluster-n-slider",
                        min=2,
                        max=20,
                        step=1,
                        value=2,
                        tooltip={"placement": "right"},
                        marks=None,
                    ),
                    dbc.Button(
                        "Run K-means",
                        id=f"{prefix}-cluster-run-btn",
                        color="primary",
                        size="sm",
                        style={"width": "100%", "marginTop": "0.5rem"},
                    ),
                ],
            ),
            # ── K-means scatter + selection mode ──────────────────────────────
            html.Div(
                id=f"{prefix}-cluster-scatter-cont",
                style={"display": "none"},
                children=[
                    html.Div(
                        [
                            dcc.Dropdown(
                                id=f"{prefix}-cluster-xaxis",
                                placeholder="X axis",
                                style={"flex": "1"},
                            ),
                            dcc.Dropdown(
                                id=f"{prefix}-cluster-yaxis",
                                placeholder="Y axis",
                                style={"flex": "1"},
                            ),
                        ],
                        style={"display": "flex", "gap": "0.5rem", "marginBottom": "0.25rem"},
                    ),
                    dcc.Graph(id=f"{prefix}-cluster-scatter", figure={}),
                    dbc.RadioItems(
                        id=f"{prefix}-cluster-selection-mode",
                        options=[
                            {"label": "Replace selection", "value": "replace"},
                            {"label": "Add to selection", "value": "add"},
                            {"label": "Subtract from selection", "value": "subtract"},
                        ],
                        value="replace",
                        inline=True,
                        className="sidebar-checklist",
                        labelStyle={"color": "var(--color9)", "marginRight": "1rem"},
                        style={"marginTop": "0.5rem"},
                    ),
                ],
            ),
            # ── Proximity options ─────────────────────────────────────────────
            html.Div(
                id=f"{prefix}-cluster-prox-opts",
                style={"display": "none"},
                children=[
                    html.Label("Query ID column:", style=lbl),
                    dcc.Dropdown(
                        id=f"{prefix}-cluster-prox-qp-col",
                        placeholder="Query ID column…",
                        style={"marginBottom": "0.4rem"},
                    ),
                    html.Label("Neighbor ID column:", style=lbl),
                    dcc.Dropdown(
                        id=f"{prefix}-cluster-prox-nn-col",
                        placeholder="Neighbor ID column…",
                        style={"marginBottom": "0.4rem"},
                    ),
                    html.Label("Number of components:", style=lbl),
                    dcc.Slider(
                        id=f"{prefix}-cluster-numcomp-slider",
                        min=1,
                        max=50,
                        step=1,
                        value=1,
                        tooltip={"placement": "right"},
                        marks=None,
                    ),
                    html.Label("Or minimum component size:", style={**lbl, "marginTop": "0.5rem"}),
                    dcc.Slider(
                        id=f"{prefix}-cluster-minsize-slider",
                        min=0,
                        max=100,
                        step=1,
                        value=0,
                        tooltip={"placement": "right"},
                        marks=None,
                    ),
                    dbc.Button(
                        "Run Proximity",
                        id=f"{prefix}-cluster-prox-run-btn",
                        color="primary",
                        size="sm",
                        style={"width": "100%", "marginTop": "0.5rem"},
                    ),
                ],
            ),
            html.Div(
                id=f"{prefix}-cluster-status",
                style={
                    "fontSize": "0.85rem",
                    "color": "var(--color9)",
                    "marginTop": "0.5rem",
                    "wordBreak": "break-word",
                },
            ),
        ]
    )


def register_table_cluster_callbacks(app, prefix: str, connected_store_id: str, table_grid_id=None):

    # ── Type selection: show/hide panels, populate features + PCA ────────────

    @app.callback(
        Output(f"{prefix}-cluster-kmeans-opts", "style"),
        Output(f"{prefix}-cluster-prox-opts", "style"),
        Output(f"{prefix}-cluster-pca-area", "children"),
        Output(f"{prefix}-cluster-features-check", "options"),
        Output(f"{prefix}-cluster-features-check", "value"),
        Output(f"{prefix}-cluster-prox-qp-col", "options"),
        Output(f"{prefix}-cluster-prox-nn-col", "options"),
        Input(f"{prefix}-cluster-type-dropdown", "value"),
        State(connected_store_id, "data"),
        prevent_initial_call=True,
    )
    def _select_cluster_type(cluster_type, data):
        show = {"display": "block"}
        hide = {"display": "none"}

        if not cluster_type:
            return hide, hide, no_update, no_update, no_update, no_update, no_update

        is_kmeans = cluster_type == "K-means"

        if not data:
            return (
                show if is_kmeans else hide,
                hide if is_kmeans else show,
                no_update, no_update, no_update, no_update, no_update,
            )

        df = pd.DataFrame(data)
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        all_cols = list(df.columns)

        col_options = [{"label": c, "value": c} for c in all_cols]
        feat_options = [{"label": c, "value": c} for c in numeric_cols]
        pca_children = no_update

        if is_kmeans and len(numeric_cols) > 1:
            df_feat = df[numeric_cols].dropna()
            if len(df_feat) > 1:
                try:
                    pca = _PCA()
                    pca.fit(df_feat)
                    cumulative = np.cumsum(pca.explained_variance_ratio_)
                    imp = clustering_mod.pca_feature_importance(pca, numeric_cols)
                    fig = visplot.plot_pca_summary(cumulative, imp)
                    fig.update_layout(
                        font=dict(size=10),
                        margin=dict(l=20, r=20, t=30, b=20),
                        height=200,
                    )
                    pca_children = dcc.Graph(figure=fig)
                except Exception as exc:
                    pca_children = html.Div(
                        f"PCA failed: {exc}",
                        style={"fontSize": "0.8rem", "color": "var(--color9)"},
                    )

        return (
            show if is_kmeans else hide,
            hide if is_kmeans else show,
            pca_children,
            feat_options if is_kmeans else no_update,
            [],  # all unchecked
            col_options if not is_kmeans else no_update,
            col_options if not is_kmeans else no_update,
        )

    # ── Run K-means ───────────────────────────────────────────────────────────

    @app.callback(
        Output(f"{prefix}-cluster-scatter-cont", "style"),
        Output(f"{prefix}-cluster-xaxis", "options"),
        Output(f"{prefix}-cluster-yaxis", "options"),
        Output(f"{prefix}-cluster-data-store", "data"),
        Output(f"{prefix}-cluster-status", "children"),
        Input(f"{prefix}-cluster-run-btn", "n_clicks"),
        State(f"{prefix}-cluster-features-check", "value"),
        State(f"{prefix}-cluster-n-slider", "value"),
        State(connected_store_id, "data"),
        prevent_initial_call=True,
    )
    def _run_kmeans(n_clicks, features, n_clusters, data):
        if not n_clicks or not data:
            raise dash.exceptions.PreventUpdate
        if not features:
            return no_update, no_update, no_update, no_update, "Select at least one feature."

        df = pd.DataFrame(data)

        # Keep only rows where all selected features are non-NaN
        valid_mask = df[features].notna().all(axis=1)
        df_valid = df[valid_mask].copy().reset_index(drop=True)
        orig_indices = df[valid_mask].index.tolist()

        if len(df_valid) < int(n_clusters):
            return (
                no_update, no_update, no_update, no_update,
                f"Too few valid rows ({len(df_valid)}) for {n_clusters} clusters.",
            )

        X = df_valid[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        labels = KMeans(n_clusters=int(n_clusters), n_init="auto").fit_predict(X_scaled)

        result_df = df_valid[features].copy()
        result_df["cluster"] = labels
        result_df["__row_idx__"] = orig_indices

        axis_opts = [{"label": c, "value": c} for c in features]
        return (
            {"display": "block"},
            axis_opts,
            axis_opts,
            result_df.to_dict("records"),
            f"K-means complete — {int(n_clusters)} clusters, {len(df_valid)} points.",
        )

    # ── K-means scatter (re-renders when axes change) ─────────────────────────

    @app.callback(
        Output(f"{prefix}-cluster-scatter", "figure"),
        Input(f"{prefix}-cluster-xaxis", "value"),
        Input(f"{prefix}-cluster-yaxis", "value"),
        State(f"{prefix}-cluster-data-store", "data"),
        prevent_initial_call=True,
    )
    def _update_scatter(x_col, y_col, data):
        if not data or not x_col or not y_col or x_col == y_col:
            raise dash.exceptions.PreventUpdate
        df = pd.DataFrame(data)
        fig = px.scatter(df, x=x_col, y=y_col, color=df["cluster"].astype(str))
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=20, b=20),
            dragmode="select",
            legend_title_text="cluster",
        )
        return fig

    # ── Proximity clustering ──────────────────────────────────────────────────

    @app.callback(
        Output(f"{prefix}-cluster-status", "children", allow_duplicate=True),
        Input(f"{prefix}-cluster-prox-run-btn", "n_clicks"),
        State(f"{prefix}-cluster-prox-qp-col", "value"),
        State(f"{prefix}-cluster-prox-nn-col", "value"),
        State(f"{prefix}-cluster-numcomp-slider", "value"),
        State(f"{prefix}-cluster-minsize-slider", "value"),
        State(connected_store_id, "data"),
        prevent_initial_call=True,
    )
    def _run_proximity(n_clicks, qp_col, nn_col, num_comps, min_size, data):
        if not n_clicks or not data:
            raise dash.exceptions.PreventUpdate
        if not qp_col or not nn_col:
            return "Select query ID and neighbor ID columns."

        df = pd.DataFrame(data)
        if qp_col not in df.columns or nn_col not in df.columns:
            return f"Columns '{qp_col}' / '{nn_col}' not found in data."

        use_min_size = int(min_size) > 0
        try:
            comps = clustering_mod.connected_component_clusters(
                df[qp_col],
                df[nn_col],
                num_components=int(num_comps),
                min_size=int(min_size) if use_min_size else None,
            )
        except Exception as exc:
            return f"Proximity clustering failed: {exc}"

        sizes = sorted([len(g.nodes) for g in comps], reverse=True)
        preview = sizes[:10]
        suffix = "…" if len(sizes) > 10 else ""
        return f"{len(comps)} component(s). Sizes: {preview}{suffix}"

    # ── Graph → table selection sync ─────────────────────────────────────────

    if table_grid_id is not None:

        @app.callback(
            Output(table_grid_id, "selectedRows", allow_duplicate=True),
            Input(f"{prefix}-cluster-scatter", "clickData"),
            Input(f"{prefix}-cluster-scatter", "selectedData"),
            State(f"{prefix}-cluster-data-store", "data"),
            State(table_grid_id, "rowData"),
            State(f"{prefix}-cluster-selection-mode", "value"),
            State(table_grid_id, "selectedRows"),
            prevent_initial_call=True,
        )
        def _sync_selection(click_data, sel_data, cluster_data, row_data, sel_mode, current_selected):
            data_value = ctx.triggered[0]["value"] if ctx.triggered else None
            if not data_value or not data_value.get("points"):
                raise dash.exceptions.PreventUpdate
            if not cluster_data or not row_data:
                raise dash.exceptions.PreventUpdate

            points = data_value["points"]
            indices = [p["pointIndex"] for p in points if "pointIndex" in p]

            new_rows = []
            n = len(row_data)
            for cluster_idx in indices:
                if 0 <= cluster_idx < len(cluster_data):
                    row_idx = cluster_data[cluster_idx].get("__row_idx__")
                    if row_idx is not None and 0 <= row_idx < n:
                        new_rows.append(row_data[row_idx])

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
