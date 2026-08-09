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
from cryocat.app.formgen import make_dropdown


def get_table_cluster_component(prefix: str, is_motl=False, motl_cols=None):
    lbl = {"fontWeight": "bold", "marginBottom": "0.3rem"}
    return html.Div(
        children=[
            dcc.Store(id=f"{prefix}-cluster-data-store"),
            html.Div(
                [
                    html.Label("Clustering type:", style=lbl),
                    make_dropdown(
                        f"{prefix}-cluster-type-dropdown",
                        ["K-means", "Proximity"],
                        None,
                        clearable=True,
                        placeholder="Choose type…",
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
                            make_dropdown(
                                f"{prefix}-cluster-xaxis",
                                [],
                                None,
                                placeholder="X axis",
                                style={"flex": "1"},
                            ),
                            make_dropdown(
                                f"{prefix}-cluster-yaxis",
                                [],
                                None,
                                placeholder="Y axis",
                                style={"flex": "1"},
                            ),
                        ],
                        style={**{"display": "flex", "gap": "0.5rem"}, "marginBottom": "0.25rem"},
                    ),
                    dcc.Graph(id=f"{prefix}-cluster-scatter", figure={}),
                    html.Div(
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
                        ),
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
                    make_dropdown(
                        f"{prefix}-cluster-prox-qp-col",
                        [],
                        None,
                        placeholder="Query ID column…",
                        style={"marginBottom": "0.4rem"},
                    ),
                    html.Label("Neighbor ID column:", style=lbl),
                    make_dropdown(
                        f"{prefix}-cluster-prox-nn-col",
                        [],
                        None,
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
                    "color": "var(--color9)",
                    "marginTop": "0.5rem",
                    "wordBreak": "break-word",
                },
            ),
        html.Div(
            id=f"{prefix}-cluster-save-wrap",
            style={"display": "none"},
            children=[
                html.Hr(style={"margin": "0.5rem 0"}),
                html.Label("Save cluster assignments to table:", style=lbl),
                html.Div(
                    dbc.Input(
                        id=f"{prefix}-cluster-save-colname",
                        placeholder="Column name (e.g. cluster1)",
                        size="sm",
                        style={"marginBottom": "0.3rem"},
                    ),
                    style={"display": "none" if is_motl else "block"},
                ),
                html.Div(
                    make_dropdown(
                        f"{prefix}-cluster-save-motlcol",
                        [{"label": c, "value": c} for c in (motl_cols or [])],
                        "class" if (motl_cols and "class" in motl_cols) else (motl_cols[0] if motl_cols else None),
                        clearable=False,
                        style={"marginBottom": "0.3rem"},
                    ),
                    style={"display": "block" if is_motl else "none"},
                ),
                dbc.Button(
                    "Save to table",
                    id=f"{prefix}-cluster-save-btn",
                    color="secondary",
                    size="sm",
                    style={"width": "100%"},
                ),
            ],
        ),
        ]
    )


def register_table_cluster_callbacks(app, prefix: str, connected_store_id: str, table_grid_id=None, is_motl=False, motl_cols=None, cluster_cols_store_id=None, pool_aware=False):

    def _df_from_store(data):
        """Return a DataFrame from a pool reference or a list[dict]."""
        if pool_aware and isinstance(data, dict) and "motl_id" in data:
            from cryocat.app.pool import get_rows, PoolPayloadMissing
            try:
                return get_rows(data["motl_id"])
            except PoolPayloadMissing:
                return pd.DataFrame()
        return pd.DataFrame(data) if data else pd.DataFrame()

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

        df = _df_from_store(data)
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
                        style={"color": "var(--color9)"},
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
        Output(f"{prefix}-cluster-save-wrap", "style"),
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
            return no_update, no_update, no_update, no_update, "Select at least one feature.", no_update

        df = _df_from_store(data)

        # Keep only rows where all selected features are non-NaN
        valid_mask = df[features].notna().all(axis=1)
        df_valid = df[valid_mask].copy().reset_index(drop=True)
        orig_indices = df[valid_mask].index.tolist()

        if len(df_valid) < int(n_clusters):
            return (
                no_update, no_update, no_update, no_update,
                f"Too few valid rows ({len(df_valid)}) for {n_clusters} clusters.",
                no_update,
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
            {"display": "block"},
        )

    # ── K-means scatter (re-renders when axes change) ─────────────────────────

    @app.callback(
        Output(f"{prefix}-cluster-scatter", "figure"),
        Input(f"{prefix}-cluster-xaxis", "value"),
        Input(f"{prefix}-cluster-yaxis", "value"),
        Input(f"{prefix}-cluster-data-store", "data"),
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

        df = _df_from_store(data)
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

    # ── Save K-means cluster assignments to the data table ────────────────────

    _save_out = []
    if pool_aware:
        from cryocat.app import ids as _ids
        _save_out += [
            Output(_ids.POOL_REGISTRY, "data", allow_duplicate=True),
            Output(_ids.POOL_META, "data", allow_duplicate=True),
            Output(_ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        ]
    _save_out += [
        Output(connected_store_id, "data", allow_duplicate=True),
        Output(f"{prefix}-cluster-status", "children", allow_duplicate=True),
    ]
    if cluster_cols_store_id:
        _save_out.append(Output(cluster_cols_store_id, "data", allow_duplicate=True))

    _save_states = [
        State(f"{prefix}-cluster-data-store", "data"),
        State(connected_store_id, "data"),
        State(f"{prefix}-cluster-save-colname", "value"),
        State(f"{prefix}-cluster-save-motlcol", "value"),
    ]
    if cluster_cols_store_id:
        _save_states.append(State(cluster_cols_store_id, "data"))
    if pool_aware:
        from cryocat.app import ids as _ids
        _save_states += [
            State(_ids.POOL_REGISTRY, "data"),
            State(_ids.POOL_META, "data"),
            State(_ids.POOL_NEXT_ID, "data"),
        ]

    @app.callback(
        *_save_out,
        Input(f"{prefix}-cluster-save-btn", "n_clicks"),
        *_save_states,
        prevent_initial_call=True,
    )
    def _save_cluster(n_clicks, cluster_data, main_data, col_name, motl_col, *extra):
        # unpack extra positional args based on mode
        if pool_aware:
            if cluster_cols_store_id:
                existing_cols, registry, pool_meta, next_id = extra
            else:
                registry, pool_meta, next_id = extra
                existing_cols = None
        else:
            existing_cols = extra[0] if extra else None
            registry = pool_meta = next_id = None

        n_pool_out = 3 if pool_aware else 0
        n_out = len(_save_out)

        def _ret(*vals):
            return vals[0] if n_out == 1 else vals

        if not n_clicks or not cluster_data or not main_data:
            raise dash.exceptions.PreventUpdate

        col_target = motl_col if is_motl else col_name
        if not col_target:
            nu = [no_update] * n_pool_out + [no_update, "Enter a column name first."]
            if cluster_cols_store_id:
                nu.append(no_update)
            return tuple(nu) if n_out > 1 else nu[0]

        df = _df_from_store(main_data)
        cluster_df = pd.DataFrame(cluster_data)

        df = df.copy()
        df[col_target] = np.nan
        for _, row in cluster_df.iterrows():
            idx = int(row["__row_idx__"])
            if 0 <= idx < len(df):
                df.loc[idx, col_target] = int(row["cluster"])

        status = f"Saved cluster assignments to column '{col_target}'."

        if pool_aware:
            from cryocat.app.pool import replace_motl_rows, PoolState
            motl_id = main_data["motl_id"]
            state = PoolState.from_stores(registry, pool_meta, next_id)
            state = replace_motl_rows(state, motl_id, df)
            new_rev = state.registry[motl_id]["revision"]
            new_ref = {"motl_id": motl_id, "rev": new_rev}
            result = [*state.to_stores(), new_ref, status]
            if cluster_cols_store_id:
                cols = list(existing_cols or [])
                if col_target not in cols:
                    cols.append(col_target)
                result.append(cols)
            return tuple(result)

        new_data = df.to_dict("records")
        if cluster_cols_store_id:
            cols = list(existing_cols or [])
            if col_target not in cols:
                cols.append(col_target)
            return new_data, status, cols

        return new_data, status

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
