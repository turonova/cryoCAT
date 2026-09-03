"""Generic table-level clustering panel.

``get_table_cluster_component(prefix)`` builds the offcanvas body.
``register_table_cluster_callbacks(app, prefix, connected_store_id, ...)``
wires all clustering callbacks.

Contract
--------
Stores owned:
  ``{prefix}-cluster-data-store``   — serialised cluster-result DataFrame
  ``{prefix}-cluster-kdist-store``  — k-distance curve as JSON list
  ``{prefix}-cluster-screen-store`` — {excluded, warnings, defaults, columns};
                                      computed once per table load, invalidated by
                                      method-select or connected-store change

Pattern ids emitted:
  ``{"type": "cluster-feat-row",     "owner": prefix, "col": <column>}`` — per-column checkbox
  ``{"type": "cluster-method-param", "owner": prefix, "param": ..., "tag": ..., "method": <method>}``
    — method-specific form controls generated from function signatures
"""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import dash
from dash import html, dcc, ctx, Input, Output, State, no_update, ALL
import dash_bootstrap_components as dbc

from cryocat.analysis import clustering as clustering_mod
from cryocat.app import formgen, styles
from cryocat.app.formgen import make_dropdown


# ── Feature-selector helpers ─────────────────────────────────────────────────

def _build_feature_rows(
    prefix: str,
    df: pd.DataFrame,
    defaults: list[str],
    excluded: dict[str, str] | None = None,
    warnings: dict[str, str] | None = None,
) -> list:
    """One dbc.Checklist row per column, with screening and identifier badges.

    Badge rules:
    - ``identifier``: column excluded by ``default_feature_columns``; always shown.
    - screen reason: from ``screen_feature_columns`` (non-numeric, constant, etc.).
    - near-constant: amber warning badge.
    - A column with NO badge is one the user manually deselected — nothing else.
    When a column has both reasons (e.g. ``geom3`` = identifier AND all-zero),
    both badges appear side-by-side.
    """
    if excluded is None or warnings is None:
        excluded, warnings = clustering_mod.screen_feature_columns(df)
    default_set = set(defaults)
    rows = []
    for col in df.columns:
        screen_reason = excluded.get(col, "")
        warn = warnings.get(col, "")
        is_identifier = col not in default_set
        disabled = screen_reason == "non-numeric"
        checked = not is_identifier and not screen_reason
        badges = []
        if is_identifier:
            badges.append(html.Span("identifier", style={**styles.HINT_SM, "marginLeft": "0.4rem"}))
        if screen_reason:
            badges.append(html.Span(screen_reason, style={**styles.HINT_SM, "marginLeft": "0.4rem"}))
        if warn:
            badges.append(html.Span(
                warn,
                style={**styles.HINT_SM, "marginLeft": "0.4rem", "color": "var(--bs-warning)"},
            ))
        rows.append(html.Div(
            [
                dbc.Checklist(
                    options=[{"label": col, "value": col, "disabled": disabled}],
                    value=[col] if checked else [],
                    id={"type": "cluster-feat-row", "owner": prefix, "col": col},
                    inputStyle={"verticalAlign": "middle", "marginTop": "-2px"},
                    labelStyle={"verticalAlign": "middle"},
                ),
                *badges,
            ],
            style={"display": "flex", "alignItems": "center"},
        ))
    return rows


def _get_checked_features(feat_row_values: list) -> list[str]:
    """Collect checked column names from pattern-matched per-column Checklist values."""
    result = []
    for v in feat_row_values:
        if isinstance(v, list):
            result.extend(v)
    return result


def _compute_feature_breakdown(
    features: list[str],
    defaults: list[str],
    excluded_screen: dict[str, str],
    all_columns: list[str],
) -> str:
    """Format 'clustered on X of Y columns; excluded N identifiers, M constant, …'.

    Takes pre-computed primitives so this function can be called without re-reading
    the DataFrame on every toggle.  Pass the values from the screen-store.

    Identifier = column excluded by ``default_feature_columns_fn`` but not by
    ``screen_feature_columns`` (i.e. it is usable but not a meaningful feature
    by convention).  Screen reasons are reported verbatim.  Columns that are in
    ``defaults`` but manually deselected by the user are silently omitted.
    """
    from collections import Counter
    default_set = set(defaults)
    feature_set = set(features)
    not_used = [c for c in all_columns if c not in feature_set]
    reason_counts: Counter = Counter()
    for c in not_used:
        if c in excluded_screen:
            reason_counts[excluded_screen[c]] += 1
        elif c not in default_set:
            reason_counts["identifier"] += 1
    n_total = len(all_columns)
    n_used = len(features)
    line = f"clustered on {n_used} of {n_total} columns"
    excl_parts = []
    for reason in ["identifier", "non-numeric", "constant", "all NaN", "all zero"]:
        count = reason_counts.get(reason, 0)
        if count:
            label = f"identifier{'s' if count != 1 else ''}" if reason == "identifier" else reason
            excl_parts.append(f"{count} {label}")
    if excl_parts:
        line += "; excluded " + ", ".join(excl_parts)
    return line


# ── K-distance curve helpers ─────────────────────────────────────────────────

def _build_kdist_figure(kdist: np.ndarray, min_samples: int, eps: float | None) -> go.Figure:
    """Build the k-distance curve figure with an optional eps horizontal line."""
    fig = go.Figure()
    x = list(range(len(kdist)))
    fig.add_scatter(
        x=x, y=kdist.tolist(), mode="lines", name="k-distance",
        line=dict(width=1.5),
    )
    if eps is not None:
        fig.add_hline(
            y=eps, line_dash="dash",
            line_color=styles.COLOR_POSITIVE,
            annotation_text=f"eps = {eps}",
            annotation_position="bottom right",
        )
    fig.update_layout(
        xaxis_title="Points (sorted by distance)",
        yaxis_title=f"Distance to {min_samples}th neighbour",
        height=150,
        margin=dict(l=40, r=10, t=10, b=30),
        showlegend=False,
        uirevision="kdist",
    )
    return fig


# ── Scatter helpers ──────────────────────────────────────────────────────────

# Fixed palette for real clusters; noise always gets neutral grey.
_CLUSTER_PALETTE = [
    "#457b9d", "#e63946", "#2a9d8f", "#f4a261", "#264653",
    "#a8dadc", "#e9c46a", "#f77f00", "#023e8a", "#8338ec",
]


def _build_cluster_scatter(df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    """Scatter with -1 coloured neutral grey and real clusters in a fixed palette.

    Each point carries ``customdata=[[row_idx]]`` so the selection-sync
    callback can read the original DataFrame row index from ``customdata[0]``
    rather than from ``pointIndex`` (which is trace-local in multi-trace figures).
    """
    labels = df["cluster"].astype(int)
    row_idx_col = df["__row_idx__"].astype(int) if "__row_idx__" in df.columns else pd.Series(range(len(df)))
    real_clusters = sorted(c for c in labels.unique() if c != -1)
    noise_mask = labels == -1

    fig = go.Figure()
    for i, cl in enumerate(real_clusters):
        mask = labels == cl
        fig.add_scatter(
            x=df.loc[mask, x_col].tolist(),
            y=df.loc[mask, y_col].tolist(),
            customdata=[[int(idx)] for idx in row_idx_col[mask]],
            mode="markers",
            marker=dict(color=_CLUSTER_PALETTE[i % len(_CLUSTER_PALETTE)]),
            name=str(cl),
        )
    if noise_mask.any():
        fig.add_scatter(
            x=df.loc[noise_mask, x_col].tolist(),
            y=df.loc[noise_mask, y_col].tolist(),
            customdata=[[int(idx)] for idx in row_idx_col[noise_mask]],
            mode="markers",
            marker=dict(color="rgba(128,128,128,0.45)"),
            name="noise",
        )
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=20, b=20),
        dragmode="select",
        legend_title_text="cluster",
        uirevision="cluster-scatter",
    )
    return fig


def _cluster_status(labels: np.ndarray) -> str:
    """Format E2-compliant status: 'N clusters, M noise points (P%)'.

    -1 is never counted as a cluster.  0 clusters + all noise is a
    legitimate result and is reported plainly.
    """
    noise = int((labels == -1).sum())
    real = sorted(set(int(v) for v in labels if v != -1))
    n_real = len(real)
    total = len(labels)
    if n_real == 0:
        return f"0 clusters, {noise:,} noise points — all-noise result."
    s = "s" if n_real != 1 else ""
    if noise == 0:
        return f"{n_real} cluster{s}, {total:,} points."
    pct = noise / total * 100
    return f"{n_real} cluster{s}, {noise:,} noise points ({pct:.1f}%)."


# ── Serialisation helpers ────────────────────────────────────────────────────

def _cluster_df_to_store(df: pd.DataFrame) -> list[dict]:
    return df.to_dict("records")


def _cluster_store_to_df(data) -> pd.DataFrame:
    return pd.DataFrame(data)


# ── Layout ───────────────────────────────────────────────────────────────────

def get_table_cluster_component(prefix: str) -> html.Div:
    """Return the clustering panel layout for ``prefix``."""
    return html.Div([
        dcc.Store(id=f"{prefix}-cluster-data-store"),
        dcc.Store(id=f"{prefix}-cluster-kdist-store"),
        dcc.Store(id=f"{prefix}-cluster-screen-store"),

        # ── Method selector ───────────────────────────────────────────────────
        formgen.form_row(
            "method",
            make_dropdown(
                f"{prefix}-cluster-type-dropdown",
                [
                    {"label": "K-means",   "value": "K-means"},
                    {"label": "DBSCAN",    "value": "DBSCAN"},
                    {"label": "Proximity", "value": "Proximity"},
                ],
                None,
                clearable=True,
                placeholder="Choose method…",
            ),
            "Clustering algorithm to apply to the descriptor data",
            label_id=f"{prefix}-cluster-method-lbl",
            label_text="Method",
        ),

        # ── Feature selector (K-means and DBSCAN) ────────────────────────────
        html.Div(
            id=f"{prefix}-cluster-feat-section",
            style={"display": "none"},
            children=[
                html.Div(
                    [
                        html.Span("Features", style=styles.SECTION_HEADER),
                        html.Div(
                            [
                                dbc.Button("all",      id=f"{prefix}-cluster-feat-all",          size="sm", color=styles.BTN_NEUTRAL, style={"padding": "1px 7px"}),
                                dbc.Button("none",     id=f"{prefix}-cluster-feat-none",         size="sm", color=styles.BTN_NEUTRAL, style={"padding": "1px 7px"}),
                                dbc.Button("defaults", id=f"{prefix}-cluster-feat-defaults-btn", size="sm", color=styles.BTN_NEUTRAL, style={"padding": "1px 7px"}),
                            ],
                            style={"display": "flex", "gap": "0.25rem"},
                        ),
                    ],
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "0.2rem"},
                ),
                html.Div(id=f"{prefix}-cluster-feat-rows"),
                html.Div(id=f"{prefix}-cluster-feat-summary", style=styles.FORM_HINT),
            ],
        ),

        # ── Method-specific param form (generated from signature by callback) ──
        html.Div(id=f"{prefix}-cluster-method-form"),

        # ── K-distance curve (DBSCAN only) ────────────────────────────────────
        html.Div(
            id=f"{prefix}-cluster-kdist-section",
            style={"display": "none"},
            children=[
                dcc.Graph(id=f"{prefix}-cluster-kdist-graph", figure={}, style={"height": "150px"}),
                html.Div(
                    "The knee is the conventional starting point for eps.",
                    style=styles.FORM_HINT,
                ),
            ],
        ),

        # ── Run button (K-means / DBSCAN) ─────────────────────────────────────
        html.Div(
            id=f"{prefix}-cluster-run-wrap",
            style={"display": "none"},
            children=[
                dbc.Button(
                    "Run clustering",
                    id=f"{prefix}-cluster-run-btn",
                    color=styles.BTN_PRIMARY,
                    size="sm",
                    style={"width": "100%", "marginTop": "0.5rem"},
                ),
            ],
        ),

        # ── Proximity options ─────────────────────────────────────────────────
        html.Div(
            id=f"{prefix}-cluster-prox-opts",
            style={"display": "none"},
            children=[
                formgen.form_row(
                    "qp_col",
                    make_dropdown(f"{prefix}-cluster-prox-qp-col", [], None, placeholder="Query ID column…"),
                    "Column containing query-particle IDs",
                    label_id=f"{prefix}-cluster-prox-qp-lbl",
                    label_text="Query ID col",
                ),
                formgen.form_row(
                    "nn_col",
                    make_dropdown(f"{prefix}-cluster-prox-nn-col", [], None, placeholder="Neighbor ID column…"),
                    "Column containing nearest-neighbor IDs",
                    label_id=f"{prefix}-cluster-prox-nn-lbl",
                    label_text="Neighbor ID col",
                ),
                formgen.form_row(
                    "n_comp",
                    dcc.Slider(id=f"{prefix}-cluster-numcomp-slider", min=1, max=50, step=1, value=1, tooltip={"placement": "right"}, marks=None),
                    "Number of largest connected components to return",
                    label_id=f"{prefix}-cluster-prox-ncomp-lbl",
                    label_text="Num components",
                ),
                formgen.form_row(
                    "min_size",
                    dcc.Slider(id=f"{prefix}-cluster-minsize-slider", min=0, max=100, step=1, value=0, tooltip={"placement": "right"}, marks=None),
                    "Return all components at least this large (overrides num components when > 0)",
                    label_id=f"{prefix}-cluster-prox-minsize-lbl",
                    label_text="Min size",
                ),
                dbc.Button(
                    "Run Proximity",
                    id=f"{prefix}-cluster-prox-run-btn",
                    color=styles.BTN_PRIMARY,
                    size="sm",
                    style={"width": "100%", "marginTop": "0.5rem"},
                ),
            ],
        ),

        # ── Status ────────────────────────────────────────────────────────────
        html.Div(
            id=f"{prefix}-cluster-status",
            style={**styles.HINT, "wordBreak": "break-word", "marginTop": "0.5rem"},
        ),

        # ── Scatter + E2 cluster selection ────────────────────────────────────
        html.Div(
            id=f"{prefix}-cluster-scatter-cont",
            style={"display": "none"},
            children=[
                html.Div(
                    [
                        make_dropdown(f"{prefix}-cluster-xaxis", [], None, placeholder="X axis", style={"flex": "1"}),
                        make_dropdown(f"{prefix}-cluster-yaxis", [], None, placeholder="Y axis", style={"flex": "1"}),
                    ],
                    style={"display": "flex", "gap": "0.5rem", "marginBottom": "0.25rem"},
                ),
                dcc.Graph(id=f"{prefix}-cluster-scatter", figure={}),
                formgen.form_row(
                    "selection_mode",
                    dbc.RadioItems(
                        id=f"{prefix}-cluster-selection-mode",
                        options=[
                            {"label": "Replace",  "value": "replace"},
                            {"label": "Add",      "value": "add"},
                            {"label": "Subtract", "value": "subtract"},
                        ],
                        value="replace",
                        inline=True,
                        inputStyle=styles.RADIO_INLINE_INPUT,
                        labelStyle=styles.RADIO_INLINE_LABEL,
                    ),
                    "How scatter selection affects table row selection",
                    label_id=f"{prefix}-cluster-selmode-lbl",
                    label_text="Select mode",
                ),
                html.Hr(style={"margin": "0.3rem 0"}),
                # E2: pick a cluster label (including -1 shown as "noise") and select those rows
                formgen.form_row(
                    "cluster_label",
                    make_dropdown(
                        f"{prefix}-cluster-label-dd",
                        [],
                        None,
                        clearable=True,
                        placeholder="Pick cluster…",
                    ),
                    "Select a cluster label to highlight those rows in the table. "
                    "Label -1 is shown as 'noise' and is selectable like any cluster.",
                    label_id=f"{prefix}-cluster-label-lbl",
                    label_text="Select cluster",
                ),
            ],
        ),

        # ── Save cluster assignments ──────────────────────────────────────────
        html.Div(
            id=f"{prefix}-cluster-save-wrap",
            style={"display": "none"},
            children=[
                html.Hr(style={"margin": "0.5rem 0"}),
                formgen.form_row(
                    "save_col",
                    dbc.Input(
                        id=f"{prefix}-cluster-save-colname",
                        placeholder="Column name (e.g. cluster1)",
                        size="sm",
                    ),
                    "Column name to write cluster assignments into the data table. "
                    "Label -1 (noise) is written as-is and never remapped.",
                    label_id=f"{prefix}-cluster-save-lbl",
                    label_text="Save as column",
                ),
                dbc.Button(
                    "Save to table",
                    id=f"{prefix}-cluster-save-btn",
                    color=styles.BTN_SECONDARY,
                    size="sm",
                    style={"width": "100%"},
                ),
            ],
        ),
    ])


# ── Callbacks ────────────────────────────────────────────────────────────────

def register_table_cluster_callbacks(
    app,
    prefix: str,
    connected_store_id: str,
    table_grid_id: str | None = None,
    cluster_cols_store_id: str | None = None,
    pool_aware: bool = False,
    resolve_df: Callable | None = None,
    default_feature_columns_fn: Callable | None = None,
) -> None:
    """Register all clustering callbacks for one instance.

    Parameters
    ----------
    app:
        The Dash application.
    prefix:
        Must match the prefix passed to :func:`get_table_cluster_component`.
    connected_store_id:
        Id of the store whose data the clustering operates on.
    table_grid_id:
        Optional id of the AgGrid whose ``selectedRows`` is synced from the scatter.
    cluster_cols_store_id:
        Optional id of a store that accumulates written cluster column names.
    pool_aware:
        When True, save also writes back to the motl pool.
    resolve_df:
        Callable ``(store_data) -> pd.DataFrame | None`` for pool-backed stores.
    default_feature_columns_fn:
        Optional ``(df) -> list[str]`` returning the names of columns that are
        meaningful features for the data type in this panel.  Falls back to all
        numeric columns excluding common id columns when not supplied.
    """

    def _df_from_store(data) -> pd.DataFrame:
        if resolve_df is not None:
            df = resolve_df(data)
            return df if df is not None else pd.DataFrame()
        if not data or not isinstance(data, list):
            return pd.DataFrame()
        return pd.DataFrame.from_records(data)

    def _get_defaults(df: pd.DataFrame) -> list[str]:
        if default_feature_columns_fn is not None:
            try:
                return default_feature_columns_fn(df)
            except Exception:
                pass
        skip = {"qp_id", "nn_id", "__row_idx__", "subtomo_id"}
        excluded, _ = clustering_mod.screen_feature_columns(df)
        return [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in skip and c not in excluded
        ]

    # ── Method selection ──────────────────────────────────────────────────────

    @app.callback(
        Output(f"{prefix}-cluster-feat-section",  "style"),
        Output(f"{prefix}-cluster-feat-rows",      "children"),
        Output(f"{prefix}-cluster-method-form",    "children"),
        Output(f"{prefix}-cluster-kdist-section",  "style"),
        Output(f"{prefix}-cluster-run-wrap",       "style"),
        Output(f"{prefix}-cluster-run-btn",        "children"),
        Output(f"{prefix}-cluster-prox-opts",      "style"),
        Output(f"{prefix}-cluster-prox-qp-col",    "options"),
        Output(f"{prefix}-cluster-prox-nn-col",    "options"),
        Output(f"{prefix}-cluster-screen-store",   "data"),
        Input(f"{prefix}-cluster-type-dropdown", "value"),
        State(connected_store_id, "data"),
        prevent_initial_call=True,
    )
    def _select_method(method, data):
        show = {"display": "block"}
        hide = {"display": "none"}
        is_km   = method == "K-means"
        is_db   = method == "DBSCAN"
        is_prox = method == "Proximity"
        show_feat = is_km or is_db
        col_opts, feat_rows, screen_data = [], [], None
        if data:
            df = _df_from_store(data)
            col_opts = [{"label": c, "value": c} for c in df.columns]
            if show_feat:
                excluded, warnings = clustering_mod.screen_feature_columns(df)
                defaults = _get_defaults(df)
                feat_rows = _build_feature_rows(prefix, df, defaults, excluded, warnings)
                screen_data = {
                    "excluded": excluded,
                    "warnings": warnings,
                    "defaults": defaults,
                    "columns": list(df.columns),
                }
        method_form = []
        if is_km:
            method_form = formgen.build_form(
                clustering_mod.kmeans_cluster,
                id_type="cluster-method-param",
                id_extra={"owner": prefix, "method": "kmeans"},
                exclude=["input_df", "feature_ids", "id_columns", "nan_drop", "pca_dict"],
            )
        elif is_db:
            method_form = formgen.build_form(
                clustering_mod.dbscan_cluster,
                id_type="cluster-method-param",
                id_extra={"owner": prefix, "method": "dbscan"},
                exclude=["input_df", "feature_ids", "id_columns", "nan_drop", "pca_dict"],
            )
        btn_lbl = "Run DBSCAN" if is_db else "Run K-means" if is_km else "Run"
        return (
            show if show_feat else hide,
            feat_rows, method_form,
            show if is_db else hide,
            show if (is_km or is_db) else hide,
            btn_lbl,
            show if is_prox else hide,
            col_opts, col_opts,
            screen_data,
        )

    # ── Refresh screen-store when the connected data changes ─────────────────────
    # Called when the underlying table is replaced; updates screen-store and
    # feat-rows so the feature selector reflects the new data columns.
    # Only fires when a feature-based method (K-means / DBSCAN) is active.

    @app.callback(
        Output(f"{prefix}-cluster-screen-store", "data",     allow_duplicate=True),
        Output(f"{prefix}-cluster-feat-rows",    "children", allow_duplicate=True),
        Input(connected_store_id, "data"),
        State(f"{prefix}-cluster-type-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _refresh_screen_on_data_change(data, method):
        show_feat = method in ("K-means", "DBSCAN")
        if not data or not show_feat:
            raise dash.exceptions.PreventUpdate
        df = _df_from_store(data)
        if df.empty:
            raise dash.exceptions.PreventUpdate
        excluded, warnings = clustering_mod.screen_feature_columns(df)
        defaults = _get_defaults(df)
        feat_rows = _build_feature_rows(prefix, df, defaults, excluded, warnings)
        screen_data = {
            "excluded": excluded,
            "warnings": warnings,
            "defaults": defaults,
            "columns": list(df.columns),
        }
        return screen_data, feat_rows

    # ── Feature [all] / [none] / [defaults] buttons ───────────────────────────
    # States: 0=screen-store (scalar), 1=feat-row options (ALL pattern-matched)
    # Col name for each row is read from ctx.states_list[1][i]["id"]["col"].

    @app.callback(
        Output({"type": "cluster-feat-row", "owner": prefix, "col": ALL}, "value"),
        Input(f"{prefix}-cluster-feat-all",          "n_clicks"),
        Input(f"{prefix}-cluster-feat-none",         "n_clicks"),
        Input(f"{prefix}-cluster-feat-defaults-btn", "n_clicks"),
        State(f"{prefix}-cluster-screen-store", "data"),
        State({"type": "cluster-feat-row", "owner": prefix, "col": ALL}, "options"),
        prevent_initial_call=True,
    )
    def _feat_buttons(all_n, none_n, defaults_n, screen_data, _options_list):
        trigger = ctx.triggered_id
        default_cols = screen_data.get("defaults", []) if screen_data else []
        default_set = set(default_cols)
        result = []
        # ctx.states_list[1]: list of {"id": {"col": ...}, "value": [options]} for each row
        for entry in ctx.states_list[1]:
            col = entry["id"]["col"]
            opt_list = entry["value"]
            disabled = any(o.get("disabled") for o in (opt_list or []))
            if disabled:
                result.append([])
                continue
            if trigger == f"{prefix}-cluster-feat-all":
                result.append([col])
            elif trigger == f"{prefix}-cluster-feat-none":
                result.append([])
            else:
                result.append([col] if col in default_set else [])
        return result

    # ── Feature summary line ──────────────────────────────────────────────────

    @app.callback(
        Output(f"{prefix}-cluster-feat-summary", "children"),
        Input({"type": "cluster-feat-row", "owner": prefix, "col": ALL}, "value"),
        State(f"{prefix}-cluster-screen-store", "data"),
        prevent_initial_call=True,
    )
    def _feat_summary(values, screen_data):
        checked = _get_checked_features(values)
        if not checked:
            return "No features selected."
        if not screen_data:
            return f"Clustering on {len(checked)} columns."
        excluded = screen_data.get("excluded", {})
        defaults = screen_data.get("defaults", [])
        all_columns = screen_data.get("columns", [])
        if not all_columns:
            return f"Clustering on {len(checked)} columns."
        breakdown = _compute_feature_breakdown(checked, defaults, excluded, all_columns)
        return breakdown.capitalize() + "."

    # ── K-distance curve (recomputes on min_samples or feature-selection change) ─
    # Uses ALL-pattern matching so the callback tolerates the case where the DBSCAN
    # form has not yet been rendered (all_values will be empty or contain only
    # K-means params; we extract min_samples by inspecting the ids).

    @app.callback(
        Output(f"{prefix}-cluster-kdist-store", "data"),
        Output(f"{prefix}-cluster-kdist-graph", "figure"),
        Input({"type": "cluster-method-param", "owner": prefix, "param": ALL, "tag": ALL, "method": ALL}, "value"),
        Input({"type": "cluster-feat-row", "owner": prefix, "col": ALL}, "value"),
        State(connected_store_id, "data"),
        State(f"{prefix}-cluster-type-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _update_kdist(all_param_values, feat_values, data, method):
        # If triggered solely by eps change, let _update_eps_line handle it.
        tid = ctx.triggered_id
        if isinstance(tid, dict) and tid.get("param") == "eps" and tid.get("method") == "dbscan":
            raise dash.exceptions.PreventUpdate
        if method != "DBSCAN" or not data:
            raise dash.exceptions.PreventUpdate
        features = _get_checked_features(feat_values)
        if not features:
            raise dash.exceptions.PreventUpdate
        all_param_ids = ctx.inputs_list[0]
        ms, eps_val = 5, None
        for entry, val in zip(all_param_ids, all_param_values):
            pid = entry["id"]
            if pid.get("method") == "dbscan":
                if pid.get("param") == "min_samples" and val is not None:
                    ms = int(val)
                elif pid.get("param") == "eps" and val is not None:
                    eps_val = float(val)
        df = _df_from_store(data)
        try:
            kdist = clustering_mod.k_distance_curve(df, min_samples=ms, feature_ids=features)
            return kdist.tolist(), _build_kdist_figure(kdist, ms, eps_val)
        except Exception as exc:
            err_fig = go.Figure()
            err_fig.update_layout(title_text=f"k-distance failed: {exc}", height=150)
            return no_update, err_fig

    # ── Eps line update (only the horizontal line moves; curve unchanged) ─────

    @app.callback(
        Output(f"{prefix}-cluster-kdist-graph", "figure", allow_duplicate=True),
        Input({"type": "cluster-method-param", "owner": prefix, "param": ALL, "tag": ALL, "method": ALL}, "value"),
        State(f"{prefix}-cluster-kdist-store", "data"),
        State(f"{prefix}-cluster-type-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _update_eps_line(all_param_values, kdist_data, method):
        # Only act when eps specifically triggered this callback.
        tid = ctx.triggered_id
        if not isinstance(tid, dict) or tid.get("param") != "eps" or tid.get("method") != "dbscan":
            raise dash.exceptions.PreventUpdate
        if method != "DBSCAN" or not kdist_data:
            raise dash.exceptions.PreventUpdate
        all_param_ids = ctx.inputs_list[0]
        eps_val, ms = None, 5
        for entry, val in zip(all_param_ids, all_param_values):
            pid = entry["id"]
            if pid.get("method") == "dbscan":
                if pid.get("param") == "eps" and val is not None:
                    eps_val = float(val)
                elif pid.get("param") == "min_samples" and val is not None:
                    ms = int(val)
        kdist = np.array(kdist_data)
        return _build_kdist_figure(kdist, ms, eps_val)

    # ── Run K-means or DBSCAN ─────────────────────────────────────────────────
    # States: 0=method (scalar), 1=feat-row values (ALL pattern-matched),
    #         2=method-param values (ALL pattern-matched), 3=connected-store (scalar),
    #         4=screen-store (scalar)
    # ctx.states_list[2] is used below to extract method-param ids alongside values.

    @app.callback(
        Output(f"{prefix}-cluster-scatter-cont", "style"),
        Output(f"{prefix}-cluster-xaxis",        "options"),
        Output(f"{prefix}-cluster-yaxis",        "options"),
        Output(f"{prefix}-cluster-data-store",   "data"),
        Output(f"{prefix}-cluster-status",       "children"),
        Output(f"{prefix}-cluster-save-wrap",    "style"),
        Output(f"{prefix}-cluster-label-dd",     "options"),
        Input(f"{prefix}-cluster-run-btn", "n_clicks"),
        State(f"{prefix}-cluster-type-dropdown", "value"),
        State({"type": "cluster-feat-row",     "owner": prefix, "col": ALL}, "value"),
        State({"type": "cluster-method-param", "owner": prefix, "param": ALL, "tag": ALL, "method": ALL}, "value"),
        State(connected_store_id, "data"),
        State(f"{prefix}-cluster-screen-store", "data"),
        prevent_initial_call=True,
    )
    def _run_clustering(n_clicks, method, feat_values, param_values, data, screen_data):
        show = {"display": "block"}
        hide = {"display": "none"}
        if not n_clicks or not data or not method:
            raise dash.exceptions.PreventUpdate
        features = _get_checked_features(feat_values)
        if not features:
            return hide, no_update, no_update, no_update, "Select at least one feature.", hide, no_update
        df = _df_from_store(data)
        try:
            from cryocat.app.apputils import generate_kwargs, run_operation
            from cryocat.app import session as _session
            from cryocat.app.event import message_event as _message_event
            # Retrieve the pattern-matched ids from ctx (index 2 = the method-param State)
            param_entries = ctx.states_list[2] if len(ctx.states_list) > 2 else []
            param_ids = [e["id"] for e in param_entries]
            active_method_key = "kmeans" if method == "K-means" else "dbscan"
            active_ids  = [d for d in param_ids  if d.get("method") == active_method_key]
            active_vals = [param_values[param_ids.index(d)] for d in active_ids]
            kwargs = generate_kwargs(active_ids, active_vals) if active_ids else {}

            if method == "K-means":
                n_clusters = int(kwargs.get("n_clusters", 2))
                scale_data = kwargs.get("scale_data", True)
                if isinstance(scale_data, str):
                    scale_data = scale_data == "True"
                result_df = run_operation(
                    clustering_mod.kmeans_cluster,
                    {"input_df": df, "n_clusters": n_clusters, "feature_ids": features, "scale_data": scale_data},
                )
            elif method == "DBSCAN":
                eps = float(kwargs.get("eps") or 0.5)
                min_samples = int(kwargs.get("min_samples", 5))
                metric = str(kwargs.get("metric") or "euclidean")
                result_df = run_operation(
                    clustering_mod.dbscan_cluster,
                    {"input_df": df, "eps": eps, "min_samples": min_samples, "feature_ids": features, "metric": metric},
                )
            else:
                raise dash.exceptions.PreventUpdate

            # Record feature-column breakdown in the session stream using cached screen data.
            if screen_data:
                excluded = screen_data.get("excluded", {})
                defaults = screen_data.get("defaults", [])
                all_columns = screen_data.get("columns", list(df.columns))
            else:
                excluded, _ = clustering_mod.screen_feature_columns(df)
                defaults = _get_defaults(df)
                all_columns = list(df.columns)
            breakdown = _compute_feature_breakdown(features, defaults, excluded, all_columns)
            _session.emit(_message_event(breakdown))

        except dash.exceptions.PreventUpdate:
            raise
        except Exception as exc:
            return hide, no_update, no_update, no_update, f"Error: {exc}", hide, no_update

        # Map back original row indices via qp_id (if available) or positional
        if "qp_id" in result_df.columns and "qp_id" in df.columns:
            id_to_orig = {v: i for i, v in enumerate(df["qp_id"].tolist())}
            result_df["__row_idx__"] = result_df["qp_id"].map(id_to_orig).astype(int)
        else:
            result_df["__row_idx__"] = result_df.index.astype(int)

        labels = result_df["cluster"].astype(int).values
        status = _cluster_status(labels)

        feat_cols = [c for c in result_df.columns if c not in {"cluster", "__row_idx__", "qp_id"}]
        axis_opts = [{"label": c, "value": c} for c in feat_cols]

        # E2: populate cluster-label dropdown; -1 shown as "noise"
        unique_labels = sorted(set(int(v) for v in labels))
        label_opts = [
            {"label": "noise" if lbl == -1 else str(lbl), "value": lbl}
            for lbl in unique_labels
        ]
        return show, axis_opts, axis_opts, _cluster_df_to_store(result_df), status, show, label_opts

    # ── Scatter (re-renders when axes or result data changes) ─────────────────

    @app.callback(
        Output(f"{prefix}-cluster-scatter", "figure"),
        Input(f"{prefix}-cluster-xaxis",      "value"),
        Input(f"{prefix}-cluster-yaxis",      "value"),
        Input(f"{prefix}-cluster-data-store", "data"),
        prevent_initial_call=True,
    )
    def _update_scatter(x_col, y_col, data):
        if not data or not x_col or not y_col or x_col == y_col:
            raise dash.exceptions.PreventUpdate
        df = _cluster_store_to_df(data)
        if "cluster" not in df.columns:
            raise dash.exceptions.PreventUpdate
        return _build_cluster_scatter(df, x_col, y_col)

    # ── Proximity clustering ──────────────────────────────────────────────────
    # Assigns component labels into a cluster-data-store-compatible DataFrame so
    # the same save path as K-means / DBSCAN works without any extra code.
    # Rows whose qp_id belongs to no returned component get cluster label -1.

    @app.callback(
        Output(f"{prefix}-cluster-status",       "children",  allow_duplicate=True),
        Output(f"{prefix}-cluster-data-store",   "data",      allow_duplicate=True),
        Output(f"{prefix}-cluster-scatter-cont", "style",     allow_duplicate=True),
        Output(f"{prefix}-cluster-xaxis",        "options",   allow_duplicate=True),
        Output(f"{prefix}-cluster-yaxis",        "options",   allow_duplicate=True),
        Output(f"{prefix}-cluster-save-wrap",    "style",     allow_duplicate=True),
        Output(f"{prefix}-cluster-label-dd",     "options",   allow_duplicate=True),
        Input(f"{prefix}-cluster-prox-run-btn", "n_clicks"),
        State(f"{prefix}-cluster-prox-qp-col",    "value"),
        State(f"{prefix}-cluster-prox-nn-col",    "value"),
        State(f"{prefix}-cluster-numcomp-slider", "value"),
        State(f"{prefix}-cluster-minsize-slider", "value"),
        State(connected_store_id, "data"),
        prevent_initial_call=True,
    )
    def _run_proximity(n_clicks, qp_col, nn_col, num_comps, min_size, data):
        show = {"display": "block"}

        if not n_clicks or not data:
            raise dash.exceptions.PreventUpdate
        if not qp_col or not nn_col:
            return ("Select query ID and neighbor ID columns.",) + (no_update,) * 6
        df = _df_from_store(data)
        if qp_col not in df.columns or nn_col not in df.columns:
            return (f"Columns '{qp_col}' / '{nn_col}' not found in data.",) + (no_update,) * 6

        use_min_size = int(min_size or 0) > 0
        try:
            comps = clustering_mod.connected_component_clusters(
                df[qp_col], df[nn_col],
                num_components=int(num_comps),
                min_size=int(min_size) if use_min_size else None,
            )
        except Exception as exc:
            return (f"Proximity clustering failed: {exc}",) + (no_update,) * 6

        # Map node → component label (0-based); rows not in any component get -1.
        node_to_label: dict = {}
        for lbl_idx, graph in enumerate(comps):
            for node_id in graph.nodes:
                node_to_label[node_id] = lbl_idx

        result_df = df.copy()
        result_df["cluster"] = [node_to_label.get(v, -1) for v in df[qp_col]]
        result_df["__row_idx__"] = list(range(len(df)))

        labels = result_df["cluster"].astype(int).values
        sizes = sorted([len(g.nodes) for g in comps], reverse=True)
        size_preview = sizes[:10]
        suffix = "…" if len(sizes) > 10 else ""
        status = _cluster_status(labels) + f" Component sizes: {size_preview}{suffix}"

        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in {qp_col, nn_col, "cluster", "__row_idx__"}
        ]
        axis_opts = [{"label": c, "value": c} for c in numeric_cols]

        unique_labels = sorted(set(int(v) for v in labels))
        label_opts = [
            {"label": "noise" if lbl == -1 else str(lbl), "value": lbl}
            for lbl in unique_labels
        ]

        return (
            status,
            _cluster_df_to_store(result_df),
            show, axis_opts, axis_opts,
            show,
            label_opts,
        )

    # ── Save cluster assignments to table ─────────────────────────────────────
    # -1 (noise) is written as-is; no remapping, no row dropping.

    _save_out: list = []
    if pool_aware:
        from cryocat.app import ids as _ids
        _save_out += [
            Output(_ids.POOL_REGISTRY, "data", allow_duplicate=True),
            Output(_ids.POOL_META,     "data", allow_duplicate=True),
            Output(_ids.POOL_NEXT_ID,  "data", allow_duplicate=True),
        ]
    _save_out += [
        Output(connected_store_id,          "data",     allow_duplicate=True),
        Output(f"{prefix}-cluster-status",  "children", allow_duplicate=True),
    ]
    if cluster_cols_store_id:
        _save_out.append(Output(cluster_cols_store_id, "data", allow_duplicate=True))

    _save_states: list = [
        State(f"{prefix}-cluster-data-store",    "data"),
        State(connected_store_id,                "data"),
        State(f"{prefix}-cluster-save-colname",  "value"),
    ]
    if cluster_cols_store_id:
        _save_states.append(State(cluster_cols_store_id, "data"))
    if pool_aware:
        from cryocat.app import ids as _ids
        _save_states += [
            State(_ids.POOL_REGISTRY, "data"),
            State(_ids.POOL_META,     "data"),
            State(_ids.POOL_NEXT_ID,  "data"),
        ]

    @app.callback(
        *_save_out,
        Input(f"{prefix}-cluster-save-btn", "n_clicks"),
        *_save_states,
        prevent_initial_call=True,
    )
    def _save_cluster(n_clicks, cluster_data, main_data, col_name, *extra):
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

        if not n_clicks or not cluster_data or not main_data:
            raise dash.exceptions.PreventUpdate

        if not col_name:
            nu = [no_update] * n_pool_out + [no_update, "Enter a column name first."]
            if cluster_cols_store_id:
                nu.append(no_update)
            return tuple(nu) if n_out > 1 else nu[0]

        df = _df_from_store(main_data).copy()
        cluster_df = _cluster_store_to_df(cluster_data)
        df[col_name] = np.nan
        for rec in cluster_df.to_dict("records"):
            idx = int(rec.get("__row_idx__", -1))
            if 0 <= idx < len(df):
                df.iloc[idx, df.columns.get_loc(col_name)] = int(rec["cluster"])

        status = f"Saved cluster assignments to column '{col_name}'."

        if pool_aware:
            from cryocat.app import pool as _pool
            pool_reg, pool_meta_out, pool_next_id, new_ref = _pool.commit_rows(
                main_data, df, registry, pool_meta, next_id
            )
            # Record the cluster column in pool.meta so it survives cross-tab plotting.
            # Only possible for motl-pool refs (table-pool commit_rows returns no_update for meta).
            motl_id = (new_ref or {}).get("motl_id") if isinstance(new_ref, dict) else None
            if motl_id and isinstance(pool_meta_out, dict):
                cols = list(existing_cols or [])
                if col_name not in cols:
                    cols.append(col_name)
                existing_motl_meta = (pool_meta_out.get(motl_id) or {})
                pool_meta_out = {
                    **pool_meta_out,
                    motl_id: {**existing_motl_meta, "cluster_cols": cols},
                }
            result = [pool_reg, pool_meta_out, pool_next_id, new_ref, status]
            if cluster_cols_store_id:
                cols = list(existing_cols or [])
                if col_name not in cols:
                    cols.append(col_name)
                result.append(cols)
            return tuple(result)

        new_data = _cluster_df_to_store(df)
        if cluster_cols_store_id:
            cols = list(existing_cols or [])
            if col_name not in cols:
                cols.append(col_name)
            return new_data, status, cols
        return new_data, status

    # ── Table-grid interactions (optional) ───────────────────────────────────

    if table_grid_id is not None:

        # E2: select rows of a chosen cluster label (including -1 = noise)
        @app.callback(
            Output(table_grid_id, "selectedRows", allow_duplicate=True),
            Input(f"{prefix}-cluster-label-dd", "value"),
            State(f"{prefix}-cluster-data-store", "data"),
            State(connected_store_id, "data"),
            prevent_initial_call=True,
        )
        def _select_by_cluster_label(cluster_label, cluster_data, store_data):
            if cluster_label is None or not cluster_data:
                raise dash.exceptions.PreventUpdate
            cluster_df = _cluster_store_to_df(cluster_data)
            row_records = _df_from_store(store_data).to_dict("records")
            selected = []
            for rec in cluster_df.to_dict("records"):
                if int(rec.get("cluster", -999)) == int(cluster_label):
                    idx = int(rec.get("__row_idx__", -1))
                    if 0 <= idx < len(row_records):
                        selected.append(row_records[idx])
            return selected

        # Scatter → table selection sync; reads customdata[0] (row_idx) not pointIndex
        @app.callback(
            Output(table_grid_id, "selectedRows", allow_duplicate=True),
            Input(f"{prefix}-cluster-scatter",  "clickData"),
            Input(f"{prefix}-cluster-scatter",  "selectedData"),
            State(f"{prefix}-cluster-data-store", "data"),
            State(connected_store_id,            "data"),
            State(f"{prefix}-cluster-selection-mode", "value"),
            State(table_grid_id,                 "selectedRows"),
            prevent_initial_call=True,
        )
        def _sync_selection(click_data, sel_data, cluster_data, store_data, sel_mode, current):
            event = ctx.triggered[0]["value"] if ctx.triggered else None
            if not event or not event.get("points"):
                raise dash.exceptions.PreventUpdate
            if not cluster_data:
                raise dash.exceptions.PreventUpdate

            row_records = _df_from_store(store_data).to_dict("records")
            n = len(row_records)
            new_rows = []
            for p in event["points"]:
                cd = p.get("customdata")
                if cd is None:
                    continue
                row_idx = cd[0] if isinstance(cd, (list, tuple)) else cd
                if 0 <= int(row_idx) < n:
                    new_rows.append(row_records[int(row_idx)])

            if not new_rows and sel_mode != "subtract":
                raise dash.exceptions.PreventUpdate

            current = current or []
            if sel_mode == "replace":
                return new_rows
            if sel_mode == "add":
                existing = {json.dumps(r, sort_keys=True) for r in current}
                merged = list(current)
                for r in new_rows:
                    if json.dumps(r, sort_keys=True) not in existing:
                        merged.append(r)
                return merged
            if sel_mode == "subtract":
                remove = {json.dumps(r, sort_keys=True) for r in new_rows}
                return [r for r in current if json.dumps(r, sort_keys=True) not in remove]
            return new_rows
