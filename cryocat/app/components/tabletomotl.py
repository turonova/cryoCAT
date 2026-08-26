"""Shared "table → motl" sidebar component.

``get_table_to_motl(prefix, *, allow_modal=True)`` builds the layout.
``register_table_to_motl_callbacks(app, prefix, *, source_table_id, id_column="qp_id")``
wires the callbacks.

The component merges on ``subtomo_id`` (one-to-one), optionally writes a value
column from the source table into a motl column, and can create a new clean-subset
motl from the current filter/selection.
"""

from __future__ import annotations

import pandas as pd

from dash import html, dcc, ctx, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from cryocat.core.cryomotl import Motl
from cryocat.app import ids as _ids
from cryocat.app import styles
from cryocat.app import formgen
from cryocat.app.formgen import make_dropdown
from cryocat.app.pool import PoolState, insert_motl as _insert_motl, get_rows as _get_rows, PoolPayloadMissing


_NU3 = (no_update, no_update, no_update)


def _do_write_col(target_id, val_col, dst_col, id_column, active_rows, registry, pool_meta, next_id):
    """Write val_col from active_rows into dst_col of target motl."""
    from cryocat.app.pool import replace_motl_rows
    try:
        motl_rows = _get_rows(target_id)
    except PoolPayloadMissing:
        return "Target motl not found in pool.", *_NU3
    src_df = pd.DataFrame(active_rows)
    if id_column not in src_df.columns:
        return f"Source table has no column '{id_column}'.", *_NU3
    motl_df = pd.DataFrame(motl_rows).copy()
    id_to_val = src_df.drop_duplicates(subset=[id_column]).set_index(id_column)[val_col].dropna().to_dict()
    motl_df[dst_col] = motl_df["subtomo_id"].map(id_to_val)
    matched = int(motl_df["subtomo_id"].isin(id_to_val).sum())
    pool_state = PoolState.from_stores(registry, pool_meta, next_id)
    pool_state = replace_motl_rows(pool_state, target_id, motl_df)
    return f"Wrote '{val_col}' → '{dst_col}' for {matched} of {len(motl_df)} particles.", *pool_state.to_stores()


def _do_create_motl(target_id, id_column, val_col, dst_col, label, active_rows, registry, pool_meta, next_id):
    """Create a clean motl subset from active_rows matching target motl by subtomo_id."""
    try:
        motl_rows = _get_rows(target_id)
    except PoolPayloadMissing:
        return "Target motl not found in pool.", *_NU3
    src_df = pd.DataFrame(active_rows)
    if id_column not in src_df.columns:
        return f"Source table has no column '{id_column}'.", *_NU3
    motl_df = pd.DataFrame(motl_rows)
    ids = set(src_df[id_column].dropna().astype(float))
    subset = motl_df[motl_df["subtomo_id"].isin(ids)].copy()
    matched = len(subset)
    if matched == 0:
        return "No subtomo_id values matched the source table.", *_NU3
    if val_col and dst_col:
        id_to_val = src_df.drop_duplicates(subset=[id_column]).set_index(id_column)[val_col].dropna().to_dict()
        subset[dst_col] = subset["subtomo_id"].map(id_to_val)
    pool_state = PoolState.from_stores(registry, pool_meta, next_id)
    pool_state, new_id = _insert_motl(pool_state, subset.to_dict("records"), label=label)
    display_label = pool_state.registry[new_id]["label"]
    return f"Created '{display_label}' with {matched} particles (matched {matched}/{len(motl_df)}).", *pool_state.to_stores()


def get_table_to_motl(prefix: str, *, allow_modal: bool = True) -> html.Div:
    """Return sidebar content for table→motl operations.

    Parameters
    ----------
    prefix:
        Unique prefix for all ids in this instance.
    allow_modal:
        Reserved for future modal variant; currently the component is always
        rendered inline.
    """
    _motl_opts = [{"label": c, "value": c} for c in Motl.motl_columns]
    return html.Div(
        [
            html.Div(id=f"{prefix}-ttm-status", style={"fontSize": styles.FONT_SM, "color": styles.COLOR_MUTED, "marginBottom": "0.4rem"}),
            formgen.form_row(
                f"{prefix}_target_motl",
                make_dropdown(f"{prefix}-ttm-target-motl", [], None, clearable=True, placeholder="Choose motl from pool…"),
                "Target motl from the editor pool",
                label_id=f"{prefix}-ttm-target-motl-lbl",
                label_text="Target motl",
            ),
            formgen.form_row(
                f"{prefix}_val_col",
                html.Div(
                    [
                        make_dropdown(f"{prefix}-ttm-val-col", [], None, clearable=True, placeholder="Source column…", style={"flex": "1"}),
                        html.Span("→", style={"padding": "0 0.4rem", "lineHeight": "2"}),
                        make_dropdown(f"{prefix}-ttm-dst-col", _motl_opts, None, clearable=True, placeholder="Dest column…", style={"flex": "1"}),
                    ],
                    style={"display": "flex", "gap": "0.25rem", "marginBottom": styles.SECTION_GAP},
                ),
                "Optional: copy a value column from the source table into a motl column",
                label_id=f"{prefix}-ttm-val-col-lbl",
                label_text="Copy value column",
                truly_optional=True,
            ),
            html.Hr(style={"margin": "0.4rem 0"}),
            formgen.form_row(
                f"{prefix}_rows_mode",
                dbc.RadioItems(
                    id=f"{prefix}-ttm-rows-mode",
                    options=[
                        {"label": "All rows", "value": "all"},
                        {"label": "Selected rows only", "value": "selected"},
                    ],
                    value="all",
                    inline=True,
                    className="sidebar-checklist",
                    labelStyle={"marginRight": "0.7rem"},
                ),
                "Which rows to include when creating a new motl",
                label_id=f"{prefix}-ttm-rows-mode-lbl",
                label_text="Rows to include",
            ),
            formgen.form_row(
                f"{prefix}_motl_label",
                dbc.Input(id=f"{prefix}-ttm-label", placeholder="Optional label", size="sm"),
                "Label for the new motl entry in the editor",
                label_id=f"{prefix}-ttm-label-lbl",
                label_text="New motl label",
                truly_optional=True,
            ),
            html.Div(
                [
                    dbc.Button(
                        "Write column to motl",
                        id=f"{prefix}-ttm-write-btn",
                        color=styles.BTN_PRIMARY,
                        size="sm",
                        style={"width": "100%", "marginBottom": "0.3rem"},
                    ),
                    dbc.Button(
                        "Create new motl",
                        id=f"{prefix}-ttm-create-btn",
                        color=styles.BTN_SECONDARY,
                        size="sm",
                        style={"width": "100%"},
                    ),
                ]
            ),
        ],
        style={"padding": "0.5rem 0"},
    )


def register_table_to_motl_callbacks(
    app,
    prefix: str,
    *,
    source_table_id: str,
    id_column: str = "qp_id",
) -> None:
    """Register all callbacks for a table→motl component instance.

    Parameters
    ----------
    app:
        The Dash application.
    prefix:
        Must match the prefix passed to :func:`get_table_to_motl`.
    source_table_id:
        Id of the ``AgGrid`` component whose ``rowData`` / ``selectedRows``
        supply the source rows.
    id_column:
        Column in the source table whose values are matched against
        ``subtomo_id`` in the target motl.  Defaults to ``"qp_id"``.
    """

    @app.callback(
        Output(f"{prefix}-ttm-target-motl", "options"),
        Input(_ids.POOL_REGISTRY, "data"),
        prevent_initial_call=True,
    )
    def _populate_target(registry):
        registry = registry or {}
        return [{"label": v.get("label", k), "value": k} for k, v in registry.items()]

    @app.callback(
        Output(f"{prefix}-ttm-val-col", "options"),
        Input(source_table_id, "rowData"),
        prevent_initial_call=True,
    )
    def _populate_val_col(row_data):
        if not row_data:
            return []
        return [{"label": c, "value": c} for c in pd.DataFrame(row_data or []).columns]

    @app.callback(
        Output(f"{prefix}-ttm-status", "children"),
        Output(_ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(_ids.POOL_META, "data", allow_duplicate=True),
        Output(_ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Input(f"{prefix}-ttm-write-btn", "n_clicks"),
        Input(f"{prefix}-ttm-create-btn", "n_clicks"),
        State(f"{prefix}-ttm-target-motl", "value"),
        State(f"{prefix}-ttm-val-col", "value"),
        State(f"{prefix}-ttm-dst-col", "value"),
        State(f"{prefix}-ttm-rows-mode", "value"),
        State(f"{prefix}-ttm-label", "value"),
        State(source_table_id, "rowData"),
        State(source_table_id, "selectedRows"),
        State(_ids.POOL_REGISTRY, "data"),
        State(_ids.POOL_META, "data"),
        State(_ids.POOL_NEXT_ID, "data"),
        prevent_initial_call=True,
    )
    def _act(
        _write_click, _create_click,
        target_id, val_col, dst_col, rows_mode, label,
        all_rows, selected_rows,
        registry, pool_meta, next_id,
    ):
        active = (selected_rows or []) if (rows_mode or "all") == "selected" else (all_rows or [])
        if not active:
            return ("No rows selected." if rows_mode == "selected" else "No rows in table."), *_NU3
        if ctx.triggered_id == f"{prefix}-ttm-write-btn":
            if not val_col or not dst_col:
                return "Choose both source and destination column.", *_NU3
            return _do_write_col(target_id, val_col, dst_col, id_column, active, registry, pool_meta, next_id)
        if ctx.triggered_id == f"{prefix}-ttm-create-btn":
            return _do_create_motl(target_id, id_column, val_col, dst_col, label, active, registry, pool_meta, next_id)
        return no_update, *_NU3
