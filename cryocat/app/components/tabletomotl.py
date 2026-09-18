"""Shared "table → motl" sidebar component.

``get_table_to_motl(prefix, *, allow_modal=True)`` builds the layout.
``register_table_to_motl_callbacks(app, prefix, *, source_table_id, id_column="qp_id")``
wires the callbacks.

The component merges on ``subtomo_id`` (one-to-one), optionally writes a value
column from the source table into a motl column, and can create a new clean-subset
motl from the current filter/selection.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from dash import html, dcc, ctx, Input, Output, State, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from cryocat.core.cryomotl import Motl
from cryocat.app import ids as _ids
from cryocat.app import styles
from cryocat.app import formgen
from cryocat.app.formgen import make_dropdown
from cryocat.app.pool import PoolState, insert_motl as _insert_motl, get_rows as _get_rows, PoolPayloadMissing


_NU3 = (no_update, no_update, no_update)

_N_PAIRS = 4


def _do_write_cols(target_id, pairs, id_column, active_rows, registry, pool_meta, next_id):
    """Write multiple source→destination column pairs atomically.

    *pairs* is a list of (val_col, dst_col) tuples where both are non-None.
    All pairs are validated before any write; if one fails, none is applied.
    """
    from cryocat.app.pool import replace_motl_rows
    if not pairs:
        return "No column pairs specified.", *_NU3
    try:
        motl_rows = _get_rows(target_id)
    except PoolPayloadMissing:
        return "Target motl not found in pool.", *_NU3
    src_df = pd.DataFrame(active_rows)
    if id_column not in src_df.columns:
        return f"Source table has no column '{id_column}'.", *_NU3
    motl_df = pd.DataFrame(motl_rows).copy()

    # Validate all pairs first.
    errors = []
    for val_col, dst_col in pairs:
        if val_col not in src_df.columns:
            errors.append(f"Source has no column '{val_col}'.")
        if dst_col not in Motl.motl_columns:
            errors.append(f"'{dst_col}' is not a motl column.")
    if errors:
        return "Validation failed — no changes written. " + " ".join(errors), *_NU3

    # Apply all pairs; unmatched rows receive -2 so no particle is left as NaN.
    _UNMATCHED = -2
    total_matched = 0
    for val_col, dst_col in pairs:
        id_to_val = src_df.drop_duplicates(subset=[id_column]).set_index(id_column)[val_col].dropna().to_dict()
        motl_df[dst_col] = motl_df["subtomo_id"].map(id_to_val).fillna(_UNMATCHED)
        total_matched += int(motl_df["subtomo_id"].isin(id_to_val).sum())

    pool_state = PoolState.from_stores(registry, pool_meta, next_id)
    pool_state = replace_motl_rows(pool_state, target_id, motl_df)
    n_pairs = len(pairs)
    pair_desc = ", ".join(f"'{s}'→'{d}'" for s, d in pairs)
    matched_per_pair = total_matched // n_pairs
    unmatched = len(motl_df) - matched_per_pair
    rest_note = f", {unmatched} set to {_UNMATCHED}" if unmatched > 0 else ""
    return (
        f"Wrote {n_pairs} pair(s) ({pair_desc}) — {matched_per_pair} particles matched{rest_note}.",
        *pool_state.to_stores()
    )


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


def _do_create_table_entry(target_id, val_cols, id_column, active_rows, dp_registry, dp_next_id):
    """Create a data pool table entry containing motl rows plus selected source columns.

    The motl itself is not modified.  The new entry can hold any columns.
    """
    from cryocat.app.datapool import DataPoolState as _DPState
    from cryocat.app import datapool as _datapool
    try:
        motl_rows = _get_rows(target_id)
    except PoolPayloadMissing:
        return "Target motl not found in pool.", no_update, no_update
    src_df = pd.DataFrame(active_rows)
    motl_df = pd.DataFrame(motl_rows).copy()
    src_cols = [c for c in val_cols if c and c in src_df.columns]
    if not src_cols:
        return "Select at least one source column.", no_update, no_update
    if id_column not in src_df.columns:
        return f"Source table has no column '{id_column}'.", no_update, no_update
    merge_cols = [id_column] + [c for c in src_cols if c != id_column]
    merge_src = src_df[merge_cols].drop_duplicates(subset=[id_column])
    result = motl_df.merge(merge_src, left_on="subtomo_id", right_on=id_column, how="left")
    if id_column != "subtomo_id" and id_column in result.columns:
        result = result.drop(columns=[id_column])
    ds = _DPState.from_stores(dp_registry, dp_next_id)
    ds, new_id = _datapool.insert_entry(
        ds, result, label=f"table from {target_id}", reader="table-from-motl", source_path="",
    )
    new_dp_reg, new_dp_next = ds.to_stores()
    n_added = len(src_cols)
    return (
        f"Created table entry '{new_id}' ({len(result):,} rows, {n_added} new column(s)). Motl unchanged.",
        new_dp_reg,
        new_dp_next,
    )


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
            formgen.form_row(
                f"{prefix}_target_motl",
                make_dropdown(f"{prefix}-ttm-target-motl", [], None, clearable=True, placeholder="Choose motl from pool…"),
                "Target motl from the editor pool",
                label_id=f"{prefix}-ttm-target-motl-lbl",
                label_text="Target motl",
            ),
            html.Div(
                [
                    html.Label(
                        "Column pairs (source → dest):",
                        style={"fontSize": styles.FONT_SM, "fontWeight": 600, "marginBottom": "0.2rem"},
                    ),
                    html.P(
                        "Select source and destination for each pair. Empty rows are skipped. "
                        "All or none are written.",
                        style={"fontSize": styles.FONT_SM, "color": styles.COLOR_MUTED, "marginBottom": "0.4rem"},
                    ),
                    *[
                        html.Div(
                            [
                                make_dropdown(
                                    f"{prefix}-ttm-val-col-{i}",
                                    [],
                                    None,
                                    clearable=True,
                                    placeholder=f"Source {i + 1}…",
                                    style={"flex": "1"},
                                ),
                                html.Span("→", style={"padding": "0 0.4rem", "lineHeight": "2"}),
                                make_dropdown(
                                    f"{prefix}-ttm-dst-col-{i}",
                                    _motl_opts,
                                    None,
                                    clearable=True,
                                    placeholder=f"Dest {i + 1}…",
                                    style={"flex": "1"},
                                ),
                            ],
                            style={"display": "flex", "gap": "0.25rem", "marginBottom": "0.25rem"},
                        )
                        for i in range(_N_PAIRS)
                    ],
                ],
                style={"marginBottom": styles.SECTION_GAP},
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
                        "Write columns to motl",
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
                        style={"width": "100%", "marginBottom": "0.3rem"},
                    ),
                    dbc.Button(
                        "Create table entry instead",
                        id=f"{prefix}-ttm-create-te-btn",
                        color=styles.BTN_SECONDARY,
                        size="sm",
                        style={"width": "100%"},
                    ),
                ]
            ),
            html.Div(id=f"{prefix}-ttm-status", style={"fontSize": styles.FONT_SM, "color": styles.COLOR_MUTED, "marginTop": "0.4rem"}),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Create table entry")),
                    dbc.ModalBody(
                        "A motl cannot hold a new column. "
                        "The motl's rows plus the selected source columns will be saved as a new table entry. "
                        "The motl itself is unchanged. Continue?"
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button("Create", id=f"{prefix}-ttm-te-confirm", color="primary", size="sm"),
                            dbc.Button("Cancel", id=f"{prefix}-ttm-te-cancel", color="secondary", size="sm", className="ms-2"),
                        ]
                    ),
                ],
                id=f"{prefix}-ttm-te-modal",
                is_open=False,
            ),
        ],
        style={"padding": "0.5rem 0"},
    )


def register_table_to_motl_callbacks(
    app,
    prefix: str,
    *,
    source_table_id: str = "",
    id_column: str = "qp_id",
    source_store_id: str | None = None,
    source_sel_store_id: str | None = None,
    resolve_df: Callable | None = None,
) -> None:
    """Register all callbacks for a table→motl component instance.

    Parameters
    ----------
    app:
        The Dash application.
    prefix:
        Must match the prefix passed to :func:`get_table_to_motl`.
    source_table_id:
        Id of the ``AgGrid`` component whose ``selectedRows`` supply selected
        source rows.  ``rowData`` is read for columns only when *source_store_id*
        is not provided (plain list-model grids).  Unused when both
        *source_store_id* and *source_sel_store_id* are provided.
    id_column:
        Column in the source table whose values are matched against
        ``subtomo_id`` in the target motl.  Defaults to ``"qp_id"``.
    source_store_id:
        Optional id of a ``dcc.Store`` holding the table's data reference.
        When provided the column-list callbacks watch this store instead of
        ``source_table_id.rowData`` (which is never populated in the
        infinite/server-side row model used by :mod:`tablegrid`).
    source_sel_store_id:
        Optional id of a ``dcc.Store`` whose ``data`` holds the selected rows
        list.  When provided, replaces ``State(source_table_id, "selectedRows")``
        in all action callbacks.  Use this for sidebar panels that serve
        multiple grids via a relay store.
    resolve_df:
        Callable ``(store_data) -> pd.DataFrame | None`` used when
        *source_store_id* is provided.  Must match the resolver used by the
        table (e.g. ``datapool.resolve_df``).
    """

    _has_store = source_store_id is not None and resolve_df is not None
    _has_sel_store = source_sel_store_id is not None

    app.clientside_callback(
        """function(registry) {
            var reg = registry || {};
            return Object.keys(reg).map(function(k) {
                return {label: (reg[k].label || k), value: k};
            });
        }""",
        Output(f"{prefix}-ttm-target-motl", "options"),
        Input(_ids.POOL_REGISTRY, "data"),
        prevent_initial_call=True,
    )

    if _has_store:
        @app.callback(
            *[Output(f"{prefix}-ttm-val-col-{i}", "options") for i in range(_N_PAIRS)],
            Input(source_store_id, "data"),
            prevent_initial_call=True,
        )
        def _populate_val_cols(ref):
            df = resolve_df(ref)
            cols = [{"label": c, "value": c} for c in df.columns] if df is not None and not df.empty else []
            return tuple(cols for _ in range(_N_PAIRS))
    else:
        @app.callback(
            *[Output(f"{prefix}-ttm-val-col-{i}", "options") for i in range(_N_PAIRS)],
            Input(source_table_id, "rowData"),
            prevent_initial_call=True,
        )
        def _populate_val_cols(row_data):
            cols = [{"label": c, "value": c} for c in pd.DataFrame(row_data or []).columns] if row_data else []
            return tuple(cols for _ in range(_N_PAIRS))

    _act_extra = [State(source_store_id, "data")] if _has_store else [State(source_table_id, "rowData")]
    _sel_state = State(source_sel_store_id, "data") if _has_sel_store else State(source_table_id, "selectedRows")

    @app.callback(
        Output(f"{prefix}-ttm-status", "children"),
        Output(_ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(_ids.POOL_META, "data", allow_duplicate=True),
        Output(_ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Input(f"{prefix}-ttm-write-btn", "n_clicks"),
        Input(f"{prefix}-ttm-create-btn", "n_clicks"),
        State(f"{prefix}-ttm-target-motl", "value"),
        *[State(f"{prefix}-ttm-val-col-{i}", "value") for i in range(_N_PAIRS)],
        *[State(f"{prefix}-ttm-dst-col-{i}", "value") for i in range(_N_PAIRS)],
        State(f"{prefix}-ttm-rows-mode", "value"),
        State(f"{prefix}-ttm-label", "value"),
        *_act_extra,
        _sel_state,
        State(_ids.POOL_REGISTRY, "data"),
        State(_ids.POOL_META, "data"),
        State(_ids.POOL_NEXT_ID, "data"),
        prevent_initial_call=True,
    )
    def _act(
        _write_click, _create_click,
        target_id,
        *rest,
    ):
        # Unpack: N val_cols, N dst_cols, rows_mode, label, raw_data, selected_rows,
        # registry, pool_meta, next_id
        n = _N_PAIRS
        val_cols = list(rest[:n])
        dst_cols = list(rest[n:2 * n])
        rows_mode, label, raw_data, selected_rows, registry, pool_meta, next_id = rest[2 * n:]

        if _has_store:
            df = resolve_df(raw_data)
            all_rows = df.to_dict("records") if df is not None else []
        else:
            all_rows = raw_data or []

        active = (selected_rows or []) if (rows_mode or "all") == "selected" else all_rows
        if not active:
            return ("No rows selected." if rows_mode == "selected" else "No rows in table."), *_NU3
        if ctx.triggered_id == f"{prefix}-ttm-write-btn":
            pairs = [(vc, dc) for vc, dc in zip(val_cols, dst_cols) if vc and dc]
            if not pairs:
                return "Choose at least one source and destination column pair.", *_NU3
            return _do_write_cols(target_id, pairs, id_column, active, registry, pool_meta, next_id)
        if ctx.triggered_id == f"{prefix}-ttm-create-btn":
            val_col = val_cols[0] if val_cols else None
            dst_col = dst_cols[0] if dst_cols else None
            return _do_create_motl(target_id, id_column, val_col, dst_col, label, active, registry, pool_meta, next_id)
        return no_update, *_NU3

    _te_extra = [State(source_store_id, "data")] if _has_store else [State(source_table_id, "rowData")]

    @app.callback(
        Output(f"{prefix}-ttm-te-modal", "is_open"),
        Output(f"{prefix}-ttm-status", "children", allow_duplicate=True),
        Output(_ids.DATA_POOL_REGISTRY, "data", allow_duplicate=True),
        Output(_ids.DATA_POOL_NEXT_ID, "data", allow_duplicate=True),
        Input(f"{prefix}-ttm-create-te-btn", "n_clicks"),
        Input(f"{prefix}-ttm-te-confirm", "n_clicks"),
        Input(f"{prefix}-ttm-te-cancel", "n_clicks"),
        State(f"{prefix}-ttm-target-motl", "value"),
        *[State(f"{prefix}-ttm-val-col-{i}", "value") for i in range(_N_PAIRS)],
        State(f"{prefix}-ttm-rows-mode", "value"),
        _sel_state,
        *_te_extra,
        State(_ids.DATA_POOL_REGISTRY, "data"),
        State(_ids.DATA_POOL_NEXT_ID, "data"),
        prevent_initial_call=True,
    )
    def _handle_te_modal(_open_n, _confirm_n, _cancel_n, target_id, *rest):
        n = _N_PAIRS
        val_cols = list(rest[:n])
        rows_mode = rest[n]
        selected_rows = rest[n + 1]
        raw_data = rest[n + 2]
        dp_registry = rest[n + 3]
        dp_next_id = rest[n + 4]

        tid = ctx.triggered_id
        if tid == f"{prefix}-ttm-create-te-btn":
            return True, no_update, no_update, no_update
        if tid == f"{prefix}-ttm-te-cancel":
            return False, no_update, no_update, no_update
        if tid == f"{prefix}-ttm-te-confirm":
            if _has_store:
                df = resolve_df(raw_data)
                all_rows = df.to_dict("records") if df is not None else []
            else:
                all_rows = raw_data or []
            active = (selected_rows or []) if (rows_mode or "all") == "selected" else all_rows
            status, new_dp_reg, new_dp_next = _do_create_table_entry(
                target_id, val_cols, id_column, active, dp_registry, dp_next_id,
            )
            return False, status, new_dp_reg, new_dp_next
        raise PreventUpdate
