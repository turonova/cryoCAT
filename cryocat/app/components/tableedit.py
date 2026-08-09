"""Row editing callbacks for tableview."""
from __future__ import annotations

import pandas as pd
from dash import Input, Output, State, exceptions

from cryocat.app import ids
from cryocat.app.pool import PoolState, get_rows, replace_motl_rows, PoolPayloadMissing


def _remove_rows(
    ref: dict, all_rows: list, selected: list, registry, pool_meta, next_id
) -> tuple:
    """Pure: compute kept DF and new PoolState after removing selected rows."""
    motl_id = ref.get("motl_id")
    df = get_rows(motl_id)  # raises PoolPayloadMissing if absent
    sel_set = {frozenset(r.items()) for r in selected}
    kept = [r for r in (all_rows or []) if frozenset(r.items()) not in sel_set]
    kept_df = pd.DataFrame(kept) if kept else df.iloc[:0].copy()
    state = PoolState.from_stores(registry, pool_meta, next_id)
    return replace_motl_rows(state, motl_id, kept_df), motl_id, kept


def register_tableedit_callbacks(app, prefix: str) -> None:
    @app.callback(
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output(f"{prefix}-global-data-store", "data", allow_duplicate=True),
        Output(f"{prefix}-grid", "rowData", allow_duplicate=True),
        Input(f"{prefix}-remove-rows-btn", "n_clicks"),
        State(f"{prefix}-grid", "rowData"),
        State(f"{prefix}-grid", "selectedRows"),
        State(f"{prefix}-global-data-store", "data"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        prevent_initial_call=True,
    )
    def remove_selected_rows(_, all_rows, selected, ref, registry, pool_meta, next_id):
        if not selected or not isinstance(ref, dict) or not ref.get("motl_id"):
            raise exceptions.PreventUpdate
        try:
            state, motl_id, kept = _remove_rows(ref, all_rows, selected, registry, pool_meta, next_id)
        except PoolPayloadMissing:
            raise exceptions.PreventUpdate
        new_rev = state.registry[motl_id]["revision"]
        return *state.to_stores(), {"motl_id": motl_id, "rev": new_rev}, kept

    @app.callback(
        Output(f"{prefix}-grid", "selectedRows", allow_duplicate=True),
        Input(f"{prefix}-select-inverse-btn", "n_clicks"),
        State(f"{prefix}-grid", "rowData"),
        State(f"{prefix}-grid", "selectedRows"),
        prevent_initial_call=True,
    )
    def select_inverse_rows(_, all_rows, selected):
        if not all_rows:
            raise exceptions.PreventUpdate
        selected_set = {frozenset(row.items()) for row in (selected or [])}
        return [row for row in all_rows if frozenset(row.items()) not in selected_set]
