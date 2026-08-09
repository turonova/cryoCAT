"""Row editing callbacks for tableview — infinite-row-model aware.

All bulk operations (remove, inverse select) use the selection-ids-store
(list of subtomo_ids) as the source of truth, not the grid's rowData which
is partial in the infinite model.
"""
from __future__ import annotations

from dash import Input, Output, State, exceptions

from cryocat.app import ids
from cryocat.app.pool import PoolState, get_rows, replace_motl_rows, PoolPayloadMissing
from cryocat.app.components.tablegrid import apply_filter_model, resolve_select_all_ids


def register_tableedit_callbacks(app, prefix: str) -> None:
    @app.callback(
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output(f"{prefix}-global-data-store", "data", allow_duplicate=True),
        Output(f"{prefix}-selection-ids-store", "data", allow_duplicate=True),
        Input(f"{prefix}-remove-rows-btn", "n_clicks"),
        State(f"{prefix}-selection-ids-store", "data"),
        State(f"{prefix}-global-data-store", "data"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        prevent_initial_call=True,
    )
    def remove_selected_rows(_, selection_ids, ref, registry, pool_meta, next_id):
        """Remove all rows whose subtomo_id is in selection_ids from the pool entry."""
        if not selection_ids or not isinstance(ref, dict) or not ref.get("motl_id"):
            raise exceptions.PreventUpdate
        motl_id = ref["motl_id"]
        try:
            df = get_rows(motl_id)
        except PoolPayloadMissing:
            raise exceptions.PreventUpdate
        if "subtomo_id" not in df.columns:
            raise exceptions.PreventUpdate
        id_set = set(selection_ids)
        kept_df = df[~df["subtomo_id"].isin(id_set)]
        state = PoolState.from_stores(registry, pool_meta, next_id)
        state = replace_motl_rows(state, motl_id, kept_df)
        new_rev = state.registry[motl_id]["revision"]
        return *state.to_stores(), {"motl_id": motl_id, "rev": new_rev}, []

    @app.callback(
        Output(f"{prefix}-selection-ids-store", "data", allow_duplicate=True),
        Input(f"{prefix}-select-inverse-btn", "n_clicks"),
        State(f"{prefix}-global-data-store", "data"),
        State(f"{prefix}-grid", "filterModel"),
        State(f"{prefix}-selection-ids-store", "data"),
        prevent_initial_call=True,
    )
    def select_inverse_rows(_, ref, filter_model, current_ids):
        """Invert the selection: all filtered rows not currently selected."""
        if not isinstance(ref, dict) or not ref.get("motl_id"):
            raise exceptions.PreventUpdate
        motl_id = ref["motl_id"]
        try:
            df = get_rows(motl_id)
        except PoolPayloadMissing:
            raise exceptions.PreventUpdate
        all_filtered_ids = resolve_select_all_ids(df, filter_model or {}, {})
        current_set = set(current_ids or [])
        return [i for i in all_filtered_ids if i not in current_set]
