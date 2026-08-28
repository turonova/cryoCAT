"""Row editing callbacks for tableview — infinite-row-model aware.

All bulk operations (remove, inverse select) use the selection-ids-store
(list of identity column values) as the source of truth, not the grid's
rowData which is partial in the infinite model.
"""
from __future__ import annotations

from dash import Input, Output, State, exceptions

from cryocat.app import ids, pool
from cryocat.app.components.tablegrid import resolve_select_all_ids


def register_tableedit_callbacks(app, prefix: str) -> None:
    @app.callback(
        Output(f"{prefix}-remove-rows-btn", "disabled", allow_duplicate=True),
        Output(f"{prefix}-remove-rows-btn", "title", allow_duplicate=True),
        Output(f"{prefix}-select-inverse-btn", "disabled", allow_duplicate=True),
        Output(f"{prefix}-select-inverse-btn", "title", allow_duplicate=True),
        Input(f"{prefix}-global-data-store", "data"),
        prevent_initial_call=True,
    )
    def _update_edit_btn_states(ref):
        """Disable edit buttons when the active dataset has no identity column (W3)."""
        if not isinstance(ref, dict) or pool.get_id_column(ref) is None:
            msg = "No row identity column — row editing is unavailable for this dataset."
            return True, msg, True, msg
        return False, "", False, ""

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
        """Remove rows whose identity column value is in selection_ids."""
        if not selection_ids or not isinstance(ref, dict):
            raise exceptions.PreventUpdate
        id_col = pool.get_id_column(ref)
        if not id_col:
            raise exceptions.PreventUpdate
        df = pool.get_table_df(ref)
        if df is None or id_col not in df.columns:
            raise exceptions.PreventUpdate
        id_set = set(selection_ids)
        kept_df = df[~df[id_col].isin(id_set)]
        return (*pool.commit_rows(ref, kept_df, registry, pool_meta, next_id), [])

    @app.callback(
        Output(f"{prefix}-selection-ids-store", "data", allow_duplicate=True),
        Input(f"{prefix}-select-inverse-btn", "n_clicks"),
        State(f"{prefix}-global-data-store", "data"),
        State(f"{prefix}-grid", "filterModel"),
        State(f"{prefix}-slider-filters-store", "data"),
        State(f"{prefix}-selection-ids-store", "data"),
        prevent_initial_call=True,
    )
    def select_inverse_rows(_, ref, filter_model, slider_filters, current_ids):
        """Invert the selection: all filtered rows not currently selected."""
        if not isinstance(ref, dict):
            raise exceptions.PreventUpdate
        id_col = pool.get_id_column(ref)
        if not id_col:
            raise exceptions.PreventUpdate
        df = pool.get_table_df(ref)
        if df is None:
            raise exceptions.PreventUpdate
        all_filtered_ids = resolve_select_all_ids(
            df, filter_model or {}, slider_filters or {}, id_column=id_col
        )
        current_set = set(current_ids or [])
        return [i for i in all_filtered_ids if i not in current_set]
