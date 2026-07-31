"""Row editing callbacks for tableview."""
from __future__ import annotations

from dash import Input, Output, State, exceptions


def register_tableedit_callbacks(app, prefix: str) -> None:
    @app.callback(
        Output(f"{prefix}-global-data-store", "data", allow_duplicate=True),
        Output(f"{prefix}-grid", "rowData", allow_duplicate=True),
        Input(f"{prefix}-remove-rows-btn", "n_clicks"),
        State(f"{prefix}-grid", "rowData"),
        State(f"{prefix}-grid", "selectedRows"),
        prevent_initial_call=True,
    )
    def remove_selected_rows(_, all_rows, selected):
        if not selected:
            raise exceptions.PreventUpdate
        selected_set = {frozenset(row.items()) for row in selected}
        kept = [row for row in all_rows if frozenset(row.items()) not in selected_set]
        return kept, kept

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
