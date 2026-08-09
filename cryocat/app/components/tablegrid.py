"""AG Grid loading and column sizing for tableview."""
from __future__ import annotations

import pandas as pd
from dash import Input, Output, State, exceptions, no_update, html
import dash_ag_grid as dag

from cryocat.app.pool import get_rows, PoolPayloadMissing

# Cap rows sent to the browser per load event.  AG Grid virtualises rows so the
# user can still scroll the visible window efficiently, but serialising 60k+ rows
# to JSON on every load caused ~10 s roundtrip times.  (W2 / PERF_FINISH)
_MAX_GRID_ROWS = 2_000


def get_grid_row_count_notice(prefix: str) -> html.Div:
    """Companion component to get_grid — shows 'Showing N of M rows' when truncated."""
    return html.Div(
        id=f"{prefix}-grid-row-count",
        style={"fontSize": "0.75rem", "color": "var(--color9)", "marginTop": "2px"},
    )


def get_grid(prefix: str) -> dag.AgGrid:
    return dag.AgGrid(
        id=f"{prefix}-grid",
        columnDefs=[],
        rowData=[],
        defaultColDef={
            "sortable": True,
            "filter": True,
            "editable": True,
            "resizable": True,
        },
        dashGridOptions={
            "rowSelection": "multiple",
            "suppressRowClickSelection": True,
        },
        style={"height": "300px", "width": "100%"},
        className="ag-theme-balham",
        columnSizeOptions={"skipHeader": False},
    )


def col_defs_from_df(df: pd.DataFrame) -> list[dict]:
    """Build AG Grid columnDefs from a DataFrame.  Pure."""
    col_defs = []
    for i, col in enumerate(df.columns):
        col_def = {
            "field": col,
            "headerName": col,
            "headerTooltip": col,
            "checkboxSelection": i == 0,
            "filter": True,
            "floatingFilter": False,
            "minWidth": 80 if i == 0 else None,
        }
        if pd.api.types.is_float_dtype(df[col]):
            col_def["valueFormatter"] = {
                "function": "(params.value != null) ? params.value.toFixed(3) : ''"
            }
        col_defs.append(col_def)
    return col_defs


def register_tablegrid_callbacks(app, prefix: str) -> None:
    @app.callback(
        Output(f"{prefix}-grid", "columnDefs", allow_duplicate=True),
        Output(f"{prefix}-grid", "rowData", allow_duplicate=True),
        Output(f"{prefix}-grid-row-count", "children"),
        Input(f"{prefix}-global-data-store", "data"),
        prevent_initial_call=True,
    )
    def load_data_to_grid(ref):
        if not ref or not isinstance(ref, dict):
            raise exceptions.PreventUpdate
        motl_id = ref.get("motl_id")
        if not motl_id:
            raise exceptions.PreventUpdate
        try:
            df = get_rows(motl_id)
        except PoolPayloadMissing:
            raise exceptions.PreventUpdate
        total = len(df)
        visible = df.head(_MAX_GRID_ROWS)
        notice = (
            f"Showing {len(visible):,} of {total:,} rows — apply filters to narrow"
            if total > _MAX_GRID_ROWS else f"{total:,} rows"
        )
        return col_defs_from_df(visible), visible.to_dict("records"), notice

    @app.callback(
        Output(f"{prefix}-grid", "columnSize"),
        Input(f"{prefix}-grid", "columnDefs"),
        Input(f"{prefix}-grid", "rowData"),
        prevent_initial_call=True,
    )
    def adapt_column_size(col, rows):
        return "sizeToFit"

    @app.callback(
        Output(f"{prefix}-grid", "selectedRows", allow_duplicate=True),
        Output(f"{prefix}-select-all-btn", "children"),
        Input(f"{prefix}-select-all-btn", "n_clicks"),
        State(f"{prefix}-grid", "virtualRowData"),
        State(f"{prefix}-grid", "selectedRows"),
        prevent_initial_call=True,
    )
    def _toggle_select_all(n_clicks, virtual_rows, selected):
        if not n_clicks:
            raise exceptions.PreventUpdate
        visible = virtual_rows or []
        currently_all = selected and len(selected) >= len(visible) and bool(visible)
        if currently_all:
            return [], "Select All Visible"
        return visible, "Deselect All"
