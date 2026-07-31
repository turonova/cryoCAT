"""AG Grid loading and column sizing for tableview."""
from __future__ import annotations

import pandas as pd
from dash import Input, Output, exceptions
import dash_ag_grid as dag


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
    """Build AG Grid columnDefs from a DataFrame. Pure."""
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
        Input(f"{prefix}-global-data-store", "data"),
        prevent_initial_call=True,
    )
    def load_data_to_grid(global_data):
        if not global_data:
            raise exceptions.PreventUpdate
        df = pd.DataFrame(global_data)
        return col_defs_from_df(df), df.to_dict("records")

    @app.callback(
        Output(f"{prefix}-grid", "columnSize"),
        Input(f"{prefix}-grid", "columnDefs"),
        Input(f"{prefix}-grid", "rowData"),
        prevent_initial_call=True,
    )
    def adapt_column_size(col, rows):
        return "sizeToFit"
