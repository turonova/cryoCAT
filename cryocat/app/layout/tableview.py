from cryocat.app.logger import dash_logger, print_dash

from dash import html, dcc, Input, Output, State, callback, exceptions, MATCH, ALL
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import pandas as pd
from cryocat import cryomotl

def get_table_component(prefix: str):
    return html.Div(
        id=f"{prefix}-table-container",
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Button("Apply Changes", id=f"{prefix}-apply-btn", color="primary", className="me-1"),
                            dbc.Button("Save Snapshot", id=f"{prefix}-save-btn", color="secondary", className="me-1"),
                            dbc.Button(
                                "Remove Selected Rows", id=f"{prefix}-remove-rows-btn", color="danger", className="me-1"
                            ),
                            # dbc.Button("Remove Selected Columns", id=f"{prefix}-remove-cols-btn", color="danger"),
                        ],
                        className="d-flex justify-content-end",
                    ),
                ],
                className="mb-2",
            ),
            dag.AgGrid(
                id=f"{prefix}-grid",
                columnDefs=[],
                rowData=[],
                defaultColDef={
                    "sortable": True,
                    "filter": True,
                    "editable": True,
                    "resizable": True,
                    # "maxWidth": 100,
                    # "suppressSizeToFit": False,
                },
                dashGridOptions={
                    "rowSelection": "multiple",
                    "suppressRowClickSelection": True,
                },
                style={"height": "300px", "width": "100%"},
                className="ag-theme-balham",
                # columnSize="autoSize",
                columnSizeOptions={"skipHeader": False},
            ),
            html.Div(
                id=f"{prefix}-filters-container",  # Range sliders go here
                style={"marginBottom": "2rem", "marginTop": "2rem"},
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Save table")),
                    dbc.ModalBody(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        dcc.Input(
                                            type="text",
                                            value="",
                                            placeholder="File path",
                                            id=f"{prefix}-save-path-form",
                                        ),
                                    ),
                                    dbc.Tooltip(
                                        "Specify path and name of the file.",
                                        target=f"{prefix}-save-path-form",
                                        placement="top",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dbc.ModalFooter(dbc.Button("Save", id=f"{prefix}-save-file", className="ms-auto", n_clicks=0)),
                ],
                id=f"{prefix}-save-modal",
                is_open=False,
            ),
            dcc.Store(id=f"{prefix}-snapshot-store"),
        ],
    )


def register_table_callbacks(prefix: str, csv_only=True):

    @callback(
        Output(f"{prefix}-grid", "columnDefs", allow_duplicate=True),
        Output(f"{prefix}-grid", "rowData", allow_duplicate=True),
        Input(f"{prefix}-global-data-store", "data"),
        prevent_initial_call=True,
    )
    def load_data_to_grid(global_data):
        if not global_data:
            raise exceptions.PreventUpdate

        df = pd.DataFrame(global_data)

        col_defs = []
        for i, col in enumerate(df.columns):
            col_def = {
                "field": col,
                "headerName": col,
                "checkboxSelection": True if i == 0 else False,
                "filter": True,
                "floatingFilter": False,
                "minWidth": 80 if i == 0 else None,
            }

            if pd.api.types.is_float_dtype(df[col]):
                col_def["valueFormatter"] = {"function": "(params.value != null) ? params.value.toFixed(3) : ''"}

            col_defs.append(col_def)

        return col_defs, df.to_dict("records")

    @callback(
        Output(f"{prefix}-grid", "columnSize"),
        Input(f"{prefix}-grid", "columnDefs"),
        Input(f"{prefix}-grid", "rowData"),
        prevent_initial_call=True,
    )
    def adapt_column_size(col, rows):
        return "sizeToFit"

    @callback(
        Output(f"{prefix}-global-data-store", "data", allow_duplicate=True),
        Output(f"{prefix}-grid", "rowData", allow_duplicate=True),
        Input(f"{prefix}-apply-btn", "n_clicks"),
        State(f"{prefix}-grid", "rowData"),
        prevent_initial_call=True,
    )
    def apply_changes(_, rows):
        if not rows:
            raise exceptions.PreventUpdate
        return rows, rows

    @callback(
        Output(f"{prefix}-save-modal", "is_open", allow_duplicate=True),
        Input(f"{prefix}-save-btn", "n_clicks"),
        # State(f"{prefix}-save-modal", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_save_modal(_):
        return True

    @callback(
        Output(f"{prefix}-save-modal", "is_open", allow_duplicate=True),
        Input(f"{prefix}-save-file", "n_clicks"),
        State(f"{prefix}-save-path-form", "value"),
        State(f"{prefix}-grid", "rowData"),
        prevent_initial_call=True,
    )
    def save_table(_, file_path, grid_data):
        df = pd.DataFrame(grid_data)
        if file_path.endswith(".csv"):
            df.to_csv(file_path)
        elif csv_only:
            print_dash("The table can be saved only to a csv file.")
        elif file_path.endswith(".em"):
            m = cryomotl.Motl(df)
            m.write_out(file_path)

        return False

    @callback(
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
        filtered = [row for row in all_rows if frozenset(row.items()) not in selected_set]
        return filtered

    @callback(
        Output(f"{prefix}-grid", "columnDefs", allow_duplicate=True),
        Output(f"{prefix}-grid", "rowData", allow_duplicate=True),
        Input(f"{prefix}-remove-cols-btn", "n_clicks"),
        State(f"{prefix}-grid", "columnDefs"),
        State(f"{prefix}-grid", "rowData"),
        State(f"{prefix}-grid", "columnState"),
        prevent_initial_call=True,
    )
    def remove_selected_columns(_, col_defs, row_data, column_state):
        if not column_state:
            raise exceptions.PreventUpdate
        selected_fields = [col["colId"] for col in column_state if col.get("selected")]
        if not selected_fields:
            raise exceptions.PreventUpdate

        df = pd.DataFrame(row_data)
        df = df.drop(columns=selected_fields)
        new_col_defs = [{"field": col, "headerName": col, "checkboxSelection": True} for col in df.columns]
        return new_col_defs, df.to_dict("records")

    @callback(
        Output(f"{prefix}-filters-container", "children"),
        Input(f"{prefix}-global-data-store", "data"),
    )
    def build_range_sliders(data):
        if not data:
            raise exceptions.PreventUpdate

        df = pd.DataFrame(data)
        filter_columns = [col for col in df.select_dtypes(include="number").columns]
        col_components = []

        divFlex = {"display": "flex", "align-content": "stretch", "justifyContent": "flex-end"}
        labelFlex = {"flexShrink": 0}
        slideFlex = {"width": "100%", "padding": "0", "margin": "0", "height": "16px"}

        for col in filter_columns:
            if df[col].max() == df[col].min():
                continue

            min_val = df[col].min()
            max_val = df[col].max()

            col_components.append(
                dbc.Col(
                    html.Div(
                        [
                            html.Div(
                                f"{col}:",
                                style=labelFlex,
                                # style={
                                #     "width": "60px",
                                #     "whiteSpace": "nowrap",
                                #     # "marginRight": "0px",
                                #     "display": "flex",
                                #     "justifyContent": "flex-end",  # <-- right-aligns the text
                                # },
                            ),
                            html.Div(
                                dcc.RangeSlider(
                                    id={"type": f"{prefix}-filter-slider", "column": col},
                                    min=min_val,
                                    max=max_val,
                                    step=(max_val - min_val) / 100 or 1,
                                    value=[min_val, max_val],
                                    tooltip={"placement": "right"},
                                    marks=None,
                                    allowCross=False,
                                ),
                                # style={"flex": "1", "height": "16px", "padding": "0", "margin": "0"},
                                style=slideFlex,
                            ),
                        ],
                        # style={
                        #     "display": "flex",
                        #     "alignItems": "center",
                        #     "width": "100%",
                        #     "gap": "4px",
                        # },
                        style=divFlex,
                    ),
                    width=3,
                    style={"marginBottom": "0.5rem"},
                )
            )

        rows = [dbc.Row(col_components[i : i + 4], className="g-1") for i in range(0, len(col_components), 4)]

        return rows

    @callback(
        Output(f"{prefix}-grid", "rowData", allow_duplicate=True),
        Input(f"{prefix}-global-data-store", "data"),
        Input({"type": f"{prefix}-filter-slider", "column": ALL}, "value"),
        State({"type": f"{prefix}-filter-slider", "column": ALL}, "id"),
        prevent_initial_call=True,
    )
    def filter_data_by_sliders(global_data, slider_values, slider_ids):
        if not global_data:
            raise exceptions.PreventUpdate

        df = pd.DataFrame(global_data)

        print_dash("Original rows:", len(df))

        for slider_id, (min_val, max_val) in zip(slider_ids, slider_values):
            col = slider_id.get("column")
            if col in df.columns:
                df = df[df[col].between(min_val, max_val)]

        print_dash("Filtered rows:", len(df))
        return df.to_dict("records")
