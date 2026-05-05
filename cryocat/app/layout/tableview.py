from cryocat.app.logger import dash_logger, print_dash

from dash import html, dcc, Input, Output, State, callback, exceptions, MATCH, ALL, no_update, ctx
import base64
import tempfile
import os
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import pandas as pd
from cryocat.core import cryomotl
from cryocat.app.apputils import save_output, save_motl
from cryocat.app.layout.tableplot import get_table_plot_component
from cryocat.app.layout.customel import InlineLabeledDropdown, InlineInputForm
from cryocat.utils.ioutils import dimensions_load


_MOTL_TYPES = [
    {"label": "EM", "value": "emmotl"},
    {"label": "STOPGAP", "value": "stopgap"},
    {"label": "Relion", "value": "relion"},
    {"label": "Dynamo", "value": "dynamo"},
]

_RELION_VERSIONS = [
    {"label": "Version 3.0", "value": 3.0},
    {"label": "Version 3.1", "value": 3.1},
    {"label": "Version 4.x", "value": 4.0},
    {"label": "Version 5.x", "value": 5.0},
]


# ── Modal builders ─────────────────────────────────────────────────────────────


def _csv_save_modal(prefix):
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Save as CSV")),
            dbc.ModalBody(
                InlineInputForm(
                    id_=f"{prefix}-csv-path",
                    label="Filename:",
                    type="text",
                    placeholder="Full path including .csv extension",
                )
            ),
            dbc.ModalFooter(
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "width": "100%",
                    },
                    children=[
                        html.H5("", id=f"{prefix}-csv-status-label", style={"margin": 0}),
                        dbc.Button("Save", id=f"{prefix}-csv-save-btn", className="ms-auto", n_clicks=0),
                    ],
                )
            ),
        ],
        id=f"{prefix}-csv-modal",
        is_open=False,
    )


def _motl_save_modal(prefix):
    """Full motl Save-As modal."""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Save As")),
            dbc.ModalBody(
                html.Div(
                    [
                        InlineLabeledDropdown(
                            id_=f"{prefix}-save-type-dropdown",
                            options=_MOTL_TYPES,
                            label="Output type:",
                            multi=False,
                            placeholder="Output type",
                        ),
                        InlineLabeledDropdown(
                            id_=f"{prefix}-save-rln-version-dropdown",
                            label="Version:",
                            default_visibility="hidden",
                            options=_RELION_VERSIONS,
                        ),
                        html.Div(
                            id=f"{prefix}-save-rln-options",
                            className="hidden",
                            style={"flex": "1", "alignItems": "center"},
                            children=[
                                InlineInputForm(
                                    id_=f"{prefix}-save-rln-pixelsize",
                                    type="number",
                                    placeholder="Pixel size",
                                    min=1.0,
                                    step=1,
                                    label="Pixel size (A):",
                                    style={"width": "35%", "marginRight": "10px"},
                                ),
                                InlineInputForm(
                                    id_=f"{prefix}-save-rln-binning",
                                    type="number",
                                    placeholder="Binning",
                                    min=1.0,
                                    step=1,
                                    label="Binning:",
                                    style={"width": "35%", "marginRight": "10px"},
                                ),
                                dbc.Tooltip(
                                    "Use the original input particle list entries where possible.",
                                    target=f"{prefix}-save-rln-use-original",
                                ),
                                dbc.Checkbox(
                                    id=f"{prefix}-save-rln-use-original",
                                    label="Use original entries",
                                    value=False,
                                    inputStyle={"marginRight": "5px"},
                                    className="sidebar-checklist",
                                    labelStyle={"color": "var(--color9)"},
                                    disabled=True,
                                    style={"width": "30%"},
                                ),
                            ],
                        ),
                        html.Div(
                            [
                                html.Div(
                                    "Currently no tomogram file loaded",
                                    id=f"{prefix}-save-rln5-tomos-status",
                                    className="hidden",
                                    style={"marginRight": "10px", "color": "var(--color9)"},
                                ),
                                dcc.Upload(
                                    id=f"{prefix}-save-rln5-tomos-upload",
                                    children=[
                                        dbc.Button(
                                            "Upload tomogram file",
                                            id=f"{prefix}-save-rln5-tomos-upload-btn",
                                        ),
                                    ],
                                    multiple=False,
                                    className="hidden",
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "marginBottom": "0.5rem",
                                "width": "100%",
                                "justifyContent": "space-between",
                            },
                        ),
                        InlineInputForm(
                            id_=f"{prefix}-save-path",
                            label="Filename:",
                            type="text",
                            placeholder="Filename (including its path)",
                        ),
                    ]
                )
            ),
            dbc.ModalFooter(
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "width": "100%",
                    },
                    children=[
                        html.H5("", id=f"{prefix}-save-status-label", style={"margin": 0}),
                        dbc.Button("Save", id=f"{prefix}-save-file", className="ms-auto", n_clicks=0),
                    ],
                )
            ),
        ],
        id=f"{prefix}-save-modal",
        is_open=False,
    )


def _overwrite_confirm_modal(prefix):
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Confirm overwrite")),
            dbc.ModalBody(html.P(id=f"{prefix}-overwrite-body-text", children="")),
            dbc.ModalFooter(
                [
                    dbc.Button("Yes", id=f"{prefix}-overwrite-yes-btn", color="danger", className="me-2"),
                    dbc.Button("No", id=f"{prefix}-overwrite-no-btn", color="secondary"),
                ]
            ),
        ],
        id=f"{prefix}-overwrite-modal",
        is_open=False,
    )


# ── Component ──────────────────────────────────────────────────────────────────


def get_table_component(prefix: str, connected_motl_prefix=None, show_create_from_selected=True):
    motl_mode = connected_motl_prefix is not None

    button_children = [
        dbc.Button("Apply Changes", id=f"{prefix}-apply-btn", color="primary", className="me-1"),
    ]

    if motl_mode:
        button_children += [
            dbc.Button("Save As", id=f"{prefix}-save-btn", color="primary", className="me-1"),
            dbc.Button("Save", id=f"{prefix}-save-overwrite-btn", color="secondary", className="me-1"),
        ]
        if show_create_from_selected:
            button_children += [
                dbc.Button(
                    "Create new from selected",
                    id=f"{prefix}-create-from-selected-btn",
                    color="secondary",
                    className="me-1",
                ),
            ]

    button_children += [
        dbc.Button("Save as CSV", id=f"{prefix}-save-csv-btn", color="primary", className="me-1"),
        dbc.Button(
            "Remove Selected Rows",
            id=f"{prefix}-remove-rows-btn",
            color="primary",
            className="me-1",
        ),
        dbc.Button(
            "Plot",
            id=f"{prefix}-plot-graphs-btn",
            color="primary",
            className="me-1",
            n_clicks=0,
        ),
        dbc.Offcanvas(
            [get_table_plot_component(f"{prefix}-table-plot")],
            id=f"{prefix}-plot-graph-panel",
            title="Plotting options",
            placement="end",
            scrollable=True,
            style={"width": "1100px"},
            is_open=False,
        ),
    ]

    extra_children = [_csv_save_modal(prefix)]

    if motl_mode:
        extra_children += [
            _motl_save_modal(prefix),
            _overwrite_confirm_modal(prefix),
            dcc.Store(id=f"{prefix}-save-rln5-tomos-store"),
            dcc.Store(id=f"{prefix}-save-rln5-tomos-filename"),
            dcc.Store(id=f"{prefix}-last-save-params-store"),
        ]

    return html.Div(
        id=f"{prefix}-table-container",
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        button_children,
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
                },
                dashGridOptions={
                    "rowSelection": "multiple",
                    "suppressRowClickSelection": True,
                },
                style={"height": "300px", "width": "100%"},
                className="ag-theme-balham",
                columnSizeOptions={"skipHeader": False},
            ),
            html.H5(
                "Filters",
                style={"marginBottom": "1rem", "marginTop": "1rem"},
            ),
            html.Div(
                id=f"{prefix}-filters-container",
                style={"marginBottom": "2rem"},
            ),
            *extra_children,
            dcc.Store(id=f"{prefix}-snapshot-store"),
        ],
    )


# ── Callbacks ──────────────────────────────────────────────────────────────────


def register_table_callbacks(prefix: str, csv_only=True, connected_motl_prefix=None, slot_idx=None, max_motls=None):

    motl_mode = connected_motl_prefix is not None

    # ── Always-present callbacks ───────────────────────────────────────────────

    @callback(
        Output(f"{prefix}-plot-graph-panel", "is_open", allow_duplicate=True),
        Input(f"{prefix}-plot-graphs-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_offcanvas(n_clicks):
        return True

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
        State(f"{prefix}-global-data-store", "data"),
        State({"type": f"{prefix}-filter-slider", "column": ALL}, "value"),
        State({"type": f"{prefix}-filter-slider", "column": ALL}, "id"),
        prevent_initial_call=True,
    )
    def apply_filters(_, global_data, slider_values, slider_ids):
        if not global_data:
            raise exceptions.PreventUpdate
        df = pd.DataFrame(global_data)
        for slider_id, (min_val, max_val) in zip(slider_ids, slider_values):
            col = slider_id.get("column")
            if col in df.columns:
                df = df[df[col].between(min_val, max_val)]
        filtered = df.to_dict("records")
        return filtered, filtered

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
        return [row for row in all_rows if frozenset(row.items()) not in selected_set]

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
        filter_columns = list(df.select_dtypes(include="number").columns)
        col_components = []
        divFlex = {"display": "flex", "flexDirection": "row", "alignItems": "center"}
        labelFlex = {
            "flexShrink": 0,
            "fontSize": "10px",
            "fontWeight": "600",
            "color": "var(--color10)",
            "whiteSpace": "nowrap",
            "marginRight": "4px",
            "marginLeft": "6px",
            "marginTop": "-10px",
        }
        slideFlex = {"flex": "1", "minWidth": "0", "padding": "0", "marginTop": "-10px"}

        inputStyle = {
            "width": "52px",
            "flexShrink": 0,
            "marginTop": "-10px",
        }

        for col in filter_columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min == col_max or pd.isna(col_min) or pd.isna(col_max):
                continue
            min_val = float(col_min)
            max_val = float(col_max)
            is_int_col = pd.api.types.is_integer_dtype(df[col])
            step = 1.0 if is_int_col else ((max_val - min_val) / 100 or 1.0)
            col_components.append(
                dbc.Col(
                    html.Div(
                        [
                            html.Div(f"{col}:", style=labelFlex),
                            dcc.Input(
                                id={"type": f"{prefix}-filter-min", "column": col},
                                type="number",
                                value=min_val,
                                debounce=True,
                                className="filter-range-input",
                                style=inputStyle,
                            ),
                            html.Div(
                                dcc.RangeSlider(
                                    id={"type": f"{prefix}-filter-slider", "column": col},
                                    min=min_val,
                                    max=max_val,
                                    step=step,
                                    value=[min_val, max_val],
                                    tooltip={"placement": "top"},
                                    marks=None,
                                    allowCross=False,
                                ),
                                style=slideFlex,
                                className="filter-slider-wrapper",
                            ),
                            dcc.Input(
                                id={"type": f"{prefix}-filter-max", "column": col},
                                type="number",
                                value=max_val,
                                debounce=True,
                                className="filter-range-input",
                                style=inputStyle,
                            ),
                        ],
                        style=divFlex,
                    ),
                    width=3,
                )
            )

        return [dbc.Row(col_components, className="gx-1 gy-0")]

    @callback(
        Output({"type": f"{prefix}-filter-slider", "column": MATCH}, "value", allow_duplicate=True),
        Output({"type": f"{prefix}-filter-min", "column": MATCH}, "value", allow_duplicate=True),
        Output({"type": f"{prefix}-filter-max", "column": MATCH}, "value", allow_duplicate=True),
        Input({"type": f"{prefix}-filter-slider", "column": MATCH}, "value"),
        Input({"type": f"{prefix}-filter-min", "column": MATCH}, "value"),
        Input({"type": f"{prefix}-filter-max", "column": MATCH}, "value"),
        State({"type": f"{prefix}-filter-slider", "column": MATCH}, "min"),
        State({"type": f"{prefix}-filter-slider", "column": MATCH}, "max"),
        prevent_initial_call=True,
    )
    def sync_filter_controls(slider_val, min_input, max_input, slider_min, slider_max):
        triggered = ctx.triggered_id
        if triggered is None or slider_val is None:
            raise exceptions.PreventUpdate

        ttype = triggered.get("type", "") if isinstance(triggered, dict) else ""

        if f"{prefix}-filter-slider" in ttype:
            return no_update, slider_val[0], slider_val[1]
        elif f"{prefix}-filter-min" in ttype:
            if min_input is None:
                raise exceptions.PreventUpdate
            clamped = float(max(slider_min, min(min_input, slider_val[1])))
            return [clamped, slider_val[1]], clamped, no_update
        elif f"{prefix}-filter-max" in ttype:
            if max_input is None:
                raise exceptions.PreventUpdate
            clamped = float(min(slider_max, max(max_input, slider_val[0])))
            return [slider_val[0], clamped], no_update, clamped

        raise exceptions.PreventUpdate

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
        for slider_id, (min_val, max_val) in zip(slider_ids, slider_values):
            col = slider_id.get("column")
            if col in df.columns:
                df = df[df[col].between(min_val, max_val)]
        return df.to_dict("records")

    # ── CSV save (always available) ────────────────────────────────────────────

    @callback(
        Output(f"{prefix}-csv-modal", "is_open", allow_duplicate=True),
        Input(f"{prefix}-save-csv-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_csv_modal(_):
        return True

    @callback(
        Output(f"{prefix}-csv-modal", "is_open", allow_duplicate=True),
        Output(f"{prefix}-csv-status-label", "children"),
        Input(f"{prefix}-csv-save-btn", "n_clicks"),
        State(f"{prefix}-csv-path", "value"),
        State(f"{prefix}-grid", "rowData"),
        prevent_initial_call=True,
    )
    def do_csv_save(_, path, grid_data):
        if not path or not grid_data:
            return no_update, "Specify a filename."
        try:
            pd.DataFrame(grid_data).to_csv(path, index=False)
            return False, f"Saved to {path}"
        except Exception as e:
            return no_update, str(e)

    # ── Motl-mode save callbacks ───────────────────────────────────────────────

    if motl_mode:

        @callback(
            Output(f"{prefix}-save-modal", "is_open", allow_duplicate=True),
            Input(f"{prefix}-save-btn", "n_clicks"),
            prevent_initial_call=True,
        )
        def open_save_as_modal(_):
            return True

        @callback(
            Output(f"{prefix}-save-rln-version-dropdown", "value", allow_duplicate=True),
            Output(f"{prefix}-save-rln-version-dropdown-topdiv", "className", allow_duplicate=True),
            Output(f"{prefix}-save-rln-options", "className", allow_duplicate=True),
            Input(f"{prefix}-save-type-dropdown", "value"),
            prevent_initial_call=True,
        )
        def toggle_rln_type(motl_type):
            if motl_type == "relion":
                return 3.0, "flex", "hidden"
            return 3.0, "hidden", "hidden"

        @callback(
            Output(f"{prefix}-save-rln5-tomos-store", "data", allow_duplicate=True),
            Output(f"{prefix}-save-rln5-tomos-filename", "data", allow_duplicate=True),
            Input(f"{prefix}-save-rln5-tomos-upload", "contents"),
            State(f"{prefix}-save-rln5-tomos-upload", "filename"),
            prevent_initial_call=True,
        )
        def load_rln5_tomos(contents, filename):
            if contents is None:
                return no_update, no_update
            _, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[-1]) as tmp:
                tmp.write(decoded)
                tmp_path = tmp.name
            rln_tomos = dimensions_load(tmp_path)
            os.remove(tmp_path)
            return rln_tomos.to_dict("records"), filename

        @callback(
            Output(f"{prefix}-save-rln-options", "className", allow_duplicate=True),
            Output(f"{prefix}-save-rln5-tomos-status", "className", allow_duplicate=True),
            Output(f"{prefix}-save-rln5-tomos-status", "children", allow_duplicate=True),
            Output(f"{prefix}-save-rln5-tomos-upload", "className", allow_duplicate=True),
            Output(f"{prefix}-save-rln5-tomos-upload-btn", "children"),
            Output(f"{prefix}-save-rln-use-original", "disabled"),
            Output(f"{prefix}-save-rln-use-original", "value"),
            Input(f"{prefix}-save-rln-version-dropdown", "value"),
            Input(f"{prefix}-save-rln5-tomos-store", "data"),
            State(f"{connected_motl_prefix}-motl-data-type", "data"),
            State(f"{prefix}-save-rln5-tomos-filename", "data"),
            State(f"{connected_motl_prefix}-relion5-tomos-store", "data"),
            State(f"{connected_motl_prefix}-relion5-tomos-filename", "data"),
            prevent_initial_call=True,
        )
        def toggle_rln_version_options(
            rln_version,
            rln_tomos,
            input_motl_type,
            rln_tomos_name,
            rln_tomos_orig,
            rln_tomos_name_orig,
        ):
            btn_title = "Upload a tomogram file for Relion5"
            if input_motl_type == "relion" and rln_version in [3.0, 3.1, 4.0]:
                disable_orig = (False, False)
            elif input_motl_type == "relion5" and rln_version == 5.0:
                disable_orig = (False, False)
            else:
                disable_orig = (True, False)

            if rln_version in [3.0, 3.1]:
                return "hidden", "hidden", "", "hidden", btn_title, *disable_orig
            elif rln_version == 4.0:
                return "flex", "hidden", "", "hidden", btn_title, *disable_orig
            else:
                status = "Currently no tomogram file loaded"
                if rln_tomos:
                    status = f"Currently loaded: {rln_tomos_name}"
                    btn_title = "Upload a different tomogram file"
                elif input_motl_type == "relion5" and rln_tomos_orig:
                    status = f"Currently loaded: {rln_tomos_name_orig}"
                    btn_title = "Upload a different tomogram file"
                return "flex", "flex", status, "flex", btn_title, *disable_orig

        _update_registry_on_save = slot_idx is not None

        @callback(
            Output(f"{prefix}-save-modal", "is_open", allow_duplicate=True),
            Output(f"{prefix}-save-status-label", "children", allow_duplicate=True),
            Output(f"{prefix}-last-save-params-store", "data"),
            *([Output("motls-registry", "data", allow_duplicate=True)] if _update_registry_on_save else []),
            Input(f"{prefix}-save-file", "n_clicks"),
            State(f"{prefix}-save-path", "value"),
            State(f"{prefix}-save-type-dropdown", "value"),
            State(f"{prefix}-save-rln5-tomos-store", "data"),
            State(f"{connected_motl_prefix}-motl-extra-data-store", "data"),
            State(f"{connected_motl_prefix}-relion-optics-store", "data"),
            State(f"{connected_motl_prefix}-relion5-tomos-store", "data"),
            State(f"{prefix}-save-rln-binning", "value"),
            State(f"{prefix}-save-rln-pixelsize", "value"),
            State(f"{prefix}-save-rln-version-dropdown", "value"),
            State(f"{prefix}-save-rln-use-original", "value"),
            State(f"{prefix}-global-data-store", "data"),
            *([State("motls-registry", "data")] if _update_registry_on_save else []),
            prevent_initial_call=True,
        )
        def save_as(*args):
            if _update_registry_on_save:
                (
                    n_clicks,
                    path,
                    motl_type,
                    rln5_tomos,
                    extra_df,
                    rln_optics,
                    rln5_tomos_orig,
                    rln_binning,
                    rln_pixelsize,
                    rln_version,
                    use_original,
                    data,
                    registry,
                ) = args
            else:
                (
                    n_clicks,
                    path,
                    motl_type,
                    rln5_tomos,
                    extra_df,
                    rln_optics,
                    rln5_tomos_orig,
                    rln_binning,
                    rln_pixelsize,
                    rln_version,
                    use_original,
                    data,
                ) = args
                registry = None

            if not path or not motl_type or not data:
                base_out = (no_update, "Specify output type and filename.", no_update)
                return (*base_out, no_update) if _update_registry_on_save else base_out

            rln_tomos = rln5_tomos or rln5_tomos_orig
            status = save_motl(
                file_path=path,
                data_to_save=data,
                motl_type=motl_type,
                extra_df=extra_df,
                rln_optics=rln_optics,
                rln_tomos=rln_tomos,
                rln_binning=rln_binning,
                rln_pixel_size=rln_pixelsize,
                rln_version=rln_version,
                rln_use_original=use_original,
            )
            params = {
                "path": path,
                "motl_type": motl_type,
                "relion_version": rln_version,
                "relion_pixel_size": rln_pixelsize,
                "relion_binning": rln_binning,
                "relion_use_original": use_original,
            }

            if _update_registry_on_save:
                new_registry = dict(registry or {})
                filename = os.path.basename(path)
                entry = new_registry.get(str(slot_idx), {})
                new_registry[str(slot_idx)] = {"label": filename, "active": entry.get("active", True)}
                return False, status, params, new_registry

            return False, status, params

        @callback(
            Output(f"{prefix}-overwrite-modal", "is_open"),
            Output(f"{prefix}-overwrite-body-text", "children"),
            Output(f"{prefix}-overwrite-yes-btn", "disabled"),
            Input(f"{prefix}-save-overwrite-btn", "n_clicks"),
            State(f"{prefix}-last-save-params-store", "data"),
            prevent_initial_call=True,
        )
        def open_overwrite_confirm(_, params):
            if not params or not params.get("path"):
                return True, "No output path set. Please use 'Save As' first to specify the file.", True
            return True, f"Overwrite '{params['path']}'?", False

        @callback(
            Output(f"{prefix}-overwrite-modal", "is_open", allow_duplicate=True),
            Input(f"{prefix}-overwrite-yes-btn", "n_clicks"),
            Input(f"{prefix}-overwrite-no-btn", "n_clicks"),
            State(f"{prefix}-last-save-params-store", "data"),
            State(f"{prefix}-save-rln5-tomos-store", "data"),
            State(f"{connected_motl_prefix}-motl-extra-data-store", "data"),
            State(f"{connected_motl_prefix}-relion-optics-store", "data"),
            State(f"{connected_motl_prefix}-relion5-tomos-store", "data"),
            State(f"{prefix}-global-data-store", "data"),
            prevent_initial_call=True,
        )
        def do_overwrite_save(
            _yes,
            _no,
            params,
            rln5_tomos,
            extra_df,
            rln_optics,
            rln5_tomos_orig,
            data,
        ):
            if ctx.triggered_id == f"{prefix}-overwrite-no-btn":
                return False
            if not params or not data:
                return False
            rln_tomos = rln5_tomos or rln5_tomos_orig
            save_motl(
                file_path=params["path"],
                data_to_save=data,
                motl_type=params["motl_type"],
                extra_df=extra_df,
                rln_optics=rln_optics,
                rln_tomos=rln_tomos,
                rln_binning=params.get("relion_binning"),
                rln_pixel_size=params.get("relion_pixel_size"),
                rln_version=params.get("relion_version"),
                rln_use_original=params.get("relion_use_original", False),
            )
            return False

        if slot_idx is not None and max_motls is not None:

            @callback(
                *[Output(f"me-{i}-motl-data-store", "data", allow_duplicate=True) for i in range(max_motls)],
                *[Output(f"me-{i}-motl-data-type", "data", allow_duplicate=True) for i in range(max_motls)],
                Output("motls-registry", "data", allow_duplicate=True),
                Output("me-tabs", "active_tab", allow_duplicate=True),
                Input(f"{prefix}-create-from-selected-btn", "n_clicks"),
                State(f"{prefix}-grid", "selectedRows"),
                State(f"{connected_motl_prefix}-motl-data-type", "data"),
                State("motls-registry", "data"),
                prevent_initial_call=True,
            )
            def create_from_selected(n_clicks, selected_rows, motl_type, registry, _si=slot_idx, _mm=max_motls):
                if not n_clicks or not selected_rows:
                    raise exceptions.PreventUpdate

                registry = registry or {}
                target = None
                for i in range(_mm):
                    if str(i) not in registry or not registry[str(i)].get("active"):
                        target = i
                        break

                nones = [no_update] * _mm
                if target is None:
                    return (*nones, *nones, no_update, no_update)

                data_out = list(nones)
                type_out = list(nones)
                data_out[target] = selected_rows
                type_out[target] = motl_type

                source_label = registry.get(str(_si), {}).get("label", f"Slot {_si + 1}")
                short = source_label[:15] + "…" if len(source_label) > 15 else source_label
                new_label = f"Sel from {short} ({len(selected_rows)})"

                new_registry = dict(registry)
                new_registry[str(target)] = {"label": new_label, "active": True}

                return (*data_out, *type_out, new_registry, f"me-tab-{target}")
