"""Table component — composition of grid, filter, edit, and save submodules."""
from __future__ import annotations

from dash import html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from cryocat.app.components.tableplot import get_table_plot_component
from cryocat.app.components.tablecluster import get_table_cluster_component
from cryocat.app.components.tablegrid import get_grid_container, register_tablegrid_callbacks
from cryocat.app.components.tablefilter import register_tablefilter_callbacks
from cryocat.app.components.tableedit import register_tableedit_callbacks
from cryocat.app.components.tablesave import (
    get_csv_save_modal,
    get_motl_save_modal,
    get_overwrite_confirm_modal,
    register_tablesave_csv_callbacks,
    register_table_save_callbacks,
)


def get_table_component(
    prefix: str,
    connected_motl_prefix=None,
    show_create_from_selected=True,
    save_dialog_prefix: str | None = None,
    show_editor: bool = False,
):
    motl_mode = connected_motl_prefix is not None
    use_save_dialog = motl_mode and save_dialog_prefix is not None

    button_children = [
        dbc.Button("Apply Changes", id=f"{prefix}-apply-btn", color="primary", className="me-1"),
    ]

    if motl_mode:
        button_children.append(
            dbc.Button("Save As", id=f"{prefix}-save-btn", color="primary", className="me-1"),
        )
        if not use_save_dialog:
            button_children.append(
                dbc.Button("Save", id=f"{prefix}-save-overwrite-btn", color="secondary", className="me-1"),
            )
        if show_create_from_selected:
            button_children.append(
                dbc.Button(
                    "Create new from selected",
                    id=f"{prefix}-create-from-selected-btn",
                    color="secondary",
                    className="me-1",
                )
            )

    button_children += [
        dbc.Button("Save as CSV", id=f"{prefix}-save-csv-btn", color="primary", className="me-1"),
        dbc.Button("Remove Selected Rows", id=f"{prefix}-remove-rows-btn", color="primary", className="me-1"),
        dbc.Button("Select All Filtered", id=f"{prefix}-select-all-btn", color="secondary", className="me-1"),
        dbc.Button(
            "Create from filtered",
            id=f"{prefix}-pool-from-filtered-btn",
            color="secondary",
            className="me-1",
            disabled=True,
            style={"display": "none"},
        ),
        dbc.Button(
            "Create from selected",
            id=f"{prefix}-pool-from-selected-btn",
            color="secondary",
            className="me-1",
            disabled=True,
            style={"display": "none"},
        ),
        dbc.Button("Select Inverse", id=f"{prefix}-select-inverse-btn", color="secondary", className="me-1"),
        dbc.Button("Plot", id=f"{prefix}-plot-graphs-btn", color="primary", className="me-1", n_clicks=0),
        dbc.Offcanvas(
            [get_table_plot_component(f"{prefix}-table-plot")],
            id=f"{prefix}-plot-graph-panel",
            title="Plotting options",
            placement="end",
            scrollable=True,
            style={"width": "1100px"},
            is_open=False,
        ),
        dbc.Button("Cluster", id=f"{prefix}-cluster-btn", color="primary", className="me-1", n_clicks=0),
        dbc.Offcanvas(
            [get_table_cluster_component(f"{prefix}-table-cluster")],
            id=f"{prefix}-cluster-panel",
            title="Clustering options",
            placement="end",
            scrollable=True,
            style={"width": "700px"},
            is_open=False,
        ),
    ]

    if show_editor:
        from cryocat.app.components import tableeditor
        button_children.insert(
            0,
            dbc.Button(
                "Edit table",
                id=f"{prefix}-edit-open-btn",
                color="secondary",
                size="sm",
                className="me-1",
            ),
        )

    extra_children = [get_csv_save_modal(prefix)]

    if show_editor:
        from cryocat.app.components import tableeditor
        extra_children.append(
            dbc.Offcanvas(
                [tableeditor.get_table_editor(f"{prefix}-edit")],
                id=f"{prefix}-edit-offcanvas",
                title="Edit table",
                placement="end",
                scrollable=True,
                style={"width": "520px"},
                is_open=False,
            )
        )

    if use_save_dialog:
        from cryocat.app.components.savedialog import get_save_dialog
        extra_children.append(
            dbc.Offcanvas(
                [get_save_dialog(save_dialog_prefix, mode="single")],
                id=f"{save_dialog_prefix}-offcanvas",
                title="Save Motl",
                placement="end",
                scrollable=True,
                style={"width": "500px"},
                is_open=False,
            )
        )
    elif motl_mode:
        extra_children += [
            get_motl_save_modal(prefix),
            get_overwrite_confirm_modal(prefix),
            dcc.Store(id=f"{prefix}-last-save-params-store"),
        ]

    button_row_children = []
    if motl_mode:
        button_row_children.append(
            dbc.Col(
                html.Div(
                    id=f"{connected_motl_prefix}-relion-params-inline",
                    style={"color": "var(--color9)", "whiteSpace": "nowrap"},
                ),
                width="auto",
                className="d-flex align-items-center",
            )
        )
    button_row_children.append(
        dbc.Col(
            html.Div(
                id=f"{prefix}-selection-count",
                style={"color": "var(--color9)", "whiteSpace": "nowrap"},
            ),
            width="auto",
            className="d-flex align-items-center",
        )
    )
    button_row_children.append(
        dbc.Col(button_children, className="d-flex justify-content-end flex-wrap gap-1")
    )

    return html.Div(
        id=f"{prefix}-table-container",
        children=[
            dbc.Row(button_row_children, className="mb-2"),
            get_grid_container(prefix),
            html.Div(
                id=f"{prefix}-active-filter-count",
                style={"color": "var(--color9)", "marginTop": "4px", "marginBottom": "4px"},
            ),
            html.H5("Filters", style={"marginBottom": "1rem", "marginTop": "1rem"}),
            html.Div(id=f"{prefix}-filters-container", style={"marginBottom": "2rem"}),
            *extra_children,
            dcc.Store(id=f"{prefix}-snapshot-store"),
            dcc.Store(id=f"{prefix}-selection-ids-store", data=[]),
        ],
    )


def register_table_callbacks(
    app,
    prefix: str,
    *,
    resolve_df,
    resolve_n_rows,
    tabs_id: str | None = None,
    tab_value: str | None = None,
    extra_csv_states=None,
    custom_csv_save_fn=None,
    show_editor: bool = False,
):
    """Register grid, filter, edit and CSV-save callbacks for *prefix*.

    To add motl-mode saves, also call ``register_table_save_callbacks``.
    resolve_df and resolve_n_rows are forwarded to register_tablegrid_callbacks;
    pass :func:`cryocat.app.pool.resolve_df` / :func:`~resolve_n_rows` for
    pool-backed grids.  tabs_id / tab_value gate _mount_grid on the active tab.
    When show_editor=True, also registers the table editor and its modal toggle.
    """
    register_tablegrid_callbacks(
        app, prefix,
        resolve_df=resolve_df, resolve_n_rows=resolve_n_rows,
        tabs_id=tabs_id, tab_value=tab_value,
    )
    register_tablefilter_callbacks(app, prefix, resolve_df=resolve_df)
    register_tableedit_callbacks(app, prefix)
    register_tablesave_csv_callbacks(
        app, prefix,
        extra_csv_states=extra_csv_states,
        custom_csv_save_fn=custom_csv_save_fn,
    )

    @app.callback(
        Output(f"{prefix}-plot-graph-panel", "is_open", allow_duplicate=True),
        Input(f"{prefix}-plot-graphs-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_offcanvas(_):
        return True

    @app.callback(
        Output(f"{prefix}-cluster-panel", "is_open", allow_duplicate=True),
        Input(f"{prefix}-cluster-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_cluster_panel(_):
        return True

    if show_editor:
        from cryocat.app.components import tableeditor

        @app.callback(
            Output(f"{prefix}-edit-offcanvas", "is_open", allow_duplicate=True),
            Input(f"{prefix}-edit-open-btn", "n_clicks"),
            prevent_initial_call=True,
        )
        def open_edit_offcanvas(_):
            return True

        # Pre-select the slot's current entry in the picker when the offcanvas opens
        @app.callback(
            Output(f"{prefix}-edit-src-dd", "value", allow_duplicate=True),
            Input(f"{prefix}-edit-offcanvas", "is_open"),
            State(f"{prefix}-global-data-store", "data"),
            prevent_initial_call=True,
        )
        def _preset_editor_source(is_open, ref):
            if not is_open or not ref:
                return no_update
            if "motl_id" in (ref or {}):
                mid = ref["motl_id"]
                if mid == "dp-view":
                    return no_update
                return f"motl:{mid}"
            if "data_id" in (ref or {}):
                return f"data:{ref['data_id']}"
            return no_update

        tableeditor.register_table_editor_callbacks(app, f"{prefix}-edit")

