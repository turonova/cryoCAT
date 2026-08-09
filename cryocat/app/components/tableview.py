"""Table component — composition of grid, filter, edit, and save submodules."""
from __future__ import annotations

from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

from cryocat.app.components.tableplot import get_table_plot_component
from cryocat.app.components.tablecluster import get_table_cluster_component
from cryocat.app.components.tablegrid import get_grid, register_tablegrid_callbacks
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

    extra_children = [get_csv_save_modal(prefix)]

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
                style={"color": "var(--color9)", "whiteSpace": "nowrap", "fontSize": "0.85rem"},
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
            get_grid(prefix),
            html.Div(
                id=f"{prefix}-active-filter-count",
                style={"fontSize": "0.85rem", "color": "var(--color9)", "marginTop": "4px", "marginBottom": "4px"},
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
    extra_csv_states=None,
    custom_csv_save_fn=None,
):
    """Register grid, filter, edit and CSV-save callbacks for *prefix*.

    To add motl-mode saves, also call ``register_table_save_callbacks``.
    """
    register_tablegrid_callbacks(app, prefix)
    register_tablefilter_callbacks(app, prefix)
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
