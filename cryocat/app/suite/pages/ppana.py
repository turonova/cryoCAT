"""Peak Analysis (pana) tool.

Form-heavy tool: the detailed :func:`cryocat.analysis.pana.run_analysis`
parameter form lives in the (wide) main area; the sidebar holds the
pool-aware extraction motl source plus the Run / summary-save actions.

* Extraction input — ``get_motl_source("pana-extract", show_table=True)`` lets
  the user inspect pool entries and pick which motl to extract from.
* Run — calls :func:`cryocat.analysis.pana.run_analysis` over a template-list
  CSV; the (in-place updated) template list is shown via a tableview.
* Summary — calls :func:`cryocat.analysis.pana.create_summary_pdf`.

Peak Analysis produces CSVs/PDFs, not motls — so this page does **not** use
``motlsink``.

Contract: exposes ``layout`` and ``register_callbacks(app)``.
"""

import dash
from dash import html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd

from cryocat.analysis import pana
from cryocat.app.components.motlsource import get_motl_source, register_motl_source_callbacks
from cryocat.app.components.tableview import get_table_component, register_table_callbacks
from cryocat.app.components.tableplot import register_table_plot_callbacks
from cryocat.app.components.logpanel import get_log_panel, register_log_panel_callbacks


# ── Layout ──────────────────────────────────────────────────────────────────────

def _path_input(label, comp_id, placeholder):
    return html.Div(
        [
            html.Label(label, style={"fontSize": "0.85rem", "marginBottom": "2px"}),
            dbc.Input(id=comp_id, type="text", placeholder=placeholder, size="sm"),
        ],
        style={"marginBottom": "0.5rem"},
    )


def _run_analysis_form():
    return html.Div(
        [
            html.H5("run_analysis parameters", style={"marginBottom": "0.75rem"}),
            _path_input("Template list CSV", "pana-template-list", "path to the template-list .csv"),
            _path_input("Indices (comma-separated, 0-based)", "pana-indices", "e.g. 0,1,2  (empty = all rows)"),
            _path_input("Angle list path", "pana-angle-path", "directory with angle-list files"),
            _path_input("Wedge path", "pana-wedge-path", "directory with wedge-mask files"),
            _path_input("Parent folder path", "pana-parent-path", "root folder with structures/tomograms/masks"),
            html.Div(
                [
                    html.Label("CC radius tolerance (voxels)", style={"fontSize": "0.85rem", "marginBottom": "2px"}),
                    dbc.Input(id="pana-cc-radius", type="number", min=1, step=1, value=10, size="sm"),
                ],
                style={"marginBottom": "0.5rem", "maxWidth": "260px"},
            ),
        ],
        style={"maxWidth": "640px"},
    )


def _sidebar():
    return dbc.Col(
        html.Div(
            [
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            get_motl_source("pana-extract", show_table=True),
                            title="Extraction motl source",
                            item_id="pana-acc-extract",
                        ),
                        dbc.AccordionItem(
                            [
                                dbc.Button(
                                    "Run analysis",
                                    id="pana-run-btn",
                                    color="primary",
                                    size="sm",
                                    style={"width": "100%", "marginBottom": "0.4rem"},
                                ),
                                dbc.Button(
                                    "Save summary PDF",
                                    id="pana-summary-save-btn",
                                    color="secondary",
                                    size="sm",
                                    style={"width": "100%"},
                                ),
                                html.Div(
                                    id="pana-run-status",
                                    style={
                                        "fontSize": "0.8rem",
                                        "color": "var(--color9)",
                                        "marginTop": "0.5rem",
                                        "wordBreak": "break-word",
                                    },
                                ),
                            ],
                            title="Run / Save",
                            item_id="pana-acc-run",
                        ),
                    ],
                    always_open=True,
                    active_item=["pana-acc-extract", "pana-acc-run"],
                ),
                html.Div(
                    dbc.Button(
                        "Show log",
                        id="pana-open-log-btn",
                        className="custom-radius-button",
                        style={"width": "100%"},
                    ),
                    style={"padding": "0.5rem", "marginTop": "auto"},
                ),
            ],
            className="sidebar",
            style={
                "padding": "0.5rem",
                "overflowY": "auto",
                "height": "100vh",
                "display": "flex",
                "flexDirection": "column",
            },
        ),
        width=3,
        style={"margin": "0", "padding": "0", "height": "100vh", "position": "sticky", "top": "0px"},
    )


def _main():
    return dbc.Col(
        html.Div(
            [
                html.H4("Peak Analysis", style={"marginBottom": "1rem"}),
                _run_analysis_form(),
                html.Hr(style={"margin": "0.75rem 0"}),
                html.H5("Template list / results", style={"marginBottom": "0.5rem"}),
                dcc.Store(id="pana-out-tabv-global-data-store"),
                get_table_component("pana-out-tabv"),
            ],
            style={"padding": "0.5rem"},
        ),
        width=9,
        style={"margin": "0", "padding": "0"},
    )


layout = html.Div(
    [
        dbc.Row([_sidebar(), _main()], className="g-0", style={"margin": "0", "padding": "0"}),
        *get_log_panel("pana-log"),
    ],
    style={"margin": "0", "padding": "0"},
)


# ── Callbacks ───────────────────────────────────────────────────────────────────

def _parse_indices(indices_str, template_list):
    """Parse the comma-separated indices field; empty -> all rows of the CSV."""
    if indices_str and indices_str.strip():
        return [int(x) for x in indices_str.split(",") if x.strip()]
    return list(range(len(pd.read_csv(template_list))))


def register_callbacks(app):
    register_motl_source_callbacks(app, "pana-extract", show_table=True)
    register_table_callbacks(app, "pana-out-tabv", csv_only=True)
    register_table_plot_callbacks(
        app, "pana-out-tabv-table-plot", "pana-out-tabv-global-data-store", table_grid_id="pana-out-tabv-grid"
    )
    register_log_panel_callbacks(app, "pana-log")

    @app.callback(
        Output("pana-out-tabv-global-data-store", "data"),
        Output("pana-run-status", "children", allow_duplicate=True),
        Input("pana-run-btn", "n_clicks"),
        State("pana-template-list", "value"),
        State("pana-indices", "value"),
        State("pana-angle-path", "value"),
        State("pana-wedge-path", "value"),
        State("pana-parent-path", "value"),
        State("pana-cc-radius", "value"),
        prevent_initial_call=True,
    )
    def run_analysis(n_clicks, template_list, indices_str, angle_path, wedge_path, parent_path, cc_radius):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not (template_list and angle_path and wedge_path and parent_path):
            return no_update, "Provide the template list CSV and the angle / wedge / parent folder paths."

        try:
            indices = _parse_indices(indices_str, template_list)
            pana.run_analysis(
                template_list,
                indices,
                angle_path,
                wedge_path,
                parent_path,
                cc_radius_tol=cc_radius or 10,
            )
            # run_analysis updates the template-list CSV in place — show it.
            df = pd.read_csv(template_list)
            return df.to_dict("records"), f"Analysis complete — processed {len(indices)} row(s)."
        except Exception as exc:
            return no_update, f"Error: {exc}"

    @app.callback(
        Output("pana-run-status", "children", allow_duplicate=True),
        Input("pana-summary-save-btn", "n_clicks"),
        State("pana-template-list", "value"),
        State("pana-indices", "value"),
        State("pana-parent-path", "value"),
        prevent_initial_call=True,
    )
    def save_summary(n_clicks, template_list, indices_str, parent_path):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not (template_list and parent_path):
            return "Provide the template list CSV and the parent folder path."
        try:
            indices = _parse_indices(indices_str, template_list)
            pana.create_summary_pdf(template_list, indices, parent_path)
            return f"Summary PDF written for {len(indices)} row(s)."
        except Exception as exc:
            return f"Error: {exc}"
