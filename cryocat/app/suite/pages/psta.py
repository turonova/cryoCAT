"""STA tool — subtomogram-averaging iteration analysis.

Three input paths:
  * **From parameter file** — load a novaSTA (.txt) or STOPGAP (.star) parameter
    file; the iteration range, motl base name and motl type are read from the
    file automatically, and the per-iteration motls are loaded from disk.
  * **Iteration batch (bulk loader)** — given a filename core + iteration count,
    loads per-iteration motls from disk.
  * ``get_motl_source("sta-single")`` — pick one motl from the suite pool.

The main area shows aggregate plots over the batch (particle count and score
statistics per iteration) and, when a parameter file was loaded, a table of the
parameter values stored in the file.

Contract: exposes ``layout`` and ``register_callbacks(app)``.
"""

import os

import dash
import dash_ag_grid as dag
from dash import html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from cryocat.core.cryomotl import Motl
from cryocat.analysis.sta import StaParameters, get_motl_filename
from cryocat.app.components.motlsource import get_motl_source, register_motl_source_callbacks
from cryocat.app.components.motlsink import get_send_to_editor_button, register_send_to_editor_callbacks
from cryocat.app.components.logpanel import get_log_panel, register_log_panel_callbacks


# ── Layout helpers ───────────────────────────────────────────────────────────────

def _param_file_loader():
    lbl = {"fontSize": "0.85rem", "marginBottom": "2px"}
    return html.Div(
        [
            html.Label("Parameter file path", style=lbl),
            dbc.Input(
                id="sta-param-path",
                type="text",
                placeholder="path/to/params.txt  or  .star",
                size="sm",
                style={"marginBottom": "0.4rem"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("File type", style=lbl),
                            dcc.Dropdown(
                                id="sta-param-type",
                                options=[
                                    {"label": "Auto-detect", "value": "auto"},
                                    {"label": "novaSTA (.txt)", "value": "novasta"},
                                    {"label": "STOPGAP (.star)", "value": "stopgap"},
                                ],
                                value="auto",
                                clearable=False,
                                searchable=False,
                                style={"fontSize": "0.8rem"},
                            ),
                        ],
                        width=7,
                    ),
                    dbc.Col(
                        [
                            html.Label("Motl separator", style=lbl),
                            dbc.Input(id="sta-param-sep", type="text", value="_", size="sm"),
                        ],
                        width=5,
                    ),
                ],
                className="g-1",
                style={"marginBottom": "0.4rem"},
            ),
            dbc.Button(
                "Load from parameter file",
                id="sta-param-load-btn",
                color="primary",
                size="sm",
                style={"width": "100%", "marginTop": "0.25rem"},
            ),
            html.Div(
                id="sta-param-status",
                style={
                    "fontSize": "0.8rem",
                    "color": "var(--color9)",
                    "marginTop": "0.4rem",
                    "wordBreak": "break-word",
                },
            ),
        ]
    )


def _bulk_loader():
    lbl = {"fontSize": "0.85rem", "marginBottom": "2px"}
    return html.Div(
        [
            html.Label("Filename core (path + prefix)", style=lbl),
            dbc.Input(
                id="sta-core-input",
                type="text",
                placeholder="e.g. /data/run1/allmotl_",
                size="sm",
                style={"marginBottom": "0.4rem"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Iterations", style=lbl),
                            dbc.Input(id="sta-iter-count", type="number", min=1, step=1, value=10, size="sm"),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Label("Extension", style=lbl),
                            dbc.Input(id="sta-ext-input", type="text", value=".em", size="sm"),
                        ],
                        width=6,
                    ),
                ],
                className="g-1",
            ),
            dbc.Button(
                "Load iteration batch",
                id="sta-load-btn",
                color="primary",
                size="sm",
                style={"width": "100%", "marginTop": "0.5rem"},
            ),
            html.Div(
                id="sta-load-status",
                style={"fontSize": "0.8rem", "color": "var(--color9)", "marginTop": "0.4rem", "wordBreak": "break-word"},
            ),
        ]
    )


def _sidebar():
    return dbc.Col(
        html.Div(
            [
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            _param_file_loader(),
                            title="From parameter file",
                            item_id="sta-acc-params",
                        ),
                        dbc.AccordionItem(
                            _bulk_loader(),
                            title="Iteration batch (bulk loader)",
                            item_id="sta-acc-batch",
                        ),
                        dbc.AccordionItem(
                            get_motl_source("sta-single"),
                            title="Single motl (from pool)",
                            item_id="sta-acc-single",
                        ),
                        dbc.AccordionItem(
                            get_send_to_editor_button("sta"),
                            title="Send result to editor",
                            item_id="sta-acc-output",
                        ),
                    ],
                    always_open=True,
                    active_item=["sta-acc-params"],
                ),
                html.Div(
                    dbc.Button(
                        "Show log",
                        id="sta-open-log-btn",
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


def _params_table_section():
    """Collapsible section that shows the parameter df when a param file is loaded."""
    return dbc.Collapse(
        html.Div(
            [
                html.H6("Parameter file contents", style={"marginBottom": "0.3rem"}),
                dag.AgGrid(
                    id="sta-params-grid",
                    rowData=[],
                    columnDefs=[],
                    defaultColDef={
                        "resizable": True,
                        "sortable": True,
                        "filter": False,
                        "flex": 1,
                        "minWidth": 80,
                    },
                    style={"height": "220px"},
                    className="ag-theme-balham",
                    dashGridOptions={"suppressMovableColumns": True},
                ),
                html.Hr(style={"margin": "0.5rem 0"}),
            ],
            style={"marginBottom": "0.5rem"},
        ),
        id="sta-params-collapse",
        is_open=False,
    )


def _main():
    return dbc.Col(
        html.Div(
            [
                _params_table_section(),
                dcc.Graph(id="sta-progress-graph"),
                html.Hr(style={"margin": "0.5rem 0"}),
                dcc.Graph(id="sta-score-graph"),
            ],
            style={"padding": "0.5rem"},
        ),
        width=9,
        style={"margin": "0", "padding": "0"},
    )


layout = html.Div(
    [
        dcc.Store(id="sta-batch"),         # per-iteration aggregate summaries
        dcc.Store(id="sta-result"),        # future: edited motl for editor
        dcc.Store(id="sta-params-store"),  # serialised params df for the table
        dbc.Row([_sidebar(), _main()], className="g-0", style={"margin": "0", "padding": "0"}),
        *get_log_panel("sta-log"),
    ],
    style={"margin": "0", "padding": "0"},
)


# ── Callbacks ────────────────────────────────────────────────────────────────────

def register_callbacks(app):
    register_motl_source_callbacks(app, "sta-single")
    register_send_to_editor_callbacks(app, "sta", "sta-result")
    register_log_panel_callbacks(app, "sta-log", open_btn_id="sta-open-log-btn")

    # ── Load from parameter file ──────────────────────────────────────────────

    @app.callback(
        Output("sta-batch", "data", allow_duplicate=True),
        Output("sta-params-store", "data"),
        Output("sta-param-status", "children"),
        Input("sta-param-load-btn", "n_clicks"),
        State("sta-param-path", "value"),
        State("sta-param-type", "value"),
        State("sta-param-sep", "value"),
        prevent_initial_call=True,
    )
    def load_from_params(n_clicks, path, sta_type, separator):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not path:
            return no_update, no_update, "Provide a parameter file path."

        separator = separator or "_"
        sta_type_arg = None if sta_type == "auto" else sta_type

        try:
            params = StaParameters.load(path.strip(), sta_type=sta_type_arg)
        except Exception as exc:
            return no_update, no_update, f"Failed to load parameter file: {exc}"

        motl_base = params.get_motl_base_name(separator)
        if motl_base is None:
            return no_update, no_update, "Parameter file has no motl path — cannot locate motl files."

        motl_type = params.motl_type
        start_it  = params.start_iteration
        end_it    = params.end_iteration

        if start_it is None or end_it is None:
            return no_update, no_update, "Parameter file contains no alignment iterations."

        batch = []
        missing = []
        for i in range(start_it, end_it + 1):
            fname = get_motl_filename(motl_base, i, motl_type)
            if not os.path.isfile(fname):
                missing.append(i)
                continue
            try:
                motl = Motl.load(fname, motl_type=motl_type)
                df = motl.df
                batch.append(
                    {
                        "iteration": i,
                        "n_particles": int(df.shape[0]),
                        "score_mean":   float(df["score"].mean())   if "score" in df else None,
                        "score_median": float(df["score"].median()) if "score" in df else None,
                        "score_std":    float(df["score"].std())    if "score" in df else None,
                    }
                )
            except Exception:
                missing.append(i)

        # Serialise params df for the table
        params_data = {
            "records": params.df.to_dict("records"),
            "columns": list(params.df.columns),
        }

        if not batch:
            status = (
                f"No motl files found for iterations {start_it}–{end_it} "
                f"at '{motl_base}*' (type: {motl_type})."
            )
            return no_update, params_data, status

        status = (
            f"Loaded {len(batch)} iteration motl(s) "
            f"[{motl_type}, iterations {start_it}–{end_it}]."
        )
        if missing:
            status += f" Missing: {missing}."
        return batch, params_data, status

    # ── Populate params table ─────────────────────────────────────────────────

    @app.callback(
        Output("sta-params-grid", "rowData"),
        Output("sta-params-grid", "columnDefs"),
        Output("sta-params-collapse", "is_open"),
        Input("sta-params-store", "data"),
        prevent_initial_call=True,
    )
    def update_params_table(data):
        if not data or not data.get("columns"):
            return [], [], False
        col_defs = [
            {"field": c, "headerName": c, "flex": 1, "minWidth": 80}
            for c in data["columns"]
        ]
        return data["records"], col_defs, True

    # ── Load iteration batch (existing) ───────────────────────────────────────

    @app.callback(
        Output("sta-batch", "data", allow_duplicate=True),
        Output("sta-load-status", "children"),
        Input("sta-load-btn", "n_clicks"),
        State("sta-core-input", "value"),
        State("sta-iter-count", "value"),
        State("sta-ext-input", "value"),
        prevent_initial_call=True,
    )
    def load_batch(n_clicks, core, count, ext):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not core or not count:
            return no_update, "Provide a filename core and an iteration count."

        ext = ext or ".em"
        if not ext.startswith("."):
            ext = "." + ext

        batch = []
        missing = []
        for i in range(1, int(count) + 1):
            path = f"{core}{i}{ext}"
            if not os.path.isfile(path):
                missing.append(i)
                continue
            try:
                motl = Motl.load(path)
                df = motl.df
                batch.append(
                    {
                        "iteration": i,
                        "n_particles": int(df.shape[0]),
                        "score_mean":   float(df["score"].mean())   if "score" in df else None,
                        "score_median": float(df["score"].median()) if "score" in df else None,
                        "score_std":    float(df["score"].std())    if "score" in df else None,
                    }
                )
            except Exception:
                missing.append(i)

        if not batch:
            return no_update, f"No iteration motls loaded (checked {count} files at '{core}*{ext}')."

        status = f"Loaded {len(batch)} iteration motl(s)."
        if missing:
            status += f" Skipped iterations: {missing}."
        return batch, status

    # ── Plot batch (shared by both loaders) ───────────────────────────────────

    @app.callback(
        Output("sta-progress-graph", "figure"),
        Output("sta-score-graph", "figure"),
        Input("sta-batch", "data"),
        State("graph-settings-store", "data"),
        prevent_initial_call=True,
    )
    def plot_batch(batch, settings):
        if not batch:
            raise dash.exceptions.PreventUpdate

        iters       = [row["iteration"]  for row in batch]
        n_particles = [row["n_particles"] for row in batch]
        score_mean  = [row.get("score_mean")  for row in batch]
        score_std   = [row.get("score_std") or 0.0 for row in batch]

        progress = go.Figure(
            go.Scatter(x=iters, y=n_particles, mode="lines+markers", name="Particles")
        )
        progress.update_layout(
            title="Particle count per iteration",
            xaxis_title="Iteration",
            yaxis_title="Number of particles",
            height=350,
            margin=dict(t=40, b=40, l=50, r=20),
        )

        score = go.Figure(
            go.Scatter(
                x=iters,
                y=score_mean,
                mode="lines+markers",
                name="Mean score",
                error_y=dict(type="data", array=score_std, visible=True),
            )
        )
        score.update_layout(
            title="Mean score per iteration (±1 std)",
            xaxis_title="Iteration",
            yaxis_title="Score",
            height=350,
            margin=dict(t=40, b=40, l=50, r=20),
        )
        return progress, score
