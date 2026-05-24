"""STA tool — subtomogram-averaging iteration analysis.

Two input paths:
  * a **pattern bulk-loader** — given a filename core + iteration count, loads
    the per-iteration motls from disk and stores per-iteration aggregate
    summaries in the page-local ``sta-batch`` store. This batch is never
    surfaced as per-motl table/viewer surfaces (it is aggregate-plots-only).
  * ``get_motl_source("sta-single")`` — pick one motl from the suite pool.

The main area shows aggregate plots over the batch (particle count and score
statistics per iteration).

Contract: exposes ``layout`` and ``register_callbacks(app)``.
"""

import os

import dash
from dash import html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from cryocat.core.cryomotl import Motl
from cryocat.app.components.motlsource import get_motl_source, register_motl_source_callbacks
from cryocat.app.components.motlsink import get_send_to_editor_button, register_send_to_editor_callbacks
from cryocat.app.components.logpanel import get_log_panel, register_log_panel_callbacks


# ── Layout ──────────────────────────────────────────────────────────────────────

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
                    active_item=["sta-acc-batch", "sta-acc-single"],
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


def _main():
    return dbc.Col(
        html.Div(
            [
                html.H4("STA — Iteration Analysis", style={"marginBottom": "1rem"}),
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
        dcc.Store(id="sta-batch"),     # per-iteration aggregate summaries (never surfaced)
        dcc.Store(id="sta-result"),    # future: an edited motl to hand to the editor
        dbc.Row([_sidebar(), _main()], className="g-0", style={"margin": "0", "padding": "0"}),
        *get_log_panel("sta-log"),
    ],
    style={"margin": "0", "padding": "0"},
)


# ── Callbacks ───────────────────────────────────────────────────────────────────

def register_callbacks(app):
    register_motl_source_callbacks(app, "sta-single")
    register_send_to_editor_callbacks(app, "sta", "sta-result")
    register_log_panel_callbacks(app, "sta-log")

    @app.callback(
        Output("sta-batch", "data"),
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
                        "score_mean": float(df["score"].mean()) if "score" in df else None,
                        "score_median": float(df["score"].median()) if "score" in df else None,
                        "score_std": float(df["score"].std()) if "score" in df else None,
                    }
                )
            except Exception as exc:
                missing.append(i)

        if not batch:
            return no_update, f"No iteration motls loaded (checked {count} files at '{core}*{ext}')."

        status = f"Loaded {len(batch)} iteration motl(s)."
        if missing:
            status += f" Skipped iterations: {missing}."
        return batch, status

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

        iters = [row["iteration"] for row in batch]
        n_particles = [row["n_particles"] for row in batch]
        score_mean = [row.get("score_mean") for row in batch]
        score_std = [row.get("score_std") or 0.0 for row in batch]

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
