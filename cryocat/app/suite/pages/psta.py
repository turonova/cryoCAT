"""STA tool — subtomogram-averaging iteration analysis and setup.

Sidebar accordion (top-level menu):

  * **Evaluation** — either-or loader (parameter file or motl base + iter
    range).  Once a config is loaded, the two action buttons dispatch:

    - *Alignment evaluation* -> ``sta.compute_alignment_statistics`` ->
      every column of the result rendered as a line plot in a 3-column
      grid (tab "Alignment evaluation").
    - *Classification evaluation* -> ``sta.evaluate_classification`` ->
      ``visplot.plot_classification_convergence`` (tab "Classification
      evaluation").

  * **STA setup** — full parameter-file form (novaSTA convention).  A
    "Create" button builds the parameter DataFrame and shows it in tab 4
    ("STA setup output") where the user can edit cells and save as either
    novaSTA (.txt) or STOPGAP (.star).  Angle parameters can be picked
    via an embedded preview of :func:`cryocat.utils.geom.generate_angles`
    (visualisation only -- no file is created from the angles panel
    itself) and copied into the form with the "Use these angles" button.

  * **Classification setup** — placeholder, set up later.

Main area: ``dcc.Tabs`` with four tabs (Parameter file table /
Alignment / Classification / STA setup output).

Contract: exposes ``layout`` and ``register_callbacks(app)``.
"""
from __future__ import annotations

import math

import dash
import dash_ag_grid as dag
from dash import html, dcc, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cryocat.analysis import sta as sta_mod
from cryocat.analysis import visplot
from cryocat.app import formgen, ids
from cryocat.app.components.anglesbuilder import register_angles_builder_callbacks
from cryocat.app.components.graphsettings import styled_figure, error_figure
from cryocat.utils.geom import generate_angles
from cryocat.app.pageshell import page_shell, sidebar_accordion


_HINT = {"fontSize": "0.8rem", "color": "var(--color9)", "margin": "0.3rem 0"}
_LBL = {"fontSize": "0.85rem", "marginBottom": "2px"}
_PLACEHOLDER = "Coming later — set up by the user."
_SECTION_HEADER = {"fontSize": "0.9rem", "fontWeight": 600,
                   "margin": "0.4rem 0 0.2rem"}

# Motl types accepted by the direct loader.
_MOTL_TYPE_OPTIONS = [
    {"label": "emmotl (.em)", "value": "emmotl"},
    {"label": "stopgap (.star)", "value": "stopgap"},
    {"label": "relion (.star)", "value": "relion"},
    {"label": "relion5 (.star)", "value": "relion5"},
    {"label": "relion5_1 (.star)", "value": "relion5_1"},
]

# Angles-builder prefix used by the slim panel + register_angles_builder_callbacks.
_ANGLES_PREFIX = "sta-setup-angles"


# ── Small form-field helpers ─────────────────────────────────────────────────
#
# Fields render in a single horizontal row: ``[label  |  input]`` so the
# sidebar form is dense and the label aligns with its control.  Checkboxes
# keep their inline ``dbc.Checkbox`` label (which is already next to the
# box) so the look matches the rest of the suite.

_FIELD_ROW = {
    "display": "flex",
    "alignItems": "center",
    "gap": "0.5rem",
    "marginBottom": "0.4rem",
}
_FIELD_LABEL = {**_LBL, "flex": "0 0 45%", "margin": 0, "alignSelf": "center"}
_FIELD_INPUT = {"flex": "1 1 0", "minWidth": "0"}


def _field_text(label: str, id_: str, placeholder: str = "") -> html.Div:
    return html.Div(
        [
            html.Label(label, style=_FIELD_LABEL),
            dbc.Input(id=id_, type="text", placeholder=placeholder, size="sm",
                      style=_FIELD_INPUT),
        ],
        style=_FIELD_ROW,
    )


def _field_num(label: str, id_: str, value=None, **kwargs) -> html.Div:
    return html.Div(
        [
            html.Label(label, style=_FIELD_LABEL),
            dbc.Input(id=id_, type="number", value=value, size="sm",
                      style=_FIELD_INPUT, **kwargs),
        ],
        style=_FIELD_ROW,
    )


def _field_check(label: str, id_: str, value: bool = False) -> dbc.Checkbox:
    return dbc.Checkbox(id=id_, label=label, value=value,
                        style={"marginBottom": "0.4rem"})


# ── Evaluation accordion panel ───────────────────────────────────────────────


def _param_file_form() -> html.Div:
    return html.Div(
        [
            html.Label("Parameter file path", style=_LBL),
            dbc.Input(
                id="sta-param-path",
                type="text",
                placeholder="path/to/params.txt  or  .star",
                size="sm",
                style={"marginBottom": "0.4rem"},
            ),
            html.Label("Working directory (optional override)", style=_LBL),
            dbc.Input(
                id="sta-param-working-dir",
                type="text",
                placeholder="leave blank to use rootdir / stored path",
                size="sm",
                style={"marginBottom": "0.4rem"},
            ),
            html.Div(
                "STOPGAP: replaces the params' rootdir (lists/, masks/, refs/ "
                "subdirs are still appended). novaSTA: joins onto relative motl "
                "paths and replaces the directory portion of absolute ones.",
                style={
                    "fontSize": "0.75rem", "color": "var(--color9)",
                    "marginBottom": "0.5rem",
                },
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("File type", style=_LBL),
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
                            html.Label("Motl separator", style=_LBL),
                            dbc.Input(id="sta-param-sep", type="text",
                                      value="_", size="sm"),
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
                    "fontSize": "0.8rem", "color": "var(--color9)",
                    "marginTop": "0.4rem", "wordBreak": "break-word",
                },
            ),
        ],
        id="sta-eval-params-block",
    )


def _direct_form() -> html.Div:
    return html.Div(
        [
            html.Label("Motl base name (path + prefix)", style=_LBL),
            dbc.Input(
                id="sta-direct-base",
                type="text",
                placeholder="e.g. /data/run1/allmotl_",
                size="sm",
                style={"marginBottom": "0.4rem"},
            ),
            html.Label("Motl type", style=_LBL),
            dcc.Dropdown(
                id="sta-direct-type",
                options=_MOTL_TYPE_OPTIONS,
                value="emmotl",
                clearable=False,
                searchable=False,
                style={"fontSize": "0.8rem", "marginBottom": "0.4rem"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Starting iteration", style=_LBL),
                            dbc.Input(id="sta-direct-start", type="number",
                                      min=0, step=1, value=1, size="sm"),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Label("Number of iterations", style=_LBL),
                            dbc.Input(id="sta-direct-count", type="number",
                                      min=1, step=1, value=10, size="sm"),
                        ],
                        width=6,
                    ),
                ],
                className="g-1",
                style={"marginBottom": "0.4rem"},
            ),
            html.Div(
                "Filenames are derived as <base><iter><ext>, e.g. allmotl_3.em -- "
                "the extension and any Relion suffix come from the motl type.",
                style=_HINT,
            ),
            dbc.Button(
                "Load from motls",
                id="sta-direct-load-btn",
                color="primary",
                size="sm",
                style={"width": "100%", "marginTop": "0.25rem"},
            ),
            html.Div(
                id="sta-direct-status",
                style={
                    "fontSize": "0.8rem", "color": "var(--color9)",
                    "marginTop": "0.4rem", "wordBreak": "break-word",
                },
            ),
        ],
        id="sta-eval-direct-block",
    )


def _evaluation_panel() -> html.Div:
    return html.Div(
        [
            html.Label("Loader", style=_LBL),
            dcc.RadioItems(
                id="sta-eval-mode",
                options=[
                    {"label": " From parameter file", "value": "params"},
                    {"label": " From motls directly", "value": "direct"},
                ],
                value="params",
                inputStyle={"marginRight": "0.3rem"},
                labelStyle={"display": "block", "marginBottom": "0.2rem",
                            "fontSize": "0.85rem"},
                style={"marginBottom": "0.6rem"},
            ),
            html.Div(_param_file_form(), id="sta-eval-params-wrapper",
                     style={"display": "block"}),
            html.Div(_direct_form(), id="sta-eval-direct-wrapper",
                     style={"display": "none"}),
            html.Hr(style={"margin": "0.5rem 0"}),
            html.Div(
                id="sta-loaded-readout",
                style={**_HINT, "wordBreak": "break-word"},
            ),
            html.Div("Run evaluation", style={**_LBL, "marginTop": "0.5rem"}),
            dbc.Button(
                "Alignment evaluation",
                id="sta-run-alignment-btn",
                color="info",
                size="sm",
                style={"width": "100%", "marginBottom": "0.3rem"},
            ),
            dbc.Button(
                "Classification evaluation",
                id="sta-run-classification-btn",
                color="info",
                size="sm",
                style={"width": "100%"},
            ),
            html.Div(
                id="sta-action-status",
                style={**_HINT, "marginTop": "0.4rem", "wordBreak": "break-word"},
            ),
        ]
    )


# ── STA setup accordion panel ────────────────────────────────────────────────


def _slim_angles_builder(prefix: str) -> html.Div:
    """Visualisation-only version of the angles builder.

    Re-uses :func:`cryocat.utils.geom.generate_angles` via
    :func:`cryocat.app.formgen.build_form` so the form rows + the two
    preview graphs are identical to the canonical builder.  The
    ``output_path`` input + the "Create angle list" button are kept in
    the DOM (hidden) so the public
    :func:`cryocat.app.components.anglesbuilder.register_angles_builder_callbacks`
    can wire its three callbacks without missing-id warnings, but the
    user only sees the form + the previews + the page's own
    "Use these angles" button.
    """
    form_rows = formgen.build_form(
        generate_angles,
        id_type="angles-param",
        id_extra={"builder": prefix},
    )
    return html.Div(
        [
            html.Div(form_rows, style={"marginBottom": "0.5rem"}),
            dbc.Button(
                "Visualize",
                id=f"{prefix}-visualize-btn",
                color="primary",
                size="sm",
                style={"width": "100%", "marginBottom": "0.5rem"},
            ),
            # Hidden plumbing so the builder's create-file callback can
            # safely wire (it never fires because the button is invisible).
            html.Div(
                [
                    dbc.Input(id=f"{prefix}-output-path", type="text"),
                    dbc.Button(id=f"{prefix}-create"),
                    html.Span(id=f"{prefix}-status"),
                ],
                style={"display": "none"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(id={"type": "styled-graph", "owner": prefix, "name": "preview"},
                                  style={"height": "260px"}),
                        width=6,
                    ),
                    dbc.Col(
                        dcc.Graph(id={"type": "styled-graph", "owner": prefix, "name": "inplane"},
                                  style={"height": "260px"}),
                        width=6,
                    ),
                ],
                className="g-1",
            ),
            dcc.Store(id=f"{prefix}-params"),
            dcc.Store(id=f"{prefix}-angles"),
            dcc.Store(id=f"{prefix}-created-path"),
        ],
    )


def _sta_setup_panel() -> html.Div:
    return html.Div(
        [
            html.Div("Iterations", style=_SECTION_HEADER),
            _field_num("Number of iterations", "sta-setup-iter",
                       value=1, min=1, step=1),
            _field_num("Starting iteration", "sta-setup-start-index",
                       value=1, min=0, step=1),
            _field_check("Create reference (averaging pre-step)",
                         "sta-setup-create-ref", value=False),
            _field_check("Multi-reference mode",
                         "sta-setup-multiref", value=False),
            html.Hr(style={"margin": "0.4rem 0"}),

            html.Div("Angles (novaSTA convention)", style=_SECTION_HEADER),
            _field_text(
                "Cone angle", "sta-setup-cone-angle",
                placeholder="single value or space-separated per-iter, e.g. 30 20 10",
            ),
            _field_text("Cone sampling", "sta-setup-cone-sampling",
                        placeholder="e.g. 5"),
            _field_text("Inplane angle", "sta-setup-inplane-angle",
                        placeholder="e.g. 360"),
            _field_text("Inplane sampling", "sta-setup-inplane-sampling",
                        placeholder="e.g. 5"),
            dbc.Button(
                "Build angles…",
                id="sta-setup-angles-open-btn",
                color="secondary",
                size="sm",
                style={"width": "100%", "marginTop": "0.2rem",
                       "marginBottom": "0.4rem"},
            ),
            html.Div(
                id="sta-setup-use-angles-status",
                style={**_HINT, "marginBottom": "0.3rem"},
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Build angles (preview)")),
                    dbc.ModalBody(
                        [
                            html.Div(
                                "Fill the form to see the cone + inplane "
                                "sampling preview. No file is created -- "
                                "click \"Use these angles\" to copy the four "
                                "values into the STA setup form.",
                                style={**_HINT, "marginBottom": "0.5rem"},
                            ),
                            _slim_angles_builder(_ANGLES_PREFIX),
                        ],
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "Use these angles",
                                id="sta-setup-use-angles-btn",
                                color="primary",
                                className="me-2",
                            ),
                            dbc.Button(
                                "Close",
                                id="sta-setup-angles-close-btn",
                                color="secondary",
                            ),
                        ]
                    ),
                ],
                id="sta-setup-angles-modal",
                size="xl",
                is_open=False,
                centered=True,
                scrollable=True,
            ),
            html.Hr(style={"margin": "0.4rem 0"}),

            html.Div("Filters", style=_SECTION_HEADER),
            _field_text("High pass (required)", "sta-setup-high-pass",
                        placeholder="e.g. 25 20 15"),
            _field_text("Low pass", "sta-setup-low-pass",
                        placeholder="e.g. 30"),
            _field_text("Score threshold", "sta-setup-threshold",
                        placeholder="e.g. 0.0"),
            html.Hr(style={"margin": "0.4rem 0"}),

            html.Div("Paths", style=_SECTION_HEADER),
            _field_text("Motl name / path", "sta-setup-motl"),
            _field_text("Reference", "sta-setup-ref"),
            _field_text("Mask", "sta-setup-mask"),
            _field_text("CC mask", "sta-setup-cc-mask"),
            _field_text("Wedge list", "sta-setup-wedge-list"),
            _field_text("Subtomograms", "sta-setup-subtomograms"),
            _field_text("Tomograms", "sta-setup-tomograms"),
            html.Hr(style={"margin": "0.4rem 0"}),

            html.Div("Other", style=_SECTION_HEADER),
            _field_num("Symmetry (Cn order)", "sta-setup-symmetry",
                       value=1, min=1, step=1),
            _field_num("Class", "sta-setup-class", value=1, min=0, step=1),
            html.Hr(style={"margin": "0.4rem 0"}),

            dbc.Button(
                "Create parameter dataframe",
                id="sta-setup-create-btn",
                color="primary",
                size="sm",
                style={"width": "100%"},
            ),
            html.Div(
                id="sta-setup-create-status",
                style={**_HINT, "marginTop": "0.3rem"},
            ),
        ]
    )


def _placeholder_panel(text: str = _PLACEHOLDER) -> html.Div:
    return html.Div(text, style=_HINT)


def _sidebar() -> list:
    return [
        sidebar_accordion(
            [
                dbc.AccordionItem(
                    _evaluation_panel(),
                    title="Evaluation",
                    item_id="sta-acc-eval",
                ),
                dbc.AccordionItem(
                    _sta_setup_panel(),
                    title="STA setup",
                    item_id="sta-acc-sta-setup",
                ),
                dbc.AccordionItem(
                    _placeholder_panel(),
                    title="Classification setup",
                    item_id="sta-acc-class-setup",
                ),
            ],
            active_item=["sta-acc-eval"],
        ),
    ]



# ── Tab content builders ─────────────────────────────────────────────────────


def _tab_params() -> html.Div:
    return html.Div(
        [
            html.Div(
                "Parameters loaded from a STOPGAP .star or novaSTA .txt file. "
                "Empty until a parameter file is loaded.",
                style=_HINT,
            ),
            dag.AgGrid(
                id="sta-params-grid",
                rowData=[],
                columnDefs=[],
                defaultColDef={
                    "resizable": True, "sortable": True, "filter": False,
                    "flex": 1, "minWidth": 80,
                },
                style={"height": "75vh"},
                className="ag-theme-balham",
                dashGridOptions={"suppressMovableColumns": True},
            ),
        ],
        style={"padding": "0.5rem"},
    )


def _tab_alignment() -> html.Div:
    return html.Div(
        [
            dcc.Graph(
                id="sta-alignment-graph",
                figure=error_figure(
                    "Load a config and click 'Alignment evaluation' to compute "
                    "compute_alignment_statistics."
                ),
                style={"width": "100%"},
            ),
        ],
        style={"padding": "0.5rem"},
    )


def _tab_classification() -> html.Div:
    return html.Div(
        [
            dcc.Graph(
                id="sta-classification-graph",
                figure=error_figure(
                    "Load a config and click 'Classification evaluation' to "
                    "compute evaluate_classification."
                ),
                style={"width": "100%", "height": "420px"},
            ),
        ],
        style={"padding": "0.5rem"},
    )


def _tab_setup_output() -> html.Div:
    return html.Div(
        [
            html.Div(
                "Edit cells inline (single-click). Provide an output path and "
                "save as novaSTA (.txt) or STOPGAP (.star). Angle conversion is "
                "applied automatically when saving as STOPGAP.",
                style=_HINT,
            ),
            dag.AgGrid(
                id="sta-setup-grid",
                rowData=[],
                columnDefs=[],
                defaultColDef={
                    "resizable": True, "sortable": False, "filter": False,
                    "editable": True, "flex": 1, "minWidth": 100,
                },
                style={"height": "55vh"},
                className="ag-theme-balham",
                dashGridOptions={
                    "singleClickEdit": True,
                    "suppressMovableColumns": True,
                    "stopEditingWhenCellsLoseFocus": True,
                },
            ),
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Output path", style=_LBL),
                                    dbc.Input(
                                        id="sta-setup-save-path",
                                        type="text",
                                        placeholder="path/to/params.txt  or  .star",
                                        size="sm",
                                    ),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Save as novaSTA",
                                    id="sta-setup-save-novasta-btn",
                                    color="info", size="sm",
                                    style={"width": "100%",
                                           "marginTop": "1.4rem"},
                                ),
                                width=3,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Save as STOPGAP",
                                    id="sta-setup-save-stopgap-btn",
                                    color="info", size="sm",
                                    style={"width": "100%",
                                           "marginTop": "1.4rem"},
                                ),
                                width=3,
                            ),
                        ],
                        className="g-1",
                        style={"marginTop": "0.5rem"},
                    ),
                    html.Div(
                        id="sta-setup-save-status",
                        style={**_HINT, "marginTop": "0.4rem",
                               "wordBreak": "break-word"},
                    ),
                ],
            ),
        ],
        style={"padding": "0.5rem"},
    )


def _main() -> list:
    return [
        dcc.Tabs(
            id="sta-main-tabs",
            value="tab-params",
            children=[
                dcc.Tab(label="Parameter file",
                        value="tab-params",
                        children=_tab_params()),
                dcc.Tab(label="Alignment evaluation",
                        value="tab-align",
                        children=_tab_alignment()),
                dcc.Tab(label="Classification evaluation",
                        value="tab-class",
                        children=_tab_classification()),
                dcc.Tab(label="STA setup output",
                        value="tab-setup",
                        children=_tab_setup_output()),
            ],
        ),
    ]


layout = html.Div(
    [
        # Resolved run config from the loader: {motl_base, motl_type, ...}.
        dcc.Store(id="sta-loader-config"),
        # Serialised params df (only populated by the parameter-file loader).
        dcc.Store(id="sta-params-store"),
        # Created setup df: {"records": ..., "columns": ...}.
        dcc.Store(id="sta-setup-df-store"),
        page_shell(_sidebar(), _main()),
    ],
    style={"margin": "0", "padding": "0"},
)


# ── Stat-grid helper ────────────────────────────────────────────────────────


def _build_stats_grid(records: list[dict], x_key: str, cols: int = 3) -> go.Figure:
    """Render every numeric column in ``records`` (except ``x_key``) as a
    line plot, arranged in a ``cols``-wide grid.  Tighter vertical spacing
    than the default so the grid stays compact when there are many stats.
    """
    if not records:
        return error_figure("No iterations to plot.")

    xs = [row.get(x_key) for row in records]
    stat_cols: list[str] = []
    for key in records[0]:
        if key == x_key:
            continue
        if any(row.get(key) is not None for row in records):
            stat_cols.append(key)

    if not stat_cols:
        return error_figure("No numeric statistics available in the result.")

    rows = math.ceil(len(stat_cols) / cols)
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=stat_cols,
        shared_xaxes=False,
        vertical_spacing=0.08,    # tighter than the default 0.18
        horizontal_spacing=0.08,
    )
    for idx, col in enumerate(stat_cols):
        r = idx // cols + 1
        c = idx % cols + 1
        ys = [row.get(col) for row in records]
        fig.add_trace(
            go.Scatter(x=xs, y=ys, mode="lines+markers", name=col,
                       showlegend=False),
            row=r, col=c,
        )
        fig.update_xaxes(title_text=x_key, row=r, col=c)

    fig.update_layout(
        height=max(260, rows * 220),
        margin=dict(t=40, b=30, l=40, r=20),
    )
    return fig


# ── Build a parameter dict from the STA setup form ──────────────────────────


def _expand_for_n(text: str | None, n_align: int) -> str | None:
    """Broadcast a single-token text value to ``n_align`` repeats when
    ``n_align > 1``.  ``None`` / empty input is left unchanged.
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    tokens = s.split()
    if len(tokens) == 1 and n_align > 1:
        return " ".join(tokens * n_align)
    return s


def _setup_form_to_params_dict(
    *,
    iter_n: int, start_index: int,
    create_ref: bool, multiref: bool,
    cone_angle: str, cone_sampling: str,
    inplane_angle: str, inplane_sampling: str,
    high_pass: str, low_pass: str, threshold: str,
    motl: str, ref: str, mask: str, cc_mask: str,
    wedge_list: str, subtomograms: str, tomograms: str,
    symmetry, class_id,
) -> dict:
    """Pack the form values into the snake_case dict that
    :meth:`sta.StaParameters.from_dict` accepts.  All per-iter values are
    broadcast to ``iter_n`` repeats; empty fields are dropped.
    """
    iter_n = max(int(iter_n or 1), 1)
    # Mandatory keys -- always emitted.
    out: dict = {
        "cone_angle": _expand_for_n(cone_angle, iter_n),
        "cone_sampling": _expand_for_n(cone_sampling, iter_n),
        "inplane_angle": _expand_for_n(inplane_angle, iter_n),
        "inplane_sampling": _expand_for_n(inplane_sampling, iter_n),
        "high_pass": _expand_for_n(high_pass, iter_n),
        "start_index": int(start_index or 1),
        "create_ref": 1 if create_ref else 0,
        "multiref": 1 if multiref else 0,
    }
    # Optional, only if filled.
    optional = {
        "low_pass": _expand_for_n(low_pass, iter_n),
        "threshold": _expand_for_n(threshold, iter_n),
        "motl": (motl or "").strip() or None,
        "ref": (ref or "").strip() or None,
        "mask": (mask or "").strip() or None,
        "cc_mask": (cc_mask or "").strip() or None,
        "wedge_list": (wedge_list or "").strip() or None,
        "subtomograms": (subtomograms or "").strip() or None,
        "tomograms": (tomograms or "").strip() or None,
        "symmetry": int(symmetry) if symmetry not in (None, "") else None,
        "class": int(class_id) if class_id not in (None, "") else None,
    }
    for k, v in optional.items():
        if v is not None:
            out[k] = v
    return out


# ── Callbacks ────────────────────────────────────────────────────────────────


def register_callbacks(app):

    # Slim angles builder: params collect comes for free; we replace the
    # auto-preview with a manual "Visualize" button so the user controls
    # when the preview re-renders.  The hidden create-file button never
    # fires because it's invisible.
    register_angles_builder_callbacks(app, _ANGLES_PREFIX, skip_preview=True)

    # ── Toggle which loader form is visible ──────────────────────────────────
    @app.callback(
        Output("sta-eval-params-wrapper", "style"),
        Output("sta-eval-direct-wrapper", "style"),
        Input("sta-eval-mode", "value"),
    )
    def _toggle_loader(mode):
        if mode == "direct":
            return {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}

    # ── Load from parameter file → populate loader-config + params-store ─────
    @app.callback(
        Output("sta-loader-config", "data", allow_duplicate=True),
        Output("sta-params-store", "data", allow_duplicate=True),
        Output("sta-param-status", "children"),
        Input("sta-param-load-btn", "n_clicks"),
        State("sta-param-path", "value"),
        State("sta-param-type", "value"),
        State("sta-param-sep", "value"),
        State("sta-param-working-dir", "value"),
        prevent_initial_call=True,
    )
    def load_from_params(n_clicks, path, sta_type, separator, working_dir):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not path:
            return no_update, no_update, "Provide a parameter file path."

        separator = separator or "_"
        sta_type_arg = None if sta_type == "auto" else sta_type
        working_dir = (working_dir or "").strip() or None

        try:
            params = sta_mod.StaParameters.load(path.strip(), sta_type=sta_type_arg)
        except Exception as exc:
            return no_update, no_update, f"Failed to load parameter file: {exc}"

        motl_base = params.get_motl_base_name(separator, working_dir=working_dir)
        if motl_base is None:
            return (
                no_update, no_update,
                "Parameter file has no motl path — cannot locate motl files.",
            )

        start_it = params.start_iteration
        end_it = params.end_iteration
        if start_it is None or end_it is None:
            return no_update, no_update, "Parameter file contains no alignment iterations."

        config = {
            "motl_base": motl_base,
            "motl_type": params.motl_type,
            "start_it": int(start_it),
            "end_it": int(end_it),
            "source": "params",
            "params_path": path.strip(),
            "sta_type": sta_type_arg,
            "motl_separator": separator,
            "working_dir": working_dir,
        }
        params_data = {
            "records": params.df.to_dict("records"),
            "columns": list(params.df.columns),
        }
        return (
            config, params_data,
            f"Loaded {params.motl_type} config from parameter file "
            f"(iterations {start_it}–{end_it}). Click an evaluation button to run.",
        )

    # ── Load from motls directly → populate loader-config ────────────────────
    @app.callback(
        Output("sta-loader-config", "data", allow_duplicate=True),
        Output("sta-params-store", "data", allow_duplicate=True),
        Output("sta-direct-status", "children"),
        Input("sta-direct-load-btn", "n_clicks"),
        State("sta-direct-base", "value"),
        State("sta-direct-type", "value"),
        State("sta-direct-start", "value"),
        State("sta-direct-count", "value"),
        prevent_initial_call=True,
    )
    def load_from_motls(n_clicks, motl_base, motl_type, start_it, count):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not motl_base or not motl_type or count is None or start_it is None:
            return no_update, no_update, (
                "Provide motl base, motl type, starting iteration and count."
            )

        start_it = int(start_it)
        count = int(count)
        if count < 1:
            return no_update, no_update, "Number of iterations must be at least 1."

        end_it = start_it + count - 1
        config = {
            "motl_base": motl_base.strip(),
            "motl_type": motl_type,
            "start_it": start_it,
            "end_it": end_it,
            "source": "direct",
        }
        return (
            config, None,
            f"Loaded {motl_type} config (iterations {start_it}–{end_it}). "
            "Click an evaluation button to run.",
        )

    # ── Loaded-config readout ────────────────────────────────────────────────
    @app.callback(
        Output("sta-loaded-readout", "children"),
        Input("sta-loader-config", "data"),
    )
    def _show_config(config):
        if not config:
            return "No config loaded yet."
        return (
            f"Loaded: motl base \"{config['motl_base']}\"  |  "
            f"type {config['motl_type']}  |  "
            f"iterations {config['start_it']}–{config['end_it']}  |  "
            f"source: {config['source']}"
        )

    # ── Populate params table (tab 1) ────────────────────────────────────────
    @app.callback(
        Output("sta-params-grid", "rowData"),
        Output("sta-params-grid", "columnDefs"),
        Input("sta-params-store", "data"),
    )
    def update_params_table(data):
        if not data or not data.get("columns"):
            return [], []
        col_defs = [
            {"field": c, "headerName": c, "flex": 1, "minWidth": 80}
            for c in data["columns"]
        ]
        return data["records"], col_defs

    # ── Alignment evaluation: compute + plot (tab 2) ─────────────────────────
    @app.callback(
        Output("sta-alignment-graph", "figure"),
        Output("sta-action-status", "children", allow_duplicate=True),
        Output("sta-main-tabs", "value", allow_duplicate=True),
        Input("sta-run-alignment-btn", "n_clicks"),
        State("sta-loader-config", "data"),
        prevent_initial_call=True,
    )
    def run_alignment_evaluation(n_clicks, config):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not config:
            return (no_update,
                    "Load a config first (parameter file or direct loader).",
                    no_update)

        try:
            stats_df = sta_mod.compute_alignment_statistics(
                config["motl_base"],
                config["start_it"], config["end_it"],
                motl_type=config["motl_type"],
            )
        except Exception as exc:
            return (error_figure(f"Alignment evaluation failed: {exc}"),
                    f"Alignment evaluation failed: {exc}",
                    no_update)

        if stats_df is None or stats_df.empty:
            return (error_figure("Alignment evaluation returned no rows."),
                    "Alignment evaluation returned no rows.",
                    no_update)

        # Each row is the transition motl[i] → motl[i+1]; label by the
        # second iteration in the pair.
        stats_df = stats_df.copy()
        stats_df["iteration"] = list(
            range(config["start_it"] + 1, config["start_it"] + 1 + len(stats_df))
        )
        non_iter_cols = [c for c in stats_df.columns if c != "iteration"]
        keep = stats_df[non_iter_cols].notna().any(axis=1)
        stats_df = stats_df.loc[keep].reset_index(drop=True)

        records = stats_df.to_dict("records")
        fig = _build_stats_grid(records, x_key="iteration", cols=3)
        return (
            fig,
            f"Alignment evaluation done: {len(records)} transition(s) "
            f"with {len(non_iter_cols)} statistic column(s).",
            "tab-align",
        )

    # ── Classification evaluation: compute + plot (tab 3) ────────────────────
    @app.callback(
        Output("sta-classification-graph", "figure"),
        Output("sta-action-status", "children", allow_duplicate=True),
        Output("sta-main-tabs", "value", allow_duplicate=True),
        Input("sta-run-classification-btn", "n_clicks"),
        State("sta-loader-config", "data"),
        prevent_initial_call=True,
    )
    def run_classification_evaluation(n_clicks, config):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not config:
            return (no_update,
                    "Load a config first (parameter file or direct loader).",
                    no_update)

        try:
            occupancy, changes = sta_mod.evaluate_classification(
                config["motl_base"],
                config["start_it"], config["end_it"],
                motl_type=config["motl_type"],
                plot_results=False,
            )
        except Exception as exc:
            return (error_figure(f"Classification evaluation failed: {exc}"),
                    f"Classification evaluation failed: {exc}",
                    no_update)

        if not occupancy:
            return (error_figure("Classification evaluation returned no classes."),
                    "Classification evaluation returned no classes.",
                    no_update)

        try:
            fig = visplot.plot_classification_convergence(
                occupancy, changes, graph_title="Classification progress",
            )
        except Exception as exc:
            return (error_figure(f"Classification plot failed: {exc}"),
                    f"Classification plot failed: {exc}",
                    no_update)

        n_classes = len(occupancy)
        n_iters = max((len(v) for v in occupancy.values()), default=0)
        return (
            fig,
            f"Classification evaluation done: {n_classes} class(es) "
            f"over {n_iters} iteration(s).",
            "tab-class",
        )

    # ── Angles modal: open / close / use these angles ───────────────────────
    @app.callback(
        Output("sta-setup-angles-modal", "is_open"),
        Input("sta-setup-angles-open-btn", "n_clicks"),
        Input("sta-setup-angles-close-btn", "n_clicks"),
        Input("sta-setup-use-angles-btn", "n_clicks"),
        State("sta-setup-angles-modal", "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_angles_modal(n_open, n_close, n_use, is_open):
        triggered = ctx.triggered_id
        if triggered == "sta-setup-angles-open-btn":
            return True
        # Both Close and Use close the modal; the form-copy callback below
        # handles the value transfer when Use is clicked.
        if triggered in ("sta-setup-angles-close-btn",
                         "sta-setup-use-angles-btn"):
            return False
        return is_open

    # ── Visualize: manual preview triggered by the modal's button ───────────
    #
    # Reproduces the auto-preview in
    # :func:`cryocat.app.components.anglesbuilder.register_angles_builder_callbacks`
    # but with two differences:
    #
    # 1. Trigger is the explicit ``Visualize`` button (not every form edit).
    # 2. The inplane polar plot is centred on 0: ``generate_angles`` emits
    #    phi values in ``[0, inplane_angle)``, which the user prefers to see
    #    as ``[-inplane_angle/2, inplane_angle/2)`` (so inplane_angle=80 lays
    #    markers between roughly -40 and +40 instead of 0..70).
    @app.callback(
        Output(f"{_ANGLES_PREFIX}-angles", "data"),
        Output({"type": "styled-graph", "owner": _ANGLES_PREFIX, "name": "preview"}, "figure"),
        Output({"type": "styled-graph", "owner": _ANGLES_PREFIX, "name": "inplane"}, "figure"),
        Input(f"{_ANGLES_PREFIX}-visualize-btn", "n_clicks"),
        State(f"{_ANGLES_PREFIX}-params", "data"),
        State(ids.GRAPH_SETTINGS_STORE, "data"),
        prevent_initial_call=True,
    )
    def _visualize_angles(n_clicks, params, gs):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not params:
            err = error_figure("Fill the form first.")
            return no_update, err, err
        if params.get("cone_angle") is None or params.get("cone_sampling") is None:
            err = error_figure(
                "Set at least cone_angle and cone_sampling.")
            return no_update, err, err

        try:
            kwargs = {k: v for k, v in params.items() if v is not None}
            angles = generate_angles(**kwargs)
        except Exception as exc:
            err = error_figure(f"generate_angles failed: {exc}")
            return no_update, err, err

        # Cone-sphere panel: same path as the canonical builder.
        try:
            sphere_raw = visplot.plot_rotation_normals(angles)
            sphere_fig = styled_figure(
                sphere_raw, gs or {},
                uirevision=f"{_ANGLES_PREFIX}-preview",
                margin={"l": 0, "r": 0, "t": 30, "b": 0},
            )
        except Exception as exc:
            sphere_fig = error_figure(f"Sphere plot error: {exc}")

        # Inplane polar panel: centre on 0 so e.g. inplane_angle=80 shows
        # markers symmetric around the X axis instead of 0..80.
        try:
            inplane_max = float(params.get("inplane_angle", 360.0) or 360.0)
            phi = np.unique(np.round(angles[:, 0], 8))
            phi_centered = phi - inplane_max / 2.0
            inplane_fig = go.Figure(
                go.Scatterpolar(
                    r=[1.0] * len(phi_centered),
                    theta=phi_centered,
                    mode="markers",
                    marker=dict(size=6, opacity=0.8),
                )
            )
            inplane_fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=False, range=[0, 1.3]),
                    angularaxis=dict(
                        direction="counterclockwise",
                        tickmode="array",
                        tickvals=[-180, -135, -90, -45, 0, 45, 90, 135],
                    ),
                ),
                showlegend=False,
                title=dict(
                    text=f"Inplane sampling (φ) — {len(phi_centered)} "
                         f"angle(s), centred on 0",
                    font=dict(size=12),
                ),
                margin=dict(l=40, r=40, t=40, b=30),
            )
            inplane_fig = styled_figure(inplane_fig, gs or {}, uirevision=f"{_ANGLES_PREFIX}-inplane")
        except Exception as exc:
            inplane_fig = error_figure(f"Inplane plot error: {exc}")

        return angles.tolist(), sphere_fig, inplane_fig

    # ── 'Use these angles' → copy from angles-builder params into form ──────
    @app.callback(
        Output("sta-setup-cone-angle", "value"),
        Output("sta-setup-cone-sampling", "value"),
        Output("sta-setup-inplane-angle", "value"),
        Output("sta-setup-inplane-sampling", "value"),
        Output("sta-setup-use-angles-status", "children"),
        Input("sta-setup-use-angles-btn", "n_clicks"),
        State(f"{_ANGLES_PREFIX}-params", "data"),
        prevent_initial_call=True,
    )
    def use_angles(n_clicks, params):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not params:
            return (no_update, no_update, no_update, no_update,
                    "Fill the angle visualization form first.")
        # The angles builder stores the form values keyed by the generate_angles
        # signature (cone_angle, cone_sampling, inplane_angle, inplane_sampling).
        def _to_str(v):
            return "" if v in (None, "") else str(v)
        return (
            _to_str(params.get("cone_angle")),
            _to_str(params.get("cone_sampling")),
            _to_str(params.get("inplane_angle")),
            _to_str(params.get("inplane_sampling")),
            "Angle values copied into the STA setup form.",
        )

    # ── Create the setup dataframe (tab 4 grid) ──────────────────────────────
    @app.callback(
        Output("sta-setup-df-store", "data"),
        Output("sta-setup-create-status", "children"),
        Output("sta-main-tabs", "value", allow_duplicate=True),
        Input("sta-setup-create-btn", "n_clicks"),
        State("sta-setup-iter", "value"),
        State("sta-setup-start-index", "value"),
        State("sta-setup-create-ref", "value"),
        State("sta-setup-multiref", "value"),
        State("sta-setup-cone-angle", "value"),
        State("sta-setup-cone-sampling", "value"),
        State("sta-setup-inplane-angle", "value"),
        State("sta-setup-inplane-sampling", "value"),
        State("sta-setup-high-pass", "value"),
        State("sta-setup-low-pass", "value"),
        State("sta-setup-threshold", "value"),
        State("sta-setup-motl", "value"),
        State("sta-setup-ref", "value"),
        State("sta-setup-mask", "value"),
        State("sta-setup-cc-mask", "value"),
        State("sta-setup-wedge-list", "value"),
        State("sta-setup-subtomograms", "value"),
        State("sta-setup-tomograms", "value"),
        State("sta-setup-symmetry", "value"),
        State("sta-setup-class", "value"),
        prevent_initial_call=True,
    )
    def create_setup_df(
        n_clicks,
        iter_n, start_index, create_ref, multiref,
        cone_angle, cone_sampling, inplane_angle, inplane_sampling,
        high_pass, low_pass, threshold,
        motl, ref, mask, cc_mask, wedge_list, subtomograms, tomograms,
        symmetry, class_id,
    ):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        # Validate mandatory fields.
        required = {
            "cone angle": cone_angle, "cone sampling": cone_sampling,
            "inplane angle": inplane_angle, "inplane sampling": inplane_sampling,
            "high pass": high_pass,
        }
        missing = [name for name, v in required.items() if not (v and str(v).strip())]
        if missing:
            return (no_update,
                    "Missing required field(s): " + ", ".join(missing),
                    no_update)

        try:
            params_dict = _setup_form_to_params_dict(
                iter_n=iter_n, start_index=start_index,
                create_ref=bool(create_ref), multiref=bool(multiref),
                cone_angle=cone_angle, cone_sampling=cone_sampling,
                inplane_angle=inplane_angle, inplane_sampling=inplane_sampling,
                high_pass=high_pass, low_pass=low_pass, threshold=threshold,
                motl=motl, ref=ref, mask=mask, cc_mask=cc_mask,
                wedge_list=wedge_list, subtomograms=subtomograms,
                tomograms=tomograms,
                symmetry=symmetry, class_id=class_id,
            )
            params = sta_mod.StaParameters.from_dict(params_dict, sta_type="novasta")
        except Exception as exc:
            return no_update, f"Create failed: {exc}", no_update

        if params.df.empty:
            return (no_update,
                    "Create produced an empty dataframe — check the form values.",
                    no_update)

        data = {
            "records": params.df.to_dict("records"),
            "columns": list(params.df.columns),
        }
        return (
            data,
            f"Created parameter dataframe ({len(params.df)} iteration(s), "
            f"{len(params.df.columns)} column(s)). Edit cells in the "
            f"\"STA setup output\" tab, then save.",
            "tab-setup",
        )

    # ── Populate setup grid from store ───────────────────────────────────────
    @app.callback(
        Output("sta-setup-grid", "rowData"),
        Output("sta-setup-grid", "columnDefs"),
        Input("sta-setup-df-store", "data"),
    )
    def populate_setup_grid(data):
        if not data or not data.get("columns"):
            return [], []
        col_defs = [
            {"field": c, "headerName": c, "editable": True,
             "flex": 1, "minWidth": 100}
            for c in data["columns"]
        ]
        return data["records"], col_defs

    # ── Save setup dataframe to disk ─────────────────────────────────────────
    @app.callback(
        Output("sta-setup-save-status", "children"),
        Input("sta-setup-save-novasta-btn", "n_clicks"),
        Input("sta-setup-save-stopgap-btn", "n_clicks"),
        State("sta-setup-grid", "rowData"),
        State("sta-setup-df-store", "data"),
        State("sta-setup-save-path", "value"),
        prevent_initial_call=True,
    )
    def save_setup(n_nova, n_sg, rows, store, path):
        triggered = ctx.triggered_id
        if triggered is None:
            raise dash.exceptions.PreventUpdate
        if not path or not str(path).strip():
            return "Provide an output path."
        if not rows or not store or not store.get("columns"):
            return "Create the parameter dataframe first."

        target = "stopgap" if triggered == "sta-setup-save-stopgap-btn" else "novasta"
        columns = store["columns"]
        try:
            # Preserve the canonical column order from the store; AgGrid may
            # re-order rowData fields after editing.
            df = pd.DataFrame(rows)[columns]
            params = sta_mod.NovaStaParams(df)
            if target == "stopgap":
                params = params.to_stopgap()
            params.write_out(str(path).strip())
        except Exception as exc:
            return f"Save as {target} failed: {exc}"
        return f"Saved as {target} → {str(path).strip()}"
