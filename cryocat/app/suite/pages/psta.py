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
from cryocat.app.apputils import run_operation
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


def _field_dropdown(label: str, id_: str, options: list, value=None) -> html.Div:
    return html.Div(
        [
            html.Label(label, style=_FIELD_LABEL),
            dcc.Dropdown(
                id=id_,
                options=options,
                value=value,
                clearable=False,
                searchable=False,
                style={**_FIELD_INPUT, "fontSize": "0.82rem"},
            ),
        ],
        style=_FIELD_ROW,
    )


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
            dcc.Store(id=f"{prefix}-value"),
        ],
    )


_REF_FAMILY_OPTIONS = [
    {"label": "Single reference", "value": "singleref"},
    {"label": "Multi-reference",  "value": "multiref"},
    {"label": "Multi-class",      "value": "multiclass"},
]


_STA_TYPE_OPTIONS = [
    {"label": "novaSTA (.txt)", "value": "novasta"},
    {"label": "STOPGAP (.star)", "value": "stopgap"},
]

_SHOW = {"display": "block"}
_HIDE = {"display": "none"}


def _sta_setup_panel() -> html.Div:
    return html.Div(
        [
            # ── Format ───────────────────────────────────────────────────────
            html.Div("Format", style=_SECTION_HEADER),
            _field_dropdown("STA format", "sta-setup-sta-type",
                            options=_STA_TYPE_OPTIONS, value="novasta"),
            html.Hr(style={"margin": "0.4rem 0"}),

            # ── Iterations ────────────────────────────────────────────────────
            html.Div("Iterations", style=_SECTION_HEADER),
            _field_num("Number of iterations", "sta-setup-iter",
                       value=1, min=1, step=1),
            _field_num("Starting iteration", "sta-setup-start-index",
                       value=1, min=0, step=1),
            html.Hr(style={"margin": "0.4rem 0"}),

            # ── Common — both formats ─────────────────────────────────────────
            html.Div("Common — both formats", style=_SECTION_HEADER),
            _field_check("Create reference (averaging pre-step)",
                         "sta-setup-create-ref", value=False),
            _field_check("Split into even / odd halfsets",
                         "sta-setup-split-even-odd", value=True),
            html.Hr(style={"margin": "0.2rem 0"}),
            _field_text("Motl name / path", "sta-setup-motl"),
            _field_text("Reference", "sta-setup-ref"),
            _field_text("Mask", "sta-setup-mask"),
            _field_text("CC mask", "sta-setup-cc-mask"),
            _field_text("Wedge list", "sta-setup-wedge-list"),
            _field_text("Subtomogram path", "sta-setup-subtomo-path",
                        placeholder="path to subtomograms (name / pattern)"),
            html.Hr(style={"margin": "0.2rem 0"}),
            html.Div(
                [
                    html.Span("Angles", style={**_HINT, "fontWeight": 600}),
                    dbc.Button(
                        "Build angles…",
                        id="sta-setup-angles-open-btn",
                        color="secondary",
                        size="sm",
                        style={"marginLeft": "auto"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center",
                       "marginBottom": "0.25rem"},
            ),
            _field_text(
                "Cone angle", "sta-setup-cone-angle",
                placeholder="single value or space-separated, e.g. 30 20 10",
            ),
            _field_text("Cone sampling", "sta-setup-cone-sampling",
                        placeholder="e.g. 5"),
            _field_text("Inplane angle", "sta-setup-inplane-angle",
                        placeholder="e.g. 360"),
            _field_text("Inplane sampling", "sta-setup-inplane-sampling",
                        placeholder="e.g. 5"),
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
            html.Hr(style={"margin": "0.2rem 0"}),
            _field_text("High pass (px)", "sta-setup-high-pass",
                        placeholder="e.g. 25 20 15"),
            _field_num("High pass sigma", "sta-setup-high-pass-sigma",
                       value=2.0, step=0.5),
            _field_text("Low pass (px)", "sta-setup-low-pass",
                        placeholder="e.g. 30"),
            _field_num("Low pass sigma", "sta-setup-low-pass-sigma",
                       value=3.0, step=0.5),
            _field_text("Score threshold", "sta-setup-threshold",
                        placeholder="e.g. 0.0"),
            _field_num("Symmetry (Cn order)", "sta-setup-symmetry",
                       value=1, min=1, step=1),
            html.Hr(style={"margin": "0.4rem 0"}),

            # ── novaSTA only ──────────────────────────────────────────────────
            html.Div(
                id="sta-setup-novasta-section",
                children=[
                    html.Div("novaSTA only", style=_SECTION_HEADER),
                    _field_num("Class", "sta-setup-class", value=1, min=0, step=1),
                    _field_text("FSC mask", "sta-setup-fsc-mask",
                                placeholder="path or 'none'"),
                    _field_num("Pixel size (Å)", "sta-setup-pixel-size",
                               value=None, step=0.001),
                    _field_check("Extract subtomograms from tomograms",
                                 "sta-setup-extract-subtomos", value=False),
                    dbc.Collapse(
                        [
                            _field_num("Subtomo size (px)", "sta-setup-subtomo-size",
                                       value=None, min=1, step=1),
                            _field_text("Tomograms", "sta-setup-tomograms",
                                        placeholder="path to tomograms"),
                            _field_num("Tomo digits", "sta-setup-tomo-digits",
                                       value=None, min=1, step=1),
                        ],
                        id="sta-setup-extract-fields",
                        is_open=False,
                    ),
                    html.Hr(style={"margin": "0.4rem 0"}),
                ],
                style=_SHOW,
            ),

            # ── STOPGAP only ──────────────────────────────────────────────────
            html.Div(
                id="sta-setup-stopgap-section",
                children=[
                    html.Div("STOPGAP only", style=_SECTION_HEADER),
                    _field_text("Root directory", "sta-setup-rootdir",
                                placeholder="e.g. ./run42"),
                    _field_num("Binning", "sta-setup-binning",
                               value=None, min=1, step=1),
                    html.Hr(style={"margin": "0.2rem 0"}),
                    _field_dropdown("Reference family", "sta-setup-ref-family",
                                    options=_REF_FAMILY_OPTIONS, value="singleref"),
                    html.Hr(style={"margin": "0.2rem 0"}),
                    html.Div("Preprocessing", style={**_HINT, "fontWeight": 600,
                                                     "marginBottom": "0.25rem"}),
                    _field_check("Apply Laplacian filter",
                                 "sta-setup-apply-laplacian", value=False),
                    _field_check("Calculate exposure weights (calc_exp)",
                                 "sta-setup-calc-exp", value=True),
                    _field_check("Calculate CTF weights (calc_ctf)",
                                 "sta-setup-calc-ctf", value=True),
                    _field_num("Cosine weighting exponent", "sta-setup-cos-weight",
                               value=0.0, step=0.1),
                    _field_num("Score weight (Nyquist pass-through)",
                               "sta-setup-score-weight", value=0.01, step=0.005),
                    html.Hr(style={"margin": "0.2rem 0"}),
                    html.Div("Search / averaging", style={**_HINT, "fontWeight": 600,
                                                          "marginBottom": "0.25rem"}),
                    _field_dropdown("Search mode", "sta-setup-search-mode",
                                    options=[{"label": v, "value": v}
                                             for v in ("hc", "shc")],
                                    value="hc"),
                    _field_dropdown("Cone search type", "sta-setup-cone-search-type",
                                    options=[{"label": v, "value": v}
                                             for v in ("coarse", "complete")],
                                    value="coarse"),
                    _field_dropdown("Scoring function", "sta-setup-scoring-fcn",
                                    options=[{"label": v, "value": v}
                                             for v in ("flcf", "pearson")],
                                    value="flcf"),
                    _field_dropdown("Rotation mode", "sta-setup-rot-mode",
                                    options=[{"label": v, "value": v}
                                             for v in ("linear", "cubic")],
                                    value="linear"),
                    _field_dropdown("Averaging mode", "sta-setup-avg-mode",
                                    options=[{"label": v, "value": v}
                                             for v in ("full", "partial")],
                                    value="full"),
                    _field_num("Subset (%)", "sta-setup-subset",
                               value=100, min=1, max=100, step=1),
                    _field_num("F-threshold (fthresh)", "sta-setup-fthresh",
                               value=800, min=1, step=1),
                    _field_text("Temperature (annealing, 0=off)",
                                "sta-setup-temperature", placeholder="e.g. 0"),
                    html.Hr(style={"margin": "0.2rem 0"}),
                    _field_check("Use Euler search (replaces cone search)",
                                 "sta-setup-use-euler-search", value=False),
                    dbc.Collapse(
                        [
                            _field_text("Euler axes (e.g. ZYZ)", "sta-setup-euler-axes",
                                        placeholder="e.g. ZYZ"),
                            _field_num("Euler 1 incr (°)", "sta-setup-euler-1-incr",
                                       value=None, step=0.1),
                            _field_num("Euler 1 iter", "sta-setup-euler-1-iter",
                                       value=None, min=1, step=1),
                            _field_num("Euler 2 incr (°)", "sta-setup-euler-2-incr",
                                       value=None, step=0.1),
                            _field_num("Euler 2 iter", "sta-setup-euler-2-iter",
                                       value=None, min=1, step=1),
                            _field_num("Euler 3 incr (°)", "sta-setup-euler-3-incr",
                                       value=None, step=0.1),
                            _field_num("Euler 3 iter", "sta-setup-euler-3-iter",
                                       value=None, min=1, step=1),
                        ],
                        id="sta-setup-euler-fields",
                        is_open=False,
                    ),
                    html.Hr(style={"margin": "0.4rem 0"}),
                ],
                style=_HIDE,
            ),

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
    cone_angle: str, cone_sampling: str,
    inplane_angle: str, inplane_sampling: str,
    high_pass: str, high_pass_sigma, low_pass: str, low_pass_sigma,
    threshold: str,
    motl: str, ref: str, mask: str, cc_mask: str,
    wedge_list: str, subtomo_path: str,
    symmetry, class_id,
    split_even_odd: bool = True,
    fsc_mask: str = None, pixel_size=None,
    extract_subtomos: bool = False,
    subtomo_size=None, tomograms: str = None, tomo_digits=None,
    # STOPGAP-only
    rootdir: str = None, binning=None,
    create_ref: bool = False, ref_family: str = "singleref",
    apply_laplacian: bool = False,
    calc_exp: bool = True, calc_ctf: bool = True,
    cos_weight=None, score_weight=None,
    search_mode: str = "hc", cone_search_type: str = "coarse",
    scoring_fcn: str = "flcf", rot_mode: str = "linear",
    avg_mode: str = "full", subset=None, fthresh=None,
    temperature: str = None,
    use_euler_search: bool = False,
    euler_axes: str = None,
    euler_1_incr=None, euler_1_iter=None,
    euler_2_incr=None, euler_2_iter=None,
    euler_3_incr=None, euler_3_iter=None,
) -> dict:
    """Pack the form values into the canonical dict that
    :meth:`sta.StaParameters.from_dict` accepts.  All per-iter values are
    broadcast to ``iter_n`` repeats; empty fields are dropped.
    Keys use canonical names (or snake_case, which from_dict normalises).
    """
    iter_n = max(int(iter_n or 1), 1)
    # Control keys (always emitted so from_dict can set create_ref / ref_family).
    out: dict = {
        "start_index":          int(start_index or 1),
        "create_ref":           1 if create_ref else 0,
        "ref_family":           ref_family or "singleref",
        "use_euler_search":     1 if use_euler_search else 0,
        "split_into_even_odd":  1 if split_even_odd else 0,
    }
    # Per-iteration: only if filled.
    per_iter = {
        "cone_angle":        _expand_for_n(cone_angle, iter_n),
        "cone_sampling":     _expand_for_n(cone_sampling, iter_n),
        "inplane_angle":     _expand_for_n(inplane_angle, iter_n),
        "inplane_sampling":  _expand_for_n(inplane_sampling, iter_n),
        "high_pass":         _expand_for_n(high_pass, iter_n),
        "low_pass":          _expand_for_n(low_pass, iter_n),
        "threshold":         _expand_for_n(threshold, iter_n),
        "temperature":       _expand_for_n(temperature, iter_n),
    }
    for k, v in per_iter.items():
        if v is not None:
            out[k] = v
    # Scalar paths / other — only if filled.
    optional: dict = {
        "motl":             (motl or "").strip() or None,
        "ref":              (ref or "").strip() or None,
        "mask":             (mask or "").strip() or None,
        "cc_mask":          (cc_mask or "").strip() or None,
        "wedge_list":       (wedge_list or "").strip() or None,
        # Merged subtomogram path — maps to canonical "subtomo name".
        "subtomo_name":     (subtomo_path or "").strip() or None,
        "symmetry":         int(symmetry) if symmetry not in (None, "") else None,
        "class":            int(class_id) if class_id not in (None, "") else None,
        "high_pass_sigma":  float(high_pass_sigma) if high_pass_sigma not in (None, "") else None,
        "low_pass_sigma":   float(low_pass_sigma) if low_pass_sigma not in (None, "") else None,
        # novaSTA-only
        "fsc_mask":         (fsc_mask or "").strip() or None,
        "pixel_size":       float(pixel_size) if pixel_size not in (None, "") else None,
        # STOPGAP-only scalars
        "rootdir":          (rootdir or "").strip() or None,
        "binning":          int(binning) if binning not in (None, "") else None,
        "apply_laplacian":  1 if apply_laplacian else 0,
        "calc_exp":         1 if calc_exp else 0,
        "calc_ctf":         1 if calc_ctf else 0,
        "cos_weight":       float(cos_weight) if cos_weight not in (None, "") else None,
        "score_weight":     float(score_weight) if score_weight not in (None, "") else None,
        "search_mode":      search_mode or None,
        "cone_search_type": cone_search_type or None,
        "scoring_fcn":      scoring_fcn or None,
        "rot_mode":         rot_mode or None,
        "avg_mode":         avg_mode or None,
        "subset":           int(subset) if subset not in (None, "") else None,
        "fthresh":          int(fthresh) if fthresh not in (None, "") else None,
        # Euler (STOPGAP with euler search)
        "euler_axes":       (euler_axes or "").strip() or None,
        "euler_1_incr":     float(euler_1_incr) if euler_1_incr not in (None, "") else None,
        "euler_1_iter":     int(euler_1_iter) if euler_1_iter not in (None, "") else None,
        "euler_2_incr":     float(euler_2_incr) if euler_2_incr not in (None, "") else None,
        "euler_2_iter":     int(euler_2_iter) if euler_2_iter not in (None, "") else None,
        "euler_3_incr":     float(euler_3_incr) if euler_3_incr not in (None, "") else None,
        "euler_3_iter":     int(euler_3_iter) if euler_3_iter not in (None, "") else None,
    }
    # Extraction fields (novaSTA only, only when extract_subtomos=True).
    if extract_subtomos:
        optional["extract_subtomos"] = 1
        optional["subtomo_size"] = int(subtomo_size) if subtomo_size not in (None, "") else None
        optional["tomograms"] = (tomograms or "").strip() or None
        optional["tomo_digits"] = int(tomo_digits) if tomo_digits not in (None, "") else None
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
    register_angles_builder_callbacks(app, _ANGLES_PREFIX, with_graphs=False)

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
        State("sta-setup-sta-type", "value"),
        State("sta-setup-iter", "value"),
        State("sta-setup-start-index", "value"),
        State("sta-setup-cone-angle", "value"),
        State("sta-setup-cone-sampling", "value"),
        State("sta-setup-inplane-angle", "value"),
        State("sta-setup-inplane-sampling", "value"),
        State("sta-setup-high-pass", "value"),
        State("sta-setup-high-pass-sigma", "value"),
        State("sta-setup-low-pass", "value"),
        State("sta-setup-low-pass-sigma", "value"),
        State("sta-setup-threshold", "value"),
        State("sta-setup-motl", "value"),
        State("sta-setup-ref", "value"),
        State("sta-setup-mask", "value"),
        State("sta-setup-cc-mask", "value"),
        State("sta-setup-wedge-list", "value"),
        State("sta-setup-subtomo-path", "value"),
        State("sta-setup-symmetry", "value"),
        State("sta-setup-class", "value"),
        State("sta-setup-split-even-odd", "value"),
        State("sta-setup-fsc-mask", "value"),
        State("sta-setup-pixel-size", "value"),
        State("sta-setup-extract-subtomos", "value"),
        State("sta-setup-subtomo-size", "value"),
        State("sta-setup-tomograms", "value"),
        State("sta-setup-tomo-digits", "value"),
        State("sta-setup-rootdir", "value"),
        State("sta-setup-binning", "value"),
        State("sta-setup-create-ref", "value"),
        State("sta-setup-ref-family", "value"),
        State("sta-setup-apply-laplacian", "value"),
        State("sta-setup-calc-exp", "value"),
        State("sta-setup-calc-ctf", "value"),
        State("sta-setup-cos-weight", "value"),
        State("sta-setup-score-weight", "value"),
        State("sta-setup-search-mode", "value"),
        State("sta-setup-cone-search-type", "value"),
        State("sta-setup-scoring-fcn", "value"),
        State("sta-setup-rot-mode", "value"),
        State("sta-setup-avg-mode", "value"),
        State("sta-setup-subset", "value"),
        State("sta-setup-fthresh", "value"),
        State("sta-setup-temperature", "value"),
        State("sta-setup-use-euler-search", "value"),
        State("sta-setup-euler-axes", "value"),
        State("sta-setup-euler-1-incr", "value"),
        State("sta-setup-euler-1-iter", "value"),
        State("sta-setup-euler-2-incr", "value"),
        State("sta-setup-euler-2-iter", "value"),
        State("sta-setup-euler-3-incr", "value"),
        State("sta-setup-euler-3-iter", "value"),
        prevent_initial_call=True,
    )
    def create_setup_df(
        n_clicks,
        sta_type,
        iter_n, start_index,
        cone_angle, cone_sampling, inplane_angle, inplane_sampling,
        high_pass, high_pass_sigma, low_pass, low_pass_sigma, threshold,
        motl, ref, mask, cc_mask, wedge_list, subtomo_path,
        symmetry, class_id,
        split_even_odd, fsc_mask, pixel_size,
        extract_subtomos, subtomo_size, tomograms, tomo_digits,
        rootdir, binning,
        create_ref, ref_family,
        apply_laplacian, calc_exp, calc_ctf,
        cos_weight, score_weight,
        search_mode, cone_search_type, scoring_fcn, rot_mode,
        avg_mode, subset, fthresh, temperature,
        use_euler_search,
        euler_axes,
        euler_1_incr, euler_1_iter,
        euler_2_incr, euler_2_iter,
        euler_3_incr, euler_3_iter,
    ):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        try:
            resolved_sta_type = (sta_type or "novasta").lower()
            params_dict = _setup_form_to_params_dict(
                iter_n=iter_n, start_index=start_index,
                cone_angle=cone_angle, cone_sampling=cone_sampling,
                inplane_angle=inplane_angle, inplane_sampling=inplane_sampling,
                high_pass=high_pass, high_pass_sigma=high_pass_sigma,
                low_pass=low_pass, low_pass_sigma=low_pass_sigma,
                threshold=threshold,
                motl=motl, ref=ref, mask=mask, cc_mask=cc_mask,
                wedge_list=wedge_list, subtomo_path=subtomo_path,
                symmetry=symmetry, class_id=class_id,
                split_even_odd=bool(split_even_odd) if split_even_odd is not None else True,
                fsc_mask=fsc_mask, pixel_size=pixel_size,
                extract_subtomos=bool(extract_subtomos),
                subtomo_size=subtomo_size,
                tomograms=tomograms, tomo_digits=tomo_digits,
                rootdir=rootdir, binning=binning,
                create_ref=bool(create_ref), ref_family=ref_family or "singleref",
                apply_laplacian=bool(apply_laplacian),
                calc_exp=bool(calc_exp) if calc_exp is not None else True,
                calc_ctf=bool(calc_ctf) if calc_ctf is not None else True,
                cos_weight=cos_weight, score_weight=score_weight,
                search_mode=search_mode, cone_search_type=cone_search_type,
                scoring_fcn=scoring_fcn, rot_mode=rot_mode,
                avg_mode=avg_mode, subset=subset, fthresh=fthresh,
                temperature=temperature,
                use_euler_search=bool(use_euler_search),
                euler_axes=euler_axes,
                euler_1_incr=euler_1_incr, euler_1_iter=euler_1_iter,
                euler_2_incr=euler_2_incr, euler_2_iter=euler_2_iter,
                euler_3_incr=euler_3_incr, euler_3_iter=euler_3_iter,
            )
            params = sta_mod.StaParameters.from_dict(params_dict, sta_type=resolved_sta_type)
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

    # ── Show/hide format-specific sections based on STA type ─────────────────
    @app.callback(
        Output("sta-setup-novasta-section", "style"),
        Output("sta-setup-stopgap-section", "style"),
        Input("sta-setup-sta-type", "value"),
    )
    def _toggle_sta_type_sections(sta_type):
        if sta_type == "stopgap":
            return _HIDE, _SHOW
        return _SHOW, _HIDE

    # ── Show/hide Euler fields when Euler search checkbox is toggled ──────────
    @app.callback(
        Output("sta-setup-euler-fields", "is_open"),
        Input("sta-setup-use-euler-search", "value"),
    )
    def _toggle_euler_fields(use_euler):
        return bool(use_euler)

    # ── Show/hide tomogram extraction fields ─────────────────────────────────
    @app.callback(
        Output("sta-setup-extract-fields", "is_open"),
        Input("sta-setup-extract-subtomos", "value"),
    )
    def _toggle_extract_fields(extract):
        return bool(extract)

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
            run_operation(params.write_out, {"path": str(path).strip()})
        except Exception as exc:
            return f"Save as {target} failed: {exc}"
        return f"Saved as {target} → {str(path).strip()}"
