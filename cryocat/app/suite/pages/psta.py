"""STA tool — subtomogram-averaging setup, classification and run-folder creation.

Sidebar accordion (top-level menu):

  * **Evaluation** — either-or loader (parameter file or motl base + iter
    range).  Once a config is loaded, the two action buttons dispatch:

    - *Alignment evaluation* -> ``sta.compute_alignment_statistics`` ->
      every column of the result rendered as a line plot in a 3-column
      grid (tab "Alignment evaluation").
    - *Classification evaluation* -> ``sta.evaluate_classification`` ->
      ``visplot.plot_classification_convergence`` (tab "Classification
      evaluation").

  * **STA setup** — block-schedule parameter-file form.  A schedule table
    replaces the old flat "N iterations" field; each row is one
    :class:`~cryocat.analysis.sta.Block`.  A "Create" button builds the
    parameter DataFrame (tab 4 "STA setup output") for inline editing and
    save as novaSTA or STOPGAP.  Templates are offered when
    ``sta_type == "stopgap"`` and ``ref_family == "multiref"``.

  * **Classification setup** — motl preparation for multi-reference runs.
    De-novo (random reference motls) and existing-references modes.

  * **Run folder creation** — STOPGAP folder layout, preflight, creation
    and (optional) reference renaming.  STOPGAP only.

Main area: ``dcc.Tabs`` with five tabs (Parameter file table /
Alignment / Classification / STA setup output / Run folder manifest).

Contract: exposes ``layout`` and ``register_callbacks(app)``.
"""
from __future__ import annotations

import base64
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
from cryocat.app import formgen, ids, styles as _styles
from cryocat.app.formgen import make_dropdown
from cryocat.app.components.pathfield import get_path_field
from cryocat.app.apputils import run_operation
from cryocat.app.components.anglesbuilder import register_angles_builder_callbacks
from cryocat.app.components.poolpicker import get_pool_picker, register_pool_picker_callbacks
from cryocat.app.components.graphsettings import styled_figure, error_figure
from cryocat.utils.geom import generate_angles
from cryocat.app.pageshell import page_shell, sidebar_accordion
from cryocat.analysis.sta import (
    CoassignmentFactor,
    build_coassignment_factor,
    consensus_groups,
    reliability_summary,
)
from cryocat.app.pool import PoolState, get_rows, insert_motl


_HINT = {"color": "var(--color9)", "margin": "0.3rem 0"}
_LBL = {"marginBottom": "2px"}
_PLACEHOLDER = "Coming later — set up by the user."
_SECTION_HEADER = {"fontWeight": 600,
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
# use inline dbc.Checklist (single option) for consistent alignment.

_FIELD_ROW = {
    "display": "flex",
    "alignItems": "center",
    "gap": "0.5rem",
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
        style={**_FIELD_ROW, "marginBottom": "0.4rem"},
    )


def _field_num(label: str, id_: str, value=None, **kwargs) -> html.Div:
    return html.Div(
        [
            html.Label(label, style=_FIELD_LABEL),
            dbc.Input(id=id_, type="number", value=value, size="sm",
                      style=_FIELD_INPUT, **kwargs),
        ],
        style={**_FIELD_ROW, "marginBottom": "0.4rem"},
    )


def _field_check(label: str, id_: str, value: bool = False) -> dbc.Checklist:
    return dbc.Checklist(
        id=id_,
        options=[{"label": label, "value": "on"}],
        value=["on"] if value else [],
        inline=True,
        inputStyle={"verticalAlign": "middle", "marginTop": "-2px"},
        labelStyle={"verticalAlign": "middle"},
        style={"marginBottom": "0.4rem"},
    )


def _field_dropdown(label: str, id_: str, options: list, value=None) -> html.Div:
    return html.Div(
        [
            html.Label(label, style=_FIELD_LABEL),
            make_dropdown(id_, options, value, clearable=False,
                          style=_FIELD_INPUT),
        ],
        style={**_FIELD_ROW, "marginBottom": "0.4rem"},
    )


# ── Evaluation accordion panel ───────────────────────────────────────────────


def _param_file_form() -> html.Div:
    return html.Div(
        [
            html.Label("Parameter file path", style=_LBL),
            html.Div(get_path_field("sta-param-path", extensions=(".txt", ".star"),
                                    placeholder="path/to/params.txt  or  .star"),
                     style={"marginBottom": "0.4rem"}),
            html.Label("Working directory (optional override)", style=_LBL),
            html.Div(get_path_field("sta-param-working-dir", mode="directory",
                                    placeholder="leave blank to use rootdir / stored path"),
                     style={"marginBottom": "0.4rem"}),
            html.Div(
                "STOPGAP: replaces the params' rootdir (lists/, masks/, refs/ "
                "subdirs are still appended). novaSTA: joins onto relative motl "
                "paths and replaces the directory portion of absolute ones.",
                style={
                    "color": "var(--color9)",
                    "marginBottom": "0.5rem",
                },
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("File type", style=_LBL),
                            make_dropdown("sta-param-type", [
                                {"label": "Auto-detect", "value": "auto"},
                                {"label": "novaSTA (.txt)", "value": "novasta"},
                                {"label": "STOPGAP (.star)", "value": "stopgap"},
                            ], "auto", clearable=False),
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
                    "color": "var(--color9)",
                    "marginTop": "0.4rem", "wordBreak": "break-word",
                },
            ),
        ],
        id="sta-eval-params-block",
    )


def _evaluation_panel() -> html.Div:
    return html.Div(
        [
            html.Label("Loader", style=_LBL),
            html.Div(
                dcc.RadioItems(
                    id="sta-eval-mode",
                    options=[
                        {"label": " From parameter file", "value": "params"},
                        {"label": " From loaded motls", "value": "pool"},
                    ],
                    value="params",
                    inputStyle=_styles.RADIO_INLINE_INPUT,
                    labelStyle=_styles.RADIO_INLINE_LABEL,
                    style={"display": "flex", "flexWrap": "wrap", "alignItems": "center"},
                ),
                style={"marginBottom": "0.6rem"},
            ),
            html.Div(_param_file_form(), id="sta-eval-params-wrapper",
                     style={"display": "block"}),
            html.Div(
                id="sta-pool-wrapper",
                style={"display": "none"},
                children=[
                    get_pool_picker("sta-pool", label="Pool source (one group = one motl per iteration)"),
                    dbc.Button(
                        "Load from pool",
                        id="sta-pool-load-btn",
                        color="primary",
                        size="sm",
                        style={"width": "100%", "marginTop": "0.4rem", "marginBottom": "0.3rem"},
                    ),
                    html.Div(
                        id="sta-pool-status",
                        style={
                            "color": "var(--color9)",
                            "marginTop": "0.2rem", "wordBreak": "break-word",
                        },
                    ),
                ],
            ),
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

_ANG_RANGE_COL_DEFS = [
    {"field": "from_iter",        "headerName": "Iteration from",   "editable": True, "flex": 1, "type": "numericColumn"},
    {"field": "to_iter",          "headerName": "Iteration to",     "editable": True, "flex": 1, "type": "numericColumn"},
    {"field": "cone_angle",       "headerName": "Cone angle",       "editable": True, "flex": 2},
    {"field": "cone_sampling",    "headerName": "Cone step (deg)",  "editable": True, "flex": 2},
    {"field": "inplane_angle",    "headerName": "Inplane angle",    "editable": True, "flex": 2},
    {"field": "inplane_sampling", "headerName": "Inplane step (deg)", "editable": True, "flex": 2},
]

_BP_RANGE_COL_DEFS = [
    {"field": "from_iter",       "headerName": "Iteration from",  "editable": True, "flex": 1, "type": "numericColumn"},
    {"field": "to_iter",         "headerName": "Iteration to",    "editable": True, "flex": 1, "type": "numericColumn"},
    {"field": "high_pass",       "headerName": "High pass (px)",  "editable": True, "flex": 2},
    {"field": "high_pass_sigma", "headerName": "High pass sigma", "editable": True, "flex": 2},
    {"field": "low_pass",        "headerName": "Low pass (px)",   "editable": True, "flex": 2},
    {"field": "low_pass_sigma",  "headerName": "Low pass sigma",  "editable": True, "flex": 2},
]


def _sep_range_modal(group: str, title: str, col_defs: list) -> dbc.Modal:
    """Build the 'set separately' range-table modal for *group* (``'ang'`` or ``'bp'``)."""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(title)),
            dbc.ModalBody(
                [
                    html.Div(
                        "Specify iteration ranges. Ranges must be contiguous, "
                        "start at 1 and end at the total number of iterations.",
                        style={**_HINT, "marginBottom": "0.4rem"},
                    ),
                    dag.AgGrid(
                        id=f"sta-{group}-sep-grid",
                        rowData=[],
                        columnDefs=col_defs,
                        defaultColDef={
                            "resizable": True, "sortable": False, "filter": False,
                            "cellStyle": {"textAlign": "center"},
                            "wrapHeaderText": True,
                            "autoHeaderHeight": True,
                        },
                        dashGridOptions={
                            "singleClickEdit": True,
                            "stopEditingWhenCellsLoseFocus": True,
                            "rowSelection": "single",
                            "suppressMovableColumns": True,
                        },
                        style={"height": "220px", "width": "100%"},
                        className="ag-theme-balham sta-sep-grid",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(dbc.Button("Split", id=f"sta-{group}-sep-split-btn",
                                               color="secondary", size="sm",
                                               style={"width": "100%"}), width=4),
                            dbc.Col(dbc.Button("Remove", id=f"sta-{group}-sep-remove-btn",
                                               color="secondary", size="sm",
                                               style={"width": "100%"}), width=4),
                            dbc.Col(dbc.Button("Reset", id=f"sta-{group}-sep-reset-btn",
                                               color="warning", size="sm",
                                               style={"width": "100%"}), width=4),
                        ],
                        className="g-1",
                        style={"marginTop": "0.3rem"},
                    ),
                    html.Div(
                        id=f"sta-{group}-sep-error",
                        style={"color": "var(--bs-danger)", "marginTop": "0.3rem",
                               "fontSize": _styles.FONT_SM, "whiteSpace": "pre-wrap"},
                    ),
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button("Accept", id=f"sta-{group}-sep-accept-btn",
                               color="primary", className="me-2"),
                    dbc.Button("Cancel", id=f"sta-{group}-sep-cancel-btn",
                               color="secondary"),
                ]
            ),
        ],
        id=f"sta-{group}-sep-modal",
        size="xl",
        is_open=False,
        centered=True,
        scrollable=True,
    )


def _sta_setup_panel() -> html.Div:
    return html.Div(
        [
            # ── Format ───────────────────────────────────────────────────────
            html.Div("Format", style=_SECTION_HEADER),
            _field_dropdown("STA format", "sta-setup-sta-type",
                            options=_STA_TYPE_OPTIONS, value="novasta"),
            html.Hr(style={"margin": "0.4rem 0"}),

            # ── Iterations ────────────────────────────────────────────────────────────
            html.Div("Iterations", style=_SECTION_HEADER),
            _field_num("Number of iterations", "sta-setup-iter", value=10, min=1, step=1),
            _field_num("Starting iteration", "sta-setup-start-index",
                       value=1, min=1, step=1),
            _field_check("Continue an existing run (prefill from last param row)",
                         "sta-setup-continue-run"),
            html.Div(
                id="sta-setup-continue-status",
                style={**_HINT, "marginBottom": "0.3rem", "wordBreak": "break-word"},
            ),
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
                    dbc.Button(
                        "Set separately…",
                        id="sta-ang-sep-btn",
                        color="secondary",
                        size="sm",
                    ),
                ],
                style={**{"display": "flex", "alignItems": "center", "gap": "0.25rem"},
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
                id="sta-ang-sep-display",
                style={**_HINT, "fontSize": _styles.FONT_TIGHT, "display": "none"},
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
            html.Hr(style={"margin": "0.2rem 0"}),
            html.Div(
                [
                    html.Span("Bandpass filter", style={**_HINT, "fontWeight": 600}),
                    dbc.Button(
                        "Set separately…",
                        id="sta-bp-sep-btn",
                        color="secondary",
                        size="sm",
                        style={"marginLeft": "auto"},
                    ),
                ],
                style={**{"display": "flex", "alignItems": "center"}, "marginBottom": "0.25rem"},
            ),
            _field_text("High pass (px)", "sta-setup-high-pass",
                        placeholder="e.g. 25 20 15"),
            _field_num("High pass sigma", "sta-setup-high-pass-sigma",
                       value=2.0, step=0.5),
            _field_text("Low pass (px)", "sta-setup-low-pass",
                        placeholder="e.g. 30"),
            _field_num("Low pass sigma", "sta-setup-low-pass-sigma",
                       value=3.0, step=0.5),
            html.Div(
                id="sta-bp-sep-display",
                style={**_HINT, "fontSize": _styles.FONT_TIGHT, "display": "none"},
            ),
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


def _classification_setup_panel() -> html.Div:
    """Part B — classification motl preparation. Classification is always multiref."""
    return html.Div(
        [
            # B1 vs B2 workflow choice — run mode is always multiref
            dcc.RadioItems(
                id="sta-cls-workflow",
                options=[
                    {"label": " De-novo (create random reference motls)",
                     "value": "denovo"},
                    {"label": " Existing references (assign random classes)",
                     "value": "exrefs"},
                ],
                value="denovo",
                inputStyle={"verticalAlign": "middle", "marginTop": "-2px", "marginRight": "0.4rem"},
                labelStyle={"display": "block", "marginBottom": "0.3rem", "verticalAlign": "middle"},
                style={"marginBottom": "0.5rem"},
            ),

            # ── B1: De-novo ───────────────────────────────────────────────────
            html.Div(
                id="sta-cls-denovo-section",
                children=[
                    html.Div("De-novo (random reference motls)", style=_SECTION_HEADER),
                    html.Label("Input motl", style=_LBL),
                    html.Div(
                        get_path_field("sta-cls-dn-input-motl",
                                       placeholder="path to input motl"),
                        style={"marginBottom": "0.4rem"},
                    ),
                    _field_dropdown("Input motl type", "sta-cls-dn-input-type",
                                    options=_MOTL_TYPE_OPTIONS, value="emmotl"),
                    _field_num("Number of classes", "sta-cls-dn-n-classes",
                               value=6, min=2, step=1),
                    _field_num("Class occupancy (particles/class)",
                               "sta-cls-dn-occupancy", value=None, min=1, step=1),
                    html.Small(
                        "Leave blank for auto (10% of particles).",
                        style={**_HINT, "marginBottom": "0.3rem"},
                    ),
                    _field_num("Number of runs", "sta-cls-dn-n-runs",
                               value=4, min=1, step=1),
                    _field_num("Iteration number", "sta-cls-dn-iter",
                               value=1, min=1, step=1),
                    _field_dropdown("Output motl type", "sta-cls-dn-output-type",
                                    options=[
                                        {"label": "stopgap (.star)", "value": "stopgap"},
                                        {"label": "emmotl (.em)", "value": "emmotl"},
                                    ], value="stopgap"),
                    html.Label("Output motl base path", style=_LBL),
                    html.Div(
                        get_path_field("sta-cls-dn-output-base",
                                       mode="save",
                                       placeholder="e.g. /data/run1/ref_motl"),
                        style={"marginBottom": "0.4rem"},
                    ),
                    dbc.Button(
                        "Create de-novo motls",
                        id="sta-cls-dn-create-btn",
                        color="primary", size="sm",
                        style={"width": "100%"},
                    ),
                    html.Div(
                        id="sta-cls-dn-status",
                        style={**_HINT, "marginTop": "0.4rem", "wordBreak": "break-word"},
                    ),
                ],
                style=_SHOW,
            ),

            # ── B2: Existing references ───────────────────────────────────────
            html.Div(
                id="sta-cls-exrefs-section",
                children=[
                    html.Div(
                        "Existing references (random class assignment)",
                        style=_SECTION_HEADER,
                    ),
                    html.Small(
                        "Creates one motl with randomly assigned classes, shared by "
                        "all runs (they differ only in STOPGAP's stochastic search order).",
                        style={**_HINT, "marginBottom": "0.3rem"},
                    ),
                    html.Label("Input motl", style=_LBL),
                    html.Div(
                        get_path_field("sta-cls-er-input-motl",
                                       placeholder="path to input motl"),
                        style={"marginBottom": "0.4rem"},
                    ),
                    _field_dropdown("Input motl type", "sta-cls-er-input-type",
                                    options=_MOTL_TYPE_OPTIONS, value="emmotl"),
                    _field_num("Number of classes", "sta-cls-er-n-classes",
                               value=6, min=2, step=1),
                    _field_num("Iteration number", "sta-cls-er-iter",
                               value=1, min=1, step=1),
                    _field_dropdown("Output motl type", "sta-cls-er-output-type",
                                    options=[
                                        {"label": "stopgap (.star)", "value": "stopgap"},
                                        {"label": "emmotl (.em)", "value": "emmotl"},
                                    ], value="stopgap"),
                    html.Label("Output motl base path", style=_LBL),
                    html.Div(
                        get_path_field("sta-cls-er-output-base",
                                       mode="save",
                                       placeholder="e.g. /data/run1/motl"),
                        style={"marginBottom": "0.4rem"},
                    ),
                    dbc.Button(
                        "Create motl with random classes",
                        id="sta-cls-er-create-btn",
                        color="primary", size="sm",
                        style={"width": "100%"},
                    ),
                    html.Div(
                        id="sta-cls-er-status",
                        style={**_HINT, "marginTop": "0.4rem", "wordBreak": "break-word"},
                    ),
                ],
                style=_HIDE,
            ),

            # ── R4: Generate parameter table ──────────────────────────────────
            html.Hr(style={"margin": "0.5rem 0"}),
            html.Div("Generate parameter table", style=_SECTION_HEADER),
            html.Small(
                "Writes STOPGAP multiref rows from the current STA setup form values. "
                "The table is fully editable after generation.",
                style={**_HINT, "marginBottom": "0.4rem"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "De-novo (31 rows)",
                            id="sta-cls-gen-denovo-btn",
                            color="secondary", size="sm",
                            style={"width": "100%"},
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Existing refs (30 rows)",
                            id="sta-cls-gen-exrefs-btn",
                            color="secondary", size="sm",
                            style={"width": "100%"},
                        ),
                        width=6,
                    ),
                ],
                className="g-1",
            ),
            html.Div(
                id="sta-cls-gen-status",
                style={**_HINT, "marginTop": "0.3rem", "wordBreak": "break-word"},
            ),
        ]
    )


def _run_folder_panel() -> html.Div:
    """Part C — STOPGAP run-folder creation."""
    return html.Div(
        [
            html.Div(
                id="sta-rf-prefill-source",
                style={**_HINT, "marginBottom": "0.4rem"},
            ),
            # ── Paths ─────────────────────────────────────────────────────────
            html.Div("Paths", style=_SECTION_HEADER),
            html.Label("Output base directory", style=_LBL),
            html.Div(
                get_path_field("sta-rf-output-base", mode="directory",
                               placeholder="parent folder for run directories"),
                style={"marginBottom": "0.4rem"},
            ),
            _field_text("Folder name (base)", "sta-rf-folder-name",
                        placeholder="e.g. run01"),
            _field_check(
                "Classification multirun (create one folder per run)",
                "sta-rf-multirun",
            ),
            html.Label("Subtomogram path (will be symlinked)", style=_LBL),
            html.Div(
                get_path_field("sta-rf-subtomo-path", mode="directory",
                               placeholder="path to subtomograms directory"),
                style={"marginBottom": "0.4rem"},
            ),
            html.Hr(style={"margin": "0.4rem 0"}),

            # ── Masks / wedgelist / motls ─────────────────────────────────────
            html.Div("Files to copy", style=_SECTION_HEADER),
            html.Label("Alignment mask (mask_name)", style=_LBL),
            html.Div(
                get_path_field("sta-rf-mask", extensions=(".em",),
                               placeholder="path to mask file"),
                style={"marginBottom": "0.4rem"},
            ),
            html.Label("CC mask (ccmask_name)", style=_LBL),
            html.Div(
                get_path_field("sta-rf-ccmask", extensions=(".em",),
                               placeholder="path to CC mask file"),
                style={"marginBottom": "0.4rem"},
            ),
            html.Label("Wedge list (wedgelist_name)", style=_LBL),
            html.Div(
                get_path_field("sta-rf-wedgelist",
                               placeholder="path to wedge list file"),
                style={"marginBottom": "0.4rem"},
            ),
            html.Label(
                "Motl files (comma-separated paths or from classification setup)",
                style=_LBL,
            ),
            dbc.Textarea(
                id="sta-rf-motl-paths",
                placeholder="path1.star, path2.star  (or leave blank to use classification output)",
                rows=2,
                style={"fontSize": _styles.FONT_SM, "marginBottom": "0.4rem"},
            ),
            html.Label("Existing reference files (comma-separated)", style=_LBL),
            dbc.Textarea(
                id="sta-rf-ref-files",
                placeholder="ref_1.em, ref_2.em …",
                rows=2,
                style={"fontSize": _styles.FONT_SM, "marginBottom": "0.4rem"},
            ),
            html.Hr(style={"margin": "0.4rem 0"}),

            # ── Actions ───────────────────────────────────────────────────────
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button("Preflight check", id="sta-rf-preflight-btn",
                                   color="secondary", size="sm",
                                   style={"width": "100%"}),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Button("Create folders", id="sta-rf-create-btn",
                                   color="primary", size="sm",
                                   style={"width": "100%"}),
                        width=6,
                    ),
                ],
                className="g-1",
            ),
            _field_check("Overwrite existing folders", "sta-rf-overwrite"),
            html.Div(
                id="sta-rf-status",
                style={**_HINT, "marginTop": "0.5rem", "wordBreak": "break-word",
                       "whiteSpace": "pre-wrap"},
            ),
        ]
    )


# ── Multi-class consensus helpers ────────────────────────────────────────────

_MC_STYLE_BASE     = _styles.INLINE_CTRL_ROW
_MC_STYLE_DISABLED = {**_MC_STYLE_BASE, "opacity": "0.4", "pointerEvents": "none"}
_MC_STYLE_ENABLED  = _MC_STYLE_BASE


def _mc_serialize_factor(factor):
    def _enc(arr):
        return {
            "b64": base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode(),
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        }
    return {
        "labels": _enc(factor.labels),
        "particle_ids": _enc(factor.particle_ids),
        "run_labels": factor.run_labels,
        "n_classes": _enc(factor.n_classes),
    }


def _mc_deserialize_factor(data):
    def _dec(d, dtype):
        buf = base64.b64decode(d["b64"])
        return np.frombuffer(buf, dtype=dtype).reshape(d["shape"]).copy()
    return CoassignmentFactor(
        labels=_dec(data["labels"], np.int32),
        particle_ids=_dec(data["particle_ids"], data["particle_ids"]["dtype"]),
        run_labels=data["run_labels"],
        n_classes=_dec(data["n_classes"], np.int32),
    )


def _mc_build_factor(motl_ids, registry):
    """Return (factor, error_msg) from a flat list of pool motl IDs."""
    pool = PoolState.from_stores(registry or {})
    try:
        dfs = [get_rows(mid, state=pool) for mid in motl_ids]
        names = [(registry or {}).get(mid, {}).get("label", mid) for mid in motl_ids]
        return build_coassignment_factor(dfs, run_labels=names), None
    except Exception as exc:
        return None, str(exc)


def _mc_slider_marks(R):
    if R <= 8:
        return {k: str(k) for k in range(R + 1)}
    step = max(1, R // 5)
    marks = {k: str(k) for k in range(0, R + 1, step)}
    marks[R] = str(R)
    return marks


def _mc_verdict_children(summary):
    ok = summary["reliable"]
    color = "#EAAE47" if ok else "var(--bs-warning)"
    icon = "✓" if ok else "⚠"
    return [
        html.Span(f"{icon}  ", style={"color": color, "fontWeight": 700}),
        html.Span(summary["verdict"], style={"fontSize": _styles.FONT_SM}),
        html.Br(),
        html.Small(
            f"{summary['n_particles']} particles · {summary['n_runs']} runs",
            style={"color": "var(--color9)"},
        ),
    ]


def _mc_diagnose_figures(summary, gs, factor=None):
    settings = gs or {}
    row1 = []  # agreement histogram + cumulative + eigenvalue spectrum
    row2 = []  # ARI heatmap + bar chart (if factor supplied)

    hist = summary.get("histogram")
    if hist is not None:
        R = summary["n_runs"]
        is_exact = "exact" not in hist.columns or bool(hist["exact"].iloc[0])
        t_note = "" if is_exact else "estimated from 1M sampled pairs"
        t_margin = 30 if is_exact else 42

        # --- agreement histogram (fraction of pairs) ---
        fig_h = go.Figure(go.Bar(
            x=hist["n_runs_agreeing"].tolist(),
            y=hist["fraction_of_pairs"].tolist(),
            marker_color="steelblue",
        ))
        fig_h.update_layout(
            title_text=t_note, title_x=0.5, title_font_size=10,
            xaxis_title=f"Runs agreeing (0–{R})",
            yaxis_title="Fraction of pairs",
            margin={"l": 50, "r": 10, "t": t_margin, "b": 40},
        )
        row1.append(dcc.Graph(
            figure=styled_figure(fig_h, settings, uirevision="mc-hist"),
            style={"flex": "1", "minWidth": 0, "height": "220px"},
        ))

        # --- cumulative histogram (threshold preview, C3) ---
        fig_c = go.Figure(go.Scatter(
            x=hist["n_runs_agreeing"].tolist(),
            y=hist["fraction_at_least_k"].tolist(),
            mode="lines+markers",
            marker={"size": 5},
            line={"color": "steelblue"},
        ))
        fig_c.update_layout(
            title_text="Threshold preview" + (f" ({t_note})" if t_note else ""),
            title_x=0.5, title_font_size=10,
            xaxis_title=f"Threshold k (0–{R})",
            yaxis_title="Fraction assigned ≥ k",
            yaxis={"range": [0, 1]},
            margin={"l": 50, "r": 10, "t": t_margin, "b": 40},
        )
        row1.append(dcc.Graph(
            figure=styled_figure(fig_c, settings, uirevision="mc-cumul"),
            style={"flex": "1", "minWidth": 0, "height": "220px"},
        ))

    evals = summary.get("eigenvalues")
    if evals is not None and len(evals):
        ev_list = evals.tolist() if hasattr(evals, "tolist") else list(evals)
        # De-emphasise leading eigenvalue (reflects mean similarity, not structure)
        # Plot from component 2; annotate component 1 value separately
        if len(ev_list) >= 2:
            x_vals = list(range(2, len(ev_list) + 1))
            y_vals = ev_list[1:]
            ev1_note = f"λ₁={ev_list[0]:.3g} (mean similarity, de-emphasised)"
        else:
            x_vals = [1]
            y_vals = ev_list
            ev1_note = ""
        fig_e = go.Figure(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines+markers", marker={"size": 5},
        ))
        fig_e.update_layout(
            xaxis_title="Component",
            yaxis_title="Eigenvalue",
            margin={"l": 50, "r": 10, "t": 42 if ev1_note else 30, "b": 40},
            title_text=ev1_note, title_x=0.5, title_font_size=10,
        )
        row1.append(dcc.Graph(
            figure=styled_figure(fig_e, settings, uirevision="mc-evals"),
            style={"flex": "1", "minWidth": 0, "height": "220px"},
        ))

    # --- per-run ARI (C2) ---
    if factor is not None:
        try:
            ari_data = factor.run_agreement()
            mat = ari_data["matrix"]
            smry = ari_data["summary"].sort_values("mean_ari")

            # heatmap
            fig_heat = go.Figure(go.Heatmap(
                z=mat.values.tolist(),
                x=mat.columns.tolist(),
                y=mat.index.tolist(),
                colorscale="RdYlGn",
                zmin=-0.2, zmax=1.0,
                showscale=True,
                colorbar={"thickness": 10, "len": 0.8},
            ))
            fig_heat.update_layout(
                title_text="Per-run ARI", title_x=0.5, title_font_size=11,
                xaxis={"tickangle": -45, "tickfont": {"size": 8}},
                yaxis={"tickfont": {"size": 8}},
                margin={"l": 70, "r": 10, "t": 36, "b": 60},
            )
            row2.append(dcc.Graph(
                figure=styled_figure(fig_heat, settings, uirevision="mc-ari-heat"),
                style={"flex": "1", "minWidth": 0, "height": "260px"},
            ))

            # bar chart
            fig_bar = go.Figure(go.Bar(
                x=smry["mean_ari"].tolist(),
                y=smry["run"].tolist(),
                orientation="h",
                marker_color="steelblue",
                error_x={"type": "data",
                          "arrayminus": (smry["mean_ari"] - smry["min_ari"]).tolist(),
                          "array":      (smry["max_ari"]  - smry["mean_ari"]).tolist()},
            ))
            fig_bar.update_layout(
                title_text="Mean ARI per run", title_x=0.5, title_font_size=11,
                xaxis_title="ARI", xaxis={"range": [-0.2, 1.0]},
                yaxis={"tickfont": {"size": 8}},
                margin={"l": 120, "r": 10, "t": 36, "b": 40},
            )
            row2.append(dcc.Graph(
                figure=styled_figure(fig_bar, settings, uirevision="mc-ari-bar"),
                style={"flex": "1", "minWidth": 0, "height": "260px"},
            ))
        except Exception:
            pass

    verdict = html.Div(_mc_verdict_children(summary),
                       style={"marginBottom": "0.5rem", "fontSize": _styles.FONT_SM})
    sections = [verdict]
    if row1:
        sections.append(html.Div(row1, style={"display": "flex", "gap": "0.5rem"}))
    if row2:
        sections.append(html.Small(
            "Per-run agreement (ARI — adjusted Rand index, corrected for chance)",
            style={"color": "var(--color9)", "display": "block", "marginTop": "0.75rem"},
        ))
        sections.append(html.Div(
            html.Div(row2, style={"display": "flex", "gap": "0.5rem"}),
            style={"marginTop": "0.25rem"},
        ))
    if len(sections) == 1:
        return verdict
    return html.Div(sections)


def _mc_run_consensus(factor, k_val, linkage, min_gs):
    R = factor.n_runs
    k = int(k_val) if k_val is not None else R
    t = k / R if R > 0 else 1.0
    sz = int(min_gs) if min_gs else 1
    return consensus_groups(
        factor, min_agreement=t, linkage=linkage or "complete", min_group_size=sz
    )


def _mc_preview_children(result, gs):
    settings = gs or {}
    real = result.group_sizes[result.group_sizes.index != result.junk_class].sort_values(
        ascending=False
    )
    fig_s = go.Figure(go.Bar(
        x=[str(i) for i in real.index],
        y=real.values.tolist(),
        marker_color="steelblue",
    ))
    fig_s.update_layout(xaxis_title="Class", yaxis_title="Particles",
                        margin={"l": 50, "r": 10, "t": 30, "b": 40})
    fig_b = go.Figure([
        go.Bar(name="Assigned", x=[""], y=[result.n_assigned], marker_color="steelblue"),
        go.Bar(name="Junk",     x=[""], y=[result.n_junk],     marker_color="salmon"),
    ])
    fig_b.update_layout(barmode="stack", margin={"l": 50, "r": 10, "t": 30, "b": 40})
    n_g = int((result.group_sizes.index != result.junk_class).sum())
    return [
        html.P(
            f"{result.n_assigned} assigned · {n_g} groups · {result.n_junk} junk"
            f"  (t={result.min_agreement:.3f}, {result.method})",
            style={"fontSize": _styles.FONT_SM, "color": "var(--color9)", "margin": "0 0 0.4rem"},
        ),
        html.Div(
            [
                dcc.Graph(figure=styled_figure(fig_s, settings, uirevision="mc-sizes"),
                          style={"flex": "2", "minWidth": 0, "height": "200px"}),
                dcc.Graph(figure=styled_figure(fig_b, settings, uirevision="mc-bar"),
                          style={"flex": "1", "minWidth": 0, "height": "200px"}),
            ],
            style={"display": "flex", "gap": "0.5rem"},
        ),
    ]


def _mc_heatmap_div(factor, result, gs):
    settings = gs or {}
    try:
        m = factor.matrix()
    except Exception:
        return html.P("Heatmap skipped: too many particles (N > 5 000).",
                      style={"color": "var(--color9)", "fontSize": _styles.FONT_SM})
    order = np.argsort(result.labels)
    m_ord = m[order][:, order]
    fig = go.Figure(go.Heatmap(z=m_ord.tolist(), colorscale="Viridis", zmin=0.0, zmax=1.0))
    fig.update_layout(
        xaxis={"showticklabels": False},
        yaxis={"showticklabels": False, "autorange": "reversed"},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
    )
    return dcc.Graph(figure=styled_figure(fig, settings, uirevision="mc-heatmap"),
                     style={"height": "400px"})


def _mc_apply_labels(result, source_df, class_column="class"):
    df = source_df.copy()
    pids = df["subtomo_id"].to_numpy()
    idx = np.searchsorted(result.particle_ids, pids)
    idx_c = np.clip(idx, 0, len(result.particle_ids) - 1)
    in_f = result.particle_ids[idx_c] == pids
    labels = np.where(in_f, result.labels[idx_c], result.junk_class).astype(np.int32)
    df[class_column] = labels
    return df


def _mc_execute_produce(factor, k_val, linkage, min_gs, motl_ids, registry, meta, next_id):
    R = factor.n_runs
    k = int(k_val) if k_val is not None else R
    t = k / R if R > 0 else 1.0
    sz = int(min_gs) if min_gs else 1
    result = run_operation(consensus_groups, {
        "factor": factor,
        "min_agreement": t,
        "linkage": linkage or "complete",
        "min_group_size": sz,
    })
    pool = PoolState.from_stores(registry, meta, next_id)
    src_df = get_rows(motl_ids[0], state=pool)
    out_df = _mc_apply_labels(result, src_df)
    n_g = int((result.group_sizes.index != result.junk_class).sum())
    label = f"MultiClass({n_g} classes, t={result.min_agreement:.2f})"
    new_state, mid = insert_motl(pool, out_df, label=label)
    return new_state, mid, result


def _mc_produce_status(mid, result):
    n_g = int((result.group_sizes.index != result.junk_class).sum())
    n_tot = result.n_assigned + result.n_junk
    return html.Span(
        f"Added to pool as {mid}: {n_g} groups, {result.n_assigned}/{n_tot} assigned.",
        style={"color": "#EAAE47"},
    )


def _mc_panel() -> html.Div:
    return html.Div([
        get_pool_picker("sta-mc-pool", label="Motls for consensus (select ≥ 2)"),
        html.Small(
            "Select motls in the dropdown above — no separate load step needed.",
            style={"color": "var(--color9)", "display": "block", "margin": "0.3rem 0 0.5rem"},
        ),
        html.Hr(style={"margin": "0.5rem 0"}),
        dbc.Button(
            "Run analysis",
            id="sta-mc-run-btn",
            color="primary",
            size="sm",
            style={"width": "100%", "marginBottom": "0.3rem"},
        ),
        html.Div(id="sta-mc-run-status", style={**_HINT, "wordBreak": "break-word"}),
        html.Hr(style={"margin": "0.5rem 0"}),
        html.Label("Threshold (k)", style=_LBL),
        dcc.Slider(
            id="sta-mc-t-slider", min=0, max=0, step=1, value=0, marks={},
            updatemode="drag",
        ),
        html.Small(
            "k/R runs must co-assign a pair; drag left = more lenient.",
            style={"color": "var(--color9)", "display": "block", "marginBottom": "0.5rem"},
        ),
        html.Div(
            html.Div(
                [
                    html.Span("Linkage:", style={**_LBL, "whiteSpace": "nowrap", "marginRight": "0.25rem"}),
                    dcc.RadioItems(
                        id="sta-mc-linkage",
                        options=[
                            {"label": html.Span("Complete", id="sta-mc-lnk-complete"), "value": "complete"},
                            {"label": html.Span("Average",  id="sta-mc-lnk-average"),  "value": "average"},
                            {"label": html.Span("Single",   id="sta-mc-lnk-single"),   "value": "single"},
                        ],
                        value="complete",
                        style={"display": "flex", "alignItems": "center", "flexWrap": "nowrap"},
                        inputStyle=_styles.RADIO_INLINE_INPUT,
                        labelStyle=_styles.RADIO_INLINE_LABEL,
                    ),
                    dbc.Tooltip(
                        "Strictest: two groups merge only when all member pairs are close. "
                        "Produces compact, well-separated clusters.",
                        target="sta-mc-lnk-complete", placement="bottom",
                    ),
                    dbc.Tooltip(
                        "Merges based on the average pairwise distance between groups. "
                        "Intermediate tightness between complete and single.",
                        target="sta-mc-lnk-average", placement="bottom",
                    ),
                    dbc.Tooltip(
                        "Most lenient: merges if any single pair is close enough. "
                        "Can chain distant groups through weak links — use with caution.",
                        target="sta-mc-lnk-single", placement="bottom",
                    ),
                ],
                id="sta-mc-linkage-wrapper",
                style=_MC_STYLE_DISABLED,
            ),
            style={"marginBottom": "0.3rem"},
        ),
        _field_num("Minimal class size", "sta-mc-min-group-size", value=10, min=1),
        html.Hr(style={"margin": "0.5rem 0"}),
        dbc.Button(
            "Produce consensus",
            id="sta-mc-produce-btn",
            color="info",
            size="sm",
            disabled=True,
            style={"width": "100%", "marginBottom": "0.3rem"},
        ),
        html.Div(id="sta-mc-produce-status", style={**_HINT, "wordBreak": "break-word"}),
    ])


def _tab_mc() -> html.Div:
    return html.Div(
        [
            html.Div(id="sta-mc-diagnose-area"),
            html.Div(id="sta-mc-preview-area", style={"marginTop": "0.75rem"}),
            html.Div(id="sta-mc-heatmap-area", style={"marginTop": "0.75rem"}),
        ],
        style={"padding": "0.5rem"},
    )


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
                    _classification_setup_panel(),
                    title="Classification setup",
                    item_id="sta-acc-class-setup",
                ),
                dbc.AccordionItem(
                    _run_folder_panel(),
                    title="Create STA folders (STOPGAP only)",
                    item_id="sta-acc-run-folder",
                ),
                dbc.AccordionItem(
                    _mc_panel(),
                    title="Multi-class consensus",
                    item_id="sta-acc-mc",
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
                id="sta-setup-table-hint",
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
                                    get_path_field("sta-setup-save-path", mode="save",
                                                   extensions=(".txt", ".star"),
                                                   placeholder="path/to/params.txt  or  .star"),
                                ],
                                width=8,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Save",
                                    id="sta-setup-save-btn",
                                    color="info", size="sm",
                                    style={"width": "100%", "marginTop": "1.4rem"},
                                ),
                                width=4,
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


def _tab_run_folder_manifest() -> html.Div:
    return html.Div(
        [
            html.Div(
                "Manifest of directories, files and symlinks from the last "
                "\"Create folders\" action.",
                style=_HINT,
            ),
            dag.AgGrid(
                id="sta-rf-manifest-grid",
                rowData=[],
                columnDefs=[
                    {"field": "kind", "headerName": "Kind", "width": 90},
                    {"field": "path", "headerName": "Path", "flex": 1},
                ],
                defaultColDef={"resizable": True, "sortable": True, "filter": True,
                               "flex": 1, "minWidth": 80},
                style={"height": "75vh"},
                className="ag-theme-balham",
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
                dcc.Tab(label="Run folder manifest",
                        value="tab-rf-manifest",
                        children=_tab_run_folder_manifest()),
                dcc.Tab(label="Multi-class consensus",
                        value="tab-mc",
                        children=_tab_mc()),
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
        # Paths created by the classification setup (B panel) to hand to Part C.
        dcc.Store(id="sta-cls-created-paths"),
        dcc.Store(id="sta-ang-ranges-store"),
        dcc.Store(id="sta-bp-ranges-store"),
        dcc.Store(id="sta-mc-factor-store"),
        dcc.Store(id="sta-mc-run-counter", data=0),
        _sep_range_modal("ang", "Angular search — set separately", _ANG_RANGE_COL_DEFS),
        _sep_range_modal("bp", "Bandpass filter — set separately", _BP_RANGE_COL_DEFS),
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
        "high_pass_sigma":   _expand_for_n(
            str(high_pass_sigma) if high_pass_sigma not in (None, "") else None, iter_n),
        "low_pass":          _expand_for_n(low_pass, iter_n),
        "low_pass_sigma":    _expand_for_n(
            str(low_pass_sigma) if low_pass_sigma not in (None, "") else None, iter_n),
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


def _build_display_df(params: "sta_mod.StaParameters") -> pd.DataFrame:
    """Return a display-ready DF with format-specific column names and values.

    For StopgapParams: renames canonical → STOPGAP column names, computes
    angincr/angiter/phi_angincr/phi_angiter from canonical angle columns,
    removes novaSTA-only columns, applies to_format conversions.
    For NovaStaParams: removes STOPGAP-only columns; keeps canonical names.
    """
    is_sg = isinstance(params, sta_mod.StopgapParams)
    df = params.df.copy()

    if is_sg:
        # Remove novaSTA-only columns (stopgap=None in schema)
        nova_only = {
            s.canonical
            for s in sta_mod._STA_SCHEMA
            if s.stopgap is None and s.canonical is not None
        }
        df = df.drop(columns=[c for c in nova_only if c in df.columns], errors="ignore")

        # Compute STOPGAP angle columns from canonical angle columns
        angle_canonical = {"cone angle", "cone sampling", "inplane angle", "inplane sampling"}
        if angle_canonical.issubset(df.columns) or "cone angle" in df.columns:
            ai_list, ac_list, pi_list, pc_list = [], [], [], []
            for _, row in df.iterrows():
                ca = float(row.get("cone angle") or 0)
                cs = float(row.get("cone sampling") or 1) or 1.0
                ia = float(row.get("inplane angle") or 360)
                isp = float(row.get("inplane sampling") or cs)
                ai, ac, pi, pc = sta_mod.nova_to_stopgap_angles(ca, cs, ia, isp)
                ai_list.append(ai); ac_list.append(ac)
                pi_list.append(pi); pc_list.append(pc)
            df = df.drop(columns=[c for c in angle_canonical if c in df.columns], errors="ignore")
            df["angiter"] = ai_list
            df["angincr"] = ac_list
            df["phi_angiter"] = pi_list
            df["phi_angincr"] = pc_list

        # Apply to_format conversions (symmetry, split into even odd, etc.)
        for spec in sta_mod._STA_SCHEMA:
            if spec.to_format is not None and spec.canonical is not None:
                if spec.canonical in df.columns:
                    col = spec.canonical
                    df[col] = df[col].apply(
                        lambda v, s=spec: s.to_format(v, "stopgap") if v is not None else v
                    )

        # Rename canonical → STOPGAP column names
        rename_map = {
            s.canonical: s.stopgap
            for s in sta_mod._STA_SCHEMA
            if s.canonical is not None and s.stopgap is not None
        }
        df = df.rename(columns=rename_map)

    else:
        # novaSTA: remove STOPGAP-only columns (novasta=None in schema)
        sg_only = {
            s.canonical
            for s in sta_mod._STA_SCHEMA
            if s.novasta is None and s.canonical is not None
        }
        df = df.drop(columns=[c for c in sg_only if c in df.columns], errors="ignore")
        # Column names stay canonical (same as novaSTA camelCase keys after mapping)

    return df


def _display_rows_to_params(
    rows: list[dict],
    columns: list[str],
    sta_type: str,
) -> "sta_mod.StaParameters":
    """Reconstruct a StaParameters from display-formatted grid rows.

    Reverses the format-specific column names and value conversions applied
    by :func:`_build_display_df`, then wraps in the appropriate subclass.
    """
    df = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)

    if sta_type == "stopgap":
        # Convert STOPGAP angle columns back to canonical
        angle_sg = {"angiter", "angincr", "phi_angiter", "phi_angincr"}
        if angle_sg.issubset(df.columns):
            ca_list, cs_list, ia_list, isp_list = [], [], [], []
            for _, row in df.iterrows():
                ai = float(row.get("angiter") or 0)
                ac = float(row.get("angincr") or 1) or 1.0
                pi = float(row.get("phi_angiter") or 0)
                pc = float(row.get("phi_angincr") or 1) or 1.0
                ca, cs, ia, isp = sta_mod.stopgap_to_nova_angles(ai, ac, pi, pc)
                ca_list.append(ca); cs_list.append(cs)
                ia_list.append(ia); isp_list.append(isp)
            df = df.drop(columns=[c for c in angle_sg if c in df.columns], errors="ignore")
            df["cone angle"] = ca_list
            df["cone sampling"] = cs_list
            df["inplane angle"] = ia_list
            df["inplane sampling"] = isp_list

        # Apply from_format conversions (symmetry, split into even odd)
        for spec in sta_mod._STA_SCHEMA:
            if spec.from_format is not None and spec.stopgap is not None:
                col_sg = spec.stopgap
                if col_sg in df.columns:
                    df[col_sg] = df[col_sg].apply(
                        lambda v, s=spec: s.from_format(v, "stopgap") if v is not None else v
                    )

        # Rename STOPGAP → canonical column names
        rename_map = {
            s.stopgap: s.canonical
            for s in sta_mod._STA_SCHEMA
            if s.canonical is not None and s.stopgap is not None
        }
        df = df.rename(columns=rename_map)
        return sta_mod.StopgapParams(df)

    else:
        return sta_mod.NovaStaParams(df)


def _build_sta_run_from_form(
    output_base, folder_name, n_runs, subtomo_path, run_mode,
    motl_paths_text, cls_paths, iter_n,
    mask, ccmask, wedgelist, ref_files_text,
):
    """Build a :class:`~cryocat.analysis.sta.StaRun` from run-folder panel form values.

    Returns ``(sta_run, motl_paths, errors)`` where *errors* is a list of
    validation messages (empty on success).  *sta_run* and *motl_paths* are
    ``None`` when there are errors.
    """
    from pathlib import Path as _Path

    errors = []
    if not output_base or not str(output_base).strip():
        errors.append("Output base directory is required.")
    if not folder_name or not str(folder_name).strip():
        errors.append("Folder name is required.")
    if not subtomo_path or not str(subtomo_path).strip():
        errors.append("Subtomogram path is required.")

    # Motl paths: prefer classification output, fall back to text area
    motl_paths = []
    if cls_paths:
        motl_paths = [_Path(p) for p in cls_paths if p]
    elif motl_paths_text and str(motl_paths_text).strip():
        motl_paths = [_Path(p.strip()) for p in str(motl_paths_text).split(",") if p.strip()]

    # Reference files
    references = []
    if ref_files_text and str(ref_files_text).strip():
        references = [_Path(p.strip()) for p in str(ref_files_text).split(",") if p.strip()]

    # Base params dict (canonical names)
    base_params: dict = {}
    if mask and str(mask).strip():
        base_params["mask name"] = str(mask).strip()
    if ccmask and str(ccmask).strip():
        base_params["cc mask name"] = str(ccmask).strip()
    if wedgelist and str(wedgelist).strip():
        base_params["wedge list"] = str(wedgelist).strip()

    if errors:
        return None, None, errors

    blocks = [sta_mod.Block(int(iter_n or 10), "ali", "{base}")]
    sta_run = sta_mod.StaRun(
        input_motl_id="",
        run_mode=run_mode or "singleref",
        n_runs=int(n_runs or 1),
        output_base=_Path(str(output_base).strip()),
        folder_name=str(folder_name).strip(),
        subtomo_path=_Path(str(subtomo_path).strip()),
        base_params=base_params,
        schedule=blocks,
        references=references,
    )
    return sta_run, motl_paths, []


# ── Callbacks ────────────────────────────────────────────────────────────────


def _validate_sep_ranges(rows: list[dict], n_total: int) -> list[str]:
    """Validate that the range-table rows form a contiguous cover of 1..n_total."""
    errors: list[str] = []
    if not rows:
        errors.append("At least one range row is required.")
        return errors
    rows_s = sorted(rows, key=lambda r: int(r.get("from_iter") or 0))
    if int(rows_s[0].get("from_iter") or 0) != 1:
        errors.append(f"First range must start at 1 (got {rows_s[0].get('from_iter')}).")
    if int(rows_s[-1].get("to_iter") or 0) != n_total:
        errors.append(
            f"Last range must end at {n_total} (got {rows_s[-1].get('to_iter')})."
        )
    for a, b in zip(rows_s, rows_s[1:]):
        a_to = int(a.get("to_iter") or 0)
        b_from = int(b.get("from_iter") or 0)
        if b_from != a_to + 1:
            if b_from <= a_to:
                errors.append(
                    f"Overlap between range ending at {a_to} "
                    f"and range starting at {b_from}."
                )
            else:
                errors.append(
                    f"Gap between range ending at {a_to} "
                    f"and range starting at {b_from}."
                )
    return errors


def _expand_sep_ranges(
    ranges: list[dict], fields: list[str], iter_n: int, start_idx: int,
) -> dict[str, str | None]:
    """Expand range-table rows into space-separated per-iteration value strings."""
    per_field: dict[str, list[str]] = {f: [] for f in fields}
    for i in range(start_idx, start_idx + iter_n):
        matched = None
        for rng in ranges or []:
            fr = rng.get("from_iter") or 0
            to = rng.get("to_iter") or 0
            if fr <= i <= to:
                matched = rng
                break
        for f in fields:
            per_field[f].append(str((matched or {}).get(f) or "") if matched else "")
    return {
        f: " ".join(vals) if any(v for v in vals) else None
        for f, vals in per_field.items()
    }


def _pool_paths_to_sta_config(source_paths: list[str | None], motl_type: str) -> dict | str:
    """Derive STA loader config from pool-member source paths.

    Returns a config dict on success, or an error string on failure.
    The paths must follow the ``{base}{iteration}{ext}`` naming convention
    and the iteration numbers must form a contiguous range.
    """
    import re
    import os
    from pathlib import Path

    if not source_paths:
        return "Selected pool entries have no source paths — load motls from files first."

    none_paths = [p for p in source_paths if not p]
    if none_paths:
        return (
            f"{len(none_paths)} pool member(s) have no source path. "
            "Ensure motls were loaded from files, not created in-memory."
        )

    dirs  = [str(Path(p).parent) for p in source_paths]
    stems = [Path(p).stem for p in source_paths]

    if len(set(dirs)) > 1:
        return (
            "Pool motls are in different directories — "
            "use 'From motls directly' with a common base path."
        )

    num_re  = re.compile(r"^(.*?)(\d+)$")
    matches = [num_re.match(s) for s in stems]
    if not all(matches):
        return (
            "Pool motl filenames don't end with iteration numbers — "
            "use 'From motls directly' to specify the base path manually."
        )

    prefixes = [m.group(1) for m in matches]
    if len(set(prefixes)) > 1:
        return (
            "Pool motl filenames don't share a common prefix — "
            "use 'From motls directly' to specify the base path manually."
        )

    motl_base = os.path.join(dirs[0], prefixes[0])
    numbers   = sorted(int(m.group(2)) for m in matches)
    start_it, end_it = numbers[0], numbers[-1]

    expected = list(range(start_it, end_it + 1))
    if numbers != expected:
        return (
            f"Iteration numbers have gaps (got {numbers}, expected {expected}). "
            "Use 'From motls directly' and specify the iteration range manually."
        )

    return {
        "motl_base": motl_base,
        "motl_type": motl_type,
        "start_it":  start_it,
        "end_it":    end_it,
        "source":    "pool",
    }


def register_callbacks(app):

    # Slim angles builder: params collect comes for free; we replace the
    # auto-preview with a manual "Visualize" button so the user controls
    # when the preview re-renders.  The hidden create-file button never
    # fires because it's invisible.
    register_angles_builder_callbacks(app, _ANGLES_PREFIX, with_graphs=False)
    register_pool_picker_callbacks(app, "sta-pool")
    register_pool_picker_callbacks(app, "sta-mc-pool")

    # ── Toggle which loader form is visible ──────────────────────────────────
    @app.callback(
        Output("sta-eval-params-wrapper", "style"),
        Output("sta-pool-wrapper", "style"),
        Input("sta-eval-mode", "value"),
    )
    def _toggle_loader(mode):
        show, hide = {"display": "block"}, {"display": "none"}
        if mode == "pool":
            return hide, show
        return show, hide

    # ── Load from parameter file → populate loader-config + params-store ─────
    @app.callback(
        Output("sta-loader-config", "data", allow_duplicate=True),
        Output("sta-params-store", "data", allow_duplicate=True),
        Output("sta-param-status", "children"),
        Input("sta-param-load-btn", "n_clicks"),
        State({"type": "path-input", "owner": "sta-param-path"}, "value"),
        State("sta-param-type", "value"),
        State("sta-param-sep", "value"),
        State({"type": "path-input", "owner": "sta-param-working-dir"}, "value"),
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
            params = run_operation(sta_mod.StaParameters.load, {
                "path": path.strip(),
                "sta_type": sta_type_arg,
            })
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

    # ── Load from pool → populate loader-config ──────────────────────────────
    @app.callback(
        Output("sta-loader-config", "data", allow_duplicate=True),
        Output("sta-params-store", "data", allow_duplicate=True),
        Output("sta-pool-status", "children"),
        Input("sta-pool-load-btn", "n_clicks"),
        State("sta-pool-value", "data"),
        State(ids.POOL_REGISTRY, "data"),
        prevent_initial_call=True,
    )
    def _load_from_pool(n_clicks, selected_ids, registry):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not selected_ids:
            return no_update, no_update, "Select motls or a group first."

        registry = registry or {}
        motl_type = next(
            ((registry.get(mid) or {}).get("type") for mid in selected_ids
             if (registry.get(mid) or {}).get("type")),
            "emmotl",
        )
        source_paths = [(registry.get(mid) or {}).get("source_path") for mid in selected_ids]

        result = _pool_paths_to_sta_config(source_paths, motl_type)
        if isinstance(result, str):
            return no_update, no_update, result

        return (
            result, None,
            f"Pool config: {result['motl_type']} · "
            f"iterations {result['start_it']}–{result['end_it']}. "
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
            stats_df = run_operation(sta_mod.compute_alignment_statistics, {
                "motl_base_name": config["motl_base"],
                "start_it": config["start_it"],
                "end_it": config["end_it"],
                "motl_type": config["motl_type"],
            })
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
            occupancy, changes = run_operation(sta_mod.evaluate_classification, {
                "motl_base_name": config["motl_base"],
                "start_it": config["start_it"],
                "end_it": config["end_it"],
                "motl_type": config["motl_type"],
                "plot_results": False,
            })
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
        State("sta-ang-ranges-store", "data"),
        State("sta-bp-ranges-store", "data"),
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
        iter_n_val, start_index,
        ang_ranges, bp_ranges,
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
            run_mode = ref_family or "singleref"
            iter_n = max(int(iter_n_val or 10), 1)
            starting = int(start_index or 1)

            # Expand angular ranges if active
            if ang_ranges:
                ang_exp = _expand_sep_ranges(
                    ang_ranges,
                    ["cone_angle", "cone_sampling", "inplane_angle", "inplane_sampling"],
                    iter_n, starting,
                )
                cone_angle = ang_exp["cone_angle"]
                cone_sampling = ang_exp["cone_sampling"]
                inplane_angle = ang_exp["inplane_angle"]
                inplane_sampling = ang_exp["inplane_sampling"]

            # Expand bandpass ranges if active
            if bp_ranges:
                bp_exp = _expand_sep_ranges(
                    bp_ranges,
                    ["high_pass", "high_pass_sigma", "low_pass", "low_pass_sigma"],
                    iter_n, starting,
                )
                high_pass = bp_exp["high_pass"]
                high_pass_sigma = bp_exp["high_pass_sigma"]
                low_pass = bp_exp["low_pass"]
                low_pass_sigma = bp_exp["low_pass_sigma"]

            params_dict = _setup_form_to_params_dict(
                iter_n=iter_n,
                start_index=starting,
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
                create_ref=bool(create_ref), ref_family=run_mode,
                apply_laplacian=bool(apply_laplacian),
                calc_exp=bool(calc_exp) if calc_exp is not None else True,
                calc_ctf=bool(calc_ctf) if calc_ctf is not None else True,
                cos_weight=cos_weight, score_weight=score_weight,
                search_mode=search_mode or "hc",
                cone_search_type=cone_search_type,
                scoring_fcn=scoring_fcn, rot_mode=rot_mode,
                avg_mode=avg_mode, subset=subset, fthresh=fthresh,
                temperature=temperature,
                use_euler_search=bool(use_euler_search),
                euler_axes=euler_axes,
                euler_1_incr=euler_1_incr, euler_1_iter=euler_1_iter,
                euler_2_incr=euler_2_incr, euler_2_iter=euler_2_iter,
                euler_3_incr=euler_3_incr, euler_3_iter=euler_3_iter,
            )
            if resolved_sta_type == "stopgap":
                params_dict["subtomo_mode"] = sta_mod.compose_subtomo_mode("ali", run_mode)

            params = sta_mod.StaParameters.from_dict(params_dict, sta_type=resolved_sta_type)
        except Exception as exc:
            return no_update, f"Create failed: {exc}", no_update

        if params.df.empty:
            return (no_update,
                    "Create produced an empty dataframe — check the form values.",
                    no_update)

        display_df = _build_display_df(params)
        data = {
            "records": display_df.to_dict("records"),
            "columns": list(display_df.columns),
            "sta_type": resolved_sta_type,
        }
        return (
            data,
            f"Created {resolved_sta_type.upper()} parameter dataframe "
            f"({len(params.df)} iteration(s), {len(display_df.columns)} column(s)). "
            f'Edit cells in the "STA setup output" tab, then save.',
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
        Output("sta-setup-table-hint", "children"),
        Input("sta-setup-df-store", "data"),
    )
    def populate_setup_grid(data):
        if not data or not data.get("columns"):
            return [], [], "Edit cells inline (single-click). Provide an output path and save."
        col_defs = [
            {"field": c, "headerName": c, "editable": True,
             "flex": 1, "minWidth": 100}
            for c in data["columns"]
        ]
        sta_type = data.get("sta_type", "novasta")
        if sta_type == "stopgap":
            hint = "The angles were recomputed to follow STOPGAP conventions."
        else:
            hint = "Edit cells inline (single-click). Provide an output path and save."
        return data["records"], col_defs, hint

    # ── Save setup dataframe to disk ─────────────────────────────────────────
    @app.callback(
        Output("sta-setup-save-status", "children"),
        Input("sta-setup-save-btn", "n_clicks"),
        State("sta-setup-grid", "rowData"),
        State("sta-setup-df-store", "data"),
        State({"type": "path-input", "owner": "sta-setup-save-path"}, "value"),
        prevent_initial_call=True,
    )
    def save_setup(n_save, rows, store, path):
        if not n_save:
            raise dash.exceptions.PreventUpdate
        if not path or not str(path).strip():
            return "Provide an output path."
        if not rows or not store or not store.get("columns"):
            return "Create the parameter dataframe first."

        target = (store or {}).get("sta_type", "novasta")
        columns = store["columns"]
        try:
            params = _display_rows_to_params(rows, columns, target)
            run_operation(params.write_out, {"path": str(path).strip()})
        except Exception as exc:
            return f"Save as {target} failed: {exc}"
        return f"Saved as {target} → {str(path).strip()}"

    # ── Continue-run: prefill form fields from last param row ────────────────
    @app.callback(
        Output("sta-setup-start-index", "value", allow_duplicate=True),
        Output("sta-setup-continue-status", "children"),
        Output("sta-setup-motl", "value", allow_duplicate=True),
        Output("sta-setup-ref", "value", allow_duplicate=True),
        Output("sta-setup-mask", "value", allow_duplicate=True),
        Output("sta-setup-cc-mask", "value", allow_duplicate=True),
        Output("sta-setup-wedge-list", "value", allow_duplicate=True),
        Output("sta-setup-subtomo-path", "value", allow_duplicate=True),
        Output("sta-setup-cone-angle", "value", allow_duplicate=True),
        Output("sta-setup-cone-sampling", "value", allow_duplicate=True),
        Output("sta-setup-inplane-angle", "value", allow_duplicate=True),
        Output("sta-setup-inplane-sampling", "value", allow_duplicate=True),
        Output("sta-setup-high-pass", "value", allow_duplicate=True),
        Output("sta-setup-high-pass-sigma", "value", allow_duplicate=True),
        Output("sta-setup-low-pass", "value", allow_duplicate=True),
        Output("sta-setup-low-pass-sigma", "value", allow_duplicate=True),
        Output("sta-setup-symmetry", "value", allow_duplicate=True),
        Output("sta-setup-rootdir", "value", allow_duplicate=True),
        Output("sta-setup-binning", "value", allow_duplicate=True),
        Output("sta-setup-iter", "value", allow_duplicate=True),
        Input("sta-setup-continue-run", "value"),
        State("sta-params-store", "data"),
        prevent_initial_call=True,
    )
    def _continue_run_prefill(continue_flag, params_data):
        _none20 = (no_update,) * 20
        if not continue_flag:
            return (no_update, "") + _none20[2:]
        if not params_data or not params_data.get("records"):
            return (
                no_update,
                "No parameter file loaded — use the Evaluation panel to load one first.",
            ) + _none20[2:]
        try:
            last = params_data["records"][-1]
            prefill = sta_mod.continue_run_prefill(last)
            si = prefill["starting_iter"]
            bp = prefill["base_params"]

            warn = ""
            raw_temp = last.get("temperature", 0)
            try:
                if float(raw_temp or 0) != 0.0:
                    warn = (
                        " Warning: the loaded run used a non-zero temperature. "
                        "Continuing with simulated annealing is incorrect — "
                        "annealing belongs only to the opening iterations of a de-novo run."
                    )
            except (TypeError, ValueError):
                pass

            def _s(v):
                return str(v).strip() if v not in (None, "") else None

            return (
                si,
                f"Prefilled from last row (iteration {last.get('iteration', '?')}). "
                f"Set 'Number of iterations' before creating.{warn}",
                _s(bp.get("motl")),
                _s(bp.get("ref")),
                _s(bp.get("mask name") or bp.get("mask")),
                _s(bp.get("cc mask name") or bp.get("cc_mask")),
                _s(bp.get("wedge list") or bp.get("wedge_list")),
                _s(bp.get("subtomo name") or bp.get("subtomo_name")),
                _s(bp.get("cone angle") or bp.get("cone_angle")),
                _s(bp.get("cone sampling") or bp.get("cone_sampling")),
                _s(bp.get("inplane angle") or bp.get("inplane_angle")),
                _s(bp.get("inplane sampling") or bp.get("inplane_sampling")),
                _s(bp.get("high pass") or bp.get("high_pass")),
                bp.get("high pass sigma") or bp.get("high_pass_sigma"),
                _s(bp.get("low pass") or bp.get("low_pass")),
                bp.get("low pass sigma") or bp.get("low_pass_sigma"),
                bp.get("symmetry"),
                _s(bp.get("rootdir")),
                bp.get("binning"),
                None,  # sta-setup-iter: leave empty, user must set
            )
        except Exception as exc:
            return (no_update, f"Prefill error: {exc}") + _none20[2:]

    # ── Classification setup: show/hide denovo vs exrefs section ─────────────
    @app.callback(
        Output("sta-cls-denovo-section", "style"),
        Output("sta-cls-exrefs-section", "style"),
        Input("sta-cls-workflow", "value"),
    )
    def _toggle_cls_sections(workflow):
        if (workflow or "denovo") == "exrefs":
            return _HIDE, _SHOW
        return _SHOW, _HIDE

    # ── B1: create de-novo reference motls ───────────────────────────────────
    @app.callback(
        Output("sta-cls-dn-status", "children"),
        Output("sta-cls-created-paths", "data"),
        Input("sta-cls-dn-create-btn", "n_clicks"),
        State({"type": "path-input", "owner": "sta-cls-dn-input-motl"}, "value"),
        State("sta-cls-dn-input-type", "value"),
        State("sta-cls-dn-n-classes", "value"),
        State("sta-cls-dn-occupancy", "value"),
        State("sta-cls-dn-n-runs", "value"),
        State("sta-cls-dn-iter", "value"),
        State("sta-cls-dn-output-type", "value"),
        State({"type": "path-input", "owner": "sta-cls-dn-output-base"}, "value"),
        prevent_initial_call=True,
    )
    def _create_denovo_motls(
        n_clicks,
        input_motl, input_type, n_classes, occupancy, n_runs, iter_num,
        output_type, output_base,
    ):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        errors = []
        if not input_motl or not str(input_motl).strip():
            errors.append("Input motl path is required.")
        if not output_base or not str(output_base).strip():
            errors.append("Output motl base path is required.")
        if errors:
            return " | ".join(errors), no_update
        try:
            kwargs = {
                "input_motl": str(input_motl).strip(),
                "number_of_classes": int(n_classes or 6),
                "output_motl_base": str(output_base).strip(),
                "input_motl_type": input_type or "emmotl",
                "iteration_number": int(iter_num or 1),
                "number_of_runs": int(n_runs or 1),
                "output_motl_type": output_type or "stopgap",
            }
            if occupancy not in (None, ""):
                kwargs["class_occupancy"] = int(occupancy)
            paths = run_operation(sta_mod.create_denovo_multiref_run, kwargs)
            path_strs = [str(p) for p in (paths or [])]
            return (
                f"Created {len(path_strs)} motl file(s): {', '.join(path_strs)}",
                path_strs,
            )
        except Exception as exc:
            return f"De-novo create failed: {exc}", no_update

    # ── B2: assign random classes (existing references) ───────────────────────
    @app.callback(
        Output("sta-cls-er-status", "children"),
        Output("sta-cls-created-paths", "data", allow_duplicate=True),
        Input("sta-cls-er-create-btn", "n_clicks"),
        State({"type": "path-input", "owner": "sta-cls-er-input-motl"}, "value"),
        State("sta-cls-er-input-type", "value"),
        State("sta-cls-er-n-classes", "value"),
        State("sta-cls-er-iter", "value"),
        State("sta-cls-er-output-type", "value"),
        State({"type": "path-input", "owner": "sta-cls-er-output-base"}, "value"),
        prevent_initial_call=True,
    )
    def _create_exrefs_motl(
        n_clicks,
        input_motl, input_type, n_classes, iter_num, output_type, output_base,
    ):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        errors = []
        if not input_motl or not str(input_motl).strip():
            errors.append("Input motl path is required.")
        if not output_base or not str(output_base).strip():
            errors.append("Output motl base path is required.")
        if errors:
            return " | ".join(errors), no_update
        try:
            kwargs = {
                "input_motl": str(input_motl).strip(),
                "number_of_classes": int(n_classes or 6),
                "output_motl_base": str(output_base).strip(),
                "input_motl_type": input_type or "emmotl",
                "iteration_number": int(iter_num or 1),
                "number_of_runs": 1,
                "output_motl_type": output_type or "stopgap",
            }
            paths = run_operation(sta_mod.create_multiref_run, kwargs)
            path_strs = [str(p) for p in (paths or [])]
            return (
                f"Created {len(path_strs)} motl file(s): {', '.join(path_strs)}",
                path_strs,
            )
        except Exception as exc:
            return f"Existing-refs create failed: {exc}", no_update

    # ── C: run folder preflight ───────────────────────────────────────────────
    @app.callback(
        Output("sta-rf-status", "children"),
        Input("sta-rf-preflight-btn", "n_clicks"),
        State({"type": "path-input", "owner": "sta-rf-output-base"}, "value"),
        State("sta-rf-folder-name", "value"),
        State("sta-rf-multirun", "value"),
        State("sta-cls-dn-n-runs", "value"),
        State({"type": "path-input", "owner": "sta-rf-subtomo-path"}, "value"),
        State("sta-rf-motl-paths", "value"),
        State("sta-cls-created-paths", "data"),
        State("sta-setup-iter", "value"),
        State("sta-setup-start-index", "value"),
        State({"type": "path-input", "owner": "sta-rf-mask"}, "value"),
        State({"type": "path-input", "owner": "sta-rf-ccmask"}, "value"),
        State({"type": "path-input", "owner": "sta-rf-wedgelist"}, "value"),
        State("sta-rf-ref-files", "value"),
        prevent_initial_call=True,
    )
    def _run_folder_preflight(
        n_clicks,
        output_base, folder_name, multirun, cls_n_runs, subtomo_path,
        motl_paths_text, cls_paths, iter_n, start_index,
        mask, ccmask, wedgelist, ref_files_text,
    ):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        n_runs = int(cls_n_runs or 1) if multirun else 1
        run_mode = "multiref" if multirun else "singleref"
        try:
            sta_run, motl_paths, errors = _build_sta_run_from_form(
                output_base, folder_name, n_runs, subtomo_path, run_mode,
                motl_paths_text, cls_paths, iter_n,
                mask, ccmask, wedgelist, ref_files_text,
            )
            if errors:
                return "Form errors:\n" + "\n".join(f"• {e}" for e in errors)
            problems = sta_mod.preflight_run_folder(
                sta_run, motl_paths, starting_iter=int(start_index or 1)
            )
            if problems:
                return "Preflight issues:\n" + "\n".join(f"• {p}" for p in problems)
            return f"Preflight OK — {len(sta_run.schedule)} block(s), {sta_run.n_runs} run(s)."
        except Exception as exc:
            return f"Preflight error: {exc}"

    # ── C: run folder creation ────────────────────────────────────────────────
    @app.callback(
        Output("sta-rf-status", "children", allow_duplicate=True),
        Output("sta-rf-manifest-grid", "rowData"),
        Input("sta-rf-create-btn", "n_clicks"),
        State({"type": "path-input", "owner": "sta-rf-output-base"}, "value"),
        State("sta-rf-folder-name", "value"),
        State("sta-rf-multirun", "value"),
        State("sta-cls-dn-n-runs", "value"),
        State({"type": "path-input", "owner": "sta-rf-subtomo-path"}, "value"),
        State("sta-rf-motl-paths", "value"),
        State("sta-cls-created-paths", "data"),
        State("sta-setup-iter", "value"),
        State("sta-setup-start-index", "value"),
        State({"type": "path-input", "owner": "sta-rf-mask"}, "value"),
        State({"type": "path-input", "owner": "sta-rf-ccmask"}, "value"),
        State({"type": "path-input", "owner": "sta-rf-wedgelist"}, "value"),
        State("sta-rf-ref-files", "value"),
        State("sta-rf-overwrite", "value"),
        prevent_initial_call=True,
    )
    def _create_run_folders(
        n_clicks,
        output_base, folder_name, multirun, cls_n_runs, subtomo_path,
        motl_paths_text, cls_paths, iter_n, start_index,
        mask, ccmask, wedgelist, ref_files_text, overwrite,
    ):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        n_runs = int(cls_n_runs or 1) if multirun else 1
        run_mode = "multiref" if multirun else "singleref"
        try:
            sta_run, motl_paths, errors = _build_sta_run_from_form(
                output_base, folder_name, n_runs, subtomo_path, run_mode,
                motl_paths_text, cls_paths, iter_n,
                mask, ccmask, wedgelist, ref_files_text,
            )
            if errors:
                return "Form errors:\n" + "\n".join(f"• {e}" for e in errors), no_update
            manifest = run_operation(
                sta_mod.create_run_folder,
                {
                    "sta_run": sta_run,
                    "motl_paths": motl_paths,
                    "starting_iter": int(start_index or 1),
                    "overwrite": bool(overwrite),
                },
            )
            manifest = manifest or {}
            grid_rows = []
            for action, paths in manifest.items():
                for item in (paths if isinstance(paths, list) else [paths]):
                    grid_rows.append({"action": action, "path": str(item)})
            n_dirs = len(manifest.get("dirs_created", []))
            n_files = (len(manifest.get("files_copied", [])) +
                       len(manifest.get("symlinks_created", [])))
            return (
                f"Created {n_dirs} director(ies) and {n_files} file(s)/link(s).",
                grid_rows,
            )
        except Exception as exc:
            return f"Create failed: {exc}", no_update

    # ── R2: Angular search "Set separately" modal ────────────────────────────

    @app.callback(
        Output("sta-ang-sep-modal", "is_open"),
        Output("sta-ang-sep-grid", "rowData", allow_duplicate=True),
        Input("sta-ang-sep-btn", "n_clicks"),
        Input("sta-ang-sep-cancel-btn", "n_clicks"),
        State("sta-ang-sep-modal", "is_open"),
        State("sta-setup-iter", "value"),
        State("sta-setup-cone-angle", "value"),
        State("sta-setup-cone-sampling", "value"),
        State("sta-setup-inplane-angle", "value"),
        State("sta-setup-inplane-sampling", "value"),
        State("sta-ang-ranges-store", "data"),
        prevent_initial_call=True,
    )
    def _toggle_ang_sep_modal(
        n_open, n_cancel, is_open,
        iter_n, cone_angle, cone_sampling, inplane_angle, inplane_sampling,
        stored_ranges,
    ):
        tid = ctx.triggered_id
        if tid == "sta-ang-sep-btn":
            if stored_ranges:
                return True, stored_ranges
            n = max(int(iter_n or 10), 1)
            default_row = {
                "from_iter": 1, "to_iter": n,
                "cone_angle": cone_angle or "",
                "cone_sampling": cone_sampling or "",
                "inplane_angle": inplane_angle or "",
                "inplane_sampling": inplane_sampling or "",
            }
            return True, [default_row]
        return False, no_update

    @app.callback(
        Output("sta-ang-sep-grid", "rowData", allow_duplicate=True),
        Input("sta-ang-sep-split-btn", "n_clicks"),
        Input("sta-ang-sep-remove-btn", "n_clicks"),
        Input("sta-ang-sep-reset-btn", "n_clicks"),
        State("sta-ang-sep-grid", "rowData"),
        State("sta-ang-sep-grid", "selectedRows"),
        State("sta-setup-iter", "value"),
        State("sta-setup-cone-angle", "value"),
        State("sta-setup-cone-sampling", "value"),
        State("sta-setup-inplane-angle", "value"),
        State("sta-setup-inplane-sampling", "value"),
        prevent_initial_call=True,
    )
    def _manage_ang_sep_rows(
        n_split, n_remove, n_reset, rows, selected,
        iter_n, cone_angle, cone_sampling, inplane_angle, inplane_sampling,
    ):
        tid = ctx.triggered_id
        rows = list(rows or [])
        if tid == "sta-ang-sep-reset-btn":
            n = max(int(iter_n or 10), 1)
            return [{
                "from_iter": 1, "to_iter": n,
                "cone_angle": cone_angle or "",
                "cone_sampling": cone_sampling or "",
                "inplane_angle": inplane_angle or "",
                "inplane_sampling": inplane_sampling or "",
            }]
        if tid == "sta-ang-sep-remove-btn":
            if selected and rows:
                sel = selected[0]
                for i, r in enumerate(rows):
                    if r == sel:
                        rows.pop(i)
                        break
            return rows
        if tid == "sta-ang-sep-split-btn":
            src = (selected or [None])[0] or (rows[-1] if rows else {})
            fr = int(src.get("from_iter") or 1)
            to = int(src.get("to_iter") or 1)
            if to > fr:
                mid = fr + (to - fr) // 2
                idx = next((i for i, r in enumerate(rows) if r == src), len(rows) - 1)
                rows[idx] = {**src, "to_iter": mid}
                new_row = {**src, "from_iter": mid + 1}
                rows.insert(idx + 1, new_row)
        return rows

    @app.callback(
        Output("sta-ang-ranges-store", "data", allow_duplicate=True),
        Output("sta-ang-sep-modal", "is_open", allow_duplicate=True),
        Output("sta-ang-sep-btn", "children"),
        Output("sta-ang-sep-display", "children"),
        Output("sta-ang-sep-display", "style"),
        Output("sta-ang-sep-error", "children"),
        Input("sta-ang-sep-accept-btn", "n_clicks"),
        Input("sta-ang-sep-reset-btn", "n_clicks"),
        State("sta-ang-sep-grid", "rowData"),
        State("sta-setup-iter", "value"),
        prevent_initial_call=True,
    )
    def _accept_ang_sep(n_accept, n_reset, rows, iter_n):
        tid = ctx.triggered_id
        if tid == "sta-ang-sep-reset-btn":
            return (
                None, False,
                "Set separately…", "",
                {"display": "none"},
                "",
            )
        if not n_accept:
            raise dash.exceptions.PreventUpdate
        n = max(int(iter_n or 10), 1)
        errors = _validate_sep_ranges(rows or [], n)
        if errors:
            return (
                no_update, no_update,
                no_update, no_update, no_update,
                "\n".join(errors),
            )
        n_ranges = len(rows or [])
        return (
            rows, False,
            "Edit separately…",
            f"varies ({n_ranges} range{'s' if n_ranges != 1 else ''})",
            {**_HINT, "fontSize": _styles.FONT_TIGHT},
            "",
        )

    # ── R2: Bandpass "Set separately" modal ─────────────────────────────────

    @app.callback(
        Output("sta-bp-sep-modal", "is_open"),
        Output("sta-bp-sep-grid", "rowData", allow_duplicate=True),
        Input("sta-bp-sep-btn", "n_clicks"),
        Input("sta-bp-sep-cancel-btn", "n_clicks"),
        State("sta-bp-sep-modal", "is_open"),
        State("sta-setup-iter", "value"),
        State("sta-setup-high-pass", "value"),
        State("sta-setup-high-pass-sigma", "value"),
        State("sta-setup-low-pass", "value"),
        State("sta-setup-low-pass-sigma", "value"),
        State("sta-bp-ranges-store", "data"),
        prevent_initial_call=True,
    )
    def _toggle_bp_sep_modal(
        n_open, n_cancel, is_open,
        iter_n, high_pass, high_pass_sigma, low_pass, low_pass_sigma,
        stored_ranges,
    ):
        tid = ctx.triggered_id
        if tid == "sta-bp-sep-btn":
            if stored_ranges:
                return True, stored_ranges
            n = max(int(iter_n or 10), 1)
            default_row = {
                "from_iter": 1, "to_iter": n,
                "high_pass": high_pass or "",
                "high_pass_sigma": high_pass_sigma if high_pass_sigma not in (None, "") else "",
                "low_pass": low_pass or "",
                "low_pass_sigma": low_pass_sigma if low_pass_sigma not in (None, "") else "",
            }
            return True, [default_row]
        return False, no_update

    @app.callback(
        Output("sta-bp-sep-grid", "rowData", allow_duplicate=True),
        Input("sta-bp-sep-split-btn", "n_clicks"),
        Input("sta-bp-sep-remove-btn", "n_clicks"),
        Input("sta-bp-sep-reset-btn", "n_clicks"),
        State("sta-bp-sep-grid", "rowData"),
        State("sta-bp-sep-grid", "selectedRows"),
        State("sta-setup-iter", "value"),
        State("sta-setup-high-pass", "value"),
        State("sta-setup-high-pass-sigma", "value"),
        State("sta-setup-low-pass", "value"),
        State("sta-setup-low-pass-sigma", "value"),
        prevent_initial_call=True,
    )
    def _manage_bp_sep_rows(
        n_split, n_remove, n_reset, rows, selected,
        iter_n, high_pass, high_pass_sigma, low_pass, low_pass_sigma,
    ):
        tid = ctx.triggered_id
        rows = list(rows or [])
        if tid == "sta-bp-sep-reset-btn":
            n = max(int(iter_n or 10), 1)
            return [{
                "from_iter": 1, "to_iter": n,
                "high_pass": high_pass or "",
                "high_pass_sigma": high_pass_sigma if high_pass_sigma not in (None, "") else "",
                "low_pass": low_pass or "",
                "low_pass_sigma": low_pass_sigma if low_pass_sigma not in (None, "") else "",
            }]
        if tid == "sta-bp-sep-remove-btn":
            if selected and rows:
                sel = selected[0]
                for i, r in enumerate(rows):
                    if r == sel:
                        rows.pop(i)
                        break
            return rows
        if tid == "sta-bp-sep-split-btn":
            src = (selected or [None])[0] or (rows[-1] if rows else {})
            fr = int(src.get("from_iter") or 1)
            to = int(src.get("to_iter") or 1)
            if to > fr:
                mid = fr + (to - fr) // 2
                idx = next((i for i, r in enumerate(rows) if r == src), len(rows) - 1)
                rows[idx] = {**src, "to_iter": mid}
                rows.insert(idx + 1, {**src, "from_iter": mid + 1})
        return rows

    @app.callback(
        Output("sta-bp-ranges-store", "data", allow_duplicate=True),
        Output("sta-bp-sep-modal", "is_open", allow_duplicate=True),
        Output("sta-bp-sep-btn", "children"),
        Output("sta-bp-sep-display", "children"),
        Output("sta-bp-sep-display", "style"),
        Output("sta-bp-sep-error", "children"),
        Input("sta-bp-sep-accept-btn", "n_clicks"),
        Input("sta-bp-sep-reset-btn", "n_clicks"),
        State("sta-bp-sep-grid", "rowData"),
        State("sta-setup-iter", "value"),
        prevent_initial_call=True,
    )
    def _accept_bp_sep(n_accept, n_reset, rows, iter_n):
        tid = ctx.triggered_id
        if tid == "sta-bp-sep-reset-btn":
            return (
                None, False,
                "Set separately…", "",
                {"display": "none"},
                "",
            )
        if not n_accept:
            raise dash.exceptions.PreventUpdate
        n = max(int(iter_n or 10), 1)
        errors = _validate_sep_ranges(rows or [], n)
        if errors:
            return (
                no_update, no_update,
                no_update, no_update, no_update,
                "\n".join(errors),
            )
        n_ranges = len(rows or [])
        return (
            rows, False,
            "Edit separately…",
            f"varies ({n_ranges} range{'s' if n_ranges != 1 else ''})",
            {**_HINT, "fontSize": _styles.FONT_TIGHT},
            "",
        )

    # ── R4: Generate parameter table from classification templates ────────────

    @app.callback(
        Output("sta-setup-df-store", "data", allow_duplicate=True),
        Output("sta-cls-gen-status", "children"),
        Output("sta-main-tabs", "value", allow_duplicate=True),
        Input("sta-cls-gen-denovo-btn", "n_clicks"),
        Input("sta-cls-gen-exrefs-btn", "n_clicks"),
        State("sta-setup-sta-type", "value"),
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
        State("sta-setup-rootdir", "value"),
        State("sta-setup-binning", "value"),
        State("sta-setup-apply-laplacian", "value"),
        State("sta-setup-calc-exp", "value"),
        State("sta-setup-calc-ctf", "value"),
        State("sta-setup-cos-weight", "value"),
        State("sta-setup-score-weight", "value"),
        State("sta-setup-cone-search-type", "value"),
        State("sta-setup-scoring-fcn", "value"),
        State("sta-setup-rot-mode", "value"),
        State("sta-setup-avg-mode", "value"),
        State("sta-setup-subset", "value"),
        State("sta-setup-fthresh", "value"),
        prevent_initial_call=True,
    )
    def _generate_cls_template(
        n_denovo, n_exrefs,
        sta_type, start_index,
        cone_angle, cone_sampling, inplane_angle, inplane_sampling,
        high_pass, high_pass_sigma, low_pass, low_pass_sigma, threshold,
        motl, ref, mask, cc_mask, wedge_list, subtomo_path,
        symmetry, class_id, split_even_odd,
        rootdir, binning,
        apply_laplacian, calc_exp, calc_ctf,
        cos_weight, score_weight,
        cone_search_type, scoring_fcn, rot_mode, avg_mode, subset, fthresh,
    ):
        tid = ctx.triggered_id
        if tid is None:
            raise dash.exceptions.PreventUpdate
        if (sta_type or "novasta").lower() != "stopgap":
            return no_update, "Templates are STOPGAP only.", no_update

        is_denovo = (tid == "sta-cls-gen-denovo-btn")
        # Template block layout per R4 spec:
        # De-novo: 1 avg (motl={base}_ref_mr{run}) + 10 ali shc temp=10 + 20 ali shc temp=0
        # Existing: 1 ali hc temp=0 + 29 ali shc temp=0
        if is_denovo:
            template_blocks = [
                (1,  "avg", "{base}_ref_mr{run}", None,  "0"),
                (10, "ali", "{base}",              "shc", "10"),
                (20, "ali", "{base}",              "shc", "0"),
            ]
        else:
            template_blocks = [
                (1,  "ali", "{base}", "hc",  "0"),
                (29, "ali", "{base}", "shc", "0"),
            ]

        starting = int(start_index or 1)
        run_mode = "multiref"

        try:
            block_dfs = []
            sidx = starting
            for n_iters, job, motl_pat, sm, temp in template_blocks:
                params_dict = _setup_form_to_params_dict(
                    iter_n=n_iters,
                    start_index=sidx,
                    cone_angle=cone_angle, cone_sampling=cone_sampling,
                    inplane_angle=inplane_angle, inplane_sampling=inplane_sampling,
                    high_pass=high_pass, high_pass_sigma=high_pass_sigma,
                    low_pass=low_pass, low_pass_sigma=low_pass_sigma,
                    threshold=threshold,
                    motl=motl_pat, ref=ref, mask=mask, cc_mask=cc_mask,
                    wedge_list=wedge_list, subtomo_path=subtomo_path,
                    symmetry=symmetry, class_id=class_id,
                    split_even_odd=bool(split_even_odd) if split_even_odd is not None else True,
                    rootdir=rootdir, binning=binning,
                    create_ref=(job == "avg"), ref_family=run_mode,
                    apply_laplacian=bool(apply_laplacian),
                    calc_exp=bool(calc_exp) if calc_exp is not None else True,
                    calc_ctf=bool(calc_ctf) if calc_ctf is not None else True,
                    cos_weight=cos_weight, score_weight=score_weight,
                    search_mode=sm or "hc",
                    cone_search_type=cone_search_type,
                    scoring_fcn=scoring_fcn, rot_mode=rot_mode,
                    avg_mode=avg_mode, subset=subset, fthresh=fthresh,
                    temperature=temp,
                    use_euler_search=False,
                )
                params_dict["subtomo_mode"] = sta_mod.compose_subtomo_mode(job, run_mode)
                block_params = sta_mod.StaParameters.from_dict(params_dict, sta_type="stopgap")
                block_dfs.append(block_params.df)
                sidx += n_iters

            combined_df = pd.concat(block_dfs, ignore_index=True)
            params = sta_mod.StopgapParams(
                combined_df, create_ref=is_denovo, ref_family=run_mode,
            )
        except Exception as exc:
            return no_update, f"Template generation failed: {exc}", no_update

        if params.df.empty:
            return no_update, "Template produced an empty dataframe.", no_update

        display_df = _build_display_df(params)
        data = {
            "records": display_df.to_dict("records"),
            "columns": list(display_df.columns),
            "sta_type": "stopgap",
        }
        label = "de-novo (31 rows)" if is_denovo else "existing refs (30 rows)"
        return (
            data,
            f"Generated {label} parameter table. Edit in the 'STA setup output' tab.",
            "tab-setup",
        )

    # ── R7: Prefill run-folder panel from loaded parameter file ──────────────

    @app.callback(
        Output({"type": "path-input", "owner": "sta-rf-output-base"}, "value",
               allow_duplicate=True),
        Output("sta-rf-folder-name", "value", allow_duplicate=True),
        Output({"type": "path-input", "owner": "sta-rf-mask"}, "value",
               allow_duplicate=True),
        Output({"type": "path-input", "owner": "sta-rf-ccmask"}, "value",
               allow_duplicate=True),
        Output({"type": "path-input", "owner": "sta-rf-wedgelist"}, "value",
               allow_duplicate=True),
        Output("sta-rf-motl-paths", "value", allow_duplicate=True),
        Output("sta-rf-ref-files", "value", allow_duplicate=True),
        Output("sta-rf-prefill-source", "children", allow_duplicate=True),
        Input("sta-setup-df-store", "data"),
        Input("sta-params-store", "data"),
        prevent_initial_call=True,
    )
    def _prefill_run_folder_from_params(setup_data, params_data):
        # Prefer newly created (setup_data) over loaded file (params_data)
        data = setup_data if setup_data and setup_data.get("records") else params_data
        if not data or not data.get("records"):
            return (no_update,) * 8
        try:
            last = data["records"][-1]
            is_setup = bool(setup_data and setup_data.get("records"))
            sta_type = data.get("sta_type", "novasta") if is_setup else "canonical"

            def _s(v):
                return str(v).strip() if v not in (None, "", "nan") else None

            rootdir = None
            output_base = None
            folder_name = None

            if sta_type == "stopgap":
                # Display DF uses STOPGAP column names
                rootdir = _s(last.get("rootdir"))
                mask = _s(last.get("mask_name") or last.get("mask"))
                ccmask = _s(last.get("ccmask_name") or last.get("cc_mask"))
                wedgelist = _s(last.get("wedgelist_name") or last.get("wedge_list"))
                motl = _s(last.get("motl_name") or last.get("motl"))
                ref = _s(last.get("ref_name") or last.get("ref"))
            else:
                # Loaded params store uses canonical column names
                rootdir = _s(last.get("rootdir"))
                mask = _s(last.get("mask") or last.get("mask name"))
                ccmask = _s(last.get("cc mask") or last.get("cc mask name") or last.get("cc_mask"))
                wedgelist = _s(last.get("wedge list") or last.get("wedge_list"))
                motl = _s(last.get("motl"))
                ref = _s(last.get("ref"))

            if rootdir:
                from pathlib import Path as _Path
                p = _Path(rootdir.rstrip("/").rstrip("\\"))
                output_base = str(p.parent) if p.parent != p else None
                folder_name = p.name or None

            if is_setup:
                source_msg = "Prefilled from newly created parameter table."
            else:
                source_msg = "Prefilled from loaded parameter file."

            return (
                output_base, folder_name,
                mask, ccmask, wedgelist,
                motl, ref,
                source_msg,
            )
        except Exception as exc:
            return (no_update,) * 7 + (f"Prefill error: {exc}",)

    # ── Multi-class consensus callbacks ──────────────────────────────────────

    @app.callback(
        Output("sta-mc-factor-store", "data"),
        Output("sta-mc-run-counter", "data"),
        Output("sta-mc-t-slider", "max"),
        Output("sta-mc-t-slider", "value"),
        Output("sta-mc-t-slider", "marks"),
        Output("sta-mc-linkage-wrapper", "style", allow_duplicate=True),
        Output("sta-mc-diagnose-area", "children"),
        Output("sta-mc-run-status", "children"),
        Output("sta-mc-produce-btn", "disabled"),
        Output("sta-main-tabs", "value", allow_duplicate=True),
        Input("sta-mc-run-btn", "n_clicks"),
        State("sta-mc-pool-value", "data"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.GRAPH_SETTINGS_STORE, "data"),
        prevent_initial_call=True,
    )
    def _mc_run(n_clicks, motl_ids, registry, gs):
        _tab = "tab-mc"
        if not motl_ids or len(motl_ids) < 2:
            msg = html.Span("Select ≥ 2 motls in the picker above, then click Run.",
                            style={"color": "var(--bs-warning)"})
            return (no_update, no_update, no_update, no_update, no_update,
                    no_update, msg, msg, True, _tab)
        factor, err = _mc_build_factor(motl_ids, registry)
        if err:
            msg = html.Span(f"Error: {err}", style={"color": "var(--bs-danger)"})
            return (no_update, no_update, no_update, no_update, no_update,
                    no_update, msg, msg, True, _tab)
        try:
            summary = reliability_summary(factor)
            R = factor.n_runs
            marks = _mc_slider_marks(R)
            k_default = round(R * 0.7) if R > 1 else R
            linkage_style = _MC_STYLE_ENABLED if k_default < R else _MC_STYLE_DISABLED
            diagnose = _mc_diagnose_figures(summary, gs, factor)
        except Exception as exc:
            msg = html.Span(f"Error during analysis: {exc}", style={"color": "var(--bs-danger)"})
            return (no_update, no_update, no_update, no_update, no_update,
                    no_update, msg, msg, True, _tab)
        status = html.Span(f"Ready — {R} runs, {summary['n_particles']} particles.",
                           style={"color": "#EAAE47"})
        return (
            _mc_serialize_factor(factor),
            n_clicks,
            R, k_default, marks,
            linkage_style,
            diagnose,
            status,
            False,
            _tab,
        )

    @app.callback(
        Output("sta-mc-preview-area", "children"),
        Input("sta-mc-t-slider", "value"),
        Input("sta-mc-linkage", "value"),
        Input("sta-mc-min-group-size", "value"),
        Input("sta-mc-run-counter", "data"),
        State("sta-mc-factor-store", "data"),
        State(ids.GRAPH_SETTINGS_STORE, "data"),
        prevent_initial_call=True,
    )
    def _mc_preview(k_val, linkage, min_gs, _run_ctr, factor_data, gs):
        if not factor_data:
            raise dash.exceptions.PreventUpdate
        factor = _mc_deserialize_factor(factor_data)
        result = _mc_run_consensus(factor, k_val, linkage, min_gs)
        return _mc_preview_children(result, gs)

    @app.callback(
        Output("sta-mc-linkage-wrapper", "style", allow_duplicate=True),
        Input("sta-mc-t-slider", "value"),
        State("sta-mc-factor-store", "data"),
        prevent_initial_call=True,
    )
    def _mc_toggle_linkage(k_val, factor_data):
        if not factor_data:
            raise dash.exceptions.PreventUpdate
        factor = _mc_deserialize_factor(factor_data)
        k = int(k_val) if k_val is not None else factor.n_runs
        return _MC_STYLE_ENABLED if k < factor.n_runs else _MC_STYLE_DISABLED

    @app.callback(
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output("sta-mc-produce-status", "children"),
        Output("sta-mc-heatmap-area", "children"),
        Input("sta-mc-produce-btn", "n_clicks"),
        State("sta-mc-factor-store", "data"),
        State("sta-mc-t-slider", "value"),
        State("sta-mc-linkage", "value"),
        State("sta-mc-min-group-size", "value"),
        State("sta-mc-pool-value", "data"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        State(ids.GRAPH_SETTINGS_STORE, "data"),
        prevent_initial_call=True,
    )
    def _mc_produce(n_clicks, factor_data, k_val, linkage, min_gs,
                    motl_ids, registry, meta, next_id, gs):
        if not factor_data or not motl_ids:
            raise dash.exceptions.PreventUpdate
        factor = _mc_deserialize_factor(factor_data)
        new_state, mid, result = _mc_execute_produce(
            factor, k_val, linkage, min_gs, motl_ids, registry, meta, next_id
        )
        reg, m, nid = new_state.to_stores()
        return reg, m, nid, _mc_produce_status(mid, result), _mc_heatmap_div(factor, result, gs)
