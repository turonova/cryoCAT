"""Memthick page — M1 (Configure + Generate code) + M2 (Analyze + Visualize).

The pipeline itself runs on cluster login nodes; this tab covers the two
non-execution halves of the workflow:

* **Configure** -- a :func:`cryocat.app.formgen.build_form` form on
  :func:`cryocat.analysis.memthick.run_full_pipeline` for all scalar /
  Literal / numeric-tuple params, plus three composite widgets from
  :mod:`cryocat.app.components.memthick_widgets`: ``membrane_labels`` (dict
  editor), ``surface_separation_mode`` (single-mode + per-membrane override),
  and a nested ``IntensityProfileAnalyzer`` sub-form.
* **Generate code** -- a "Build" button that snapshots the form into a
  preview area, plus Save/Download buttons for ``.py``, ``.ipynb`` and an
  optional SLURM wrapper. All artifacts come from
  :mod:`cryocat.app.suite.pages._memthick_codegen`.
* **Analyze** (M2) -- load the pipeline's output CSVs / pickles back in,
  show summary statistics (boundary-mode counts), render every plot
  produced by ``docs/.../memthick_analyze_plot.py``, and export the
  per-surface motls. Heavy ``MembraneData`` / analysis-result objects
  live server-side in :mod:`cryocat.app.components.memthick_registry`;
  only handles cross the wire.

The page never calls ``run_full_pipeline`` itself — that's the explicit
M1 invariant. Execution happens off-app.

Contract: exposes :data:`layout` and :func:`register_callbacks(app)`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import dash
from dash import html, dcc, Input, Output, State, ALL, no_update, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from cryocat.analysis import memthick
from cryocat.app import formgen, ids
from cryocat.app.apputils import generate_kwargs, run_operation
from cryocat.app.components import memthick_widgets as mw
from cryocat.app.components import memthick_registry as mreg
from cryocat.app.components.graphsettings import apply_settings_to_figure, error_figure
from cryocat.app.components.logpanel import get_log_panel, register_log_panel_callbacks
from cryocat.app.components.motlsink import (
    get_send_to_editor_button, register_send_to_editor_callbacks,
)
from cryocat.app.suite.pages import _memthick_analysis as analysis_helpers
from cryocat.app.suite.pages import _memthick_codegen as codegen
from cryocat.app.pageshell import page_shell, sidebar_accordion


# Lazy import of the tutorial's analyze/plot module. It lives next to the
# notebooks (``docs/.../memthick_analyze_plot.py``) rather than under
# ``cryocat.analysis``, so we try the obvious cryocat path first, then fall
# back to inserting the tutorial directory on ``sys.path``. The Analyze
# section surfaces a clear status message on failure.
_MAPLE: Any = None
_MAPLE_IMPORT_ERROR: Exception | None = None


def _import_maple() -> Any:
    """Return the ``memthick_analyze_plot`` module or raise ``ImportError``.

    Cached on first call. Looks under ``cryocat.analysis.memthick_analyze_plot``
    first, then under any directory matching the
    ``docs/source/tutorials/membrane_thickness`` checkout layout.
    """
    global _MAPLE, _MAPLE_IMPORT_ERROR
    if _MAPLE is not None:
        return _MAPLE
    if _MAPLE_IMPORT_ERROR is not None:
        raise _MAPLE_IMPORT_ERROR
    try:
        from cryocat.analysis import memthick_analyze_plot as maple  # type: ignore
        _MAPLE = maple
        return maple
    except Exception:
        pass
    candidates = [
        Path(__file__).resolve().parents[4] / "docs" / "source" / "tutorials" / "membrane_thickness",
        Path.cwd() / "docs" / "source" / "tutorials" / "membrane_thickness",
    ]
    for d in candidates:
        if (d / "memthick_analyze_plot.py").exists():
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
            try:
                import memthick_analyze_plot as maple  # type: ignore
                _MAPLE = maple
                return maple
            except Exception as exc:
                _MAPLE_IMPORT_ERROR = exc
                raise
    _MAPLE_IMPORT_ERROR = ImportError(
        "Could not find memthick_analyze_plot. Expected it under "
        "cryocat.analysis or docs/source/tutorials/membrane_thickness/."
    )
    raise _MAPLE_IMPORT_ERROR


# Three params are handled by composite widgets and excluded from the auto-form.
_COMPOSITE_PARAMS = ("membrane_labels", "surface_separation_mode", "analyzer")

# Pipeline-form id namespace (round-trips via :func:`generate_kwargs`).
_PIPELINE_ID_TYPE = "memthick-param"

# Analyzer sub-form prefix; the widget passes this through to ``build_form``.
_ANALYZER_PREFIX = "memthick-analyzer"


# Stage groupings mirror the comment sections in
# :func:`cryocat.analysis.memthick.run_full_pipeline`. The accordion in the
# Configure sidebar mounts one item per stage, each rendered by a separate
# :func:`build_form` call that *excludes* every param outside that stage so
# the layout matches the source's own organisation.
_STAGE_GENERAL = ("segmentation_map", "output_path")
_STAGE_SURFACE = (
    "step_size_marching_cubes", "smooth_sigma_segmentation",
    "subdivision_iterations", "snap_vertices_to_boundary",
    "refine_normals", "radius_hit", "flip_normals",
    "save_vertices_mrc", "save_split_surface_meshes",
)
_STAGE_MATCH = (
    "max_distance_nm", "max_angle", "direction", "use_gpu", "num_cpu_threads",
    "batch_size", "query_batch_size", "pixel_size_nm",
)
_STAGE_PROFILE = (
    "extract_intensity_profiles", "tomogram_map", "profile_half_width_nm",
    "intensity_save_profiles", "intensity_save_statistics",
    "compatibility_tolerance_nm", "save_thickness_mrc",
)
_ALL_STAGE_PARAMS = _STAGE_GENERAL + _STAGE_SURFACE + _STAGE_MATCH + _STAGE_PROFILE


# ── Layout helpers ───────────────────────────────────────────────────────────


_HINT = {"fontSize": "0.8rem", "color": "var(--color9)", "margin": "0.3rem 0"}
_SECTION_HEADER = {"fontSize": "0.95rem", "fontWeight": 600, "margin": "0.4rem 0 0.2rem"}


def _section(title: str, body) -> html.Div:
    return html.Div(
        [
            html.Div(title, style=_SECTION_HEADER),
            html.Div(body),
            html.Hr(style={"margin": "0.4rem 0"}),
        ]
    )


def _stage_form(stage_params: tuple[str, ...]) -> list:
    """Build a form row list for one stage of :func:`run_full_pipeline`.

    The trick: call :func:`build_form` with an ``exclude`` set covering every
    pipeline param *not* in this stage (plus the three composite params).
    The output is a flat list of rows for the requested stage only, so the
    accordion items mirror the source's stage-comment groupings without
    duplicating the parameter list.
    """
    exclude = set(_ALL_STAGE_PARAMS) - set(stage_params) | set(_COMPOSITE_PARAMS)
    return formgen.build_form(
        memthick.run_full_pipeline,
        id_type=_PIPELINE_ID_TYPE,
        exclude=exclude,
    )


def _configure_section() -> html.Div:
    """Sidebar form: composite widgets interleaved with a stage-grouped accordion.

    The accordion items mirror the comment sections in
    :func:`cryocat.analysis.memthick.run_full_pipeline`: General &
    Surface extraction & Geometric matching & Profile / boundary. Each item
    pairs the build_form-derived rows for that stage with the composite
    widget that lives in the same logical area:

    * General + ``membrane_labels``
    * Surface extraction + ``surface_separation_mode``
    * Geometric matching (scalars only)
    * Profile / boundary + ``IntensityProfileAnalyzer`` sub-form
    """
    return dbc.Accordion(
        [
            dbc.AccordionItem(
                html.Div(
                    [
                        html.Div(_stage_form(_STAGE_GENERAL)),
                        html.Hr(style={"margin": "0.4rem 0"}),
                        html.Div("Membrane labels", style=_SECTION_HEADER),
                        mw.get_label_dict_field("memthick-labels"),
                    ]
                ),
                title="General",
                item_id="memthick-stage-general",
            ),
            dbc.AccordionItem(
                html.Div(
                    [
                        html.Div(_stage_form(_STAGE_SURFACE)),
                        html.Hr(style={"margin": "0.4rem 0"}),
                        html.Div("Surface separation mode", style=_SECTION_HEADER),
                        mw.get_per_membrane_mode_field("memthick-mode", default_mode="planar"),
                    ]
                ),
                title="Surface extraction",
                item_id="memthick-stage-surface",
            ),
            dbc.AccordionItem(
                html.Div(_stage_form(_STAGE_MATCH)),
                title="Geometric matching",
                item_id="memthick-stage-match",
            ),
            dbc.AccordionItem(
                html.Div(
                    [
                        html.Div(_stage_form(_STAGE_PROFILE)),
                        html.Hr(style={"margin": "0.4rem 0"}),
                        html.Div("Intensity profile analyzer", style=_SECTION_HEADER),
                        mw.get_analyzer_subform(_ANALYZER_PREFIX),
                    ]
                ),
                title="Profile / boundary",
                item_id="memthick-stage-profile",
            ),
        ],
        always_open=True,
        active_item=["memthick-stage-general"],
    )


def _generate_section() -> html.Div:
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Build code",
                            id="memthick-build-btn",
                            color="primary",
                            size="sm",
                            style={"width": "100%"},
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="memthick-format",
                            options=[
                                {"label": ".py", "value": "py"},
                                {"label": ".ipynb", "value": "ipynb"},
                                {"label": ".py + SLURM wrapper", "value": "slurm"},
                            ],
                            value="py",
                            clearable=False,
                            style={"fontSize": "0.85rem"},
                        ),
                        width=6,
                    ),
                ],
                className="g-1",
            ),
            dbc.Collapse(
                html.Div(
                    [
                        html.Small(
                            "SBATCH directives (one per line, e.g. --mem=32G or -N 1)",
                            style=_HINT,
                        ),
                        dcc.Textarea(
                            id="memthick-sbatch",
                            placeholder="--mem=32G\n--time=24:00:00\n-N 1",
                            style={"width": "100%", "minHeight": "60px", "fontFamily": "monospace"},
                        ),
                        html.Small("Module loads (one per line)", style=_HINT),
                        dcc.Textarea(
                            id="memthick-modules",
                            placeholder="cryocat/1.0\ncuda/12.1",
                            style={"width": "100%", "minHeight": "40px", "fontFamily": "monospace"},
                        ),
                    ],
                    style={"marginTop": "0.4rem"},
                ),
                id="memthick-slurm-collapse",
                is_open=False,
            ),
            html.Hr(style={"margin": "0.4rem 0"}),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Input(
                            id="memthick-save-path",
                            type="text",
                            placeholder="/path/to/run_memthick.py",
                            style={"width": "100%", "fontSize": "0.85rem"},
                        ),
                        width=8,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Save",
                            id="memthick-save-btn",
                            color="secondary",
                            size="sm",
                            style={"width": "100%"},
                        ),
                        width=4,
                    ),
                ],
                className="g-1",
            ),
            html.Div(id="memthick-save-status", style=_HINT),
        ]
    )


# ── Analyze section ──────────────────────────────────────────────────────────


_PLOT_TABS: tuple[tuple[str, str], ...] = (
    ("code",        "Generated code"),
    ("boundary",    "Boundary summary"),
    ("thickness",   "Thickness distribution"),
    ("min_to_min",  "Min-to-min distances"),
    ("thick3d",     "3D map"),
    ("profiles",    "Intensity profiles"),
    ("binned",      "Binned profiles"),
    ("surfaces",    "Surfaces"),
)

_THICKNESS_REGIMES = [
    {"label": "All", "value": ""},
    {"label": "max_max", "value": "max_max"},
    {"label": "max_anchor / anchor_max", "value": "max_anchor"},
    {"label": "minima_only", "value": "minima_only"},
]
_OUTLIER_METHODS = [
    {"label": "None", "value": ""},
    {"label": "IQR", "value": "iqr"},
    {"label": "Percentile", "value": "percentile"},
    {"label": "Std dev", "value": "std"},
]
_THICKNESS_MODES = [
    {"label": "Auto (from CSV name)", "value": "auto"},
    {"label": "Inflection points", "value": "inflection_points"},
    {"label": "Minima", "value": "minima"},
    {"label": "Segmentation boundaries", "value": "segmentation_boundaries"},
]


def _filter_panel() -> html.Div:
    """Shared filter controls that apply to every plot."""
    return html.Div(
        [
            html.Small("Thickness range [nm]", style=_HINT),
            dbc.Row(
                [
                    dbc.Col(dcc.Input(id="memthick-filter-thick-min", type="number",
                                      placeholder="min", step="any",
                                      style={"width": "100%", "fontSize": "0.85rem"}), width=6),
                    dbc.Col(dcc.Input(id="memthick-filter-thick-max", type="number",
                                      placeholder="max", step="any",
                                      style={"width": "100%", "fontSize": "0.85rem"}), width=6),
                ],
                className="g-1",
            ),
            html.Small("Minima-separation range [nm]", style=_HINT),
            dbc.Row(
                [
                    dbc.Col(dcc.Input(id="memthick-filter-msep-min", type="number",
                                      placeholder="min", step="any",
                                      style={"width": "100%", "fontSize": "0.85rem"}), width=6),
                    dbc.Col(dcc.Input(id="memthick-filter-msep-max", type="number",
                                      placeholder="max", step="any",
                                      style={"width": "100%", "fontSize": "0.85rem"}), width=6),
                ],
                className="g-1",
            ),
            html.Small("Detection-mode regime", style=_HINT),
            dcc.Dropdown(id="memthick-filter-regime", options=_THICKNESS_REGIMES,
                         value="", clearable=False,
                         style={"fontSize": "0.85rem"}),
            html.Small("Outlier removal", style=_HINT),
            dcc.Dropdown(id="memthick-filter-outlier", options=_OUTLIER_METHODS,
                         value="", clearable=False,
                         style={"fontSize": "0.85rem"}),
            dbc.Row(
                [
                    dbc.Col(dcc.Input(id="memthick-filter-iqr", type="number",
                                      value=1.5, step=0.1,
                                      style={"width": "100%", "fontSize": "0.85rem"}), width=4),
                    dbc.Col(dcc.Input(id="memthick-filter-std", type="number",
                                      value=2.0, step=0.1,
                                      style={"width": "100%", "fontSize": "0.85rem"}), width=4),
                    dbc.Col(dcc.Input(id="memthick-filter-pmin", type="number",
                                      value=5, step=1,
                                      style={"width": "100%", "fontSize": "0.85rem"}), width=2),
                    dbc.Col(dcc.Input(id="memthick-filter-pmax", type="number",
                                      value=95, step=1,
                                      style={"width": "100%", "fontSize": "0.85rem"}), width=2),
                ],
                className="g-1",
            ),
            html.Small(
                "IQR factor / std factor / percentile-low / percentile-high",
                style=_HINT,
            ),
        ]
    )


def _membrane_selector() -> html.Div:
    """Multi-select dropdown of loaded membranes; drives the plot overlays."""
    return html.Div(
        [
            html.Small("Membranes to overlay", style=_HINT),
            dcc.Dropdown(
                id="memthick-membrane-select",
                options=[],
                value=[],
                multi=True,
                placeholder="Load some membranes first",
                style={"fontSize": "0.85rem"},
            ),
        ]
    )


def _load_panel() -> html.Div:
    return html.Div(
        [
            html.Small("Pipeline output folder", style=_HINT),
            dcc.Input(id="memthick-load-output", type="text",
                      placeholder="path/to/outputs",
                      style={"width": "100%", "fontSize": "0.85rem"}),
            html.Small("Segmentation base name (matches the M1 segmentation stem)", style=_HINT),
            dcc.Input(id="memthick-load-seg-base", type="text",
                      placeholder="e.g. 2140_z150to400_segmented",
                      style={"width": "100%", "fontSize": "0.85rem"}),
            html.Small("Membrane names (comma / newline separated)", style=_HINT),
            dcc.Textarea(id="memthick-load-membranes",
                         placeholder="ER, IMM, OMM",
                         style={"width": "100%", "minHeight": "50px", "fontSize": "0.85rem",
                                "fontFamily": "monospace"}),
            html.Small("Pixel size [nm] (blank = auto from pickle)", style=_HINT),
            dcc.Input(id="memthick-load-pixel-size", type="number", step="any",
                      placeholder="auto",
                      style={"width": "100%", "fontSize": "0.85rem"}),
            html.Hr(style={"margin": "0.4rem 0"}),
            dbc.Button("Load", id="memthick-load-btn", color="primary", size="sm",
                       style={"width": "100%"}),
            html.Div(id="memthick-load-status",
                     style={**_HINT, "marginTop": "0.4rem", "wordBreak": "break-word"}),
        ]
    )


def _plot_controls() -> html.Div:
    """Per-plot key controls. The list comes straight from the spec table."""
    return html.Div(
        [
            html.Small("Histogram bins", style=_HINT),
            dcc.Input(id="memthick-plot-bins", type="number", value=60, step=1,
                      style={"width": "100%", "fontSize": "0.85rem"}),
            dbc.Checkbox(id="memthick-plot-density", value=True,
                         label="Density-normalised histograms"),
            html.Small("3D color scale", style=_HINT),
            dcc.Input(id="memthick-plot-color-scale", type="text", value="OrRd",
                      style={"width": "100%", "fontSize": "0.85rem"}),
            dbc.Row(
                [
                    dbc.Col(dcc.Input(id="memthick-plot-color-min", type="number",
                                      placeholder="cmin", step="any",
                                      style={"width": "100%", "fontSize": "0.85rem"}), width=6),
                    dbc.Col(dcc.Input(id="memthick-plot-color-max", type="number",
                                      placeholder="cmax", step="any",
                                      style={"width": "100%", "fontSize": "0.85rem"}), width=6),
                ],
                className="g-1",
            ),
            html.Small("3D marker size / sample fraction", style=_HINT),
            dbc.Row(
                [
                    dbc.Col(dcc.Input(id="memthick-plot-marker-size", type="number",
                                      value=2, step=1,
                                      style={"width": "100%", "fontSize": "0.85rem"}), width=6),
                    dbc.Col(dcc.Input(id="memthick-plot-sample-frac", type="number",
                                      value=0.2, step=0.05, min=0, max=1,
                                      style={"width": "100%", "fontSize": "0.85rem"}), width=6),
                ],
                className="g-1",
            ),
            dbc.Checkbox(id="memthick-plot-color-by-mean", value=False,
                         label="3D: color by membrane mean"),
            html.Small("Profile extension range [nm]", style=_HINT),
            dbc.Row(
                [
                    dbc.Col(dcc.Input(id="memthick-plot-ext-min", type="number",
                                      placeholder="min", step="any",
                                      style={"width": "100%", "fontSize": "0.85rem"}), width=6),
                    dbc.Col(dcc.Input(id="memthick-plot-ext-max", type="number",
                                      placeholder="max", step="any",
                                      style={"width": "100%", "fontSize": "0.85rem"}), width=6),
                ],
                className="g-1",
            ),
            html.Small("Profile-summary toggles", style=_HINT),
            dbc.Checklist(
                id="memthick-plot-profile-toggles",
                options=[
                    {"label": "Segmentation-boundary markers",
                     "value": "show_segmentation_boundary_markers"},
                    {"label": "Segmentation-boundary distributions",
                     "value": "show_segmentation_boundary_distributions"},
                    {"label": "Inflection-point markers",
                     "value": "show_inflection_point_markers"},
                    {"label": "Inflection-point distributions",
                     "value": "show_inflection_point_distributions"},
                    {"label": "Outward maxima", "value": "show_outward_maxima"},
                    {"label": "Minima", "value": "show_minima"},
                    {"label": "Minima midpoint", "value": "show_minima_midpoint"},
                    {"label": "Percentile bands", "value": "show_percentile_bands"},
                ],
                value=[
                    "show_segmentation_boundary_markers",
                    "show_inflection_point_markers",
                    "show_minima_midpoint",
                ],
                style={"fontSize": "0.85rem"},
            ),
            html.Small("Binned: bins + method", style=_HINT),
            dbc.Row(
                [
                    dbc.Col(dcc.Input(id="memthick-plot-thick-bins", type="number",
                                      value=4, step=1,
                                      style={"width": "100%", "fontSize": "0.85rem"}), width=6),
                    dbc.Col(dcc.Dropdown(id="memthick-plot-bin-method",
                                         options=[{"label": "quantile", "value": "quantile"},
                                                  {"label": "equal-width", "value": "equal_width"}],
                                         value="quantile", clearable=False,
                                         style={"fontSize": "0.85rem"}), width=6),
                ],
                className="g-1",
            ),
            html.Small("Surfaces: .ply base path + opacity", style=_HINT),
            dcc.Input(id="memthick-plot-ply-base", type="text",
                      placeholder="blank → scatter only",
                      style={"width": "100%", "fontSize": "0.85rem"}),
            dbc.Row(
                [
                    dbc.Col(dcc.Input(id="memthick-plot-mesh-opacity", type="number",
                                      value=0.5, step=0.05, min=0, max=1,
                                      style={"width": "100%", "fontSize": "0.85rem"}), width=6),
                    dbc.Col(dbc.Checkbox(id="memthick-plot-show-scatter", value=True,
                                         label="Show scatter overlay"), width=6),
                ],
                className="g-1",
            ),
        ]
    )


def _export_panel() -> html.Div:
    return html.Div(
        [
            html.Small("Membrane to export", style=_HINT),
            dcc.Dropdown(id="memthick-export-membrane", options=[], value=None,
                         clearable=False, style={"fontSize": "0.85rem"}),
            html.Small("Surface", style=_HINT),
            dcc.Dropdown(
                id="memthick-export-surface",
                options=[{"label": "Surface 1", "value": "surface1"},
                         {"label": "Surface 2", "value": "surface2"}],
                value="surface1", clearable=False, style={"fontSize": "0.85rem"},
            ),
            html.Small("Score column", style=_HINT),
            dcc.Input(id="memthick-export-score-col", type="text", value="thickness_nm",
                      style={"width": "100%", "fontSize": "0.85rem"}),
            html.Small("Sample fraction (blank = all)", style=_HINT),
            dcc.Input(id="memthick-export-sample-frac", type="number", step=0.05,
                      min=0, max=1, placeholder="all",
                      style={"width": "100%", "fontSize": "0.85rem"}),
            html.Hr(style={"margin": "0.4rem 0"}),
            dbc.Button("Create motls", id="memthick-export-build-btn", color="secondary",
                       size="sm", style={"width": "100%", "marginBottom": "0.3rem"}),
            html.Div(id="memthick-export-status", style=_HINT),
            html.Hr(style={"margin": "0.4rem 0"}),
            html.Small("Send to motl editor", style=_HINT),
            get_send_to_editor_button("memthick-export"),
            html.Hr(style={"margin": "0.4rem 0"}),
            html.Small("Save to disk", style=_HINT),
            html.Small("Output directory", style=_HINT),
            dcc.Input(id="memthick-export-save-dir", type="text",
                      placeholder="defaults to CSV directory",
                      style={"width": "100%", "fontSize": "0.85rem"}),
            html.Small("Thickness mode (filename tag)", style=_HINT),
            dcc.Dropdown(id="memthick-export-mode", options=_THICKNESS_MODES,
                         value="auto", clearable=False, style={"fontSize": "0.85rem"}),
            dbc.Button("Save motls", id="memthick-export-save-btn", color="secondary",
                       size="sm", style={"width": "100%", "marginTop": "0.3rem"}),
            html.Div(id="memthick-export-save-status",
                     style={**_HINT, "wordBreak": "break-word"}),
            dcc.Store(id="memthick-export-result"),
        ]
    )


def _analyze_section() -> html.Div:
    return html.Div(
        [
            dbc.Accordion(
                [
                    dbc.AccordionItem(_load_panel(), title="Load",
                                      item_id="memthick-an-load"),
                    dbc.AccordionItem(_membrane_selector(), title="Membranes",
                                      item_id="memthick-an-select"),
                    dbc.AccordionItem(_filter_panel(), title="Filters",
                                      item_id="memthick-an-filter"),
                    dbc.AccordionItem(_plot_controls(), title="Plot controls",
                                      item_id="memthick-an-controls"),
                    dbc.AccordionItem(_export_panel(), title="Export motls",
                                      item_id="memthick-an-export"),
                ],
                always_open=True,
                active_item=["memthick-an-load"],
            ),
            dcc.Store(id="memthick-results-handles", data=[]),
        ]
    )


def _sidebar() -> list:
    return [
        sidebar_accordion(
            [
                dbc.AccordionItem(_configure_section(), title="Configure",
                                  item_id="memthick-acc-config"),
                dbc.AccordionItem(_generate_section(), title="Generate code",
                                  item_id="memthick-acc-gen"),
                dbc.AccordionItem(_analyze_section(), title="Analyze",
                                  item_id="memthick-acc-analyze"),
            ],
            active_item=["memthick-acc-config", "memthick-acc-gen"],
        ),
    ]


def _plot_tab(tab_id: str, label: str) -> dcc.Tab:
    if tab_id == "code":
        body = html.Pre(
            id="memthick-code-preview",
            style={
                "fontFamily": "monospace",
                "fontSize": "0.85rem",
                "background": "var(--bs-light)",
                "padding": "0.75rem",
                "borderRadius": "4px",
                "overflow": "auto",
                "minHeight": "60vh",
                "whiteSpace": "pre-wrap",
                "wordBreak": "break-word",
            },
        )
    elif tab_id == "boundary":
        body = html.Div(
            id="memthick-boundary-table",
            style={"padding": "0.5rem"},
        )
    else:
        body = dcc.Graph(
            id={"type": "styled-graph", "owner": "memthick", "name": tab_id},
            style={"height": "calc(100vh - 140px)", "width": "100%"},
            config={"displayModeBar": True, "scrollZoom": True},
        )
    return dcc.Tab(label=label, value=tab_id, children=body)


def _main() -> list:
    return [
        dcc.Tabs(
            id="memthick-main-tabs",
            value="code",
            children=[_plot_tab(tab_id, label) for tab_id, label in _PLOT_TABS],
            style={"marginBottom": "0.25rem"},
        ),
        dcc.Store(id="memthick-built-store"),
        dcc.Download(id="memthick-download"),
    ]


layout = html.Div(
    [
        page_shell(_sidebar(), _main(), sidebar_width=4),
        *get_log_panel("memthick-log"),
    ],
    style={"margin": 0, "padding": 0},
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_lines_to_dict(text: str) -> dict:
    """Parse one ``key=value`` or ``key value`` per line into a dict.

    Used for the SBATCH textarea.
    """
    out: dict[str, str] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        else:
            parts = line.split(None, 1)
            key, value = parts[0], (parts[1] if len(parts) > 1 else "")
        out[key.strip()] = value.strip()
    return out


def _parse_lines_to_list(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def _build_kwargs(
    pipeline_ids: list[dict],
    pipeline_values: list,
    labels_rows: list[dict],
    mode_toggle: bool,
    mode_single: str,
    mode_per_label_ids: list[dict],
    mode_per_label_values: list[str],
    analyzer_ids: list[dict],
    analyzer_values: list,
) -> tuple[dict, dict]:
    """Combine the form + composite-widget states into pipeline + analyzer kwargs.

    Empty / None values are stripped so the generated script only sets what
    the user explicitly chose; everything else falls back to the function
    signature's default.
    """
    pipeline_kwargs = generate_kwargs(pipeline_ids or [], pipeline_values or [])
    pipeline_kwargs = {k: v for k, v in pipeline_kwargs.items() if v not in (None, "", [])}

    labels = mw.read_label_dict(labels_rows)
    if labels:
        pipeline_kwargs["membrane_labels"] = labels

    pipeline_kwargs["surface_separation_mode"] = mw.read_per_membrane_mode(
        mode_toggle, mode_single, mode_per_label_ids, mode_per_label_values,
    )

    analyzer_kwargs = mw.read_analyzer_kwargs(analyzer_ids or [], analyzer_values or [])
    return pipeline_kwargs, analyzer_kwargs


# ── Callbacks ────────────────────────────────────────────────────────────────


def register_callbacks(app):
    register_log_panel_callbacks(app, "memthick-log")
    mw.register_label_dict_callbacks(app, "memthick-labels")
    mw.register_per_membrane_mode_callbacks(app, "memthick-mode")
    mw.register_analyzer_subform_callbacks(app, _ANALYZER_PREFIX)

    # Mirror label-dict names into the per-membrane mode widget so its
    # override switch can render one dropdown per membrane.
    @app.callback(
        Output("memthick-mode-labels-store", "data"),
        Input("memthick-labels-rows", "data"),
    )
    def _mirror_labels(rows):
        return sorted({(r.get("name") or "").strip() for r in (rows or []) if (r.get("name") or "").strip()})

    # Show the SLURM extra fields only when the format requires them.
    @app.callback(
        Output("memthick-slurm-collapse", "is_open"),
        Input("memthick-format", "value"),
    )
    def _toggle_slurm(fmt):
        return fmt == "slurm"

    # ── Build code ───────────────────────────────────────────────────────────
    @app.callback(
        Output("memthick-code-preview", "children"),
        Output("memthick-built-store", "data"),
        Output("memthick-save-status", "children", allow_duplicate=True),
        Input("memthick-build-btn", "n_clicks"),
        # pipeline form
        State({"type": _PIPELINE_ID_TYPE, "param": ALL, "tag": ALL}, "value"),
        State({"type": _PIPELINE_ID_TYPE, "param": ALL, "tag": ALL}, "id"),
        # pipeline form -- Tuple slot ids carry extra `slot`/`elem` keys.
        State({"type": _PIPELINE_ID_TYPE, "param": ALL, "tag": ALL, "slot": ALL, "elem": ALL}, "value"),
        State({"type": _PIPELINE_ID_TYPE, "param": ALL, "tag": ALL, "slot": ALL, "elem": ALL}, "id"),
        # label dict
        State("memthick-labels-rows", "data"),
        # per-membrane mode
        State("memthick-mode-toggle", "value"),
        State("memthick-mode-single-mode", "value"),
        State({"type": "memthick-mode-per-label-mode", "label": ALL}, "value"),
        State({"type": "memthick-mode-per-label-mode", "label": ALL}, "id"),
        # analyzer sub-form (scalar + tuple slots)
        State({"type": _ANALYZER_PREFIX, "param": ALL, "tag": ALL}, "value"),
        State({"type": _ANALYZER_PREFIX, "param": ALL, "tag": ALL}, "id"),
        State({"type": _ANALYZER_PREFIX, "param": ALL, "tag": ALL,
               "slot": ALL, "elem": ALL}, "value"),
        State({"type": _ANALYZER_PREFIX, "param": ALL, "tag": ALL,
               "slot": ALL, "elem": ALL}, "id"),
        # output format + slurm extras
        State("memthick-format", "value"),
        State("memthick-sbatch", "value"),
        State("memthick-modules", "value"),
        State("memthick-save-path", "value"),
        prevent_initial_call=True,
    )
    def _build(
        n_clicks,
        pipe_scalar_vals, pipe_scalar_ids,
        pipe_tuple_vals, pipe_tuple_ids,
        labels_rows,
        mode_toggle, mode_single, mode_per_label_vals, mode_per_label_ids,
        an_scalar_vals, an_scalar_ids, an_tuple_vals, an_tuple_ids,
        fmt, sbatch_text, modules_text, save_path,
    ):
        if not n_clicks:
            raise PreventUpdate

        # Round-trip scalar + tuple-slot controls through one generate_kwargs call.
        pipe_ids = (pipe_scalar_ids or []) + (pipe_tuple_ids or [])
        pipe_vals = (pipe_scalar_vals or []) + (pipe_tuple_vals or [])
        an_ids = (an_scalar_ids or []) + (an_tuple_ids or [])
        an_vals = (an_scalar_vals or []) + (an_tuple_vals or [])

        try:
            pipeline_kwargs, analyzer_kwargs = _build_kwargs(
                pipe_ids, pipe_vals,
                labels_rows, mode_toggle, mode_single, mode_per_label_ids, mode_per_label_vals,
                an_ids, an_vals,
            )
        except Exception as exc:
            return f"# error building kwargs: {exc}", no_update, f"Build error: {exc}"

        if not pipeline_kwargs.get("segmentation_map"):
            return (
                "# segmentation_map is required.",
                no_update,
                "segmentation_map is required.",
            )

        # Merge analyzer kwargs into the assembled dict under the
        # ``analyzer`` key so the codegen layer can render both the
        # constructor and the pipeline call from a single payload.
        merged = dict(pipeline_kwargs)
        if analyzer_kwargs:
            merged["analyzer"] = analyzer_kwargs

        py_src = codegen.render_pipeline_py(merged)
        artifact = {
            "py": py_src,
            "ipynb_json": codegen.render_pipeline_ipynb_json(merged),
            "fmt": fmt,
        }
        if fmt == "slurm":
            slurm_target = save_path or "run_memthick.py"
            artifact["slurm"] = codegen.render_slurm_wrapper(
                slurm_target,
                cluster_params=_parse_lines_to_dict(sbatch_text),
                module_loads=_parse_lines_to_list(modules_text),
            )
        return py_src, artifact, "Built — click Save to write to disk."

    # ── Save to disk ─────────────────────────────────────────────────────────
    @app.callback(
        Output("memthick-save-status", "children", allow_duplicate=True),
        Output("memthick-download", "data"),
        Input("memthick-save-btn", "n_clicks"),
        State("memthick-built-store", "data"),
        State("memthick-save-path", "value"),
        State("memthick-format", "value"),
        prevent_initial_call=True,
    )
    def _save(n_clicks, artifact, save_path, fmt):
        if not n_clicks:
            raise PreventUpdate
        if not artifact:
            return "Build the code first.", no_update
        if not (save_path and str(save_path).strip()):
            # Fall back to a browser download if no server-side path is set.
            content, fname = _payload_for_download(artifact, fmt)
            return "Downloaded.", dict(content=content, filename=fname)

        save_path = str(save_path).strip()
        try:
            base = Path(save_path)
            base.parent.mkdir(parents=True, exist_ok=True)
            written: list[str] = []
            if fmt in ("py", "slurm"):
                py_path = base.with_suffix(".py")
                py_path.write_text(artifact["py"], encoding="utf-8")
                written.append(str(py_path))
                if fmt == "slurm":
                    sh_path = base.with_suffix(".sh")
                    sh_path.write_text(artifact.get("slurm", ""), encoding="utf-8")
                    written.append(str(sh_path))
            elif fmt == "ipynb":
                nb_path = base.with_suffix(".ipynb")
                nb_path.write_text(artifact["ipynb_json"], encoding="utf-8")
                written.append(str(nb_path))
            else:
                return f"Unknown format: {fmt}", no_update
            return "Saved: " + ", ".join(written), no_update
        except Exception as exc:
            return f"Save error: {exc}", no_update

    # ── M2: Send-to-editor wiring ────────────────────────────────────────────
    register_send_to_editor_callbacks(app, "memthick-export", "memthick-export-result")

    # ── M2: Load ─────────────────────────────────────────────────────────────
    @app.callback(
        Output("memthick-load-status", "children"),
        Output("memthick-results-handles", "data"),
        Output("memthick-membrane-select", "options"),
        Output("memthick-membrane-select", "value"),
        Output("memthick-export-membrane", "options"),
        Output("memthick-export-membrane", "value"),
        Input("memthick-load-btn", "n_clicks"),
        State("memthick-load-output", "value"),
        State("memthick-load-seg-base", "value"),
        State("memthick-load-membranes", "value"),
        State("memthick-load-pixel-size", "value"),
        State("memthick-results-handles", "data"),
        prevent_initial_call=True,
    )
    def _load(n_clicks, output_path, seg_base, membranes_text, pixel_size, existing_handles):
        if not n_clicks:
            raise PreventUpdate

        try:
            maple = _import_maple()
        except Exception as exc:
            return (
                f"memthick_analyze_plot not importable: {exc}",
                no_update, no_update, no_update, no_update, no_update,
            )

        membranes = analysis_helpers.parse_membrane_names(membranes_text)
        if not (output_path and seg_base and membranes):
            return (
                "Provide output folder, seg-base, and at least one membrane name.",
                no_update, no_update, no_update, no_update, no_update,
            )

        loaded: list[dict] = list(existing_handles or [])
        existing_names = {h.get("membrane") for h in loaded}
        loaded_now: list[str] = []
        failed: list[str] = []
        for membrane in membranes:
            if membrane in existing_names:
                continue
            csv_path = analysis_helpers.resolve_thickness_csv(output_path, seg_base, membrane)
            if not csv_path.exists():
                failed.append(f"{membrane} (missing {csv_path.name})")
                continue
            try:
                md = run_operation(
                    maple.load_membrane_data,
                    dict(
                        thickness_csv=str(csv_path),
                        auto_discover_related_files=True,
                        pixel_size_nm=pixel_size if pixel_size not in (None, "") else None,
                    ),
                )
                tr = maple.analyze_membrane_thickness(data=md)
                pr = maple.analyze_intensity_profiles(data=md)
                info = maple.return_boundary_info(md)
                # Resolve the pixel size that was actually used.
                resolved_ps = pixel_size if pixel_size not in (None, "") else None
                if resolved_ps is None and md.intensity_profiles:
                    resolved_ps = md.intensity_profiles[0].get("pixel_size")
                bundle = mreg.MembraneResults(
                    membrane=membrane,
                    membrane_data=md,
                    thickness_results=tr,
                    profile_results=pr,
                    boundary_info=info,
                    thickness_csv=str(csv_path),
                    pixel_size_nm=float(resolved_ps) if resolved_ps else None,
                )
                rid = mreg.registry.add(bundle)
                handle = mreg.make_handle(bundle)
                handle["id"] = rid
                loaded.append(handle)
                loaded_now.append(membrane)
            except Exception as exc:
                failed.append(f"{membrane} ({exc})")

        options = [{"label": h["membrane"], "value": h["id"]} for h in loaded]
        all_values = [h["id"] for h in loaded]
        status_parts = []
        if loaded_now:
            status_parts.append(f"Loaded: {', '.join(loaded_now)}")
        if failed:
            status_parts.append(f"Failed: {'; '.join(failed)}")
        if not status_parts:
            status_parts.append("Nothing to load — all selected membranes are already present.")
        first_membrane = options[0]["value"] if options else None
        return (
            " | ".join(status_parts),
            loaded,
            options, all_values,
            options, first_membrane,
        )

    # ── M2: Boundary-mode summary table ──────────────────────────────────────
    @app.callback(
        Output("memthick-boundary-table", "children"),
        Input("memthick-results-handles", "data"),
    )
    def _render_boundary_table(handles):
        if not handles:
            return html.Small("No membranes loaded yet.", style=_HINT)
        modes = sorted({m for h in handles for m in (h.get("by_detection_mode") or {})})
        header = [html.Th("Membrane"), html.Th("Rows"), html.Th("Resolved"), html.Th("Unresolved")]
        for m in modes:
            header.append(html.Th(m))
        rows = [html.Tr(header)]
        for h in handles:
            cells = [
                html.Td(h.get("membrane", "")),
                html.Td(f"{h.get('n_rows', 0):,}"),
                html.Td(f"{h.get('n_resolved', 0):,}"),
                html.Td(f"{h.get('n_unresolved', 0):,}"),
            ]
            for m in modes:
                cells.append(html.Td(f"{(h.get('by_detection_mode') or {}).get(m, 0):,}"))
            rows.append(html.Tr(cells))
        return dbc.Table(
            rows, bordered=True, striped=True, hover=True, size="sm",
            style={"fontSize": "0.85rem"},
        )

    # ── M2: Render plots ─────────────────────────────────────────────────────
    @app.callback(
        Output({"type": "styled-graph", "owner": "memthick", "name": ALL}, "figure"),
        Input("memthick-main-tabs", "value"),
        Input("memthick-membrane-select", "value"),
        Input("memthick-filter-thick-min", "value"),
        Input("memthick-filter-thick-max", "value"),
        Input("memthick-filter-msep-min", "value"),
        Input("memthick-filter-msep-max", "value"),
        Input("memthick-filter-regime", "value"),
        Input("memthick-filter-outlier", "value"),
        Input("memthick-filter-iqr", "value"),
        Input("memthick-filter-std", "value"),
        Input("memthick-filter-pmin", "value"),
        Input("memthick-filter-pmax", "value"),
        Input("memthick-plot-bins", "value"),
        Input("memthick-plot-density", "value"),
        Input("memthick-plot-color-scale", "value"),
        Input("memthick-plot-color-min", "value"),
        Input("memthick-plot-color-max", "value"),
        Input("memthick-plot-marker-size", "value"),
        Input("memthick-plot-sample-frac", "value"),
        Input("memthick-plot-color-by-mean", "value"),
        Input("memthick-plot-ext-min", "value"),
        Input("memthick-plot-ext-max", "value"),
        Input("memthick-plot-profile-toggles", "value"),
        Input("memthick-plot-thick-bins", "value"),
        Input("memthick-plot-bin-method", "value"),
        Input("memthick-plot-ply-base", "value"),
        Input("memthick-plot-mesh-opacity", "value"),
        Input("memthick-plot-show-scatter", "value"),
        State(ids.GRAPH_SETTINGS_STORE, "data"),
        State({"type": "styled-graph", "owner": "memthick", "name": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _render_plots(
        active_tab, selected_ids,
        t_min, t_max, m_min, m_max, regime, outlier_method,
        iqr_factor, std_factor, p_min, p_max,
        bins, density,
        color_scale, c_min, c_max, marker_size, sample_frac, color_by_mean,
        ext_min, ext_max, profile_toggles,
        thick_bins, bin_method,
        ply_base, mesh_opacity, show_scatter,
        graph_settings, tab_ids,
    ):
        if not tab_ids:
            raise PreventUpdate
        # Render only the active plot tab; leave the others alone.
        figures = [no_update] * len(tab_ids)
        if active_tab in ("code", "boundary"):
            return figures
        try:
            maple = _import_maple()
        except Exception:
            return figures
        if not selected_ids:
            return figures
        bundles = [b for b in (mreg.registry.get(rid) for rid in selected_ids) if b is not None]
        if not bundles:
            return figures

        thickness_data = [b.thickness_results for b in bundles]
        profile_data = [b.profile_results for b in bundles]
        names = [b.membrane for b in bundles]

        thickness_range = _tuple_or_none(t_min, t_max)
        msep_range = _tuple_or_none(m_min, m_max)
        ext_range = _tuple_or_none(ext_min, ext_max)
        color_range = _tuple_or_none(c_min, c_max)
        outlier_kwargs = {
            "outlier_removal_method": outlier_method or None,
            "outlier_iqr_factor": iqr_factor if iqr_factor is not None else 1.5,
            "outlier_std_factor": std_factor if std_factor is not None else 2.0,
            "outlier_percentile_range": (
                p_min if p_min is not None else 5,
                p_max if p_max is not None else 95,
            ),
        }
        toggles = {name: (name in (profile_toggles or [])) for name in (
            "show_segmentation_boundary_markers",
            "show_segmentation_boundary_distributions",
            "show_inflection_point_markers",
            "show_inflection_point_distributions",
            "show_outward_maxima",
            "show_minima",
            "show_minima_midpoint",
            "show_percentile_bands",
        )}

        try:
            fig = _build_plot_for_tab(
                maple, active_tab,
                thickness_data, profile_data, names,
                thickness_range=thickness_range, msep_range=msep_range,
                regime=regime or None, bins=bins,
                density=bool(density), color_scale=color_scale or None,
                color_range=color_range, marker_size=marker_size,
                sample_fraction=sample_frac, color_by_mean=bool(color_by_mean),
                ext_range=ext_range, toggles=toggles,
                thick_bins=thick_bins, bin_method=bin_method,
                ply_base=ply_base, mesh_opacity=mesh_opacity,
                show_scatter=bool(show_scatter),
                outlier_kwargs=outlier_kwargs,
            )
        except Exception as exc:
            err = error_figure(str(exc)).to_dict()
            for i, tid in enumerate(tab_ids):
                if tid.get("name") == active_tab:
                    figures[i] = err
            return figures

        fig_dict = fig.to_dict() if hasattr(fig, "to_dict") else fig
        apply_settings_to_figure(fig_dict, graph_settings or {})
        for i, tid in enumerate(tab_ids):
            if tid.get("name") == active_tab:
                figures[i] = fig_dict
        return figures

    # ── M2: Build motls (Create) ─────────────────────────────────────────────
    @app.callback(
        Output("memthick-export-status", "children"),
        Output("memthick-export-result", "data"),
        Input("memthick-export-build-btn", "n_clicks"),
        State("memthick-export-membrane", "value"),
        State("memthick-export-surface", "value"),
        State("memthick-export-score-col", "value"),
        State("memthick-export-sample-frac", "value"),
        prevent_initial_call=True,
    )
    def _build_motls(n_clicks, membrane_id, surface, score_col, sample_frac):
        if not n_clicks:
            raise PreventUpdate
        if not membrane_id:
            return "Load a membrane first.", no_update
        bundle = mreg.registry.get(membrane_id)
        if bundle is None:
            return "Selected membrane is no longer in the registry.", no_update
        try:
            maple = _import_maple()
            motl1, motl2 = run_operation(
                maple.create_thickness_motls,
                dict(
                    thickness_csv=bundle.thickness_csv,
                    score_column=score_col or "thickness_nm",
                    sample_fraction=sample_frac if sample_frac not in (None, "") else None,
                ),
            )
        except Exception as exc:
            return f"Error: {exc}", no_update
        chosen = motl1 if surface == "surface1" else motl2
        rows = analysis_helpers.motl_to_pool_rows(chosen)
        return (
            f"Built {len(rows)} {surface} rows for {bundle.membrane}.",
            rows,
        )

    # ── M2: Save motls to disk ───────────────────────────────────────────────
    @app.callback(
        Output("memthick-export-save-status", "children"),
        Input("memthick-export-save-btn", "n_clicks"),
        State("memthick-export-membrane", "value"),
        State("memthick-export-save-dir", "value"),
        State("memthick-export-mode", "value"),
        State("memthick-export-sample-frac", "value"),
        prevent_initial_call=True,
    )
    def _save_motls(n_clicks, membrane_id, save_dir, mode, sample_frac):
        if not n_clicks:
            raise PreventUpdate
        if not membrane_id:
            return "Load a membrane first."
        bundle = mreg.registry.get(membrane_id)
        if bundle is None:
            return "Selected membrane is no longer in the registry."
        try:
            maple = _import_maple()
            run_operation(
                maple.save_thickness_motls,
                dict(
                    thickness_csv=bundle.thickness_csv,
                    output_path=save_dir if save_dir else None,
                    sample_fraction=sample_frac if sample_frac not in (None, "") else None,
                    thickness_mode=mode or "auto",
                ),
            )
        except Exception as exc:
            return f"Error: {exc}"
        target = save_dir if save_dir else str(Path(bundle.thickness_csv).parent)
        return f"Saved motls for {bundle.membrane} to {target}."


def _tuple_or_none(lo, hi):
    if lo in (None, "") and hi in (None, ""):
        return None
    return (
        float(lo) if lo not in (None, "") else None,
        float(hi) if hi not in (None, "") else None,
    )


def _build_plot_for_tab(
    maple,
    tab: str,
    thickness_data,
    profile_data,
    names: list[str],
    *,
    thickness_range, msep_range, regime,
    bins, density, color_scale, color_range,
    marker_size, sample_fraction, color_by_mean,
    ext_range, toggles,
    thick_bins, bin_method,
    ply_base, mesh_opacity, show_scatter,
    outlier_kwargs,
):
    """Dispatch the active tab to its matching plot function.

    Each branch passes through the curated control set the spec calls out;
    irrelevant kwargs are dropped so the underlying signatures stay clean.
    """
    if tab == "thickness":
        return maple.plot_thickness_distribution(
            thickness_data, membrane_names=names,
            thickness_range_nm=thickness_range,
            minima_separation_range_nm=msep_range,
            thickness_regime=regime,
            histogram_bins=bins or 60,
            density_normalization=density,
            show_mean_lines=True,
            **outlier_kwargs,
        )
    if tab == "min_to_min":
        return maple.plot_min_to_min_distribution(
            thickness_data, membrane_names=names,
            minima_separation_range_nm=msep_range,
            thickness_regime=regime,
            histogram_bins=bins or 60,
            density_normalization=density,
            show_mean_lines=True,
            **outlier_kwargs,
        )
    if tab == "thick3d":
        return maple.plot_thickness_3d(
            thickness_data, membrane_names=names,
            thickness_range_nm=thickness_range,
            minima_separation_range_nm=msep_range,
            thickness_regime=regime,
            color_scale=color_scale,
            color_range=color_range,
            marker_size=marker_size,
            sample_fraction=sample_fraction,
            color_by_mean=color_by_mean,
            **outlier_kwargs,
        )
    if tab == "profiles":
        return maple.plot_intensity_profile_summary(
            profile_data, membrane_names=names,
            extension_range_nm=ext_range,
            thickness_range_nm=thickness_range,
            minima_separation_range_nm=msep_range,
            thickness_regime=regime,
            **toggles,
        )
    if tab == "binned":
        return maple.plot_intensity_profile_binned(
            profile_data, membrane_names=names,
            extension_range_nm=ext_range,
            thickness_bins=thick_bins or 4,
            binning_method=bin_method or "quantile",
            thickness_range_nm=thickness_range,
            minima_separation_range_nm=msep_range,
            thickness_regime=regime,
            **toggles,
        )
    if tab == "surfaces":
        return maple.plot_surfaces(
            thickness_data, membrane_names=names,
            thickness_range_nm=thickness_range,
            minima_separation_range_nm=msep_range,
            thickness_regime=regime,
            marker_size=marker_size,
            sample_fraction=sample_fraction,
            ply_base_path=ply_base or None,
            mesh_opacity=mesh_opacity,
            show_scatter=show_scatter,
        )
    raise ValueError(f"Unknown plot tab: {tab}")


def _payload_for_download(artifact: dict, fmt: str) -> tuple[str, str]:
    if fmt == "ipynb":
        return artifact["ipynb_json"], "run_memthick.ipynb"
    if fmt == "slurm":
        return artifact.get("slurm", ""), "run_memthick.sh"
    return artifact["py"], "run_memthick.py"
