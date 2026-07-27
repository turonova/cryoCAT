"""Peak Analysis (pana) tool.

Three usage modes share the same sidebar + 10-tab main area:

* **Single case** — fill in paths and output directory; runs
  :func:`~cryocat.analysis.pana.run_single_case` then opens the HTML
  summary in the next free result slot.
* **Visualize existing** — point to a case directory or load a template-list
  CSV; select a row to open its HTML summary in the next free slot.
* **Generate script** — uses the CSV path from the Visualize panel to
  produce a ready-to-submit ``.py`` / SLURM script (nothing runs in-app).

Main area tabs
--------------
* **Results table** (tab 0) — filename-only summary of all runs, populated
  from single-case runs or from loading a template-list CSV.  Rows are
  selectable; a "Visualize selected row" button opens the result HTML.
* **Result 1 – Result 9** (tabs 1–9) — fixed iframe slots; each holds one
  self-contained HTML summary page.

Contract: exposes ``layout`` and ``register_callbacks(app)``.
"""

from __future__ import annotations

import os
from pathlib import Path

import dash
from dash import html, dcc, dash_table, Input, Output, State, no_update, ctx, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd

from cryocat.analysis import pana
from cryocat.app import formgen
from cryocat.app.styles import (
    HINT as _HINT,
    CTRL_ROW as _ROW_STYLE,
    CTRL_LABEL as _LABEL_STYLE,
    CTRL_INPUT as _INPUT_WRAPPER,
    SECTION_HEADER as _SECTION_HEADER,
)
from cryocat.app.apputils import run_operation, generate_kwargs
from cryocat.app.components.logpanel import get_log_panel, register_log_panel_callbacks
from cryocat.app.components.anglesfield import get_angles_field, register_angles_field_callbacks
from cryocat.utils.wedgeutils import generate_wedge_mask
from cryocat.app.components.wedgepreview import wedge_xz_figure
from cryocat.app.suite.pages import _pana_codegen as codegen
from cryocat.app.pageshell import page_shell, sidebar_accordion


# ── Form helpers ─────────────────────────────────────────────────────────────


_COLLAPSE_INNER = {
    "backgroundColor": "rgba(var(--bs-secondary-rgb), 0.05)",
    "borderRadius": "0.25rem",
    "padding": "0.5rem",
    "marginBottom": "0.35rem",
}

_WEDGE_ID_TYPE = "ppana-wedge-param"
_WEDGE_BUILDER = "ppana-wedge"

_N_SLOTS = 9


def _label(text):
    return html.Label(text, style=_LABEL_STYLE)


def _row(label_text, component):
    return html.Div([_label(label_text), html.Div(component, style=_INPUT_WRAPPER)], style=_ROW_STYLE)


def _text(label, comp_id, placeholder="", value=None):
    return _row(label, dbc.Input(id=comp_id, type="text", placeholder=placeholder, value=value, size="sm"))


def _number(label, comp_id, value=None, step=None, min_=None, placeholder=""):
    return _row(
        label,
        dbc.Input(id=comp_id, type="number", value=value, step=step, min=min_, placeholder=placeholder, size="sm"),
    )


def _check(label, comp_id, value=False):
    return _row(label, dbc.Checkbox(id=comp_id, value=value))


def _dropdown(label, comp_id, options, value=None):
    return _row(label, dbc.Select(id=comp_id, options=options, value=value, size="sm"))


# ── Wedge mask builder modal ──────────────────────────────────────────────────

def _wedge_mask_modal() -> dbc.Modal:
    form_rows = formgen.build_form(
        generate_wedge_mask,
        id_type=_WEDGE_ID_TYPE,
        id_extra={"builder": _WEDGE_BUILDER},
    )
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Generate wedge mask")),
            dbc.ModalBody(
                [
                    html.Div(form_rows, style={"marginBottom": "0.75rem"}),
                    dbc.Input(
                        id="ppana-wedge-output-path",
                        type="text",
                        placeholder="Output path (e.g. /path/to/wedge_mask.em)",
                        size="sm",
                        style={"marginBottom": "0.4rem"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button("Preview (XZ slice)", id="ppana-wedge-preview-btn",
                                           color="secondary", size="sm", style={"width": "100%"}),
                                width=6,
                            ),
                            dbc.Col(
                                dbc.Button("Generate mask", id="ppana-wedge-generate-btn",
                                           color="primary", size="sm", style={"width": "100%"}),
                                width=6,
                            ),
                        ],
                        className="g-1",
                        style={"marginBottom": "0.4rem"},
                    ),
                    html.Div(id="ppana-wedge-modal-status", style={**_HINT, "wordBreak": "break-word"}),
                    html.Div(id="ppana-wedge-preview-area", style={"marginTop": "0.5rem"}),
                    dcc.Store(id="ppana-wedge-params"),
                    dcc.Store(id="ppana-wedge-created-path"),
                ],
            ),
            dbc.ModalFooter(
                [
                    dbc.Button("Use as target", id="ppana-wedge-use-target-btn", color="primary", size="sm", className="me-1"),
                    dbc.Button("Use as template", id="ppana-wedge-use-tmpl-btn", color="primary", size="sm", className="me-1"),
                    dbc.Button("Use as both", id="ppana-wedge-use-both-btn", color="success", size="sm", className="me-2"),
                    dbc.Button("Close", id="ppana-wedge-close-btn", color="secondary", size="sm"),
                ]
            ),
        ],
        id="ppana-wedge-modal",
        size="lg",
        is_open=False,
        centered=True,
        scrollable=True,
    )


# ── Mode-specific forms ───────────────────────────────────────────────────────

def _single_case_form():
    return html.Div(
        [
            # ─ Subtomogram extraction (optional) ────────────────────────────
            _check("Use subtomogram as target map", "ppana-s-use-subtomo"),
            dbc.Collapse(
                html.Div(
                    [
                        html.Small(
                            "Extract the highest-scoring subtomogram; output path is copied to Target map and "
                            "the best-peak rotation pre-fills Starting angles.",
                            style={**_HINT, "marginBottom": "0.3rem", "display": "block"},
                        ),
                        _text("Source tomogram (.em / .mrc)", "ppana-sub-tomo", "path/to/tomogram.em"),
                        _text("Motl file (.em / .csv)", "ppana-sub-motl", "path/to/motl.em"),
                        _number("Box size (voxels)", "ppana-sub-boxsize", step=1, min_=1, placeholder="e.g. 80"),
                        _text("Output path (optional)", "ppana-sub-output",
                              "blank → <output dir>/<case name>/subtomogram.em"),
                        html.Div(
                            dbc.Button("Extract and use this file", id="ppana-sub-btn",
                                       color="secondary", size="sm", style={"width": "100%", "marginTop": "0.3rem"}),
                        ),
                        html.Div(id="ppana-sub-status",
                                 style={**_HINT, "marginTop": "0.3rem", "wordBreak": "break-word"}),
                    ],
                    style=_COLLAPSE_INNER,
                ),
                id="ppana-s-subtomo-collapse",
                is_open=False,
            ),
            # ─ Wedge mask (optional) ─────────────────────────────────────────
            _check("Apply wedge mask", "ppana-s-apply-wedge"),
            dbc.Collapse(
                html.Div(
                    [
                        html.Small(
                            "Provide pre-computed wedge masks, or generate one via the button below.",
                            style={**_HINT, "marginBottom": "0.3rem", "display": "block"},
                        ),
                        _text("Wedge mask — target", "ppana-sw-target", "path/to/wedge_mask_target.mrc"),
                        _text("Wedge mask — template", "ppana-sw-tmpl", "path/to/wedge_mask_tmpl.mrc"),
                        html.Div(
                            dbc.Button("Generate wedge masks…", id="ppana-wedge-open-btn",
                                       color="secondary", size="sm",
                                       style={"width": "100%", "marginTop": "0.3rem"}),
                        ),
                    ],
                    style=_COLLAPSE_INNER,
                ),
                id="ppana-s-wedge-collapse",
                is_open=False,
            ),
            # ─ Map file paths ────────────────────────────────────────────────
            html.Small("Paths to map files", style={**_HINT, "marginBottom": "0.3rem", "display": "block"}),
            _text("Target map", "ppana-s-tomogram", "path/to/tomo.mrc"),
            _text("Template", "ppana-s-template", "path/to/template.mrc"),
            _text("Template mask", "ppana-s-mask", "path/to/mask.mrc"),
            _text("Tight mask (optional)", "ppana-s-tight-mask", "path/to/tight_mask.mrc"),
            _row("Angles file", get_angles_field("ppana-angles")),
            html.Hr(style={"margin": "0.4rem 0"}),
            # ─ Output ────────────────────────────────────────────────────────
            html.Small("Output", style={**_HINT, "marginBottom": "0.3rem", "display": "block"}),
            _text("Output directory", "ppana-s-output-dir", "path/to/output/"),
            _text("Case name", "ppana-s-case-name", "e.g. ribosome_c1"),
            _dropdown(
                "If output exists",
                "ppana-s-if-exists",
                [
                    {"label": "Overwrite", "value": "overwrite"},
                    {"label": "Error", "value": "error"},
                    {"label": "Timestamp (new subdir)", "value": "timestamp"},
                ],
                value="overwrite",
            ),
            html.Hr(style={"margin": "0.4rem 0"}),
            # ─ Analysis options ───────────────────────────────────────────────
            html.Small("Analysis options", style={**_HINT, "marginBottom": "0.3rem", "display": "block"}),
            _text("Starting angles (φ,θ,ψ)", "ppana-s-starting-angle",
                  "e.g. 0, 0, 0  (leave blank for default)"),
            _number("Cyclic symmetry", "ppana-s-symmetry", value=1, step=1, min_=1),
            _number("CC radius (voxels)", "ppana-s-cc-radius", value=10, step=1, min_=1),
            _check("Compute distance map", "ppana-s-compute-dist", value=True),
            _check("Compute peak statistics", "ppana-s-compute-peak", value=True),
            dbc.Collapse(
                html.Div(
                    [
                        _number("Degrees threshold (°)", "ppana-s-degrees",
                                placeholder="search angle increment", step="any"),
                    ],
                    style=_COLLAPSE_INNER,
                ),
                id="ppana-s-peak-collapse",
                is_open=True,
            ),
            # ─ Angular histograms (optional) ──────────────────────────────────
            _check("Compute angular histograms", "ppana-s-compute-gradual"),
            dbc.Collapse(
                html.Div(
                    [
                        html.Small(
                            "Sweeps integer angles 0 … angular_range−1 for combined, cone-only, and "
                            "in-plane rotations. Writes gradual CSVs and updates the HTML summary.",
                            style={**_HINT, "marginBottom": "0.3rem", "display": "block"},
                        ),
                        _number("Angular range (°)", "ppana-s-angular-range", value=359, step=1, min_=1),
                    ],
                    style=_COLLAPSE_INNER,
                ),
                id="ppana-s-gradual-collapse",
                is_open=False,
            ),
            html.Hr(style={"margin": "0.4rem 0"}),
            dbc.Button("Run", id="ppana-s-run-btn", color="primary", size="sm", style={"width": "100%"}),
            dbc.Button(
                "Recompute peak statistics",
                id="ppana-s-recompute-btn",
                color="secondary",
                size="sm",
                disabled=True,
                style={"width": "100%", "marginTop": "0.3rem"},
            ),
            dbc.Button(
                "Recompute angular histograms",
                id="ppana-s-recompute-gradual-btn",
                color="secondary",
                size="sm",
                disabled=True,
                style={"width": "100%", "marginTop": "0.3rem"},
            ),
        ],
        id="ppana-single-form",
    )


def _visualize_form():
    return html.Div(
        [
            # ─ Case directory section (existing) ────────────────────────────
            html.Div("Case directory", style=_SECTION_HEADER),
            html.Small(
                "Point to a directory produced by run_single_case (contains scores.em, angles.em, etc.).",
                style={**_HINT, "marginBottom": "0.3rem", "display": "block"},
            ),
            _text("Case directory", "ppana-v-case-dir", "path/to/case_name/"),
            _check("Compute distance map (needs angles.em + angles.csv)", "ppana-v-compute-dist", value=False),
            _number("  CC radius (voxels)", "ppana-v-cc-radius", value=10, step=1, min_=1),
            _check("Compute peak stats (needs scores.em)", "ppana-v-compute-peak", value=False),
            _number("  Degrees threshold", "ppana-v-degrees", placeholder="required for peak stats", step="any"),
            html.Div(id="ppana-v-artifacts",
                     style={**_HINT, "marginTop": "0.4rem", "wordBreak": "break-word"}),
            dbc.Button("Visualize", id="ppana-v-run-btn", color="primary", size="sm",
                       style={"width": "100%"}),
            html.Hr(style={"margin": "0.8rem 0"}),
            # ─ Template list CSV section ─────────────────────────────────────
            html.Div("Template list CSV", style=_SECTION_HEADER),
            html.Small(
                "Load an existing template-list CSV to browse and visualize results. "
                "This CSV path is also used by Generate script.",
                style={**_HINT, "marginBottom": "0.3rem", "display": "block"},
            ),
            _text("Template list CSV", "ppana-v-csv-path", "path/to/template_list.csv"),
            _text("Parent folder path", "ppana-v-csv-parent", "base directory for structure folders"),
            _row(
                "If table has data",
                dbc.RadioItems(
                    id="ppana-v-csv-mode",
                    options=[
                        {"label": "Replace", "value": "replace"},
                        {"label": "Append", "value": "append"},
                    ],
                    value="replace",
                    inline=True,
                    style={"fontSize": "0.82rem"},
                ),
            ),
            dbc.Button("Load to table", id="ppana-v-load-csv-btn", color="secondary", size="sm",
                       style={"width": "100%"}),
            html.Div(id="ppana-v-csv-status",
                     style={**_HINT, "marginTop": "0.3rem", "wordBreak": "break-word"}),
        ],
        id="ppana-visualize-form",
    )


def _generate_script_form():
    return html.Div(
        [
            html.Small(
                "Uses the template-list CSV and parent folder from the Visualize panel. "
                "Nothing runs in-app — the script is saved to disk for cluster submission.",
                style={**_HINT, "marginBottom": "0.4rem", "display": "block"},
            ),
            _text("Angle list folder", "ppana-g-angle-path", "directory with angle-list files"),
            _text("Wedge mask folder (optional)", "ppana-g-wedge-path", "directory with wedge-mask files"),
            _number("CC radius tolerance", "ppana-g-cc-radius", value=10, step=1, min_=1),
            html.Hr(style={"margin": "0.4rem 0"}),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id="ppana-g-format",
                            options=[
                                {"label": ".py script", "value": "py"},
                                {"label": ".py + SLURM wrapper", "value": "slurm"},
                            ],
                            value="py",
                            clearable=False,
                            style={"fontSize": "0.85rem"},
                        ),
                        width=12,
                    ),
                ],
                className="g-1",
                style={"marginBottom": "0.35rem"},
            ),
            dbc.Collapse(
                html.Div(
                    [
                        html.Small("SBATCH directives (one per line, e.g. --mem=32G)", style=_HINT),
                        dcc.Textarea(
                            id="ppana-g-sbatch",
                            placeholder="--mem=32G\n--time=24:00:00\n-N 1",
                            style={"width": "100%", "minHeight": "60px", "fontFamily": "monospace",
                                   "fontSize": "0.82rem"},
                        ),
                        html.Small("Module loads (one per line)", style=_HINT),
                        dcc.Textarea(
                            id="ppana-g-modules",
                            placeholder="cryocat/1.0",
                            style={"width": "100%", "minHeight": "40px", "fontFamily": "monospace",
                                   "fontSize": "0.82rem"},
                        ),
                    ],
                    style={"marginTop": "0.4rem"},
                ),
                id="ppana-g-slurm-collapse",
                is_open=False,
            ),
            html.Hr(style={"margin": "0.4rem 0"}),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Input(id="ppana-g-save-path", type="text",
                                  placeholder="/path/to/run_pana.py", size="sm"),
                        width=8,
                    ),
                    dbc.Col(
                        dbc.Button("Generate", id="ppana-g-generate-btn", color="primary", size="sm",
                                   style={"width": "100%"}),
                        width=4,
                    ),
                ],
                className="g-1",
            ),
            html.Div(id="ppana-g-status", style={**_HINT, "marginTop": "0.3rem", "wordBreak": "break-word"}),
        ],
        id="ppana-script-form",
    )


# ── Layout ────────────────────────────────────────────────────────────────────

def _sidebar() -> list:
    return [
        sidebar_accordion(
            [
                dbc.AccordionItem(
                    _single_case_form(),
                    title="Single case",
                    item_id="ppana-acc-single",
                ),
                dbc.AccordionItem(
                    _visualize_form(),
                    title="Visualize existing",
                    item_id="ppana-acc-visualize",
                ),
                dbc.AccordionItem(
                    _generate_script_form(),
                    title="Generate script",
                    item_id="ppana-acc-script",
                ),
            ],
            active_item=["ppana-acc-single"],
        ),
    ]


def _csv_tab_content():
    """Fixed content for the Results table tab (DataTable + visualize button)."""
    return html.Div(
        [
            dash_table.DataTable(
                id="ppana-csv-table",
                columns=[],
                data=[],
                row_selectable="single",
                selected_rows=[],
                style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "calc(100vh - 160px)"},
                style_cell={
                    "fontSize": "0.82rem",
                    "padding": "4px 8px",
                    "maxWidth": "220px",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                    "whiteSpace": "nowrap",
                },
                style_header={"fontWeight": "bold", "backgroundColor": "#f0f2f5"},
                tooltip_delay=0,
                tooltip_duration=None,
                page_size=25,
            ),
            html.Div(
                [
                    dbc.Button(
                        "Visualize selected row",
                        id="ppana-csv-visualize-btn",
                        color="secondary",
                        size="sm",
                        style={"marginTop": "0.5rem"},
                    ),
                    html.Span(
                        id="ppana-csv-vis-status",
                        style={**_HINT, "marginLeft": "0.75rem"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center"},
            ),
        ],
        style={"padding": "0.5rem"},
    )


def _slot_placeholder(i: int):
    return html.Div(
        f"Result slot {i} — run a single case or visualize an existing result to populate this slot.",
        style={"color": "#aaa", "padding": "2rem", "fontSize": "0.9rem"},
    )


def _main() -> list:
    slot_tabs = [dcc.Tab(label=f"Result {i}", value=f"tab-r{i}",
                         children=[html.Div(id=f"ppana-slot-{i}", children=_slot_placeholder(i))])
                 for i in range(1, _N_SLOTS + 1)]

    return [
        html.Div(
            id="ppana-status",
            style={
                "fontSize": "0.9rem",
                "color": "var(--color9)",
                "padding": "0.3rem 0.5rem 0",
                "whiteSpace": "pre-wrap",
                "wordBreak": "break-word",
                "minHeight": "1.4rem",
            },
        ),
        dcc.Tabs(
            id="ppana-main-tabs",
            value="tab-csv",
            children=[
                dcc.Tab(
                    label="Results table",
                    value="tab-csv",
                    children=[_csv_tab_content()],
                ),
                *slot_tabs,
            ],
            style={"marginBottom": "0"},
        ),
        # Stores
        dcc.Store(id="ppana-slots-store", data={}),
        dcc.Store(id="ppana-csv-rows-store", data=[]),
        dcc.Store(id="ppana-next-slot", data=1),
        dcc.Store(id="ppana-s-last-run", data={}),
    ]


layout = html.Div(
    [
        page_shell(_sidebar(), _main()),
        _wedge_mask_modal(),
        *get_log_panel("ppana-log"),
    ],
    style={"margin": "0", "padding": "0"},
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_starting_angle(s):
    if not s or not str(s).strip():
        return None
    parts = [float(x.strip()) for x in str(s).replace(";", ",").split(",")]
    if len(parts) != 3:
        raise ValueError(f"Starting angles: expected 3 comma-separated values, got {len(parts)}.")
    return tuple(parts)


_LABELS = {
    "score_slices": "Score slices",
    "line_profiles": "Line profiles",
    "peak_shape": "Peak shape",
    "distance_slices_all": "Dist: all",
    "distance_slices_normals": "Dist: normals",
    "distance_slices_inplane": "Dist: in-plane",
    "angle_distribution": "Angle dist.",
}

# Columns shown in the CSV table (subset; hidden keys like _write_dir are not listed here)
_TABLE_DISPLAY_COLS = ["Template", "Target map", "Mask", "Case name", "Output dir", "Status"]

_RESIZE_SCRIPT = (
    "<script>\n"
    "window.addEventListener('load', function() {\n"
    "  if (window.Plotly) {\n"
    "    document.querySelectorAll('.js-plotly-plot').forEach(\n"
    "      function(d) { Plotly.Plots.resize(d); }\n"
    "    );\n"
    "  }\n"
    "});\n"
    "</script>"
)


def _make_summary_html(figs: dict, run_params: dict | None = None) -> str:
    """Build a standalone HTML page from :func:`pana.visualize_results` output."""
    parts: list[str] = []

    if run_params:
        rows_html = "".join(
            f"<tr><td class='k'>{k}</td><td>{v}</td></tr>"
            for k, v in run_params.items() if v
        )
        parts.append(f"<h2>Run parameters</h2><table class='params'>{rows_html}</table>")

    first = True
    for key, fig in figs.items():
        include_plotlyjs = "cdn" if first else False
        first = False
        fig_html = fig.to_html(
            include_plotlyjs=include_plotlyjs,
            full_html=False,
            default_width="100%",
            config={"responsive": True},
        )
        label = _LABELS.get(key, key)
        parts.append(f"<h2>{label}</h2>\n{fig_html}")

    css = (
        "body{font-family:sans-serif;padding:1rem;max-width:1400px;margin:0 auto;}"
        "h2{font-size:1.05rem;margin:1rem 0 0.3rem;color:#333;"
        "border-bottom:1px solid #eee;padding-bottom:0.2rem;}"
        "table.params{border-collapse:collapse;margin-bottom:0.5rem;font-size:0.85rem;}"
        "table.params td{padding:0.2rem 0.6rem;border:1px solid #ddd;}"
        "table.params td.k{background:#f5f5f5;font-weight:500;}"
    )
    body = "\n".join(parts)
    return (
        f"<!DOCTYPE html>\n<html>\n"
        f"<head><meta charset='utf-8'><style>{css}</style></head>\n"
        f"<body>\n{body}\n{_RESIZE_SCRIPT}\n</body>\n</html>"
    )


def _load_or_generate_summary_html(
    case_dir: str,
    figs: dict | None = None,
    run_params: dict | None = None,
) -> str:
    """Return the summary HTML for *case_dir*.

    Priority:
    1. ``params.csv`` present → generate via :func:`pana.create_summary_html_from_folder`
       (writes ``summary.html`` to disk, returns its content).
    2. Pre-existing ``summary.html`` or ``id_*_summary.html`` → read and return.
    3. *figs* provided → fall back to :func:`_make_summary_html`.
    """
    p = Path(case_dir)
    if (p / "params.csv").exists():
        html_path = pana.create_summary_html_from_folder(case_dir)
        with open(html_path, encoding="utf-8") as fh:
            return fh.read()
    for cand in [p / "summary.html", *sorted(p.glob("id_*_summary.html"))]:
        if cand.exists():
            with open(cand, encoding="utf-8") as fh:
                return fh.read()
    if figs:
        return _make_summary_html(figs, run_params)
    raise FileNotFoundError(f"No summary HTML or params.csv found in {case_dir!r}")


def _assign_slot(slots: dict, next_slot: int, html_content: str) -> tuple[dict, int, str]:
    """Put html_content in slot *next_slot*, return updated stores + new active tab."""
    updated = dict(slots)
    updated[str(next_slot)] = html_content
    new_next = (next_slot % _N_SLOTS) + 1
    active_tab = f"tab-r{next_slot}"
    return updated, new_next, active_tab


def _file_row(tomo, tmpl, mask, angles, outdir, name, write_dir=None) -> dict:
    """Build a display row dict for the CSV table from single-case inputs."""
    def fname(p):
        return Path(p).name if p else ""
    row = {
        "Template":   fname(tmpl),
        "Target map": fname(tomo),
        "Mask":       fname(mask),
        "Case name":  name or "",
        "Output dir": outdir or "",
        "Status":     "Done",
        "_write_dir": write_dir or "",
    }
    return row


def _table_columns() -> list[dict]:
    return [{"name": c, "id": c} for c in _TABLE_DISPLAY_COLS]


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):
    register_log_panel_callbacks(app, "ppana-log")
    register_angles_field_callbacks(app, "ppana-angles")

    # ── toggle subtomogram sub-form ──────────────────────────────────────────
    @app.callback(
        Output("ppana-s-subtomo-collapse", "is_open"),
        Input("ppana-s-use-subtomo", "value"),
    )
    def _toggle_subtomo_form(use_subtomo):
        return bool(use_subtomo)

    # ── extract subtomogram → pre-fill target map + starting angles ──────────
    @app.callback(
        Output("ppana-s-tomogram", "value"),
        Output("ppana-s-starting-angle", "value"),
        Output("ppana-sub-status", "children"),
        Input("ppana-sub-btn", "n_clicks"),
        State("ppana-sub-tomo", "value"),
        State("ppana-sub-motl", "value"),
        State("ppana-sub-boxsize", "value"),
        State("ppana-sub-output", "value"),
        State("ppana-s-output-dir", "value"),
        State("ppana-s-case-name", "value"),
        prevent_initial_call=True,
    )
    def _extract_subtomo(n_clicks, sub_tomo, sub_motl, sub_boxsize, sub_output, s_outdir, s_name):
        if not n_clicks:
            raise PreventUpdate
        missing = [
            f for f, v in [("Source tomogram", sub_tomo), ("Motl file", sub_motl), ("Box size", sub_boxsize)]
            if not v
        ]
        if missing:
            return no_update, no_update, f"Missing: {', '.join(missing)}."
        if not sub_output:
            if not s_outdir or not s_name:
                return no_update, no_update, "Provide output path or fill Output directory + Case name first."
            sub_output = str(Path(s_outdir) / s_name / "subtomogram.em")
        try:
            result = run_operation(
                pana.extract_best_subtomogram,
                dict(tomogram=sub_tomo, motl=sub_motl, box_size=int(sub_boxsize), output_path=sub_output),
            )
            rotation = result.get("rotation")
            if rotation is not None:
                r = np.asarray(rotation).flatten()
                angles_str = f"{r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f}"
            else:
                angles_str = no_update
            return sub_output, angles_str, f"Extracted to: {sub_output}"
        except Exception as exc:
            return no_update, no_update, f"Error: {exc}"

    # ── toggle wedge mask collapse ───────────────────────────────────────────
    @app.callback(
        Output("ppana-s-wedge-collapse", "is_open"),
        Input("ppana-s-apply-wedge", "value"),
    )
    def _toggle_wedge_form(apply_wedge):
        return bool(apply_wedge)

    # ── wedge mask builder: collect params ───────────────────────────────────
    @app.callback(
        Output("ppana-wedge-params", "data"),
        Input({"type": _WEDGE_ID_TYPE, "builder": _WEDGE_BUILDER, "param": ALL, "tag": ALL}, "value"),
        State({"type": _WEDGE_ID_TYPE, "builder": _WEDGE_BUILDER, "param": ALL, "tag": ALL}, "id"),
    )
    def _collect_wedge_params(values, ids):
        if not values or not ids:
            raise PreventUpdate
        return generate_kwargs(ids, values)

    # ── wedge mask builder: generate ─────────────────────────────────────────
    @app.callback(
        Output("ppana-wedge-modal-status", "children"),
        Output("ppana-wedge-created-path", "data"),
        Input("ppana-wedge-generate-btn", "n_clicks"),
        State("ppana-wedge-params", "data"),
        State("ppana-wedge-output-path", "value"),
        prevent_initial_call=True,
    )
    def _generate_wedge_mask_cb(n_clicks, params, out_path):
        if not n_clicks:
            raise PreventUpdate
        if not params:
            return "Fill in the form parameters first.", no_update
        required = ["map_size", "wedgelist", "tomo_number"]
        missing = [r for r in required if params.get(r) is None]
        if missing:
            return f"Missing required fields: {', '.join(missing)}.", no_update
        if not out_path or not str(out_path).strip():
            return "Specify an output path first.", no_update
        try:
            kwargs = {k: v for k, v in params.items() if v is not None}
            kwargs["output_path"] = out_path
            run_operation(generate_wedge_mask, kwargs)
            return f"Generated → {out_path}", out_path
        except Exception as exc:
            return f"Error: {exc}", no_update

    # ── wedge mask builder: in-modal XZ preview ──────────────────────────────
    @app.callback(
        Output("ppana-wedge-preview-area", "children"),
        Output("ppana-wedge-modal-status", "children", allow_duplicate=True),
        Input("ppana-wedge-preview-btn", "n_clicks"),
        State("ppana-wedge-params", "data"),
        prevent_initial_call=True,
    )
    def _preview_wedge_mask(n_clicks, params):
        if not n_clicks:
            raise PreventUpdate
        if not params:
            return no_update, "Fill in the form parameters first."
        required = ["map_size", "wedgelist", "tomo_number"]
        missing = [r for r in required if params.get(r) is None]
        if missing:
            return no_update, f"Missing required fields: {', '.join(missing)}."
        try:
            kwargs = {k: v for k, v in params.items() if v is not None and k != "output_path"}
            result = generate_wedge_mask(**kwargs)
            mask = result["mask"] if isinstance(result, dict) else result
            graph = dcc.Graph(figure=wedge_xz_figure(mask),
                              style={"height": "420px", "width": "100%"})
            return graph, f"Preview rendered (mask shape {mask.shape})."
        except Exception as exc:
            return no_update, f"Preview error: {exc}"

    # ── wedge mask modal: open / close / use ─────────────────────────────────
    @app.callback(
        Output("ppana-wedge-modal", "is_open"),
        Output("ppana-sw-target", "value"),
        Output("ppana-sw-tmpl", "value"),
        Input("ppana-wedge-open-btn", "n_clicks"),
        Input("ppana-wedge-close-btn", "n_clicks"),
        Input("ppana-wedge-use-target-btn", "n_clicks"),
        Input("ppana-wedge-use-tmpl-btn", "n_clicks"),
        Input("ppana-wedge-use-both-btn", "n_clicks"),
        State("ppana-wedge-modal", "is_open"),
        State("ppana-wedge-created-path", "data"),
        prevent_initial_call=True,
    )
    def _wedge_modal_dispatch(n_open, n_close, n_target, n_tmpl, n_both, is_open, created):
        t = ctx.triggered_id
        if t == "ppana-wedge-open-btn":
            return True, no_update, no_update
        if t == "ppana-wedge-close-btn":
            return False, no_update, no_update
        if t == "ppana-wedge-use-target-btn":
            return False, created or no_update, no_update
        if t == "ppana-wedge-use-tmpl-btn":
            return False, no_update, created or no_update
        if t == "ppana-wedge-use-both-btn":
            return False, created or no_update, created or no_update
        raise PreventUpdate

    # ── scan case dir for artifacts ──────────────────────────────────────────
    @app.callback(
        Output("ppana-v-artifacts", "children"),
        Input("ppana-v-case-dir", "value"),
    )
    def _scan_case_dir(case_dir):
        if not case_dir or not os.path.isdir(case_dir):
            return ""
        found = [f for f in pana._ARTIFACT_FILES if (Path(case_dir) / f).exists()]
        if not found:
            return "No recognised artifacts found in this directory."
        return "Found: " + ", ".join(found)

    # ── toggle SLURM options ─────────────────────────────────────────────────
    @app.callback(
        Output("ppana-g-slurm-collapse", "is_open"),
        Input("ppana-g-format", "value"),
    )
    def _toggle_slurm_collapse(fmt):
        return fmt == "slurm"

    # ── toggle peak stats sub-form ───────────────────────────────────────────
    @app.callback(
        Output("ppana-s-peak-collapse", "is_open"),
        Input("ppana-s-compute-peak", "value"),
    )
    def _toggle_peak_form(checked):
        return bool(checked)

    # ── toggle angular histograms sub-form ──────────────────────────────────
    @app.callback(
        Output("ppana-s-gradual-collapse", "is_open"),
        Input("ppana-s-compute-gradual", "value"),
    )
    def _toggle_gradual_form(checked):
        return bool(checked)

    # ── main run dispatch (single case + visualize existing) ─────────────────
    @app.callback(
        Output("ppana-status", "children"),
        Output("ppana-slots-store", "data"),
        Output("ppana-csv-rows-store", "data"),
        Output("ppana-next-slot", "data"),
        Output("ppana-main-tabs", "value"),
        Output("ppana-s-last-run", "data"),
        Output("ppana-s-recompute-btn", "disabled"),
        Output("ppana-s-recompute-gradual-btn", "disabled"),
        Input("ppana-s-run-btn", "n_clicks"),
        Input("ppana-v-run-btn", "n_clicks"),
        # single-case states
        State("ppana-s-tomogram", "value"),
        State("ppana-s-template", "value"),
        State("ppana-s-mask", "value"),
        State("ppana-s-tight-mask", "value"),
        State("ppana-angles-path", "value"),
        State("ppana-s-output-dir", "value"),
        State("ppana-s-case-name", "value"),
        State("ppana-s-if-exists", "value"),
        State("ppana-s-starting-angle", "value"),
        State("ppana-s-symmetry", "value"),
        State("ppana-s-cc-radius", "value"),
        State("ppana-s-degrees", "value"),
        State("ppana-s-compute-dist", "value"),
        State("ppana-s-compute-peak", "value"),
        State("ppana-s-apply-wedge", "value"),
        State("ppana-sw-target", "value"),
        State("ppana-sw-tmpl", "value"),
        State("ppana-s-compute-gradual", "value"),
        State("ppana-s-angular-range", "value"),
        # visualize states
        State("ppana-v-case-dir", "value"),
        State("ppana-v-compute-dist", "value"),
        State("ppana-v-cc-radius", "value"),
        State("ppana-v-compute-peak", "value"),
        State("ppana-v-degrees", "value"),
        # shared stores
        State("ppana-slots-store", "data"),
        State("ppana-csv-rows-store", "data"),
        State("ppana-next-slot", "data"),
        prevent_initial_call=True,
    )
    def _run(
        n_single, n_vis,
        s_tomo, s_tmpl, s_mask, s_tight_mask, s_angles, s_outdir, s_name, s_if_exists,
        s_starting_angle, s_sym, s_cc, s_deg, s_compute_dist, s_compute_peak,
        s_apply_wedge, sw_target, sw_tmpl, s_compute_gradual, s_angular_range,
        v_case_dir, v_compute_dist, v_cc, v_compute_peak, v_deg,
        slots, csv_rows, next_slot,
    ):
        triggered = ctx.triggered_id
        if not triggered:
            raise PreventUpdate
        slots = slots or {}
        csv_rows = list(csv_rows or [])
        next_slot = int(next_slot or 1)
        try:
            if triggered == "ppana-v-run-btn":
                status, new_slots, new_csv, new_next, active_tab = _run_visualize(
                    v_case_dir, bool(v_compute_dist), int(v_cc or 10),
                    bool(v_compute_peak), float(v_deg) if v_deg else None,
                    slots, csv_rows, next_slot,
                )
                return status, new_slots, new_csv, new_next, active_tab, no_update, no_update, no_update
            else:
                status, new_slots, new_csv, new_next, active_tab, last_run = _run_single(
                    s_tomo, s_tmpl, s_mask, s_tight_mask, s_angles,
                    s_outdir, s_name, s_if_exists,
                    s_starting_angle,
                    int(s_sym or 1), int(s_cc or 10), float(s_deg) if s_deg else None,
                    bool(s_compute_dist), bool(s_compute_peak),
                    bool(s_apply_wedge), sw_target, sw_tmpl,
                    bool(s_compute_gradual), int(s_angular_range or 359),
                    slots, csv_rows, next_slot,
                )
                recompute_disabled = not last_run
                gradual_disabled = not (last_run and last_run.get("has_gradual"))
                return status, new_slots, new_csv, new_next, active_tab, last_run, recompute_disabled, gradual_disabled
        except Exception as exc:
            return f"Error: {exc}", no_update, no_update, no_update, no_update, no_update, no_update, no_update

    def _run_single(
        tomo, tmpl, mask, tight_mask, angles, outdir, name, if_exists,
        starting_angle_str, sym, cc, deg, compute_dist, compute_peak,
        apply_wedge, sw_target, sw_tmpl, compute_gradual, angular_range,
        slots, csv_rows, next_slot,
    ):
        missing = [f for f, v in [("Target map", tomo), ("Template", tmpl), ("Mask", mask)] if not v]
        if not angles:
            missing.append("Angles file")
        if not outdir:
            missing.append("Output directory")
        if not name:
            missing.append("Case name")
        if missing:
            return f"Missing required fields: {', '.join(missing)}.", no_update, no_update, no_update, no_update, {}

        shared = dict(
            target_map=tomo,
            template=tmpl,
            template_mask=mask,
            input_angles=angles,
            output_dir=outdir,
            case_name=name,
            starting_angle=_parse_starting_angle(starting_angle_str),
            cyclic_symmetry=sym,
            cc_radius=cc,
            degrees=deg,
            compute_distance_map=compute_dist,
            compute_peak_stats=compute_peak,
            if_exists=if_exists or "overwrite",
        )
        if apply_wedge:
            if sw_target and str(sw_target).strip():
                shared["wedge_mask_target"] = sw_target
            if sw_tmpl and str(sw_tmpl).strip():
                shared["wedge_mask_tmpl"] = sw_tmpl
        if tight_mask and str(tight_mask).strip():
            shared["tight_mask"] = tight_mask

        result = run_operation(pana.run_single_case, shared)
        write_dir = str(result.get("write_dir", Path(outdir) / name))

        if compute_gradual:
            gradual_shared = dict(
                target_map=tomo,
                template=tmpl,
                template_mask=mask,
                starting_angle=_parse_starting_angle(starting_angle_str),
                cyclic_symmetry=sym,
                cc_radius=cc,
                angular_range=angular_range,
            )
            if apply_wedge:
                if sw_target and str(sw_target).strip():
                    gradual_shared["wedge_mask_target"] = sw_target
                if sw_tmpl and str(sw_tmpl).strip():
                    gradual_shared["wedge_mask_tmpl"] = sw_tmpl
            final_df, hist_df = run_operation(pana.run_single_gradual_case, gradual_shared)
            p = Path(write_dir)
            final_df.to_csv(str(p / "gradual_angles_analysis.csv"), index=False)
            hist_df.to_csv(str(p / "gradual_angles_histograms.csv"), index=False)

        html_content = _load_or_generate_summary_html(write_dir)

        new_slots, new_next, active_tab = _assign_slot(slots, next_slot, html_content)

        csv_row = _file_row(tomo, tmpl, mask, angles, outdir, name, write_dir=write_dir)
        new_csv_rows = csv_rows + [csv_row]

        last_run = {
            "write_dir": write_dir,
            "degrees": deg,
            "cc_radius": cc,
            "slot": next_slot,
            "has_gradual": compute_gradual,
            "angular_range": angular_range,
            "tomo": tomo,
            "tmpl": tmpl,
            "mask": mask,
            "starting_angle_str": starting_angle_str,
            "sym": sym,
            "apply_wedge": apply_wedge,
            "sw_target": sw_target,
            "sw_tmpl": sw_tmpl,
        }
        status = f"Done. Results in slot {next_slot}."
        return status, new_slots, new_csv_rows, new_next, active_tab, last_run

    def _run_visualize(case_dir, compute_dist, cc, compute_peak, deg, slots, csv_rows, next_slot):
        if not case_dir or not os.path.isdir(case_dir):
            return "Case directory not found.", no_update, no_update, no_update, no_update

        p = Path(case_dir)

        def _exists(fname):
            return str(p / fname) if (p / fname).exists() else None

        if compute_dist and _exists("angles.em") and _exists("angles.csv"):
            pana.compute_distance_map(
                angles_map=str(p / "angles.em"),
                angles_list=str(p / "angles.csv"),
                output_dir=str(p),
            )
        if compute_peak and _exists("scores.em"):
            da = _exists("distance_map_all.em")
            dn = _exists("distance_map_normals.em")
            di = _exists("distance_map_inplane.em")
            if da and dn and di and deg is not None:
                pana.compute_peak_stats(
                    scores_map=str(p / "scores.em"),
                    dist_all_map=da,
                    dist_normals_map=dn,
                    dist_inplane_map=di,
                    degrees=deg,
                    output_dir=str(p),
                )

        figs = pana.visualize_results(
            scores=_exists("scores.em"),
            angles_map=_exists("angles.em"),
            angles_list=_exists("angles.csv"),
            dist_all_map=_exists("distance_map_all.em"),
            dist_normals_map=_exists("distance_map_normals.em"),
            dist_inplane_map=_exists("distance_map_inplane.em"),
            peak_stats=_exists("stats.json"),
        )
        run_params = {"Case directory": case_dir}
        try:
            html_content = _load_or_generate_summary_html(case_dir, figs or None, run_params)
        except FileNotFoundError:
            if not figs:
                return "No recognised artifacts found — nothing to visualize.", no_update, no_update, no_update, no_update
            html_content = _make_summary_html(figs, run_params)

        new_slots, new_next, active_tab = _assign_slot(slots, next_slot, html_content)

        csv_row = _file_row(None, None, None, None, case_dir, Path(case_dir).name, write_dir=case_dir)
        csv_row["Status"] = "Visualized"
        new_csv_rows = csv_rows + [csv_row]

        status = f"{len(figs)} panel(s) ready. Results in slot {next_slot}."
        return status, new_slots, new_csv_rows, new_next, active_tab

    # ── populate slot iframes from store ─────────────────────────────────────
    @app.callback(
        [Output(f"ppana-slot-{i}", "children") for i in range(1, _N_SLOTS + 1)],
        Input("ppana-slots-store", "data"),
    )
    def _update_slots(slots):
        slots = slots or {}
        outputs = []
        for i in range(1, _N_SLOTS + 1):
            html_content = slots.get(str(i), "")
            if html_content:
                outputs.append(
                    html.Iframe(
                        srcDoc=html_content,
                        style={"width": "100%", "height": "calc(100vh - 85px)", "border": "none"},
                    )
                )
            else:
                outputs.append(_slot_placeholder(i))
        return outputs

    # ── populate csv table from store ────────────────────────────────────────
    @app.callback(
        Output("ppana-csv-table", "columns"),
        Output("ppana-csv-table", "data"),
        Output("ppana-csv-table", "tooltip_data"),
        Input("ppana-csv-rows-store", "data"),
    )
    def _update_csv_table(csv_rows):
        if not csv_rows:
            return [], [], []
        # Only display the visible columns; hidden keys like _write_dir stay in data
        cols = _table_columns()
        data = [{c: row.get(c, "") for c in _TABLE_DISPLAY_COLS} for row in csv_rows]
        tooltips = [
            {c: {"value": str(row.get(c, "")), "type": "markdown"} for c in _TABLE_DISPLAY_COLS}
            for row in csv_rows
        ]
        return cols, data, tooltips

    # ── load template-list CSV into table ────────────────────────────────────
    @app.callback(
        Output("ppana-csv-rows-store", "data", allow_duplicate=True),
        Output("ppana-v-csv-status", "children"),
        Input("ppana-v-load-csv-btn", "n_clicks"),
        State("ppana-v-csv-path", "value"),
        State("ppana-v-csv-parent", "value"),
        State("ppana-v-csv-mode", "value"),
        State("ppana-csv-rows-store", "data"),
        prevent_initial_call=True,
    )
    def _load_csv_to_table(n_clicks, csv_path, parent, mode, existing_rows):
        if not n_clicks:
            raise PreventUpdate
        if not csv_path or not os.path.isfile(csv_path):
            return no_update, "CSV file not found."
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            return no_update, f"Could not read CSV: {exc}"

        def _fname(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return ""
            return Path(str(v)).name

        def _str(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return ""
            return str(v)

        new_rows = []
        for idx, row in df.iterrows():
            # Construct write_dir using pana path logic if parent is provided
            write_dir = ""
            if parent:
                try:
                    structure = _str(row.get("Structure"))
                    folder_spec = row.get("Output folder")
                    if folder_spec is not None:
                        write_dir = pana.create_output_folder_path(parent, structure, folder_spec)
                except Exception:
                    pass

            new_rows.append({
                "Template":   _fname(row.get("Template")),
                "Target map": _fname(row.get("Target map")),
                "Mask":       _fname(row.get("Mask")),
                "Case name":  _str(row.get("Structure")),
                "Output dir": _str(row.get("Output folder")),
                "Status":     "Done" if row.get("Done") else "Pending",
                "_write_dir": write_dir,
                "_csv_path":  csv_path,
                "_csv_index": int(idx),
            })

        existing = list(existing_rows or [])
        if mode == "append":
            combined = existing + new_rows
        else:
            combined = new_rows

        return combined, f"Loaded {len(new_rows)} row(s) from {Path(csv_path).name}."

    # ── visualize selected CSV row ────────────────────────────────────────────
    @app.callback(
        Output("ppana-slots-store", "data", allow_duplicate=True),
        Output("ppana-next-slot", "data", allow_duplicate=True),
        Output("ppana-main-tabs", "value", allow_duplicate=True),
        Output("ppana-csv-vis-status", "children"),
        Input("ppana-csv-visualize-btn", "n_clicks"),
        State("ppana-csv-table", "selected_rows"),
        State("ppana-csv-rows-store", "data"),
        State("ppana-slots-store", "data"),
        State("ppana-next-slot", "data"),
        prevent_initial_call=True,
    )
    def _visualize_csv_row(n_clicks, selected_rows, csv_rows, slots, next_slot):
        if not n_clicks:
            raise PreventUpdate
        if not selected_rows or not csv_rows:
            return no_update, no_update, no_update, "Select a row first."

        row = csv_rows[selected_rows[0]]
        write_dir = row.get("_write_dir", "")
        slots = slots or {}
        next_slot = int(next_slot or 1)

        if not write_dir or not os.path.isdir(write_dir):
            return no_update, no_update, no_update, (
                f"Output directory not found: {write_dir or '(none)'}"
            )

        p = Path(write_dir)

        def _exists(fname):
            return str(p / fname) if (p / fname).exists() else None

        figs = pana.visualize_results(
            scores=_exists("scores.em"),
            angles_map=_exists("angles.em"),
            angles_list=_exists("angles.csv"),
            dist_all_map=_exists("distance_map_all.em"),
            dist_normals_map=_exists("distance_map_normals.em"),
            dist_inplane_map=_exists("distance_map_inplane.em"),
            peak_stats=_exists("stats.json"),
        )
        if not figs:
            # Try batch naming convention: id_N_scores.em etc.
            csv_idx = row.get("_csv_index")
            if csv_idx is not None:
                base = f"id_{csv_idx}"
                figs = pana.visualize_results(
                    scores=_exists(f"{base}_scores.em"),
                    dist_all_map=_exists(f"{base}_angles_dist_all.em"),
                    dist_normals_map=_exists(f"{base}_angles_dist_normals.em"),
                    dist_inplane_map=_exists(f"{base}_angles_dist_inplane.em"),
                    peak_stats=_exists(f"{base}_stats.json"),
                )

        run_params = {
            "Template":   row.get("Template", ""),
            "Target map": row.get("Target map", ""),
            "Case name":  row.get("Case name", ""),
            "Output dir": write_dir,
        }
        try:
            html_content = _load_or_generate_summary_html(write_dir, figs or None, run_params)
        except FileNotFoundError:
            if not figs:
                return no_update, no_update, no_update, "No recognised artifacts found in output directory."
            html_content = _make_summary_html(figs, run_params)

        new_slots, new_next, active_tab = _assign_slot(slots, next_slot, html_content)
        return new_slots, new_next, active_tab, f"Opened in slot {next_slot}."

    # ── recompute peak statistics ────────────────────────────────────────────
    @app.callback(
        Output("ppana-status", "children", allow_duplicate=True),
        Output("ppana-slots-store", "data", allow_duplicate=True),
        Output("ppana-main-tabs", "value", allow_duplicate=True),
        Input("ppana-s-recompute-btn", "n_clicks"),
        State("ppana-s-last-run", "data"),
        State("ppana-s-degrees", "value"),
        State("ppana-s-cc-radius", "value"),
        State("ppana-slots-store", "data"),
        prevent_initial_call=True,
    )
    def _recompute_peak_stats(n_clicks, last_run, s_deg, s_cc, slots):
        if not n_clicks:
            raise PreventUpdate
        if not last_run or not last_run.get("write_dir"):
            return "No previous run found — run a single case first.", no_update, no_update

        write_dir = last_run["write_dir"]
        deg = float(s_deg) if s_deg is not None else last_run.get("degrees")
        cc = int(s_cc or last_run.get("cc_radius") or 10)
        target_slot = int(last_run.get("slot", 1))
        p = Path(write_dir)

        if deg is None:
            return "Degrees threshold required for peak statistics.", no_update, no_update

        def _e(fname):
            path = p / fname
            return str(path) if path.exists() else None

        scores = _e("scores.em")
        da = _e("distance_map_all.em")
        dn = _e("distance_map_normals.em")
        di = _e("distance_map_inplane.em")
        if not scores:
            return f"scores.em not found in {write_dir}.", no_update, no_update
        if not (da and dn and di):
            return "Distance maps not found — run with Compute distance map enabled first.", no_update, no_update

        try:
            pana.compute_peak_stats(
                scores_map=scores,
                dist_all_map=da,
                dist_normals_map=dn,
                dist_inplane_map=di,
                degrees=deg,
                cc_radius=cc,
                output_dir=write_dir,
            )
        except Exception as exc:
            return f"Error recomputing peak stats: {exc}", no_update, no_update

        params_path = p / "params.csv"
        if params_path.exists():
            try:
                params_df = pd.read_csv(str(params_path))
                row_dict = params_df.iloc[0].to_dict()
                row_dict["Degrees"] = deg
                new_results = pana.gather_case_results(
                    scores_map=scores,
                    dist_all_map=da,
                    dist_normals_map=dn,
                    dist_inplane_map=di,
                    degrees=deg,
                    cc_radius=cc,
                )
                new_record = pana.build_params_record(row_dict, new_results)
                new_record.to_csv(str(params_path), index=False)
            except Exception as exc:
                return f"Peak stats recomputed but params.csv update failed: {exc}", no_update, no_update

        try:
            html_content = _load_or_generate_summary_html(write_dir)
        except Exception as exc:
            return f"Recomputed but could not reload HTML: {exc}", no_update, no_update

        # update the original slot in-place — do not advance next_slot
        updated_slots = dict(slots or {})
        updated_slots[str(target_slot)] = html_content
        active_tab = f"tab-r{target_slot}"
        return f"Peak statistics recomputed (slot {target_slot}).", updated_slots, active_tab

    # ── recompute angular histograms ─────────────────────────────────────────
    @app.callback(
        Output("ppana-status", "children", allow_duplicate=True),
        Output("ppana-slots-store", "data", allow_duplicate=True),
        Output("ppana-main-tabs", "value", allow_duplicate=True),
        Input("ppana-s-recompute-gradual-btn", "n_clicks"),
        State("ppana-s-last-run", "data"),
        State("ppana-s-angular-range", "value"),
        State("ppana-slots-store", "data"),
        prevent_initial_call=True,
    )
    def _recompute_gradual(n_clicks, last_run, s_angular_range, slots):
        if not n_clicks:
            raise PreventUpdate
        if not last_run or not last_run.get("write_dir"):
            return "No previous run found — run a single case first.", no_update, no_update

        write_dir = last_run["write_dir"]
        target_slot = int(last_run.get("slot", 1))
        angular_range = int(s_angular_range or last_run.get("angular_range") or 359)
        tomo = last_run.get("tomo")
        tmpl = last_run.get("tmpl")
        mask = last_run.get("mask")

        if not (tomo and tmpl and mask):
            return "Source maps not recorded — run a single case first.", no_update, no_update

        gradual_shared = dict(
            target_map=tomo,
            template=tmpl,
            template_mask=mask,
            starting_angle=_parse_starting_angle(last_run.get("starting_angle_str")),
            cyclic_symmetry=int(last_run.get("sym") or 1),
            cc_radius=int(last_run.get("cc_radius") or 10),
            angular_range=angular_range,
        )
        if last_run.get("apply_wedge"):
            if last_run.get("sw_target"):
                gradual_shared["wedge_mask_target"] = last_run["sw_target"]
            if last_run.get("sw_tmpl"):
                gradual_shared["wedge_mask_tmpl"] = last_run["sw_tmpl"]

        try:
            final_df, hist_df = run_operation(pana.run_single_gradual_case, gradual_shared)
        except Exception as exc:
            return f"Error recomputing angular histograms: {exc}", no_update, no_update

        p = Path(write_dir)
        final_df.to_csv(str(p / "gradual_angles_analysis.csv"), index=False)
        hist_df.to_csv(str(p / "gradual_angles_histograms.csv"), index=False)

        try:
            html_content = _load_or_generate_summary_html(write_dir)
        except Exception as exc:
            return f"Histograms written but could not reload HTML: {exc}", no_update, no_update

        updated_slots = dict(slots or {})
        updated_slots[str(target_slot)] = html_content
        active_tab = f"tab-r{target_slot}"
        return f"Angular histograms recomputed (slot {target_slot}).", updated_slots, active_tab

    # ── generate script ──────────────────────────────────────────────────────
    @app.callback(
        Output("ppana-g-status", "children"),
        Input("ppana-g-generate-btn", "n_clicks"),
        State("ppana-v-csv-path", "value"),
        State("ppana-v-csv-parent", "value"),
        State("ppana-g-angle-path", "value"),
        State("ppana-g-wedge-path", "value"),
        State("ppana-g-cc-radius", "value"),
        State("ppana-g-format", "value"),
        State("ppana-g-sbatch", "value"),
        State("ppana-g-modules", "value"),
        State("ppana-g-save-path", "value"),
        prevent_initial_call=True,
    )
    def _generate_script(n_clicks, csv_path, parent, angle_path, wedge_path, cc,
                          fmt, sbatch_text, modules_text, save_path):
        if not n_clicks:
            raise PreventUpdate
        if not csv_path:
            return "Set the template list CSV path in the Visualize panel first."
        if not save_path or not str(save_path).strip():
            return "Specify a save path for the script."

        save_path = str(save_path).strip()
        if not save_path.endswith(".py"):
            save_path += ".py"

        kwargs: dict = {"template_list": csv_path}
        if parent:
            kwargs["parent_folder_path"] = parent
        if angle_path:
            kwargs["angle_list_path"] = angle_path
        if wedge_path:
            kwargs["wedge_path"] = wedge_path
        if cc is not None:
            kwargs["cc_radius_tol"] = int(cc)

        try:
            py_script = codegen.render_analysis_py(kwargs)
            with open(save_path, "w", encoding="utf-8") as fh:
                fh.write(py_script)

            if fmt == "slurm":
                slurm_path = save_path.replace(".py", ".sh")
                cluster_params = codegen._parse_sbatch_text(sbatch_text or "")
                module_loads = [m.strip() for m in (modules_text or "").splitlines() if m.strip()]
                slurm_script = codegen.render_slurm_wrapper(save_path, cluster_params, module_loads)
                with open(slurm_path, "w", encoding="utf-8") as fh:
                    fh.write(slurm_script)
                return f"Saved: {save_path} + {slurm_path}"

            return f"Saved: {save_path}"
        except Exception as exc:
            return f"Error: {exc}"
