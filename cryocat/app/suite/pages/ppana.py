"""Peak Analysis (pana) tool.

Three run modes:

* **Single case** — fill in paths for tomogram/template/mask/angles and an
  output directory; runs :func:`~cryocat.analysis.pana.run_single_case` then
  shows the result panels in the main area.
* **Visualize existing** — point to a case directory that already contains
  ``scores.em``, ``angles.em``, ``stats.json`` etc. and call
  :func:`~cryocat.analysis.pana.visualize_results` on whatever artifacts are
  present.
* **CSV batch** — point to an existing template-list CSV and call
  :func:`~cryocat.analysis.pana.run_analysis` over the selected indices.

Contract: exposes ``layout`` and ``register_callbacks(app)``.
"""

from __future__ import annotations

import os
from pathlib import Path
import json as _json

import dash
from dash import html, dcc, Input, Output, State, no_update, ctx, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd

from cryocat.analysis import pana
from cryocat.app import formgen
from cryocat.app.apputils import run_operation, generate_kwargs
from cryocat.app.components.logpanel import get_log_panel, register_log_panel_callbacks
from cryocat.app.components.anglesfield import get_angles_field, register_angles_field_callbacks
from cryocat.utils.wedgeutils import generate_wedge_mask


# ── Form helpers ─────────────────────────────────────────────────────────────

_ROW_STYLE = {
    "display": "flex",
    "alignItems": "center",
    "gap": "0.5rem",
    "marginBottom": "0.35rem",
}
_LABEL_STYLE = {
    "fontSize": "0.85rem",
    "flex": "0 0 45%",
    "marginBottom": "0",
}
_INPUT_WRAPPER = {"flex": "1 1 auto", "minWidth": "0"}
_HINT = {"fontSize": "0.8rem", "color": "var(--color9)"}

_COLLAPSE_INNER = {
    "backgroundColor": "rgba(var(--bs-secondary-rgb), 0.05)",
    "borderRadius": "0.25rem",
    "padding": "0.5rem",
    "marginBottom": "0.35rem",
}

_WEDGE_ID_TYPE = "ppana-wedge-param"
_WEDGE_BUILDER = "ppana-wedge"


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
    """Modal for generating a single wedge mask with use-as buttons."""
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
                    dbc.Button(
                        "Generate mask",
                        id="ppana-wedge-generate-btn",
                        color="secondary",
                        size="sm",
                        style={"width": "100%", "marginBottom": "0.4rem"},
                    ),
                    html.Div(
                        id="ppana-wedge-modal-status",
                        style={**_HINT, "wordBreak": "break-word"},
                    ),
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
                        _text(
                            "Output path (optional)",
                            "ppana-sub-output",
                            "blank → <output dir>/<case name>/subtomogram.em",
                        ),
                        html.Div(
                            dbc.Button(
                                "Extract and use this file",
                                id="ppana-sub-btn",
                                color="secondary",
                                size="sm",
                                style={"width": "100%", "marginTop": "0.3rem"},
                            ),
                        ),
                        html.Div(
                            id="ppana-sub-status",
                            style={**_HINT, "marginTop": "0.3rem", "wordBreak": "break-word"},
                        ),
                    ],
                    style=_COLLAPSE_INNER,
                ),
                id="ppana-s-subtomo-collapse",
                is_open=False,
            ),
            # ─ Wedge mask (optional) — sits here so it's above the file paths ─
            _check("Apply wedge mask", "ppana-s-apply-wedge"),
            dbc.Collapse(
                html.Div(
                    [
                        html.Small(
                            "Provide pre-computed wedge masks, or generate one via the button below.",
                            style={**_HINT, "marginBottom": "0.3rem", "display": "block"},
                        ),
                        _text("Wedge mask — target (.em)", "ppana-sw-target", "path/to/wedge_mask_target.em"),
                        _text("Wedge mask — template (.em)", "ppana-sw-tmpl", "path/to/wedge_mask_tmpl.em"),
                        html.Div(
                            dbc.Button(
                                "Generate wedge masks…",
                                id="ppana-wedge-open-btn",
                                color="secondary",
                                size="sm",
                                style={"width": "100%", "marginTop": "0.3rem"},
                            ),
                        ),
                    ],
                    style=_COLLAPSE_INNER,
                ),
                id="ppana-s-wedge-collapse",
                is_open=False,
            ),
            # ─ Map file paths ────────────────────────────────────────────────
            html.Small("Paths to map files", style={**_HINT, "marginBottom": "0.3rem", "display": "block"}),
            _text("Target map (.em)", "ppana-s-tomogram", "path/to/tomo.em"),
            _text("Template (.em)", "ppana-s-template", "path/to/template.em"),
            _text("Template mask (.em)", "ppana-s-mask", "path/to/mask.em"),
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
            _text(
                "Starting angles (φ,θ,ψ)",
                "ppana-s-starting-angle",
                "e.g. 0, 0, 0  (leave blank for default)",
            ),
            _number("Cyclic symmetry", "ppana-s-symmetry", value=1, step=1, min_=1),
            _number("CC radius (voxels)", "ppana-s-cc-radius", value=10, step=1, min_=1),
            _number("Degrees threshold", "ppana-s-degrees", placeholder="leave blank to skip peak stats", step="any"),
            _check("Compute distance map", "ppana-s-compute-dist", value=True),
            _check("Compute peak stats", "ppana-s-compute-peak", value=True),
            html.Hr(style={"margin": "0.4rem 0"}),
            dbc.Button(
                "Run",
                id="ppana-s-run-btn",
                color="primary",
                size="sm",
                style={"width": "100%"},
            ),
        ],
        id="ppana-single-form",
    )


def _visualize_form():
    return html.Div(
        [
            html.Small(
                "Point to a directory produced by run_single_case (contains scores.em, angles.em, etc.).",
                style={**_HINT, "marginBottom": "0.3rem", "display": "block"},
            ),
            _text("Case directory", "ppana-v-case-dir", "path/to/case_name/"),
            html.Hr(style={"margin": "0.4rem 0"}),
            html.Small(
                "Optional: compute missing artifacts first",
                style={**_HINT, "marginBottom": "0.3rem", "display": "block"},
            ),
            _check("Compute distance map (needs angles.em + angles.csv)", "ppana-v-compute-dist", value=False),
            _number("  CC radius (voxels)", "ppana-v-cc-radius", value=10, step=1, min_=1),
            _check("Compute peak stats (needs scores.em)", "ppana-v-compute-peak", value=False),
            _number("  Degrees threshold", "ppana-v-degrees", placeholder="required for peak stats", step="any"),
            html.Div(
                id="ppana-v-artifacts",
                style={**_HINT, "marginTop": "0.4rem", "wordBreak": "break-word"},
            ),
            html.Hr(style={"margin": "0.4rem 0"}),
            dbc.Button(
                "Visualize",
                id="ppana-v-run-btn",
                color="primary",
                size="sm",
                style={"width": "100%"},
            ),
        ],
        id="ppana-visualize-form",
    )


def _csv_batch_form():
    return html.Div(
        [
            _text("Template list CSV", "ppana-c-template-list", "path/to/template_list.csv"),
            _text("Parent folder path", "ppana-c-parent-path", "root folder containing structures/tomograms"),
            _text("Indices (optional)", "ppana-c-indices", "single row index; blank = all rows not flagged Done"),
            _text("Angle list folder", "ppana-c-angle-path", "directory with angle-list files"),
            _text("Wedge mask folder", "ppana-c-wedge-path", "directory with wedge-mask files"),
            _number("CC radius tolerance", "ppana-c-cc-radius", value=10, step=1, min_=1),
            html.Hr(style={"margin": "0.4rem 0"}),
            dbc.Button(
                "Run batch",
                id="ppana-c-run-btn",
                color="primary",
                size="sm",
                style={"width": "100%"},
            ),
        ],
        id="ppana-csv-form",
    )


# ── Layout ────────────────────────────────────────────────────────────────────

def _sidebar():
    return dbc.Col(
        html.Div(
            [
                dbc.Accordion(
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
                            _csv_batch_form(),
                            title="CSV batch",
                            item_id="ppana-acc-csv",
                        ),
                    ],
                    always_open=True,
                    active_item=["ppana-acc-single"],
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
                html.Div(
                    id="ppana-status",
                    style={
                        "fontSize": "0.95rem",
                        "color": "var(--color9)",
                        "padding": "0.5rem 0.5rem 0",
                        "whiteSpace": "pre-wrap",
                        "wordBreak": "break-word",
                    },
                ),
                html.Div(
                    [
                        dcc.Tabs(
                            id="ppana-tabs",
                            value=None,
                            children=[],
                            style={"marginBottom": "0.25rem"},
                        ),
                        dcc.Graph(
                            id="ppana-graph",
                            style={"height": "calc(100vh - 120px)", "width": "100%"},
                            config={"displayModeBar": True, "scrollZoom": True},
                        ),
                    ],
                    id="ppana-figure-area",
                    style={"display": "none", "padding": "0"},
                ),
                dcc.Store(id="ppana-figs-store"),
            ],
            style={"padding": "0.5rem 0.5rem 0 0.5rem"},
        ),
        width=9,
        style={"margin": "0", "padding": "0"},
    )


layout = html.Div(
    [
        dbc.Row([_sidebar(), _main()], className="g-0", style={"margin": "0", "padding": "0"}),
        _wedge_mask_modal(),
        *get_log_panel("ppana-log"),
    ],
    style={"margin": "0", "padding": "0"},
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _csv_indices(template_list: str, index_str) -> list[int]:
    df = pd.read_csv(template_list)
    if index_str and str(index_str).strip():
        return [int(str(index_str).strip())]
    if "Done" in df.columns:
        done = df["Done"].fillna(False).astype(bool)
        return df.index[~done].tolist()
    return df.index.tolist()


def _figs_to_store(figs: dict) -> dict:
    """Serialize a {name: go.Figure} dict for dcc.Store."""
    return {name: fig.to_plotly_json() for name, fig in figs.items()}


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
    def _generate_wedge_mask(n_clicks, params, out_path):
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

    # ── scan case dir for artifacts (visualize mode) ─────────────────────────
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

    # ── main run dispatch (three independent run buttons) ────────────────────
    @app.callback(
        Output("ppana-status", "children"),
        Output("ppana-figs-store", "data"),
        Input("ppana-s-run-btn", "n_clicks"),
        Input("ppana-v-run-btn", "n_clicks"),
        Input("ppana-c-run-btn", "n_clicks"),
        # single-case states
        State("ppana-s-tomogram", "value"),
        State("ppana-s-template", "value"),
        State("ppana-s-mask", "value"),
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
        # visualize states
        State("ppana-v-case-dir", "value"),
        State("ppana-v-compute-dist", "value"),
        State("ppana-v-cc-radius", "value"),
        State("ppana-v-compute-peak", "value"),
        State("ppana-v-degrees", "value"),
        # csv-batch states
        State("ppana-c-template-list", "value"),
        State("ppana-c-parent-path", "value"),
        State("ppana-c-indices", "value"),
        State("ppana-c-angle-path", "value"),
        State("ppana-c-wedge-path", "value"),
        State("ppana-c-cc-radius", "value"),
        prevent_initial_call=True,
    )
    def _run(
        n_single, n_vis, n_csv,
        s_tomo, s_tmpl, s_mask, s_angles, s_outdir, s_name, s_if_exists,
        s_starting_angle, s_sym, s_cc, s_deg, s_compute_dist, s_compute_peak,
        s_apply_wedge, sw_target, sw_tmpl,
        v_case_dir, v_compute_dist, v_cc, v_compute_peak, v_deg,
        c_tlist, c_parent, c_indices, c_angle_path, c_wedge_path, c_cc,
    ):
        triggered = ctx.triggered_id
        if not triggered:
            raise PreventUpdate
        try:
            if triggered == "ppana-c-run-btn":
                return _run_csv(c_tlist, c_parent, c_indices, c_angle_path, c_wedge_path, int(c_cc or 10))
            elif triggered == "ppana-v-run-btn":
                return _run_visualize(
                    v_case_dir, bool(v_compute_dist), int(v_cc or 10),
                    bool(v_compute_peak), float(v_deg) if v_deg else None,
                )
            else:  # ppana-s-run-btn
                return _run_single(
                    s_tomo, s_tmpl, s_mask, s_angles, s_outdir, s_name, s_if_exists,
                    s_starting_angle,
                    int(s_sym or 1), int(s_cc or 10), float(s_deg) if s_deg else None,
                    bool(s_compute_dist), bool(s_compute_peak),
                    bool(s_apply_wedge), sw_target, sw_tmpl,
                )
        except Exception as exc:
            return f"Error: {exc}", no_update

    def _run_single(
        tomo, tmpl, mask, angles, outdir, name, if_exists,
        starting_angle_str,
        sym, cc, deg, compute_dist, compute_peak,
        apply_wedge=False, sw_target=None, sw_tmpl=None,
    ):
        missing = [f for f, v in [("Target map", tomo), ("Template", tmpl), ("Mask", mask)] if not v]
        if not angles:
            missing.append("Angles file")
        if not outdir:
            missing.append("Output directory")
        if not name:
            missing.append("Case name")
        if missing:
            return f"Missing required fields: {', '.join(missing)}.", no_update

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

        result = run_operation(pana.run_single_case, shared)

        figs = pana.visualize_results(
            scores=result.get("scores_map"),
            angles_map=result.get("angles_map"),
            dist_all_map=result.get("dist_all_map"),
            dist_normals_map=result.get("dist_normals_map"),
            dist_inplane_map=result.get("dist_inplane_map"),
            peak_stats=result.get("peak_stats"),
        )
        status = f"Done — {len(figs)} panel(s) ready. Results in: {result.get('write_dir', outdir)}"
        return status, _figs_to_store(figs)

    def _run_visualize(case_dir, compute_dist, cc, compute_peak, deg):
        if not case_dir or not os.path.isdir(case_dir):
            return "Case directory not found.", no_update

        p = Path(case_dir)

        def _exists(name):
            return str(p / name) if (p / name).exists() else None

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
        if not figs:
            return "No recognized artifacts found — nothing to visualize.", no_update
        return f"{len(figs)} panel(s) ready.", _figs_to_store(figs)

    def _run_csv(tlist, parent, index_str, angle_path, wedge_path, cc):
        if not tlist:
            return "Provide the template list CSV path.", no_update
        if not parent:
            return "Provide the parent folder path.", no_update
        if not (angle_path and wedge_path):
            return "Provide angle list and wedge mask folder paths.", no_update
        indices = _csv_indices(tlist, index_str)
        if not indices:
            return "Nothing to run — every row is already flagged as Done.", no_update
        run_operation(
            pana.run_analysis,
            dict(
                template_list=tlist,
                indices=indices,
                angle_list_path=angle_path,
                wedge_path=wedge_path,
                parent_folder_path=parent,
                cc_radius_tol=cc,
            ),
        )
        return f"CSV batch complete — processed {len(indices)} row(s).", no_update

    # ── populate tabs from store ─────────────────────────────────────────────
    @app.callback(
        Output("ppana-tabs", "children"),
        Output("ppana-tabs", "value"),
        Output("ppana-figure-area", "style"),
        Input("ppana-figs-store", "data"),
    )
    def _populate_tabs(store_data):
        if not store_data:
            return [], None, {"display": "none"}
        tabs = [dcc.Tab(label=_LABELS.get(k, k), value=k) for k in store_data]
        first = next(iter(store_data)) if store_data else None
        return tabs, first, {"display": "block", "padding": "0"}

    # ── figure display (tight margins, fill container) ───────────────────────
    @app.callback(
        Output("ppana-graph", "figure"),
        Input("ppana-tabs", "value"),
        State("ppana-figs-store", "data"),
        prevent_initial_call=True,
    )
    def _show_figure(panel, store_data):
        if not panel or not store_data or panel not in store_data:
            raise PreventUpdate
        import plotly.graph_objects as go
        fig = go.Figure(store_data[panel])
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0, pad=0),
            autosize=True,
        )
        return fig

    # ── zoom enforcement: minimum visible span = 10 pixels ──────────────────
    try:
        @app.callback(
            Output("ppana-graph", "figure", allow_duplicate=True),
            Input("ppana-graph", "relayoutData"),
            State("ppana-graph", "figure"),
            prevent_initial_call=True,
        )
        def _enforce_zoom(relayout_data, current_fig):
            if not relayout_data or not current_fig:
                raise PreventUpdate
            MIN_SPAN = 10
            layout = current_fig.get("layout", {})
            updated = False
            for axis in ("xaxis", "yaxis"):
                r0_key = f"{axis}.range[0]"
                r1_key = f"{axis}.range[1]"
                if r0_key in relayout_data and r1_key in relayout_data:
                    r0 = float(relayout_data[r0_key])
                    r1 = float(relayout_data[r1_key])
                    if abs(r1 - r0) < MIN_SPAN:
                        center = (r0 + r1) / 2
                        half = MIN_SPAN / 2
                        if axis not in layout:
                            layout[axis] = {}
                        layout[axis]["range"] = [center - half, center + half]
                        updated = True
            if not updated:
                raise PreventUpdate
            current_fig["layout"] = layout
            return current_fig
    except Exception:
        pass
