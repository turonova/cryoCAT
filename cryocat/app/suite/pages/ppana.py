"""Peak Analysis (pana) tool.

Two run modes:

* **CSV mode** — point to an existing template-list CSV. The optional ID field
  picks a single row to process; if empty, all rows where ``Done`` is False are
  processed.
* **Single-instance mode** — fill in every CSV column as a form field; the page
  writes a one-row CSV to the user-specified output path and runs analysis on
  it. A motl from the suite pool can be selected to pre-fill ``Phi``/``Theta``/
  ``Psi`` and ``Tomogram`` from one of its rows: either the row matching the
  optional ``subtomo_id`` field, or the row with the highest ``score``.

Peak Analysis produces CSVs/PDFs, not motls — so this page does **not** use
``motlsink``. The table view used in the previous version was removed; the
sidebar holds all inputs and the main area only shows the run status.

Contract: exposes ``layout`` and ``register_callbacks(app)``.
"""

import os
from pathlib import Path

import dash
from dash import html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd

from cryocat.analysis import pana
from cryocat.app.components.motlsource import get_motl_source, register_motl_source_callbacks
from cryocat.app.components.logpanel import get_log_panel, register_log_panel_callbacks


# ── Form helpers ────────────────────────────────────────────────────────────────

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
_INPUT_WRAPPER_STYLE = {"flex": "1 1 auto", "minWidth": "0"}


def _label(text):
    return html.Label(text, style=_LABEL_STYLE)


def _text(label, comp_id, placeholder="", value=None):
    return html.Div(
        [
            _label(label),
            html.Div(
                dbc.Input(id=comp_id, type="text", placeholder=placeholder, value=value, size="sm"),
                style=_INPUT_WRAPPER_STYLE,
            ),
        ],
        style=_ROW_STYLE,
    )


def _number(label, comp_id, value=None, step=None, min_=None, placeholder=""):
    return html.Div(
        [
            _label(label),
            html.Div(
                dbc.Input(
                    id=comp_id, type="number", value=value, step=step, min=min_,
                    placeholder=placeholder, size="sm",
                ),
                style=_INPUT_WRAPPER_STYLE,
            ),
        ],
        style=_ROW_STYLE,
    )


def _check(label, comp_id, value=False):
    return html.Div(
        [
            _label(label),
            html.Div(
                dbc.Checkbox(id=comp_id, value=value),
                style=_INPUT_WRAPPER_STYLE,
            ),
        ],
        style=_ROW_STYLE,
    )


def _single_instance_form():
    return html.Div(
        [
            _text("Output CSV path", "pana-out-csv", "where the 1-row template list will be written"),
            html.Hr(style={"margin": "0.5rem 0"}),
            _text("Subtomo ID (optional)", "pana-motl-subtomo-id",
                  "blank = motl row with the highest score"),
            html.Hr(style={"margin": "0.5rem 0"}),
            _text("Structure", "pana-f-structure", "e.g. ribosome"),
            _text("Template", "pana-f-template", ".em filename"),
            _text("Mask", "pana-f-mask", ".em filename"),
            _text("Angles", "pana-f-angles", "angle list filename"),
            _text("Compare", "pana-f-compare", '"tmpl", "subtomo", or a structure name'),
            _text("Tomo map", "pana-f-tomo-map", ".em filename (used when Compare != tmpl)"),
            _text("Tomogram", "pana-f-tomogram",
                  "string containing the tomo number; auto-filled from motl"),
            _check("Apply wedge", "pana-f-apply-wedge", value=False),
            _number("Boxsize", "pana-f-boxsize", step=1, min_=1, placeholder="voxels"),
            _number("Binning", "pana-f-binning", step=1, min_=1, value=1, placeholder=""),
            _number("Phi", "pana-f-phi", step="any", value=0, placeholder="deg (auto-filled from motl)"),
            _number("Theta", "pana-f-theta", step="any", value=0, placeholder="deg (auto-filled from motl)"),
            _number("Psi", "pana-f-psi", step="any", value=0, placeholder="deg (auto-filled from motl)"),
            _check("Apply angular offset", "pana-f-apply-offset", value=False),
            _number("Degrees", "pana-f-degrees", step="any", value=10,
                    placeholder="search angle increment"),
            _text("Symmetry", "pana-f-symmetry", 'e.g. "C1"', value="C1"),
        ],
        id="pana-single-form",
    )


def _csv_mode_form():
    return html.Div(
        [
            _text("Template list CSV", "pana-template-list", "path to the template-list .csv"),
            _text("ID (optional)", "pana-indices",
                  "single row index; blank = all rows not flagged as Done"),
        ],
        id="pana-csv-form",
    )


# ── Layout ──────────────────────────────────────────────────────────────────────

def _sidebar():
    return dbc.Col(
        html.Div(
            [
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            dbc.RadioItems(
                                id="pana-mode",
                                options=[
                                    {"label": "CSV file", "value": "csv"},
                                    {"label": "Single instance", "value": "single"},
                                ],
                                value="csv",
                                inline=False,
                            ),
                            title="Input mode",
                            item_id="pana-acc-mode",
                        ),
                        dbc.AccordionItem(
                            _csv_mode_form(),
                            title="CSV file",
                            item_id="pana-acc-csv",
                        ),
                        dbc.AccordionItem(
                            [
                                get_motl_source("pana-src", multi=False),
                                _single_instance_form(),
                            ],
                            title="Single instance",
                            item_id="pana-acc-single",
                        ),
                        dbc.AccordionItem(
                            [
                                _text("Angle list path", "pana-angle-path",
                                      "directory with angle-list files"),
                                _text("Wedge path", "pana-wedge-path",
                                      "directory with wedge-mask files"),
                                _text("Parent folder path", "pana-parent-path",
                                      "root folder with structures/tomograms/masks"),
                                _number("CC radius tolerance (voxels)", "pana-cc-radius",
                                        value=10, step=1, min_=1),
                            ],
                            title="Common paths",
                            item_id="pana-acc-common",
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
                            ],
                            title="Run / Save",
                            item_id="pana-acc-run",
                        ),
                    ],
                    always_open=True,
                    active_item=["pana-acc-mode", "pana-acc-csv", "pana-acc-common", "pana-acc-run"],
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
                html.Div(
                    id="pana-run-status",
                    style={
                        "fontSize": "0.95rem",
                        "color": "var(--color9)",
                        "padding": "0.5rem",
                        "whiteSpace": "pre-wrap",
                        "wordBreak": "break-word",
                    },
                ),
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


# ── Run-mode helpers ────────────────────────────────────────────────────────────

_SINGLE_INSTANCE_COLUMNS = [
    "Structure", "Template", "Mask", "Angles", "Compare", "Tomo map", "Tomogram",
    "Apply wedge", "Boxsize", "Binning", "Phi", "Theta", "Psi",
    "Apply angular offset", "Degrees", "Symmetry",
    "Output folder", "Done",
]


def _csv_indices(template_list, index_str):
    """CSV mode: single index if given, else every row where ``Done`` is False."""
    df = pd.read_csv(template_list)
    if index_str and str(index_str).strip():
        return [int(str(index_str).strip())]
    if "Done" in df.columns:
        done = df["Done"].fillna(False).astype(bool)
        return df.index[~done].tolist()
    return df.index.tolist()


def _pick_motl_row(motl_rows, subtomo_id_str):
    """Pick one motl row by ``subtomo_id`` (blank => highest-score row)."""
    if not motl_rows:
        return None
    df = pd.DataFrame(motl_rows)
    if df.empty:
        return None
    if subtomo_id_str and str(subtomo_id_str).strip():
        sid = int(str(subtomo_id_str).strip())
        match = df[df.get("subtomo_id") == sid]
        if match.empty:
            raise ValueError(f"subtomo_id {sid} not found in selected motl.")
        return match.iloc[0]
    if "score" not in df.columns:
        return df.iloc[0]
    return df.loc[df["score"].idxmax()]


def _build_single_row(form, motl_row):
    """Assemble the single-instance row, letting ``motl_row`` override angles/tomogram."""
    row = dict(form)
    if motl_row is not None:
        for src, dst in (("phi", "Phi"), ("theta", "Theta"), ("psi", "Psi")):
            if src in motl_row.index and pd.notna(motl_row[src]):
                row[dst] = float(motl_row[src])
        if not row.get("Tomogram"):
            tid = motl_row.get("tomo_id")
            if pd.notna(tid):
                row["Tomogram"] = str(int(tid))
    row["Output folder"] = ""
    row["Done"] = False
    return row


# ── Callbacks ───────────────────────────────────────────────────────────────────

def register_callbacks(app):
    register_motl_source_callbacks(app, "pana-src", multi=False)
    register_log_panel_callbacks(app, "pana-log")

    @app.callback(
        Output("pana-csv-form", "style"),
        Output("pana-single-form", "style"),
        Output("pana-src-motl-source", "style"),
        Input("pana-mode", "value"),
    )
    def _toggle_mode(mode):
        show = {"display": "block"}
        hide = {"display": "none"}
        if mode == "single":
            return hide, show, show
        return show, hide, hide

    @app.callback(
        Output("pana-run-status", "children", allow_duplicate=True),
        Input("pana-run-btn", "n_clicks"),
        State("pana-mode", "value"),
        # CSV-mode state
        State("pana-template-list", "value"),
        State("pana-indices", "value"),
        # Single-instance state
        State("pana-out-csv", "value"),
        State("pana-src-motl-select", "value"),
        State("pool-motls", "data"),
        State("pana-motl-subtomo-id", "value"),
        State("pana-f-structure", "value"),
        State("pana-f-template", "value"),
        State("pana-f-mask", "value"),
        State("pana-f-angles", "value"),
        State("pana-f-compare", "value"),
        State("pana-f-tomo-map", "value"),
        State("pana-f-tomogram", "value"),
        State("pana-f-apply-wedge", "value"),
        State("pana-f-boxsize", "value"),
        State("pana-f-binning", "value"),
        State("pana-f-phi", "value"),
        State("pana-f-theta", "value"),
        State("pana-f-psi", "value"),
        State("pana-f-apply-offset", "value"),
        State("pana-f-degrees", "value"),
        State("pana-f-symmetry", "value"),
        # Common paths
        State("pana-angle-path", "value"),
        State("pana-wedge-path", "value"),
        State("pana-parent-path", "value"),
        State("pana-cc-radius", "value"),
        prevent_initial_call=True,
    )
    def run(n_clicks, mode, template_list, index_str,
            out_csv, motl_id, pool_motls, subtomo_id_str,
            f_structure, f_template, f_mask, f_angles, f_compare, f_tomo_map, f_tomogram,
            f_apply_wedge, f_boxsize, f_binning, f_phi, f_theta, f_psi,
            f_apply_offset, f_degrees, f_symmetry,
            angle_path, wedge_path, parent_path, cc_radius):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not (angle_path and wedge_path and parent_path):
            return "Provide the angle / wedge / parent folder paths."

        try:
            if mode == "csv":
                if not template_list:
                    return "Provide the template list CSV path."
                indices = _csv_indices(template_list, index_str)
                if not indices:
                    return "Nothing to run — every row is already flagged as Done."
                csv_to_use = template_list
            else:
                if not out_csv:
                    return "Provide the output CSV path for single-instance mode."
                pool_motls = pool_motls or {}
                motl_rows = pool_motls.get(motl_id) if motl_id else None
                motl_row = _pick_motl_row(motl_rows, subtomo_id_str)

                form = {
                    "Structure": f_structure or "",
                    "Template": f_template or "",
                    "Mask": f_mask or "",
                    "Angles": f_angles or "",
                    "Compare": f_compare or "",
                    "Tomo map": f_tomo_map or "",
                    "Tomogram": f_tomogram or "",
                    "Apply wedge": bool(f_apply_wedge),
                    "Boxsize": int(f_boxsize) if f_boxsize is not None else 0,
                    "Binning": float(f_binning) if f_binning is not None else 1.0,
                    "Phi": float(f_phi) if f_phi is not None else 0.0,
                    "Theta": float(f_theta) if f_theta is not None else 0.0,
                    "Psi": float(f_psi) if f_psi is not None else 0.0,
                    "Apply angular offset": bool(f_apply_offset),
                    "Degrees": float(f_degrees) if f_degrees is not None else 0.0,
                    "Symmetry": f_symmetry or "C1",
                }
                row = _build_single_row(form, motl_row)
                df = pd.DataFrame([row], columns=_SINGLE_INSTANCE_COLUMNS)
                Path(os.path.dirname(out_csv) or ".").mkdir(parents=True, exist_ok=True)
                df.to_csv(out_csv, index=True)
                indices = [0]
                csv_to_use = out_csv

            pana.run_analysis(
                csv_to_use,
                indices,
                angle_path,
                wedge_path,
                parent_path,
                cc_radius_tol=cc_radius or 10,
            )
            return f"Analysis complete - processed {len(indices)} row(s) of {csv_to_use}."
        except Exception as exc:
            return f"Error: {exc}"

    @app.callback(
        Output("pana-run-status", "children", allow_duplicate=True),
        Input("pana-summary-save-btn", "n_clicks"),
        State("pana-mode", "value"),
        State("pana-template-list", "value"),
        State("pana-indices", "value"),
        State("pana-out-csv", "value"),
        State("pana-parent-path", "value"),
        prevent_initial_call=True,
    )
    def save_summary(n_clicks, mode, template_list, index_str, out_csv, parent_path):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not parent_path:
            return "Provide the parent folder path."
        try:
            if mode == "csv":
                if not template_list:
                    return "Provide the template list CSV path."
                indices = _csv_indices(template_list, index_str)
                csv_to_use = template_list
            else:
                if not out_csv:
                    return "Provide the output CSV path for single-instance mode."
                indices = [0]
                csv_to_use = out_csv
            pana.create_summary_pdf(csv_to_use, indices, parent_path)
            return f"Summary PDF written for {len(indices)} row(s) of {csv_to_use}."
        except Exception as exc:
            return f"Error: {exc}"
