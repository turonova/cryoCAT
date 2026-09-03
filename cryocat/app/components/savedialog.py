"""Save dialog — one reusable component for pool-backed motl saves.

Owned stores
    {prefix}-value    SaveSpec as dict; updated on every form change
    {prefix}-prefill  pool meta dict set by caller to pre-populate fields
    {prefix}-motl-id  motl_id to save (single mode only, set by caller)

Published result id: {prefix}-status

IDs owned
    {prefix}-format            format dropdown
    {prefix}-writer-form       dynamic writer-params form (swaps on format change)
    {prefix}-rln-*             Relion options (via relionopts component)
    {prefix}-dest              destination path field  (single mode)
    {prefix}-dest-dir          destination directory   (batch mode)
    {prefix}-filename-policy   stem|suffix             (batch mode)
    {prefix}-filename-suffix   suffix text             (batch mode)
    {prefix}-overwrite         refuse|overwrite
    {prefix}-resolve-hint      resolved-path display   (single mode)
    {prefix}-validation        validation messages
    {prefix}-status            save status
    {prefix}-save-btn          action button
"""
from __future__ import annotations

import os
from typing import Literal

import dash
import pandas as pd
from dash import html, dcc, Input, Output, State, no_update, ALL, ctx
import dash_bootstrap_components as dbc

from cryocat.core.cryomotl import Motl, EmMotl, StopgapMotl, RelionMotl, RelionMotlv5, DynamoMotl
from cryocat.app.pool import get_rows, get_extra
from cryocat.app.apputils import run_operation, generate_kwargs
from cryocat.app.components.pathfield import get_path_field
from cryocat.app.components.relionopts import (
    get_relion_options,
    register_relion_options_callbacks,
    read_relion_kwargs,
)
from cryocat.app.formgen import make_dropdown, build_form, form_row, register_form_callbacks
from cryocat.app import styles


SAVE_FORMATS = [
    {"label": "EM (.em)", "value": "emmotl"},
    {"label": "STOPGAP (.star)", "value": "stopgap"},
    {"label": "Relion (.star)", "value": "relion"},
    {"label": "Dynamo (.tbl)", "value": "dynamo"},
]

EXTENSION_MAP = {
    "emmotl": ".em",
    "stopgap": ".star",
    "relion": ".star",
    "dynamo": ".tbl",
}

_OVERWRITE_OPTS = [
    {"label": "Refuse (list conflicts)", "value": "refuse"},
    {"label": "Overwrite existing", "value": "overwrite"},
]

_POLICY_OPTS = [
    {"label": "Keep stem + new extension", "value": "stem"},
    {"label": "Add suffix before extension", "value": "suffix"},
]

_DATA_TYPE_TO_FMT = {
    "emmotl": "emmotl", "stopgap": "stopgap", "dynamo": "dynamo",
    "relion": "relion", "relion5": "relion", "relion5_1": "relion",
}

# Params already handled elsewhere (motl itself, output path, or relionopts).
# Everything NOT in this exclude set is offered via the dynamic writer form.
_RELION_WRITER_EXCLUDE = (
    "output_path",
    "version",              # controls constructor routing; handled by relionopts
    "write_optics",         # derived from whether optics_data is loaded
    "pixel_size",           # handled by relionopts
    "binning",              # handled by relionopts
    "tomo_format",          # handled by relionopts
    "subtomo_format",       # handled by relionopts
    "use_original_entries", # handled by relionopts
    "subtomo_size",         # handled by relionopts
    "optics_data",          # complex DataFrame; handled internally
    "convert",              # v5 only; handled by relionopts
)


# ── Writer-form helpers ───────────────────────────────────────────────────────

def _writer_form_for(fmt: str, prefix: str) -> list:
    """Return build_form rows for the writer of *fmt*, or [] if no params to show.

    The form uses id_type ``{prefix}-writer-param`` so the save callback can
    collect all values with a single ALL-pattern State.
    """
    id_type = f"{prefix}-writer-param"
    if fmt == "stopgap":
        return build_form(StopgapMotl.write_out, id_type=id_type, exclude=("output_path",))
    if fmt == "relion":
        return build_form(RelionMotl.write_out, id_type=id_type, exclude=_RELION_WRITER_EXCLUDE)
    # emmotl and dynamo: write_out has only output_path — nothing to show.
    return []


# ── Pure helpers ─────────────────────────────────────────────────────────────

def resolve_extension(path: str, fmt: str) -> str:
    """Return *path* with the canonical extension for *fmt*."""
    stem = os.path.splitext(path)[0]
    return stem + EXTENSION_MAP.get(fmt, os.path.splitext(path)[1])


def validate_save(
    path: str | None,
    fmt: str | None,
    rln_value: dict | None,
    *,
    mode: str = "single",
    members: list | None = None,
    paths: dict | None = None,
    overwrite: str = "refuse",
) -> list[str]:
    """Return every problem as a human-readable string; empty → OK."""
    probs: list[str] = []
    if not fmt:
        probs.append("Select an output format.")
    if mode == "single":
        if not path:
            probs.append("Specify an output path.")
    else:
        if not path:
            probs.append("Specify an output directory.")
        if not members:
            probs.append("No group is active (click a group in the pool list).")
    if fmt == "relion":
        ver = float((rln_value or {}).get("version") or 0)
        if ver >= 5.0 and not (rln_value or {}).get("tomos"):
            probs.append("Relion 5.x save requires a tomogram file — load one in the Relion options.")
    if mode == "batch" and overwrite == "refuse" and paths:
        conflicts = [p for p in paths.values() if os.path.exists(p)]
        if conflicts:
            snippet = "\n".join(conflicts[:5])
            tail = f"\n… and {len(conflicts) - 5} more." if len(conflicts) > 5 else ""
            probs.append(f"{len(conflicts)} file(s) already exist and would be overwritten:\n{snippet}{tail}")
    return probs


def _build_motl(fmt: str, df, extra_df, rln_value: dict | None, writer_kwargs: dict) -> tuple:
    """Return (motl_instance, write_out_kwargs) for the given format."""
    if fmt == "emmotl":
        return EmMotl(df), {}

    if fmt == "stopgap":
        m = StopgapMotl(df)
        if extra_df is not None:
            src = pd.DataFrame(extra_df) if not isinstance(extra_df, pd.DataFrame) else extra_df
            m.sg_df = src
        return m, writer_kwargs

    if fmt == "dynamo":
        m = DynamoMotl(df)
        if extra_df is not None:
            src = pd.DataFrame(extra_df) if not isinstance(extra_df, pd.DataFrame) else extra_df
            m.dynamo_df = src
        return m, {}

    if fmt == "relion":
        rln = rln_value or {}
        ver = float(rln.get("version") or 3.1)
        px = rln.get("pixel_size") or 1.0
        bn = rln.get("binning") or 1
        optics_raw = rln.get("optics")
        optics = pd.DataFrame(optics_raw) if optics_raw else None
        tomos_raw = rln.get("tomos")
        write_kw: dict = {
            "tomo_format": rln.get("tomo_format") or "",
            "subtomo_format": rln.get("subtomo_format") or "",
            "use_original_entries": bool(rln.get("use_original")),
            "write_optics": optics is not None,
        }
        # Merge writer-form params (keep_all_entries, add_object_id, add_subunit_id).
        write_kw.update(writer_kwargs)
        if ver >= 5.0:
            tomo_df = pd.DataFrame(tomos_raw) if tomos_raw else None
            m = RelionMotlv5(df, input_tomograms=tomo_df, pixel_size=px, binning=bn, optics_data=optics)
            if rln.get("subtomo_size"):
                write_kw["subtomo_size"] = rln["subtomo_size"]
            if rln.get("convert"):
                write_kw["convert"] = True
        else:
            m = RelionMotl(df, version=ver, pixel_size=px, binning=bn, optics_data=optics)
        if extra_df is not None:
            src = pd.DataFrame(extra_df) if not isinstance(extra_df, pd.DataFrame) else extra_df
            m.relion_df = src
        return m, write_kw

    raise ValueError(f"Unknown save format {fmt!r}")


def _prefill_to_form_values(prefill: dict | None) -> tuple:
    """Pool meta dict → (fmt, version, pixel_size, binning)."""
    if not prefill:
        return "emmotl", no_update, no_update, no_update
    fmt = _DATA_TYPE_TO_FMT.get(prefill.get("data_type", "emmotl"), "emmotl")
    rln = prefill.get("relion_params") or {}
    ver = rln.get("version")
    px = rln.get("pixel_size")
    bn = rln.get("binning")
    return fmt, (ver if ver is not None else no_update), (px or no_update), (bn or no_update)


def execute_save_single(motl_id: str, path: str, fmt: str, rln_value: dict | None, writer_kwargs: dict) -> str:
    """Load motl from pool and write to *path*. Returns status string."""
    df = get_rows(motl_id)
    extra_df = get_extra(motl_id)
    resolved = resolve_extension(path, fmt)
    out_dir = os.path.dirname(resolved)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    m, write_kw = _build_motl(fmt, df, extra_df, rln_value, writer_kwargs)
    run_operation(m.write_out, {"output_path": resolved, **write_kw})
    return f"Saved {len(df)} particles → {resolved}"


def execute_batch_save(
    members: list[str],
    paths: dict[str, str],
    fmt: str,
    rln_value: dict | None,
    writer_kwargs: dict,
    registry: dict,
) -> tuple[str, str]:
    """Write each member to its path. Returns (status, validation) strings."""
    done, errs = 0, []
    for mid in members:
        out_path = paths.get(mid)
        if not out_path:
            continue
        try:
            df = get_rows(mid)
            extra_df = get_extra(mid)
            m, write_kw = _build_motl(fmt, df, extra_df, rln_value, writer_kwargs)
            run_operation(m.write_out, {"output_path": out_path, **write_kw})
            done += 1
        except Exception as exc:
            label = (registry.get(mid) or {}).get("label", mid)
            errs.append(f"{label}: {exc}")
    status = f"Saved {done}/{len(members)} motl(s) to {fmt} format."
    if errs:
        status += "  Errors: " + "; ".join(errs[:3])
        if len(errs) > 3:
            status += f" … (+{len(errs) - 3} more)"
    return status, ""


def build_batch_paths(
    members: list[str],
    out_dir: str,
    fmt: str,
    policy: str,
    suffix: str | None,
    registry: dict,
) -> dict[str, str]:
    """Map motl_id → output path for batch convert."""
    import pathlib
    ext = EXTENSION_MAP.get(fmt, ".em")
    paths: dict[str, str] = {}
    for mid in members:
        meta = registry.get(mid) or {}
        src = meta.get("source_path") or meta.get("label", mid)
        stem = pathlib.Path(src).stem if src else mid
        if policy == "suffix" and suffix:
            stem = stem + suffix
        paths[mid] = os.path.join(out_dir, stem + ext)
    return paths


# ── Layout ───────────────────────────────────────────────────────────────────

def get_save_dialog(prefix: str, *, mode: Literal["single", "batch"] = "single") -> html.Div:
    """Save dialog component.

    Single mode: saves one pool motl identified by {prefix}-motl-id store.
    Batch mode: converts all members of the active group; caller wires the
    save callback (register_save_dialog_callbacks skips it for batch).

    Writer parameters are rendered dynamically when the format changes —
    the same pattern as the clustering method form.  ``{prefix}-writer-form``
    is populated by the ``_on_format`` callback in register_save_dialog_callbacks.
    """
    format_row = form_row(
        "Output format",
        make_dropdown(f"{prefix}-format", SAVE_FORMATS, "emmotl"),
        "Select the file format to write.",
        label_id=f"{prefix}-lbl-format",
    )

    # Dynamic writer-params form: populated by callback when format changes.
    # Empty on load; for emmotl/dynamo it stays empty (no extra params).
    writer_form = html.Div(id=f"{prefix}-writer-form", children=[])

    relion_panel = get_relion_options(prefix, for_load=False)

    if mode == "single":
        dest_section = html.Div([
            form_row(
                "Output file",
                get_path_field(f"{prefix}-dest", mode="save", extensions=(".em", ".star", ".tbl")),
                "Full path to the output file. Extension is corrected to match the format.",
                label_id=f"{prefix}-lbl-dest",
            ),
            html.Div(
                id=f"{prefix}-resolve-hint",
                style={"fontSize": styles.FONT_TIGHT, "color": styles.COLOR_MUTED, "marginTop": "0.15rem"},
            ),
        ])
    else:
        dest_section = html.Div([
            form_row(
                "Output directory",
                get_path_field(f"{prefix}-dest-dir", mode="directory", kind="output"),
                "Directory where all converted files will be written.",
                label_id=f"{prefix}-lbl-dest-dir",
            ),
            form_row(
                "Filename",
                make_dropdown(f"{prefix}-filename-policy", _POLICY_OPTS, "stem"),
                "How to derive the output filename from the source.",
                label_id=f"{prefix}-lbl-filename",
            ),
            html.Div(
                id=f"{prefix}-suffix-row",
                style={"display": "none"},
                children=[
                    form_row(
                        "Suffix",
                        dbc.Input(
                            id=f"{prefix}-filename-suffix",
                            placeholder="e.g. _v2",
                            size="sm",
                        ),
                        "Text to append to the stem before the extension.",
                        label_id=f"{prefix}-lbl-suffix",
                    ),
                ],
            ),
        ])

    overwrite_row = form_row(
        "On conflict",
        make_dropdown(f"{prefix}-overwrite", _OVERWRITE_OPTS, "refuse"),
        "What to do when the output file already exists.",
        label_id=f"{prefix}-lbl-conflict",
    )

    btn_label = "Save" if mode == "single" else "Convert All"
    action_btn = dbc.Button(
        btn_label,
        id=f"{prefix}-save-btn",
        color="primary",
        style={"width": "100%", "marginTop": "0.5rem"},
    )

    stores = [
        dcc.Store(id=f"{prefix}-value"),
        dcc.Store(id=f"{prefix}-prefill"),
    ]
    if mode == "single":
        stores.append(dcc.Store(id=f"{prefix}-motl-id"))

    return html.Div(
        id=f"{prefix}-container",
        children=[
            format_row,
            writer_form,
            relion_panel,
            dest_section,
            overwrite_row,
            html.Div(id=f"{prefix}-validation", style={"color": "red", "marginTop": "0.3rem", "whiteSpace": "pre-line"}),
            action_btn,
            html.Div(id=f"{prefix}-status", style={"marginTop": "0.3rem", "color": "var(--color9)"}),
            *stores,
        ],
    )


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_save_dialog_callbacks(
    app,
    prefix: str,
    *,
    mode: Literal["single", "batch"] = "single",
) -> None:
    """Wire all non-save interactions for the dialog.

    Single mode also registers the save action (reads {prefix}-motl-id store).
    Batch mode skips the save action — caller registers it (needs pool globals).
    """
    register_relion_options_callbacks(
        app, prefix, for_load=False, type_input_id=f"{prefix}-format",
    )

    # Register all formgen callbacks for the dynamic writer-params form type.
    register_form_callbacks(app, f"{prefix}-writer-param")

    @app.callback(
        Output(f"{prefix}-writer-form", "children"),
        Input(f"{prefix}-format", "value"),
        prevent_initial_call=True,
    )
    def _on_format(fmt):
        return _writer_form_for(fmt or "", prefix)

    @app.callback(
        Output(f"{prefix}-format", "value"),
        Output(f"{prefix}-rln-version", "value", allow_duplicate=True),
        Output(f"{prefix}-rln-pixelsize", "value"),
        Output(f"{prefix}-rln-binning", "value"),
        Input(f"{prefix}-prefill", "data"),
        prevent_initial_call=True,
    )
    def _on_prefill(prefill):
        if not prefill:
            raise dash.exceptions.PreventUpdate
        return _prefill_to_form_values(prefill)

    if mode == "single":
        @app.callback(
            Output(f"{prefix}-resolve-hint", "children"),
            Input({"type": "path-input", "owner": f"{prefix}-dest"}, "value"),
            Input(f"{prefix}-format", "value"),
            prevent_initial_call=True,
        )
        def _on_path_fmt(path, fmt):
            if not path or not fmt:
                return ""
            resolved = resolve_extension(path, fmt)
            return f"→ {resolved}" if resolved != path else ""

        @app.callback(
            Output(f"{prefix}-status", "children"),
            Output(f"{prefix}-validation", "children"),
            Input(f"{prefix}-save-btn", "n_clicks"),
            State(f"{prefix}-format", "value"),
            State({"type": "path-input", "owner": f"{prefix}-dest"}, "value"),
            State(f"{prefix}-overwrite", "value"),
            State(f"{prefix}-rln-value", "data"),
            State({"type": f"{prefix}-writer-param", "owner": ALL, "param": ALL, "tag": ALL}, "value"),
            State({"type": f"{prefix}-writer-param", "owner": ALL, "param": ALL, "tag": ALL}, "id"),
            State(f"{prefix}-motl-id", "data"),
            prevent_initial_call=True,
        )
        def _save_single(n_clicks, fmt, path, overwrite, rln_value, writer_vals, writer_ids, motl_id):
            if not n_clicks:
                raise dash.exceptions.PreventUpdate
            probs = validate_save(path, fmt, rln_value, mode="single")
            if not probs and overwrite == "refuse" and path and fmt:
                resolved = resolve_extension(path, fmt)
                if os.path.exists(resolved):
                    probs = [f"File already exists: {resolved}"]
            if probs:
                return no_update, "\n".join(probs)
            if not motl_id:
                return no_update, "No motl selected for this slot."
            writer_kwargs = generate_kwargs(writer_ids, writer_vals) if writer_ids else {}
            try:
                status = execute_save_single(motl_id, path, fmt, rln_value, writer_kwargs)
                return status, ""
            except Exception as exc:
                return no_update, str(exc)

    else:  # batch — register only the suffix-row visibility
        @app.callback(
            Output(f"{prefix}-suffix-row", "style"),
            Input(f"{prefix}-filename-policy", "value"),
            prevent_initial_call=True,
        )
        def _on_policy(policy):
            return {"display": "block"} if policy == "suffix" else {"display": "none"}


def read_save_dialog(state: dict | None) -> dict:
    """Pure: return the SaveSpec stored in {prefix}-value (or empty dict)."""
    return state or {}
