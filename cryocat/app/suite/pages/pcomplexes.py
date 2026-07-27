"""Complexes page — one accordion item per complex type.

First accordion item is "Motl" (open on load) for pool selection.
Remaining items are collapsed by default:

* *(CnComplex only)* A sub-type selector (CnComplex base vs NPC).
* An input-motl picker that feeds the class constructor.
* An **Init parameters** form built automatically by
  :func:`cryocat.app.formgen.build_form` from the class ``__init__``
  signature.
* A **Create** button that validates and instantiates the complex.
* A **Method** dropdown listing all instance methods for the selected complex.
* A dynamic method-parameter form that updates when the method changes.
* A **Run** button and a per-complex status line.

CnComplex additionally exposes NPC static operations via a dropdown that
appears only when the NPC sub-type is selected.

Dispatch
--------
All commit actions route through
:func:`cryocat.app.apputils.run_operation`.  No previews.

Contract: exposes :data:`layout` and :func:`register_callbacks(app)`.
"""
from __future__ import annotations

from typing import Any

import dash
from dash import html, dcc, Input, Output, State, ALL, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd

from cryocat.core.cryomotl import EmMotl, Motl
from cryocat.analysis.structure import (
    CnComplex, DnComplex, NPC,
    TetrahedralComplex, OctahedralComplex, IcosahedralComplex,
)
from cryocat.app import ids, formgen
from cryocat.app.apputils import generate_kwargs, run_operation
from cryocat.app.pool import PoolState, insert_motl
from cryocat.app.components.logpanel import get_log_panel, register_log_panel_callbacks
from cryocat.app.components.motlsource import (
    get_motl_source, register_motl_source_callbacks,
)
from cryocat.app.components.motlsink import (
    get_send_to_editor_button, register_send_to_editor_callbacks,
)
from cryocat.app.pageshell import page_shell


_HINT = {"fontSize": "0.8rem", "color": "var(--color9)", "margin": "0.3rem 0"}
_SECTION_HEADER = {"fontSize": "0.95rem", "fontWeight": 600, "margin": "0.4rem 0 0.2rem"}


# ── Layout helpers ────────────────────────────────────────────────────────────

def _row(label: str, control: Any) -> html.Div:
    """Two-column label + control row matching formgen layout."""
    return html.Div([
        html.Div(
            html.Label(label, style={"fontSize": "0.85rem", "margin": 0}),
            style={"width": "45%", "display": "flex", "alignItems": "center",
                   "boxSizing": "border-box", "paddingRight": "4px"},
        ),
        html.Div(control, style={"width": "55%"}),
    ], style={"display": "flex", "flexDirection": "row", "marginBottom": "0.25rem",
              "width": "100%", "alignItems": "center"})


def _dd(
    dd_id: str,
    options: list,
    value: Any = None,
    placeholder: str | None = None,
    clearable: bool = False,
) -> dcc.Dropdown:
    """Compact dropdown; search enabled only when options > 10."""
    return dcc.Dropdown(
        id=dd_id,
        options=options,
        value=value,
        placeholder=placeholder,
        clearable=clearable,
        searchable=len(options) > 10,
        style={"fontSize": "0.85rem"},
    )


# ── Result type constants ─────────────────────────────────────────────────────

_R_MOTL    = "motl"
_R_MOTLS   = "motls"
_R_DF      = "dataframe"
_R_DIAM    = "diameter"
_R_INPLACE = "inplace"
_R_COORDS  = "coords"


# ── Method spec helpers ───────────────────────────────────────────────────────

def _inst(label: str, fn_attr: str, result: str, excluded: tuple = ()) -> dict:
    return {
        "label": label, "kind": "instance", "fn_attr": fn_attr,
        "excluded": excluded, "result": result,
    }


def _clsm(label: str, fn_attr: str, result: str, excluded: tuple = ()) -> dict:
    return {
        "label": label, "kind": "classmethod", "fn_attr": fn_attr,
        "excluded": excluded, "result": result,
    }


# ── CnComplex / NPC instance methods ─────────────────────────────────────────

_CN_BASE_METHODS: dict[str, dict] = {
    "create_affiliation": _inst("Create affiliation",    "create_affiliation",   _R_MOTL),
    "assign_order":       _inst("Assign subunit order",  "assign_subunit_order", _R_INPLACE),
    "get_centers":        _inst("Get centers as motl",   "get_centers_as_motl",  _R_MOTL),
    "circumference":      _inst("Circumference",         "circumference",        _R_DF),
    "diameter":           _inst("Diameter",              "diameter",             _R_DIAM),
    "get_object_stats":   _inst("Get object stats",      "get_object_stats",     _R_DF),
    "occupancy":          _inst("Occupancy",             "occupancy",            _R_DF),
    "clean_per_object":   _inst("Clean per object",      "clean_per_object",     _R_MOTL),
    "merge_subunits":     _inst("Merge subunits",        "merge_subunits",       _R_INPLACE),
}

_NPC_INSTANCE_METHODS: dict[str, dict] = {
    **_CN_BASE_METHODS,
    "npc_unify": _inst("Unify NN orientations", "unify_nn_orientations", _R_INPLACE),
}

_NPC_STATIC_OPS: dict[str, dict] = {
    "cluster_subunits": {
        "label": "Cluster subunits → rings",
        "method_name": "cluster_subunits_to_rings",
        "pickers": [("input_motl", False)],
        "excluded_params": (
            "input_motl", "entry_mask", "exit_mask",
            "entry_mask_coord", "exit_mask_coord", "mask_size",
        ),
        "result_kind": _R_MOTL,
        "needs_mask_widget": True,
    },
    "merge_rings": {
        "label": "Merge rings (multi-motl)",
        "method_name": "merge_rings",
        "pickers": [("input_motls", True)],
        "excluded_params": ("input_motls",),
        "result_kind": _R_MOTLS,
        "needs_mask_widget": False,
    },
    "npc_centers": {
        "label": "Get centers as motl",
        "method_name": "get_centers_as_motl",
        "pickers": [("tomo_motl", False)],
        "excluded_params": ("tomo_motl",),
        "result_kind": _R_MOTL,
        "needs_mask_widget": False,
    },
    "npc_diameter": {
        "label": "Compute diameter",
        "method_name": "compute_diameter",
        "pickers": [("input_motl", False)],
        "excluded_params": ("input_motl",),
        "result_kind": _R_DIAM,
        "needs_mask_widget": False,
    },
}

_NPC_OP_IDS = list(_NPC_STATIC_OPS)


# ── DnComplex methods ─────────────────────────────────────────────────────────

_DN_METHODS: dict[str, dict] = {
    **_CN_BASE_METHODS,
    "split_rings":       _inst("Split rings",       "split_rings",       _R_MOTL),
    "ring_spacing":      _inst("Ring spacing",      "ring_spacing",      _R_DF),
    "inter_ring_twist":  _inst("Inter-ring twist",  "inter_ring_twist",  _R_DF),
}


# ── Polyhedral methods (shared by Tet / Oct / Ico) ────────────────────────────

_POLY_METHODS: dict[str, dict] = {
    "create_affiliation": _inst("Create affiliation",   "create_affiliation",   _R_MOTL),
    "assign_order":       _inst("Assign subunit order", "assign_subunit_order", _R_INPLACE),
    "get_centers":        _inst("Get centers as motl",  "get_centers_as_motl",  _R_MOTL),
    "occupancy":          _inst("Occupancy",            "occupancy",            _R_DF),
    "clean_per_object":   _inst("Clean per object",     "clean_per_object",     _R_MOTL),
    "merge_subunits":     _inst("Merge subunits",       "merge_subunits",       _R_INPLACE),
    "feature_vectors":    _inst("Feature vectors",      "feature_vectors",      _R_COORDS),
    "expand":             _inst("Expand",               "expand",               _R_MOTL,
                                excluded=("output_path", "output_motl_type", "shift_vecs")),
    "recover_features":   _clsm("Recover features",     "recover_features",     _R_COORDS),
}


# ── Pure helpers ──────────────────────────────────────────────────────────────


def motl_from_pool_rows(pool_motls: dict | None, motl_id: str | None) -> Motl | None:
    if not motl_id:
        return None
    rows = (pool_motls or {}).get(motl_id) or []
    if not rows:
        return None
    return EmMotl(pd.DataFrame(rows))


def motl_to_pool_rows(motl: Motl | None) -> list[dict]:
    if motl is None:
        return []
    return motl.df.to_dict("records")


def resolve_mask_kwargs(
    mode: str,
    mask_size: Any,
    entry_coord: Any,
    exit_coord: Any,
    entry_path: str | None,
    exit_path: str | None,
) -> dict:
    if mode == "paths":
        out: dict = {}
        if entry_path and str(entry_path).strip():
            out["entry_mask"] = str(entry_path).strip()
        if exit_path and str(exit_path).strip():
            out["exit_mask"] = str(exit_path).strip()
        return out
    out = {}
    if mask_size not in (None, ""):
        out["mask_size"] = mask_size
    if entry_coord:
        out["entry_mask_coord"] = entry_coord
    if exit_coord:
        out["exit_mask_coord"] = exit_coord
    return out


def _parse_triplet(text: str | None) -> tuple[int, int, int] | None:
    if not text:
        return None
    parts = [p.strip() for p in str(text).replace(";", ",").split(",")]
    parts = [p for p in parts if p]
    if len(parts) != 3:
        return None
    try:
        return tuple(int(round(float(p))) for p in parts)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _build_method_form(method_id: str, methods: dict, cls: type, cpx_id: str) -> list:
    spec = methods.get(method_id)
    if not spec:
        return []
    fn = getattr(cls, spec["fn_attr"])
    return formgen.build_form(
        fn,
        id_type="cpx-meth-param",
        id_extra={"cpx": cpx_id},
        exclude=list(spec.get("excluded", ())),
    )


def _dispatch_result(spec: dict, result: Any, instance: Any = None) -> tuple:
    label = spec["label"]
    kind = spec["result"]

    if kind == _R_MOTL:
        if not isinstance(result, Motl):
            return (no_update, no_update, no_update, no_update,
                    f"{label}: expected Motl, got {type(result).__name__}.")
        rows = motl_to_pool_rows(result)
        return rows, no_update, no_update, no_update, f"{label} → {len(rows)} particles."

    if kind == _R_INPLACE:
        if instance is None:
            return (no_update, no_update, no_update, no_update,
                    f"{label}: in-place method but no instance available.")
        rows = motl_to_pool_rows(instance.motl)
        return rows, no_update, no_update, no_update, f"{label} done → {len(rows)} particles."

    if kind == _R_DF:
        records = result.to_dict("records") if isinstance(result, pd.DataFrame) else []
        return no_update, no_update, no_update, records, f"{label} → {len(records)} rows."

    if kind == _R_DIAM:
        if not (isinstance(result, tuple) and len(result) == 2):
            return (no_update, no_update, no_update, no_update,
                    f"{label}: expected (DataFrame, Motl) tuple.")
        summary_df, motl_out = result
        diam_records = summary_df.to_dict("records") if not summary_df.empty else []
        rows = motl_to_pool_rows(motl_out)
        return rows, no_update, diam_records, no_update, f"{label} → {len(diam_records)} object(s)."

    if kind == _R_COORDS:
        arr = result[0] if isinstance(result, tuple) else result
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            ncols = arr.shape[1]
            cols = (["x", "y", "z"] + [str(i) for i in range(3, ncols)])[:ncols]
            records = pd.DataFrame(arr, columns=cols).to_dict("records")
        else:
            records = []
        return no_update, no_update, no_update, records, f"{label} → {len(records)} points."

    if kind == _R_MOTLS:
        motls_rows = [motl_to_pool_rows(m) for m in result] if isinstance(result, list) else []
        return no_update, motls_rows, no_update, no_update, f"{label} → {len(motls_rows)} motls."

    return no_update, no_update, no_update, no_update, f"{label}: unknown result kind {kind!r}."


def _run_instance_method(
    cls: type,
    spec: dict,
    motl_id: str | None,
    pool_motls: dict | None,
    init_ids: list,
    init_vals: list,
    meth_ids: list,
    meth_vals: list,
) -> tuple:
    meth_kwargs = generate_kwargs(meth_ids, meth_vals) if (meth_ids and meth_vals) else {}
    meth_kwargs = {k: v for k, v in meth_kwargs.items() if v not in (None, "", [])}

    if spec["kind"] == "classmethod":
        method = getattr(cls, spec["fn_attr"])
        try:
            result = run_operation(method, meth_kwargs)
        except Exception as exc:
            return no_update, no_update, no_update, no_update, f"{spec['label']} failed: {exc}"
        return _dispatch_result(spec, result)

    motl = motl_from_pool_rows(pool_motls, motl_id)
    if motl is None:
        return no_update, no_update, no_update, no_update, "Pick a non-empty input motl."

    init_kwargs = generate_kwargs(init_ids, init_vals) if (init_ids and init_vals) else {}
    init_kwargs = {k: v for k, v in init_kwargs.items() if v not in (None, "", [])}

    try:
        instance = cls(motl, **init_kwargs)
    except Exception as exc:
        return no_update, no_update, no_update, no_update, f"Init failed: {exc}"

    method = getattr(instance, spec["fn_attr"])
    try:
        result = run_operation(method, meth_kwargs)
    except Exception as exc:
        return no_update, no_update, no_update, no_update, f"{spec['label']} failed: {exc}"

    return _dispatch_result(spec, result, instance)


def _render_df_table(records: list | None, empty_msg: str) -> Any:
    if not records:
        return html.Small(empty_msg, style=_HINT)
    df = pd.DataFrame(records)
    if df.empty:
        return html.Small("No results.", style=_HINT)
    header = html.Thead([html.Tr([
        html.Th(c, style={"fontSize": "0.85rem"}) for c in df.columns
    ])])
    body_rows = []
    for r in records:
        cells = []
        for c in df.columns:
            val = r.get(c)
            try:
                is_nan = pd.isna(val)
            except (TypeError, ValueError):
                is_nan = False
            if is_nan:
                cells.append(html.Td(""))
            elif isinstance(val, float):
                cells.append(html.Td(f"{val:.4g}"))
            else:
                cells.append(html.Td(str(val)))
        body_rows.append(html.Tr(cells))
    return dbc.Table(
        [header, html.Tbody(body_rows)],
        bordered=True, striped=True, hover=True, size="sm",
        style={"fontSize": "0.85rem"},
    )


# ── NPC mask source widget ────────────────────────────────────────────────────


def _triplet_input(input_id: str, placeholder: str = "x,y,z") -> dcc.Input:
    return dcc.Input(
        id=input_id, type="text", placeholder=placeholder,
        style={"width": "100%", "fontSize": "0.85rem",
               "height": "22px", "padding": "0 6px", "lineHeight": "20px"},
    )


def _mask_source_widget(prefix: str) -> html.Div:
    return html.Div([
        html.Small("Mask source", style=_HINT),
        dcc.RadioItems(
            id=f"{prefix}-mode",
            options=[
                {"label": " Shift (coord + size)", "value": "shift"},
                {"label": " Mask files (paths)",   "value": "paths"},
            ],
            value="shift",
            style={"fontSize": "0.85rem"},
            inputStyle={"marginRight": "0.25rem"},
            labelStyle={"display": "block", "marginBottom": "0.15rem"},
        ),
        html.Div([
            _row("Mask size", _triplet_input(f"{prefix}-size", "72 or 72,72,72")),
            _row("Entry centre", _triplet_input(f"{prefix}-entry-coord", "e.g. 34,61,36")),
            _row("Exit centre",  _triplet_input(f"{prefix}-exit-coord",  "e.g. 34,17,36")),
        ], id=f"{prefix}-shift-section", style={"display": "block"}),
        html.Div([
            _row("Entry mask", dcc.Input(id=f"{prefix}-entry-path", type="text",
                 placeholder="path/to/entry_mask.em",
                 style={"width": "100%", "fontSize": "0.85rem",
                        "height": "22px", "padding": "0 6px"})),
            _row("Exit mask",  dcc.Input(id=f"{prefix}-exit-path", type="text",
                 placeholder="path/to/exit_mask.em",
                 style={"width": "100%", "fontSize": "0.85rem",
                        "height": "22px", "padding": "0 6px"})),
        ], id=f"{prefix}-paths-section", style={"display": "none"}),
    ])


# ── Per-complex sidebar accordion items ───────────────────────────────────────


def _motl_accordion_item() -> dbc.AccordionItem:
    return dbc.AccordionItem([
        html.Small(
            "Select the motl from the pool. "
            "All complex-specific pickers below are pre-populated from the same pool.",
            style=_HINT,
        ),
        get_motl_source("cpx-global", multi=False),
    ],
    title="Motl",
    item_id="cpx-motl",
    )


def _cn_accordion_item() -> dbc.AccordionItem:
    init_form = formgen.build_form(
        CnComplex,
        id_type="cpx-init-param",
        id_extra={"cpx": "cn"},
        exclude=["motl"],
    )

    _cn_subtype_opts = [
        {"label": "CnComplex (base)", "value": "cn_base"},
        {"label": "NPC (Nuclear Pore Complex)", "value": "npc"},
    ]
    _cn_method_opts = [
        {"label": spec["label"], "value": mid} for mid, spec in _CN_BASE_METHODS.items()
    ]
    _npc_op_opts = [
        {"label": op["label"], "value": op_id} for op_id, op in _NPC_STATIC_OPS.items()
    ]

    # Pre-render NPC static op pickers (must be in DOM for callback registration).
    npc_picker_divs = []
    for op_id, op in _NPC_STATIC_OPS.items():
        pfx = f"cpx-cn-npc-{op_id}"
        inner: list = []
        for picker_name, multi in op["pickers"]:
            inner.append(html.Div([
                html.Small(picker_name, style=_HINT),
                get_motl_source(f"{pfx}-{picker_name}", multi=multi),
            ], style={"marginBottom": "0.3rem"}))
        if op["needs_mask_widget"]:
            inner.append(_mask_source_widget(f"{pfx}-mask"))
        npc_picker_divs.append(
            html.Div(inner, id=f"cpx-cn-npc-picker-{op_id}", style={"display": "none"})
        )

    return dbc.AccordionItem([
        _row("Sub-type", _dd("cpx-cn-subtype", _cn_subtype_opts, value="cn_base")),
        get_motl_source("cpx-cn-main", multi=False),
        html.Hr(style={"margin": "0.4rem 0"}),
        html.Div("Init parameters", style=_SECTION_HEADER),
        html.Div(init_form),
        dbc.Button(
            "Create", id="cpx-cn-create-btn", color="secondary", size="sm",
            style={"width": "100%", "marginTop": "0.4rem"},
        ),
        html.Div(id="cpx-cn-create-status",
                 style={**_HINT, "marginTop": "0.3rem", "wordBreak": "break-word"}),
        html.Hr(style={"margin": "0.4rem 0"}),
        _row("Method", _dd("cpx-cn-method-dd", _cn_method_opts,
                           placeholder="Select method…", clearable=True)),
        html.Div(id="cpx-cn-method-form", style={"marginBottom": "0.4rem"}),
        dbc.Button(
            "Run", id="cpx-cn-run-btn", color="primary", size="sm",
            style={"width": "100%"},
        ),
        html.Div(id="cpx-cn-status",
                 style={**_HINT, "marginTop": "0.3rem", "wordBreak": "break-word"}),
        # NPC static ops — shown only when sub-type is NPC
        html.Div(
            id="cpx-cn-npc-ops",
            children=[
                html.Hr(style={"margin": "0.6rem 0"}),
                html.Div("NPC static operation", style=_SECTION_HEADER),
                _row("Operation", _dd("cpx-cn-npc-op-dd", _npc_op_opts,
                                      placeholder="Select…", clearable=True)),
                *npc_picker_divs,
                html.Div(id="cpx-cn-npc-op-form", style={"marginBottom": "0.4rem"}),
                dbc.Button(
                    "Run NPC op", id="cpx-cn-npc-op-run-btn", color="primary", size="sm",
                    style={"width": "100%"},
                ),
                html.Div(id="cpx-cn-npc-op-status",
                         style={**_HINT, "marginTop": "0.3rem", "wordBreak": "break-word"}),
            ],
            style={"display": "none"},
        ),
    ],
    title="CnComplex",
    item_id="cpx-cn",
    )


def _dn_accordion_item() -> dbc.AccordionItem:
    init_form = formgen.build_form(
        DnComplex,
        id_type="cpx-init-param",
        id_extra={"cpx": "dn"},
        exclude=["motl"],
    )
    _dn_method_opts = [
        {"label": spec["label"], "value": mid} for mid, spec in _DN_METHODS.items()
    ]
    return dbc.AccordionItem([
        get_motl_source("cpx-dn-main", multi=False),
        html.Hr(style={"margin": "0.4rem 0"}),
        html.Div("Init parameters", style=_SECTION_HEADER),
        html.Div(init_form),
        dbc.Button(
            "Create", id="cpx-dn-create-btn", color="secondary", size="sm",
            style={"width": "100%", "marginTop": "0.4rem"},
        ),
        html.Div(id="cpx-dn-create-status",
                 style={**_HINT, "marginTop": "0.3rem", "wordBreak": "break-word"}),
        html.Hr(style={"margin": "0.4rem 0"}),
        _row("Method", _dd("cpx-dn-method-dd", _dn_method_opts,
                           placeholder="Select method…", clearable=True)),
        html.Div(id="cpx-dn-method-form", style={"marginBottom": "0.4rem"}),
        dbc.Button(
            "Run", id="cpx-dn-run-btn", color="primary", size="sm",
            style={"width": "100%"},
        ),
        html.Div(id="cpx-dn-status",
                 style={**_HINT, "marginTop": "0.3rem", "wordBreak": "break-word"}),
    ],
    title="DnComplex",
    item_id="cpx-dn",
    )


def _poly_accordion_item(cpx_id: str, label: str, cls: type) -> dbc.AccordionItem:
    init_form = formgen.build_form(
        cls,
        id_type="cpx-init-param",
        id_extra={"cpx": cpx_id},
        exclude=["motl"],
    )
    _poly_method_opts = [
        {"label": spec["label"], "value": mid} for mid, spec in _POLY_METHODS.items()
    ]
    return dbc.AccordionItem([
        get_motl_source(f"cpx-{cpx_id}-main", multi=False),
        html.Hr(style={"margin": "0.4rem 0"}),
        html.Div("Init parameters", style=_SECTION_HEADER),
        html.Div(init_form),
        dbc.Button(
            "Create", id=f"cpx-{cpx_id}-create-btn", color="secondary", size="sm",
            style={"width": "100%", "marginTop": "0.4rem"},
        ),
        html.Div(id=f"cpx-{cpx_id}-create-status",
                 style={**_HINT, "marginTop": "0.3rem", "wordBreak": "break-word"}),
        html.Hr(style={"margin": "0.4rem 0"}),
        _row("Method", _dd(f"cpx-{cpx_id}-method-dd", _poly_method_opts,
                           placeholder="Select method…", clearable=True)),
        html.Div(id=f"cpx-{cpx_id}-method-form", style={"marginBottom": "0.4rem"}),
        dbc.Button(
            "Run", id=f"cpx-{cpx_id}-run-btn", color="primary", size="sm",
            style={"width": "100%"},
        ),
        html.Div(id=f"cpx-{cpx_id}-status",
                 style={**_HINT, "marginTop": "0.3rem", "wordBreak": "break-word"}),
    ],
    title=label,
    item_id=f"cpx-{cpx_id}",
    )


# ── Full layout ───────────────────────────────────────────────────────────────


def _sidebar() -> list:
    return [
        dbc.Accordion(
            [
                _motl_accordion_item(),
                _cn_accordion_item(),
                _dn_accordion_item(),
                _poly_accordion_item("tet", "Tetrahedral", TetrahedralComplex),
                _poly_accordion_item("oct", "Octahedral",  OctahedralComplex),
                _poly_accordion_item("ico", "Icosahedral", IcosahedralComplex),
            ],
            always_open=False,
            active_item="cpx-motl",
            id="cpx-main-accordion",
        ),
    ]


def _main() -> list:
    return [
        html.Div("Result motl(s)", style=_SECTION_HEADER),
        get_send_to_editor_button("complexes-export"),
        html.Div(id="complexes-export-extra",
                 style={**_HINT, "marginTop": "0.3rem"}),
        dcc.Store(id="complexes-result-motl"),
        dcc.Store(id="complexes-result-motls"),
        html.Hr(style={"margin": "0.6rem 0"}),
        html.Div("Results table", style=_SECTION_HEADER),
        html.Div(id="complexes-result-table", style={"padding": "0.25rem"}),
        dcc.Store(id="complexes-result-df-store"),
        html.Hr(style={"margin": "0.6rem 0"}),
        html.Div("Diameter results", style=_SECTION_HEADER),
        html.Div(id="complexes-diameter-table", style={"padding": "0.25rem"}),
        dcc.Store(id="complexes-diameter-store"),
    ]


layout = html.Div(
    [
        page_shell(_sidebar(), _main(), sidebar_width=4),
        *get_log_panel("complexes-log"),
    ],
    style={"margin": 0, "padding": 0},
)


# ── Callbacks ─────────────────────────────────────────────────────────────────


def register_callbacks(app: dash.Dash) -> None:
    register_log_panel_callbacks(app, "complexes-log")
    register_send_to_editor_callbacks(app, "complexes-export", "complexes-result-motl")

    # ── Motl-source picker registrations ────────────────────────────────────
    register_motl_source_callbacks(app, "cpx-global", multi=False)
    register_motl_source_callbacks(app, "cpx-cn-main", multi=False)
    register_motl_source_callbacks(app, "cpx-dn-main", multi=False)
    for _cpx in ("tet", "oct", "ico"):
        register_motl_source_callbacks(app, f"cpx-{_cpx}-main", multi=False)
    for _op_id, _op in _NPC_STATIC_OPS.items():
        for _picker_name, _multi in _op["pickers"]:
            register_motl_source_callbacks(
                app, f"cpx-cn-npc-{_op_id}-{_picker_name}", multi=_multi,
            )

    # ── CnComplex / NPC ──────────────────────────────────────────────────────

    @app.callback(
        Output("cpx-cn-method-dd", "options"),
        Input("cpx-cn-subtype", "value"),
    )
    def _update_cn_method_options(subtype: str) -> list:
        methods = _NPC_INSTANCE_METHODS if subtype == "npc" else _CN_BASE_METHODS
        return [{"label": spec["label"], "value": mid} for mid, spec in methods.items()]

    @app.callback(
        Output("cpx-cn-npc-ops", "style"),
        Input("cpx-cn-subtype", "value"),
    )
    def _toggle_npc_ops_section(subtype: str) -> dict:
        return {"display": "block"} if subtype == "npc" else {"display": "none"}

    @app.callback(
        Output("cpx-cn-method-form", "children"),
        Input("cpx-cn-method-dd", "value"),
        State("cpx-cn-subtype", "value"),
    )
    def _update_cn_method_form(method_id: str, subtype: str) -> list:
        if not method_id:
            return []
        cls = NPC if subtype == "npc" else CnComplex
        methods = _NPC_INSTANCE_METHODS if subtype == "npc" else _CN_BASE_METHODS
        return _build_method_form(method_id, methods, cls, "cn")

    @app.callback(
        Output("cpx-cn-create-status", "children"),
        Input("cpx-cn-create-btn", "n_clicks"),
        State("cpx-cn-subtype", "value"),
        State("cpx-cn-main-motl-select", "value"),
        State(ids.POOL_MOTLS, "data"),
        State({"type": "cpx-init-param", "cpx": "cn", "param": ALL, "tag": ALL}, "value"),
        State({"type": "cpx-init-param", "cpx": "cn", "param": ALL, "tag": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _create_cn(n_clicks, subtype, motl_id, pool_motls, init_vals, init_ids):
        if not n_clicks:
            raise PreventUpdate
        cls = NPC if subtype == "npc" else CnComplex
        motl = motl_from_pool_rows(pool_motls, motl_id)
        if motl is None:
            return "Pick a non-empty input motl."
        init_kwargs = generate_kwargs(init_ids, init_vals) if (init_ids and init_vals) else {}
        init_kwargs = {k: v for k, v in init_kwargs.items() if v not in (None, "", [])}
        try:
            instance = cls(motl, **init_kwargs)
            return f"{cls.__name__} created — {len(instance.motl.df)} particles."
        except Exception as exc:
            return f"Create failed: {exc}"

    @app.callback(
        Output("complexes-result-motl",    "data", allow_duplicate=True),
        Output("complexes-result-motls",   "data", allow_duplicate=True),
        Output("complexes-diameter-store", "data", allow_duplicate=True),
        Output("complexes-result-df-store","data", allow_duplicate=True),
        Output("cpx-cn-status", "children"),
        Input("cpx-cn-run-btn", "n_clicks"),
        State("cpx-cn-subtype", "value"),
        State("cpx-cn-method-dd", "value"),
        State("cpx-cn-main-motl-select", "value"),
        State(ids.POOL_MOTLS, "data"),
        State({"type": "cpx-init-param", "cpx": "cn", "param": ALL, "tag": ALL}, "value"),
        State({"type": "cpx-init-param", "cpx": "cn", "param": ALL, "tag": ALL}, "id"),
        State({"type": "cpx-meth-param", "cpx": "cn", "param": ALL, "tag": ALL}, "value"),
        State({"type": "cpx-meth-param", "cpx": "cn", "param": ALL, "tag": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _run_cn(n_clicks, subtype, method_id, motl_id, pool_motls,
                init_vals, init_ids, meth_vals, meth_ids):
        if not n_clicks or not method_id:
            raise PreventUpdate
        cls = NPC if subtype == "npc" else CnComplex
        methods = _NPC_INSTANCE_METHODS if subtype == "npc" else _CN_BASE_METHODS
        spec = methods.get(method_id)
        if not spec:
            return no_update, no_update, no_update, no_update, f"Unknown method: {method_id!r}"
        return _run_instance_method(
            cls, spec, motl_id, pool_motls,
            init_ids, init_vals, meth_ids, meth_vals,
        )

    # ── NPC static op dropdown + run ─────────────────────────────────────────

    @app.callback(
        Output("cpx-cn-npc-picker-cluster_subunits", "style"),
        Output("cpx-cn-npc-picker-merge_rings",      "style"),
        Output("cpx-cn-npc-picker-npc_centers",      "style"),
        Output("cpx-cn-npc-picker-npc_diameter",     "style"),
        Output("cpx-cn-npc-op-form", "children"),
        Input("cpx-cn-npc-op-dd", "value"),
    )
    def _update_npc_op_selection(op_id: str | None) -> tuple:
        visible = {"display": "block"}
        hidden  = {"display": "none"}
        styles = [visible if oid == op_id else hidden for oid in _NPC_OP_IDS]
        form: list = []
        if op_id and op_id in _NPC_STATIC_OPS:
            op = _NPC_STATIC_OPS[op_id]
            fn = getattr(NPC, op["method_name"])
            form = formgen.build_form(
                fn,
                id_type="cpx-npc-op-param",
                id_extra={"cpx": "npc", "op": op_id},
                exclude=list(op["excluded_params"]),
            )
        return (*styles, form)

    @app.callback(
        Output("cpx-cn-npc-cluster_subunits-mask-shift-section", "style"),
        Output("cpx-cn-npc-cluster_subunits-mask-paths-section", "style"),
        Input("cpx-cn-npc-cluster_subunits-mask-mode", "value"),
    )
    def _toggle_cluster_mask(mode: str) -> tuple:
        if mode == "paths":
            return {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}

    @app.callback(
        Output("complexes-result-motl",    "data", allow_duplicate=True),
        Output("complexes-result-motls",   "data", allow_duplicate=True),
        Output("complexes-diameter-store", "data", allow_duplicate=True),
        Output("complexes-result-df-store","data", allow_duplicate=True),
        Output("cpx-cn-npc-op-status", "children"),
        Input("cpx-cn-npc-op-run-btn", "n_clicks"),
        State("cpx-cn-npc-op-dd", "value"),
        State(ids.POOL_MOTLS, "data"),
        State({"type": "cpx-npc-op-param", "cpx": "npc", "op": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": "cpx-npc-op-param", "cpx": "npc", "op": ALL, "param": ALL, "tag": ALL}, "id"),
        State("cpx-cn-npc-cluster_subunits-input_motl-motl-select",  "value"),
        State("cpx-cn-npc-merge_rings-input_motls-motl-select",        "value"),
        State("cpx-cn-npc-npc_centers-tomo_motl-motl-select",          "value"),
        State("cpx-cn-npc-npc_diameter-input_motl-motl-select",        "value"),
        State("cpx-cn-npc-cluster_subunits-mask-mode",        "value"),
        State("cpx-cn-npc-cluster_subunits-mask-size",        "value"),
        State("cpx-cn-npc-cluster_subunits-mask-entry-coord", "value"),
        State("cpx-cn-npc-cluster_subunits-mask-exit-coord",  "value"),
        State("cpx-cn-npc-cluster_subunits-mask-entry-path",  "value"),
        State("cpx-cn-npc-cluster_subunits-mask-exit-path",   "value"),
        prevent_initial_call=True,
    )
    def _run_npc_static(
        n_clicks, op_id, pool_motls, param_values, param_ids,
        cluster_motl, merge_motls, centers_motl, diam_motl,
        mask_mode, mask_size, mask_entry_coord, mask_exit_coord,
        mask_entry_path, mask_exit_path,
    ):
        if not n_clicks or not op_id:
            raise PreventUpdate
        op = _NPC_STATIC_OPS.get(op_id)
        if not op:
            return no_update, no_update, no_update, no_update, f"Unknown op: {op_id!r}"

        label = op["label"]
        _picker_lookup: dict[str, dict] = {
            "cluster_subunits": {"input_motl":  cluster_motl},
            "merge_rings":       {"input_motls": merge_motls},
            "npc_centers":       {"tomo_motl":   centers_motl},
            "npc_diameter":      {"input_motl":  diam_motl},
        }
        picker_vals = _picker_lookup[op_id]

        kwargs: dict[str, Any] = {}
        for picker_name, multi in op["pickers"]:
            val = picker_vals[picker_name]
            if multi:
                motls = []
                for mid in (val or []):
                    m = motl_from_pool_rows(pool_motls, mid)
                    if m is None:
                        return (no_update, no_update, no_update, no_update,
                                f"Pick non-empty motl(s) for '{picker_name}'.")
                    motls.append(m)
                if len(motls) < 2 and op["method_name"] == "merge_rings":
                    return (no_update, no_update, no_update, no_update,
                            "merge_rings needs at least two motls.")
                kwargs[picker_name] = motls
            else:
                motl = motl_from_pool_rows(pool_motls, val)
                if motl is None:
                    return (no_update, no_update, no_update, no_update,
                            f"Pick a non-empty motl for '{picker_name}'.")
                kwargs[picker_name] = motl

        scalar_kwargs = generate_kwargs(param_ids, param_values) if (param_ids and param_values) else {}
        scalar_kwargs = {k: v for k, v in scalar_kwargs.items() if v not in (None, "", [])}
        kwargs.update(scalar_kwargs)

        if op["needs_mask_widget"]:
            size_val: Any = mask_size
            if size_val not in (None, "") and "," in str(size_val):
                size_val = _parse_triplet(size_val)
            elif size_val not in (None, ""):
                try:
                    size_val = int(round(float(size_val)))
                except (TypeError, ValueError):
                    size_val = None
            kwargs.update(resolve_mask_kwargs(
                mask_mode or "shift", size_val,
                _parse_triplet(mask_entry_coord),
                _parse_triplet(mask_exit_coord),
                mask_entry_path, mask_exit_path,
            ))

        spec = {"label": label, "result": op["result_kind"]}
        method = getattr(NPC, op["method_name"])

        if op["result_kind"] == _R_DIAM:
            try:
                summary_df, motl_out = run_operation(method, kwargs)
            except Exception as exc:
                return no_update, no_update, no_update, no_update, f"{label} failed: {exc}"
            if summary_df.empty:
                return (no_update, no_update, no_update, no_update,
                        f"{label} → no opposite pairs found.")
            rows = motl_to_pool_rows(motl_out)
            return (rows, no_update, summary_df.to_dict("records"), no_update,
                    f"{label} → {len(summary_df)} NPC(s).")

        try:
            result = run_operation(method, kwargs)
        except Exception as exc:
            return no_update, no_update, no_update, no_update, f"{label} failed: {exc}"

        return _dispatch_result(spec, result)

    # ── DnComplex ────────────────────────────────────────────────────────────

    @app.callback(
        Output("cpx-dn-create-status", "children"),
        Input("cpx-dn-create-btn", "n_clicks"),
        State("cpx-dn-main-motl-select", "value"),
        State(ids.POOL_MOTLS, "data"),
        State({"type": "cpx-init-param", "cpx": "dn", "param": ALL, "tag": ALL}, "value"),
        State({"type": "cpx-init-param", "cpx": "dn", "param": ALL, "tag": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _create_dn(n_clicks, motl_id, pool_motls, init_vals, init_ids):
        if not n_clicks:
            raise PreventUpdate
        motl = motl_from_pool_rows(pool_motls, motl_id)
        if motl is None:
            return "Pick a non-empty input motl."
        init_kwargs = generate_kwargs(init_ids, init_vals) if (init_ids and init_vals) else {}
        init_kwargs = {k: v for k, v in init_kwargs.items() if v not in (None, "", [])}
        try:
            instance = DnComplex(motl, **init_kwargs)
            return f"DnComplex created — {len(instance.motl.df)} particles."
        except Exception as exc:
            return f"Create failed: {exc}"

    @app.callback(
        Output("cpx-dn-method-form", "children"),
        Input("cpx-dn-method-dd", "value"),
    )
    def _update_dn_method_form(method_id: str) -> list:
        return _build_method_form(method_id, _DN_METHODS, DnComplex, "dn") if method_id else []

    @app.callback(
        Output("complexes-result-motl",    "data", allow_duplicate=True),
        Output("complexes-result-motls",   "data", allow_duplicate=True),
        Output("complexes-diameter-store", "data", allow_duplicate=True),
        Output("complexes-result-df-store","data", allow_duplicate=True),
        Output("cpx-dn-status", "children"),
        Input("cpx-dn-run-btn", "n_clicks"),
        State("cpx-dn-method-dd", "value"),
        State("cpx-dn-main-motl-select", "value"),
        State(ids.POOL_MOTLS, "data"),
        State({"type": "cpx-init-param", "cpx": "dn", "param": ALL, "tag": ALL}, "value"),
        State({"type": "cpx-init-param", "cpx": "dn", "param": ALL, "tag": ALL}, "id"),
        State({"type": "cpx-meth-param", "cpx": "dn", "param": ALL, "tag": ALL}, "value"),
        State({"type": "cpx-meth-param", "cpx": "dn", "param": ALL, "tag": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _run_dn(n_clicks, method_id, motl_id, pool_motls,
                init_vals, init_ids, meth_vals, meth_ids):
        if not n_clicks or not method_id:
            raise PreventUpdate
        spec = _DN_METHODS.get(method_id)
        if not spec:
            return no_update, no_update, no_update, no_update, f"Unknown method: {method_id!r}"
        return _run_instance_method(
            DnComplex, spec, motl_id, pool_motls,
            init_ids, init_vals, meth_ids, meth_vals,
        )

    # ── Polyhedral complexes ──────────────────────────────────────────────────

    for _cpx_id, _cls in [
        ("tet", TetrahedralComplex),
        ("oct", OctahedralComplex),
        ("ico", IcosahedralComplex),
    ]:
        _register_poly_callbacks(app, _cpx_id, _cls)

    # ── Result-area renderers ─────────────────────────────────────────────────

    @app.callback(
        Output("complexes-diameter-table", "children"),
        Input("complexes-diameter-store", "data"),
    )
    def _render_diameter(records):
        return _render_df_table(records, "Run a diameter method to populate this table.")

    @app.callback(
        Output("complexes-result-table", "children"),
        Input("complexes-result-df-store", "data"),
    )
    def _render_result_df(records):
        return _render_df_table(records, "Run an analysis method to see results here.")

    @app.callback(
        Output("complexes-export-extra", "children"),
        Input("complexes-result-motls", "data"),
    )
    def _show_motls_summary(motls_rows):
        if not motls_rows:
            return ""
        return html.Small(
            f"{len(motls_rows)} motl(s) ready (merge_rings). Use the button above to push each.",
            style=_HINT,
        )

    # ── Push merge_rings motls to pool ────────────────────────────────────────

    @app.callback(
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_MOTLS,    "data", allow_duplicate=True),
        Output(ids.POOL_EXTRA,    "data", allow_duplicate=True),
        Output(ids.POOL_META,     "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID,  "data", allow_duplicate=True),
        Output("complexes-export-extra", "children", allow_duplicate=True),
        Input("complexes-result-motls", "data"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_MOTLS,    "data"),
        State(ids.POOL_EXTRA,    "data"),
        State(ids.POOL_META,     "data"),
        State(ids.POOL_NEXT_ID,  "data"),
        prevent_initial_call=True,
    )
    def _push_motls_list(motls_rows, registry, pool_motls, pool_extra, pool_meta, next_id):
        if not motls_rows:
            raise PreventUpdate
        state = PoolState.from_stores(registry, pool_motls, pool_extra, pool_meta, next_id)
        pushed: list[str] = []
        for i, rows in enumerate(motls_rows):
            if not rows:
                continue
            # TODO(doc-2): route through run_operation_to_pool
            state, mid = insert_motl(state, rows, label=f"merged-{i + 1}")
            pushed.append(mid)
        msg = html.Small(
            f"Pushed {len(pushed)} motl(s) to pool: {', '.join(pushed)}.",
            style=_HINT,
        )
        return (*state.to_stores(), msg)


# ── Out-of-line callback helpers ──────────────────────────────────────────────


def _register_poly_callbacks(app: dash.Dash, cpx_id: str, cls: type) -> None:
    """Register Create + method-form update + Run callbacks for one poly complex."""

    @app.callback(
        Output(f"cpx-{cpx_id}-create-status", "children"),
        Input(f"cpx-{cpx_id}-create-btn", "n_clicks"),
        State(f"cpx-{cpx_id}-main-motl-select", "value"),
        State(ids.POOL_MOTLS, "data"),
        State({"type": "cpx-init-param", "cpx": cpx_id, "param": ALL, "tag": ALL}, "value"),
        State({"type": "cpx-init-param", "cpx": cpx_id, "param": ALL, "tag": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _create(n_clicks, motl_id, pool_motls, init_vals, init_ids, _cls=cls):
        if not n_clicks:
            raise PreventUpdate
        motl = motl_from_pool_rows(pool_motls, motl_id)
        if motl is None:
            return "Pick a non-empty input motl."
        init_kwargs = generate_kwargs(init_ids, init_vals) if (init_ids and init_vals) else {}
        init_kwargs = {k: v for k, v in init_kwargs.items() if v not in (None, "", [])}
        try:
            instance = _cls(motl, **init_kwargs)
            return f"{_cls.__name__} created — {len(instance.motl.df)} particles."
        except Exception as exc:
            return f"Create failed: {exc}"

    _create.__name__ = f"_create_{cpx_id}"

    @app.callback(
        Output(f"cpx-{cpx_id}-method-form", "children"),
        Input(f"cpx-{cpx_id}-method-dd", "value"),
    )
    def _update_form(method_id, _cpx=cpx_id, _cls=cls):
        return _build_method_form(method_id, _POLY_METHODS, _cls, _cpx) if method_id else []

    _update_form.__name__ = f"_update_{cpx_id}_method_form"

    @app.callback(
        Output("complexes-result-motl",    "data", allow_duplicate=True),
        Output("complexes-result-motls",   "data", allow_duplicate=True),
        Output("complexes-diameter-store", "data", allow_duplicate=True),
        Output("complexes-result-df-store","data", allow_duplicate=True),
        Output(f"cpx-{cpx_id}-status", "children"),
        Input(f"cpx-{cpx_id}-run-btn", "n_clicks"),
        State(f"cpx-{cpx_id}-method-dd", "value"),
        State(f"cpx-{cpx_id}-main-motl-select", "value"),
        State(ids.POOL_MOTLS, "data"),
        State({"type": "cpx-init-param", "cpx": cpx_id, "param": ALL, "tag": ALL}, "value"),
        State({"type": "cpx-init-param", "cpx": cpx_id, "param": ALL, "tag": ALL}, "id"),
        State({"type": "cpx-meth-param", "cpx": cpx_id, "param": ALL, "tag": ALL}, "value"),
        State({"type": "cpx-meth-param", "cpx": cpx_id, "param": ALL, "tag": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _run(n_clicks, method_id, motl_id, pool_motls,
             init_vals, init_ids, meth_vals, meth_ids,
             _cpx=cpx_id, _cls=cls):
        if not n_clicks or not method_id:
            raise PreventUpdate
        spec = _POLY_METHODS.get(method_id)
        if not spec:
            return no_update, no_update, no_update, no_update, f"Unknown method: {method_id!r}"
        return _run_instance_method(
            _cls, spec, motl_id, pool_motls,
            init_ids, init_vals, meth_ids, meth_vals,
        )

    _run.__name__ = f"_run_{cpx_id}"
