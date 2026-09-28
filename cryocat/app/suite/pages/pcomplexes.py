"""Complexes page — registry-driven; no per-complex branching.

A new symmetry complex class needs only a ``@gui_exposed``-decorated subclass
and one entry in :data:`COMPLEX_CLASSES`; the tab auto-discovers its
constructor form and all instance/class-method operations.

Contract: exposes :data:`layout` and :func:`register_callbacks(app)`.
"""
from __future__ import annotations

import inspect
from datetime import datetime
from typing import Any

import dash
from dash import html, dcc, Input, Output, State, ALL, ctx, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd

from cryocat.core.cryomotl import EmMotl, Motl
from cryocat.core.surface import Mesh
from cryocat.analysis.structure import (
    CnComplex, DnComplex, NPC,
    TetrahedralComplex, OctahedralComplex, IcosahedralComplex,
    PleomorphicSurface,
)
from cryocat.app import ids, formgen, discovery
from cryocat.utils.classutils import GuiCategory
from cryocat.app.formgen import make_dropdown
from cryocat.app.apputils import generate_kwargs, run_operation
from cryocat.app.logger import invoke_operation as _invoke_op
from cryocat.app.components import complex_registry as cr
from cryocat.app.components.complex_registry import COMPLEX_BUILDERS
from cryocat.app.components import surface_registry as sr
from cryocat.app.components.motlsource import (
    get_motl_source, register_motl_source_callbacks,
    get_multi_motl_picker, register_multi_motl_picker_callbacks,
)
from cryocat.app.components.resultsslot import (
    get_results_slot, register_results_slot_callbacks,
)
from cryocat.app.components.tabletomotl import (
    get_table_to_motl, register_table_to_motl_callbacks,
)
from cryocat.app.pageshell import page_shell
import cryocat.app.pool as _pool
import cryocat.app.datapool as _datapool
import cryocat.app.provenance as _prov
from cryocat.app import session as _session


# ── Registry of supported complex classes ────────────────────────────────────

COMPLEX_CLASSES: dict[str, type] = {
    "CnComplex":             CnComplex,
    "DnComplex":             DnComplex,
    "NPC":                   NPC,
    "TetrahedralComplex":    TetrahedralComplex,
    "OctahedralComplex":     OctahedralComplex,
    "IcosahedralComplex":    IcosahedralComplex,
    "Pleomorphic assembly":  PleomorphicSurface,
}

# Hierarchy groups for the class picker (D3).
# Each group maps to the concrete subclasses that belong to it.
_HIERARCHY: list[tuple[str, list[str]]] = [
    ("Cyclic",       ["CnComplex", "NPC"]),
    ("Dihedral",     ["DnComplex"]),
    ("Polyhedral",   ["TetrahedralComplex", "OctahedralComplex", "IcosahedralComplex"]),
    ("Pleomorphic",  ["Pleomorphic assembly"]),
]

# Grouped dropdown options: group headers (disabled) + concrete entries.
_CLASS_OPTIONS: list[dict] = []
for _group_label, _names in _HIERARCHY:
    _CLASS_OPTIONS.append({"label": f"── {_group_label} ──", "value": f"__group__{_group_label}", "disabled": True})
    for _n in _names:
        _CLASS_OPTIONS.append({"label": _n, "value": _n})

# ── Store IDs ─────────────────────────────────────────────────────────────────

_CPX_POOL    = "cpx-pool-store"      # list[dict] of ComplexHandle dicts
_CPX_SEL     = "cpx-selected-store"  # str | None — selected complex_id
_CPX_RESULTS = "cpx-results-store"   # dict[str, list[dict]] — sections per complex_id

# Form id-type strings (must not collide with any other page)
_INIT     = "cpx-init-param"
_METH     = "cpx-meth-param"

_HINT = {"color": "var(--color9)", "margin": "0.3rem 0"}
_HDR  = {"fontWeight": 600, "margin": "0.4rem 0 0.2rem"}


# ── Pure helpers ──────────────────────────────────────────────────────────────

def _cls_for_handle(handle: dict) -> type | None:
    """Return the class for a handle, matching by key or class __name__."""
    cls_str = handle.get("cls", "")
    return COMPLEX_CLASSES.get(cls_str) or next(
        (v for v in COMPLEX_CLASSES.values() if v.__name__ == cls_str), None
    )


def motl_from_pool_rows(motl_id: str | None) -> Motl | None:
    """Rebuild a :class:`~cryocat.core.cryomotl.Motl` from the server-side pool."""
    if not motl_id:
        return None
    try:
        motl = EmMotl(_pool.get_rows(motl_id))
        motl._pool_motl_id = motl_id
        return motl
    except _pool.PoolPayloadMissing:
        return None


def motl_to_pool_rows(motl: Motl | None) -> list[dict]:
    """Serialise a :class:`~cryocat.core.cryomotl.Motl` to pool-store rows."""
    return motl.df.to_dict("records") if motl is not None else []


def _motl_from_pool(motl_id: str | None) -> Motl | None:
    return motl_from_pool_rows(motl_id)


def _get_live_complex(complex_id: str, handle: dict):
    """Return the live complex: from server registry first, else reconstruct."""
    live = cr.registry.get(complex_id)
    if live is not None:
        return live
    from cryocat.app.suite.pages._motl_link import get_motl_role_id
    motl = _motl_from_pool(get_motl_role_id(handle.get("motl_links"), "source"))
    if motl is None:
        return None
    try:
        return cr.reconstruct(handle, motl)
    except Exception:
        return None


def _dispatch_result(
    entry,
    result: Any,
    cpx: Any,
    handle: dict,
) -> tuple[list, list, list, str]:
    """Route a method result to (motl_rows, df_records, feat_records, status)."""
    label = entry.label
    kind  = entry.returns

    if kind == "motl":
        if isinstance(result, list):
            combined_df = pd.concat([m.df for m in result if isinstance(m, Motl)], ignore_index=True)
            rows = combined_df.to_dict("records")
            return rows, [], [], f"{label} → {len(rows)} particles from {len(result)} merged motls."
        if not isinstance(result, Motl):
            return [], [], [], f"{label}: expected Motl, got {type(result).__name__}."
        rows = result.df.to_dict("records")
        return rows, [], [], f"{label} → {len(rows)} particles."

    if kind == "none":
        # in-place: expose the updated cpx.motl if present
        try:
            if cpx is not None:
                cr.registry.replace(handle["complex_id"], cpx)
        except KeyError:
            pass
        rows = cpx.motl.df.to_dict("records") if (cpx is not None and hasattr(cpx, "motl")) else []
        return rows, [], [], f"{label} done → {len(rows)} particles."

    if kind == "dataframe":
        records = result.to_dict("records") if isinstance(result, pd.DataFrame) else []
        return [], records, [], f"{label} → {len(records)} rows."

    if kind == "features":
        arr = result[0] if isinstance(result, tuple) else result
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            nc   = arr.shape[1]
            cols = (["x", "y", "z"] + [str(i) for i in range(3, nc)])[:nc]
            records = pd.DataFrame(arr, columns=cols).to_dict("records")
        else:
            records = []
        return [], [], records, f"{label} → {len(records)} feature points."

    if kind == "surface":
        if not isinstance(result, Mesh):
            return [], [], [], f"{label}: expected Mesh, got {type(result).__name__}."
        psurf = PleomorphicSurface(result)
        sr.registry.add(psurf)
        surf_label = f"{handle.get('label', '')} — {label}"
        return [], [], [], f"{label} → surface '{surf_label}' registered; open the Surfaces tab to use it."

    return [], [], [], f"{label}: unknown returns kind {kind!r}."


def _render_table(records: list | None, empty_msg: str = "No results.") -> Any:
    if not records:
        return html.Small(empty_msg, style=_HINT)
    df = pd.DataFrame(records)
    if df.empty:
        return html.Small("No results.", style=_HINT)
    header = html.Thead([html.Tr([
        html.Th(c) for c in df.columns
    ])])
    body = []
    for row in records:
        cells = []
        for c in df.columns:
            val = row.get(c)
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
        body.append(html.Tr(cells))
    return dbc.Table(
        [header, html.Tbody(body)],
        bordered=True, striped=True, hover=True, size="sm",
        style={"overflowX": "auto"},
    )


# ── Layout helpers ────────────────────────────────────────────────────────────

def _build_section() -> dbc.AccordionItem:
    return dbc.AccordionItem(
        [
            html.Div("Complex type", style=_HDR),
            make_dropdown("cpx-class-dd", _CLASS_OPTIONS, None, clearable=False,
                          placeholder="Select complex type…"),
            html.Hr(style={"margin": "0.4rem 0"}),
            get_multi_motl_picker("cpx-build"),
            html.Hr(style={"margin": "0.4rem 0"}),
            html.Div("Init parameters", style=_HDR),
            html.Div(id="cpx-init-form"),
            dbc.Button(
                "Create", id="cpx-create-btn", color="secondary", size="sm",
                style={"width": "100%", "marginTop": "0.5rem"},
                disabled=True,
            ),
            html.Div(id="cpx-create-status", style={**_HINT, "wordBreak": "break-word"}),
        ],
        title="Build complex",
        item_id="cpx-build-item",
    )


def _handles_section() -> dbc.AccordionItem:
    return dbc.AccordionItem(
        [html.Div(id="cpx-handles-list",
                  children=[html.Small("No complexes created yet.", style=_HINT)])],
        title="My complexes",
        item_id="cpx-handles-item",
    )


def _ops_section() -> dbc.AccordionItem:
    return dbc.AccordionItem(
        [
            html.Div(id="cpx-sel-info", style={**_HINT, "marginBottom": "0.3rem"}),
            make_dropdown("cpx-method-dd", [], None, clearable=True,
                          placeholder="Select operation…"),
            html.Div(id="cpx-meth-form", style={"marginTop": "0.4rem"}),
            dbc.Button(
                "Run", id="cpx-run-btn", color="primary", size="sm",
                style={"width": "100%", "marginTop": "0.4rem"},
            ),
            html.Div(id="cpx-run-status", style={**_HINT, "wordBreak": "break-word"}),
        ],
        title="Operations",
        item_id="cpx-ops-item",
    )


def _sidebar() -> list:
    return [
        dbc.Accordion(
            [_build_section(), _handles_section(), _ops_section()],
            always_open=True,
            active_item=["cpx-build-item"],
        ),
        dcc.Store(id=_CPX_POOL, data=[]),
        dcc.Store(id=_CPX_SEL, data=None),
        dcc.Store(id="cpx-bdef-specs", data=None),
    ]


def _main() -> list:
    return [
        html.Div("Results", style=_HDR),
        html.Div(id="cpx-results-area",
                 children=[html.Small("No complex selected.", style=_HINT)]),
        dcc.Store(id=_CPX_RESULTS, data={}),
        html.Hr(style={"margin": "0.6rem 0"}),
        html.Div("DataFrame results", style=_HDR),
        get_results_slot("cpx-df"),
        html.Hr(style={"margin": "0.6rem 0"}),
        html.Div("Table → motl", style=_HDR),
        get_table_to_motl("cpx-ttm", show_rows_mode=False),
    ]


_BDEF_MODAL = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Configure block types")),
        dbc.ModalBody(
            [
                html.Div("Single block type", style=_HDR),
                html.Div(
                    [
                        dbc.Label("Symmetry", html_for="cpx-bdef-symmetry", width=3),
                        dbc.Col(
                            dbc.Input(id="cpx-bdef-symmetry", placeholder="e.g. C3, C6", type="text"),
                            width=9,
                        ),
                    ],
                    className="row mb-2",
                ),
                html.Div("Site shift (x, y, z in voxels)", style=_HDR),
                html.Div(
                    [
                        dbc.Col(dbc.Input(id="cpx-bdef-shift-x", placeholder="x", type="number"), width=4),
                        dbc.Col(dbc.Input(id="cpx-bdef-shift-y", placeholder="y", type="number"), width=4),
                        dbc.Col(dbc.Input(id="cpx-bdef-shift-z", placeholder="z", type="number"), width=4),
                    ],
                    className="row",
                ),
            ]
        ),
        dbc.ModalFooter(
            [
                dbc.Button("Save", id="cpx-bdef-save-btn", color="primary", size="sm"),
                dbc.Button("Close", id="cpx-bdef-close-btn", color="secondary", size="sm",
                           className="ms-2"),
                html.Div(id="cpx-bdef-modal-status", style={**_HINT, "flex": "1"}),
            ],
            style={"display": "flex", "alignItems": "center", "gap": "0.4rem"},
        ),
    ],
    id="cpx-bdef-modal",
    is_open=False,
    size="md",
)


layout: Any = html.Div(
    [
        page_shell(_sidebar(), _main(), sidebar_width=4),
        _BDEF_MODAL,
    ],
    style={"margin": 0, "padding": 0},
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

def _resolve_slot_df(slot_data: dict | None) -> pd.DataFrame | None:
    """Resolve the cpx-df results-slot store to a DataFrame for table→motl."""
    if not slot_data or not slot_data.get("records"):
        return None
    return pd.DataFrame(slot_data["records"])


def register_callbacks(app: dash.Dash) -> None:  # noqa: C901
    formgen.register_form_callbacks(app, _INIT)
    formgen.register_form_callbacks(app, _METH)
    register_multi_motl_picker_callbacks(app, "cpx-build")
    register_results_slot_callbacks(app, "cpx-df", _CPX_SEL)
    register_table_to_motl_callbacks(
        app, "cpx-ttm",
        source_store_id="cpx-df-slot-data",
        resolve_df=_resolve_slot_df,
    )

    # ── Block-type modal ──────────────────────────────────────────────────────

    @app.callback(
        Output("cpx-bdef-modal", "is_open"),
        Input("cpx-bdef-open-btn", "n_clicks"),
        Input("cpx-bdef-close-btn", "n_clicks"),
        Input("cpx-bdef-save-btn", "n_clicks"),
        State("cpx-bdef-modal", "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_bdef_modal(open_clicks, close_clicks, save_clicks, is_open):
        trigger = ctx.triggered_id
        if trigger == "cpx-bdef-open-btn":
            return True
        return False

    @app.callback(
        Output("cpx-bdef-specs", "data"),
        Output("cpx-bdef-modal-status", "children"),
        Input("cpx-bdef-save-btn", "n_clicks"),
        State("cpx-bdef-symmetry", "value"),
        State("cpx-bdef-shift-x", "value"),
        State("cpx-bdef-shift-y", "value"),
        State("cpx-bdef-shift-z", "value"),
        prevent_initial_call=True,
    )
    def _save_bdef_specs(_, symmetry, sx, sy, sz):
        if not symmetry:
            return no_update, "Symmetry is required."
        try:
            shift = [float(sx or 0), float(sy or 0), float(sz or 0)]
        except (TypeError, ValueError):
            return no_update, "Site shift must be numeric."
        return {"symmetry": symmetry, "site_shift": shift}, f"Saved: {symmetry}, shift={shift}"

    # 1. Rebuild init form, show motl picker, and toggle Create button when class changes
    @app.callback(
        Output("cpx-init-form", "children"),
        Output("cpx-create-btn", "disabled"),
        Output("cpx-build-list-picker", "style"),
        Input("cpx-class-dd", "value"),
        prevent_initial_call=True,
    )
    def _update_init_form(cls_name: str | None):
        if not cls_name:
            return [], True, {"display": "none"}
        cls = COMPLEX_CLASSES.get(cls_name)
        if cls is None:
            return [], True, {"display": "none"}
        builder = COMPLEX_BUILDERS.get(cls, cls)
        first_param = next(iter(inspect.signature(builder).parameters))
        # NPC has fixed C8 symmetry; hide the symmetry param from the form.
        if cls_name == "NPC":
            exclude = [first_param, "symmetry"]
            rows = formgen.build_form(builder, id_type=_INIT, id_extra={}, exclude=exclude)
        elif cls_name == "Pleomorphic assembly":
            rows = [
                html.Div(id="cpx-bdef-btn-area", children=[
                    dbc.Button(
                        "Configure block types",
                        id="cpx-bdef-open-btn",
                        color="info",
                        size="sm",
                        style={"width": "100%", "marginTop": "0.3rem"},
                    ),
                ]),
                html.Div(id="cpx-bdef-summary", style=_HINT),
            ]
        else:
            exclude = [first_param]
            rows = formgen.build_form(builder, id_type=_INIT, id_extra={}, exclude=exclude)
        return rows, False, {}

    # 2. Create complex → add to server registry and pool
    @app.callback(
        Output(_CPX_POOL, "data"),
        Output(_CPX_SEL, "data"),
        Output("cpx-create-status", "children"),
        Output(_CPX_RESULTS, "data", allow_duplicate=True),
        Input("cpx-create-btn", "n_clicks"),
        State("cpx-class-dd", "value"),
        State("cpx-build-list-select", "value"),
        State({"type": _INIT, "owner": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": _INIT, "owner": ALL, "param": ALL, "tag": ALL}, "id"),
        State(_CPX_POOL, "data"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        State(_CPX_RESULTS, "data"),
        State("cpx-bdef-specs", "data"),
        prevent_initial_call=True,
    )
    def _create_complex(_, cls_name, motl_ids, init_vals, init_ids, pool_data, registry, pool_meta, pool_next_id, results_store, bdef_specs):
        if not cls_name or not motl_ids:
            raise PreventUpdate
        cls = COMPLEX_CLASSES.get(cls_name)
        if cls is None:
            raise PreventUpdate

        # NPC accepts a list of motls; all other classes take a single motl.
        if cls is NPC:
            motl_list = [_motl_from_pool(mid) for mid in motl_ids]
            if any(m is None for m in motl_list):
                return no_update, no_update, "One or more selected motls could not be loaded.", no_update
            motl = motl_list if len(motl_list) > 1 else motl_list[0]
            motl_id = motl_ids[0]
        else:
            motl_id = motl_ids[0] if isinstance(motl_ids, list) else motl_ids
            motl = _motl_from_pool(motl_id)
            if motl is None:
                return no_update, no_update, "No motl data found for the selected motl.", no_update

        pool_state = _pool.PoolState.from_stores(registry, pool_meta, pool_next_id)
        init_kwargs = generate_kwargs(init_ids, init_vals, pool_state) if (init_ids and init_vals) else {}
        init_kwargs = {k: v for k, v in init_kwargs.items() if v not in (None, "", [])}

        if cls is PleomorphicSurface:
            if not bdef_specs:
                return (no_update, no_update,
                        "No block types configured — click 'Configure block types' first.",
                        no_update)
            next_complex_id = cr.registry.peek_next_key()
            cpx_var = _prov.bind(next_complex_id)
            try:
                cpx = _invoke_op(
                    PleomorphicSurface.from_blocks,
                    {
                        "blocks": motl,
                        "symmetry": bdef_specs["symmetry"],
                        "site_shift": bdef_specs["site_shift"],
                        **init_kwargs,
                    },
                    assign_to=cpx_var,
                )
            except Exception as exc:
                return no_update, no_update, f"Init failed: {exc}", no_update
        else:
            builder = COMPLEX_BUILDERS.get(cls, cls)
            first_param = next(iter(inspect.signature(builder).parameters))
            next_complex_id = cr.registry.peek_next_key()
            cpx_var = _prov.bind(next_complex_id)
            try:
                cpx = _invoke_op(builder, {first_param: motl, **init_kwargs}, assign_to=cpx_var)
            except Exception as exc:
                return no_update, no_update, f"Init failed: {exc}", no_update

        complex_id = cr.registry.add(cpx)
        cpx._pool_complex_id = complex_id
        _prov.record(complex_id, _session.last_seq())
        label = f"{cpx_var} ({cls_name})"
        handle = cr.make_handle(cpx, complex_id, label, {"source": [motl_id]} if motl_id else {}, init_kwargs)

        new_pool = list(pool_data or []) + [handle]
        new_results = dict(results_store or {})
        new_results[complex_id] = []
        return new_pool, complex_id, f"Created {label}.", new_results

    # 3. Render clickable handles list with close buttons
    @app.callback(
        Output("cpx-handles-list", "children"),
        Input(_CPX_POOL, "data"),
        Input(_CPX_SEL, "data"),
        prevent_initial_call=True,
    )
    def _render_handles(pool_data, selected_id):
        if not pool_data:
            return html.Small("No complexes created yet.", style=_HINT)
        items = []
        for h in pool_data:
            cid    = h.get("complex_id", "")
            is_sel = cid == selected_id
            info   = f"{h.get('cls', '')} · n={h.get('n_subunits', '?')}"
            items.append(html.Div(
                [
                    dbc.Button(
                        [html.Strong(h.get("label", cid)), html.Br(),
                         html.Small(info, style={"color": "var(--color9)"})],
                        id={"type": "cpx-handle-btn", "id": cid},
                        color="primary" if is_sel else "secondary",
                        outline=not is_sel,
                        size="sm",
                        style={"flex": "1", "textAlign": "left"},
                        n_clicks=0,
                    ),
                    dbc.Button(
                        "×",
                        id={"type": "cpx-close-btn", "id": cid},
                        color="danger",
                        outline=True,
                        size="sm",
                        style={"marginLeft": "0.25rem", "padding": "0 0.4rem"},
                        n_clicks=0,
                    ),
                ],
                style={"display": "flex", "marginBottom": "0.25rem"},
            ))
        return items

    # 4. Select complex by clicking its handle button
    @app.callback(
        Output(_CPX_SEL, "data", allow_duplicate=True),
        Input({"type": "cpx-handle-btn", "id": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _select_handle(n_clicks_list):
        if not any(n_clicks_list):
            raise PreventUpdate
        triggered = ctx.triggered_id
        if triggered is None or not isinstance(triggered, dict):
            raise PreventUpdate
        return triggered["id"]

    # 5. Update method dropdown and info when a complex is selected
    @app.callback(
        Output("cpx-method-dd", "options"),
        Output("cpx-method-dd", "value"),
        Output("cpx-sel-info", "children"),
        Input(_CPX_SEL, "data"),
        State(_CPX_POOL, "data"),
        prevent_initial_call=True,
    )
    def _update_methods(selected_id, pool_data):
        if not selected_id:
            return [], None, ""
        handle = next(
            (h for h in (pool_data or []) if h.get("complex_id") == selected_id), None
        )
        if handle is None:
            return [], None, ""
        cls = _cls_for_handle(handle)
        if cls is None:
            return [], None, ""
        allowed_cats = {GuiCategory.MOTL_OP}
        if cls is PleomorphicSurface:
            allowed_cats.add(GuiCategory.PLEOMORPHIC_OP)
        entries = [
            e for e in discovery.entries_for_class(cls)
            if e.category in allowed_cats
        ]
        geometry_fitted = handle.get("geometry_fitted", False)

        # Build grouped options with headers
        opts: list[dict] = []
        cur_group: str | None = None
        for e in entries:
            g = e.group or ""
            if g != cur_group:
                cur_group = g
                if g:
                    opts.append({"label": f"── {g} ──", "value": f"__group__{g}", "disabled": True})
            disabled = (e.label == "Expand to subparticles" and not geometry_fitted)
            opts.append({"label": e.label, "value": e.key, "disabled": disabled})

        radius_str = f" · r={handle['radius']:.1f}px" if handle.get("radius") else ""
        geo_str = " [geometry fitted]" if geometry_fitted else " [no geometry]"
        info = f"{handle['label']} · {handle.get('n_objects', '?')} objects{radius_str}{geo_str}"
        return opts, None, info

    # 6. Rebuild method form when an operation is selected
    @app.callback(
        Output("cpx-meth-form", "children"),
        Input("cpx-method-dd", "value"),
        State(_CPX_SEL, "data"),
        State(_CPX_POOL, "data"),
        prevent_initial_call=True,
    )
    def _update_meth_form(entry_key: str | None, selected_id, pool_data):
        if not entry_key or not selected_id:
            return []
        try:
            entry = discovery.get(entry_key)
        except KeyError:
            return []
        return formgen.build_form(entry, id_type=_METH, id_extra={})

    # 7. Run the selected method
    @app.callback(
        Output("cpx-run-status", "children"),
        Output(_CPX_POOL, "data", allow_duplicate=True),
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output(ids.DATA_POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.DATA_POOL_NEXT_ID, "data", allow_duplicate=True),
        Output(_CPX_RESULTS, "data", allow_duplicate=True),
        Output("cpx-df-obj-results", "data", allow_duplicate=True),
        Input("cpx-run-btn", "n_clicks"),
        State("cpx-method-dd", "value"),
        State(_CPX_SEL, "data"),
        State(_CPX_POOL, "data"),
        State({"type": _METH, "owner": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": _METH, "owner": ALL, "param": ALL, "tag": ALL}, "id"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        State(ids.DATA_POOL_REGISTRY, "data"),
        State(ids.DATA_POOL_NEXT_ID, "data"),
        State(_CPX_RESULTS, "data"),
        State("cpx-df-obj-results", "data"),
        prevent_initial_call=True,
    )
    def _run_method(
        _, entry_key, selected_id, pool_data, meth_vals, meth_ids,
        registry, pool_meta, pool_next_id,
        dp_registry, dp_next_id, results_store, obj_results_store,
    ):
        _nu9 = (no_update,) * 9

        if not entry_key or not selected_id:
            raise PreventUpdate

        handle = next(
            (h for h in (pool_data or []) if h.get("complex_id") == selected_id), None
        )
        if handle is None:
            return "No complex selected.", *((no_update,) * 8)

        try:
            entry = discovery.get(entry_key)
        except KeyError:
            return f"Unknown entry {entry_key!r}.", *((no_update,) * 8)

        pool_state = _pool.PoolState.from_stores(registry, pool_meta, pool_next_id)
        meth_kwargs = generate_kwargs(meth_ids, meth_vals, pool_state) if (meth_ids and meth_vals) else {}
        meth_kwargs = {k: v for k, v in meth_kwargs.items() if v not in (None, "", [])}

        # Pre-compute pool id so the log event carries assign_to before adoption.
        # next_id + 1 assumes no concurrent insert between here and insert_motl;
        # safe under Dash's single-threaded callback model.
        _assign_to: str | None = None
        _pool_id: str | None = None
        _dp_state: _datapool.DataPoolState | None = None
        if entry.returns == "motl":
            _pool_id = f"motl_{pool_state.next_id + 1}"
            _assign_to = _prov.bind(_pool_id)
        elif entry.returns == "dataframe":
            # Dataframe results go to the per-page slot, not the data pool.
            _assign_to = "results_cpx"

        try:
            if entry.kind in ("classmethod", "staticmethod"):
                cls = _cls_for_handle(handle)
                if cls is None:
                    return "Unknown complex class.", *((no_update,) * 8)
                fn     = getattr(cls, entry.fn.__name__)
                result = _invoke_op(fn, meth_kwargs, assign_to=_assign_to, pool_id=_pool_id)
                cpx    = None
            else:
                cpx = _get_live_complex(handle["complex_id"], handle)
                if cpx is None:
                    return (
                        "Complex not available — reload or recreate it.", *((no_update,) * 8)
                    )
                fn     = getattr(cpx, entry.fn.__name__)
                result = _invoke_op(fn, meth_kwargs, assign_to=_assign_to, pool_id=_pool_id)
        except Exception as exc:
            return f"{entry.label} failed: {exc}", *((no_update,) * 8)

        motl_rows, df_records, feat_records, status = _dispatch_result(
            entry, result, cpx, handle
        )

        # After any in-place ("none") or motl-returning method, recompute the
        # handle so that geometry_fitted / radius are kept in sync in the pool.
        new_cpx_pool = no_update
        if entry.returns in ("none", "motl") and cpx is not None:
            updated_handle = cr.make_handle(
                cpx,
                handle["complex_id"],
                handle.get("label", handle["complex_id"]),
                handle.get("motl_links") or {},
                handle.get("init_kwargs", {}),
            )
            new_cpx_pool = [
                updated_handle if h.get("complex_id") == handle["complex_id"] else h
                for h in (pool_data or [])
            ]

        # GX4: route motl results into the motl pool; capture pool entry id
        new_pool_reg = no_update
        new_pool_meta = no_update
        new_pool_next = no_update
        pool_ref: str | None = None

        # When an in-place op runs after a motl-returning op, update the existing
        # pool entry so the editor sees the latest particles (e.g. post-merge SU IDs).
        if entry.returns == "none" and cpx is not None:
            existing_motl_section = next(
                (
                    s for s in (results_store or {}).get(selected_id, [])
                    if s.get("key") == "_results_motl" and s.get("pool_ref")
                ),
                None,
            )
            if existing_motl_section:
                pool_state = _pool.replace_motl_rows(
                    pool_state,
                    existing_motl_section["pool_ref"],
                    cpx.motl.df,
                )
                new_pool_reg, new_pool_meta, new_pool_next = pool_state.to_stores()

        if entry.returns == "motl" and isinstance(result, Motl):
            new_ps, pool_ref = _pool.insert_motl(
                pool_state, result.df, label=entry.label,
            )
            new_pool_reg, new_pool_meta, new_pool_next = new_ps.to_stores()
            result._pool_motl_id = pool_ref
            _prov.record(pool_ref, _session.last_seq())
        elif entry.returns == "motl" and isinstance(result, list) and any(
            isinstance(m, Motl) for m in result
        ):
            # Per-ring result from NPC __getattribute__ dispatch: each ring → own pool entry.
            ps = pool_state
            ring_refs: list[str] = []
            for ring_idx, ring_motl in enumerate(result):
                if not isinstance(ring_motl, Motl):
                    continue
                ps, ref = _pool.insert_motl(
                    ps, ring_motl.df, label=f"{entry.label} ring {ring_idx}",
                )
                _prov.record(ref, _session.last_seq())
                ring_refs.append(ref)
            if ring_refs:
                new_pool_reg, new_pool_meta, new_pool_next = ps.to_stores()
                pool_ref = ring_refs[0]

        # GX4: dataframe results go to the per-object slot (not the data pool).
        # Use Send to editor in the slot to create a pool entry on demand.
        new_dp_reg = no_update
        new_dp_next = no_update
        new_obj_results = no_update
        if entry.returns == "dataframe" and isinstance(result, pd.DataFrame):
            _slot = {
                "records": result.head(200).to_dict("records"),
                "label": entry.label,
                "n_rows": len(result),
                "pool_label": f"results_cpx_{selected_id}",
            }
            _nr = dict(obj_results_store or {})
            _nr[selected_id] = _slot
            new_obj_results = _nr

        # Build result section descriptor and update per-complex results store
        count = len(motl_rows or []) or len(df_records or []) or len(feat_records or [])
        ts = datetime.now().strftime("%H:%M:%S")

        # Extra metadata for in-place ("none") operations
        n_objects: int | None = None
        column_written: str | None = None
        if entry.returns == "none" and cpx is not None:
            if hasattr(cpx, "affiliation_column"):
                n_objects = int(cpx.motl.df[cpx.affiliation_column].nunique())
            fn_name = getattr(entry.fn, "__name__", "")
            if fn_name == "assign_subunit_order" and hasattr(cpx, "order_column"):
                column_written = cpx.order_column
            elif fn_name == "merge_subunits" and hasattr(cpx, "affiliation_column"):
                column_written = cpx.affiliation_column
            elif fn_name == "unify_nn_orientations":
                column_written = "phi/theta/psi"

        # motl-returning methods all overwrite the same slot so the results
        # panel shows the latest merged motl, not a growing list of old ones.
        section_key = "_results_motl" if entry.returns == "motl" else entry_key

        section = {
            "key": section_key,
            "label": entry.label,
            "time": ts,
            "kind": entry.returns,
            "pool_ref": pool_ref,
            "count": count,
            "n_objects": n_objects,
            "column_written": column_written,
        }
        new_results = dict(results_store or {})
        cpx_sections = list(new_results.get(selected_id, []))
        replaced = False
        for i, s in enumerate(cpx_sections):
            if s.get("key") == section_key:
                cpx_sections[i] = section
                replaced = True
                break
        if not replaced:
            cpx_sections.append(section)
        new_results[selected_id] = cpx_sections

        return (
            status,
            new_cpx_pool,
            new_pool_reg,
            new_pool_meta,
            new_pool_next,
            new_dp_reg,
            new_dp_next,
            new_results,
            new_obj_results,
        )

    # 8. Render collapsible result sections for the selected complex
    @app.callback(
        Output("cpx-results-area", "children"),
        Input(_CPX_SEL, "data"),
        Input(_CPX_RESULTS, "data"),
        prevent_initial_call=True,
    )
    def _render_results_area(selected_id, results_store):
        if not selected_id:
            return html.Small("No complex selected.", style=_HINT)
        sections = (results_store or {}).get(selected_id, [])
        if not sections:
            return html.Small("No results yet for this complex.", style=_HINT)
        items = []
        for i, sec in enumerate(sections):
            is_last = i == len(sections) - 1
            kind = sec.get("kind", "")
            pool_ref = sec.get("pool_ref")
            count = sec.get("count", 0)
            if kind == "dataframe":
                body = [html.Small(f"{count} rows", style=_HINT)]
                if pool_ref:
                    body.append(html.Small(f"Data pool entry: {pool_ref}", style=_HINT))
                records = sec.get("records")
                if records:
                    body.append(_render_table(records))
            elif kind == "features":
                body = [html.Small(f"{count} feature points", style=_HINT)]
                if pool_ref:
                    body.append(html.Small(f"Data pool entry: {pool_ref}", style=_HINT))
            elif kind == "motl":
                body = [html.Small(f"{count} particles", style=_HINT)]
                if pool_ref:
                    body.append(html.Small(f"Motl pool entry: {pool_ref}", style=_HINT))
            else:  # "none" — in-place operation
                n_obj = sec.get("n_objects")
                col_wr = sec.get("column_written")
                detail = f"{count} particles"
                if n_obj is not None:
                    detail += f" across {n_obj} objects"
                body = [html.Small(detail, style=_HINT)]
                if col_wr:
                    body.append(html.Small(f"→ {col_wr}", style=_HINT))
            items.append(html.Details(
                [
                    html.Summary(
                        f"{sec['label']}  ·  {sec['time']}",
                        style={"fontSize": "0.85rem", "cursor": "pointer"},
                    ),
                    *body,
                ],
                open=is_last,
                style={"marginBottom": "0.3rem"},
            ))
        return items

    # 9. Close complex — remove from pool and clear its results
    @app.callback(
        Output(_CPX_POOL, "data", allow_duplicate=True),
        Output(_CPX_SEL, "data", allow_duplicate=True),
        Output(_CPX_RESULTS, "data", allow_duplicate=True),
        Output("cpx-df-obj-results", "data", allow_duplicate=True),
        Input({"type": "cpx-close-btn", "id": ALL}, "n_clicks"),
        State(_CPX_POOL, "data"),
        State(_CPX_SEL, "data"),
        State(_CPX_RESULTS, "data"),
        State("cpx-df-obj-results", "data"),
        prevent_initial_call=True,
    )
    def _close_complex(n_clicks_list, pool_data, selected_id, results_store, obj_results):
        if not any(n_clicks_list):
            raise PreventUpdate
        triggered = ctx.triggered_id
        if triggered is None or not isinstance(triggered, dict):
            raise PreventUpdate
        cid = triggered["id"]
        new_pool = [h for h in (pool_data or []) if h.get("complex_id") != cid]
        new_sel = None if selected_id == cid else selected_id
        new_results = {k: v for k, v in (results_store or {}).items() if k != cid}
        new_obj_results = {k: v for k, v in (obj_results or {}).items() if k != cid}
        return new_pool, new_sel, new_results, new_obj_results
