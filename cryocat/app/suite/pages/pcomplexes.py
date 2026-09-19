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
    BlockDefinition,
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
)
from cryocat.app.components.motlsink import (
    get_send_to_editor_button, register_send_to_editor_callbacks,
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

_CPX_POOL     = "cpx-pool-store"        # list[dict] of ComplexHandle dicts
_CPX_SEL      = "cpx-selected-store"    # str | None — selected complex_id
_CPX_RES_MOTL = "cpx-result-motl"       # list[dict] | None — motl rows (send-to-editor)
_CPX_RESULTS  = "cpx-results-store"     # dict[str, list[dict]] — sections per complex_id

# Form id-type strings (must not collide with any other page)
_INIT     = "cpx-init-param"
_METH     = "cpx-meth-param"
_BDEF_ID  = "cpx-bdef-param"

# Block-definition creators exposed in the GUI.
_BLOCK_DEF_CREATORS: dict[str, tuple[str, Any]] = {
    "cyclic":      ("Block definition – cyclic",      BlockDefinition.cyclic),
    "microtubule": ("Block definition – microtubule", BlockDefinition.microtubule),
}

# Single current block definition — replaced on each creation, never accumulated.
_current_bdef: BlockDefinition | None = None

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
            get_motl_source("cpx-build"),
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


def _build_bdef_section() -> dbc.AccordionItem:
    creator_opts = [{"label": lbl, "value": k} for k, (lbl, _) in _BLOCK_DEF_CREATORS.items()]
    return dbc.AccordionItem(
        [
            html.Div("Creator", style=_HDR),
            make_dropdown("cpx-bdef-creator-dd", creator_opts, None,
                          placeholder="Select creator…"),
            html.Div(id="cpx-bdef-form", style={"marginTop": "0.4rem"}),
            dbc.Button(
                "Create definition",
                id="cpx-bdef-create-btn",
                color="secondary",
                size="sm",
                style={"width": "100%", "marginTop": "0.5rem"},
                disabled=True,
            ),
            html.Div(id="cpx-bdef-status", style={**_HINT, "wordBreak": "break-word"}),
            html.Hr(style={"margin": "0.4rem 0"}),
            html.Div(id="cpx-bdef-list"),
        ],
        title="Block definitions",
        item_id="cpx-bdef-item",
    )


def _sidebar() -> list:
    return [
        dbc.Accordion(
            [_build_section(), _handles_section(), _ops_section(), _build_bdef_section()],
            always_open=True,
            active_item=["cpx-build-item"],
        ),
        dcc.Store(id=_CPX_POOL, data=[]),
        dcc.Store(id=_CPX_SEL, data=None),
    ]


def _main() -> list:
    return [
        html.Div("Result motl", style=_HDR),
        get_send_to_editor_button("cpx-export"),
        dcc.Store(id=_CPX_RES_MOTL),
        html.Hr(style={"margin": "0.6rem 0"}),
        html.Div("Results", style=_HDR),
        html.Div(id="cpx-results-area",
                 children=[html.Small("No complex selected.", style=_HINT)]),
        dcc.Store(id=_CPX_RESULTS, data={}),
    ]


layout: Any = html.Div(
    [
        page_shell(_sidebar(), _main(), sidebar_width=4),
    ],
    style={"margin": 0, "padding": 0},
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app: dash.Dash) -> None:  # noqa: C901
    formgen.register_form_callbacks(app, _INIT)
    formgen.register_form_callbacks(app, _METH)
    formgen.register_form_callbacks(app, _BDEF_ID)
    register_motl_source_callbacks(app, "cpx-build")
    register_send_to_editor_callbacks(app, "cpx-export", _CPX_RES_MOTL)

    # ── Block-definition panel ────────────────────────────────────────────────

    @app.callback(
        Output("cpx-bdef-form", "children"),
        Output("cpx-bdef-create-btn", "disabled"),
        Input("cpx-bdef-creator-dd", "value"),
        prevent_initial_call=True,
    )
    def _render_bdef_form(creator_key: str | None):
        if not creator_key or creator_key not in _BLOCK_DEF_CREATORS:
            return [], True
        _, method = _BLOCK_DEF_CREATORS[creator_key]
        rows = formgen.build_form(method, id_type=_BDEF_ID, id_extra={"op": creator_key}, exclude=[])
        return rows, False

    @app.callback(
        Output("cpx-bdef-status", "children"),
        Output("cpx-bdef-list", "children"),
        Input("cpx-bdef-create-btn", "n_clicks"),
        State("cpx-bdef-creator-dd", "value"),
        State({"type": _BDEF_ID, "owner": ALL, "op": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": _BDEF_ID, "owner": ALL, "op": ALL, "param": ALL, "tag": ALL}, "id"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        prevent_initial_call=True,
    )
    def _create_block_def(_, creator_key, bdef_vals, bdef_ids, registry, pool_meta, pool_next_id):
        global _current_bdef
        if not creator_key or creator_key not in _BLOCK_DEF_CREATORS:
            raise PreventUpdate
        _, method = _BLOCK_DEF_CREATORS[creator_key]
        pool_state = _pool.PoolState.from_stores(registry, pool_meta, pool_next_id)
        kwargs = generate_kwargs(bdef_ids, bdef_vals, pool_state) if (bdef_ids and bdef_vals) else {}
        kwargs = {k: v for k, v in kwargs.items() if v not in (None, "", [])}
        try:
            block_def = method(**kwargs)
        except Exception as exc:
            return f"Failed: {exc}", no_update
        _current_bdef = block_def
        param_str = (
            ", ".join(f"{k}={v}" for k, v in sorted(kwargs.items())) or "defaults"
        )
        label = f"{creator_key} ({param_str})"
        panel = html.Small(f"Current: {label}", style=_HINT)
        return f"Created {label}.", panel

    # 1. Rebuild init form and toggle Create button when class changes
    @app.callback(
        Output("cpx-init-form", "children"),
        Output("cpx-create-btn", "disabled"),
        Input("cpx-class-dd", "value"),
        prevent_initial_call=True,
    )
    def _update_init_form(cls_name: str | None):
        if not cls_name:
            return [], True
        cls = COMPLEX_CLASSES.get(cls_name)
        if cls is None:
            return [], True
        builder = COMPLEX_BUILDERS.get(cls, cls)
        first_param = next(iter(inspect.signature(builder).parameters))
        # NPC has fixed C8 symmetry; hide the symmetry param from the form.
        if cls_name == "NPC":
            exclude = [first_param, "symmetry"]
        elif cls_name == "Pleomorphic assembly":
            exclude = [first_param, "block_definition"]
        else:
            exclude = [first_param]
        rows = formgen.build_form(builder, id_type=_INIT, id_extra={}, exclude=exclude)
        return rows, False

    # 2. Create complex → add to server registry and pool
    @app.callback(
        Output(_CPX_POOL, "data"),
        Output(_CPX_SEL, "data"),
        Output("cpx-create-status", "children"),
        Output(_CPX_RESULTS, "data", allow_duplicate=True),
        Input("cpx-create-btn", "n_clicks"),
        State("cpx-class-dd", "value"),
        State("cpx-build-motl-select", "value"),
        State({"type": _INIT, "owner": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": _INIT, "owner": ALL, "param": ALL, "tag": ALL}, "id"),
        State(_CPX_POOL, "data"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        State(_CPX_RESULTS, "data"),
        prevent_initial_call=True,
    )
    def _create_complex(_, cls_name, motl_id, init_vals, init_ids, pool_data, registry, pool_meta, pool_next_id, results_store):
        if not cls_name or not motl_id:
            raise PreventUpdate
        cls = COMPLEX_CLASSES.get(cls_name)
        if cls is None:
            raise PreventUpdate

        motl = _motl_from_pool(motl_id)
        if motl is None:
            return no_update, no_update, "No motl data found for the selected motl.", no_update

        pool_state = _pool.PoolState.from_stores(registry, pool_meta, pool_next_id)
        init_kwargs = generate_kwargs(init_ids, init_vals, pool_state) if (init_ids and init_vals) else {}
        init_kwargs = {k: v for k, v in init_kwargs.items() if v not in (None, "", [])}

        if cls is PleomorphicSurface:
            if _current_bdef is None:
                return (no_update, no_update,
                        "No block definition created — use the Block definitions panel first.",
                        no_update)
            block_def = _current_bdef
            next_complex_id = cr.registry.peek_next_key()
            cpx_var = _prov.bind(next_complex_id)
            ps_kwargs = {k: v for k, v in init_kwargs.items()
                         if k in ("pixel_size", "tomo_id_column", "block_type_column",
                                  "ideal_degree", "ideal_face_size")}
            try:
                cpx = _invoke_op(
                    PleomorphicSurface,
                    {"blocks": motl, "block_definition": block_def, **ps_kwargs},
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
        Output(_CPX_RES_MOTL, "data"),
        Output("cpx-run-status", "children"),
        Output(_CPX_POOL, "data", allow_duplicate=True),
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output(ids.DATA_POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.DATA_POOL_NEXT_ID, "data", allow_duplicate=True),
        Output(_CPX_RESULTS, "data", allow_duplicate=True),
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
        prevent_initial_call=True,
    )
    def _run_method(
        _, entry_key, selected_id, pool_data, meth_vals, meth_ids,
        registry, pool_meta, pool_next_id,
        dp_registry, dp_next_id, results_store,
    ):
        _nu9 = (no_update,) * 9

        if not entry_key or not selected_id:
            raise PreventUpdate

        handle = next(
            (h for h in (pool_data or []) if h.get("complex_id") == selected_id), None
        )
        if handle is None:
            return no_update, "No complex selected.", *((no_update,) * 7)

        try:
            entry = discovery.get(entry_key)
        except KeyError:
            return no_update, f"Unknown entry {entry_key!r}.", *((no_update,) * 7)

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
            _dp_state = _datapool.DataPoolState.from_stores(dp_registry, dp_next_id)
            _cpx_count = _dp_state.kind_counters.get("cpx", 0) + 1
            _pool_id = f"cpx_{_cpx_count}"
            _assign_to = _prov.bind(_pool_id)

        try:
            if entry.kind == "classmethod":
                cls = _cls_for_handle(handle)
                if cls is None:
                    return no_update, "Unknown complex class.", *((no_update,) * 7)
                fn     = getattr(cls, entry.fn.__name__)
                result = _invoke_op(fn, meth_kwargs, assign_to=_assign_to, pool_id=_pool_id)
                cpx    = None
            else:
                cpx = _get_live_complex(handle["complex_id"], handle)
                if cpx is None:
                    return (no_update,
                            "Complex not available — reload or recreate it.", *((no_update,) * 7))
                fn     = getattr(cpx, entry.fn.__name__)
                result = _invoke_op(fn, meth_kwargs, assign_to=_assign_to, pool_id=_pool_id)
        except Exception as exc:
            return no_update, f"{entry.label} failed: {exc}", *((no_update,) * 7)

        motl_rows, df_records, feat_records, status = _dispatch_result(
            entry, result, cpx, handle
        )

        # After any in-place ("none") method, recompute the handle so that
        # geometry_fitted / radius are kept in sync in the pool.
        new_cpx_pool = no_update
        if entry.returns == "none" and cpx is not None:
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
        if entry.returns == "motl" and isinstance(result, Motl):
            new_ps, pool_ref = _pool.insert_motl(
                pool_state, result.df, label=entry.label,
            )
            new_pool_reg, new_pool_meta, new_pool_next = new_ps.to_stores()
            result._pool_motl_id = pool_ref
            _prov.record(pool_ref, _session.last_seq())

        # GX4: route dataframe results into the data pool; capture pool entry id
        new_dp_reg = no_update
        new_dp_next = no_update
        if entry.returns == "dataframe" and isinstance(result, pd.DataFrame):
            if _dp_state is None:
                _dp_state = _datapool.DataPoolState.from_stores(dp_registry, dp_next_id)
            new_ds, pool_ref = _datapool.insert_entry(
                _dp_state, result,
                label=entry.label,
                reader="dataframe",
                source_path="",
                entry_kind="cpx",
            )
            new_dp_reg, new_dp_next = new_ds.to_stores()
            _prov.record(pool_ref, _session.last_seq())

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

        section = {
            "key": entry_key,
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
            if s.get("key") == entry_key:
                cpx_sections[i] = section
                replaced = True
                break
        if not replaced:
            cpx_sections.append(section)
        new_results[selected_id] = cpx_sections

        return (
            motl_rows or None,
            status,
            new_cpx_pool,
            new_pool_reg,
            new_pool_meta,
            new_pool_next,
            new_dp_reg,
            new_dp_next,
            new_results,
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
        Input({"type": "cpx-close-btn", "id": ALL}, "n_clicks"),
        State(_CPX_POOL, "data"),
        State(_CPX_SEL, "data"),
        State(_CPX_RESULTS, "data"),
        prevent_initial_call=True,
    )
    def _close_complex(n_clicks_list, pool_data, selected_id, results_store):
        if not any(n_clicks_list):
            raise PreventUpdate
        triggered = ctx.triggered_id
        if triggered is None or not isinstance(triggered, dict):
            raise PreventUpdate
        cid = triggered["id"]
        new_pool = [h for h in (pool_data or []) if h.get("complex_id") != cid]
        new_sel = None if selected_id == cid else selected_id
        new_results = {k: v for k, v in (results_store or {}).items() if k != cid}
        return new_pool, new_sel, new_results
