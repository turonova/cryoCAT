"""Complexes page — registry-driven; no per-complex branching.

A new symmetry complex class needs only a ``@gui_exposed``-decorated subclass
and one entry in :data:`COMPLEX_CLASSES`; the tab auto-discovers its
constructor form and all instance/class-method operations.

Contract: exposes :data:`layout` and :func:`register_callbacks(app)`.
"""
from __future__ import annotations

from typing import Any

import dash
from dash import html, dcc, Input, Output, State, ALL, ctx, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd

from cryocat.core.cryomotl import EmMotl, Motl
from cryocat.analysis.structure import (
    CnComplex, DnComplex, NPC,
    TetrahedralComplex, OctahedralComplex, IcosahedralComplex,
)
from cryocat.app import ids, formgen, discovery
from cryocat.app.formgen import make_dropdown
from cryocat.app.apputils import generate_kwargs, run_operation
from cryocat.app.components import complex_registry as cr
from cryocat.app.components.motlsource import (
    get_motl_source, register_motl_source_callbacks,
)
from cryocat.app.components.motlsink import (
    get_send_to_editor_button, register_send_to_editor_callbacks,
)
from cryocat.app.pageshell import page_shell
import cryocat.app.pool as _pool


# ── Registry of supported complex classes ────────────────────────────────────

COMPLEX_CLASSES: dict[str, type] = {
    "CnComplex":          CnComplex,
    "DnComplex":          DnComplex,
    "NPC":                NPC,
    "TetrahedralComplex": TetrahedralComplex,
    "OctahedralComplex":  OctahedralComplex,
    "IcosahedralComplex": IcosahedralComplex,
}

# Hierarchy groups for the class picker (D3).
# Each group maps to the concrete subclasses that belong to it.
_HIERARCHY: list[tuple[str, list[str]]] = [
    ("Cyclic",     ["CnComplex", "NPC"]),
    ("Dihedral",   ["DnComplex"]),
    ("Polyhedral", ["TetrahedralComplex", "OctahedralComplex", "IcosahedralComplex"]),
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
_CPX_RES_MOTL = "cpx-result-motl"       # list[dict] | None — motl rows
_CPX_RES_DF   = "cpx-result-df"         # list[dict] | None — dataframe records
_CPX_RES_FEAT = "cpx-result-feat"       # list[dict] | None — feature vectors

# Form id-type strings (must not collide with any other page)
_INIT = "cpx-init-param"
_METH = "cpx-meth-param"

_HINT = {"color": "var(--color9)", "margin": "0.3rem 0"}
_HDR  = {"fontWeight": 600, "margin": "0.4rem 0 0.2rem"}


# ── Pure helpers ──────────────────────────────────────────────────────────────

def motl_from_pool_rows(motl_id: str | None) -> Motl | None:
    """Rebuild a :class:`~cryocat.core.cryomotl.Motl` from the server-side pool."""
    if not motl_id:
        return None
    try:
        return EmMotl(_pool.get_rows(motl_id))
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
    motl = _motl_from_pool(handle.get("source_motl_id"))
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
        [html.Div(id="cpx-handles-list")],
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
    ]


def _main() -> list:
    return [
        html.Div("Result motl", style=_HDR),
        get_send_to_editor_button("cpx-export"),
        dcc.Store(id=_CPX_RES_MOTL),
        html.Hr(style={"margin": "0.6rem 0"}),
        html.Div("Results table", style=_HDR),
        html.Div(id="cpx-result-table"),
        dcc.Store(id=_CPX_RES_DF),
        html.Hr(style={"margin": "0.6rem 0"}),
        html.Div("Feature vectors", style=_HDR),
        html.Div(id="cpx-feat-table"),
        dcc.Store(id=_CPX_RES_FEAT),
    ]


layout: Any = html.Div(
    [
        page_shell(_sidebar(), _main(), sidebar_width=4),
    ],
    style={"margin": 0, "padding": 0},
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app: dash.Dash) -> None:  # noqa: C901
    register_motl_source_callbacks(app, "cpx-build")
    register_send_to_editor_callbacks(app, "cpx-export", _CPX_RES_MOTL)

    # 1. Rebuild init form and toggle Create button when class changes
    @app.callback(
        Output("cpx-init-form", "children"),
        Output("cpx-create-btn", "disabled"),
        Input("cpx-class-dd", "value"),
    )
    def _update_init_form(cls_name: str | None):
        if not cls_name:
            return [], True
        cls = COMPLEX_CLASSES.get(cls_name)
        if cls is None:
            return [], True
        rows = formgen.build_form(cls, id_type=_INIT, id_extra={}, exclude=["motl"])
        return rows, False

    # 2. Create complex → add to server registry and pool
    @app.callback(
        Output(_CPX_POOL, "data"),
        Output(_CPX_SEL, "data"),
        Output("cpx-create-status", "children"),
        Input("cpx-create-btn", "n_clicks"),
        State("cpx-class-dd", "value"),
        State("cpx-build-motl-select", "value"),
        State({"type": _INIT, "owner": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": _INIT, "owner": ALL, "param": ALL, "tag": ALL}, "id"),
        State(_CPX_POOL, "data"),
        prevent_initial_call=True,
    )
    def _create_complex(_, cls_name, motl_id, init_vals, init_ids, pool_data):
        if not cls_name or not motl_id:
            raise PreventUpdate
        cls = COMPLEX_CLASSES.get(cls_name)
        if cls is None:
            raise PreventUpdate

        motl = _motl_from_pool(motl_id)
        if motl is None:
            return no_update, no_update, "No motl data found for the selected motl."

        init_kwargs = generate_kwargs(init_ids, init_vals) if (init_ids and init_vals) else {}
        init_kwargs = {k: v for k, v in init_kwargs.items() if v not in (None, "", [])}

        try:
            cpx = cls(motl, **init_kwargs)
        except Exception as exc:
            return no_update, no_update, f"Init failed: {exc}"

        complex_id = cr.registry.add(cpx)
        var = complex_id.replace("-", "_")
        label = f"{var} ({cls_name})"
        handle = cr.make_handle(cpx, complex_id, label, motl_id, init_kwargs)

        new_pool = list(pool_data or []) + [handle]
        return new_pool, complex_id, f"Created {label}."

    # 3. Render clickable handles list
    @app.callback(
        Output("cpx-handles-list", "children"),
        Input(_CPX_POOL, "data"),
        Input(_CPX_SEL, "data"),
    )
    def _render_handles(pool_data, selected_id):
        if not pool_data:
            return html.Small("No complexes created yet.", style=_HINT)
        items = []
        for h in pool_data:
            cid    = h.get("complex_id", "")
            is_sel = cid == selected_id
            info   = f"{h.get('cls', '')} · n={h.get('n_subunits', '?')}"
            items.append(dbc.Button(
                [html.Strong(h.get("label", cid)), html.Br(),
                 html.Small(info, style={"color": "var(--color9)"})],
                id={"type": "cpx-handle-btn", "id": cid},
                color="primary" if is_sel else "secondary",
                outline=not is_sel,
                size="sm",
                style={"width": "100%", "textAlign": "left", "marginBottom": "0.25rem"},
                n_clicks=0,
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
    )
    def _update_methods(selected_id, pool_data):
        if not selected_id:
            return [], None, ""
        handle = next(
            (h for h in (pool_data or []) if h.get("complex_id") == selected_id), None
        )
        if handle is None:
            return [], None, ""
        cls = COMPLEX_CLASSES.get(handle.get("cls", ""))
        if cls is None:
            return [], None, ""
        entries = discovery.entries_for_class(cls)
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
        Output(_CPX_RES_DF, "data"),
        Output(_CPX_RES_FEAT, "data"),
        Output("cpx-run-status", "children"),
        Output(_CPX_POOL, "data", allow_duplicate=True),
        Input("cpx-run-btn", "n_clicks"),
        State("cpx-method-dd", "value"),
        State(_CPX_SEL, "data"),
        State(_CPX_POOL, "data"),
        State({"type": _METH, "owner": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": _METH, "owner": ALL, "param": ALL, "tag": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _run_method(_, entry_key, selected_id, pool_data, meth_vals, meth_ids):
        if not entry_key or not selected_id:
            raise PreventUpdate

        handle = next(
            (h for h in (pool_data or []) if h.get("complex_id") == selected_id), None
        )
        if handle is None:
            return no_update, no_update, no_update, "No complex selected.", no_update

        try:
            entry = discovery.get(entry_key)
        except KeyError:
            return no_update, no_update, no_update, f"Unknown entry {entry_key!r}.", no_update

        meth_kwargs = generate_kwargs(meth_ids, meth_vals) if (meth_ids and meth_vals) else {}
        meth_kwargs = {k: v for k, v in meth_kwargs.items() if v not in (None, "", [])}

        try:
            if entry.kind == "classmethod":
                cls = COMPLEX_CLASSES.get(handle.get("cls", ""))
                if cls is None:
                    return no_update, no_update, no_update, "Unknown complex class.", no_update
                fn     = getattr(cls, entry.fn.__name__)
                result = run_operation(fn, meth_kwargs)
                cpx    = None
            else:
                cpx = _get_live_complex(handle["complex_id"], handle)
                if cpx is None:
                    return (no_update, no_update, no_update,
                            "Complex not available — reload or recreate it.", no_update)
                fn     = getattr(cpx, entry.fn.__name__)
                result = run_operation(fn, meth_kwargs)
        except Exception as exc:
            return no_update, no_update, no_update, f"{entry.label} failed: {exc}", no_update

        motl_rows, df_records, feat_records, status = _dispatch_result(
            entry, result, cpx, handle
        )

        # After any in-place ("none") method, recompute the handle so that
        # geometry_fitted / radius are kept in sync in the pool.
        new_pool = no_update
        if entry.returns == "none" and cpx is not None:
            updated_handle = cr.make_handle(
                cpx,
                handle["complex_id"],
                handle.get("label", handle["complex_id"]),
                handle.get("source_motl_id", ""),
                handle.get("init_kwargs", {}),
            )
            new_pool = [
                updated_handle if h.get("complex_id") == handle["complex_id"] else h
                for h in (pool_data or [])
            ]

        return (
            motl_rows or None,
            df_records or None,
            feat_records or None,
            status,
            new_pool,
        )

    # 8. Render dataframe result table
    @app.callback(
        Output("cpx-result-table", "children"),
        Input(_CPX_RES_DF, "data"),
    )
    def _render_result_table(data):
        return _render_table(data, "No dataframe result yet.")

    # 9. Render feature-vector table
    @app.callback(
        Output("cpx-feat-table", "children"),
        Input(_CPX_RES_FEAT, "data"),
    )
    def _render_feat_table(data):
        return _render_table(data, "No feature vectors yet.")
