"""Editor sidebar — pool-driven.

Loads motls into the suite-global pool (``pool-*`` stores, declared in
:mod:`cryocat.app.suite.app`) and drives the editor's view slots.

Model
-----
* The **pool** is the source of truth for motl data; it is unbounded.
* The editor renders into a fixed set of ``N_SLOTS`` pre-wired *view slots*
  (the table/viewer/save surfaces, whose callbacks are registered once up
  front with literal ``me-{i}`` prefixes).
* ``me-slot-map`` (a length-``N_SLOTS`` list of ``motl_id`` or ``None``) maps
  each view slot to a pool entry. Loading a motl auto-assigns it to a free
  slot; when the pool exceeds the slots the user re-assigns via the
  "Slot assignment" dropdowns.
"""

import inspect
import pandas as pd

import dash
from dash import html, dcc, Input, Output, State, no_update, ctx, ALL
import dash_bootstrap_components as dbc

from cryocat.core.cryomotl import Motl
from cryocat.app.components.motlio import get_motl_load_component, register_motl_load_callbacks
from cryocat.app.components.motlsource import (
    get_multi_motl_picker, register_multi_motl_picker_callbacks,
)
from cryocat.app.apputils import (
    generate_kwargs, get_single_motl_methods, get_multi_motl_methods,
)
from cryocat.app.formgen import build_form
from cryocat.app.logger import dash_logger, invoke_operation

# Number of editor *view slots* (rendered table/viewer surfaces). The motl pool
# itself is unbounded — this only caps how many motls are open as tabs at once.
N_SLOTS = 5

# Fetched once at import time from the live Motl class — no hardcoding.
_MOTL_METHODS = get_single_motl_methods()
_MULTI_MOTL_METHODS = get_multi_motl_methods()
# Lookup of `method_name -> _gui["motls"]` spec for the run callback. Built from
# the same collector so adding a new multi-motl op (decorator change only) flows
# through with no edits here.
_MULTI_MOTL_SPECS = {m["value"]: m["motls"] for m in _MULTI_MOTL_METHODS}

_NONE_OPT = "__none__"  # dropdown sentinel for an empty slot


def _slot_assignment_rows():
    """One dropdown per view slot — assigns a pool motl to that slot."""
    rows = []
    for i in range(N_SLOTS):
        rows.append(
            dbc.Row(
                [
                    dbc.Col(
                        html.Label(
                            f"Slot {i + 1}:",
                            style={"fontSize": "0.85rem"},
                        ),
                        width=3,
                        className="d-flex align-items-center",
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id={"type": "me-slot-assign", "slot": i},
                            options=[{"label": "(empty)", "value": _NONE_OPT}],
                            value=_NONE_OPT,
                            clearable=False,
                            style={"fontSize": "0.85rem"},
                        ),
                        width=9,
                    ),
                ],
                className="mb-1",
            )
        )
    return rows


def get_motl_editor_sidebar():
    return dbc.Col(
        html.Div(
            [
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            [
                                get_motl_load_component("me-load"),
                                html.Div(
                                    id="me-load-status",
                                    style={
                                        "marginTop": "0.5rem",
                                        "fontSize": "0.9rem",
                                        "color": "var(--color9)",
                                        "wordBreak": "break-word",
                                    },
                                ),
                            ],
                            title="Load Motl",
                            item_id="me-sidebar-load",
                        ),
                        dbc.AccordionItem(
                            [
                                html.Div(
                                    id="me-motl-list",
                                    children=html.Div(
                                        "No motls loaded.",
                                        style={"color": "var(--color9)", "fontSize": "0.9rem", "padding": "4px"},
                                    ),
                                ),
                            ],
                            title="Loaded Motls (pool)",
                            item_id="me-sidebar-list",
                        ),
                        dbc.AccordionItem(
                            [
                                html.Div(
                                    "Assign which pool motl is rendered in each editor slot. "
                                    "Loading auto-fills free slots; use these when the pool exceeds "
                                    f"{N_SLOTS} motls.",
                                    style={"fontSize": "0.8rem", "color": "var(--color9)", "marginBottom": "0.5rem"},
                                ),
                                *_slot_assignment_rows(),
                            ],
                            title="Slot assignment",
                            item_id="me-sidebar-slots",
                        ),
                        dbc.AccordionItem(
                            [
                                dcc.Dropdown(
                                    id="me-op-func-select",
                                    options=_MOTL_METHODS,
                                    placeholder="Select operation",
                                    style={"marginBottom": "0.5rem"},
                                ),
                                html.Div(id="me-op-func-form", style={"marginBottom": "0.5rem"}),
                                html.Div(
                                    [
                                        dbc.Button(
                                            "Apply",
                                            id="me-op-apply-btn",
                                            color="primary",
                                            size="sm",
                                            className="me-1",
                                            style={"width": "48%"},
                                        ),
                                        dbc.Button(
                                            "Undo",
                                            id="me-op-undo-btn",
                                            color="secondary",
                                            size="sm",
                                            style={"width": "48%"},
                                        ),
                                    ],
                                    style={"display": "flex", "marginBottom": "0.5rem"},
                                ),
                                html.Div(
                                    id="me-op-status",
                                    style={"fontSize": "0.85rem", "color": "var(--color9)", "wordBreak": "break-word"},
                                ),
                            ],
                            title="Single Motl Operations",
                            item_id="me-sidebar-singlemotl",
                        ),
                        dbc.AccordionItem(
                            [
                                dcc.Dropdown(
                                    id="me-multi-op-select",
                                    options=_MULTI_MOTL_METHODS,
                                    placeholder="Select operation",
                                    style={"marginBottom": "0.5rem"},
                                ),
                                get_multi_motl_picker("me-multi"),
                                html.Div(id="me-multi-form", style={"marginBottom": "0.5rem"}),
                                dbc.Button(
                                    "Run",
                                    id="me-multi-run-btn",
                                    color="primary",
                                    size="sm",
                                    style={"width": "100%"},
                                ),
                                html.Div(
                                    id="me-multi-op-status",
                                    style={
                                        "marginTop": "0.5rem", "fontSize": "0.9rem",
                                        "color": "var(--color9)", "wordBreak": "break-word",
                                    },
                                ),
                            ],
                            title="Multiple motl operations",
                            item_id="me-sidebar-multimotl",
                        ),
                    ],
                    always_open=True,
                    active_item=["me-sidebar-load", "me-sidebar-list"],
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
        id="me-sidebar",
        width=3,
        style={
            "margin": "0",
            "padding": "0",
            "height": "100vh",
            "position": "sticky",
            "top": "0px",
        },
    )


def _first_free_slot(slot_map):
    for i in range(N_SLOTS):
        if i >= len(slot_map) or not slot_map[i]:
            return i
    return None


def _relion_params_summary(relion_params):
    """Human-readable one-liner appended to the load status."""
    if not relion_params:
        return ""

    def _scalar(v):
        if v is None:
            return None
        try:
            if hasattr(v, "__len__"):
                v = v[0] if len(v) > 0 else None
            return float(v)
        except (TypeError, ValueError, IndexError):
            return None

    parts = []
    ps = _scalar(relion_params.get("pixel_size"))
    bn = _scalar(relion_params.get("binning"))
    if ps is not None:
        parts.append(f"pixel size: {ps:.4g} Å")
    if bn is not None:
        parts.append(f"binning: {bn:.4g}")
    if relion_params.get("tomo_format"):
        parts.append(f"tomo fmt: {relion_params['tomo_format']}")
    if relion_params.get("subtomo_format"):
        parts.append(f"subtomo fmt: {relion_params['subtomo_format']}")
    return ("  |  " + ",  ".join(parts)) if parts else ""


def _register_rotation_fields_for_form(app, id_type: str, methods: list) -> None:
    """Pre-register rotation-builder modal callbacks for @gui_exposed Motl methods
    that have RotationLike parameters and use the given id_type."""
    import typing
    from cryocat.utils.classutils import resolve_param_type
    from cryocat.app.components.rotationbuilder import register_rotation_builder_callbacks

    for method_info in methods:
        fn = getattr(Motl, method_info["value"], None)
        if fn is None:
            continue
        try:
            hints = typing.get_type_hints(fn)
        except Exception:
            hints = {}
        for param, ann in hints.items():
            tag, _ = resolve_param_type(ann)
            if tag != "RotationLike":
                continue
            rprefix = f"rotfld-{id_type}-{param}"
            register_rotation_builder_callbacks(app, f"{rprefix}-inner")

            def _register(rprefix_=rprefix, param_=param, id_type_=id_type):
                @app.callback(
                    Output(f"{rprefix_}-modal", "is_open"),
                    Input(f"{rprefix_}-build-btn", "n_clicks"),
                    Input(f"{rprefix_}-close-btn", "n_clicks"),
                    Input(f"{rprefix_}-use-btn", "n_clicks"),
                    State(f"{rprefix_}-modal", "is_open"),
                    prevent_initial_call=True,
                )
                def _toggle(_open, _close, _use, is_open):
                    return not is_open

                @app.callback(
                    Output({"type": id_type_, "param": param_, "tag": "RotationLike"}, "value", allow_duplicate=True),
                    Input(f"{rprefix_}-use-btn", "n_clicks"),
                    State(f"{rprefix_}-inner-rot-euler-store", "data"),
                    prevent_initial_call=True,
                )
                def _use(_n, euler_str):
                    return euler_str or no_update

            _register()


def register_motl_editor_sidebar_callbacks(app):

    register_motl_load_callbacks(app, "me-load")
    _register_rotation_fields_for_form(app, "me-op-param", _MOTL_METHODS)
    _register_rotation_fields_for_form(app, "me-multi-param", _MULTI_MOTL_METHODS)

    # ── Load → pool ────────────────────────────────────────────────────────────
    # A freshly loaded motl is appended to the pool with a new motl_id and
    # auto-assigned to the first free view slot (if any).
    @app.callback(
        Output("pool-registry", "data", allow_duplicate=True),
        Output("pool-motls", "data", allow_duplicate=True),
        Output("pool-extra", "data", allow_duplicate=True),
        Output("pool-meta", "data", allow_duplicate=True),
        Output("pool-next-id", "data", allow_duplicate=True),
        Output("me-slot-map", "data", allow_duplicate=True),
        Output("me-tabs", "active_tab", allow_duplicate=True),
        Output("me-load-status", "children", allow_duplicate=True),
        Input("me-load-motl-data-store", "data"),
        State("me-load-motl-extra-data-store", "data"),
        State("me-load-motl-data-type", "data"),
        State("me-load-relion-optics-store", "data"),
        State("me-load-relion5-tomos-store", "data"),
        State("me-load-relion5-tomos-filename", "data"),
        State("me-load-motl-upload", "filename"),
        State("me-load-relion-params-store", "data"),
        State("pool-registry", "data"),
        State("pool-motls", "data"),
        State("pool-extra", "data"),
        State("pool-meta", "data"),
        State("pool-next-id", "data"),
        State("me-slot-map", "data"),
        prevent_initial_call=True,
    )
    def route_motl(
        motl_data, extra, dtype, optics, r5t, r5tn, filename, relion_params,
        registry, pool_motls, pool_extra, pool_meta, next_id, slot_map,
    ):
        if not motl_data:
            raise dash.exceptions.PreventUpdate

        registry = dict(registry or {})
        pool_motls = dict(pool_motls or {})
        pool_extra = dict(pool_extra or {})
        pool_meta = dict(pool_meta or {})
        next_id = next_id or 0
        slot_map = list(slot_map or [None] * N_SLOTS)
        while len(slot_map) < N_SLOTS:
            slot_map.append(None)

        mid = f"motl-{next_id}"
        label = filename or f"Motl {next_id + 1}"

        registry[mid] = {
            "label": label,
            "type": dtype or "emmotl",
            "n_rows": len(motl_data),
            "active": True,
        }
        pool_motls[mid] = motl_data
        pool_extra[mid] = extra
        pool_meta[mid] = {
            "data_type": dtype,
            "relion_optics": optics,
            "relion5_tomos": r5t,
            "relion5_tomos_filename": r5tn,
            "relion_params": relion_params,
            "script_expr": f"cryomotl.Motl.load({(filename or label)!r}, {(dtype or 'emmotl')!r})",
        }

        free = _first_free_slot(slot_map)
        if free is not None:
            slot_map[free] = mid
            active_tab = f"me-tab-{free}"
            status = f"Loaded: {label} ({len(motl_data)} particles) → slot {free + 1}"
        else:
            active_tab = no_update
            status = (
                f"Loaded: {label} ({len(motl_data)} particles) → pool "
                f"(all {N_SLOTS} slots in use; assign it via 'Slot assignment')"
            )
        status += _relion_params_summary(relion_params)

        return registry, pool_motls, pool_extra, pool_meta, next_id + 1, slot_map, active_tab, status

    # ── Pool motl list ─────────────────────────────────────────────────────────
    @app.callback(
        Output("me-motl-list", "children"),
        Input("pool-registry", "data"),
        Input("me-slot-map", "data"),
        prevent_initial_call=True,
    )
    def update_motl_list(registry, _slot_map):
        registry = registry or {}
        items = []
        for mid, meta in registry.items():
            if not meta.get("active", True):
                continue
            label = meta.get("label", mid)
            items.append(
                dbc.ListGroupItem(
                    [
                        html.Span(
                            label,
                            style={
                                "flex": "1",
                                "overflow": "hidden",
                                "textOverflow": "ellipsis",
                                "whiteSpace": "nowrap",
                                "fontSize": "0.9rem",
                            },
                        ),
                        dbc.Button(
                            "×",
                            id={"type": "me-close-motl", "mid": mid},
                            color="link",
                            size="sm",
                            style={"padding": "0 6px", "color": "var(--color9)", "lineHeight": "1"},
                        ),
                    ],
                    id={"type": "me-motl-list-item", "mid": mid},
                    action=True,
                    n_clicks=0,
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "padding": "4px 8px",
                        "cursor": "pointer",
                    },
                )
            )

        if not items:
            return html.Div(
                "No motls loaded.",
                style={"color": "var(--color9)", "fontSize": "0.9rem", "padding": "4px"},
            )
        return dbc.ListGroup(items, flush=True)

    # ── Slot-assignment dropdowns: render from slot_map + registry ─────────────
    @app.callback(
        Output({"type": "me-slot-assign", "slot": ALL}, "options"),
        Output({"type": "me-slot-assign", "slot": ALL}, "value"),
        Input("pool-registry", "data"),
        Input("me-slot-map", "data"),
        prevent_initial_call=True,
    )
    def render_slot_assignment(registry, slot_map):
        registry = registry or {}
        slot_map = list(slot_map or [None] * N_SLOTS)
        motl_opts = [
            {"label": m.get("label", mid), "value": mid}
            for mid, m in registry.items()
            if m.get("active", True)
        ]
        options = [[{"label": "(empty)", "value": _NONE_OPT}] + motl_opts for _ in range(N_SLOTS)]
        values = [
            (slot_map[i] if i < len(slot_map) and slot_map[i] else _NONE_OPT)
            for i in range(N_SLOTS)
        ]
        return options, values

    # ── Slot-assignment dropdowns: apply user change → slot_map ────────────────
    @app.callback(
        Output("me-slot-map", "data", allow_duplicate=True),
        Input({"type": "me-slot-assign", "slot": ALL}, "value"),
        State({"type": "me-slot-assign", "slot": ALL}, "id"),
        State("me-slot-map", "data"),
        prevent_initial_call=True,
    )
    def apply_slot_assignment(values, ids, slot_map):
        slot_map = list(slot_map or [None] * N_SLOTS)
        while len(slot_map) < N_SLOTS:
            slot_map.append(None)

        # Current value of each slot's dropdown (None for the "(empty)" sentinel).
        by_slot = {}
        for id_, val in zip(ids, values):
            by_slot[id_["slot"]] = val if (val and val != _NONE_OPT) else None
        new_map = [by_slot.get(i) for i in range(N_SLOTS)]

        # A motl may occupy at most one slot. The dropdown the user just changed
        # is authoritative: its choice moves into that slot and is cleared from
        # any other slot it previously occupied (so re-assigning replaces,
        # rather than emptying the target slot).
        triggered = ctx.triggered_id
        if isinstance(triggered, dict) and triggered.get("type") == "me-slot-assign":
            changed_slot = triggered.get("slot")
            chosen = new_map[changed_slot] if changed_slot is not None else None
            if chosen is not None:
                for i in range(N_SLOTS):
                    if i != changed_slot and new_map[i] == chosen:
                        new_map[i] = None
        else:
            # No specific trigger (e.g. a programmatic refresh): keep first.
            seen = set()
            for i in range(N_SLOTS):
                if new_map[i] is not None:
                    if new_map[i] in seen:
                        new_map[i] = None
                    else:
                        seen.add(new_map[i])

        if new_map == slot_map:
            raise dash.exceptions.PreventUpdate
        return new_map

    # ── Clicking a pool motl activates its slot tab (if assigned) ──────────────
    @app.callback(
        Output("me-tabs", "active_tab", allow_duplicate=True),
        Input({"type": "me-motl-list-item", "mid": ALL}, "n_clicks"),
        State("me-slot-map", "data"),
        prevent_initial_call=True,
    )
    def activate_tab_from_list(n_clicks_list, slot_map):
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "mid" in triggered):
            raise dash.exceptions.PreventUpdate
        mid = triggered["mid"]
        slot_map = slot_map or [None] * N_SLOTS
        for i, m in enumerate(slot_map):
            if m == mid:
                return f"me-tab-{i}"
        raise dash.exceptions.PreventUpdate

    # ── Close a pool motl: drop it from the pool and free its slot ─────────────
    @app.callback(
        Output("pool-registry", "data", allow_duplicate=True),
        Output("pool-motls", "data", allow_duplicate=True),
        Output("pool-extra", "data", allow_duplicate=True),
        Output("pool-meta", "data", allow_duplicate=True),
        Output("me-slot-map", "data", allow_duplicate=True),
        Output("me-tabs", "active_tab", allow_duplicate=True),
        Input({"type": "me-close-motl", "mid": ALL}, "n_clicks"),
        State("pool-registry", "data"),
        State("pool-motls", "data"),
        State("pool-extra", "data"),
        State("pool-meta", "data"),
        State("me-slot-map", "data"),
        State("me-tabs", "active_tab"),
        prevent_initial_call=True,
    )
    def close_motl(n_clicks_list, registry, pool_motls, pool_extra, pool_meta, slot_map, active_tab):
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "mid" in triggered):
            raise dash.exceptions.PreventUpdate

        mid = triggered["mid"]
        registry = {k: v for k, v in (registry or {}).items() if k != mid}
        pool_motls = {k: v for k, v in (pool_motls or {}).items() if k != mid}
        pool_extra = {k: v for k, v in (pool_extra or {}).items() if k != mid}
        pool_meta = {k: v for k, v in (pool_meta or {}).items() if k != mid}

        old_map = list(slot_map or [None] * N_SLOTS)
        closed_slot = next((i for i, m in enumerate(old_map) if m == mid), None)
        new_map = [None if m == mid else m for m in old_map]

        # If the closed motl's tab was active, move focus off the destroyed tab.
        new_active = no_update
        if closed_slot is not None and active_tab == f"me-tab-{closed_slot}":
            nxt = next((i for i, m in enumerate(new_map) if m), None)
            new_active = f"me-tab-{nxt}" if nxt is not None else "me-tab-0"

        return registry, pool_motls, pool_extra, pool_meta, new_map, new_active

    # ── Multiple-motl operations: collector-driven (pair / list) ───────────────
    register_multi_motl_picker_callbacks(app, "me-multi")

    @app.callback(
        Output("me-multi-pair-picker", "style"),
        Output("me-multi-list-picker", "style"),
        Output("me-multi-list-label", "children"),
        Input("me-multi-op-select", "value"),
    )
    def _toggle_multi_picker(method_name):
        if not method_name:
            return {"display": "none"}, {"display": "none"}, "Motls"
        spec = _MULTI_MOTL_SPECS.get(method_name) or {}
        arity = spec.get("arity")
        if arity == "pair":
            return {"display": "block"}, {"display": "none"}, "Motls"
        if arity == "list":
            label = (
                "Motls (first = kept on duplicates)"
                if spec.get("main_first") else
                "Motls to merge (order preserved)"
            )
            return {"display": "none"}, {"display": "block"}, label
        return {"display": "none"}, {"display": "none"}, "Motls"

    @app.callback(
        Output("me-multi-form", "children"),
        Input("me-multi-op-select", "value"),
    )
    def _build_multi_form(method_name):
        # Scalar params only — the motl params (motl1/motl2 for pair ops,
        # motls.param for list ops) are supplied by the picker, not the form.
        if not method_name:
            return []
        spec = _MULTI_MOTL_SPECS.get(method_name) or {}
        if spec.get("arity") == "pair":
            exclude = ("motl1", "motl2")
        elif spec.get("arity") == "list":
            exclude = (spec.get("param", "motl_list"),)
        else:
            exclude = ()
        fn = getattr(Motl, method_name)
        return build_form(fn, id_type="me-multi-param", exclude=exclude)

    @app.callback(
        Output("pool-registry", "data", allow_duplicate=True),
        Output("pool-motls", "data", allow_duplicate=True),
        Output("pool-extra", "data", allow_duplicate=True),
        Output("pool-meta", "data", allow_duplicate=True),
        Output("pool-next-id", "data", allow_duplicate=True),
        Output("me-slot-map", "data", allow_duplicate=True),
        Output("me-tabs", "active_tab", allow_duplicate=True),
        Output("me-multi-op-status", "children", allow_duplicate=True),
        Input("me-multi-run-btn", "n_clicks"),
        State("me-multi-op-select", "value"),
        State("me-multi-main-select", "value"),
        State("me-multi-second-select", "value"),
        State("me-multi-list-select", "value"),
        State({"type": "me-multi-param", "param": ALL, "tag": ALL}, "value"),
        State({"type": "me-multi-param", "param": ALL, "tag": ALL}, "id"),
        State("pool-registry", "data"),
        State("pool-motls", "data"),
        State("pool-extra", "data"),
        State("pool-meta", "data"),
        State("pool-next-id", "data"),
        State("me-slot-map", "data"),
        prevent_initial_call=True,
    )
    def run_multi_op(
        n_clicks, method_name, main_id, second_id, list_ids,
        param_values, param_ids,
        registry, pool_motls, pool_extra, pool_meta, next_id, slot_map,
    ):
        pool_noup = (no_update,) * 7

        def _err(msg):
            return (*pool_noup, msg)

        if not n_clicks or not method_name:
            raise dash.exceptions.PreventUpdate

        spec = _MULTI_MOTL_SPECS.get(method_name)
        if spec is None:
            return _err(f"Operation '{method_name}' is not registered as multi-motl.")

        pool_motls = pool_motls or {}

        # 1) Resolve selected motl_ids -> Motl instances, preserving order.
        try:
            if spec["arity"] == "pair":
                if not main_id or not second_id:
                    return _err("Select both Main and Second motls.")
                if main_id == second_id:
                    return _err("Main and Second motl must differ.")
                ordered_ids = [main_id, second_id]
            else:  # list
                if not list_ids or len(list_ids) < 2:
                    return _err("Select at least two motls for this operation.")
                ordered_ids = list(list_ids)

            motls = []
            for mid in ordered_ids:
                rows = pool_motls.get(mid)
                if not rows:
                    return _err(f"Pool entry '{mid}' has no data.")
                motl_obj = Motl(pd.DataFrame(rows))
                src_expr = (pool_meta.get(mid) or {}).get("script_expr")
                if src_expr:
                    dash_logger.record_motl_source(motl_obj, src_expr)
                motls.append(motl_obj)
        except Exception as exc:
            return _err(f"Error preparing motls: {exc}")

        # 2) Scalar kwargs from the auto-form.
        kwargs = generate_kwargs(param_ids, param_values) if param_ids else {}

        # 3) Build + run the call. These ops are classmethods on Motl.
        try:
            fn = getattr(Motl, method_name)
            if spec["arity"] == "pair":
                sig_params = list(inspect.signature(fn).parameters.keys())
                full_kwargs = {sig_params[0]: motls[0], sig_params[1]: motls[1], **kwargs}
            else:
                list_param = spec.get("param", "motl_list")
                full_kwargs = {list_param: motls, **kwargs}
            result = invoke_operation(fn, full_kwargs)
        except Exception as exc:
            return _err(f"Error running '{method_name}': {exc}")

        if not isinstance(result, Motl):
            return _err(f"'{method_name}' did not return a Motl (got {type(result).__name__}).")

        # 4) Add the new motl to the pool and assign to a free slot.
        registry = dict(registry or {})
        pool_motls = dict(pool_motls or {})
        pool_extra = dict(pool_extra or {})
        pool_meta = dict(pool_meta or {})
        next_id = next_id or 0
        slot_map = list(slot_map or [None] * N_SLOTS)
        while len(slot_map) < N_SLOTS:
            slot_map.append(None)

        new_rows = result.df.to_dict("records")
        gui = getattr(getattr(Motl, method_name).__func__, "_gui", {})
        op_label = gui.get("label", method_name)
        src_labels = [(registry.get(mid) or {}).get("label", mid) for mid in ordered_ids]
        mid = f"motl-{next_id}"
        registry[mid] = {
            "label": f"{op_label} of {' + '.join(src_labels)}",
            "type": "emmotl",
            "n_rows": len(new_rows),
            "active": True,
        }
        pool_motls[mid] = new_rows
        pool_extra[mid] = None
        pool_meta[mid] = {
            "data_type": None, "relion_optics": None, "relion5_tomos": None,
            "relion5_tomos_filename": None, "relion_params": None,
            "script_expr": dash_logger.last_script_line,
        }

        free = _first_free_slot(slot_map)
        if free is not None:
            slot_map[free] = mid
            active = f"me-tab-{free}"
            status = (
                f"'{op_label}' -> new motl in slot {free + 1} "
                f"({len(new_rows)} particles, from {len(motls)} input motl(s))."
            )
        else:
            active = no_update
            status = (
                f"'{op_label}' -> new motl in the pool "
                f"({len(new_rows)} particles; no free slot, use 'Slot assignment')."
            )

        return registry, pool_motls, pool_extra, pool_meta, next_id + 1, slot_map, active, status

    # ── Single-motl operation form ─────────────────────────────────────────────
    @app.callback(
        Output("me-op-func-form", "children"),
        Input("me-op-func-select", "value"),
        prevent_initial_call=True,
    )
    def generate_op_form(method_name):
        if not method_name:
            return []
        return build_form(getattr(Motl, method_name), id_type="me-op-param")

    # ── Apply a method to the active slot's motl ───────────────────────────────
    # In-place ops (gui output=None) update the active slot; ops that produce a
    # new motl (gui output="motl", e.g. get_random_subset) are added to the pool
    # as a separate entry so the source motl is preserved.
    @app.callback(
        *[Output(f"me-{i}-motl-data-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        *[Output(f"me-{i}-undo-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        Output("me-op-status", "children", allow_duplicate=True),
        Output("pool-registry", "data", allow_duplicate=True),
        Output("pool-motls", "data", allow_duplicate=True),
        Output("pool-extra", "data", allow_duplicate=True),
        Output("pool-meta", "data", allow_duplicate=True),
        Output("pool-next-id", "data", allow_duplicate=True),
        Output("me-slot-map", "data", allow_duplicate=True),
        Output("me-tabs", "active_tab", allow_duplicate=True),
        Input("me-op-apply-btn", "n_clicks"),
        State("me-op-func-select", "value"),
        State("me-tabs", "active_tab"),
        State({"type": "me-op-param", "param": ALL, "tag": ALL}, "value"),
        State({"type": "me-op-param", "param": ALL, "tag": ALL}, "id"),
        *[State(f"me-{i}-motl-data-store", "data") for i in range(N_SLOTS)],
        State("pool-registry", "data"),
        State("pool-motls", "data"),
        State("pool-extra", "data"),
        State("pool-meta", "data"),
        State("pool-next-id", "data"),
        State("me-slot-map", "data"),
        prevent_initial_call=True,
    )
    def apply_operation(n_clicks, method_name, active_tab, param_values, param_ids, *rest):
        all_slot_data = rest[:N_SLOTS]
        registry, pool_motls, pool_extra, pool_meta, next_id, slot_map = rest[N_SLOTS:]

        # 7 pool-related outputs: registry, motls, extra, meta, next_id, slot_map, active_tab
        pool_noup = (no_update,) * 7
        nochange = [no_update] * N_SLOTS

        def _ret(data_out, undo_out, status, pool=pool_noup):
            return (*data_out, *undo_out, status, *pool)

        if not n_clicks or not method_name or not active_tab:
            raise dash.exceptions.PreventUpdate
        try:
            slot_idx = int(str(active_tab).replace("me-tab-", ""))
        except (ValueError, AttributeError):
            raise dash.exceptions.PreventUpdate
        if slot_idx >= N_SLOTS:
            raise dash.exceptions.PreventUpdate

        current_data = all_slot_data[slot_idx]
        if not current_data:
            return _ret(nochange, nochange, "No data in the active slot.")

        kwargs = generate_kwargs(param_ids, param_values) if param_ids else {}

        try:
            slot_mid = slot_map[slot_idx] if slot_idx < len(slot_map or []) else None
            src_expr = (pool_meta.get(slot_mid) or {}).get("script_expr") if slot_mid else None
            motl = Motl(pd.DataFrame(current_data))
            if src_expr:
                dash_logger.record_motl_source(motl, src_expr)
            result = invoke_operation(getattr(motl, method_name), kwargs)
        except Exception:
            return _ret(nochange, nochange, f"Error running '{method_name}' — see log.")

        gui = getattr(getattr(Motl, method_name), "_gui", {})

        # Operation produces a NEW motl -> add it to the pool, keep the source.
        if gui.get("output") == "motl" and isinstance(result, Motl):
            registry = dict(registry or {})
            pool_motls = dict(pool_motls or {})
            pool_extra = dict(pool_extra or {})
            pool_meta = dict(pool_meta or {})
            next_id = next_id or 0
            slot_map = list(slot_map or [None] * N_SLOTS)
            while len(slot_map) < N_SLOTS:
                slot_map.append(None)

            new_rows = result.df.to_dict("records")
            mid = f"motl-{next_id}"
            src_label = (registry.get(slot_map[slot_idx]) or {}).get("label", f"Slot {slot_idx + 1}")
            registry[mid] = {
                "label": f"{gui['label']} of {src_label}",
                "type": "emmotl",
                "n_rows": len(new_rows),
                "active": True,
            }
            pool_motls[mid] = new_rows
            pool_extra[mid] = None
            pool_meta[mid] = {
                "data_type": None, "relion_optics": None, "relion5_tomos": None,
                "relion5_tomos_filename": None, "relion_params": None,
                "script_expr": dash_logger.last_script_line,
            }
            free = next((i for i in range(N_SLOTS) if not slot_map[i]), None)
            if free is not None:
                slot_map[free] = mid
                active = f"me-tab-{free}"
                status = f"'{method_name}' -> new motl in slot {free + 1} ({len(new_rows)} particles)."
            else:
                active = no_update
                status = f"'{method_name}' -> new motl in the pool (no free slot; use 'Slot assignment')."
            return _ret(
                nochange, nochange, status,
                pool=(registry, pool_motls, pool_extra, pool_meta, next_id + 1, slot_map, active),
            )

        # In-place operation — update the active slot.
        if isinstance(result, Motl):
            new_data = result.df.to_dict("records")
        elif result is None:
            new_data = motl.df.to_dict("records")
        else:
            return _ret(nochange, nochange, f"Ran '{method_name}' — result: {result!r} (table unchanged).")

        data_out = [no_update] * N_SLOTS
        data_out[slot_idx] = new_data
        undo_out = [no_update] * N_SLOTS
        undo_out[slot_idx] = current_data
        status = f"'{method_name}' applied. Particles: {len(current_data)} → {len(new_data)}."
        return _ret(data_out, undo_out, status)

    # ── Undo the last operation on the active slot ─────────────────────────────
    @app.callback(
        *[Output(f"me-{i}-motl-data-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        *[Output(f"me-{i}-undo-store", "data", allow_duplicate=True) for i in range(N_SLOTS)],
        Output("me-op-status", "children", allow_duplicate=True),
        Input("me-op-undo-btn", "n_clicks"),
        State("me-tabs", "active_tab"),
        *[State(f"me-{i}-undo-store", "data") for i in range(N_SLOTS)],
        prevent_initial_call=True,
    )
    def undo_operation(n_clicks, active_tab, *all_undo_data):
        if not n_clicks or not active_tab:
            raise dash.exceptions.PreventUpdate

        try:
            slot_idx = int(str(active_tab).replace("me-tab-", ""))
        except (ValueError, AttributeError):
            raise dash.exceptions.PreventUpdate
        if slot_idx >= N_SLOTS:
            raise dash.exceptions.PreventUpdate

        undo_data = all_undo_data[slot_idx]
        if not undo_data:
            empty = [no_update] * N_SLOTS
            return (*empty, *empty, "Nothing to undo for this slot.")

        data_out = [no_update] * N_SLOTS
        data_out[slot_idx] = undo_data
        undo_out = [no_update] * N_SLOTS
        undo_out[slot_idx] = None  # one level of undo

        return (*data_out, *undo_out, "Undo successful.")
