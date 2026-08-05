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
from cryocat.app.components.motlio import motl_types, register_motl_load_callbacks
from cryocat.app.components.relionopts import get_relion_options
from cryocat.app.components.pathfield import get_path_field
from cryocat.app.components.motlsource import (
    get_multi_motl_picker, register_multi_motl_picker_callbacks,
)
from cryocat.app.apputils import (
    generate_kwargs, run_operation, run_operation_to_pool, record_load_to_pool,
)
from cryocat.app import ids
from cryocat.app import discovery as _discovery
from cryocat.app.formgen import build_form, make_dropdown
from cryocat.app.logger import invoke_operation
from cryocat.app import session as _session
from cryocat.app.event import message_event
from cryocat.app.pageshell import _SIDEBAR_STYLE, _SIDEBAR_COL_STYLE
from cryocat.app.pool import (
    PoolState, remove_motl, get_rows, PoolPayloadMissing,
    GroupState, create_group, delete_group, reorder_group, remove_from_group,
    purge_motl_from_groups, set_has_tab, natural_sort_key, insert_motl,
)

# Number of editor *view slots* (rendered table/viewer surfaces). The motl pool
# itself is unbounded — this only caps how many motls are open as tabs at once.
N_SLOTS = 5

# Fetched at import time from GUI_REGISTRY — adding a new @gui_exposed method
# to Motl is sufficient; no edits needed here.
_MOTL_METHODS       = [{"label": e.label, "value": e.fn.__name__}
                       for e in _discovery.single_motl_ops()
                       if e.fn.__module__ == "cryocat.core.cryomotl"]
_MULTI_MOTL_METHODS = [{"label": e.label, "value": e.fn.__name__, "motls": e.motls}
                       for e in _discovery.multi_motl_ops()]
# Lookup of `method_name -> motls spec` for the run callback.
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
                        html.Label(f"Slot {i + 1}:"),
                        width=3,
                        className="d-flex align-items-center",
                    ),
                    dbc.Col(
                        make_dropdown(
                            {"type": "me-slot-assign", "slot": i},
                            [{"label": "(empty)", "value": _NONE_OPT}],
                            _NONE_OPT,
                            clearable=False,
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
                                dbc.RadioItems(
                                    id="me-load-mode",
                                    options=[
                                        {"label": "Single file", "value": "single"},
                                        {"label": "Multiple (glob)", "value": "multi"},
                                    ],
                                    value="single",
                                    inline=True,
                                    style={"display": "flex", "gap": "1.5rem", "marginBottom": "0.5rem"},
                                ),
                                # Shared: type + Relion options — identical for single and glob
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.Div("Motl type: ", style={"fontStyle": "bold"}),
                                            width=4,
                                            className="d-flex align-items-center",
                                        ),
                                        dbc.Col(
                                            make_dropdown(
                                                "me-load-motl-dropdown",
                                                motl_types,
                                                "emmotl",
                                                style={"padding": "0"},
                                            ),
                                            width=8,
                                        ),
                                    ],
                                    style={"marginTop": "1rem", "marginBottom": "0.4rem"},
                                ),
                                get_relion_options("me-load", for_load=True),
                                # Single-file section
                                html.Div(
                                    id="me-load-single-section",
                                    children=[
                                        html.Div(
                                            get_path_field(
                                                "me-load-motl-path",
                                                mode="open",
                                                kind="motl",
                                                extensions=(".em", ".star", ".csv", ".tbl"),
                                                placeholder="Path to motl file",
                                            ),
                                            style={"marginTop": "0.5rem"},
                                        ),
                                        dbc.Button(
                                            "Load",
                                            id="me-load-motl-load-btn",
                                            color="primary",
                                            style={"width": "100%", "marginTop": "0.4rem"},
                                        ),
                                    ],
                                ),
                                # Multi/glob section
                                html.Div(
                                    id="me-load-multi-section",
                                    style={"display": "none"},
                                    children=[
                                        dbc.Input(
                                            id="me-mload-pattern",
                                            placeholder="e.g. /data/**/*.em or /data/runs/ (glob syntax supported)",
                                            size="sm",
                                            style={"marginBottom": "0.3rem", "marginTop": "0.5rem"},
                                        ),
                                        html.Div(
                                            id="me-mload-count",
                                            style={
                                                "color": "var(--color9)",
                                                "marginBottom": "0.3rem",
                                                "fontSize": "0.85rem",
                                            },
                                        ),
                                        dbc.Input(
                                            id="me-mload-group-name",
                                            placeholder="Group name (optional)",
                                            size="sm",
                                            style={"marginBottom": "0.4rem"},
                                        ),
                                        dbc.Button(
                                            "Load All",
                                            id="me-mload-btn",
                                            color="primary",
                                            size="sm",
                                            style={"width": "100%", "marginBottom": "0.3rem"},
                                        ),
                                        html.Div(
                                            id="me-mload-status",
                                            style={"color": "var(--color9)", "fontSize": "0.85rem"},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="me-load-status",
                                    style={
                                        "marginTop": "0.5rem",
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
                                        style={"color": "var(--color9)", "padding": "4px"},
                                    ),
                                ),
                            ],
                            title="Loaded Motls (pool)",
                            item_id="me-sidebar-list",
                        ),
                        dbc.AccordionItem(
                            [
                                html.Div(
                                    "Create a named group from pool motls. "
                                    "Groups appear collapsed in the list; "
                                    "order is preserved for multi-motl operations.",
                                    style={"color": "var(--color9)", "marginBottom": "0.5rem"},
                                ),
                                dbc.Input(
                                    id="me-create-group-name",
                                    placeholder="Group name (optional)",
                                    size="sm",
                                    style={"marginBottom": "0.4rem"},
                                ),
                                make_dropdown(
                                    "me-create-group-select",
                                    [],
                                    [],
                                    multi=True,
                                    placeholder="Select motls to group",
                                    style={"marginBottom": "0.4rem"},
                                ),
                                dbc.Button(
                                    "Create Group",
                                    id="me-create-group-btn",
                                    color="primary",
                                    size="sm",
                                    style={"width": "100%"},
                                ),
                                html.Div(
                                    id="me-create-group-status",
                                    style={"color": "var(--color9)", "marginTop": "0.4rem"},
                                ),
                            ],
                            title="Groups",
                            item_id="me-sidebar-groups",
                        ),
                        dbc.AccordionItem(
                            [
                                html.Div(
                                    "Convert all members of the active group to a single format. "
                                    "Click a group label in the pool list to select the target group.",
                                    style={"color": "var(--color9)", "marginBottom": "0.5rem"},
                                ),
                                html.Div(
                                    id="me-batch-convert-target",
                                    style={
                                        "color": "var(--color9)",
                                        "marginBottom": "0.4rem",
                                    },
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.Label("Output type:"),
                                            width=4,
                                            className="d-flex align-items-center",
                                        ),
                                        dbc.Col(
                                            make_dropdown(
                                                "me-batch-convert-format",
                                                motl_types,
                                                "emmotl",
                                                style={"padding": "0"},
                                            ),
                                            width=8,
                                        ),
                                    ],
                                    style={"marginBottom": "0.4rem"},
                                ),
                                html.Div(
                                    get_path_field(
                                        "me-batch-convert-dir",
                                        mode="directory",
                                        kind="output",
                                        placeholder="Output directory",
                                    ),
                                    style={"marginBottom": "0.4rem"},
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.Label("Filename:"),
                                            width=4,
                                            className="d-flex align-items-center",
                                        ),
                                        dbc.Col(
                                            make_dropdown(
                                                "me-batch-convert-filename-policy",
                                                [
                                                    {"label": "Keep stem + new extension", "value": "stem"},
                                                    {"label": "Add suffix", "value": "suffix"},
                                                ],
                                                "stem",
                                                style={"padding": "0"},
                                            ),
                                            width=8,
                                        ),
                                    ],
                                    style={"marginBottom": "0.3rem"},
                                ),
                                html.Div(
                                    id="me-batch-convert-suffix-row",
                                    style={"display": "none"},
                                    children=[
                                        dbc.Input(
                                            id="me-batch-convert-suffix",
                                            placeholder="Suffix to append before extension (e.g. _v2)",
                                            size="sm",
                                            style={"marginBottom": "0.3rem"},
                                        ),
                                    ],
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.Label("On conflict:"),
                                            width=4,
                                            className="d-flex align-items-center",
                                        ),
                                        dbc.Col(
                                            make_dropdown(
                                                "me-batch-convert-overwrite",
                                                [
                                                    {"label": "Refuse (list conflicts)", "value": "refuse"},
                                                    {"label": "Overwrite", "value": "overwrite"},
                                                ],
                                                "refuse",
                                                style={"padding": "0"},
                                            ),
                                            width=8,
                                        ),
                                    ],
                                    style={"marginBottom": "0.4rem"},
                                ),
                                dbc.Button(
                                    "Convert All",
                                    id="me-batch-convert-btn",
                                    color="primary",
                                    size="sm",
                                    style={"width": "100%"},
                                ),
                                html.Div(
                                    id="me-batch-convert-status",
                                    style={
                                        "color": "var(--color9)",
                                        "marginTop": "0.4rem",
                                        "wordBreak": "break-word",
                                    },
                                ),
                            ],
                            title="Batch Convert",
                            item_id="me-sidebar-batch-convert",
                        ),
                        dbc.AccordionItem(
                            [
                                html.Div(
                                    "Assign which pool motl is rendered in each editor slot. "
                                    "Loading auto-fills free slots; use these when the pool exceeds "
                                    f"{N_SLOTS} motls.",
                                    style={"color": "var(--color9)", "marginBottom": "0.5rem"},
                                ),
                                *_slot_assignment_rows(),
                            ],
                            title="Slot assignment",
                            item_id="me-sidebar-slots",
                        ),
                        dbc.AccordionItem(
                            [
                                html.Div(
                                    id="me-single-op-target-label",
                                    style={
                                        "color": "var(--color9)",
                                        "marginBottom": "0.4rem",
                                        "fontSize": "0.85rem",
                                    },
                                ),
                                make_dropdown(
                                    "me-op-func-select",
                                    _MOTL_METHODS,
                                    None,
                                    placeholder="Select operation",
                                    style={"marginBottom": "0.5rem"},
                                ),
                                html.Div(id="me-op-func-form", style={"marginBottom": "0.5rem"}),
                                html.Div(
                                    id="me-op-group-options",
                                    style={"display": "none"},
                                    children=[
                                        dbc.Checklist(
                                            id="me-op-create-group",
                                            options=[{"label": "Create a group", "value": "create"}],
                                            value=["create"],
                                            style={"marginBottom": "0.3rem"},
                                        ),
                                        dbc.Checklist(
                                            id="me-op-save-to-disk",
                                            options=[{"label": "Save to disk", "value": "save"}],
                                            value=[],
                                            style={"marginBottom": "0.3rem"},
                                        ),
                                        html.Div(
                                            id="me-op-save-options",
                                            style={"display": "none"},
                                            children=[
                                                html.Div(
                                                    get_path_field(
                                                        "me-op-save-dir",
                                                        mode="directory",
                                                        kind="output",
                                                        placeholder="Output directory",
                                                    ),
                                                    style={"marginBottom": "0.3rem"},
                                                ),
                                                make_dropdown(
                                                    "me-op-save-format",
                                                    motl_types,
                                                    "emmotl",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            id="me-op-group-validation",
                                            style={
                                                "color": "var(--color9)",
                                                "marginTop": "0.2rem",
                                            },
                                        ),
                                    ],
                                ),
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
                                    style={**{"display": "flex"}, "marginBottom": "0.5rem"},
                                ),
                                html.Div(
                                    id="me-op-status",
                                    style={"color": "var(--color9)", "wordBreak": "break-word"},
                                ),
                            ],
                            title="Single Motl Operations",
                            item_id="me-sidebar-singlemotl",
                        ),
                        dbc.AccordionItem(
                            [
                                html.Div(
                                    id="me-multi-op-target-label",
                                    style={
                                        "color": "var(--color9)",
                                        "marginBottom": "0.4rem",
                                        "fontSize": "0.85rem",
                                    },
                                ),
                                make_dropdown(
                                    "me-multi-op-select",
                                    _MULTI_MOTL_METHODS,
                                    None,
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
                                        "marginTop": "0.5rem",                                        "color": "var(--color9)", "wordBreak": "break-word",
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
            style=_SIDEBAR_STYLE,
        ),
        id="me-sidebar",
        width=3,
        style=_SIDEBAR_COL_STYLE,
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


def register_motl_editor_sidebar_callbacks(app):

    register_motl_load_callbacks(app, "me-load")

    # ── Load → pool ────────────────────────────────────────────────────────────
    # A freshly loaded motl is appended to the pool with a new motl_id and
    # auto-assigned to the first free view slot (if any).
    @app.callback(
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output("me-slot-map", "data", allow_duplicate=True),
        Output("me-tabs", "active_tab", allow_duplicate=True),
        Output("me-load-status", "children", allow_duplicate=True),
        Input("me-load-motl-data-store", "data"),
        State("me-load-motl-extra-data-store", "data"),
        State("me-load-motl-data-type", "data"),
        State("me-load-relion-optics-store", "data"),
        State("me-load-rln-tomos-store", "data"),
        State("me-load-rln-tomos-filename", "data"),
        State({"type": "path-input", "owner": "me-load-motl-path"}, "value"),
        State("me-load-relion-params-store", "data"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        State("me-slot-map", "data"),
        prevent_initial_call=True,
    )
    def route_motl(
        motl_data, extra, dtype, optics, r5t, r5tn, motl_path, relion_params,
        registry, pool_meta, next_id, slot_map,
    ):
        import os as _os
        from cryocat.app.components.filesystem import resolve_input as _resolve

        if not motl_data:
            raise dash.exceptions.PreventUpdate

        slot_map = list(slot_map or [None] * N_SLOTS)
        while len(slot_map) < N_SLOTS:
            slot_map.append(None)
        # Drop stale slot references from a previous session if the pool restarted.
        _live_ids = set((registry or {}).keys())
        slot_map = [m if (m and m in _live_ids) else None for m in slot_map]

        _nid = next_id or 0
        # Use the basename of the resolved path as the label; fall back to a counter.
        resolved_path, _err = _resolve(motl_path or "")
        label = _os.path.basename(resolved_path) if resolved_path and not _err else f"Motl {_nid + 1}"
        effective_type = dtype or "emmotl"

        # Build rln_kwargs for script rendering (pixel_size/binning/version/formats).
        rln_kwargs: dict = {}
        if effective_type in ("relion", "relion5", "relion5_1") and relion_params:
            if effective_type == "relion":
                ver_str = relion_params.get("version", "")
                try:
                    rln_kwargs["version"] = float(str(ver_str).split()[-1])
                except (ValueError, IndexError):
                    pass
            ps = relion_params.get("pixel_size")
            if ps:
                try:
                    rln_kwargs["pixel_size"] = float(ps)
                except (TypeError, ValueError):
                    pass  # per-particle (multi-optics): not representable as a single kwarg
            bn = relion_params.get("binning")
            if bn:
                rln_kwargs["binning"] = float(bn)
            tf = relion_params.get("tomo_format")
            if tf:
                rln_kwargs["tomo_format"] = tf
            sf = relion_params.get("subtomo_format")
            if sf:
                rln_kwargs["subtomo_format"] = sf

        current_pool = PoolState.from_stores(registry, pool_meta, next_id)
        try:
            pool_state, mid, _ = record_load_to_pool(
                motl_data, effective_type, resolved_path or label, rln_kwargs,
                current_pool,
                label=label,
                extra=extra,
                meta={
                    "data_type": dtype,
                    "relion_optics": optics,
                    "relion5_tomos": r5t,
                    "relion5_tomos_filename": r5tn,
                    "relion_params": relion_params,
                },
            )
        except Exception as exc:
            return (no_update, no_update, no_update, slot_map, no_update, f"Load failed: {exc}")

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

        return (*pool_state.to_stores(), slot_map, active_tab, status)

    # ── Pool motl list ─────────────────────────────────────────────────────────
    @app.callback(
        Output("me-motl-list", "children"),
        Input(ids.POOL_REGISTRY, "data"),
        Input(ids.POOL_GROUPS, "data"),
        Input("me-slot-map", "data"),
        Input("me-group-expand", "data"),
        Input("me-active-target", "data"),
        prevent_initial_call=True,
    )
    def update_motl_list(registry, groups_data, _slot_map, expand_data, active_target):
        registry = registry or {}
        gstate = GroupState.from_store(groups_data)
        expand_data = expand_data or {}
        active_target = active_target or {}
        at_type = active_target.get("type")
        at_id = active_target.get("id")
        items = []

        # ── Groups first ──────────────────────────────────────────────────────
        for gid, g in gstate.groups.items():
            members = list(g.get("members", []))
            glabel = g.get("label", gid)
            expanded = expand_data.get(gid, False)
            toggle_icon = "▾" if expanded else "▶"
            is_active_group = (at_type == "group" and at_id == gid)
            group_row_style = {"display": "flex", "alignItems": "center", "padding": "4px 8px"}
            if is_active_group:
                group_row_style["backgroundColor"] = "var(--bs-primary-bg-subtle)"
            items.append(
                dbc.ListGroupItem(
                    [
                        dbc.Button(
                            toggle_icon,
                            id={"type": "me-group-toggle", "gid": gid},
                            color="link",
                            size="sm",
                            style={"padding": "0 4px", "fontFamily": "monospace", "color": "inherit"},
                        ),
                        html.Span(
                            f"{glabel} ({len(members)})",
                            id={"type": "me-group-label-click", "gid": gid},
                            n_clicks=0,
                            style={
                                "flex": "1",
                                "fontWeight": "500",
                                "overflow": "hidden",
                                "textOverflow": "ellipsis",
                                "whiteSpace": "nowrap",
                                "cursor": "pointer",
                            },
                        ),
                        dbc.Button(
                            "×",
                            id={"type": "me-group-delete", "gid": gid},
                            color="link",
                            size="sm",
                            style={"padding": "0 6px", "color": "var(--color9)"},
                        ),
                    ],
                    style=group_row_style,
                )
            )
            if expanded:
                for idx, mid in enumerate(members):
                    mmeta = registry.get(mid) or {}
                    mlabel = mmeta.get("label", mid)
                    is_first = idx == 0
                    is_last = idx == len(members) - 1
                    items.append(
                        dbc.ListGroupItem(
                            [
                                html.Span(
                                    f"{mlabel} ({mid.replace('-', '_')})",
                                    style={
                                        "flex": "1",
                                        "paddingLeft": "1.5rem",
                                        "overflow": "hidden",
                                        "textOverflow": "ellipsis",
                                        "whiteSpace": "nowrap",
                                        "color": "var(--color9)" if not mmeta else "inherit",
                                    },
                                ),
                                dbc.Button(
                                    "↑",
                                    id={"type": "me-member-up", "gid": gid, "mid": mid},
                                    color="link",
                                    size="sm",
                                    disabled=is_first,
                                    style={"padding": "0 3px", "color": "inherit"},
                                ),
                                dbc.Button(
                                    "↓",
                                    id={"type": "me-member-down", "gid": gid, "mid": mid},
                                    color="link",
                                    size="sm",
                                    disabled=is_last,
                                    style={"padding": "0 3px", "color": "inherit"},
                                ),
                                dbc.Button(
                                    "Open",
                                    id={"type": "me-member-open", "gid": gid, "mid": mid},
                                    color="link",
                                    size="sm",
                                    style={"padding": "0 4px", "color": "inherit"},
                                ),
                                dbc.Button(
                                    "−",
                                    id={"type": "me-member-remove", "gid": gid, "mid": mid},
                                    color="link",
                                    size="sm",
                                    style={"padding": "0 4px", "color": "var(--color9)"},
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center", "padding": "2px 8px"},
                        )
                    )

        # ── Ungrouped motls (R2: grouped motls never appear here) ────────────
        all_grouped = {
            mid
            for g in gstate.groups.values()
            for mid in g.get("members", [])
        }
        for mid, meta in registry.items():
            if not meta.get("active", True):
                continue
            if mid in all_grouped:
                continue  # R2: grouped motls appear only under their group
            label = meta.get("label", mid)
            is_active_motl = (at_type == "motl" and at_id == mid)
            motl_row_style = {"display": "flex", "alignItems": "center", "padding": "4px 8px", "cursor": "pointer"}
            if is_active_motl:
                motl_row_style["backgroundColor"] = "var(--bs-primary-bg-subtle)"
            items.append(
                dbc.ListGroupItem(
                    [
                        html.Span(
                            f"{label} (id: {mid.replace('-', '_')})",
                            style={
                                "flex": "1",
                                "overflow": "hidden",
                                "textOverflow": "ellipsis",
                                "whiteSpace": "nowrap",
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
                    style=motl_row_style,
                )
            )

        if not items:
            return html.Div("No motls loaded.", style={"color": "var(--color9)", "padding": "4px"})
        return dbc.ListGroup(items, flush=True)

    # ── Slot-assignment dropdowns: render from slot_map + registry ─────────────
    @app.callback(
        Output({"type": "me-slot-assign", "slot": ALL}, "options"),
        Output({"type": "me-slot-assign", "slot": ALL}, "value"),
        Input(ids.POOL_REGISTRY, "data"),
        Input("me-slot-map", "data"),
        prevent_initial_call=True,
    )
    def render_slot_assignment(registry, slot_map):
        registry = registry or {}
        slot_map = list(slot_map or [None] * N_SLOTS)
        motl_opts = [
            {"label": m.get("label", mid), "value": mid}
            for mid, m in registry.items()
            if m.get("active", True) and m.get("has_tab", True)
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

    # ── Clicking a pool motl activates its slot tab and sets the active target ──
    @app.callback(
        Output("me-tabs", "active_tab", allow_duplicate=True),
        Output("me-active-target", "data", allow_duplicate=True),
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
        active_target = {"type": "motl", "id": mid}
        slot_map = slot_map or [None] * N_SLOTS
        for i, m in enumerate(slot_map):
            if m == mid:
                return f"me-tab-{i}", active_target
        return no_update, active_target  # not in any slot but still becomes active target

    # ── Close a pool motl: drop it from the pool, free its slot, purge from groups
    @app.callback(
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output(ids.POOL_GROUPS, "data", allow_duplicate=True),  # remove from any group
        Output("me-slot-map", "data", allow_duplicate=True),
        Output("me-tabs", "active_tab", allow_duplicate=True),
        Input({"type": "me-close-motl", "mid": ALL}, "n_clicks"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        State(ids.POOL_GROUPS, "data"),
        State("me-slot-map", "data"),
        State("me-tabs", "active_tab"),
        prevent_initial_call=True,
    )
    def close_motl(n_clicks_list, registry, pool_meta, next_id, groups_data, slot_map, active_tab):
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "mid" in triggered):
            raise dash.exceptions.PreventUpdate

        mid = triggered["mid"]
        pool_state = remove_motl(PoolState.from_stores(registry, pool_meta, next_id), mid)
        gstate = purge_motl_from_groups(GroupState.from_store(groups_data), mid)

        old_map = list(slot_map or [None] * N_SLOTS)
        closed_slot = next((i for i, m in enumerate(old_map) if m == mid), None)
        new_map = [None if m == mid else m for m in old_map]

        new_active = no_update
        if closed_slot is not None and active_tab == f"me-tab-{closed_slot}":
            nxt = next((i for i, m in enumerate(new_map) if m), None)
            new_active = f"me-tab-{nxt}" if nxt is not None else "me-tab-0"

        return (*pool_state.to_stores(), gstate.to_store(), new_map, new_active)

    # ── Group: toggle expand/collapse ─────────────────────────────────────────
    @app.callback(
        Output("me-group-expand", "data", allow_duplicate=True),
        Input({"type": "me-group-toggle", "gid": ALL}, "n_clicks"),
        State("me-group-expand", "data"),
        prevent_initial_call=True,
    )
    def toggle_group(n_clicks_list, expand_data):
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "gid" in triggered):
            raise dash.exceptions.PreventUpdate
        gid = triggered["gid"]
        expand_data = dict(expand_data or {})
        expand_data[gid] = not expand_data.get(gid, False)
        return expand_data

    # ── Group: delete group (motls remain in pool) ─────────────────────────────
    @app.callback(
        Output(ids.POOL_GROUPS, "data", allow_duplicate=True),
        Output("me-group-expand", "data", allow_duplicate=True),
        Input({"type": "me-group-delete", "gid": ALL}, "n_clicks"),
        State(ids.POOL_GROUPS, "data"),
        State("me-group-expand", "data"),
        prevent_initial_call=True,
    )
    def delete_group_cb(n_clicks_list, groups_data, expand_data):
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "gid" in triggered):
            raise dash.exceptions.PreventUpdate
        gid = triggered["gid"]
        gstate = delete_group(GroupState.from_store(groups_data), gid)
        expand_data = {k: v for k, v in (expand_data or {}).items() if k != gid}
        return gstate.to_store(), expand_data

    # ── Group member: move up in order ─────────────────────────────────────────
    @app.callback(
        Output(ids.POOL_GROUPS, "data", allow_duplicate=True),
        Input({"type": "me-member-up", "gid": ALL, "mid": ALL}, "n_clicks"),
        State(ids.POOL_GROUPS, "data"),
        prevent_initial_call=True,
    )
    def move_member_up(n_clicks_list, groups_data):
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "gid" in triggered):
            raise dash.exceptions.PreventUpdate
        gid, mid = triggered["gid"], triggered["mid"]
        gstate = GroupState.from_store(groups_data)
        members = list((gstate.groups.get(gid) or {}).get("members", []))
        idx = next((i for i, m in enumerate(members) if m == mid), None)
        if idx is None or idx == 0:
            raise dash.exceptions.PreventUpdate
        members[idx - 1], members[idx] = members[idx], members[idx - 1]
        return reorder_group(gstate, gid, members).to_store()

    # ── Group member: move down in order ───────────────────────────────────────
    @app.callback(
        Output(ids.POOL_GROUPS, "data", allow_duplicate=True),
        Input({"type": "me-member-down", "gid": ALL, "mid": ALL}, "n_clicks"),
        State(ids.POOL_GROUPS, "data"),
        prevent_initial_call=True,
    )
    def move_member_down(n_clicks_list, groups_data):
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "gid" in triggered):
            raise dash.exceptions.PreventUpdate
        gid, mid = triggered["gid"], triggered["mid"]
        gstate = GroupState.from_store(groups_data)
        members = list((gstate.groups.get(gid) or {}).get("members", []))
        idx = next((i for i, m in enumerate(members) if m == mid), None)
        if idx is None or idx == len(members) - 1:
            raise dash.exceptions.PreventUpdate
        members[idx], members[idx + 1] = members[idx + 1], members[idx]
        return reorder_group(gstate, gid, members).to_store()

    # ── Group member: remove from group (motl stays in pool) ───────────────────
    @app.callback(
        Output(ids.POOL_GROUPS, "data", allow_duplicate=True),
        Input({"type": "me-member-remove", "gid": ALL, "mid": ALL}, "n_clicks"),
        State(ids.POOL_GROUPS, "data"),
        prevent_initial_call=True,
    )
    def remove_member_cb(n_clicks_list, groups_data):
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "gid" in triggered):
            raise dash.exceptions.PreventUpdate
        gid, mid = triggered["gid"], triggered["mid"]
        return remove_from_group(GroupState.from_store(groups_data), gid, mid).to_store()

    # ── Group member: open in editor (sets has_tab=True, assigns to free slot) ──
    @app.callback(
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output("me-slot-map", "data", allow_duplicate=True),
        Output("me-tabs", "active_tab", allow_duplicate=True),
        Input({"type": "me-member-open", "gid": ALL, "mid": ALL}, "n_clicks"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        State("me-slot-map", "data"),
        prevent_initial_call=True,
    )
    def open_member_in_editor(n_clicks_list, registry, pool_meta, next_id, slot_map):
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "mid" in triggered):
            raise dash.exceptions.PreventUpdate
        mid = triggered["mid"]
        pool_state = set_has_tab(PoolState.from_stores(registry, pool_meta, next_id), mid, True)
        slot_map = list(slot_map or [None] * N_SLOTS)
        while len(slot_map) < N_SLOTS:
            slot_map.append(None)
        free = _first_free_slot(slot_map)
        if free is not None:
            slot_map[free] = mid
            active = f"me-tab-{free}"
        else:
            active = no_update
        return (*pool_state.to_stores(), slot_map, active)

    # ── Group: create from selected pool motls ─────────────────────────────────
    @app.callback(
        Output(ids.POOL_GROUPS, "data", allow_duplicate=True),
        Output("me-create-group-status", "children"),
        Input("me-create-group-btn", "n_clicks"),
        State("me-create-group-select", "value"),
        State("me-create-group-name", "value"),
        State(ids.POOL_GROUPS, "data"),
        prevent_initial_call=True,
    )
    def create_group_cb(n_clicks, selected_motls, group_name, groups_data):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not selected_motls:
            return no_update, "Select at least one motl to form a group."
        gstate, gid = create_group(
            GroupState.from_store(groups_data),
            selected_motls,
            label=group_name.strip() if group_name and group_name.strip() else None,
        )
        g = gstate.groups[gid]
        return gstate.to_store(), f"Group '{g['label']}' created with {len(g['members'])} motl(s)."

    # ── Group: populate create-group dropdown from pool ────────────────────────
    @app.callback(
        Output("me-create-group-select", "options"),
        Input(ids.POOL_REGISTRY, "data"),
    )
    def populate_create_group_select(registry):
        registry = registry or {}
        return [
            {"label": f"{m.get('label', mid)} ({mid.replace('-', '_')})", "value": mid}
            for mid, m in registry.items()
            if m.get("active", True)
        ]

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
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output("me-slot-map", "data", allow_duplicate=True),
        Output("me-tabs", "active_tab", allow_duplicate=True),
        Output("me-multi-op-status", "children", allow_duplicate=True),
        Input("me-multi-run-btn", "n_clicks"),
        State("me-multi-op-select", "value"),
        State("me-multi-main-select", "value"),
        State("me-multi-second-select", "value"),
        State("me-multi-list-select", "value"),
        State({"type": "me-multi-param", "owner": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": "me-multi-param", "owner": ALL, "param": ALL, "tag": ALL}, "id"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        State("me-slot-map", "data"),
        prevent_initial_call=True,
    )
    def run_multi_op(
        n_clicks, method_name, main_id, second_id, list_ids,
        param_values, param_ids,
        registry, pool_meta, next_id, slot_map,
    ):
        pool_noup = (no_update,) * 5

        def _err(msg):
            return (*pool_noup, msg)

        if not n_clicks or not method_name:
            raise dash.exceptions.PreventUpdate

        spec = _MULTI_MOTL_SPECS.get(method_name)
        if spec is None:
            return _err(f"Operation '{method_name}' is not registered as multi-motl.")

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
                try:
                    rows_df = get_rows(mid)
                except PoolPayloadMissing as exc:
                    return _err(str(exc))
                motl_obj = Motl(rows_df)
                motl_obj._pool_motl_id = mid
                motls.append(motl_obj)
        except Exception as exc:
            _session.emit(message_event(f"Error preparing motls: {exc}", level="error"))
            return _err(f"Error preparing motls: {exc}")

        # 2) Scalar kwargs from the auto-form.
        current_pool = PoolState.from_stores(registry, pool_meta, next_id)
        kwargs = generate_kwargs(param_ids, param_values, pool_state=current_pool) if param_ids else {}

        # 3) Pre-compute label (does not depend on result).
        gui = getattr(getattr(Motl, method_name).__func__, "_gui", {})
        op_label = gui.get("label", method_name)
        src_labels = [(registry.get(oid) or {}).get("label", oid) for oid in ordered_ids]
        slot_map = list(slot_map or [None] * N_SLOTS)
        while len(slot_map) < N_SLOTS:
            slot_map.append(None)

        # 4) Build kwargs and run through the atomic chokepoint.
        try:
            fn = getattr(Motl, method_name)
            if spec["arity"] == "pair":
                sig_params = list(inspect.signature(fn).parameters.keys())
                full_kwargs = {sig_params[0]: motls[0], sig_params[1]: motls[1], **kwargs}
            else:
                list_param = spec.get("param", "motl_list")
                full_kwargs = {list_param: motls, **kwargs}
            pool_state, mid, result = run_operation_to_pool(
                fn, full_kwargs, current_pool,
                label=f"{op_label} of {' + '.join(src_labels)}",
            )
        except Exception as exc:
            return _err(f"Error running '{method_name}': {exc}")

        if not isinstance(result, Motl):
            return _err(f"'{method_name}' did not return a Motl (got {type(result).__name__}).")

        free = _first_free_slot(slot_map)
        if free is not None:
            slot_map[free] = mid
            active = f"me-tab-{free}"
            status = (
                f"'{op_label}' -> new motl in slot {free + 1} "
                f"({len(result.df)} particles, from {len(motls)} input motl(s))."
            )
        else:
            active = no_update
            status = (
                f"'{op_label}' -> new motl in the pool "
                f"({len(result.df)} particles; no free slot, use 'Slot assignment')."
            )

        return (*pool_state.to_stores(), slot_map, active, status)

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
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output("me-slot-map", "data", allow_duplicate=True),
        Output("me-tabs", "active_tab", allow_duplicate=True),
        Output(ids.POOL_GROUPS, "data", allow_duplicate=True),
        Input("me-op-apply-btn", "n_clicks"),
        State("me-op-func-select", "value"),
        State("me-tabs", "active_tab"),
        State({"type": "me-op-param", "owner": ALL, "param": ALL, "tag": ALL}, "value"),
        State({"type": "me-op-param", "owner": ALL, "param": ALL, "tag": ALL}, "id"),
        *[State(f"me-{i}-motl-data-store", "data") for i in range(N_SLOTS)],
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        State("me-slot-map", "data"),
        State("me-op-create-group", "value"),
        State("me-op-save-to-disk", "value"),
        State({"type": "path-input", "owner": "me-op-save-dir"}, "value"),
        State("me-op-save-format", "value"),
        State(ids.POOL_GROUPS, "data"),
        prevent_initial_call=True,
    )
    def apply_operation(n_clicks, method_name, active_tab, param_values, param_ids, *rest):
        all_slot_data = rest[:N_SLOTS]
        (registry, pool_meta, next_id, slot_map,
         create_group_val, save_to_disk_val, save_dir_val, save_fmt_val, groups_data) = rest[N_SLOTS:]

        # 5 pool-related outputs: registry, meta, next_id, slot_map, active_tab
        pool_noup = (no_update,) * 5
        nochange = [no_update] * N_SLOTS

        def _ret(data_out, undo_out, status, pool=pool_noup, groups=no_update):
            return (*data_out, *undo_out, status, *pool, groups)

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

        current_pool = PoolState.from_stores(registry, pool_meta, next_id)
        kwargs = generate_kwargs(param_ids, param_values, pool_state=current_pool) if param_ids else {}
        gui = getattr(getattr(Motl, method_name), "_gui", {})

        # Operation produces a list of Motls — route through the group output path.
        if gui.get("output") == "motl_group":
            import os as _os
            want_group = bool(create_group_val)
            want_save = bool(save_to_disk_val)
            slot_map = list(slot_map or [None] * N_SLOTS)
            while len(slot_map) < N_SLOTS:
                slot_map.append(None)
            src_mid = slot_map[slot_idx]
            src_label = (registry.get(src_mid) or {}).get("label", f"Slot {slot_idx + 1}")
            motl = Motl(pd.DataFrame(current_data))
            motl._pool_motl_id = src_mid
            op_label = gui.get("label", method_name)
            try:
                result_list = invoke_operation(getattr(motl, method_name), kwargs)
            except Exception as exc:
                return _ret(nochange, nochange, f"Error running '{method_name}': {exc}")
            if not isinstance(result_list, list):
                return _ret(nochange, nochange,
                            f"'{method_name}' did not return a list (got {type(result_list).__name__}).")
            col_name = kwargs.get("column_name", "")
            new_ids = []
            for i, m in enumerate(result_list):
                df = m.df if hasattr(m, "df") else pd.DataFrame()
                stem = (
                    f"{src_label}_{col_name}_{i + 1}"
                    if col_name else
                    f"{src_label}_{op_label}_{i + 1}"
                )
                current_pool, new_mid = insert_motl(current_pool, df, label=stem, has_tab=False)
                new_ids.append(new_mid)
            new_gstate = GroupState.from_store(groups_data)
            if want_group and new_ids:
                glabel = (
                    f"{src_label}_by_{col_name}" if col_name
                    else f"{op_label} of {src_label}"
                )
                new_gstate, _ = create_group(new_gstate, new_ids, label=glabel)
            if want_save and new_ids and save_dir_val:
                _ext_map = {"emmotl": ".em", "stopgap": ".csv", "dynamo": ".tbl",
                            "relion": ".star", "relion5": ".star", "relion5_1": ".star"}
                _fmt = save_fmt_val or "emmotl"
                _ext = _ext_map.get(_fmt, ".em")
                for new_mid, m in zip(new_ids, result_list):
                    _lbl = (current_pool.registry.get(new_mid) or {}).get("label", new_mid)
                    _path = _os.path.join(save_dir_val.strip(), _lbl + _ext)
                    try:
                        run_operation(m.write_out, {"output_path": _path, "motl_type": _fmt})
                    except Exception as exc:
                        pass
            status = f"'{op_label}' → {len(result_list)} motl(s)"
            if want_group:
                status += ", grouped in pool"
            if want_save:
                status += ", saved to disk"
            return _ret(nochange, nochange, status,
                        pool=(*current_pool.to_stores(), slot_map, no_update),
                        groups=new_gstate.to_store())

        # Operation produces a NEW motl — route through the atomic chokepoint.
        if gui.get("output") == "motl":
            slot_map = list(slot_map or [None] * N_SLOTS)
            while len(slot_map) < N_SLOTS:
                slot_map.append(None)
            src_label = (registry.get(slot_map[slot_idx]) or {}).get("label", f"Slot {slot_idx + 1}")
            motl = Motl(pd.DataFrame(current_data))
            motl._pool_motl_id = slot_map[slot_idx]
            try:
                pool_state, mid, result = run_operation_to_pool(
                    getattr(motl, method_name), kwargs, current_pool,
                    label=f"{gui.get('label', method_name)} of {src_label}",
                )
            except Exception as exc:
                return _ret(nochange, nochange, f"Error running '{method_name}': {exc}")
            free = next((i for i in range(N_SLOTS) if not slot_map[i]), None)
            if free is not None:
                slot_map[free] = mid
                active = f"me-tab-{free}"
                status = f"'{method_name}' -> new motl in slot {free + 1} ({len(result.df)} particles)."
            else:
                active = no_update
                status = f"'{method_name}' -> new motl in the pool (no free slot; use 'Slot assignment')."
            return _ret(
                nochange, nochange, status,
                pool=(*pool_state.to_stores(), slot_map, active),
            )

        # In-place operation — update the active slot.
        try:
            motl = Motl(pd.DataFrame(current_data))
            slot_map_list = list(slot_map or [])
            motl._pool_motl_id = slot_map_list[slot_idx] if slot_idx < len(slot_map_list) else None
            result = invoke_operation(getattr(motl, method_name), kwargs)
        except Exception as exc:
            return _ret(nochange, nochange, f"Error running '{method_name}': {exc}")

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

    # ── R1: toggle single / multiple load mode ─────────────────────────────────
    @app.callback(
        Output("me-load-single-section", "style"),
        Output("me-load-multi-section", "style"),
        Input("me-load-mode", "value"),
    )
    def _toggle_load_mode(mode):
        if mode == "multi":
            return {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}

    # ── R1: live match count ────────────────────────────────────────────────────
    @app.callback(
        Output("me-mload-count", "children"),
        Input("me-mload-pattern", "value"),
        prevent_initial_call=True,
    )
    def _count_matches(pattern):
        import glob as _glob
        import os as _os
        from pathlib import Path as _Path
        if not pattern:
            return ""
        pattern = pattern.strip()
        if _os.path.isdir(pattern):
            matches = [
                str(p) for p in _Path(pattern).iterdir()
                if p.is_file() and p.suffix.lower() in (".em", ".star", ".csv", ".tbl")
            ]
        else:
            matches = [m for m in _glob.glob(pattern, recursive=True) if _os.path.isfile(m)]
        n = len(matches)
        return f"{n} file(s) matched" if n else "No files matched"

    # ── R1: Load All — expand glob, load each, insert into pool, create group ──
    @app.callback(
        Output(ids.POOL_REGISTRY, "data", allow_duplicate=True),
        Output(ids.POOL_META, "data", allow_duplicate=True),
        Output(ids.POOL_NEXT_ID, "data", allow_duplicate=True),
        Output(ids.POOL_GROUPS, "data", allow_duplicate=True),
        Output("me-mload-status", "children"),
        Input("me-mload-btn", "n_clicks"),
        State("me-load-motl-dropdown", "value"),
        State("me-load-rln-value", "data"),
        State("me-load-rln-tomos-store", "data"),
        State("me-mload-pattern", "value"),
        State("me-mload-group-name", "value"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_META, "data"),
        State(ids.POOL_NEXT_ID, "data"),
        State(ids.POOL_GROUPS, "data"),
        prevent_initial_call=True,
    )
    def _load_multi(n_clicks, motl_type, rln_value, rln_tomos, pattern, group_name, registry, pool_meta, next_id, groups_data):
        import glob as _glob
        import os as _os
        from pathlib import Path as _Path
        from cryocat.app.components._motlio_ops import load_motl_from_path as _load, load_kwargs_from_store as _lkfs

        _noup4 = (no_update,) * 4

        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not pattern:
            return (*_noup4, "Enter a glob pattern or folder path.")

        motl_type = motl_type or "emmotl"
        pattern = pattern.strip()

        if _os.path.isdir(pattern):
            ext_map = {"emmotl": ".em", "stopgap": ".csv", "dynamo": ".tbl", "relion": ".star"}
            ext = ext_map.get(motl_type, ".em")
            raw = [str(p) for p in _Path(pattern).iterdir() if p.is_file() and p.suffix.lower() == ext]
        else:
            raw = [m for m in _glob.glob(pattern, recursive=True) if _os.path.isfile(m)]

        if not raw:
            return (*_noup4, "No files matched.")

        matches = sorted(raw, key=lambda p: natural_sort_key(_Path(p).stem))

        pool_state = PoolState.from_stores(registry, pool_meta, next_id)
        gstate = GroupState.from_store(groups_data)
        new_mids = []
        failed = []
        _rln_kws = _lkfs(rln_value)

        for path in matches:
            label = _Path(path).stem
            try:
                table_data, extra_data, _optics, _rln_t, _dtype, _rln_params = _load(
                    path, motl_type, rln_tomos=rln_tomos, **_rln_kws,
                )
            except Exception as exc:
                failed.append(f"{label}: {exc}")
                continue
            try:
                pool_state, mid, _ = record_load_to_pool(
                    table_data, motl_type, path, {},
                    pool_state, label=label, extra=extra_data,
                )
                new_mids.append(mid)
            except Exception as exc:
                failed.append(f"{label}: {exc}")

        if not new_mids:
            return (*_noup4, f"All loads failed: {'; '.join(failed[:3])}")

        gname = group_name.strip() if group_name and group_name.strip() else None
        gstate, gid = create_group(gstate, new_mids, label=gname)
        status = f"Loaded {len(new_mids)} motl(s) → group '{gstate.groups[gid]['label']}'."
        if failed:
            status += f"  {len(failed)} failed: {failed[0]}{'…' if len(failed) > 1 else ''}."

        return (*pool_state.to_stores(), gstate.to_store(), status)

    # ── R3: clicking a group label sets it as the active target ────────────────
    @app.callback(
        Output("me-active-target", "data", allow_duplicate=True),
        Input({"type": "me-group-label-click", "gid": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _set_active_from_group(n_clicks_list):
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered_id
        if not (isinstance(triggered, dict) and "gid" in triggered):
            raise dash.exceptions.PreventUpdate
        return {"type": "group", "id": triggered["gid"]}

    # ── R3: active target label for both operation panels ─────────────────────
    @app.callback(
        Output("me-single-op-target-label", "children"),
        Output("me-multi-op-target-label", "children"),
        Input("me-active-target", "data"),
        Input(ids.POOL_REGISTRY, "data"),
        Input(ids.POOL_GROUPS, "data"),
    )
    def _update_op_target_labels(active_target, registry, groups_data):
        if not active_target:
            return "", ""
        registry = registry or {}
        at_type = active_target.get("type")
        at_id = active_target.get("id")
        if at_type == "motl":
            meta = registry.get(at_id) or {}
            label = meta.get("label", at_id)
            n = meta.get("n_rows", "?")
            base = f"Target: {label} ({n} particles)"
            return base, base
        if at_type == "group":
            gstate = GroupState.from_store(groups_data)
            g = gstate.groups.get(at_id) or {}
            glabel = g.get("label", at_id)
            n = len(g.get("members", []))
            return (
                f"Target: {glabel} ({n} motls — apply to each)",
                f"Target: {glabel} ({n} motls — apply together)",
            )
        return "", ""

    # ── Show/hide group-options panel when a motl_group op is selected ─────────
    @app.callback(
        Output("me-op-group-options", "style"),
        Input("me-op-func-select", "value"),
    )
    def _toggle_group_options(method_name):
        if not method_name:
            return {"display": "none"}
        gui = getattr(getattr(Motl, method_name, None), "_gui", {})
        if gui.get("output") == "motl_group":
            return {"display": "block"}
        return {"display": "none"}

    @app.callback(
        Output("me-op-save-options", "style"),
        Input("me-op-save-to-disk", "value"),
    )
    def _toggle_save_options(val):
        return {"display": "block"} if val else {"display": "none"}

    @app.callback(
        Output("me-op-apply-btn", "disabled"),
        Output("me-op-group-validation", "children"),
        Input("me-op-create-group", "value"),
        Input("me-op-save-to-disk", "value"),
        Input("me-op-func-select", "value"),
    )
    def _validate_group_options(create_val, save_val, method_name):
        if not method_name:
            return False, ""
        gui = getattr(getattr(Motl, method_name, None), "_gui", {})
        if gui.get("output") != "motl_group":
            return False, ""
        if not create_val and not save_val:
            return True, "Select at least one output option (group or save)."
        return False, ""

    # ── Batch Convert callbacks ────────────────────────────────────────────────
    @app.callback(
        Output("me-batch-convert-target", "children"),
        Input("me-active-target", "data"),
        Input(ids.POOL_GROUPS, "data"),
    )
    def _update_batch_target(active_target, groups_data):
        if not active_target or active_target.get("type") != "group":
            return "No group selected. Click a group label in the pool list to select it."
        gid = active_target["id"]
        gstate = GroupState.from_store(groups_data)
        g = gstate.groups.get(gid) or {}
        glabel = g.get("label", gid)
        n = len(g.get("members", []))
        return f"Group: {glabel} ({n} motl(s))"

    @app.callback(
        Output("me-batch-convert-suffix-row", "style"),
        Input("me-batch-convert-filename-policy", "value"),
    )
    def _toggle_suffix_row(policy):
        return {"display": "block"} if policy == "suffix" else {"display": "none"}

    @app.callback(
        Output("me-batch-convert-status", "children"),
        Input("me-batch-convert-btn", "n_clicks"),
        State("me-active-target", "data"),
        State("me-batch-convert-format", "value"),
        State({"type": "path-input", "owner": "me-batch-convert-dir"}, "value"),
        State("me-batch-convert-filename-policy", "value"),
        State("me-batch-convert-suffix", "value"),
        State("me-batch-convert-overwrite", "value"),
        State(ids.POOL_REGISTRY, "data"),
        State(ids.POOL_GROUPS, "data"),
        prevent_initial_call=True,
    )
    def _batch_convert(n_clicks, active_target, motl_type, out_dir,
                       filename_policy, suffix, overwrite, registry, groups_data):
        import os as _os
        import pathlib as _pathlib

        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not active_target or active_target.get("type") != "group":
            return "Select a group first (click its label in the pool list)."
        gid = active_target["id"]
        gstate = GroupState.from_store(groups_data)
        g = gstate.groups.get(gid) or {}
        members = list(g.get("members", []))
        if not members:
            return "The selected group has no members."
        if not out_dir:
            return "Specify an output directory."
        out_dir = out_dir.strip()
        motl_type = motl_type or "emmotl"
        _ext_map = {
            "emmotl": ".em", "stopgap": ".csv", "dynamo": ".tbl",
            "relion": ".star", "relion5": ".star", "relion5_1": ".star",
        }
        _ext = _ext_map.get(motl_type, ".em")

        # Build output paths
        registry = registry or {}
        paths = {}
        for mid in members:
            meta = registry.get(mid) or {}
            src_path = meta.get("source_path") or meta.get("label", mid)
            stem = _pathlib.Path(src_path).stem if src_path else mid
            if filename_policy == "suffix" and suffix:
                stem = stem + suffix
            paths[mid] = _os.path.join(out_dir, stem + _ext)

        # Overwrite check
        if overwrite == "refuse":
            conflicts = [p for p in paths.values() if _os.path.exists(p)]
            if conflicts:
                lines = "\n".join(conflicts[:10])
                tail = f"\n… and {len(conflicts) - 10} more." if len(conflicts) > 10 else ""
                return f"Refused — {len(conflicts)} file(s) already exist:\n{lines}{tail}"

        done, errs = 0, []
        for mid, out_path in paths.items():
            try:
                df = get_rows(mid)
                m = Motl(df)
                run_operation(m.write_out, {"output_path": out_path, "motl_type": motl_type})
                done += 1
            except PoolPayloadMissing as exc:
                errs.append(f"{mid}: {exc}")
            except Exception as exc:
                errs.append(f"{mid}: {exc}")

        status = f"Converted {done}/{len(members)} motl(s) to {motl_type}."
        if errs:
            status += "  Errors: " + "; ".join(errs[:3])
            if len(errs) > 3:
                status += f" … (+{len(errs) - 3} more)"
        return status

    # ── Part F: write-back @-variable results to the form text inputs ──────────
    from cryocat.app.formgen import register_var_picker_writeback
    register_var_picker_writeback(app, "me-op-param")
    register_var_picker_writeback(app, "me-multi-param")
