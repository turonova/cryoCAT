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

import pandas as pd

import dash
from dash import html, dcc, Input, Output, State, no_update, ctx, ALL
import dash_bootstrap_components as dbc

from cryocat.core.cryomotl import Motl
from cryocat.app.components.motlio import get_motl_load_component, register_motl_load_callbacks
from cryocat.app.apputils import generate_kwargs, get_motl_operation_methods
from cryocat.app.formgen import build_form
from cryocat.app.logger import dash_logger

# Number of editor *view slots* (rendered table/viewer surfaces). The motl pool
# itself is unbounded — this only caps how many motls are open as tabs at once.
N_SLOTS = 5

# Fetched once at import time from the live Motl class — no hardcoding.
_MOTL_METHODS = get_motl_operation_methods()

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
                                    id="me-op-motl-select",
                                    multi=True,
                                    placeholder="Select motls",
                                    style={"marginBottom": "0.5rem"},
                                ),
                                dcc.Dropdown(
                                    id="me-op-column-select",
                                    placeholder="Match column (common/unique)",
                                    style={"marginBottom": "0.75rem"},
                                ),
                                html.Div(
                                    [
                                        dbc.Button(
                                            "Merge",
                                            id="me-merge-btn",
                                            color="primary",
                                            size="sm",
                                            className="mb-1",
                                            style={"width": "100%"},
                                        ),
                                        dbc.Button(
                                            "Common rows",
                                            id="me-intersect-btn",
                                            color="secondary",
                                            size="sm",
                                            className="mb-1",
                                            style={"width": "100%"},
                                        ),
                                        dbc.Button(
                                            "Unique rows",
                                            id="me-unique-btn",
                                            color="secondary",
                                            size="sm",
                                            style={"width": "100%"},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="me-multiop-status",
                                    style={"marginTop": "0.5rem", "fontSize": "0.9rem", "color": "var(--color9)"},
                                ),
                            ],
                            title="Multi-Motl Operations",
                            item_id="me-sidebar-ops",
                        ),
                    ],
                    always_open=True,
                    active_item=["me-sidebar-load", "me-sidebar-list"],
                ),
                html.Div(
                    dbc.Button(
                        "Show log",
                        id="me-open-log-btn",
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


def register_motl_editor_sidebar_callbacks(app):

    register_motl_load_callbacks(app, "me-load")

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

    # ── Match-column options (from the first pool motl) ────────────────────────
    @app.callback(
        Output("me-op-column-select", "options"),
        Output("me-op-column-select", "value"),
        Input("pool-registry", "data"),
        State("pool-motls", "data"),
        prevent_initial_call=True,
    )
    def update_column_options(registry, pool_motls):
        pool_motls = pool_motls or {}
        for rows in pool_motls.values():
            if rows:
                cols = list(pd.DataFrame(rows).columns)
                options = [{"label": c, "value": c} for c in cols]
                default = "subtomo_id" if "subtomo_id" in cols else cols[0]
                return options, default
        return [], None

    # ── Pool motl list + multi-op selector ─────────────────────────────────────
    @app.callback(
        Output("me-motl-list", "children"),
        Output("me-op-motl-select", "options"),
        Input("pool-registry", "data"),
        prevent_initial_call=True,
    )
    def update_motl_list(registry):
        registry = registry or {}
        items = []
        options = []
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
            options.append({"label": label, "value": mid})

        if not items:
            return (
                html.Div("No motls loaded.", style={"color": "var(--color9)", "fontSize": "0.9rem", "padding": "4px"}),
                [],
            )
        return dbc.ListGroup(items, flush=True), options

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

    # ── Multi-motl operations (write the Results tab) ──────────────────────────
    @app.callback(
        Output("me-results-store", "data", allow_duplicate=True),
        Output("me-results-label-store", "data", allow_duplicate=True),
        Output("me-multiop-status", "children", allow_duplicate=True),
        Input("me-merge-btn", "n_clicks"),
        State("me-op-motl-select", "value"),
        State("pool-motls", "data"),
        prevent_initial_call=True,
    )
    def merge_motls(n_clicks, selected, pool_motls):
        if not n_clicks or not selected:
            raise dash.exceptions.PreventUpdate
        pool_motls = pool_motls or {}
        dfs = [pd.DataFrame(pool_motls[m]) for m in selected if pool_motls.get(m)]
        if not dfs:
            return no_update, no_update, "No data in selected motls."
        merged = pd.concat(dfs, ignore_index=True)
        label = f"Merged ({len(merged)} particles)"
        dash_logger.write(
            f"merged = pd.concat([{', '.join(selected)}], ignore_index=True)", source="cryocat"
        )
        return merged.to_dict("records"), label, f"Merged {len(dfs)} motls: {len(merged)} particles total."

    @app.callback(
        Output("me-results-store", "data", allow_duplicate=True),
        Output("me-results-label-store", "data", allow_duplicate=True),
        Output("me-multiop-status", "children", allow_duplicate=True),
        Input("me-intersect-btn", "n_clicks"),
        State("me-op-motl-select", "value"),
        State("me-op-column-select", "value"),
        State("pool-motls", "data"),
        prevent_initial_call=True,
    )
    def intersect_motls(n_clicks, selected, match_col, pool_motls):
        if not n_clicks or not selected or len(selected) < 2:
            raise dash.exceptions.PreventUpdate
        if not match_col:
            return no_update, no_update, "Select a column to match on."
        pool_motls = pool_motls or {}
        dfs = [pd.DataFrame(pool_motls[m]) for m in selected if pool_motls.get(m)]
        if len(dfs) < 2:
            return no_update, no_update, "Need at least 2 motls with data."
        if match_col not in dfs[0].columns:
            return no_update, no_update, f"Column '{match_col}' not found in motl."
        common_vals = set(dfs[0][match_col])
        for df in dfs[1:]:
            common_vals &= set(df[match_col])
        result = dfs[0][dfs[0][match_col].isin(common_vals)].reset_index(drop=True)
        label = f"Common rows ({len(result)} particles)"
        dash_logger.write(
            f"# Common rows on '{match_col}' across [{', '.join(selected)}]\n"
            f"common_vals = set.intersection(*[set(m['{match_col}']) for m in [{', '.join(selected)}]])\n"
            f"result = {selected[0]}[{selected[0]}['{match_col}'].isin(common_vals)]",
            source="cryocat",
        )
        return (
            result.to_dict("records"),
            label,
            f"Found {len(result)} rows with common '{match_col}' across {len(dfs)} motls.",
        )

    @app.callback(
        Output("me-results-store", "data", allow_duplicate=True),
        Output("me-results-label-store", "data", allow_duplicate=True),
        Output("me-multiop-status", "children", allow_duplicate=True),
        Input("me-unique-btn", "n_clicks"),
        State("me-op-motl-select", "value"),
        State("me-op-column-select", "value"),
        State("pool-motls", "data"),
        prevent_initial_call=True,
    )
    def unique_motls(n_clicks, selected, match_col, pool_motls):
        if not n_clicks or not selected or len(selected) < 2:
            raise dash.exceptions.PreventUpdate
        if not match_col:
            return no_update, no_update, "Select a column to match on."
        pool_motls = pool_motls or {}
        dfs = [pd.DataFrame(pool_motls[m]) for m in selected if pool_motls.get(m)]
        if len(dfs) < 2:
            return no_update, no_update, "Need at least 2 motls with data."
        if match_col not in dfs[0].columns:
            return no_update, no_update, f"Column '{match_col}' not found in motl."
        all_val_sets = [set(df[match_col]) for df in dfs]
        common = all_val_sets[0].intersection(*all_val_sets[1:])
        unique_vals = set().union(*all_val_sets) - common
        result = pd.concat([df[df[match_col].isin(unique_vals)] for df in dfs], ignore_index=True)
        label = f"Unique rows ({len(result)} particles)"
        dash_logger.write(
            f"# Unique rows on '{match_col}' across [{', '.join(selected)}]\n"
            f"all_sets = [set(m['{match_col}']) for m in [{', '.join(selected)}]]\n"
            f"unique_vals = set.union(*all_sets) - set.intersection(*all_sets)\n"
            f"result = pd.concat([m[m['{match_col}'].isin(unique_vals)] for m in [{', '.join(selected)}]])",
            source="cryocat",
        )
        return (
            result.to_dict("records"),
            label,
            f"Found {len(result)} rows with unique '{match_col}' across {len(dfs)} motls.",
        )

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
            motl = Motl(pd.DataFrame(current_data))
            result = getattr(motl, method_name)(**kwargs)
        except Exception as exc:
            return _ret(nochange, nochange, f"Error: {exc}")

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
