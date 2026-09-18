"""PoolSlotList — generic pool-plus-slot-map list component.

Renders a flat list where each pool entry carries its own slot dropdown.
Accepts any registry dict of shape {item_id: {label, kind?, ...}} plus a
slot map (list[item_id | None]) of length n_slots.  No pool-specific logic.

Two registration modes:

register_pool_slot_list_callbacks
    Full mode: also registers a render callback that populates the container
    returned by get_pool_slot_list().  Pass row_extra_fn to append pool-specific
    controls (e.g. ✕ remove button) after the slot dropdown on each row.
    Pass active_id_store_id to enable click-to-select and active-row
    highlighting; the store is set to the clicked item_id.

register_slot_change_callback
    Slot-only mode: registers just the slot-change callback.  Use when the
    calling module owns its own rendering (e.g. motlsidebar's group-aware list).
    The slot dropdowns in the custom render must use id pattern
    {"type": "{prefix}-psl-slot", "item_id": item_id}.

_first_free_slot(slot_map, n_slots)
    Shared helper — returns the lowest-numbered free slot index, or None.
    Import and call this from any page that needs next-free-slot logic so that
    there is exactly one implementation used by all callers.
"""
from __future__ import annotations

import dash
from dash import html, dcc, Input, Output, State, no_update, ctx, ALL
import dash_bootstrap_components as dbc

from cryocat.app import styles
from cryocat.app.formgen import make_dropdown

_UNASSIGNED = "__unassigned__"


def _first_free_slot(slot_map: list, n_slots: int) -> int | None:
    """Return the lowest-numbered free slot index, or None if all occupied."""
    for i in range(n_slots):
        if i >= len(slot_map) or not slot_map[i]:
            return i
    return None


def _current_slot(item_id: str, slot_map: list) -> str:
    for i, sid in enumerate(slot_map):
        if sid == item_id:
            return str(i)
    return _UNASSIGNED


def _slot_options(item_id: str, slot_map: list, n_slots: int) -> list[dict]:
    opts = [{"label": "—", "value": _UNASSIGNED}]
    for i in range(n_slots):
        occupant = slot_map[i] if i < len(slot_map) else None
        if occupant and occupant != item_id:
            label = f"Slot {i + 1} (taken)"
        else:
            label = f"Slot {i + 1}"
        opts.append({"label": label, "value": str(i)})
    return opts


def get_pool_slot_list(prefix: str) -> html.Div:
    """Return the container div; register_pool_slot_list_callbacks populates it."""
    return html.Div([
        dcc.Store(id=f"{prefix}-psl-seq", data=0),
        html.Div(
            id=f"{prefix}-psl-list",
            children=[html.Div("No entries.", style=styles.HINT)],
        ),
    ])


def _build_rows(reg, sm, n_slots, active_id, prefix, row_extra_fn):
    """Build dbc.ListGroupItem rows for the pool list, or a hint when empty."""
    if not reg:
        return html.Div("No entries.", style=styles.HINT)
    rows = []
    for item_id, entry in reg.items():
        label = entry.get("label", item_id)
        kind = entry.get("kind") or ""
        current = _current_slot(item_id, sm)
        opts = _slot_options(item_id, sm, n_slots)
        is_active = (item_id == active_id)

        row_style = {
            "display": "flex",
            "alignItems": "center",
            "gap": "0.4rem",
            "padding": "3px 4px",
        }
        if is_active:
            row_style["backgroundColor"] = "var(--bs-primary-bg-subtle)"

        label_parts = [
            html.Span(
                label,
                style={
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                    "whiteSpace": "nowrap",
                    "fontWeight": 600,
                    "fontSize": styles.FONT_SM,
                },
            ),
        ]
        if kind:
            label_parts.append(
                html.Span(
                    f"[{kind}]",
                    style={
                        "fontSize": styles.FONT_XS,
                        "color": styles.COLOR_MUTED,
                        "flexShrink": 0,
                    },
                )
            )

        row_children = [
            # id and n_clicks live on the label div only — not on the ListGroupItem
            # wrapper — so a click on the slot dropdown does not propagate to
            # _on_item_click and cannot trigger a stale re-render of the list.
            html.Div(
                label_parts,
                id={"type": f"{prefix}-psl-item", "item_id": item_id},
                n_clicks=0,
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "0.25rem",
                    "flex": "1 1 0",
                    "minWidth": 0,
                    "cursor": "pointer",
                },
            ),
            html.Div(
                make_dropdown(
                    {"type": f"{prefix}-psl-slot", "item_id": item_id},
                    opts,
                    current,
                    clearable=False,
                ),
                style={"width": "8rem", "flexShrink": 0},
            ),
        ]
        if row_extra_fn is not None:
            row_children.extend(row_extra_fn(item_id, entry))

        rows.append(
            dbc.ListGroupItem(
                row_children,
                style=row_style,
            )
        )
    return dbc.ListGroup(rows, flush=True)


def register_slot_change_callback(
    app,
    prefix: str,
    slot_map_id: str,
    n_slots: int,
) -> None:
    """Register only the slot-change callback.

    Use this when the calling module renders its own list (custom layout) and
    just needs generic non-destructive slot assignment handled for it.
    """

    @app.callback(
        Output(slot_map_id, "data"),
        Input({"type": f"{prefix}-psl-slot", "item_id": ALL}, "value"),
        State(slot_map_id, "data"),
        prevent_initial_call=True,
    )
    def _on_slot_change(values, slot_map):
        if not ctx.triggered_id or not values:
            raise dash.exceptions.PreventUpdate
        item_id = ctx.triggered_id["item_id"]
        new_val = ctx.triggered[0]["value"]
        sm = list(slot_map or [None] * n_slots)
        while len(sm) < n_slots:
            sm.append(None)
        for i in range(n_slots):
            if sm[i] == item_id:
                sm[i] = None
        if new_val == _UNASSIGNED:
            return sm
        try:
            slot_idx = int(new_val)
        except (TypeError, ValueError):
            return sm
        if 0 <= slot_idx < n_slots:
            sm[slot_idx] = item_id
        return sm


def register_pool_slot_list_callbacks(
    app,
    prefix: str,
    pool_registry_id: str,
    slot_map_id: str,
    n_slots: int,
    *,
    row_extra_fn=None,
    active_id_store_id: str | None = None,
) -> None:
    """Register render + slot-change callbacks for the PoolSlotList.

    row_extra_fn(item_id, entry) -> list[html elements]
        Optional per-row extras added after the slot dropdown.

    active_id_store_id : str | None
        When given, the render highlights the active row by reading from this
        store, and registers a click callback that writes the clicked item_id
        back to the same store.
    """
    _has_active = active_id_store_id is not None
    # psl-seq fires _render whenever the pool registry changes (via
    # _clean_stale_slots) without making pool_registry_id a direct Input.
    # This prevents _render from reading a stale slot_map concurrently with
    # _on_slot_change when threaded=True in the server.
    _all_inputs = [
        Input(f"{prefix}-psl-seq", "data"),
        State(pool_registry_id, "data"),
        Input(slot_map_id, "data"),
        *([] if not _has_active else [Input(active_id_store_id, "data")]),
    ]

    # One registration. To add a new optional store: add its id parameter
    # above, append Input(...) to _all_inputs when it is not None, add a
    # _has_<name> flag, and read args[_i] in _render. No new @app.callback block.
    @app.callback(
        Output(f"{prefix}-psl-list", "children"),
        *_all_inputs,
    )
    def _render(*args):
        _seq = args[0]  # trigger only
        registry = args[1]
        slot_map = args[2]
        active_id = args[3] if _has_active else None
        reg = registry or {}
        sm = list(slot_map or [None] * n_slots)
        while len(sm) < n_slots:
            sm.append(None)
        return _build_rows(reg, sm, n_slots, active_id, prefix, row_extra_fn)

    if active_id_store_id:
        @app.callback(
            Output(active_id_store_id, "data", allow_duplicate=True),
            Input({"type": f"{prefix}-psl-item", "item_id": ALL}, "n_clicks"),
            prevent_initial_call=True,
        )
        def _on_item_click(n_clicks):
            if not ctx.triggered_id or not any(n or 0 for n in (n_clicks or [])):
                raise dash.exceptions.PreventUpdate
            return ctx.triggered_id["item_id"]

    register_slot_change_callback(app, prefix, slot_map_id, n_slots)

    @app.callback(
        Output(slot_map_id, "data", allow_duplicate=True),
        Output(f"{prefix}-psl-seq", "data"),
        Input(pool_registry_id, "data"),
        State(slot_map_id, "data"),
        State(f"{prefix}-psl-seq", "data"),
        prevent_initial_call=True,
    )
    def _clean_stale_slots(registry, slot_map, seq):
        """Nullify stale slots and bump psl-seq to retrigger _render.

        Fires whenever the pool registry changes.  Removing a stale entry
        writes an updated slot_map; a registry-only add (no stale slots)
        skips the slot_map write but always increments psl-seq so _render
        still fires to show the new entry.  _render reads pool_registry_id
        as State rather than Input, which means it never fires concurrently
        with _on_slot_change — eliminating the slot-revert race.
        """
        reg = registry or {}
        sm = list(slot_map or [None] * n_slots)
        updated = [sid if sid in reg else None for sid in sm]
        new_seq = (seq or 0) + 1
        if updated == sm:
            return no_update, new_seq
        return updated, new_seq


def register_slot_focus_callback(
    app,
    slot_map_id: str,
    tabs_id: str,
    tab_prefix: str,
    n_slots: int,
    *,
    active_id_store_id: str | None = None,
    empty_fallback_tab_id: str | None = None,
) -> None:
    """Register a callback that keeps the tab strip on an occupied slot.

    Fires whenever the slot map changes.  If the currently active tab now maps
    to an empty slot it moves focus to the lowest-numbered occupied slot; when
    no slot is occupied it redirects to *empty_fallback_tab_id* (if given) or
    slot 0 so the user is never left on a disabled tab.

    active_id_store_id : str | None
        When given, also outputs to this store: writes the new focused slot's
        entry id (the raw slot_map value, a string), or None when no slot is
        occupied.

    empty_fallback_tab_id : str | None
        Tab id to redirect to when all slots are empty (e.g. "tango-tab-twist").
        Defaults to the first slot tab when None.

    One shared implementation — call once per strip (graphs, motls, …).
    """

    def _compute(slot_map, current_tab):
        if not current_tab or not current_tab.startswith(tab_prefix):
            return None, None, False  # tab, active_id, changed
        try:
            current_idx = int(current_tab[len(tab_prefix):])
        except ValueError:
            return None, None, False
        sm = list(slot_map or [None] * n_slots)
        if current_idx < len(sm) and sm[current_idx]:
            return None, None, False  # still occupied
        for i, sid in enumerate(sm):
            if sid:
                return f"{tab_prefix}{i}", sid, True
        # No slots occupied — redirect away from the slot strip
        fallback = empty_fallback_tab_id or f"{tab_prefix}0"
        new_tab = None if current_tab == fallback else fallback
        return new_tab, None, True

    _has_active = active_id_store_id is not None
    _all_outputs = [
        Output(tabs_id, "active_tab", allow_duplicate=True),
        *([] if not _has_active else [Output(active_id_store_id, "data", allow_duplicate=True)]),
    ]

    # One registration. To add a new optional store: add its id parameter
    # above, append Output(...) to _all_outputs when it is not None, add a
    # _has_<name> flag, and extend the return tuple in _focus_on_slot_change.
    # No new @app.callback block.
    @app.callback(
        *_all_outputs,
        Input(slot_map_id, "data"),
        State(tabs_id, "active_tab"),
        prevent_initial_call=True,
    )
    def _focus_on_slot_change(slot_map, current_tab):
        new_tab, new_id, changed = _compute(slot_map, current_tab)
        tab_val = (new_tab if new_tab is not None else no_update) if changed else no_update
        if _has_active:
            return tab_val, (new_id if changed else no_update)
        return tab_val
